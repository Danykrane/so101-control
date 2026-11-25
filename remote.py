"""RemoteDriver: a robotd client with the same API as RobotDriver.

In "robotd" mode the GUI watches and controls the robot while robotd owns the serial port. This way the GUI,
robotctl and an LLM agent (MCP) work with the same robot at the same time. No Qt.
Closing the client touches neither the torque nor robotd.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from driver import Diagnostics, RobotState
from kinematics import JOINTS
from safety import Limits

URL = "http://127.0.0.1:8765"


def robotd_alive(url: str = URL, timeout: float = 0.3) -> bool:
    """Is robotd running (then it owns the robot port and the GUI connects through it)."""
    try:
        with urllib.request.urlopen(url + "/state", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _arr(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


class RemoteDriver:
    mode = "remote"
    hz = 20.0

    def __init__(self, url: str = URL, on_state=None, on_log=None, on_diag=None):
        self.port = url
        self.on_state = on_state or (lambda s: None)
        self.on_log = on_log or (lambda m: None)
        self.on_diag = on_diag or (lambda d: None)
        self.speed_arm, self.speed_gripper = 20.0, 40.0
        self.state: RobotState | None = None
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._limits: Limits | None = None

    # --- same API as RobotDriver ---

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="remote-driver", daemon=True)
        self._thread.start()

    def close(self, release: bool = False, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def jog(self, delta) -> None:
        joints = {n: float(d) for n, d in zip(JOINTS, delta) if d}
        if joints:
            self._q.put(("/move", {"joints": joints, "relative": True, "wait": False}))

    def set_target(self, target) -> None:
        joints = {n: float(v) for n, v in zip(JOINTS, target) if not np.isnan(v)}
        if joints:
            self._q.put(("/move", {"joints": joints, "wait": False}))

    def hold(self) -> None:
        self._post("/stop", {})  # stop immediately, bypassing the queue

    def enable_torque(self) -> None:
        self._q.put(("/torque", {"on": True}))

    def disable_torque(self) -> None:
        self._q.put(("/torque", {"on": False}))

    def set_speed(self, arm_deg_s: float, gripper_pct_s: float) -> None:
        self._q.put(("/speed", {"arm": arm_deg_s, "gripper": gripper_pct_s}))

    # --- HTTP ---

    def _req(self, path: str, body: dict | None = None, timeout: float = 2.0):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.port + path, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def _post(self, path: str, body: dict) -> None:
        try:
            self._req(path, body, timeout=5.0)
        except urllib.error.HTTPError as e:
            self.on_log("robotd: " + json.loads(e.read() or b"{}").get("error", str(e)))
        except (urllib.error.URLError, OSError) as e:
            self.on_log(f"robotd is not responding: {e}")

    def _to_state(self, s: dict) -> RobotState:
        lower, upper = _arr(s["limits"]["lower"]), _arr(s["limits"]["upper"])
        if self._limits is None or not (np.array_equal(self._limits.lower, lower) and np.array_equal(self._limits.upper, upper)):
            self._limits = Limits(lower, upper)  # same object while limits are unchanged: the GUI does not rebuild sliders
        self.speed_arm, self.speed_gripper = s["speed"]["arm"], s["speed"]["gripper"]
        return RobotState(
            t=s["t"], mode="remote", connected=bool(s["connected"] and s.get("running", True)),
            torque=np.array(s["torque"], bool), present=_arr(s["present"]), command=_arr(s["command"]),
            target=_arr(s["target"]), limits=self._limits, load=_arr(s["load"]), temperature=_arr(s["temperature"]),
            voltage=float(s["voltage"] if s["voltage"] is not None else np.nan), loop_hz=float(s["loop_hz"] or 0),
            error=s.get("error", ""), bias=_arr(s.get("bias", [0.0] * 6)),
        )

    def _run(self) -> None:
        try:
            d = self._req("/diag")
            self.on_diag(Diagnostics(f"robotd {d.get('port', '')}", bool(d.get("ok")), d.get("message", ""), d.get("rows", [])))
            self.on_log(f"robotd {self.port}: connected, {d.get('message', '')}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.on_log(f"robotd is not responding at {self.port}: {e}. Start ./robotd.command")
            return
        seen: list[str] = []
        period, tick, fails = 1.0 / self.hz, 0, 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                while True:
                    path, body = self._q.get_nowait()
                    self._post(path, body)
            except queue.Empty:
                pass
            try:
                self.state = self._to_state(self._req("/state"))
                self.on_state(self.state)
                if tick % 10 == 0:  # forward the robotd log to the GUI, about 2 Hz
                    lines = self._req("/log?n=20")["log"]
                    new = [ln for ln in lines if ln not in seen]
                    for ln in new:
                        self.on_log("robotd: " + ln.split("  ", 1)[-1])
                    seen = lines
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                fails += 1
                if fails == 1:
                    self.on_log(f"robotd is not responding ({e}), waiting: a restarted robotd is picked up automatically")
                if fails == 10 and self.state is not None:
                    self.on_state(self._offline(self.state))
            else:
                if fails >= 10:
                    self.on_log("robotd is back online")
                fails = 0
            tick += 1
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))
        if self.state is not None:
            self.state = self._offline(self.state)
            self.on_state(self.state)

    @staticmethod
    def _offline(s: RobotState) -> RobotState:
        return RobotState(s.t, s.mode, False, s.torque, s.present, s.command, s.target, s.limits,
                          s.load, s.temperature, s.voltage, 0.0, s.error, s.bias)
