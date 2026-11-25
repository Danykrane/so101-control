"""SO-101 driver: SimDriver (virtual arm) and FeetechDriver (Feetech bus via LeRobot).

No Qt: designed to run inside the separate robotd process. All bus I/O happens in a single
worker thread; the UI sends commands through a queue and receives RobotState via a callback.

Safety rules (no motion on startup):
- connecting only reads: ping, calibration from EEPROM, positions, torque, telemetry;
- nothing is written to the motors before the first user command;
- torque is enabled only by an explicit command: Goal_Position = Present_Position, then Torque_Enable;
- closing does not touch torque by default (disconnect(disable_torque=False)).
"""

from __future__ import annotations

import glob
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from kinematics import JOINTS
from safety import Limits, clip_target, compensated, integrate_bias, limit_lead, step_toward

MOTOR_IDS = {name: i + 1 for i, name in enumerate(JOINTS)}
MODEL = "sts3215"
MODEL_NUMBER = 777  # what an STS3215 returns when Model_Number is read
MAX_RES = 4095  # LeRobot DEGREES: (raw - mid) * 360 / 4095
GRIPPER = 5

DIAGNOSTIC_KEYS = [  # (Diagnostics.rows key, column title)
    ("id", "ID"), ("name", "Motor"), ("model", "Model"), ("firmware", "Firmware"), ("voltage", "V"),
    ("temperature", "°C"), ("load", "Load"), ("torque", "Torque"), ("range_min", "min"),
    ("range_max", "max"), ("homing_offset", "offset"), ("present_raw", "raw"), ("present", "Position"),
]

PORT_PATTERNS = ("/dev/tty.usbmodem*", "/dev/tty.usbserial*", "/dev/tty.wchusbserial*")


def find_ports() -> list[str]:
    return sorted({p for pattern in PORT_PATTERNS for p in glob.glob(pattern)})


# --- raw (encoder ticks) <-> motor units, same as LeRobot -------------------------------


def raw_to_motor(raw, rmin, rmax) -> np.ndarray:
    raw, rmin, rmax = (np.asarray(v, dtype=float) for v in (raw, rmin, rmax))
    out = (raw - (rmin + rmax) / 2) * 360.0 / MAX_RES
    out[GRIPPER] = (raw[GRIPPER] - rmin[GRIPPER]) / (rmax[GRIPPER] - rmin[GRIPPER]) * 100.0
    return out


def motor_to_raw(value, rmin, rmax) -> np.ndarray:
    value, rmin, rmax = (np.asarray(v, dtype=float) for v in (value, rmin, rmax))
    out = value * MAX_RES / 360.0 + (rmin + rmax) / 2
    out[GRIPPER] = value[GRIPPER] / 100.0 * (rmax[GRIPPER] - rmin[GRIPPER]) + rmin[GRIPPER]
    return np.rint(out).astype(int)


def calibration_limits(rmin, rmax) -> Limits:
    return Limits(raw_to_motor(rmin, rmin, rmax), raw_to_motor(rmax, rmin, rmax))


# --- state --------------------------------------------------------------------------


@dataclass
class Diagnostics:
    port: str
    ok: bool
    message: str
    rows: list[dict] = field(default_factory=list)  # one row per motor


@dataclass
class RobotState:
    t: float
    mode: str  # "sim" | "real"
    connected: bool
    torque: np.ndarray  # (6,) bool
    present: np.ndarray  # (6,) motor units
    command: np.ndarray  # what is being sent right now (interpolated goal)
    target: np.ndarray  # what the user asked for
    limits: Limits
    load: np.ndarray  # (6,) Present_Load, 0.1 % of maximum
    temperature: np.ndarray  # (6,) °C
    voltage: float  # V
    loop_hz: float
    error: str = ""
    bias: np.ndarray = field(default_factory=lambda: np.zeros(6))  # anti-sag offset, deg


    @property
    def torque_on(self) -> bool:
        return bool(np.all(self.torque))


class RobotDriver:
    """Shared control loop. Subclasses implement only I/O (_open, _read_present, ...)."""

    mode = "base"
    hz = 30.0
    telemetry_every = 10  # ticks: load, temperature, voltage at about 3 Hz
    max_failures = 30  # consecutive ticks without a link (about 1 s) -> disconnect
    sag_ki = 2.0  # 1/s, outer I-loop against sag under gravity (0 = off), see safety.integrate_bias

    def __init__(
        self,
        urdf_limits: Limits,
        on_state: Callable[[RobotState], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_diag: Callable[[Diagnostics], None] | None = None,
    ):
        self.urdf_limits = urdf_limits
        self.on_state = on_state or (lambda s: None)
        self.on_log = on_log or (lambda m: None)
        self.on_diag = on_diag or (lambda d: None)
        self.speed_arm = 20.0  # deg/s
        self.speed_gripper = 40.0  # %/s
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self.state: RobotState | None = None

    # --- API for the UI (thread-safe) ---

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"{self.mode}-driver", daemon=True)
        self._thread.start()

    def close(self, release: bool = False, timeout: float = 3.0) -> None:
        """release=False: torque stays as is (the arm holds its pose); True: disable torque."""
        if self._thread and self._thread.is_alive():
            self._q.put(("close", release))
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def jog(self, delta) -> None:
        self._q.put(("jog", np.asarray(delta, dtype=float)))

    def set_target(self, target) -> None:
        """Absolute goal (6), NaN in a joint slot = leave unchanged."""
        self._q.put(("target", np.asarray(target, dtype=float)))

    def hold(self) -> None:
        self._q.put(("hold", None))

    def enable_torque(self) -> None:
        self._q.put(("torque", True))

    def disable_torque(self) -> None:
        self._q.put(("torque", False))

    def set_speed(self, arm_deg_s: float, gripper_pct_s: float) -> None:
        self._q.put(("speed", (float(arm_deg_s), float(gripper_pct_s))))

    # --- I/O, implemented by subclasses ---

    def _open(self) -> tuple[Diagnostics, Limits, np.ndarray, np.ndarray | None, np.ndarray]:
        """-> (diagnostics, calibration limits, present, goal from the motors or None, torque[6])."""
        raise NotImplementedError

    def _read_present(self) -> np.ndarray:
        raise NotImplementedError

    def _read_telemetry(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """-> (load[6], temperature[6], voltage, torque[6] bool)."""
        raise NotImplementedError

    def _write_command(self, command: np.ndarray) -> None:
        raise NotImplementedError

    def _torque_on(self) -> np.ndarray:
        """Write Goal_Position = Present_Position, enable torque, return present."""
        raise NotImplementedError

    def _torque_off(self) -> None:
        raise NotImplementedError

    def _close(self, release: bool) -> None:
        raise NotImplementedError

    # --- loop ---

    def _run(self) -> None:
        nan6 = np.full(6, np.nan)
        try:
            diag, cal_limits, present, goal, torque = self._open()
        except Exception as e:  # noqa: BLE001 - show any error in the UI
            self.on_log(f"Connection failed: {str(e).strip()}")
            z = np.zeros(6)
            self._publish(False, np.zeros(6, bool), z, z, z, self.urdf_limits, nan6, nan6, float("nan"), 0, str(e).strip())
            return
        self.on_diag(diag)
        limits = self.urdf_limits.intersect(cal_limits)
        torque = np.asarray(torque, bool)
        if torque.any() and not torque.all():
            self.on_log("Warning: torque is not enabled on all motors. Press \"Torque on\" or \"Disconnect\".")
        command = goal.copy() if (goal is not None and torque.all()) else present.copy()
        target, sent = command.copy(), command.copy()
        bias, speed, prev = np.zeros(6), np.zeros(6), present.copy()
        armed = False  # write nothing before the first user command
        load, temp, volt = nan6, nan6, float("nan")
        period = 1.0 / self.hz
        tick, failures, hz, release, error = 0, 0, self.hz, False, ""
        last = deadline = time.monotonic()
        self.on_log(f"{self.mode}: connected, {diag.message}")

        while True:
            stop = False
            try:
                while True:
                    cmd, arg = self._q.get_nowait()
                    if cmd == "close":
                        stop, release = True, arg
                        break
                    if cmd == "speed":
                        self.speed_arm, self.speed_gripper = arg
                    elif cmd == "torque" and arg:
                        present = self._torque_on()
                        torque[:] = True
                        command, target, sent, armed = present.copy(), present.copy(), present.copy(), True
                        bias[:] = 0.0
                        self.on_log("Torque on, the arm holds its current pose")
                    elif cmd == "torque":
                        self._torque_off()
                        torque[:] = False
                        self.on_log("Torque off")
                    elif not torque.all():
                        self.on_log("Torque is off: press \"Torque on\" first")
                    elif cmd == "hold":  # stop: exactly the current position, no bias
                        command, target, armed = present.copy(), present.copy(), True
                        bias[:] = 0.0
                        self._write_command(command)
                        sent = command.copy()
                        self.on_log("Stop: holding the current position")
                    elif cmd == "jog":
                        target, armed = clip_target(target + arg, target, limits), True
                    elif cmd == "target":  # NaN = leave the joint unchanged
                        target, armed = clip_target(np.where(np.isnan(arg), target, arg), target, limits), True
            except queue.Empty:
                pass
            except Exception as e:  # noqa: BLE001
                self.on_log(f"Command error: {e}")
            if stop:
                break

            try:
                present = self._read_present()
                speed = 0.7 * speed + 0.3 * (present - prev) / period  # smoothing: one encoder tick ~ 2.6 deg/s per tick
                prev = present.copy()
                if tick % self.telemetry_every == 0:
                    load, temp, volt, actual = self._read_telemetry()
                    if np.any(actual != torque):  # overheating/overload or another program
                        self.on_log(f"Motor torque changed externally: {actual.astype(int).tolist()}")
                        torque = actual.copy()
                if armed and torque.all():
                    command = step_toward(command, target, self.speed_arm * period, self.speed_gripper * period)
                    command = limit_lead(command, present)
                    if self.sag_ki > 0:
                        bias = integrate_bias(bias, target, present, command, speed, period, self.sag_ki)
                    out = compensated(command, bias, present, sent, limits)
                    if np.any(np.abs(out - sent) > 1e-3):
                        self._write_command(out)
                        sent = out
                failures, error = 0, ""
            except Exception as e:  # noqa: BLE001
                failures += 1
                error = f"Lost link to the motors: {e}"
                if failures == 1:
                    self.on_log(error)
                if failures >= self.max_failures:
                    self.on_log("Link lost, disconnecting (torque untouched)")
                    break
            if not torque.all():
                command, target, sent = present.copy(), present.copy(), present.copy()
                bias[:] = 0.0

            now = time.monotonic()
            hz = 0.9 * hz + 0.1 / max(now - last, 1e-6)
            last = now
            self._publish(True, torque, present, command, target, limits, load, temp, volt, hz, error, bias)
            tick += 1
            deadline = max(deadline + period, time.monotonic())  # steady 30 Hz without accumulating lag
            time.sleep(max(0.0, deadline - time.monotonic()))

        try:
            self._close(release)
            self.on_log("Disconnected" + (", torque disabled" if release else ", torque unchanged"))
        except Exception as e:  # noqa: BLE001
            self.on_log(f"Error while disconnecting: {e}")
        self._publish(False, torque, present, command, target, limits, load, temp, volt, 0, error)

    def _publish(self, connected, torque, present, command, target, limits, load, temp, volt, hz, error, bias=None):
        self.state = RobotState(
            t=time.time(),
            mode=self.mode,
            connected=connected,
            torque=np.array(torque, bool),
            present=np.array(present, float),
            command=np.array(command, float),
            target=np.array(target, float),
            limits=limits,
            load=np.array(load, float),
            temperature=np.array(temp, float),
            voltage=float(volt),
            loop_hz=float(hz),
            error=error,
            bias=np.zeros(6) if bias is None else np.array(bias, float),
        )
        self.on_state(self.state)


class SimDriver(RobotDriver):
    """Virtual arm: each servo is a first-order lag, no gravity."""

    mode = "sim"
    REST = np.array([0.0, -90.0, 85.0, 60.0, 0.0, 5.0])

    def __init__(self, urdf_limits: Limits, sag=None, **kw):
        super().__init__(urdf_limits, **kw)
        self._sag = np.zeros(6) if sag is None else np.asarray(sag, float)  # simulated sag under gravity, for tests
        self._pos = self.REST.copy()
        self._goal = self.REST.copy()
        self._torque = False

    def _open(self):
        rows = [
            dict(id=i, name=n, model="sim", firmware="-", voltage=12.0, temperature=30.0, load=0, torque=0,
                 range_min="-", range_max="-", homing_offset="-", present_raw="-", present=round(self._pos[i - 1], 1))
            for n, i in MOTOR_IDS.items()
        ]
        wide = Limits(np.array([-180.0] * 5 + [0.0]), np.array([180.0] * 5 + [100.0]))
        diag = Diagnostics("sim", True, "virtual robot, 6/6 motors", rows)
        return diag, wide, self._pos.copy(), None, np.zeros(6, bool)

    def _read_present(self):
        if self._torque:
            self._pos += (self._goal - self._sag - self._pos) * 0.35
        return self._pos.copy()

    def _read_telemetry(self):
        load = np.rint((self._goal - self._pos) * 20) if self._torque else np.zeros(6)
        return load, np.full(6, 30.0), 12.0, np.full(6, self._torque)

    def _write_command(self, command):
        self._goal = np.array(command, float)

    def _torque_on(self):
        self._goal = self._pos.copy()
        self._torque = True
        return self._pos.copy()

    def _torque_off(self):
        self._torque = False

    def _close(self, release):
        if release:
            self._torque = False


class FeetechDriver(RobotDriver):
    """Real arm via lerobot SerialMotorsBus (rustypot). SO101Follower.connect() is never called."""

    mode = "real"

    def __init__(self, port: str, urdf_limits: Limits, bus_factory=None, **kw):
        super().__init__(urdf_limits, **kw)
        self.port = port
        self._bus_factory = bus_factory or make_bus
        self.bus = None
        self.names = list(JOINTS)

    def _sync(self, reg: str) -> np.ndarray:
        values = self.bus.sync_read(reg, self.names, normalize=False)
        return np.array([values[n] for n in self.names], dtype=float)

    def _open(self):
        self.bus = self._bus_factory(self.port)
        diag = probe(self.bus, self.port)
        if not diag.ok:
            self.on_diag(diag)
            self.bus.disconnect(disable_torque=False)
            raise RuntimeError(diag.message)
        rows = diag.rows
        self.rmin = np.array([r["range_min"] for r in rows], float)
        self.rmax = np.array([r["range_max"] for r in rows], float)
        present = raw_to_motor([r["present_raw"] for r in rows], self.rmin, self.rmax)
        goal = raw_to_motor([r["goal_raw"] for r in rows], self.rmin, self.rmax)
        torque = np.array([r["torque"] for r in rows], bool)
        return diag, calibration_limits(self.rmin, self.rmax), present, goal, torque

    def _read_present(self):
        return raw_to_motor(self._sync("Present_Position"), self.rmin, self.rmax)

    def _read_telemetry(self):
        load = self._sync("Present_Load")
        temp = self._sync("Present_Temperature")
        volt = float(np.mean(self._sync("Present_Voltage"))) / 10.0
        return load, temp, volt, self._sync("Torque_Enable").astype(bool)

    def _write_command(self, command):
        raw = motor_to_raw(command, self.rmin, self.rmax)
        self.bus.sync_write("Goal_Position", {n: int(v) for n, v in zip(self.names, raw)}, normalize=False)

    def _torque_on(self):
        raw = self._sync("Present_Position").astype(int)
        for n, v in zip(self.names, raw):  # acknowledged by each motor, not sync_write
            self.bus.write("Goal_Position", n, int(v), normalize=False, num_retry=2)
        self.bus.enable_torque(num_retry=2)
        return raw_to_motor(raw, self.rmin, self.rmax)

    def _torque_off(self):
        self.bus.disable_torque(num_retry=2)

    def _close(self, release):
        if self.bus is None:
            return
        try:
            if release:
                self.bus.disable_torque(num_retry=2)
        finally:
            self.bus.disconnect(disable_torque=False)


# --- bus and diagnostics: read-only -------------------------------------------------------


def make_bus(port: str):
    """Open the port without a handshake. Writes nothing to the motors."""
    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    motors = {
        n: Motor(i, MODEL, MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
        for n, i in MOTOR_IDS.items()
    }
    bus = SerialMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)
    return bus


def probe(bus, port: str) -> Diagnostics:
    """Ping IDs 1-6 and read registers. Read-only; the EEPROM calibration is stored in bus.calibration."""
    rows = []
    for name, mid in MOTOR_IDS.items():
        model = bus.ping(mid, num_retry=2)
        rows.append(dict(id=mid, name=name, model=model))
    missing = [r for r in rows if r["model"] is None]
    wrong = [r for r in rows if r["model"] is not None and r["model"] != MODEL_NUMBER]
    if missing or wrong:
        parts = []
        if missing:
            parts.append("no answer from ID " + ", ".join(str(r["id"]) for r in missing))
        if wrong:
            parts.append("unexpected model: " + ", ".join(f"ID {r['id']}={r['model']}" for r in wrong))
        return Diagnostics(port, False, "; ".join(parts), rows)

    names = list(MOTOR_IDS)

    def sync(reg):
        return bus.sync_read(reg, names, normalize=False)

    cal = bus.read_calibration()
    bus.calibration = cal
    major, minor = sync("Firmware_Major_Version"), sync("Firmware_Minor_Version")
    present, goal, torque = sync("Present_Position"), sync("Goal_Position"), sync("Torque_Enable")
    volt, temp, load = sync("Present_Voltage"), sync("Present_Temperature"), sync("Present_Load")
    bad_cal = [n for n in names if cal[n].range_max <= cal[n].range_min]
    rmin = [cal[n].range_min for n in names]
    rmax = [cal[n].range_max for n in names]
    pos = raw_to_motor([present[n] for n in names], rmin, rmax) if not bad_cal else np.full(6, np.nan)
    for r, n, p in zip(rows, names, pos):
        r.update(
            firmware=f"{major[n]}.{minor[n]}",
            voltage=volt[n] / 10.0,
            temperature=temp[n],
            load=load[n],
            torque=int(torque[n]),
            range_min=cal[n].range_min,
            range_max=cal[n].range_max,
            homing_offset=cal[n].homing_offset,
            present_raw=present[n],
            goal_raw=goal[n],
            present=round(float(p), 1),
        )
    if bad_cal:
        return Diagnostics(port, False, f"calibration is empty or invalid: {', '.join(bad_cal)}", rows)
    on = sum(r["torque"] for r in rows)
    return Diagnostics(port, True, f"6/6 motors, torque on {on}/6, {np.mean([r['voltage'] for r in rows]):.1f} V", rows)
