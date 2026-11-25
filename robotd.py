#!/usr/bin/env python3
"""robotd: держит порт SO-101 и камеры, принимает команды по HTTP на 127.0.0.1:8765.

Клиенты: robotctl.py (CLI), robot_mcp.py (MCP для Claude), окно app.py в режиме «robotd».
Правила те же, что в driver.py: подключение только читает, момент включается только командой
(Goal = Present, потом Torque_Enable), цель клипуется по лимитам, скорость ограничена.
При выходе (Ctrl+C, закрытие окна Terminal, /shutdown) момент не трогается, если не попросить release.

Запускать из Terminal (`./robotd.command`): доступ к камере macOS выдает приложению-родителю.

API, JSON:
  GET  /state                  суставы, момент, телеметрия, поза EE
  GET  /diag                   диагностика моторов при подключении
  GET  /log?n=50               последние сообщения
  GET  /cameras                список камер
  GET  /camera/<i>.jpg?w=640   снимок камеры
  POST /torque     {"on": true}
  POST /stop                   цель = текущая позиция, удержание
  POST /move       {"joints": {"elbow_flex": 10}, "relative": false, "wait": true, "timeout": 20}
                   {"pose": "rest"} = поза покоя (config/rest_pose.json, пишется при подключении без момента)
  POST /cartesian  {"dx": мм, "dy": мм, "dz": мм, "dpitch": °, "droll": °, "wait": true}
                   или абсолютно {"x": мм, "y": мм, "z": мм, "pitch": °, "roll": °}
  POST /speed      {"arm": °/с, "gripper": %/с}
  POST /shutdown   {"release": false}
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from cameras import Grabber, list_cameras
from driver import FeetechDriver, SimDriver, find_ports
from kinematics import HERE, JOINTS, JointMap, Robot, plan_cartesian
from safety import Limits

HOST, PORT = "127.0.0.1", 8765
REST_PATH = HERE / "config" / "rest_pose.json"  # сложенная поза: робот лежит без момента
ALIASES = {
    "pan": "shoulder_pan", "lift": "shoulder_lift", "shoulder": "shoulder_lift", "elbow": "elbow_flex",
    "wrist": "wrist_flex", "flex": "wrist_flex", "roll": "wrist_roll", "grip": "gripper",
}
TOLERANCE = np.array([1.0] * 5 + [3.0])  # «дошел»: ° и %; провисание под весом снимает I-контур драйвера
STALL_S = 1.5  # команда дошла до цели, а ошибка не уменьшается столько секунд = упор или предмет


class Busy(Exception):
    """Команду нельзя выполнить сейчас (момент выключен, нет связи): HTTP 409."""


def _num(x, nd=2):
    x = float(x)
    return None if math.isnan(x) else round(x, nd)


def _arr(a, nd=2):
    return [_num(v, nd) for v in np.asarray(a, dtype=float)]


class Robotd:
    def __init__(self, sim: bool, port: str | None, cams: list[int], z_min_mm: float):
        self.robot, self.jmap = Robot(), JointMap.load()
        self.logs: deque[str] = deque(maxlen=500)
        self.diag = None
        self.z_min = z_min_mm / 1000.0
        self.motion = threading.Lock()  # одна команда движения за раз
        urdf_limits = Limits(*self.jmap.motor_limits(self.robot))
        cb = dict(on_log=self.log, on_diag=self._on_diag)
        self.driver = SimDriver(urdf_limits, **cb) if sim else FeetechDriver(port, urdf_limits, **cb)
        self.port = "sim" if sim else port
        self.camera_list = list_cameras()
        builtin = {c["index"] for c in self.camera_list if c["builtin"]}
        self.grabbers: dict[int, Grabber] = {}
        for i in cams:
            if i in builtin:
                self.log(f"Камера {i} встроенная (FaceTime), не открываю")
                continue
            self.grabbers[i] = Grabber(i)

    # --- служебное ---

    def log(self, msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S}  {msg}"
        self.logs.append(line)
        print(line, flush=True)

    def _on_diag(self, d) -> None:
        self.diag = d

    def start(self) -> None:
        names = ", ".join(f"{c['index']}={c['name']}" for c in self.camera_list)
        self.log(f"Камеры: {names or 'нет'}, открываю {sorted(self.grabbers) or 'ни одной'}")
        for g in self.grabbers.values():
            g.start()
        self.driver.start()
        end = time.time() + 15
        while time.time() < end and self.driver.state is None:
            time.sleep(0.05)
        s = self.driver.state
        if s is not None and s.connected and s.mode == "real" and not s.torque.any():
            pose = {n: round(float(v), 1) for n, v in zip(JOINTS, s.present)}
            REST_PATH.write_text(json.dumps(pose, indent=1) + "\n", encoding="utf-8")
            self.log(f"Поза покоя записана в {REST_PATH.name}: {list(pose.values())}")

    def rest_pose(self) -> np.ndarray:
        if not REST_PATH.exists():
            raise Busy("позы покоя нет: она записывается, когда robotd подключается к роботу без момента")
        data = json.loads(REST_PATH.read_text(encoding="utf-8"))
        return np.array([float(data[n]) for n in JOINTS])

    def close(self, release: bool = False) -> None:
        self.driver.close(release=release)
        for g in self.grabbers.values():
            g.stop()

    @property
    def state(self):
        return self.driver.state

    def _ready(self):
        s = self.state
        if not (self.driver.running and s and s.connected):
            raise Busy("робот не подключен")
        if not s.torque_on:
            raise Busy("момент выключен: сначала POST /torque {\"on\": true}")
        return s

    def _pose(self, q_motor) -> dict:
        p = self.robot.task(self.jmap.to_urdf(q_motor)[:5])
        return {"x": _num(p[0] * 1000, 1), "y": _num(p[1] * 1000, 1), "z": _num(p[2] * 1000, 1),
                "pitch": _num(math.degrees(p[3]), 1), "roll": _num(math.degrees(p[4]), 1)}

    # --- ответы ---

    def state_json(self) -> dict:
        s = self.state
        if s is None:
            return {"connected": False, "port": self.port}
        return {
            "t": s.t, "mode": s.mode, "port": self.port, "connected": s.connected, "running": self.driver.running,
            "torque_on": s.torque_on, "torque": [bool(v) for v in s.torque], "joints": list(JOINTS),
            "present": _arr(s.present), "target": _arr(s.target), "command": _arr(s.command),
            "limits": {"lower": _arr(s.limits.lower), "upper": _arr(s.limits.upper)},
            "load": _arr(s.load, 0), "temperature": _arr(s.temperature, 0), "voltage": _num(s.voltage, 1),
            "loop_hz": _num(s.loop_hz, 1), "error": s.error, "bias": _arr(s.bias), "speed": {"arm": self.driver.speed_arm, "gripper": self.driver.speed_gripper},
            "ee": self._pose(s.present), "ee_target": self._pose(s.target),
        }

    def diag_json(self) -> dict:
        d = self.diag
        if d is None:
            return {}
        rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in d.rows]
        return {"port": d.port, "ok": d.ok, "message": d.message, "rows": rows}

    # --- команды ---

    def torque(self, on: bool) -> dict:
        s = self.state
        if not (self.driver.running and s and s.connected):
            raise Busy("робот не подключен")
        t0 = time.time()
        self.driver.enable_torque() if on else self.driver.disable_torque()
        self._wait_tick(t0)
        return self.state_json()

    def stop(self) -> dict:
        t0 = time.time()
        self.driver.hold()
        self._wait_tick(t0)
        return self.state_json()

    def speed(self, arm: float | None, gripper: float | None) -> dict:
        arm = float(np.clip(arm if arm is not None else self.driver.speed_arm, 1, 60))
        gripper = float(np.clip(gripper if gripper is not None else 2 * arm, 2, 120))
        t0 = time.time()
        self.driver.set_speed(arm, gripper)
        self._wait_tick(t0)
        return {"arm": arm, "gripper": gripper}

    def move(self, body: dict) -> dict:
        s = self._ready()
        if body.get("pose"):
            if body["pose"] != "rest":
                raise ValueError(f"неизвестная поза {body['pose']!r}; есть rest")
            goal = self.rest_pose()
        else:
            joints = body.get("joints") or {}
            goal = np.full(6, np.nan)
            for name, value in joints.items():
                name = ALIASES.get(name, name)
                if name not in JOINTS:
                    raise ValueError(f"неизвестный сустав {name!r}; есть {', '.join(JOINTS)}")
                i = JOINTS.index(name)
                goal[i] = (s.target[i] if body.get("relative") else 0.0) + float(value)
        return self._go(goal, body)

    def cartesian(self, body: dict) -> dict:
        s = self._ready()
        cur = self.robot.task(self.jmap.to_urdf(s.target)[:5])
        goal = cur.copy()
        absolute = any(k in body for k in ("x", "y", "z", "pitch", "roll"))
        if absolute:
            for i, k in enumerate(("x", "y", "z")):
                if body.get(k) is not None:
                    goal[i] = float(body[k]) / 1000.0
            for i, k in ((3, "pitch"), (4, "roll")):
                if body.get(k) is not None:
                    goal[i] = math.radians(float(body[k]))
        else:
            for i, k in enumerate(("dx", "dy", "dz")):
                goal[i] += float(body.get(k, 0.0)) / 1000.0
            goal[3] += math.radians(float(body.get("dpitch", 0.0)))
            goal[4] += math.radians(float(body.get("droll", 0.0)))
        if goal[2] < self.z_min:
            raise ValueError(f"z = {goal[2] * 1000:.0f} мм ниже допустимого {self.z_min * 1000:.0f} мм (стол)")
        target, msg = plan_cartesian(self.robot, self.jmap, s.target, s.limits, goal)
        if target is None:
            raise ValueError(msg)
        return self._go(target, body)

    def _go(self, goal: np.ndarray, body: dict) -> dict:
        wait = bool(body.get("wait", True))
        timeout = float(body.get("timeout", 20.0))
        with self.motion:
            s = self._ready()
            requested = np.where(np.isnan(goal), s.target, goal)
            t0 = time.time()
            self.driver.set_target(goal)
            self._wait_tick(t0)
            s = self.state
            clipped = [JOINTS[i] for i in range(6) if abs(s.target[i] - requested[i]) > 0.05]
            out = {"accepted": True, "requested": _arr(requested), "target": _arr(s.target), "clipped": clipped}
            if clipped:
                self.log(f"Цель обрезана лимитами: {', '.join(clipped)}")
            if wait:
                out["reached"], out["reason"] = self._wait_reached(timeout, s.target.copy())
            out["state"] = self.state_json()
            return out

    def _wait_tick(self, t0: float, timeout: float = 1.0) -> None:
        """Дождаться тика драйвера после t0: команда из очереди применена."""
        end = time.time() + timeout
        while time.time() < end:
            s = self.state
            if s is not None and s.t > t0 + 0.04:
                return
            time.sleep(0.01)

    def _wait_reached(self, timeout: float, target: np.ndarray) -> tuple[bool, str]:
        end = time.monotonic() + timeout
        best, best_t = float("inf"), time.monotonic()
        while time.monotonic() < end:
            s = self.state
            if not (s and s.connected and s.torque_on):
                return False, "момент выключен или связь потеряна"
            if np.any(np.abs(s.target - target) > 0.05):
                return False, "цель изменил другой клиент (СТОП в окне?)"
            err = np.abs(s.present - s.target)
            settled = bool(np.all(np.abs(s.command - s.target) < 1e-3))
            if settled and np.all(err <= TOLERANCE):
                return True, ""
            e = float(np.max(err - TOLERANCE))
            if e < best - 0.1:
                best, best_t = e, time.monotonic()
            elif settled and time.monotonic() - best_t > STALL_S:
                j = JOINTS[int(np.argmax(err - TOLERANCE))]
                return False, f"остановился не дойдя: {j} отстает на {err[JOINTS.index(j)]:.1f} (упор, предмет, нагрузка)"
            time.sleep(0.03)
        return False, "таймаут"

    def camera_jpeg(self, index: int | None, width: int | None) -> bytes:
        if index is None:
            if not self.grabbers:
                raise Busy("камеры не открыты")
            index = min(self.grabbers)
        g = self.grabbers.get(index)
        if g is None:
            raise Busy(f"камера {index} не открыта; открыты {sorted(self.grabbers)}")
        data = g.jpeg(width)
        if data is None:
            raise Busy(g.error or f"камера {index}: кадров еще нет")
        return data


class Handler(BaseHTTPRequestHandler):
    server_version = "robotd/1"
    app: Robotd

    def log_message(self, fmt, *args):  # без строки на каждый запрос
        pass

    def _send(self, code: int, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, fn):
        try:
            fn()
        except Busy as e:
            self._send(409, {"error": str(e)})
        except (ValueError, KeyError, TypeError) as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        app = self.app

        def run():
            if url.path == "/state":
                self._send(200, app.state_json())
            elif url.path == "/diag":
                self._send(200, app.diag_json())
            elif url.path == "/log":
                self._send(200, {"log": list(app.logs)[-int(q.get("n", 50)):]})
            elif url.path == "/cameras":
                opened = {i: {"fps": round(g.fps, 1), "alive": g.is_alive(), "error": g.error} for i, g in app.grabbers.items()}
                self._send(200, {"cameras": app.camera_list, "opened": opened})
            elif url.path.startswith("/camera"):
                tail = url.path[len("/camera"):].strip("/").removesuffix(".jpg")
                idx = int(tail) if tail else None
                self._send(200, app.camera_jpeg(idx, int(q["w"]) if "w" in q else 640), "image/jpeg")
            else:
                self._send(404, {"error": "нет такого пути"})

        self._handle(run)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        app = self.app
        path = urlparse(self.path).path
        routes = {
            "/torque": lambda: app.torque(bool(body.get("on"))),
            "/stop": app.stop,
            "/move": lambda: app.move(body),
            "/cartesian": lambda: app.cartesian(body),
            "/speed": lambda: app.speed(body.get("arm"), body.get("gripper")),
        }
        if path == "/shutdown":
            self._send(200, {"ok": True, "release": bool(body.get("release"))})
            threading.Thread(target=shutdown, args=(bool(body.get("release")),), daemon=True).start()
            return
        if path not in routes:
            self._send(404, {"error": "нет такого пути"})
            return
        self._handle(lambda: self._send(200, routes[path]()))


SERVER: ThreadingHTTPServer | None = None
APP: Robotd | None = None


def shutdown(release: bool = False) -> None:
    try:
        if APP:
            APP.log("Выход: " + ("отключаю момент" if release else "момент не трогаю"))
            APP.close(release)
    finally:
        if SERVER:
            SERVER.shutdown()


def main() -> int:
    global SERVER, APP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="виртуальная рука вместо порта")
    ap.add_argument("--port", help="порт робота; по умолчанию автопоиск")
    ap.add_argument("--cameras", help="индексы камер через запятую; по умолчанию все внешние, '' = без камер")
    ap.add_argument("--http-port", type=int, default=PORT)
    ap.add_argument("--z-min", type=float, default=-30.0, help="мм: ниже по z декартовы цели не принимаются")
    args = ap.parse_args()

    port = None
    if not args.sim:
        ports = [args.port] if args.port else find_ports()
        if not ports:
            print("Робот не найден: нет /dev/tty.usbmodem*. Подключите USB и питание или запустите с --sim.")
            return 1
        port = ports[0]
    if args.cameras is None:
        cams = [c["index"] for c in list_cameras() if not c["builtin"]]
    else:
        cams = [int(x) for x in args.cameras.split(",") if x.strip()]

    try:
        SERVER = ThreadingHTTPServer((HOST, args.http_port), Handler)
    except OSError as e:
        print(f"Порт {HOST}:{args.http_port} занят: robotd уже запущен? ({e})")
        return 1
    SERVER.daemon_threads = True
    APP = Robotd(args.sim, port, cams, args.z_min)
    Handler.app = APP
    APP.log(f"robotd: {'sim' if args.sim else port}, http://{HOST}:{args.http_port}")
    APP.start()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: threading.Thread(target=shutdown, daemon=True).start())
    SERVER.serve_forever(poll_interval=0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
