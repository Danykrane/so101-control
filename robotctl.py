#!/usr/bin/env python3
"""robotctl: robotd commands from the terminal (also usable by an LLM agent through a shell).

  robotctl.py state                      joints, torque, EE pose
  robotctl.py torque on|off
  robotctl.py stop
  robotctl.py move pan=10 elbow=-5       absolute, deg (gripper in %)
  robotctl.py move wrist=5 --rel         relative to the current goal
  robotctl.py move --rest                to the rest pose (folded, recorded on connect without torque)
  robotctl.py cart --dz 20 --dx -10      Cartesian step, mm and deg
  robotctl.py goto 200 0 80 90           absolute EE pose: x y z (mm), pitch (deg), [roll]
  robotctl.py grip 80                    gripper, % (0 closed, 100 open)
  robotctl.py speed 15                   arm speed in deg/s (gripper 2x)
  robotctl.py snap --out cam.jpg         camera snapshot
  robotctl.py log 30
  robotctl.py shutdown [--release]

Joints: pan, lift, elbow, wrist (flex), roll, grip or full names. Exit code 0 = ok,
2 = command not executed (torque off, IK, did not arrive), 1 = robotd is not running.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8765"
SHORT = ["pan", "lift", "elbow", "wrist", "roll", "grip"]


def call(path: str, body: dict | None = None, timeout: float = 60.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return raw if r.headers.get_content_type() == "image/jpeg" else json.loads(raw)
    except urllib.error.HTTPError as e:
        return {"error": json.loads(e.read() or b"{}").get("error", str(e)), "http": e.code}
    except urllib.error.URLError as e:
        print(f"robotd is not responding at {URL}: {e.reason}. Start ./robotd.command", file=sys.stderr)
        sys.exit(1)


def fmt_state(s: dict) -> str:
    if not s.get("connected"):
        return f"not connected ({s.get('port')}) {s.get('error', '')}"
    p, t = s["present"], s["target"]
    joints = "  ".join(f"{n}={v:.1f}" + (f"→{tv:.1f}" if abs(tv - v) > 0.5 else "") for n, v, tv in zip(SHORT, p, t))
    e = s["ee"]
    torque = "ON" if s["torque_on"] else ("partial " + str(s["torque"]) if any(s["torque"]) else "off")
    load = max(abs(x) for x in s["load"] if x is not None) if any(x is not None for x in s["load"]) else 0
    temp = max(x for x in s["temperature"] if x is not None) if any(x is not None for x in s["temperature"]) else 0
    return (f"{s['mode']} {s['port']} | torque {torque} | {s['loop_hz']} Hz | {s['voltage']} V, max {temp:.0f} °C, max load {load:.0f}\n"
            f"  {joints}\n"
            f"  EE: x {e['x']} y {e['y']} z {e['z']} mm, pitch {e['pitch']}°, roll {e['roll']}°"
            + (f"\n  sag compensation, deg: {[round(b, 1) for b in s['bias'][:5]]}" if any(abs(b or 0) > 0.05 for b in s.get("bias", [])) else "")
            + (f"\n  error: {s['error']}" if s.get("error") else ""))


def report(res: dict) -> int:
    if "http" in res:  # robotd refused (4xx/5xx)
        print("ERROR:", res["error"])
        return 2
    if "state" in res:
        if res.get("clipped"):
            print("clipped by limits:", ", ".join(res["clipped"]))
        if "reached" in res:
            print("arrived" if res["reached"] else f"DID NOT arrive: {res['reason']}")
        print(fmt_state(res["state"]))
        return 0 if res.get("reached", True) else 2
    print(fmt_state(res) if "connected" in res else json.dumps(res, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the raw robotd reply")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("state")
    sub.add_parser("diag")
    p = sub.add_parser("torque"); p.add_argument("on", choices=["on", "off"])
    sub.add_parser("stop")
    p = sub.add_parser("move"); p.add_argument("joints", nargs="*", help="name=value")
    p.add_argument("--rel", action="store_true"); p.add_argument("--rest", action="store_true")
    p.add_argument("--no-wait", action="store_true"); p.add_argument("--timeout", type=float, default=20)
    p = sub.add_parser("cart")
    for k in ("dx", "dy", "dz", "dpitch", "droll"):
        p.add_argument(f"--{k}", type=float, default=0.0)
    p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("goto"); p.add_argument("xyz_pitch_roll", nargs="+", type=float)
    p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("grip"); p.add_argument("percent", type=float); p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("speed"); p.add_argument("arm", type=float); p.add_argument("gripper", type=float, nargs="?")
    p = sub.add_parser("snap"); p.add_argument("--cam", type=int); p.add_argument("--width", type=int, default=640)
    p.add_argument("--out", default="so101_cam.jpg")
    p = sub.add_parser("log"); p.add_argument("n", type=int, nargs="?", default=30)
    p = sub.add_parser("shutdown"); p.add_argument("--release", action="store_true")
    a = ap.parse_args()

    if a.cmd == "state":
        res = call("/state")
    elif a.cmd == "diag":
        res = call("/diag")
    elif a.cmd == "torque":
        res = call("/torque", {"on": a.on == "on"})
    elif a.cmd == "stop":
        res = call("/stop", {})
    elif a.cmd == "move":
        if a.rest:
            body = {"pose": "rest"}
        else:
            body = {"joints": dict((kv.split("=", 1)[0], float(kv.split("=", 1)[1])) for kv in a.joints), "relative": a.rel}
        body.update(wait=not a.no_wait, timeout=a.timeout)
        res = call("/move", body, timeout=a.timeout + 10)
    elif a.cmd == "cart":
        res = call("/cartesian", {k: getattr(a, k) for k in ("dx", "dy", "dz", "dpitch", "droll")} | {"wait": not a.no_wait})
    elif a.cmd == "goto":
        v = a.xyz_pitch_roll
        body = dict(zip(("x", "y", "z", "pitch", "roll"), v)) | {"wait": not a.no_wait}
        res = call("/cartesian", body)
    elif a.cmd == "grip":
        res = call("/move", {"joints": {"gripper": a.percent}, "wait": not a.no_wait, "timeout": 10})
    elif a.cmd == "speed":
        res = call("/speed", {"arm": a.arm, "gripper": a.gripper})
    elif a.cmd == "snap":
        res = call(f"/camera/{'' if a.cam is None else a.cam}?w={a.width}")
        if isinstance(res, bytes):
            with open(a.out, "wb") as f:
                f.write(res)
            print(a.out)
            return 0
    elif a.cmd == "log":
        res = call(f"/log?n={a.n}")
        if "log" in res:
            print("\n".join(res["log"]))
            return 0
    else:
        res = call("/shutdown", {"release": a.release})

    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 2 if "http" in res else 0
    return report(res)


if __name__ == "__main__":
    sys.exit(main())
