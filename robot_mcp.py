#!/usr/bin/env python3
"""SO-101 MCP server for an LLM agent: a thin wrapper over the robotd HTTP API (127.0.0.1:8765).

An MCP client (e.g. Claude Code) starts it from an `.mcp.json` entry, see `.mcp.json.example`:
  uv run --no-project --python 3.12 --with "mcp>=1.10,<2" python robot_mcp.py
robotd (./robotd.command) owns the port and cameras; this process only forwards commands,
so the GUI, robotctl and the agent can work with the robot at the same time.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP, Image

URL = "http://127.0.0.1:8765"

mcp = FastMCP(
    "so101",
    instructions=(
        "Controls the SO-101 arm through robotd. Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, "
        "wrist_roll (LeRobot degrees = URDF degrees, signs verified), gripper (0 closed ... 100 open). "
        "EE coordinates in mm in the base frame: +X forward from the base, +Y left, +Z up; pitch 90 deg = gripper down. "
        "Call robot_torque(on=true) before moving; robot_stop stops and holds the pose. "
        "Move in small steps and check robot_camera (wrist camera: top of the frame = +X, left edge = +Y). "
        "Never disable torque in a raised pose: the arm will fall; call robot_go_rest first."
    ),
)


def _call(path: str, body: dict | None = None, timeout: float = 60.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return raw if r.headers.get_content_type() == "image/jpeg" else json.loads(raw)
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.loads(e.read() or b"{}").get("error", str(e))) from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"robotd is not responding ({e.reason}): start ./robotd.command") from None


def _motion(res: dict) -> dict:
    """Compact motion result: arrived or not, why not, what was clipped by limits, where the arm is now."""
    s = res.get("state", {})
    return {k: v for k, v in {
        "reached": res.get("reached"), "reason": res.get("reason") or None, "clipped": res.get("clipped") or None,
        "present": s.get("present"), "target": s.get("target"), "ee": s.get("ee"), "load": s.get("load"),
    }.items() if v is not None}


@mcp.tool()
def robot_state() -> dict:
    """State: joints (present/target), torque, limits, load, temperature, voltage, EE pose (mm, deg)."""
    return _call("/state")


@mcp.tool()
def robot_torque(on: bool) -> dict:
    """Enable (Goal = Present, the arm holds its current pose) or disable torque. Disabling in a raised pose drops the arm."""
    s = _call("/torque", {"on": on})
    return {"torque_on": s.get("torque_on"), "present": s.get("present")}


@mcp.tool()
def robot_stop() -> dict:
    """Stop immediately: goal = current position, hold."""
    s = _call("/stop", {})
    return {"present": s.get("present"), "ee": s.get("ee")}


@mcp.tool()
def robot_move_joints(joints: dict[str, float], relative: bool = False, wait: bool = True, timeout_s: float = 20.0) -> dict:
    """Move joints. joints: {"elbow_flex": 30, "gripper": 80} (short names pan/lift/elbow/wrist/roll/grip work too).
    relative=True: add to the current goal. Goals are clipped to limits; wait=True waits until the arm arrives."""
    return _motion(_call("/move", {"joints": joints, "relative": relative, "wait": wait, "timeout": timeout_s}, timeout_s + 10))


@mcp.tool()
def robot_move_cartesian(dx_mm: float = 0, dy_mm: float = 0, dz_mm: float = 0, dpitch_deg: float = 0,
                         droll_deg: float = 0, wait: bool = True) -> dict:
    """Move the EE relative to the current goal: mm along base axes (+X forward, +Y left, +Z up), pitch/roll in deg.
    IK over 5 joints with limit and singularity checks; a refusal is an error with the reason."""
    body = {"dx": dx_mm, "dy": dy_mm, "dz": dz_mm, "dpitch": dpitch_deg, "droll": droll_deg, "wait": wait}
    return _motion(_call("/cartesian", body))


@mcp.tool()
def robot_move_to_pose(x_mm: float, y_mm: float, z_mm: float, pitch_deg: float | None = None,
                       roll_deg: float | None = None, wait: bool = True) -> dict:
    """Move the EE to an absolute pose (mm in the base frame). pitch 90 = gripper pointing straight down; None = keep."""
    body = {"x": x_mm, "y": y_mm, "z": z_mm, "pitch": pitch_deg, "roll": roll_deg, "wait": wait}
    return _motion(_call("/cartesian", {k: v for k, v in body.items() if v is not None}))


@mcp.tool()
def robot_gripper(percent: float, wait: bool = True) -> dict:
    """Gripper: 0 = closed, 100 = open. When grasping an object the reply is reached=false with "stopped short": that is expected."""
    return _motion(_call("/move", {"joints": {"gripper": percent}, "wait": wait, "timeout": 10}))


@mcp.tool()
def robot_go_rest() -> dict:
    """Fold the arm into the rest pose (recorded when robotd connected to the resting robot with torque off).
    After that torque can be disabled safely: the arm is already resting."""
    return _motion(_call("/move", {"pose": "rest", "wait": True, "timeout": 40}, 50))


@mcp.tool()
def robot_set_speed(arm_deg_s: float, gripper_pct_s: float | None = None) -> dict:
    """Speed: arm 1-60 deg/s (default 20), gripper %/s (default 2x the arm)."""
    return _call("/speed", {"arm": arm_deg_s, "gripper": gripper_pct_s})


@mcp.tool()
def robot_camera(index: int | None = None, width: int = 640) -> Image:
    """Camera snapshot (default: the first open camera, on the wrist). width: JPEG width in pixels."""
    q = f"/camera/{'' if index is None else index}?w={int(width)}"
    return Image(data=_call(q), format="jpeg")


@mcp.tool()
def robot_log(n: int = 30) -> list[str]:
    """Latest robotd log messages: connection, torque, limit clipping, link errors."""
    return _call(f"/log?n={int(n)}")["log"]


if __name__ == "__main__":
    mcp.run()
