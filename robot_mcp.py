#!/usr/bin/env python3
"""MCP-сервер SO-101 для Claude: тонкая обертка над HTTP API robotd (127.0.0.1:8765).

Claude Code запускает его сам по `.mcp.json` в корне LeRobotSo101:
  uv run --no-project --python 3.12 --with "mcp>=1.10,<2" python start_use_mac/robot_mcp.py
Порт и камеры держит robotd (./robotd.command), этот процесс только пересылает команды,
поэтому окно, robotctl и Claude могут работать с роботом одновременно.
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
        "Управление рукой SO-101 через robotd. Суставы: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, "
        "wrist_roll (градусы LeRobot = градусы URDF, знаки проверены), gripper (0 закрыт … 100 открыт). "
        "Координаты EE в мм в системе базы: +X вперед от базы, +Y влево, +Z вверх; pitch 90° = гриппер вниз. "
        "Перед движением нужен robot_torque(on=true); robot_stop останавливает и держит позу. "
        "Двигаться небольшими шагами, проверять robot_camera (камера на гриппере: верх кадра = +X, левый край = +Y). "
        "Не отключать момент в поднятой позе: рука упадет; сначала robot_go_rest."
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
        raise RuntimeError(f"robotd не отвечает ({e.reason}): запустите start_use_mac/robotd.command") from None


def _motion(res: dict) -> dict:
    """Ответ движения без лишнего: дошел ли, почему нет, что обрезали лимиты, где сейчас рука."""
    s = res.get("state", {})
    return {k: v for k, v in {
        "reached": res.get("reached"), "reason": res.get("reason") or None, "clipped": res.get("clipped") or None,
        "present": s.get("present"), "target": s.get("target"), "ee": s.get("ee"), "load": s.get("load"),
    }.items() if v is not None}


@mcp.tool()
def robot_state() -> dict:
    """Состояние: суставы (present/target), момент, лимиты, нагрузка, температура, напряжение, поза EE (мм, °)."""
    return _call("/state")


@mcp.tool()
def robot_torque(on: bool) -> dict:
    """Включить (Goal = Present, рука держит текущую позу) или отключить момент. Отключение в поднятой позе роняет руку."""
    s = _call("/torque", {"on": on})
    return {"torque_on": s.get("torque_on"), "present": s.get("present")}


@mcp.tool()
def robot_stop() -> dict:
    """Немедленно остановиться: цель = текущая позиция, удержание."""
    s = _call("/stop", {})
    return {"present": s.get("present"), "ee": s.get("ee")}


@mcp.tool()
def robot_move_joints(joints: dict[str, float], relative: bool = False, wait: bool = True, timeout_s: float = 20.0) -> dict:
    """Двинуть суставы. joints: {"elbow_flex": 30, "gripper": 80} (короткие имена pan/lift/elbow/wrist/roll/grip тоже).
    relative=True: прибавить к текущей цели. Цель клипуется по лимитам; wait=True ждет, пока рука дойдет."""
    return _motion(_call("/move", {"joints": joints, "relative": relative, "wait": wait, "timeout": timeout_s}, timeout_s + 10))


@mcp.tool()
def robot_move_cartesian(dx_mm: float = 0, dy_mm: float = 0, dz_mm: float = 0, dpitch_deg: float = 0,
                         droll_deg: float = 0, wait: bool = True) -> dict:
    """Сдвинуть EE относительно текущей цели: мм по осям базы (+X вперед, +Y влево, +Z вверх), pitch/roll в °.
    IK по 5 суставам с проверкой лимитов и сингулярности; отказ = ошибка с причиной."""
    body = {"dx": dx_mm, "dy": dy_mm, "dz": dz_mm, "dpitch": dpitch_deg, "droll": droll_deg, "wait": wait}
    return _motion(_call("/cartesian", body))


@mcp.tool()
def robot_move_to_pose(x_mm: float, y_mm: float, z_mm: float, pitch_deg: float | None = None,
                       roll_deg: float | None = None, wait: bool = True) -> dict:
    """Перевести EE в абсолютную позу (мм в системе базы). pitch 90 = гриппер вертикально вниз; None = не менять."""
    body = {"x": x_mm, "y": y_mm, "z": z_mm, "pitch": pitch_deg, "roll": roll_deg, "wait": wait}
    return _motion(_call("/cartesian", {k: v for k, v in body.items() if v is not None}))


@mcp.tool()
def robot_gripper(percent: float, wait: bool = True) -> dict:
    """Гриппер: 0 = закрыт, 100 = открыт. При захвате предмета ответ reached=false с причиной «не дошел» — это нормально."""
    return _motion(_call("/move", {"joints": {"gripper": percent}, "wait": wait, "timeout": 10}))


@mcp.tool()
def robot_go_rest() -> dict:
    """Сложить руку в позу покоя (записана, когда robotd подключился к лежащему роботу без момента).
    После этого момент можно отключать: рука уже лежит."""
    return _motion(_call("/move", {"pose": "rest", "wait": True, "timeout": 40}, 50))


@mcp.tool()
def robot_set_speed(arm_deg_s: float, gripper_pct_s: float | None = None) -> dict:
    """Скорость: рука 1–60 °/с (по умолчанию 20), гриппер %/с (по умолчанию 2× руки)."""
    return _call("/speed", {"arm": arm_deg_s, "gripper": gripper_pct_s})


@mcp.tool()
def robot_camera(index: int | None = None, width: int = 640) -> Image:
    """Снимок камеры (по умолчанию первая открытая, на гриппере). width: ширина JPEG в пикселях."""
    q = f"/camera/{'' if index is None else index}?w={int(width)}"
    return Image(data=_call(q), format="jpeg")


@mcp.tool()
def robot_log(n: int = 30) -> list[str]:
    """Последние сообщения robotd: подключение, момент, обрезка лимитами, ошибки связи."""
    return _call(f"/log?n={int(n)}")["log"]


if __name__ == "__main__":
    mcp.run()
