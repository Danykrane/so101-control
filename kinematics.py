"""Кинематика SO-101 по URDF: FK, численный IK, лимиты, отображение моторных углов в URDF.

Без Qt и без железа: модуль потом переедет в robotd и в C++.

Единицы:
- «моторные» значения: 5 суставов в градусах LeRobot (DEGREES, 0 = середина диапазона
  калибровки), гриппер в процентах 0..100 (RANGE_0_100);
- URDF: радианы, метры.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "urdf" / "so101_new_calib.urdf"  # from TheRobotStudio/SO-ARM100, Apache-2.0
JOINT_MAP_PATH = HERE / "config" / "joint_map.json"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = JOINTS[:5]
EE_LINK = "gripper_frame_link"


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _transform(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray  # (3,)
    lower: float  # рад
    upper: float  # рад
    revolute: bool


class Robot:
    """Цепь base_link → gripper_frame_link плюс ветка подвижной губки гриппера."""

    def __init__(self, urdf_path: Path = URDF_PATH):
        root = ET.parse(urdf_path).getroot()
        self.joints: dict[str, Joint] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.zeros(3)
            lim = j.find("limit")
            lower = float(lim.get("lower")) if lim is not None else 0.0
            upper = float(lim.get("upper")) if lim is not None else 0.0
            self.joints[j.get("name")] = Joint(
                name=j.get("name"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                origin=_transform(xyz, rpy),
                axis=axis,
                lower=lower,
                upper=upper,
                revolute=j.get("type") == "revolute",
            )
        by_child = {j.child: j for j in self.joints.values()}
        chain = []
        link = EE_LINK
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].parent
        self.chain: list[Joint] = chain[::-1]
        names = [j.name for j in self.chain if j.revolute]
        if tuple(names) != ARM:
            raise ValueError(f"Неожиданная цепь URDF: {names}")
        self.gripper = self.joints["gripper"]
        self.lower = np.array([self.joints[n].lower for n in JOINTS])
        self.upper = np.array([self.joints[n].upper for n in JOINTS])
        # Кончик подвижной губки: точка EE в системе губки при q_gripper = 0 (закрыт).
        frames = self.frames(np.zeros(6))
        jaw0 = frames["jaw"]
        self._jaw_tip = (np.linalg.inv(jaw0) @ np.append(frames["ee"][:3, 3], 1.0))[:3]

    def frames(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Системы координат (4x4, в базе): по одной на каждый сустав цепи, 'ee', 'jaw'."""
        q = np.asarray(q, dtype=float)
        out: dict[str, np.ndarray] = {}
        T = np.eye(4)
        i = 0
        for j in self.chain:
            T = T @ j.origin
            if j.revolute:
                out[j.name] = T.copy()  # система сустава до поворота: ось и точка шарнира
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j.axis, q[i])
                T = T @ R
                i += 1
            if j.child == "gripper_link":
                gripper_link = T.copy()
        out["ee"] = T
        G = gripper_link @ self.gripper.origin
        out["gripper"] = G.copy()
        R = np.eye(4)
        R[:3, :3] = _axis_angle(self.gripper.axis, q[5] if len(q) > 5 else 0.0)
        out["jaw"] = G @ R
        return out

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Поза EE (4x4) для q (рад, 5 или 6 значений)."""
        T = np.eye(4)
        i = 0
        for j in self.chain:
            T = T @ j.origin
            if j.revolute:
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j.axis, q[i])
                T = T @ R
                i += 1
        return T

    def skeleton(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Точки для 3D: (шарниры Nx3, кончик губки 3). Шарниры: база, 5 суставов, EE."""
        f = self.frames(q)
        pts = [np.zeros(3)] + [f[n][:3, 3] for n in ARM] + [f["ee"][:3, 3]]
        jaw_tip = (f["jaw"] @ np.append(self._jaw_tip, 1.0))[:3]
        return np.array(pts), jaw_tip

    # --- task space: x, y, z, pitch, roll ---------------------------------------------

    def task(self, q: np.ndarray) -> np.ndarray:
        """[x, y, z (м), pitch (рад), wrist_roll (рад)].

        pitch: наклон оси подхода (z гриппера) относительно горизонтали в плоскости руки,
        > 0 вниз. Плоскость руки задает ось shoulder_lift, поэтому pitch непрерывен и за 90°.
        """
        T = self.fk(q)
        n = self._lift_axis(q)
        u = np.cross(n, [0.0, 0.0, 1.0])  # горизонталь в плоскости руки, при pan=0 это +X
        a = T[:3, 2]
        pitch = math.atan2(-a[2], float(np.dot(a, u)))
        return np.array([*T[:3, 3], pitch, q[4]])

    def _lift_axis(self, q: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        for j in self.chain:
            T = T @ j.origin
            if j.name == "shoulder_lift":
                return T[:3, :3] @ j.axis
            if j.revolute:
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j.axis, q[0])
                T = T @ R
        raise AssertionError

    def ik(
        self,
        target: np.ndarray,
        q0: np.ndarray,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        iters: int = 200,
        damping: float = 0.003,
        ang_weight: float = 0.1,
    ) -> tuple[np.ndarray, bool]:
        """Damped least squares по 5 суставам. target = task-вектор. Возвращает (q, сошлось).

        Демпфирование адаптивное (Левенберг–Марквардт): шаг принимается, только если ошибка
        уменьшилась, иначе демпфирование растет. Так IK не уходит далеко от q0 у сингулярности
        (локоть выпрямлен) и у лимитов.
        """
        lower = self.lower[:5] if lower is None else lower[:5]
        upper = self.upper[:5] if upper is None else upper[:5]
        w = np.array([1.0, 1.0, 1.0, ang_weight, ang_weight])  # метры против радиан: 1 рад ≈ 10 см
        eps = 1e-6

        def error(q):
            e = target - self.task(q)
            e[3] = (e[3] + math.pi) % (2 * math.pi) - math.pi
            return e

        q = np.clip(np.asarray(q0, dtype=float)[:5].copy(), lower, upper)
        e = error(q)
        lam = damping
        for _ in range(iters):
            if np.all(np.abs(e[:3]) < 2e-5) and np.all(np.abs(e[3:]) < math.radians(0.01)):
                return q, True
            J = np.empty((5, 5))
            for k in range(5):
                dq = np.zeros(5)
                dq[k] = eps
                J[:, k] = (e - error(q + dq)) / eps  # d(task)/dq
            Jw, ew = w[:, None] * J, w * e
            step = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam**2 * np.eye(5), ew)
            n = np.linalg.norm(step)
            if n > 0.1:
                step *= 0.1 / n
            q_new = np.clip(q + step, lower, upper)
            e_new = error(q_new)
            if np.linalg.norm(w * e_new) < np.linalg.norm(ew):
                q, e, lam = q_new, e_new, max(lam * 0.5, 1e-4)
            else:
                lam *= 4
                if lam > 10:
                    break
        return q, False


@dataclass
class JointMap:
    """URDF-угол = sign · (моторное значение − zero), в градусах. Гриппер так же (проценты ≈ градусы)."""

    sign: np.ndarray  # (6,)
    zero: np.ndarray  # (6,)

    @classmethod
    def identity(cls) -> JointMap:
        return cls(np.ones(6), np.zeros(6))

    @classmethod
    def load(cls, path: Path = JOINT_MAP_PATH) -> JointMap:
        if not path.exists():
            return cls.identity()
        data = json.loads(path.read_text(encoding="utf-8"))
        sign = np.array([float(data.get(n, {}).get("sign", 1)) for n in JOINTS])
        zero = np.array([float(data.get(n, {}).get("zero_deg", 0.0)) for n in JOINTS])
        if not np.all(np.isin(sign, (-1.0, 1.0))):
            raise ValueError(f"{path}: sign должен быть +1 или -1")
        return cls(sign, zero)

    def to_urdf(self, motor: np.ndarray) -> np.ndarray:
        """Моторные значения (6) → радианы URDF (6)."""
        return np.radians(self.sign * (np.asarray(motor, dtype=float) - self.zero))

    def to_motor(self, urdf: np.ndarray) -> np.ndarray:
        """Радианы URDF (5 или 6) → моторные значения той же длины."""
        n = len(urdf)
        return self.zero[:n] + self.sign[:n] * np.degrees(np.asarray(urdf, dtype=float))

    def motor_limits(self, robot: Robot) -> tuple[np.ndarray, np.ndarray]:
        """Лимиты URDF в моторных единицах (с учетом знака)."""
        a, b = self.to_motor(robot.lower), self.to_motor(robot.upper)
        return np.minimum(a, b), np.maximum(a, b)


# --- декартово движение: общее для окна, robotd и MCP ------------------------------------

MAX_IK_JUMP_DEG = 15.0  # больше за один подшаг = рядом сингулярность, движение не выполняем


def plan_cartesian(
    robot: Robot,
    jmap: JointMap,
    target: np.ndarray,
    limits,
    goal: np.ndarray,
    step_m: float = 0.01,
    step_rad: float = math.radians(5),
) -> tuple[np.ndarray | None, str]:
    """Цель суставов для декартовой цели goal = [x, y, z (м), pitch, roll (рад)].

    target: текущая цель суставов (6, моторные единицы), limits: Limits в моторных единицах.
    Путь до goal режется на подшаги ≤ step_m / step_rad, IK решается по цепочке от текущей позы,
    чтобы не перескочить на другую ветвь решения. → (цель 6 с NaN у гриппера, "") или (None, причина).
    """
    q = jmap.to_urdf(target)
    a, b = jmap.to_urdf(limits.lower), jmap.to_urdf(limits.upper)
    lo = np.minimum(np.maximum(np.minimum(a, b), robot.lower), q)  # сустав за лимитом не тянем на границу
    hi = np.maximum(np.minimum(np.maximum(a, b), robot.upper), q)
    start = robot.task(q[:5])
    goal = np.asarray(goal, dtype=float)
    d = goal - start
    d[3] = (d[3] + math.pi) % (2 * math.pi) - math.pi
    n = max(1, int(math.ceil(max(np.linalg.norm(d[:3]) / step_m, np.max(np.abs(d[3:])) / step_rad))))
    sol = q[:5]
    for k in range(1, n + 1):
        new, ok = robot.ik(start + d * k / n, sol, lo, hi)
        if not ok:
            return None, f"IK: точка {k}/{n} недостижима (лимиты или рабочая зона)"
        jump = math.degrees(float(np.max(np.abs(new - sol))))
        if jump > MAX_IK_JUMP_DEG:
            return None, f"IK: подшаг требует поворота сустава на {jump:.0f}° (рядом сингулярность)"
        sol = new
    out = np.full(6, np.nan)
    out[:5] = jmap.to_motor(sol)
    return out, ""
