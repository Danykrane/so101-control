"""SO-101 kinematics from URDF: FK, numerical IK, limits, mapping motor angles to URDF.

No Qt and no hardware: the module is meant to move into robotd and later to C++.

Units:
- "motor" values: 5 joints in LeRobot degrees (DEGREES, 0 = middle of the calibration
  range), gripper in percent 0..100 (RANGE_0_100);
- URDF: radians, meters.
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
    lower: float  # rad
    upper: float  # rad
    revolute: bool


class Robot:
    """Chain base_link -> gripper_frame_link plus the branch of the moving gripper jaw."""

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
            raise ValueError(f"Unexpected URDF chain: {names}")
        self.gripper = self.joints["gripper"]
        self.lower = np.array([self.joints[n].lower for n in JOINTS])
        self.upper = np.array([self.joints[n].upper for n in JOINTS])
        # Tip of the moving jaw: the EE point in the jaw frame at q_gripper = 0 (closed).
        frames = self.frames(np.zeros(6))
        jaw0 = frames["jaw"]
        self._jaw_tip = (np.linalg.inv(jaw0) @ np.append(frames["ee"][:3, 3], 1.0))[:3]

    def frames(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Frames (4x4, in the base frame): one per chain joint, plus 'ee' and 'jaw'."""
        q = np.asarray(q, dtype=float)
        out: dict[str, np.ndarray] = {}
        T = np.eye(4)
        i = 0
        for j in self.chain:
            T = T @ j.origin
            if j.revolute:
                out[j.name] = T.copy()  # joint frame before rotation: axis and pivot point
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
        """EE pose (4x4) for q (rad, 5 or 6 values)."""
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
        """Points for 3D drawing: (joints Nx3, jaw tip 3). Joints: base, 5 joints, EE."""
        f = self.frames(q)
        pts = [np.zeros(3)] + [f[n][:3, 3] for n in ARM] + [f["ee"][:3, 3]]
        jaw_tip = (f["jaw"] @ np.append(self._jaw_tip, 1.0))[:3]
        return np.array(pts), jaw_tip

    # --- task space: x, y, z, pitch, roll ---------------------------------------------

    def task(self, q: np.ndarray) -> np.ndarray:
        """[x, y, z (m), pitch (rad), wrist_roll (rad)].

        pitch: tilt of the approach axis (gripper z) from horizontal in the arm plane,
        > 0 is down. The arm plane is defined by the shoulder_lift axis, so pitch stays continuous past 90 deg.
        """
        T = self.fk(q)
        n = self._lift_axis(q)
        u = np.cross(n, [0.0, 0.0, 1.0])  # horizontal in the arm plane, +X at pan=0
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
        """Damped least squares over 5 joints. target = task vector. Returns (q, converged).

        Adaptive damping (Levenberg-Marquardt): a step is accepted only if the error decreased,
        otherwise damping grows. This keeps IK close to q0 near the singularity
        (straight elbow) and near the limits.
        """
        lower = self.lower[:5] if lower is None else lower[:5]
        upper = self.upper[:5] if upper is None else upper[:5]
        w = np.array([1.0, 1.0, 1.0, ang_weight, ang_weight])  # meters vs radians: 1 rad ~ 10 cm
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
    """URDF angle = sign * (motor value - zero), in degrees. Same for the gripper (percent ~ degrees)."""

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
            raise ValueError(f"{path}: sign must be +1 or -1")
        return cls(sign, zero)

    def to_urdf(self, motor: np.ndarray) -> np.ndarray:
        """Motor values (6) -> URDF radians (6)."""
        return np.radians(self.sign * (np.asarray(motor, dtype=float) - self.zero))

    def to_motor(self, urdf: np.ndarray) -> np.ndarray:
        """URDF radians (5 or 6) -> motor values of the same length."""
        n = len(urdf)
        return self.zero[:n] + self.sign[:n] * np.degrees(np.asarray(urdf, dtype=float))

    def motor_limits(self, robot: Robot) -> tuple[np.ndarray, np.ndarray]:
        """URDF limits in motor units (sign taken into account)."""
        a, b = self.to_motor(robot.lower), self.to_motor(robot.upper)
        return np.minimum(a, b), np.maximum(a, b)


# --- Cartesian motion: shared by the GUI, robotd and MCP ------------------------------------

MAX_IK_JUMP_DEG = 15.0  # a larger jump in one sub-step means a nearby singularity: refuse the move


def plan_cartesian(
    robot: Robot,
    jmap: JointMap,
    target: np.ndarray,
    limits,
    goal: np.ndarray,
    step_m: float = 0.01,
    step_rad: float = math.radians(5),
) -> tuple[np.ndarray | None, str]:
    """Joint target for a Cartesian goal = [x, y, z (m), pitch, roll (rad)].

    target: current joint target (6, motor units), limits: Limits in motor units.
    The path to goal is split into sub-steps <= step_m / step_rad and IK is chained from the current pose
    so it does not jump to another solution branch. -> (target 6 with NaN for the gripper, "") or (None, reason).
    """
    q = jmap.to_urdf(target)
    a, b = jmap.to_urdf(limits.lower), jmap.to_urdf(limits.upper)
    lo = np.minimum(np.maximum(np.minimum(a, b), robot.lower), q)  # a joint outside its limit is not pulled to the boundary
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
            return None, f"IK: waypoint {k}/{n} is unreachable (limits or workspace)"
        jump = math.degrees(float(np.max(np.abs(new - sol))))
        if jump > MAX_IK_JUMP_DEG:
            return None, f"IK: sub-step needs a {jump:.0f} deg joint jump (singularity nearby)"
        sol = new
    out = np.full(6, np.nan)
    out[:5] = jmap.to_motor(sol)
    return out, ""
