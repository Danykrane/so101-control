import time
from dataclasses import dataclass

import numpy as np
import pytest

import driver as drv
import kinematics as kin
from safety import MAX_STEP_PER_TICK, Limits, clip_target, limit_lead, step_toward

ROBOT = kin.Robot()


# --- kinematics ---------------------------------------------------------------------------


def random_q(rng, margin=0.15):
    lo, hi = ROBOT.lower[:5] + margin, ROBOT.upper[:5] - margin
    return rng.uniform(lo, hi)


def test_fk_zero_pose_is_sane():
    T = ROBOT.fk(np.zeros(5))
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    x, y, z = T[:3, 3]
    assert 0.3 < x < 0.45 and abs(y) < 0.01 and 0.15 < z < 0.3  # stretched forward


def min_singular(q):
    f, J = ROBOT.task(q), np.empty((5, 5))
    for k in range(5):
        dq = np.zeros(5)
        dq[k] = 1e-6
        J[:, k] = (ROBOT.task(q + dq) - f) / 1e-6
    return np.linalg.svd(np.diag([1, 1, 1, 0.1, 0.1]) @ J, compute_uv=False)[-1]


def test_ik_of_fk_returns_q():
    rng = np.random.default_rng(0)
    n = 0
    while n < 50:
        q = random_q(rng)
        if min_singular(q) < 0.01:  # elbow almost straight (about -75 deg): solution is not unique
            continue
        n += 1
        target = ROBOT.task(q)
        sol, ok = ROBOT.ik(target, q + rng.normal(0, np.radians(4), 5))
        assert ok
        assert np.allclose(ROBOT.task(sol)[:3], target[:3], atol=1e-4)
        assert np.degrees(np.max(np.abs(sol - q))) < 0.5


def test_ik_respects_limits_and_reports_unreachable():
    q = np.zeros(5)
    far = ROBOT.task(q) + np.array([0.5, 0, 0, 0, 0])  # 50 cm beyond the stretched arm
    sol, ok = ROBOT.ik(far, q)
    assert not ok
    assert np.all(sol >= ROBOT.lower[:5] - 1e-12) and np.all(sol <= ROBOT.upper[:5] + 1e-12)


def test_cartesian_step_is_small_in_joint_space():
    q = np.radians([0, -60, 60, 50, 0])
    t = ROBOT.task(q) + np.array([0.0, 0.0, 0.005, 0, 0])  # +5 mm along Z
    sol, ok = ROBOT.ik(t, q)
    assert ok and np.degrees(np.max(np.abs(sol - q))) < 5


def test_joint_map_roundtrip_and_limits():
    m = kin.JointMap(np.array([1, -1, 1, -1, 1, 1.0]), np.array([0, 10, -5, 0, 2, 0.0]))
    motor = np.array([10, 20, 30, -40, 50, 60.0])
    assert np.allclose(m.to_motor(m.to_urdf(motor)), motor)
    lo, hi = m.motor_limits(ROBOT)
    assert np.all(lo < hi)
    assert np.allclose(kin.JointMap.load().sign, 1)  # default file: sign=+1


# --- safety -----------------------------------------------------------------------------------

LIM = Limits(np.array([-90.0] * 5 + [0.0]), np.array([90.0] * 5 + [100.0]))


def test_clip_target_limits_and_outside_start():
    old = np.zeros(6)
    assert np.allclose(clip_target(np.full(6, 500.0), old, LIM), [90] * 5 + [100])
    # joint already outside its limit: does not jump to the boundary but may move back inside
    old = np.array([120.0, 0, 0, 0, 0, 0])
    assert clip_target(old, old, LIM)[0] == 120.0
    assert clip_target(old + [10, 0, 0, 0, 0, 0], old, LIM)[0] == 120.0
    assert clip_target(old - [10, 0, 0, 0, 0, 0], old, LIM)[0] == 110.0


def test_step_toward_is_bounded_and_synchronous():
    c = np.zeros(6)
    t = np.array([100.0, 50.0, 0, 0, 0, 100.0])
    n = step_toward(c, t, max_arm=10, max_gripper=10)  # no more than 2 deg per tick
    assert np.max(np.abs(n[:5])) == pytest.approx(MAX_STEP_PER_TICK)
    assert n[1] == pytest.approx(n[0] / 2)  # synchronous: proportions are kept
    assert n[5] == pytest.approx(MAX_STEP_PER_TICK)
    near = np.array([0.5, 0, 0, 0, 0, 0.5])
    assert np.allclose(step_toward(c, near, 1, 1), near)


def test_limit_lead():
    assert np.allclose(limit_lead(np.full(6, 30.0), np.zeros(6), lead=10), 10)


def test_intersect():
    a = Limits(np.full(6, -10.0), np.full(6, 10.0))
    b = Limits(np.full(6, 0.0), np.full(6, 20.0))
    i = a.intersect(b)
    assert np.allclose(i.lower, 0) and np.allclose(i.upper, 10)


# --- driver --------------------------------------------------------------------------------


def test_raw_motor_roundtrip():
    rmin = np.array([700, 800, 900, 850, 0, 2000])
    rmax = np.array([3400, 3300, 3100, 3200, 4095, 3400])
    raw = np.array([2048, 1000, 3000, 2000, 4000, 2700])
    m = drv.raw_to_motor(raw, rmin, rmax)
    assert m[5] == pytest.approx(50.0)
    assert m[4] == pytest.approx((4000 - 2047.5) * 360 / 4095)
    assert np.array_equal(drv.motor_to_raw(m, rmin, rmax), raw)
    lim = drv.calibration_limits(rmin, rmax)
    assert lim.lower[5] == 0 and lim.upper[5] == 100 and lim.lower[0] == pytest.approx(-lim.upper[0])


def wait_for(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def sim():
    d = drv.SimDriver(Limits(*kin.JointMap.identity().motor_limits(ROBOT)))
    d.start()
    assert wait_for(lambda: d.state is not None and d.state.connected)
    yield d
    d.close()


def test_sim_no_motion_without_torque(sim):
    p0 = sim.state.present.copy()
    sim.jog([10, 0, 0, 0, 0, 0])  # torque off: the command is ignored
    time.sleep(0.3)
    assert not sim.state.torque_on
    assert np.allclose(sim.state.present, p0)
    assert np.allclose(sim.state.target, p0)


def test_sim_torque_jog_hold(sim):
    p0 = sim.state.present.copy()
    sim.enable_torque()
    assert wait_for(lambda: sim.state.torque_on)
    assert np.allclose(sim.state.present, p0)  # enabling torque does not move the arm
    sim.set_speed(60, 60)
    sim.jog([10, 0, 0, 0, 0, 0])
    prev, max_step = sim.state.command.copy(), 0.0
    end = time.monotonic() + 2
    while time.monotonic() < end and abs(sim.state.present[0] - p0[0] - 10) > 0.2:
        c = sim.state.command.copy()
        max_step = max(max_step, np.max(np.abs(c - prev)))
        prev = c
        time.sleep(0.005)
    assert abs(sim.state.present[0] - p0[0] - 10) <= 0.2
    assert max_step <= MAX_STEP_PER_TICK + 1e-6
    sim.jog([0, 0, 0, 0, 0, 1000])  # clipped to the gripper limit
    assert wait_for(lambda: sim.state.target[5] == 100.0)
    sim.hold()
    assert wait_for(lambda: np.allclose(sim.state.target, sim.state.present, atol=1.0))


# --- FeetechDriver on a fake bus: connecting writes nothing -----------------------------


@dataclass
class MotorCalibration:  # same fields as lerobot.motors.MotorCalibration, so tests run without LeRobot
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


class FakeBus:
    def __init__(self, torque=0):
        self.writes = []
        self.raw = {n: 2048 for n in drv.MOTOR_IDS}
        self.raw["gripper"] = 2500
        self.torque = torque

    def ping(self, mid, num_retry=0):
        return drv.MODEL_NUMBER

    def read_calibration(self):
        return {
            n: MotorCalibration(i, 0, 0, 2000 if n == "gripper" else 1000, 3400 if n == "gripper" else 3000)
            for n, i in drv.MOTOR_IDS.items()
        }

    def sync_read(self, reg, motors=None, normalize=True, num_retry=0):
        vals = {
            "Present_Position": self.raw,
            "Goal_Position": self.raw,
            "Torque_Enable": {n: self.torque for n in drv.MOTOR_IDS},
            "Present_Voltage": {n: 121 for n in drv.MOTOR_IDS},
        }
        return dict(vals.get(reg, {n: 1 for n in drv.MOTOR_IDS}))

    def write(self, reg, motor, value, normalize=True, num_retry=0):
        self.writes.append(("write", reg, motor, value))

    def sync_write(self, reg, values, normalize=True, num_retry=0):
        self.writes.append(("sync_write", reg, dict(values)))

    def enable_torque(self, motors=None, num_retry=0):
        self.writes.append(("enable_torque",))
        self.torque = 1

    def disable_torque(self, motors=None, num_retry=0):
        self.writes.append(("disable_torque",))
        self.torque = 0

    def disconnect(self, disable_torque=True):
        self.writes.append(("disconnect", disable_torque))


@pytest.mark.parametrize("torque", [0, 1])
def test_feetech_connect_writes_nothing(torque):
    bus = FakeBus(torque)
    lim = Limits(*kin.JointMap.identity().motor_limits(ROBOT))
    d = drv.FeetechDriver("/dev/fake", lim, bus_factory=lambda port: bus)
    diags = []
    d.on_diag = diags.append
    d.start()
    assert wait_for(lambda: d.state is not None and d.state.connected)
    time.sleep(0.3)
    assert bus.writes == []
    assert diags and diags[0].ok and len(diags[0].rows) == 6
    d.close()
    assert bus.writes == [("disconnect", False)]  # closing does not touch torque


def test_feetech_torque_on_writes_goal_first():
    bus = FakeBus()
    d = drv.FeetechDriver("/dev/fake", Limits(*kin.JointMap.identity().motor_limits(ROBOT)), bus_factory=lambda p: bus)
    d.start()
    assert wait_for(lambda: d.state is not None and d.state.connected)
    d.enable_torque()
    assert wait_for(lambda: ("enable_torque",) in bus.writes)
    first = bus.writes.index(("enable_torque",))
    goals = [w for w in bus.writes[:first] if w[:2] == ("write", "Goal_Position")]
    assert len(goals) == 6 and all(w[3] == bus.raw[w[2]] for w in goals)
    d.close(release=True)
    assert bus.writes[-2:] == [("disable_torque",), ("disconnect", False)]
