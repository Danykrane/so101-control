"""robotd in sim mode (no HTTP), Cartesian planner, camera thread shutdown."""

import threading

import numpy as np
import pytest

import cameras
import kinematics as kin
import robotd
from safety import Limits

ROBOT, JMAP = kin.Robot(), kin.JointMap.identity()
LIMITS = Limits(*JMAP.motor_limits(ROBOT))
READY = np.array([0.0, -20.0, 20.0, 90.0, 0.0, 7.0])


def test_plan_cartesian_reaches_goal_and_rejects_far():
    goal = ROBOT.task(JMAP.to_urdf(READY)[:5])
    goal[0] += 0.03
    goal[2] -= 0.04
    t, msg = kin.plan_cartesian(ROBOT, JMAP, READY, LIMITS, goal)
    assert t is not None, msg
    assert np.isnan(t[5])  # gripper untouched
    q = READY.copy()
    q[:5] = t[:5]
    assert np.allclose(ROBOT.task(JMAP.to_urdf(q)[:5])[:3], goal[:3], atol=1e-4)
    far = goal.copy()
    far[0] = 0.8
    assert kin.plan_cartesian(ROBOT, JMAP, READY, LIMITS, far)[0] is None


@pytest.fixture
def app():
    a = robotd.Robotd(sim=True, port=None, cams=[], z_min_mm=-30.0)
    a.driver.set_speed(60, 120)
    a.start()
    yield a
    a.close()


def test_robotd_needs_torque_then_moves(app):
    with pytest.raises(robotd.Busy):
        app.move({"joints": {"elbow": 10}})
    app.torque(True)
    res = app.move({"joints": {"pan": 10, "lift": -30, "elbow": 40, "wrist": 80}, "timeout": 10})
    assert res["reached"], res["reason"]
    assert abs(res["state"]["present"][0] - 10) < robotd.TOLERANCE[0]
    res = app.move({"joints": {"wrist_flex": -5}, "relative": True, "timeout": 10})
    assert res["reached"] and abs(res["target"][3] - 75) < 0.01
    with pytest.raises(ValueError):
        app.move({"joints": {"foo": 1}})


def test_robotd_clips_and_cartesian(app):
    app.torque(True)
    res = app.move({"joints": {"lift": 150}, "wait": False})
    assert res["clipped"] == ["shoulder_lift"] and res["target"][1] <= LIMITS.upper[1] + 1e-6
    app.move({"joints": dict(zip(kin.JOINTS, READY)), "timeout": 15})
    z0 = app.state_json()["ee_target"]["z"]
    res = app.cartesian({"dz": -20, "timeout": 10})  # cannot go up from READY: the wrist hits its 95 deg limit
    assert res["reached"], res["reason"]
    assert abs(res["state"]["ee_target"]["z"] - (z0 - 20)) < 0.5
    with pytest.raises(ValueError):  # below the table
        app.cartesian({"z": -80})


def test_robotd_wait_notices_foreign_stop(app):
    app.torque(True)
    app.driver.set_speed(5, 10)
    result = {}
    t = threading.Thread(target=lambda: result.update(app.move({"joints": {"pan": 60}, "timeout": 10})))
    t.start()
    while app.state.command[0] < 2:
        pass
    app.stop()  # like STOP in the GUI
    t.join(5)
    assert result["reached"] is False and "another client" in result["reason"]


def test_grabber_stop_after_thread_finished():
    class Quick(cameras.Grabber):
        def run(self):
            pass

    g = Quick(99)
    g.start()
    g.join(1)
    g.stop()  # used to crash: the _stop field shadowed Thread._stop
    assert not g.is_alive()
