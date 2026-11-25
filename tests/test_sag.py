"""Компенсация провисания под весом: внешний I-контур поверх P-регулятора мотора."""

import time

import numpy as np

import driver as drv
import kinematics as kin
import safety as sf
from safety import Limits

LIM = Limits(*kin.JointMap.identity().motor_limits(kin.Robot()))
Z = np.zeros(6)
DT = 1 / 30


def test_integrate_bias_only_when_settled_and_in_window():
    target = np.array([0, -20, 20, 90, 0, 50.0])
    present = target - np.array([0, 2.0, -1.5, 0.05, 8.0, 3.0])  # roll: 8° = упор, гриппер: предмет
    b = sf.integrate_bias(Z, target, present, target, Z, DT, ki=2.0)
    assert b[1] > 0 and b[2] < 0  # копит в сторону ошибки
    assert b[0] == 0 and b[3] == 0  # в мертвой зоне
    assert b[4] == 0  # ошибка больше окна: упор, не копим
    assert b[5] == 0  # гриппер не дожимаем
    moving = sf.integrate_bias(Z, target, present, target, np.full(6, 10.0), DT, ki=2.0)
    assert np.all(moving == 0)  # сустав едет
    not_sent = sf.integrate_bias(Z, target, present, target - 1, Z, DT, ki=2.0)
    assert np.all(not_sent == 0)  # цель еще не отправлена целиком
    big = np.full(6, 100.0)
    assert np.all(np.abs(sf.integrate_bias(big, target, present, target, Z, DT, ki=2.0)) <= sf.MAX_BIAS)


def test_compensated_respects_limits_and_step():
    present = np.array([0, 99.5, 0, 0, 0, 50.0])
    command = present.copy()
    bias = np.array([0, 3.0, 0, 0, 0, 0])
    out = sf.compensated(command, bias, present, command, LIM)
    assert out[1] <= LIM.upper[1] + 1e-9  # добавка не выводит за лимит
    out = sf.compensated(command + 10, Z, present, command, LIM)
    assert np.all(np.abs(out - command) <= sf.MAX_STEP_PER_TICK + 1e-9)


def run_sim(ki, sag, target, seconds=3.0):
    d = drv.SimDriver(LIM, sag=sag)
    d.sag_ki = ki
    d.start()
    end = time.monotonic() + 3
    while time.monotonic() < end and not (d.state and d.state.connected):
        time.sleep(0.02)
    d.set_speed(60, 120)
    d.enable_torque()
    time.sleep(0.2)
    d.set_target(target)
    time.sleep(seconds)
    return d


def test_sim_sag_is_compensated():
    sag = np.array([0, 2.0, -1.5, 1.0, 0, 3.0])
    target = np.array([5.0, -30, 40, 70, 0, 20])
    d = run_sim(drv.RobotDriver.sag_ki, sag, target)
    try:
        s = d.state
        err = np.abs(s.present - s.target)
        assert np.max(err[:5]) < 0.3, err
        assert np.allclose(s.bias[1:4], sag[1:4], atol=0.3)
        assert s.bias[5] == 0 and err[5] > 2.5  # гриппер не компенсируем
        d.hold()  # СТОП: цель = текущая позиция, добавка с нуля; потом рука держится там, где остановили
        time.sleep(0.1)
        stop_target = d.state.target.copy()
        assert np.all(np.abs(d.state.bias[1:4]) < 0.5)
        time.sleep(1.5)
        assert np.allclose(d.state.target, stop_target)
        assert np.max(np.abs(d.state.present - stop_target)[:5]) < 0.3
    finally:
        d.close()


def test_sim_sag_without_compensation_stays():
    sag = np.array([0, 2.0, -1.5, 1.0, 0, 0])
    d = run_sim(0.0, sag, np.array([5.0, -30, 40, 70, 0, 20]), seconds=2.0)
    try:
        err = d.state.target - d.state.present
        assert np.allclose(err[1:4], sag[1:4], atol=0.2)
    finally:
        d.close()
