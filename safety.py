"""Safety-фильтр: лимиты и ограничение шага. Чистые функции на numpy, без Qt и железа.

Все массивы длины 6 в моторных единицах: 5 суставов в градусах, гриппер в процентах.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_STEP_PER_TICK = 2.0  # градусов (и процентов гриппера) за тик, как max_relative_target у so101-tiptop
MAX_LEAD = 10.0  # насколько отправляемая цель может опережать фактическую позицию
# Компенсация провисания: P-регулятор мотора (P=16) под весом руки недотягивает 1–2°.
MAX_BIAS = 3.0  # ° наибольшая добавка к отправляемой цели
BIAS_WINDOW = 6.0  # ° ошибка больше = упор или столкновение, туда не копим
BIAS_DEADBAND = 0.15  # ° мертвая зона сервопривода (около 2 шагов энкодера)
SETTLED_SPEED = 3.0  # °/с сустав считается стоящим
ARM = slice(0, 5)
GRIPPER = 5


@dataclass(frozen=True)
class Limits:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        if np.any(self.lower > self.upper):
            raise ValueError(f"lower > upper: {self.lower} {self.upper}")

    def intersect(self, other: Limits) -> Limits:
        lo, hi = np.maximum(self.lower, other.lower), np.minimum(self.upper, other.upper)
        # Пустое пересечение (кривая калибровка) не делаем шире: сустав стоит на месте.
        return Limits(np.minimum(lo, hi), hi)


def clip_target(new: np.ndarray, old: np.ndarray, limits: Limits) -> np.ndarray:
    """Клип цели по лимитам. Если сустав уже за лимитом, ему можно только возвращаться внутрь,
    но не прыгать на границу: иначе включение момента вне лимитов дернуло бы руку."""
    lo = np.minimum(limits.lower, old)
    hi = np.maximum(limits.upper, old)
    return np.clip(new, lo, hi)


def step_toward(command: np.ndarray, target: np.ndarray, max_arm: float, max_gripper: float) -> np.ndarray:
    """Один тик регулятора скорости: сдвинуть command к target.

    Суставы руки двигаются синхронно (общий масштаб, все приходят одновременно, путь в
    пространстве суставов прямой), гриппер отдельно. Шаг за тик не больше MAX_STEP_PER_TICK.
    """
    max_arm = min(max_arm, MAX_STEP_PER_TICK)
    max_gripper = min(max_gripper, MAX_STEP_PER_TICK)
    delta = np.asarray(target, dtype=float) - command
    out = np.array(command, dtype=float)
    arm = delta[ARM]
    peak = np.max(np.abs(arm))
    if peak > max_arm:
        arm = arm * (max_arm / peak)
    out[ARM] += arm
    out[GRIPPER] += np.clip(delta[GRIPPER], -max_gripper, max_gripper)
    return out


def limit_lead(command: np.ndarray, present: np.ndarray, lead: float = MAX_LEAD) -> np.ndarray:
    """Не давать отправляемой цели убегать от фактической позиции (застрял, упирается, лагает)."""
    return np.clip(command, present - lead, present + lead)


def integrate_bias(bias, target, present, command, speed, dt: float, ki: float) -> np.ndarray:
    """Внешний I-контур против провисания под весом: добавка к отправляемой цели.

    Копит только по суставам руки, которые стоят (|speed| < SETTLED_SPEED), чья цель уже
    отправлена целиком (command = target) и чья ошибка в окне (BIAS_DEADBAND, BIAS_WINDOW).
    Большая ошибка значит упор или столкновение: там не копим, чтобы не давить сильнее.
    Гриппер не трогаем: сжимая предмет, он и должен недотягивать. |добавка| ≤ MAX_BIAS.
    """
    err = np.asarray(target, dtype=float) - present
    active = (
        (np.abs(err) > BIAS_DEADBAND)
        & (np.abs(err) < BIAS_WINDOW)
        & (np.abs(speed) < SETTLED_SPEED)
        & (np.abs(np.asarray(command, dtype=float) - target) < 1e-3)
    )
    active[GRIPPER] = False
    out = np.asarray(bias, dtype=float) + np.where(active, ki * err * dt, 0.0)
    out[GRIPPER] = 0.0
    return np.clip(out, -MAX_BIAS, MAX_BIAS)


def compensated(command, bias, present, sent, limits: Limits) -> np.ndarray:
    """Что отправить в моторы: command + bias в лимитах (сустав за лимитом не тянем на границу)
    и не дальше MAX_STEP_PER_TICK от прошлой отправки."""
    out = np.clip(np.asarray(command, dtype=float) + bias, np.minimum(limits.lower, present), np.maximum(limits.upper, present))
    return sent + np.clip(out - sent, -MAX_STEP_PER_TICK, MAX_STEP_PER_TICK)
