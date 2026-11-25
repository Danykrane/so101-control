"""Safety filter: joint limits and step limiting. Pure numpy functions, no Qt and no hardware.

All arrays have length 6 in motor units: 5 joints in degrees, the gripper in percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_STEP_PER_TICK = 2.0  # degrees (and gripper percent) per tick, like LeRobot max_relative_target
MAX_LEAD = 10.0  # how far the commanded goal may run ahead of the measured position
# Sag compensation: the servo P controller (P=16) falls 1-2 deg short under the arm weight.
MAX_BIAS = 3.0  # deg, largest offset added to the commanded goal
BIAS_WINDOW = 6.0  # deg, a larger error means a hard stop or collision: do not integrate there
BIAS_DEADBAND = 0.15  # deg, servo dead band (about 2 encoder ticks)
SETTLED_SPEED = 3.0  # deg/s, below this a joint counts as settled
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
        # An empty intersection (bad calibration) is not widened: the joint stays where it is.
        return Limits(np.minimum(lo, hi), hi)


def clip_target(new: np.ndarray, old: np.ndarray, limits: Limits) -> np.ndarray:
    """Clip the goal to the limits. A joint that is already outside its limit may only move back inside,
    never jump to the boundary: otherwise enabling torque outside the limits would jerk the arm."""
    lo = np.minimum(limits.lower, old)
    hi = np.maximum(limits.upper, old)
    return np.clip(new, lo, hi)


def step_toward(command: np.ndarray, target: np.ndarray, max_arm: float, max_gripper: float) -> np.ndarray:
    """One tick of the speed governor: move command towards target.

    Arm joints move synchronously (common scale, all arrive at the same time, straight path in
    joint space), the gripper separately. The step per tick is at most MAX_STEP_PER_TICK.
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
    """Keep the commanded goal from running away from the measured position (stuck, blocked, lagging)."""
    return np.clip(command, present - lead, present + lead)


def integrate_bias(bias, target, present, command, speed, dt: float, ki: float) -> np.ndarray:
    """Outer I-loop against sag under gravity: an offset added to the commanded goal.

    Integrates only for arm joints that are settled (|speed| < SETTLED_SPEED), whose goal has been
    sent completely (command = target) and whose error is within (BIAS_DEADBAND, BIAS_WINDOW).
    A large error means a hard stop or collision: no integration there, so we never push harder.
    The gripper is left alone: when squeezing an object it is supposed to fall short. |bias| <= MAX_BIAS.
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
    """What to send to the motors: command + bias within limits (a joint outside its limit is not pulled
    to the boundary) and no further than MAX_STEP_PER_TICK from the previous send."""
    out = np.clip(np.asarray(command, dtype=float) + bias, np.minimum(limits.lower, present), np.maximum(limits.upper, present))
    return sent + np.clip(out - sent, -MAX_STEP_PER_TICK, MAX_STEP_PER_TICK)
