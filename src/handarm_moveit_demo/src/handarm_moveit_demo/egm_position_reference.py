"""Pure latest-reference integrator for the Gazebo EGM-emulation profile.

ABB EGM position guidance combines a persistent position reference with a
velocity feed-forward term.  MoveIt Servo already produces the joint velocity
feed-forward required by this project.  This model integrates only the latest
velocity sample into a bounded joint-position reference and keeps that
reference unchanged when input stops.  It deliberately has no command queue.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _finite_vector(name: str, values: Sequence[float], size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        raise ValueError("{} must contain exactly {} values".format(name, size))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite values".format(name))
    return vector


@dataclass(frozen=True)
class EgmReferenceOutput:
    reference: np.ndarray
    feedforward_velocity: np.ndarray
    actual: np.ndarray
    following_error: np.ndarray
    command_age_s: float
    command_fresh: bool
    limit_clamped: bool
    following_error_clamped: bool
    time_reset: bool


class EgmPositionReferenceModel:
    """Integrate the newest velocity sample into one persistent reference."""

    def __init__(
            self,
            joint_names: Sequence[str],
            initial_reference: Sequence[float],
            lower_limits: Sequence[float],
            upper_limits: Sequence[float],
            maximum_velocity: Sequence[float],
            maximum_acceleration: Sequence[float],
            command_timeout_s: float = 0.10,
            joint_limit_margin_rad: float = 0.01,
            maximum_step_dt_s: float = 0.02,
            maximum_following_error: Optional[Sequence[float]] = None):
        self.joint_names = tuple(str(name) for name in joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        size = len(self.joint_names)
        lower = _finite_vector("lower_limits", lower_limits, size)
        upper = _finite_vector("upper_limits", upper_limits, size)
        margin = float(joint_limit_margin_rad)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("joint_limit_margin_rad must be finite and non-negative")
        self.lower_limits = lower + margin
        self.upper_limits = upper - margin
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("joint limits leave no range after applying the margin")
        self.maximum_velocity = _finite_vector(
            "maximum_velocity", maximum_velocity, size)
        self.maximum_acceleration = _finite_vector(
            "maximum_acceleration", maximum_acceleration, size)
        if maximum_following_error is None:
            self.maximum_following_error = np.full(
                size, float("inf"), dtype=float)
        else:
            self.maximum_following_error = _finite_vector(
                "maximum_following_error", maximum_following_error, size)
        # Gazebo reports wide-range revolute joints (IRB120 joint_6) modulo
        # 2*pi. Keep an equivalent reference on the turn nearest feedback so
        # neither diagnostics nor recovery mistakes one wrap for a 360deg lag.
        self.wrap_period = np.where(
            self.upper_limits - self.lower_limits > 2.0 * np.pi + 1.0e-6,
            2.0 * np.pi, 0.0)
        if np.any(self.maximum_velocity <= 0.0):
            raise ValueError("maximum_velocity values must be positive")
        if np.any(self.maximum_acceleration <= 0.0):
            raise ValueError("maximum_acceleration values must be positive")
        if np.any(self.maximum_following_error <= 0.0):
            raise ValueError(
                "maximum_following_error values must be positive")
        self.command_timeout_s = float(command_timeout_s)
        self.maximum_step_dt_s = float(maximum_step_dt_s)
        if not np.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0.0:
            raise ValueError("command_timeout_s must be finite and positive")
        if not np.isfinite(self.maximum_step_dt_s) or self.maximum_step_dt_s <= 0.0:
            raise ValueError("maximum_step_dt_s must be finite and positive")

        initial = _finite_vector("initial_reference", initial_reference, size)
        self.reference = np.clip(
            initial, self.lower_limits, self.upper_limits)
        self.actual: Optional[np.ndarray] = None
        self.latest_velocity = np.zeros(size, dtype=float)
        self.feedforward_velocity = np.zeros(size, dtype=float)
        self.latest_command_time: Optional[float] = None
        self.last_step_time: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self.joint_names)

    def update_actual(self, positions: Sequence[float]) -> None:
        self.actual = _finite_vector("actual positions", positions, self.size)

    def synchronize_reference(self, positions: Sequence[float]) -> None:
        """Bumplessly align the persistent reference with measured joints."""

        actual = _finite_vector("synchronization positions", positions, self.size)
        self.actual = actual.copy()
        self.reference = np.clip(
            actual, self.lower_limits, self.upper_limits)
        self.latest_velocity.fill(0.0)
        self.feedforward_velocity.fill(0.0)
        self.latest_command_time = None
        self.last_step_time = None

    def update_velocity(self, velocity: Sequence[float], stamp_s: float) -> None:
        stamp = float(stamp_s)
        if not np.isfinite(stamp):
            raise ValueError("velocity timestamp must be finite")
        requested = _finite_vector("velocity command", velocity, self.size)
        self.latest_velocity = np.clip(
            requested, -self.maximum_velocity, self.maximum_velocity)
        self.latest_command_time = stamp

    def _reanchor_equivalent_turns(self) -> None:
        if self.actual is None:
            return
        for index, period in enumerate(self.wrap_period):
            if period <= 0.0:
                continue
            turns = int(round(
                (self.reference[index] - self.actual[index]) / period))
            candidate = self.reference[index] - turns * period
            if (self.lower_limits[index] <= candidate <=
                    self.upper_limits[index]):
                self.reference[index] = candidate

    def _apply_following_error_leash(self) -> bool:
        """Keep the command close enough for the position servo to follow.

        An EGM position correction is advanced from measured feedback; it must
        not become an open-loop integral that runs several radians ahead of a
        saturated joint.  The moving leash preserves the persistent position
        reference while allowing it to advance as feedback catches up.
        """

        if self.actual is None:
            return False
        proposed = self.reference.copy()
        self.reference = np.clip(
            proposed,
            self.actual - self.maximum_following_error,
            self.actual + self.maximum_following_error)
        self.reference = np.clip(
            self.reference, self.lower_limits, self.upper_limits)
        return bool(np.any(np.abs(self.reference - proposed) > 1.0e-12))

    def step(self, now_s: float) -> Optional[EgmReferenceOutput]:
        if self.actual is None:
            return None
        self._reanchor_equivalent_turns()
        now = float(now_s)
        if not np.isfinite(now):
            raise ValueError("step timestamp must be finite")

        time_reset = False
        if self.last_step_time is None:
            dt = 0.0
        elif now < self.last_step_time:
            dt = 0.0
            time_reset = True
            self.feedforward_velocity.fill(0.0)
        else:
            dt = min(now - self.last_step_time, self.maximum_step_dt_s)
        self.last_step_time = now

        if self.latest_command_time is None:
            command_age = float("inf")
            fresh = False
        else:
            command_age = now - self.latest_command_time
            fresh = 0.0 <= command_age <= self.command_timeout_s
        desired_velocity = (
            self.latest_velocity if fresh else np.zeros(self.size, dtype=float))

        if dt > 0.0:
            maximum_delta = self.maximum_acceleration * dt
            delta = np.clip(
                desired_velocity - self.feedforward_velocity,
                -maximum_delta, maximum_delta)
            self.feedforward_velocity = np.clip(
                self.feedforward_velocity + delta,
                -self.maximum_velocity, self.maximum_velocity)
            proposed = self.reference + self.feedforward_velocity * dt
            bounded = np.clip(proposed, self.lower_limits, self.upper_limits)
            clamped_axes = np.abs(bounded - proposed) > 1e-12
            self.reference = bounded
            # Do not retain an outward velocity at a saturated joint reference.
            self.feedforward_velocity[clamped_axes] = 0.0
            limit_clamped = bool(np.any(clamped_axes))
        else:
            limit_clamped = False

        following_error_clamped = self._apply_following_error_leash()

        return EgmReferenceOutput(
            reference=self.reference.copy(),
            feedforward_velocity=self.feedforward_velocity.copy(),
            actual=self.actual.copy(),
            following_error=(self.reference - self.actual).copy(),
            command_age_s=command_age,
            command_fresh=fresh,
            limit_clamped=limit_clamped,
            following_error_clamped=following_error_clamped,
            time_reset=time_reset,
        )
