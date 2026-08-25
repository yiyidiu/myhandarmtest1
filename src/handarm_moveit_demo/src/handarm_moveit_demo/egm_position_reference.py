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


def collision_proximity_hold_required(
        collision_scale: float, hard_hold_scale: float,
        retreat_authorized: bool = False,
        retreat_authorization_age_s: float = float("inf"),
        retreat_authorization_timeout_s: float = 0.12) -> bool:
    """Fail closed at collision proximity while permitting a fresh retreat.

    MoveIt Servo's threshold-distance scaler may approach a small nonzero
    value instead of producing a hard halt.  A position-controlled plant must
    therefore re-anchor its reference before that residual command can creep
    into collision.  A short-lived authorization from the C-zero retreat guard
    is the only exception; stale or absent authorization always holds.
    """

    scale = float(collision_scale)
    threshold = float(hard_hold_scale)
    age = float(retreat_authorization_age_s)
    timeout = float(retreat_authorization_timeout_s)
    if (not np.isfinite(scale) or not np.isfinite(threshold) or
            not 0.0 <= scale <= 1.0 or
            not 0.0 <= threshold < 1.0 or
            np.isnan(age) or not np.isfinite(timeout) or timeout <= 0.0):
        raise ValueError("invalid collision proximity hold inputs")
    retreat_is_fresh = bool(
        retreat_authorized and 0.0 <= age <= timeout)
    return bool(scale <= threshold and not retreat_is_fresh)


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


@dataclass(frozen=True)
class EgmPositionHoldDecision:
    active: bool
    entered: bool
    released: bool
    source: str
    linear_speed: float
    angular_speed: float


class EgmPositionHoldGate:
    """Latch measured joints after a genuinely quiet Cartesian command.

    Enter and release thresholds are intentionally different.  Small residual
    MANO/IK-edge noise therefore cannot repeatedly unlock a stiff joint hold,
    while a deliberate translation or wrist rotation releases it without a
    reference jump.  Target-loss and Twist-timeout holds retain priority.
    """

    def __init__(
            self, latch_on_twist_timeout: bool = False,
            settled_hold_enabled: bool = False,
            settled_delay_s: float = 0.25,
            linear_enter_mps: float = 0.008,
            angular_enter_radps: float = 0.06,
            linear_release_mps: float = 0.025,
            angular_release_radps: float = 0.18):
        self.latch_on_twist_timeout = bool(latch_on_twist_timeout)
        self.settled_hold_enabled = bool(settled_hold_enabled)
        self.settled_delay_s = float(settled_delay_s)
        self.linear_enter_mps = float(linear_enter_mps)
        self.angular_enter_radps = float(angular_enter_radps)
        self.linear_release_mps = float(linear_release_mps)
        self.angular_release_radps = float(angular_release_radps)
        values = [
            self.settled_delay_s, self.linear_enter_mps,
            self.angular_enter_radps, self.linear_release_mps,
            self.angular_release_radps]
        if (not np.all(np.isfinite(values)) or
                self.settled_delay_s < 0.0 or
                self.linear_enter_mps < 0.0 or
                self.angular_enter_radps < 0.0 or
                self.linear_release_mps < self.linear_enter_mps or
                self.angular_release_radps < self.angular_enter_radps):
            raise ValueError("invalid EGM position-hold gate settings")
        self.active = False
        self.source = "NONE"
        self.quiet_since: Optional[float] = None

    def reset(self) -> None:
        self.active = False
        self.source = "NONE"
        self.quiet_since = None

    def update(self, now_s: float, input_fresh: bool,
               external_hold: bool,
               requested_twist: Sequence[float]) -> EgmPositionHoldDecision:
        now = float(now_s)
        if not np.isfinite(now):
            raise ValueError("position-hold timestamp must be finite")
        twist = _finite_vector("requested_twist", requested_twist, 6)
        linear_speed = float(np.linalg.norm(twist[:3]))
        angular_speed = float(np.linalg.norm(twist[3:]))
        was_active = self.active

        if external_hold:
            self.active = True
            self.source = "TARGET_LOSS"
            self.quiet_since = None
        elif self.latch_on_twist_timeout and not bool(input_fresh):
            self.active = True
            self.source = "TWIST_TIMEOUT"
            self.quiet_since = None
        elif not self.settled_hold_enabled:
            self.active = False
            self.source = "NONE"
            self.quiet_since = None
        elif self.active and self.source == "SETTLED_TARGET":
            if (linear_speed >= self.linear_release_mps or
                    angular_speed >= self.angular_release_radps):
                self.active = False
                self.source = "NONE"
                self.quiet_since = None
        else:
            # A target-loss/timeout hold must be recaptured through the same
            # quiet dwell; it must never silently become a settled hold.
            self.active = False
            self.source = "NONE"
            quiet = bool(
                linear_speed <= self.linear_enter_mps and
                angular_speed <= self.angular_enter_radps)
            if quiet:
                if self.quiet_since is None or now < self.quiet_since:
                    self.quiet_since = now
                if now - self.quiet_since >= self.settled_delay_s:
                    self.active = True
                    self.source = "SETTLED_TARGET"
            else:
                self.quiet_since = None

        return EgmPositionHoldDecision(
            active=self.active,
            entered=self.active and not was_active,
            released=was_active and not self.active,
            source=self.source,
            linear_speed=linear_speed,
            angular_speed=angular_speed)


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

    def hold_reference(self, positions: Sequence[float]) -> None:
        """Stop feed-forward while preserving the already commanded target.

        This is the settled-target behavior: unlike a target-loss safety latch,
        it must not redefine a desired pose from a gravity-deflected feedback
        sample at the instant the quiet dwell expires.
        """

        self.actual = _finite_vector(
            "hold positions", positions, self.size).copy()
        self.latest_velocity.fill(0.0)
        self.feedforward_velocity.fill(0.0)
        self.latest_command_time = None

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
