#!/usr/bin/env python3
"""Pure causal retargeting from five human flexions to a four-joint hand."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FingerRetargetingResult:
    status: str
    calibrated: bool
    command_target: Optional[np.ndarray]
    desired_target: Optional[np.ndarray]
    human_flexion_raw: Optional[np.ndarray]
    human_flexion_filtered: Optional[np.ndarray]
    normalized_robot_closure: Optional[np.ndarray]
    reference_token: str
    hold_required: bool = False


class OneEuroVectorFilter:
    """Small vector form of the causal One Euro low-pass filter."""

    def __init__(self, minimum_cutoff_hz: float, beta: float, derivative_cutoff_hz: float):
        self.minimum_cutoff_hz = float(minimum_cutoff_hz)
        self.beta = float(beta)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        if (
            not math.isfinite(self.minimum_cutoff_hz)
            or self.minimum_cutoff_hz <= 0.0
            or not math.isfinite(self.beta)
            or self.beta < 0.0
            or not math.isfinite(self.derivative_cutoff_hz)
            or self.derivative_cutoff_hz <= 0.0
        ):
            raise ValueError("One Euro parameters are invalid")
        self.timestamp = None
        self.raw = None
        self.filtered = None
        self.filtered_derivative = None

    @staticmethod
    def _alpha(cutoff_hz: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, timestamp: float, value: Sequence[float]) -> np.ndarray:
        vector = np.asarray(value, dtype=float).copy()
        self.timestamp = float(timestamp)
        self.raw = vector.copy()
        self.filtered = vector.copy()
        self.filtered_derivative = np.zeros_like(vector)
        return vector

    def update(self, timestamp: float, value: Sequence[float]) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if self.timestamp is None:
            return self.reset(timestamp, vector)
        dt = float(timestamp) - self.timestamp
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("finger timestamps must strictly increase")
        derivative = (vector - self.raw) / dt
        derivative_alpha = self._alpha(
            np.full(vector.shape, self.derivative_cutoff_hz), dt
        )
        self.filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.filtered_derivative
        )
        cutoff = self.minimum_cutoff_hz + self.beta * np.abs(
            self.filtered_derivative
        )
        value_alpha = self._alpha(cutoff, dt)
        self.filtered = value_alpha * vector + (1.0 - value_alpha) * self.filtered
        self.timestamp = float(timestamp)
        self.raw = vector.copy()
        return self.filtered.copy()


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError("{} must be a finite {}-vector".format(name, length))
    return vector


class ThreeFingerRetargeter:
    """Calibrated five-to-three synergy map with causal safety gates.

    Joint order is ``[configuration, flexion_1, flexion_2, flexion_3]``.
    The configuration target is fixed.  Only the three flexion coordinates
    are driven by the supplied 3x5 human-to-robot mixing matrix.
    """

    def __init__(
        self,
        open_target: Sequence[float],
        close_target: Sequence[float],
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        source_mixing_matrix: Sequence[Sequence[float]],
        config: Mapping[str, Any],
    ) -> None:
        self.open_target = _finite_vector(open_target, 4, "open_target")
        self.close_target = _finite_vector(close_target, 4, "close_target")
        self.lower_limits = _finite_vector(lower_limits, 4, "lower_limits")
        self.upper_limits = _finite_vector(upper_limits, 4, "upper_limits")
        self.mixing = np.asarray(source_mixing_matrix, dtype=float)
        if self.mixing.shape != (3, 5) or not np.all(np.isfinite(self.mixing)):
            raise ValueError("source_mixing_matrix must be finite 3x5")
        if np.any(self.mixing < 0.0) or not np.allclose(
            np.sum(self.mixing, axis=1), np.ones(3), atol=1.0e-9
        ):
            raise ValueError("each source mixing row must be a nonnegative convex sum")
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("joint limits must have positive range")
        if np.any(self.open_target < self.lower_limits) or np.any(
            self.open_target > self.upper_limits
        ):
            raise ValueError("open target exceeds joint limits")
        if np.any(self.close_target < self.lower_limits) or np.any(
            self.close_target > self.upper_limits
        ):
            raise ValueError("close target exceeds joint limits")
        if not math.isclose(
            self.open_target[0], self.close_target[0], abs_tol=1.0e-9
        ):
            raise ValueError("configuration joint must remain fixed")
        if np.any(self.close_target[1:] <= self.open_target[1:]):
            raise ValueError("close target must increase all flexion joints")

        self.minimum_confidence = float(config.get("minimum_confidence", 0.55))
        self.calibration_samples = int(config.get("calibration_samples", 4))
        self.reference_max_flexion = _finite_vector(
            config.get("reference_max_flexion", [0.40] * 5),
            5,
            "reference_max_flexion",
        )
        self.calibration_max_range = float(
            config.get("calibration_max_range", 0.08)
        )
        self.flexion_deadband = _finite_vector(
            config.get("human_flexion_deadband", [0.03] * 3),
            3,
            "human_flexion_deadband",
        )
        self.close_excursion = _finite_vector(
            config.get("human_close_excursion", [0.60] * 3),
            3,
            "human_close_excursion",
        )
        self.maximum_velocity = _finite_vector(
            config.get("maximum_velocity_rad_s", [0.40, 0.75, 0.75, 0.75]),
            4,
            "maximum_velocity_rad_s",
        )
        self.command_duration_s = float(config.get("command_duration_s", 0.06))
        self.maximum_human_rate = float(
            config.get("maximum_human_flexion_rate_per_s", 6.0)
        )
        self.innovation_slack = float(config.get("innovation_slack", 0.05))
        self.maximum_single_frame_innovation = float(
            config.get("maximum_single_frame_innovation", 0.25)
        )
        self.innovation_confirmation_samples = int(
            config.get("innovation_confirmation_samples", 3)
        )
        self.innovation_consistency_tolerance = float(
            config.get("innovation_consistency_tolerance", 0.08)
        )
        scalar_values = (
            self.minimum_confidence,
            self.calibration_max_range,
            self.command_duration_s,
            self.maximum_human_rate,
            self.innovation_slack,
            self.maximum_single_frame_innovation,
            self.innovation_consistency_tolerance,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in scalar_values):
            raise ValueError("finger retargeting scalar parameter is invalid")
        if (
            self.minimum_confidence > 1.0
            or self.calibration_samples < 2
            or np.any(self.reference_max_flexion <= 0.0)
            or np.any(self.reference_max_flexion > 1.0)
            or np.any(self.flexion_deadband < 0.0)
            or np.any(self.close_excursion <= self.flexion_deadband)
            or np.any(self.maximum_velocity <= 0.0)
            or self.command_duration_s <= 0.0
            or self.maximum_human_rate <= 0.0
            or self.maximum_single_frame_innovation <= 0.0
            or self.innovation_confirmation_samples < 2
            or self.innovation_consistency_tolerance <= 0.0
        ):
            raise ValueError("finger retargeting bounds are invalid")

        self.filter = OneEuroVectorFilter(
            config.get("one_euro_minimum_cutoff_hz", 2.0),
            config.get("one_euro_beta", 0.15),
            config.get("one_euro_derivative_cutoff_hz", 1.0),
        )
        self.active_token = ""
        self.blocked_tokens = set()
        self.calibration_buffer = []
        self.baseline = None
        self.last_raw = None
        self.last_timestamp = None
        self.last_target = None
        self.pending_innovation = None
        self.pending_innovation_count = 0

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    def _result(
        self,
        status: str,
        command_target=None,
        desired_target=None,
        raw=None,
        filtered=None,
        closure=None,
        hold_required=False,
    ) -> FingerRetargetingResult:
        def copied(value):
            return None if value is None else np.asarray(value, dtype=float).copy()

        return FingerRetargetingResult(
            status=str(status),
            calibrated=self.calibrated,
            command_target=copied(command_target),
            desired_target=copied(desired_target),
            human_flexion_raw=copied(raw),
            human_flexion_filtered=copied(filtered),
            normalized_robot_closure=copied(closure),
            reference_token=self.active_token,
            hold_required=bool(hold_required),
        )

    def _begin_reference(self, token: str, current_joints: np.ndarray) -> None:
        if token in self.blocked_tokens:
            raise ValueError("blocked finger reference requires a new C token")
        if self.blocked_tokens:
            self.blocked_tokens.clear()
        self.active_token = token
        self.calibration_buffer = []
        self.baseline = None
        self.last_raw = None
        self.last_timestamp = None
        self.last_target = np.clip(
            current_joints, self.lower_limits, self.upper_limits
        )
        self.pending_innovation = None
        self.pending_innovation_count = 0
        self.filter.timestamp = None

    def block_active_reference(self) -> None:
        if self.active_token:
            self.blocked_tokens.add(self.active_token)
        self.active_token = ""
        self.calibration_buffer = []
        self.baseline = None
        self.last_raw = None
        self.last_timestamp = None
        self.last_target = None
        self.pending_innovation = None
        self.pending_innovation_count = 0
        self.filter.timestamp = None

    def update(
        self,
        timestamp: Any,
        reference_token: str,
        flexion: Sequence[float],
        confidence: Any,
        current_joints: Sequence[float],
    ) -> FingerRetargetingResult:
        stamp = float(timestamp)
        token = str(reference_token)
        raw = _finite_vector(flexion, 5, "human flexion")
        current = _finite_vector(current_joints, 4, "current joints")
        confidence = float(confidence)
        if (
            not token
            or not math.isfinite(stamp)
            or stamp <= 0.0
            or not math.isfinite(confidence)
            or np.any(raw < 0.0)
            or np.any(raw > 1.0)
        ):
            return self._result("INVALID_FINGER_INPUT", raw=raw, hold_required=True)
        if token in self.blocked_tokens:
            return self._result(
                "BLOCKED_REFERENCE_REQUIRES_NEW_C", raw=raw, hold_required=True
            )
        if np.any(current < self.lower_limits - 0.02) or np.any(
            current > self.upper_limits + 0.02
        ):
            return self._result(
                "MEASURED_HAND_JOINT_OUT_OF_BOUNDS", raw=raw, hold_required=True
            )
        if confidence < self.minimum_confidence:
            return self._result(
                "LOW_FINGER_CONFIDENCE", raw=raw, hold_required=True
            )
        if token != self.active_token:
            self._begin_reference(token, current)

        if not self.calibrated:
            if np.any(raw > self.reference_max_flexion):
                self.calibration_buffer = []
                return self._result(
                    "C_REFERENCE_HAND_NOT_OPEN", raw=raw, hold_required=True
                )
            self.calibration_buffer.append(raw.copy())
            self.calibration_buffer = self.calibration_buffer[-self.calibration_samples :]
            if len(self.calibration_buffer) < self.calibration_samples:
                return self._result(
                    "CALIBRATING_OPEN_HAND", raw=raw, hold_required=True
                )
            samples = np.asarray(self.calibration_buffer)
            if float(np.max(np.ptp(samples, axis=0))) > self.calibration_max_range:
                self.calibration_buffer = [raw.copy()]
                return self._result(
                    "C_REFERENCE_HAND_MOVING", raw=raw, hold_required=True
                )
            self.baseline = np.median(samples, axis=0)
            self.last_raw = raw.copy()
            self.last_timestamp = stamp
            filtered = self.filter.reset(stamp, self.baseline)
        else:
            dt = stamp - self.last_timestamp
            if not math.isfinite(dt) or dt <= 0.0:
                return self._result(
                    "NON_MONOTONIC_FINGER_TIMESTAMP", raw=raw, hold_required=True
                )
            maximum_innovation = min(
                self.maximum_human_rate * dt + self.innovation_slack,
                self.maximum_single_frame_innovation,
            )
            if float(np.max(np.abs(raw - self.last_raw))) > maximum_innovation:
                if (
                    self.pending_innovation is None
                    or float(
                        np.max(np.abs(raw - self.pending_innovation))
                    ) > self.innovation_consistency_tolerance
                ):
                    self.pending_innovation = raw.copy()
                    self.pending_innovation_count = 1
                else:
                    self.pending_innovation_count += 1
                    self.pending_innovation = 0.5 * (
                        self.pending_innovation + raw
                    )
                if (
                    self.pending_innovation_count
                    < self.innovation_confirmation_samples
                ):
                    return self._result(
                        "FINGER_INNOVATION_REJECTED_PENDING_CONFIRMATION",
                        raw=raw,
                        hold_required=True,
                    )
                raw = self.pending_innovation.copy()
            self.pending_innovation = None
            self.pending_innovation_count = 0
            filtered = self.filter.update(stamp, raw)
            self.last_raw = raw.copy()
            self.last_timestamp = stamp

        virtual_delta = self.mixing @ (filtered - self.baseline)
        closure = np.clip(
            (virtual_delta - self.flexion_deadband)
            / (self.close_excursion - self.flexion_deadband),
            0.0,
            1.0,
        )
        desired = self.open_target.copy()
        desired[1:] += closure * (self.close_target[1:] - self.open_target[1:])
        maximum_step = self.maximum_velocity * self.command_duration_s
        command = self.last_target + np.clip(
            desired - self.last_target, -maximum_step, maximum_step
        )
        command = np.clip(command, self.lower_limits, self.upper_limits)
        self.last_target = command.copy()
        return self._result(
            "TRACKING" if np.any(closure > 0.0) else "OPEN_BASELINE",
            command_target=command,
            desired_target=desired,
            raw=raw,
            filtered=filtered,
            closure=closure,
        )


def frozen_finger_hold_target(
    existing_target: Sequence[float], current_joints: Sequence[float]
) -> np.ndarray:
    """Capture a hand hold once instead of chasing measured plant drift.

    Rebuilding a trajectory target from ``/joint_states`` on every invalid
    camera heartbeat creates a positive feedback path: gravity/contact moves
    a joint slightly, the next heartbeat blesses that displacement as the new
    target, and the hand slowly walks away.  A hold transition captures the
    measured configuration once and every later heartbeat reuses it verbatim.
    """

    current = _finite_vector(current_joints, 4, "current finger hold joints")
    if existing_target is None:
        return current.copy()
    return _finite_vector(
        existing_target, 4, "existing finger hold target"
    ).copy()


__all__ = [
    "FingerRetargetingResult",
    "OneEuroVectorFilter",
    "ThreeFingerRetargeter",
    "frozen_finger_hold_target",
]
