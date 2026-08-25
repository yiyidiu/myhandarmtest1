#!/usr/bin/env python3
"""Quality-adaptive causal filtering of a HaMeR palm frame on SO(3).

The update follows the SO(3) formulation used by the user-supplied
``teleoperation_ubuntu_core`` V9 diagnostic code::

    R_k = R_(k-1) Exp(g_k Log(R_(k-1)^T Z_k))

The default teleoperation policy follows every valid SO(3) measurement,
including large intentional rotations.  The former hard-rejection behaviour
remains available as an explicit diagnostic rollback mode.  Matrix entries
and Euler angles are never averaged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np

from .palm_frame import (
    PalmFrameError,
    project_to_so3,
    require_so3,
    rotation_matrix_to_quaternion_xyzw,
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x_value, y_value, z_value = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray(
        [[0.0, -z_value, y_value],
         [z_value, 0.0, -x_value],
         [-y_value, x_value, 0.0]],
        dtype=np.float64,
    )


def _so3_exp(rotation_vector: Any) -> np.ndarray:
    value = np.asarray(rotation_vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("rotation_vector must be a finite 3-vector")
    angle = float(np.linalg.norm(value))
    if angle < 1.0e-8:
        skew = _skew(value)
        return project_to_so3(np.eye(3) + skew + 0.5 * skew @ skew)
    axis = value / angle
    skew = _skew(axis)
    return require_so3(
        np.eye(3) + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew),
        atol=1.0e-7,
    )


def _so3_log(rotation: Any) -> np.ndarray:
    matrix = require_so3(rotation, atol=1.0e-6)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.asarray(
        [matrix[2, 1] - matrix[1, 2],
         matrix[0, 2] - matrix[2, 0],
         matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        return 0.5 * vee
    if math.pi - angle < 1.0e-5:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        index = int(np.argmax(axis))
        if axis[index] < 1.0e-10:
            axis = np.asarray([1.0, 0.0, 0.0])
        else:
            if index == 0:
                axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
                axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
            elif index == 1:
                axis[0] = math.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
                axis[2] = math.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
            else:
                axis[0] = math.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
                axis[1] = math.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
            axis /= np.linalg.norm(axis)
        return angle * axis
    return angle / (2.0 * math.sin(angle)) * vee


@dataclass(frozen=True)
class CausalWristSO3FilterConfig:
    time_constant_s: float = 0.10
    minimum_gain: float = 0.06
    maximum_gain: float = 1.0
    motion_gain_start_deg: float = 1.5
    motion_gain_full_deg: float = 8.0
    innovation_soft_deg: float = 25.0
    innovation_hard_deg: float = 60.0
    large_angle_mode: str = "follow"

    def validate(self) -> None:
        values = (
            self.time_constant_s,
            self.minimum_gain,
            self.maximum_gain,
            self.motion_gain_start_deg,
            self.motion_gain_full_deg,
            self.innovation_soft_deg,
            self.innovation_hard_deg,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("SO3 filter configuration must be finite")
        if self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive")
        if not 0.0 <= self.minimum_gain <= self.maximum_gain <= 1.0:
            raise ValueError("SO3 filter gains must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.motion_gain_start_deg < self.motion_gain_full_deg:
            raise ValueError("SO3 motion-adaptive gain thresholds are inconsistent")
        if not 0.0 < self.innovation_soft_deg < self.innovation_hard_deg < 180.0:
            raise ValueError("SO3 innovation thresholds are inconsistent")
        if self.motion_gain_full_deg >= self.innovation_hard_deg:
            raise ValueError("responsive motion threshold must be below hard innovation")
        if self.large_angle_mode not in {"follow", "reject"}:
            raise ValueError("large_angle_mode must be follow or reject")


@dataclass(frozen=True)
class CausalWristSO3FilterResult:
    timestamp_s: float
    valid: bool
    status: str
    rotation: Optional[np.ndarray]
    quaternion_xyzw: Optional[np.ndarray]
    confidence: float
    measurement_quality: float
    gain: float
    innovation_deg: Optional[float]
    reason: str
    large_angle_mode: str
    large_angle_passthrough: bool

    def as_dict(self) -> dict:
        return {
            "timestamp_s": float(self.timestamp_s),
            "valid": bool(self.valid),
            "status": str(self.status),
            "rotation": None if self.rotation is None else self.rotation.tolist(),
            "quaternion_xyzw": (
                None
                if self.quaternion_xyzw is None
                else self.quaternion_xyzw.tolist()
            ),
            "confidence": float(self.confidence),
            "measurement_quality": float(self.measurement_quality),
            "gain": float(self.gain),
            "innovation_deg": self.innovation_deg,
            "reason": str(self.reason),
            "large_angle_mode": str(self.large_angle_mode),
            "large_angle_passthrough": bool(self.large_angle_passthrough),
            "method": "quality_adaptive_causal_so3",
        }


class CausalWristSO3Filter:
    """Filter valid wrist rotations without blocking intentional motion."""

    def __init__(
        self, config: Optional[CausalWristSO3FilterConfig] = None
    ) -> None:
        self.config = config or CausalWristSO3FilterConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        self._filtered: Optional[np.ndarray] = None
        self._last_timestamp_s: Optional[float] = None

    def _invalid(
        self,
        timestamp_s: float,
        status: str,
        quality: float,
        innovation_deg: Optional[float],
        reason: str,
    ) -> CausalWristSO3FilterResult:
        return CausalWristSO3FilterResult(
            timestamp_s=float(timestamp_s),
            valid=False,
            status=status,
            rotation=None,
            quaternion_xyzw=None,
            confidence=0.0,
            measurement_quality=float(quality),
            gain=0.0,
            innovation_deg=innovation_deg,
            reason=reason,
            large_angle_mode=self.config.large_angle_mode,
            large_angle_passthrough=False,
        )

    def _valid(
        self,
        timestamp_s: float,
        status: str,
        quality: float,
        gain: float,
        innovation_deg: float,
        large_angle_passthrough: bool = False,
    ) -> CausalWristSO3FilterResult:
        assert self._filtered is not None
        rotation = require_so3(self._filtered, atol=1.0e-6)
        return CausalWristSO3FilterResult(
            timestamp_s=float(timestamp_s),
            valid=True,
            status=status,
            rotation=rotation,
            quaternion_xyzw=rotation_matrix_to_quaternion_xyzw(rotation),
            confidence=float(np.clip(quality, 0.0, 1.0)),
            measurement_quality=float(np.clip(quality, 0.0, 1.0)),
            gain=float(gain),
            innovation_deg=float(innovation_deg),
            reason="",
            large_angle_mode=self.config.large_angle_mode,
            large_angle_passthrough=bool(large_angle_passthrough),
        )

    def update(
        self, timestamp_s: Any, measurement: Any, measurement_quality: Any = 1.0
    ) -> CausalWristSO3FilterResult:
        try:
            timestamp = float(timestamp_s)
            quality = float(measurement_quality)
        except (TypeError, ValueError, OverflowError):
            return self._invalid(0.0, "invalid", 0.0, None, "invalid_scalar")
        if not math.isfinite(timestamp) or not math.isfinite(quality):
            return self._invalid(timestamp, "invalid", 0.0, None, "nonfinite_scalar")
        quality = float(np.clip(quality, 0.0, 1.0))
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            return self._invalid(
                timestamp, "invalid", quality, None, "nonmonotonic_timestamp"
            )
        try:
            observed = project_to_so3(measurement)
        except (PalmFrameError, ValueError, np.linalg.LinAlgError):
            self._last_timestamp_s = timestamp
            return self._invalid(
                timestamp, "invalid", quality, None, "invalid_rotation_measurement"
            )
        if quality <= 0.0:
            self._last_timestamp_s = timestamp
            return self._invalid(
                timestamp, "invalid", quality, None, "nonpositive_measurement_quality"
            )

        if self._filtered is None:
            self._filtered = observed
            self._last_timestamp_s = timestamp
            return self._valid(timestamp, "initialized", quality, 1.0, 0.0)

        assert self._last_timestamp_s is not None
        dt_s = timestamp - self._last_timestamp_s
        innovation = _so3_log(self._filtered.T @ observed)
        innovation_deg = math.degrees(float(np.linalg.norm(innovation)))
        self._last_timestamp_s = timestamp

        large_angle = innovation_deg >= self.config.innovation_hard_deg
        if large_angle and self.config.large_angle_mode == "reject":
            return self._invalid(
                timestamp,
                "jump_rejected",
                quality,
                innovation_deg,
                "orientation_innovation_exceeds_hard_limit",
            )

        base_gain = 1.0 - math.exp(-dt_s / self.config.time_constant_s)
        gain = base_gain * (0.15 + 0.85 * quality)
        motion_fraction = float(np.clip(
            (innovation_deg - self.config.motion_gain_start_deg) /
            (self.config.motion_gain_full_deg -
             self.config.motion_gain_start_deg),
            0.0, 1.0,
        ))
        motion_weight = motion_fraction * motion_fraction * (
            3.0 - 2.0 * motion_fraction)
        # At rest, retain the time-constant/quality low-pass. During a clear
        # intentional rotation, approach a quality-scaled responsive gain so
        # smoothing does not recreate the previous teleoperation latency.
        responsive_gain = (
            self.config.minimum_gain + quality *
            (self.config.maximum_gain - self.config.minimum_gain))
        if self.config.large_angle_mode == "follow" and motion_fraction >= 1.0:
            # Once motion is unambiguous, prioritize direct teleoperation.
            # Measurement quality still controls the stationary low-pass, but
            # it must not turn an intentional large wrist rotation into lag.
            responsive_gain = self.config.maximum_gain
        gain += motion_weight * max(0.0, responsive_gain - gain)
        if (
            self.config.large_angle_mode == "reject"
            and innovation_deg > self.config.innovation_soft_deg
        ):
            gain *= self.config.innovation_soft_deg / innovation_deg
        gain = float(
            np.clip(gain, self.config.minimum_gain, self.config.maximum_gain)
        )
        self._filtered = project_to_so3(
            self._filtered @ _so3_exp(gain * innovation)
        )
        return self._valid(
            timestamp,
            "tracking_large_angle_passthrough" if large_angle else "tracking",
            quality,
            gain,
            innovation_deg,
            large_angle_passthrough=large_angle,
        )


__all__ = [
    "CausalWristSO3Filter",
    "CausalWristSO3FilterConfig",
    "CausalWristSO3FilterResult",
]
