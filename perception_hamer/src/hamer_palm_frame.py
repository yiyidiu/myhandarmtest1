#!/usr/bin/env python3
"""Strict HaMeR/MANO palm frame and robust per-session beta calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .palm_frame import (
    PalmFrameError,
    align_quaternion_sign,
    require_so3,
    rotation_matrix_to_quaternion_xyzw,
)


WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
LITTLE_MCP = 17


@dataclass(frozen=True)
class HamerPalmFrameResult:
    valid: bool
    rotation: Optional[np.ndarray]
    quaternion_xyzw: Optional[np.ndarray]
    origin: Optional[np.ndarray]
    is_right: Optional[bool]
    failure_reason: str
    quality: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.valid:
            rotation = require_so3(self.rotation)
            quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
            origin = np.asarray(self.origin, dtype=np.float64)
            if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
                raise PalmFrameError("valid palm frame needs a finite xyzw quaternion")
            if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-6:
                raise PalmFrameError("palm quaternion is not unit length")
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                raise PalmFrameError("valid palm frame needs a finite origin")
            if self.failure_reason:
                raise PalmFrameError("valid palm frame cannot carry a failure reason")
            object.__setattr__(self, "rotation", rotation)
            object.__setattr__(self, "quaternion_xyzw", quaternion.copy())
            object.__setattr__(self, "origin", origin.copy())
        elif any(value is not None for value in (self.rotation, self.quaternion_xyzw, self.origin)):
            raise PalmFrameError("invalid palm frame must not expose identity/stale geometry")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "rotation": None if self.rotation is None else self.rotation.tolist(),
            "quaternion_xyzw": (
                None if self.quaternion_xyzw is None else self.quaternion_xyzw.tolist()
            ),
            "origin": None if self.origin is None else self.origin.tolist(),
            "is_right": self.is_right,
            "failure_reason": self.failure_reason,
            "quality": dict(self.quality),
            "definition": (
                "x=normalize(index_mcp-little_mcp); "
                "y=orthogonalized(middle_mcp-wrist); z=x_cross_y"
            ),
            "global_orient_used": False,
        }


def invalid_palm_frame(reason: str, is_right: Optional[bool] = None) -> HamerPalmFrameResult:
    return HamerPalmFrameResult(
        valid=False,
        rotation=None,
        quaternion_xyzw=None,
        origin=None,
        is_right=is_right,
        failure_reason=str(reason),
        quality={},
    )


def build_hamer_joint_palm_frame(
    joints: Any,
    is_right: bool,
    previous_quaternion_xyzw: Optional[Sequence[float]] = None,
    minimum_axis_length: float = 1e-6,
) -> HamerPalmFrameResult:
    """Build the requested wrist/index/middle/little MANO frame in source axes.

    ``joints`` must already use the crop API's source-camera axis convention.
    For either hand, +X is anatomical little-to-index, +Y is wrist-to-middle,
    and +Z completes the right-handed frame.  No reflection matrix is treated
    as a rotation.
    """

    if not isinstance(is_right, (bool, np.bool_)):
        return invalid_palm_frame("is_right_must_be_boolean")
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        return invalid_palm_frame("joints_must_be_finite_21x3", bool(is_right))
    minimum_axis_length = float(minimum_axis_length)
    if not math.isfinite(minimum_axis_length) or minimum_axis_length <= 0.0:
        raise ValueError("minimum_axis_length must be finite and positive")
    wrist = points[WRIST]
    x_raw = points[INDEX_MCP] - points[LITTLE_MCP]
    y_raw = points[MIDDLE_MCP] - wrist
    x_length = float(np.linalg.norm(x_raw))
    y_raw_length = float(np.linalg.norm(y_raw))
    if x_length <= minimum_axis_length or y_raw_length <= minimum_axis_length:
        return invalid_palm_frame("palm_axis_too_short", bool(is_right))
    x_axis = x_raw / x_length
    y_projected = y_raw - float(np.dot(y_raw, x_axis)) * x_axis
    y_projected_length = float(np.linalg.norm(y_projected))
    if y_projected_length <= minimum_axis_length:
        return invalid_palm_frame("palm_axes_collinear", bool(is_right))
    y_axis = y_projected / y_projected_length
    z_axis = np.cross(x_axis, y_axis)
    z_length = float(np.linalg.norm(z_axis))
    if z_length <= minimum_axis_length:
        return invalid_palm_frame("palm_cross_product_degenerate", bool(is_right))
    z_axis /= z_length
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    try:
        rotation = require_so3(rotation, atol=1e-6)
        quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
        quaternion = align_quaternion_sign(quaternion, previous_quaternion_xyzw)
    except (PalmFrameError, ValueError, np.linalg.LinAlgError) as exc:
        return invalid_palm_frame(f"so3_validation_failed:{exc}", bool(is_right))
    axis_norms = np.linalg.norm(rotation, axis=0)
    quality = {
        "axis_lengths_before_normalization": [x_length, y_raw_length, y_projected_length],
        "axis_unit_norms": axis_norms.tolist(),
        "orthogonality_error": float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
        ),
        "determinant": float(np.linalg.det(rotation)),
        "joint_indices": {
            "wrist": WRIST,
            "index_mcp": INDEX_MCP,
            "middle_mcp": MIDDLE_MCP,
            "little_mcp": LITTLE_MCP,
        },
        "handedness_convention": "anatomical_little_to_index_in_source_camera_axes",
    }
    return HamerPalmFrameResult(
        valid=True,
        rotation=rotation,
        quaternion_xyzw=quaternion,
        origin=wrist,
        is_right=bool(is_right),
        failure_reason="",
        quality=quality,
    )


class RobustBetasCalibrator:
    """Collect 30--60 valid beta vectors and freeze their coordinate-wise median."""

    def __init__(self, required_samples: int = 30, maximum_samples: int = 60) -> None:
        self.required_samples = int(required_samples)
        self.maximum_samples = int(maximum_samples)
        if not 30 <= self.required_samples <= self.maximum_samples <= 60:
            raise ValueError("beta calibration must use 30..60 samples")
        self.reset()

    def reset(self) -> None:
        self._samples: List[np.ndarray] = []
        self._frozen: Optional[np.ndarray] = None
        self._frozen_timestamp: Optional[float] = None
        self._raw_mad: Optional[np.ndarray] = None

    @property
    def frozen(self) -> bool:
        return self._frozen is not None

    @property
    def betas_user(self) -> Optional[np.ndarray]:
        return None if self._frozen is None else self._frozen.copy()

    def add(self, betas: Any, timestamp: float) -> bool:
        values = np.asarray(betas, dtype=np.float64)
        timestamp = float(timestamp)
        if values.shape != (10,) or not np.all(np.isfinite(values)):
            raise ValueError("betas must be a finite 10-vector")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.frozen:
            return False
        self._samples.append(values.copy())
        if len(self._samples) >= self.required_samples:
            stack = np.stack(self._samples[: self.maximum_samples])
            median = np.median(stack, axis=0)
            self._frozen = median
            self._raw_mad = np.median(np.abs(stack - median), axis=0)
            self._frozen_timestamp = timestamp
            return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        stack = np.stack(self._samples) if self._samples else None
        return {
            "required_samples": self.required_samples,
            "maximum_samples": self.maximum_samples,
            "collected_samples": len(self._samples),
            "frozen": self.frozen,
            "raw_betas_min": None if stack is None else stack.min(axis=0).tolist(),
            "raw_betas_max": None if stack is None else stack.max(axis=0).tolist(),
            "raw_betas_std": None if stack is None else stack.std(axis=0).tolist(),
            "raw_betas_mad": None if self._raw_mad is None else self._raw_mad.tolist(),
            "betas_user": None if self._frozen is None else self._frozen.tolist(),
            "frozen_timestamp": self._frozen_timestamp,
            "estimator": "coordinate_wise_median",
        }


__all__ = [
    "HamerPalmFrameResult",
    "RobustBetasCalibrator",
    "build_hamer_joint_palm_frame",
    "invalid_palm_frame",
]
