#!/usr/bin/env python3
"""Fail-closed RGB-D rigid increments from a palm ROI.

The transform convention is::

    p_current = rotation_increment @ p_previous + translation_increment

All 3-D quantities are in the aligned color-camera coordinate system and use
metres.  No similarity scale is estimated.  Invalid updates carry ``None`` for
rotation and translation; callers must never substitute an identity transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


class RigidTrackingError(RuntimeError):
    """A fail-closed geometry or tracking error with a machine-readable reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = str(reason)
        message = self.reason if not detail else f"{self.reason}: {detail}"
        super().__init__(message)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RigidTrackingError("NONFINITE_INPUT", f"{name} is not finite")
    return number


def _readonly_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise RigidTrackingError("INVALID_RESULT", f"{name} must be finite {shape}")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RGBDRigidTrackerConfig:
    """Quality and reinitialization gates for pairwise RGB-D tracking."""

    maximum_corners: int = 160
    shi_tomasi_quality: float = 0.01
    minimum_corner_distance_px: float = 6.0
    corner_block_size: int = 7
    klt_window_size_px: int = 21
    klt_pyramid_levels: int = 3
    klt_max_iterations: int = 30
    klt_epsilon: float = 0.01
    maximum_klt_error: float = 30.0
    maximum_fb_error_px: float = 1.0
    minimum_fb_tracks: int = 8
    depth_patch_radius_px: int = 1
    minimum_depth_patch_samples: int = 1
    maximum_depth_patch_mad_m: float = 0.02
    minimum_depth_m: float = 0.10
    maximum_depth_m: float = 3.0
    minimum_3d_pairs: int = 6
    ransac_iterations: int = 64
    ransac_threshold_m: float = 0.012
    minimum_ransac_inliers: int = 6
    minimum_inlier_ratio: float = 0.50
    maximum_kabsch_rms_m: float = 0.008
    minimum_spread_m: float = 0.002
    minimum_secondary_spread_ratio: float = 1e-3
    minimum_dt_s: float = 1e-4
    maximum_dt_s: float = 0.12
    maximum_frame_gap: int = 1
    palm_bbox_erosion_fraction: float = 0.15
    maximum_local_depth_range_m: float = 0.030
    maximum_depth_from_roi_median_m: float = 0.100
    maximum_rotation_increment_deg: float = 30.0
    maximum_translation_increment_m: float = 0.150
    random_seed: int = 7

    def __post_init__(self) -> None:
        positive_ints = (
            self.maximum_corners,
            self.corner_block_size,
            self.klt_window_size_px,
            self.klt_max_iterations,
            self.minimum_fb_tracks,
            self.minimum_depth_patch_samples,
            self.minimum_3d_pairs,
            self.ransac_iterations,
            self.minimum_ransac_inliers,
            self.maximum_frame_gap,
        )
        if any(int(value) <= 0 for value in positive_ints):
            raise ValueError("integer tracker thresholds must be positive")
        if self.klt_pyramid_levels < 0 or self.depth_patch_radius_px < 0:
            raise ValueError("pyramid levels and depth radius must be nonnegative")
        if self.corner_block_size % 2 == 0 or self.klt_window_size_px % 2 == 0:
            raise ValueError("corner block and KLT window sizes must be odd")
        positive_floats = (
            self.shi_tomasi_quality,
            self.minimum_corner_distance_px,
            self.klt_epsilon,
            self.maximum_klt_error,
            self.maximum_fb_error_px,
            self.maximum_depth_patch_mad_m,
            self.minimum_depth_m,
            self.maximum_depth_m,
            self.ransac_threshold_m,
            self.maximum_kabsch_rms_m,
            self.minimum_spread_m,
            self.minimum_dt_s,
            self.maximum_dt_s,
            self.maximum_local_depth_range_m,
            self.maximum_depth_from_roi_median_m,
            self.maximum_rotation_increment_deg,
            self.maximum_translation_increment_m,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive_floats):
            raise ValueError("floating tracker thresholds must be finite and positive")
        if not 0.0 < self.shi_tomasi_quality <= 1.0:
            raise ValueError("Shi-Tomasi quality must be in (0, 1]")
        if not 0.0 < self.minimum_inlier_ratio <= 1.0:
            raise ValueError("minimum inlier ratio must be in (0, 1]")
        if not 0.0 < self.minimum_secondary_spread_ratio <= 1.0:
            raise ValueError("secondary spread ratio must be in (0, 1]")
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("maximum depth must exceed minimum depth")
        if self.maximum_dt_s <= self.minimum_dt_s:
            raise ValueError("maximum dt must exceed minimum dt")
        if self.minimum_ransac_inliers < 3 or self.minimum_3d_pairs < 3:
            raise ValueError("rigid estimation needs at least three points")
        if not 0.0 <= self.palm_bbox_erosion_fraction < 0.45:
            raise ValueError("palm_bbox_erosion_fraction must be in [0,0.45)")
        if self.maximum_rotation_increment_deg > 180.0:
            raise ValueError("maximum_rotation_increment_deg cannot exceed 180")


def _validate_intrinsics(intrinsics: Dict[str, Any], shape: Tuple[int, int]) -> None:
    required = {
        "width",
        "height",
        "fx",
        "fy",
        "ppx",
        "ppy",
        "distortion_model",
        "coeffs",
    }
    if not required.issubset(intrinsics):
        raise RigidTrackingError("INVALID_INTRINSICS", "missing calibrated fields")
    if (int(intrinsics["height"]), int(intrinsics["width"])) != tuple(shape):
        raise RigidTrackingError("INVALID_INTRINSICS", "dimensions do not match RGB-D")
    fx = _finite(intrinsics["fx"], "fx")
    fy = _finite(intrinsics["fy"], "fy")
    _finite(intrinsics["ppx"], "ppx")
    _finite(intrinsics["ppy"], "ppy")
    if fx <= 0.0 or fy <= 0.0:
        raise RigidTrackingError("INVALID_INTRINSICS", "focal lengths must be positive")
    coefficients = np.asarray(intrinsics["coeffs"], dtype=np.float64)
    if coefficients.shape != (5,) or not np.all(np.isfinite(coefficients)):
        raise RigidTrackingError("INVALID_INTRINSICS", "expected five finite coefficients")
    if not str(intrinsics["distortion_model"]):
        raise RigidTrackingError("INVALID_INTRINSICS", "empty distortion model")


def _validate_bbox(bbox_xyxy: Sequence[float], shape: Tuple[int, int]) -> Tuple[float, ...]:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise RigidTrackingError("INVALID_ROI", "bbox must be four finite xyxy values")
    x1, y1, x2, y2 = bbox.tolist()
    if x2 <= x1 or y2 <= y1:
        raise RigidTrackingError("INVALID_ROI", "bbox has nonpositive area")
    height, width = shape
    if min(x2, float(width)) <= max(x1, 0.0) or min(y2, float(height)) <= max(y1, 0.0):
        raise RigidTrackingError("INVALID_ROI", "bbox does not intersect the image")
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class RGBDTrackerFrame:
    """One aligned RGB-D sample with a palm ROI and an actual device timestamp."""

    rgb: np.ndarray
    aligned_depth_raw: np.ndarray
    color_intrinsics: Dict[str, Any]
    depth_scale_m_per_unit: float
    palm_bbox_xyxy: Sequence[float]
    timestamp_s: float
    frame_number: int
    timestamp_domain: str = "device"

    def __post_init__(self) -> None:
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise RigidTrackingError("INVALID_RGB", "expected RGB uint8 HxWx3")
        if self.aligned_depth_raw.dtype != np.uint16 or self.aligned_depth_raw.ndim != 2:
            raise RigidTrackingError("INVALID_DEPTH", "expected aligned uint16 HxW")
        if self.rgb.shape[:2] != self.aligned_depth_raw.shape:
            raise RigidTrackingError("INVALID_DEPTH", "aligned depth and RGB shapes differ")
        _validate_intrinsics(self.color_intrinsics, self.rgb.shape[:2])
        _validate_bbox(self.palm_bbox_xyxy, self.rgb.shape[:2])
        scale = _finite(self.depth_scale_m_per_unit, "depth scale")
        if scale <= 0.0:
            raise RigidTrackingError("INVALID_DEPTH_SCALE", "depth scale must be positive")
        _finite(self.timestamp_s, "timestamp_s")
        if int(self.frame_number) < 0:
            raise RigidTrackingError("INVALID_FRAME_NUMBER", "frame number is negative")
        if not str(self.timestamp_domain):
            raise RigidTrackingError("INVALID_TIMESTAMP_DOMAIN", "timestamp domain is empty")


def rgbd_tracker_frame_from_d455(
    frame: Any, palm_bbox_xyxy: Sequence[float]
) -> RGBDTrackerFrame:
    """Map the synchronized D455 contract without synthesizing time or IDs."""

    # Color and depth sensors maintain independent frame-number sequences; the
    # numeric IDs need not be equal. D455Capture has already verified that raw
    # and aligned frames preserve each stream's own ID/timestamp identity.
    if str(frame.color_timestamp_domain) != str(frame.depth_timestamp_domain):
        raise RigidTrackingError("RGB_DEPTH_TIMESTAMP_DOMAIN_MISMATCH")
    if abs(float(frame.color_timestamp_ms) - float(frame.depth_timestamp_ms)) > 2.0:
        raise RigidTrackingError("RGB_DEPTH_TIMESTAMP_SKEW")
    return RGBDTrackerFrame(
        rgb=frame.rgb,
        aligned_depth_raw=frame.aligned_depth_raw,
        color_intrinsics=dict(frame.color_intrinsics),
        depth_scale_m_per_unit=float(frame.depth_scale_m_per_unit),
        palm_bbox_xyxy=palm_bbox_xyxy,
        timestamp_s=float(frame.color_timestamp_ms) / 1000.0,
        frame_number=int(frame.color_frame_number),
        timestamp_domain=str(frame.color_timestamp_domain),
    )


@dataclass(frozen=True)
class PixelTrackSet:
    previous_pixels: np.ndarray
    current_pixels: np.ndarray
    shi_tomasi_candidates: int
    forward_tracks: int
    fb_errors_px: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        previous = np.asarray(self.previous_pixels, dtype=np.float64)
        current = np.asarray(self.current_pixels, dtype=np.float64)
        if (
            previous.ndim != 2
            or previous.shape[1:] != (2,)
            or current.shape != previous.shape
            or not np.all(np.isfinite(previous))
            or not np.all(np.isfinite(current))
        ):
            raise RigidTrackingError("INVALID_PIXEL_TRACKS")
        previous = np.array(previous, copy=True)
        current = np.array(current, copy=True)
        previous.setflags(write=False)
        current.setflags(write=False)
        object.__setattr__(self, "previous_pixels", previous)
        object.__setattr__(self, "current_pixels", current)
        errors = (
            np.zeros(previous.shape[0], dtype=np.float64)
            if self.fb_errors_px is None
            else np.asarray(self.fb_errors_px, dtype=np.float64)
        )
        if errors.shape != (previous.shape[0],) or not np.all(np.isfinite(errors)):
            raise RigidTrackingError("INVALID_PIXEL_TRACKS", "invalid FB errors")
        errors = np.array(errors, copy=True)
        errors.setflags(write=False)
        object.__setattr__(self, "fb_errors_px", errors)

    @property
    def fb_tracks(self) -> int:
        return int(self.previous_pixels.shape[0])


@dataclass(frozen=True)
class RigidEstimate:
    rotation: np.ndarray
    translation: np.ndarray
    inlier_mask: np.ndarray
    rms_m: float

    def __post_init__(self) -> None:
        rotation = _readonly_array(self.rotation, (3, 3), "rotation")
        translation = _readonly_array(self.translation, (3,), "translation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise RigidTrackingError("INVALID_RESULT", "rotation is not orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
            raise RigidTrackingError("INVALID_RESULT", "rotation determinant is not +1")
        mask = np.asarray(self.inlier_mask, dtype=bool)
        if mask.ndim != 1:
            raise RigidTrackingError("INVALID_RESULT", "inlier mask must be one-dimensional")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        rms = _finite(self.rms_m, "rms_m")
        if rms < 0.0:
            raise RigidTrackingError("INVALID_RESULT", "RMS cannot be negative")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "inlier_mask", mask)
        object.__setattr__(self, "rms_m", rms)

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


@dataclass(frozen=True)
class RigidTrackResult:
    """One pairwise result; invalid transforms are always represented by ``None``."""

    valid: bool
    valid_3d_pairs: int
    ransac_inliers: int
    inlier_ratio: float
    kabsch_rms: Optional[float]
    rotation_increment: Optional[np.ndarray]
    translation_increment: Optional[np.ndarray]
    frame_gap: int
    tracker_age: int
    failure_reason: str
    dt_s: Optional[float]
    reinitialized: bool
    shi_tomasi_candidates: int = 0
    forward_tracks: int = 0
    fb_tracks: int = 0
    spatial_span_m: Optional[float] = None
    covariance_singular_values: Optional[np.ndarray] = None
    tracked_pixels_current: Optional[np.ndarray] = None
    fb_error_p50_px: Optional[float] = None
    fb_error_p95_px: Optional[float] = None
    fb_error_max_px: Optional[float] = None

    def __post_init__(self) -> None:
        if min(
            self.valid_3d_pairs,
            self.ransac_inliers,
            self.frame_gap,
            self.tracker_age,
            self.shi_tomasi_candidates,
            self.forward_tracks,
            self.fb_tracks,
        ) < 0:
            raise RigidTrackingError("INVALID_RESULT", "negative count")
        if self.ransac_inliers > self.valid_3d_pairs:
            raise RigidTrackingError("INVALID_RESULT", "inliers exceed 3-D pairs")
        ratio = _finite(self.inlier_ratio, "inlier_ratio")
        if not 0.0 <= ratio <= 1.0:
            raise RigidTrackingError("INVALID_RESULT", "inlier ratio outside [0, 1]")
        expected_ratio = (
            float(self.ransac_inliers) / float(self.valid_3d_pairs)
            if self.valid_3d_pairs > 0
            else 0.0
        )
        if not math.isclose(ratio, expected_ratio, abs_tol=1e-12):
            raise RigidTrackingError("INVALID_RESULT", "inlier ratio disagrees with counts")
        if self.dt_s is not None and _finite(self.dt_s, "dt_s") <= 0.0:
            raise RigidTrackingError("INVALID_RESULT", "dt must be positive when present")
        if self.valid:
            if self.rotation_increment is None or self.translation_increment is None:
                raise RigidTrackingError("INVALID_RESULT", "valid result is missing SE(3)")
            if self.kabsch_rms is None or self.failure_reason != "NONE":
                raise RigidTrackingError("INVALID_RESULT", "valid result has failure fields")
            if self.dt_s is None or self.reinitialized:
                raise RigidTrackingError("INVALID_RESULT", "valid result has invalid timing state")
            rotation = _readonly_array(self.rotation_increment, (3, 3), "rotation_increment")
            translation = _readonly_array(
                self.translation_increment, (3,), "translation_increment"
            )
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not math.isclose(
                float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7
            ):
                raise RigidTrackingError("INVALID_RESULT", "increment is not in SE(3)")
            _finite(self.kabsch_rms, "kabsch_rms")
            object.__setattr__(self, "rotation_increment", rotation)
            object.__setattr__(self, "translation_increment", translation)
            span = _finite(self.spatial_span_m, "spatial_span_m")
            singular_values = _readonly_array(
                self.covariance_singular_values,
                (3,),
                "covariance_singular_values",
            )
            if span <= 0.0 or np.any(singular_values < 0.0):
                raise RigidTrackingError("INVALID_RESULT", "invalid point-cloud spread")
            object.__setattr__(self, "spatial_span_m", span)
            object.__setattr__(self, "covariance_singular_values", singular_values)
        else:
            if self.rotation_increment is not None or self.translation_increment is not None:
                raise RigidTrackingError(
                    "INVALID_RESULT", "invalid result must not carry a substitute transform"
                )
            if self.kabsch_rms is not None or self.failure_reason == "NONE" or not self.reinitialized:
                raise RigidTrackingError("INVALID_RESULT", "invalid result has success fields")
            if self.spatial_span_m is not None or self.covariance_singular_values is not None:
                raise RigidTrackingError("INVALID_RESULT", "invalid result carries point geometry")
        if not self.failure_reason:
            raise RigidTrackingError("INVALID_RESULT", "failure_reason must be explicit")
        if self.tracked_pixels_current is not None:
            pixels = np.asarray(self.tracked_pixels_current, dtype=np.float64)
            if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.all(np.isfinite(pixels)):
                raise RigidTrackingError("INVALID_RESULT", "invalid tracked pixels")
            pixels = np.array(pixels, copy=True)
            pixels.setflags(write=False)
            object.__setattr__(self, "tracked_pixels_current", pixels)
        for name in ("fb_error_p50_px", "fb_error_p95_px", "fb_error_max_px"):
            value = getattr(self, name)
            if value is not None and _finite(value, name) < 0.0:
                raise RigidTrackingError("INVALID_RESULT", "negative FB error")

    def as_dict(self) -> Dict[str, Any]:
        rotation_degrees = None
        translation_norm = None
        if self.valid:
            cosine = np.clip(
                (float(np.trace(self.rotation_increment)) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
            rotation_degrees = math.degrees(math.acos(float(cosine)))
            translation_norm = float(np.linalg.norm(self.translation_increment))
        return {
            "valid": self.valid,
            "valid_3d_pairs": self.valid_3d_pairs,
            "ransac_inliers": self.ransac_inliers,
            "inlier_ratio": self.inlier_ratio,
            "kabsch_rms": self.kabsch_rms,
            "kabsch_rms_m": self.kabsch_rms,
            "rotation_increment": (
                None
                if self.rotation_increment is None
                else self.rotation_increment.reshape(-1).tolist()
            ),
            "translation_increment": (
                None
                if self.translation_increment is None
                else self.translation_increment.tolist()
            ),
            "translation_increment_m": (
                None if self.translation_increment is None else self.translation_increment.tolist()
            ),
            "translation_increment_norm_m": translation_norm,
            "rotation_increment_deg": rotation_degrees,
            "tracked_2d_points": self.forward_tracks,
            "forward_backward_valid_points": self.fb_tracks,
            "tracked_pixels_current": (
                None if self.tracked_pixels_current is None else self.tracked_pixels_current.tolist()
            ),
            "fb_error_p50_px": self.fb_error_p50_px,
            "fb_error_p95_px": self.fb_error_p95_px,
            "fb_error_max_px": self.fb_error_max_px,
            "spatial_span_m": self.spatial_span_m,
            "covariance_singular_values": (
                None
                if self.covariance_singular_values is None
                else self.covariance_singular_values.tolist()
            ),
            "frame_gap": self.frame_gap,
            "tracker_age": self.tracker_age,
            "failure_reason": self.failure_reason,
            "dt_s": self.dt_s,
            "reinitialized": self.reinitialized,
            "shi_tomasi_candidates": self.shi_tomasi_candidates,
            "forward_tracks": self.forward_tracks,
            "fb_tracks": self.fb_tracks,
            "transform_convention": "p_current = R_increment @ p_previous + t_increment",
            "translation_unit": "m",
            "scale_estimation": "DISABLED",
        }


def _roi_mask(shape: Tuple[int, int], bbox_xyxy: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = _validate_bbox(bbox_xyxy, shape)
    height, width = shape
    left = max(0, int(math.floor(x1)))
    top = max(0, int(math.floor(y1)))
    right = min(width, int(math.ceil(x2)))
    bottom = min(height, int(math.ceil(y2)))
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def erode_palm_bbox(
    bbox_xyxy: Sequence[float], shape: Tuple[int, int], fraction: float
) -> Tuple[float, float, float, float]:
    """Remove the hand contour from an already palm-only manual ROI."""

    x1, y1, x2, y2 = _validate_bbox(bbox_xyxy, shape)
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction < 0.45:
        raise RigidTrackingError("INVALID_ROI", "invalid erosion fraction")
    dx = fraction * (x2 - x1)
    dy = fraction * (y2 - y1)
    return _validate_bbox((x1 + dx, y1 + dy, x2 - dx, y2 - dy), shape)


def build_rigid_palm_mask(
    frame: RGBDTrackerFrame, config: RGBDRigidTrackerConfig
) -> np.ndarray:
    """Palm interior mask excluding contours, holes and depth discontinuities."""

    shape = frame.rgb.shape[:2]
    inner_bbox = erode_palm_bbox(
        frame.palm_bbox_xyxy, shape, config.palm_bbox_erosion_fraction
    )
    roi = _roi_mask(shape, inner_bbox).astype(bool)
    depth_m = frame.aligned_depth_raw.astype(np.float64) * frame.depth_scale_m_per_unit
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= config.minimum_depth_m)
        & (depth_m <= config.maximum_depth_m)
    )
    samples = depth_m[roi & valid]
    if samples.size < config.minimum_3d_pairs:
        return np.zeros(shape, dtype=np.uint8)
    median = float(np.median(samples))
    foreground = np.abs(depth_m - median) <= config.maximum_depth_from_roi_median_m
    kernel = np.ones((3, 3), dtype=np.uint8)
    valid_u8 = (valid & foreground).astype(np.uint8)
    eroded_valid = cv2.erode(valid_u8, kernel, iterations=1).astype(bool)
    local_max = cv2.dilate(depth_m.astype(np.float32), kernel)
    safe_depth = np.where(valid, depth_m, np.nan).astype(np.float32)
    # Invalid neighbours must be rejected, not converted to zero-depth edges.
    filled_for_min = np.where(np.isfinite(safe_depth), safe_depth, np.inf)
    local_min = cv2.erode(filled_for_min, kernel)
    continuous = (
        np.isfinite(local_min)
        & ((local_max - local_min) <= config.maximum_local_depth_range_m)
    )
    return (roi & eroded_valid & continuous).astype(np.uint8) * 255


def robust_palm_center_m(
    frame: RGBDTrackerFrame, config: RGBDRigidTrackerConfig
) -> np.ndarray:
    """Median 3-D center of the depth-continuous palm interior."""

    mask = build_rigid_palm_mask(frame, config).astype(bool)
    rows, columns = np.nonzero(mask)
    if len(rows) < config.minimum_3d_pairs:
        raise RigidTrackingError("INSUFFICIENT_PALM_CENTER_DEPTH")
    # Bounded deterministic subsampling avoids converting every ROI pixel.
    if len(rows) > 2000:
        indices = np.linspace(0, len(rows) - 1, 2000).astype(int)
        rows, columns = rows[indices], columns[indices]
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    depths = (
        frame.aligned_depth_raw[rows, columns].astype(np.float64)
        * frame.depth_scale_m_per_unit
    )
    points = deproject_pixels(pixels, depths, frame.color_intrinsics)
    center = np.median(points, axis=0)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise RigidTrackingError("INVALID_PALM_CENTER")
    return center


def _inside_bbox(points: np.ndarray, bbox_xyxy: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return (
        (points[:, 0] >= x1)
        & (points[:, 0] < x2)
        & (points[:, 1] >= y1)
        & (points[:, 1] < y2)
    )


def track_shi_tomasi_klt_fb(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_bbox_xyxy: Sequence[float],
    current_bbox_xyxy: Sequence[float],
    config: RGBDRigidTrackerConfig,
    previous_feature_mask: Optional[np.ndarray] = None,
) -> PixelTrackSet:
    """Detect in the previous palm ROI and retain forward-backward KLT tracks."""

    if previous_rgb.shape != current_rgb.shape:
        raise RigidTrackingError("RGB_SHAPE_CHANGED")
    previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    mask = _roi_mask(previous_gray.shape, previous_bbox_xyxy)
    current_inner_bbox = erode_palm_bbox(
        current_bbox_xyxy, current_gray.shape, config.palm_bbox_erosion_fraction
    )
    if previous_feature_mask is not None:
        feature_mask = np.asarray(previous_feature_mask)
        if feature_mask.shape != previous_gray.shape or feature_mask.dtype != np.uint8:
            raise RigidTrackingError("INVALID_FEATURE_MASK")
        mask = cv2.bitwise_and(mask, feature_mask)
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=config.maximum_corners,
        qualityLevel=config.shi_tomasi_quality,
        minDistance=config.minimum_corner_distance_px,
        mask=mask,
        blockSize=config.corner_block_size,
        useHarrisDetector=False,
    )
    if corners is None:
        empty = np.empty((0, 2), dtype=np.float64)
        return PixelTrackSet(empty, empty, 0, 0)
    candidates = int(corners.shape[0])
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.klt_max_iterations,
        config.klt_epsilon,
    )
    window = (config.klt_window_size_px, config.klt_window_size_px)
    current_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=window,
        maxLevel=config.klt_pyramid_levels,
        criteria=criteria,
    )
    if current_points is None or forward_status is None:
        empty = np.empty((0, 2), dtype=np.float64)
        return PixelTrackSet(empty, empty, candidates, 0)
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_points,
        None,
        winSize=window,
        maxLevel=config.klt_pyramid_levels,
        criteria=criteria,
    )
    if backward_points is None or backward_status is None:
        empty = np.empty((0, 2), dtype=np.float64)
        return PixelTrackSet(empty, empty, candidates, 0)

    previous = corners.reshape(-1, 2).astype(np.float64)
    current = current_points.reshape(-1, 2).astype(np.float64)
    backward = backward_points.reshape(-1, 2).astype(np.float64)
    forward_ok = forward_status.reshape(-1).astype(bool)
    backward_ok = backward_status.reshape(-1).astype(bool)
    if forward_error is not None:
        error = forward_error.reshape(-1)
        forward_ok &= np.isfinite(error) & (error <= config.maximum_klt_error)
    finite = (
        np.all(np.isfinite(previous), axis=1)
        & np.all(np.isfinite(current), axis=1)
        & np.all(np.isfinite(backward), axis=1)
    )
    forward_count = int(np.count_nonzero(forward_ok & finite))
    fb_error = np.linalg.norm(previous - backward, axis=1)
    keep = (
        forward_ok
        & backward_ok
        & finite
        & (fb_error <= config.maximum_fb_error_px)
        & _inside_bbox(current, current_inner_bbox)
    )
    return PixelTrackSet(
        previous[keep], current[keep], candidates, forward_count, fb_error[keep]
    )


def sample_aligned_depth_m(
    aligned_depth_raw: np.ndarray,
    pixels: np.ndarray,
    depth_scale_m_per_unit: float,
    config: RGBDRigidTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median-sample aligned Z16 around subpixel locations with an MAD edge gate."""

    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise RigidTrackingError("INVALID_PIXEL_TRACKS")
    scale = _finite(depth_scale_m_per_unit, "depth scale")
    if scale <= 0.0:
        raise RigidTrackingError("INVALID_DEPTH_SCALE")
    height, width = aligned_depth_raw.shape
    depths = np.full(points.shape[0], np.nan, dtype=np.float64)
    valid = np.zeros(points.shape[0], dtype=bool)
    radius = config.depth_patch_radius_px
    for index, (u, v) in enumerate(points):
        if not math.isfinite(float(u)) or not math.isfinite(float(v)):
            continue
        column = int(round(float(u)))
        row = int(round(float(v)))
        if column < 0 or column >= width or row < 0 or row >= height:
            continue
        left, right = max(0, column - radius), min(width, column + radius + 1)
        top, bottom = max(0, row - radius), min(height, row + radius + 1)
        patch_m = aligned_depth_raw[top:bottom, left:right].astype(np.float64) * scale
        samples = patch_m[
            np.isfinite(patch_m)
            & (patch_m >= config.minimum_depth_m)
            & (patch_m <= config.maximum_depth_m)
        ]
        if samples.size < config.minimum_depth_patch_samples:
            continue
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        if mad > config.maximum_depth_patch_mad_m:
            continue
        depths[index] = median
        valid[index] = True
    return depths, valid


def deproject_pixels(
    pixels: np.ndarray, depths_m: np.ndarray, intrinsics: Dict[str, Any]
) -> np.ndarray:
    """Deproject aligned color-grid pixels, including D455 Brown distortion."""

    points = np.asarray(pixels, dtype=np.float64)
    depths = np.asarray(depths_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or depths.shape != (points.shape[0],):
        raise RigidTrackingError("INVALID_DEPROJECTION_INPUT")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(depths)) or np.any(depths <= 0.0):
        raise RigidTrackingError("INVALID_DEPROJECTION_INPUT")
    fx = _finite(intrinsics["fx"], "fx")
    fy = _finite(intrinsics["fy"], "fy")
    ppx = _finite(intrinsics["ppx"], "ppx")
    ppy = _finite(intrinsics["ppy"], "ppy")
    coefficients = np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)
    if coefficients.shape != (5,) or not np.all(np.isfinite(coefficients)) or fx <= 0.0 or fy <= 0.0:
        raise RigidTrackingError("INVALID_INTRINSICS")
    model = str(intrinsics.get("distortion_model", "")).lower()
    normalized = np.empty_like(points)
    if np.all(np.abs(coefficients) <= np.finfo(np.float64).eps) or model.endswith("none"):
        normalized[:, 0] = (points[:, 0] - ppx) / fx
        normalized[:, 1] = (points[:, 1] - ppy) / fy
    elif any(
        model.endswith(name)
        for name in (
            "inverse_brown_conrady",
            "brown_conrady",
            "modified_brown_conrady",
        )
    ):
        camera_matrix = np.asarray(
            [[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        normalized = cv2.undistortPoints(
            points.reshape(-1, 1, 2), camera_matrix, coefficients
        ).reshape(-1, 2)
    else:
        raise RigidTrackingError(
            "UNSUPPORTED_DISTORTION_MODEL", str(intrinsics.get("distortion_model"))
        )
    result = np.column_stack(
        (normalized[:, 0] * depths, normalized[:, 1] * depths, depths)
    )
    if not np.all(np.isfinite(result)):
        raise RigidTrackingError("INVALID_DEPROJECTION_RESULT")
    return result


def build_3d_correspondences(
    previous_pixels: np.ndarray,
    current_pixels: np.ndarray,
    previous_frame: RGBDTrackerFrame,
    current_frame: RGBDTrackerFrame,
    config: RGBDRigidTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Depth-gate the same FB-consistent feature in both aligned frames."""

    previous_depths, previous_valid = sample_aligned_depth_m(
        previous_frame.aligned_depth_raw,
        previous_pixels,
        previous_frame.depth_scale_m_per_unit,
        config,
    )
    current_depths, current_valid = sample_aligned_depth_m(
        current_frame.aligned_depth_raw,
        current_pixels,
        current_frame.depth_scale_m_per_unit,
        config,
    )
    valid = previous_valid & current_valid
    if not np.any(valid):
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty.copy()
    previous_3d = deproject_pixels(
        np.asarray(previous_pixels)[valid],
        previous_depths[valid],
        previous_frame.color_intrinsics,
    )
    current_3d = deproject_pixels(
        np.asarray(current_pixels)[valid],
        current_depths[valid],
        current_frame.color_intrinsics,
    )
    return previous_3d, current_3d


def _validated_point_pair(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape != source.shape
        or source.shape[0] < 3
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
    ):
        raise RigidTrackingError("INVALID_3D_CORRESPONDENCES")
    return source, target


def _require_nondegenerate(
    points: np.ndarray, minimum_spread_m: float, minimum_secondary_ratio: float
) -> None:
    centered = points - np.mean(points, axis=0)
    try:
        singular_values = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise RigidTrackingError("KABSCH_SVD_FAILED") from exc
    primary_rms = float(singular_values[0]) / math.sqrt(float(points.shape[0]))
    ratio = (
        float(singular_values[1] / singular_values[0])
        if singular_values[0] > np.finfo(np.float64).eps
        else 0.0
    )
    if primary_rms < minimum_spread_m or ratio < minimum_secondary_ratio:
        raise RigidTrackingError("DEGENERATE_3D_GEOMETRY")


def rigid_kabsch(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    minimum_spread_m: float = 1e-6,
    minimum_secondary_spread_ratio: float = 1e-6,
) -> RigidEstimate:
    """Least-squares rigid fit with determinant correction and no scale term."""

    source, target = _validated_point_pair(source_points, target_points)
    _require_nondegenerate(source, minimum_spread_m, minimum_secondary_spread_ratio)
    _require_nondegenerate(target, minimum_spread_m, minimum_secondary_spread_ratio)
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    try:
        u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as exc:
        raise RigidTrackingError("KABSCH_SVD_FAILED") from exc
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_center - rotation @ source_center
    residuals = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return RigidEstimate(
        rotation=rotation,
        translation=translation,
        inlier_mask=np.ones(source.shape[0], dtype=bool),
        rms_m=rms,
    )


def ransac_rigid_kabsch(
    source_points: np.ndarray,
    target_points: np.ndarray,
    config: RGBDRigidTrackerConfig,
    *,
    rng: Optional[np.random.Generator] = None,
) -> RigidEstimate:
    """Robust SE(3) fit; it raises instead of manufacturing an identity update."""

    source, target = _validated_point_pair(source_points, target_points)
    count = source.shape[0]
    if count < config.minimum_3d_pairs:
        raise RigidTrackingError("INSUFFICIENT_3D_PAIRS")
    _require_nondegenerate(
        source, config.minimum_spread_m, config.minimum_secondary_spread_ratio
    )
    random = rng if rng is not None else np.random.default_rng(config.random_seed)
    required_inliers = max(
        config.minimum_ransac_inliers,
        int(math.ceil(config.minimum_inlier_ratio * float(count))),
    )
    best_mask: Optional[np.ndarray] = None
    best_count = 0
    best_rms = math.inf
    for _ in range(config.ransac_iterations):
        indices = random.choice(count, size=3, replace=False)
        try:
            candidate = rigid_kabsch(
                source[indices],
                target[indices],
                minimum_spread_m=config.minimum_spread_m,
                minimum_secondary_spread_ratio=config.minimum_secondary_spread_ratio,
            )
        except RigidTrackingError:
            continue
        residuals = np.linalg.norm(
            (candidate.rotation @ source.T).T + candidate.translation - target, axis=1
        )
        mask = residuals <= config.ransac_threshold_m
        inlier_count = int(np.count_nonzero(mask))
        if inlier_count < required_inliers:
            continue
        inlier_rms = float(np.sqrt(np.mean(np.square(residuals[mask]))))
        if inlier_count > best_count or (inlier_count == best_count and inlier_rms < best_rms):
            best_mask, best_count, best_rms = mask, inlier_count, inlier_rms
    if best_mask is None:
        raise RigidTrackingError("RANSAC_NO_CONSENSUS")

    mask = best_mask
    for _ in range(2):
        refined = rigid_kabsch(
            source[mask],
            target[mask],
            minimum_spread_m=config.minimum_spread_m,
            minimum_secondary_spread_ratio=config.minimum_secondary_spread_ratio,
        )
        residuals = np.linalg.norm(
            (refined.rotation @ source.T).T + refined.translation - target, axis=1
        )
        updated = residuals <= config.ransac_threshold_m
        if int(np.count_nonzero(updated)) < required_inliers:
            raise RigidTrackingError("RANSAC_REFINEMENT_LOST_CONSENSUS")
        if np.array_equal(updated, mask):
            break
        mask = updated
    final_mask = mask
    final: Optional[RigidEstimate] = None
    for _ in range(5):
        final = rigid_kabsch(
            source[final_mask], target[final_mask],
            minimum_spread_m=config.minimum_spread_m,
            minimum_secondary_spread_ratio=config.minimum_secondary_spread_ratio,
        )
        final_residuals = np.linalg.norm(
            (final.rotation @ source.T).T + final.translation - target, axis=1
        )
        updated_mask = final_residuals <= config.ransac_threshold_m
        if int(np.count_nonzero(updated_mask)) < required_inliers:
            raise RigidTrackingError("RANSAC_FINAL_MODEL_LOST_CONSENSUS")
        if np.array_equal(updated_mask, final_mask):
            break
        final_mask = updated_mask
    else:
        raise RigidTrackingError("RANSAC_FINAL_MASK_DID_NOT_CONVERGE")
    assert final is not None
    if final.rms_m > config.maximum_kabsch_rms_m:
        raise RigidTrackingError("KABSCH_RMS_EXCEEDS_LIMIT")
    return RigidEstimate(
        rotation=final.rotation,
        translation=final.translation,
        inlier_mask=final_mask,
        rms_m=final.rms_m,
    )


class RGBDRigidTracker:
    """Stateful pairwise palm tracker with actual-dt and frame-gap reinitialization."""

    def __init__(self, config: Optional[RGBDRigidTrackerConfig] = None) -> None:
        self.config = config or RGBDRigidTrackerConfig()
        self._previous: Optional[RGBDTrackerFrame] = None
        self._tracker_age = 0

    @property
    def tracker_age(self) -> int:
        return self._tracker_age

    def reset(self) -> None:
        self._previous = None
        self._tracker_age = 0

    def _invalid(
        self,
        current: RGBDTrackerFrame,
        reason: str,
        *,
        frame_gap: int,
        dt_s: Optional[float],
        tracks: Optional[PixelTrackSet] = None,
        valid_3d_pairs: int = 0,
        ransac_inliers: int = 0,
    ) -> RigidTrackResult:
        self._previous = current
        self._tracker_age = 0
        return RigidTrackResult(
            valid=False,
            valid_3d_pairs=valid_3d_pairs,
            ransac_inliers=ransac_inliers,
            inlier_ratio=(
                float(ransac_inliers) / float(valid_3d_pairs)
                if valid_3d_pairs > 0
                else 0.0
            ),
            kabsch_rms=None,
            rotation_increment=None,
            translation_increment=None,
            frame_gap=max(0, int(frame_gap)),
            tracker_age=0,
            failure_reason=reason,
            dt_s=dt_s,
            reinitialized=True,
            shi_tomasi_candidates=0 if tracks is None else tracks.shi_tomasi_candidates,
            forward_tracks=0 if tracks is None else tracks.forward_tracks,
            fb_tracks=0 if tracks is None else tracks.fb_tracks,
            tracked_pixels_current=(None if tracks is None else tracks.current_pixels),
            fb_error_p50_px=(None if tracks is None or tracks.fb_tracks == 0 else float(np.percentile(tracks.fb_errors_px, 50))),
            fb_error_p95_px=(None if tracks is None or tracks.fb_tracks == 0 else float(np.percentile(tracks.fb_errors_px, 95))),
            fb_error_max_px=(None if tracks is None or tracks.fb_tracks == 0 else float(np.max(tracks.fb_errors_px))),
        )

    def process(self, current: RGBDTrackerFrame) -> RigidTrackResult:
        if self._previous is None:
            return self._invalid(
                current,
                "INITIALIZING",
                frame_gap=0,
                dt_s=None,
            )
        previous = self._previous
        raw_gap = int(current.frame_number) - int(previous.frame_number)
        dt_s = float(current.timestamp_s) - float(previous.timestamp_s)
        if current.timestamp_domain != previous.timestamp_domain:
            return self._invalid(
                current, "TIMESTAMP_DOMAIN_CHANGED", frame_gap=max(0, raw_gap), dt_s=None
            )
        if raw_gap <= 0:
            return self._invalid(
                current, "FRAME_NUMBER_NON_INCREASING", frame_gap=0, dt_s=None
            )
        if not math.isfinite(dt_s) or dt_s <= self.config.minimum_dt_s:
            return self._invalid(
                current, "TIMESTAMP_NON_INCREASING", frame_gap=raw_gap, dt_s=None
            )
        if raw_gap > self.config.maximum_frame_gap:
            return self._invalid(
                current, "FRAME_GAP_EXCEEDS_MAXIMUM", frame_gap=raw_gap, dt_s=dt_s
            )
        if dt_s > self.config.maximum_dt_s:
            return self._invalid(
                current, "DT_EXCEEDS_MAXIMUM", frame_gap=raw_gap, dt_s=dt_s
            )

        try:
            tracks = track_shi_tomasi_klt_fb(
                previous.rgb,
                current.rgb,
                previous.palm_bbox_xyxy,
                current.palm_bbox_xyxy,
                self.config,
                previous_feature_mask=build_rigid_palm_mask(previous, self.config),
            )
        except RigidTrackingError as exc:
            return self._invalid(
                current, exc.reason, frame_gap=raw_gap, dt_s=dt_s
            )
        except cv2.error:
            return self._invalid(
                current, "OPTICAL_FLOW_ERROR", frame_gap=raw_gap, dt_s=dt_s
            )
        if tracks.fb_tracks < self.config.minimum_fb_tracks:
            return self._invalid(
                current,
                "INSUFFICIENT_FB_TRACKS",
                frame_gap=raw_gap,
                dt_s=dt_s,
                tracks=tracks,
            )
        try:
            previous_3d, current_3d = build_3d_correspondences(
                tracks.previous_pixels, tracks.current_pixels, previous, current, self.config
            )
        except RigidTrackingError as exc:
            return self._invalid(
                current,
                exc.reason,
                frame_gap=raw_gap,
                dt_s=dt_s,
                tracks=tracks,
            )
        pair_count = int(previous_3d.shape[0])
        if pair_count < self.config.minimum_3d_pairs:
            return self._invalid(
                current,
                "INSUFFICIENT_VALID_DEPTH_PAIRS",
                frame_gap=raw_gap,
                dt_s=dt_s,
                tracks=tracks,
                valid_3d_pairs=pair_count,
            )
        try:
            estimate = ransac_rigid_kabsch(previous_3d, current_3d, self.config)
        except RigidTrackingError as exc:
            return self._invalid(
                current,
                exc.reason,
                frame_gap=raw_gap,
                dt_s=dt_s,
                tracks=tracks,
                valid_3d_pairs=pair_count,
            )

        self._previous = current
        self._tracker_age += 1
        inlier_ratio = float(estimate.inlier_count) / float(pair_count)
        inlier_source = previous_3d[estimate.inlier_mask]
        centered = inlier_source - np.mean(inlier_source, axis=0)
        covariance = centered.T @ centered / float(len(inlier_source))
        covariance_singular_values = np.linalg.svd(covariance, compute_uv=False)
        spatial_span_m = float(np.linalg.norm(np.ptp(inlier_source, axis=0)))
        cosine = float(np.clip((np.trace(estimate.rotation) - 1.0) * 0.5, -1.0, 1.0))
        rotation_increment_deg = math.degrees(math.acos(cosine))
        translation_increment_m = float(np.linalg.norm(estimate.translation))
        if rotation_increment_deg > self.config.maximum_rotation_increment_deg:
            return self._invalid(
                current,
                "ROTATION_INCREMENT_EXCEEDS_LIMIT",
                frame_gap=raw_gap,
                dt_s=dt_s,
                tracks=tracks,
                valid_3d_pairs=pair_count,
                ransac_inliers=estimate.inlier_count,
            )
        if translation_increment_m > self.config.maximum_translation_increment_m:
            return self._invalid(
                current,
                "TRANSLATION_INCREMENT_EXCEEDS_LIMIT",
                frame_gap=raw_gap,
                dt_s=dt_s,
                tracks=tracks,
                valid_3d_pairs=pair_count,
                ransac_inliers=estimate.inlier_count,
            )
        return RigidTrackResult(
            valid=True,
            valid_3d_pairs=pair_count,
            ransac_inliers=estimate.inlier_count,
            inlier_ratio=inlier_ratio,
            kabsch_rms=estimate.rms_m,
            rotation_increment=estimate.rotation,
            translation_increment=estimate.translation,
            frame_gap=raw_gap,
            tracker_age=self._tracker_age,
            failure_reason="NONE",
            dt_s=dt_s,
            reinitialized=False,
            shi_tomasi_candidates=tracks.shi_tomasi_candidates,
            forward_tracks=tracks.forward_tracks,
            fb_tracks=tracks.fb_tracks,
            tracked_pixels_current=tracks.current_pixels,
            fb_error_p50_px=float(np.percentile(tracks.fb_errors_px, 50)),
            fb_error_p95_px=float(np.percentile(tracks.fb_errors_px, 95)),
            fb_error_max_px=float(np.max(tracks.fb_errors_px)),
            spatial_span_m=spatial_span_m,
            covariance_singular_values=covariance_singular_values,
        )


class RelativeTrackingState(str, Enum):
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    FROZEN = "FROZEN"
    LOST = "LOST"


@dataclass(frozen=True)
class RelativeOrientationResult:
    state: RelativeTrackingState
    pairwise: RigidTrackResult
    accumulated_rotation: Optional[np.ndarray]
    accumulated_translation: Optional[np.ndarray]
    orientation_updated: bool
    clutch_required: bool
    frozen_count: int
    lost_count: int
    reinitialization_count: int
    freeze_reason: str = "NONE"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "pairwise": self.pairwise.as_dict(),
            "accumulated_rotation": None if self.accumulated_rotation is None else self.accumulated_rotation.reshape(-1).tolist(),
            "accumulated_translation_m": None if self.accumulated_translation is None else self.accumulated_translation.tolist(),
            "orientation_updated": self.orientation_updated,
            "clutch_required": self.clutch_required,
            "frozen_count": self.frozen_count,
            "lost_count": self.lost_count,
            "reinitialization_count": self.reinitialization_count,
            "freeze_reason": self.freeze_reason,
            "orientation_source": "RGBD_KLT_RANSAC_KABSCH_ONLY",
            "hamer_orientation_used": False,
        }


class RGBDRelativeOrientationTracker:
    """Quality-state wrapper; clutch defines identity and LOST never auto-resumes."""

    def __init__(
        self,
        config: Optional[RGBDRigidTrackerConfig] = None,
        lost_after_s: float = 0.25,
    ) -> None:
        self.core = RGBDRigidTracker(config)
        self.lost_after_s = float(lost_after_s)
        if not math.isfinite(self.lost_after_s) or self.lost_after_s <= 0.0:
            raise ValueError("lost_after_s must be finite and positive")
        self._frozen_count = 0
        self._lost_count = 0
        self._reinitialization_count = 0
        self._state = RelativeTrackingState.LOST
        self._rotation: Optional[np.ndarray] = None
        self._translation: Optional[np.ndarray] = None
        self._last_valid_timestamp: Optional[float] = None

    @property
    def state(self) -> RelativeTrackingState:
        return self._state

    def engage_clutch(self) -> None:
        self.core.reset()
        self._rotation = np.eye(3, dtype=np.float64)
        self._translation = np.zeros(3, dtype=np.float64)
        self._last_valid_timestamp = None
        self._state = RelativeTrackingState.INITIALIZING
        self._reinitialization_count += 1

    def mark_roi_reacquired(self) -> None:
        """ROI may be found while LOST, but orientation stays unavailable."""

        self.core.reset()
        self._rotation = None
        self._translation = None
        self._last_valid_timestamp = None
        self._state = RelativeTrackingState.LOST
        self._reinitialization_count += 1

    def process(
        self, frame: RGBDTrackerFrame, *, externally_frozen: bool = False,
        freeze_reason: str = "NONE"
    ) -> RelativeOrientationResult:
        if self._state == RelativeTrackingState.LOST:
            # Produce a real initializing diagnostic without enabling output.
            pairwise = self.core.process(frame)
            return self._result(pairwise, False, clutch_required=True)
        pairwise = self.core.process(frame)
        if externally_frozen:
            if self._last_valid_timestamp is None and not pairwise.valid:
                self._state = RelativeTrackingState.INITIALIZING
            elif pairwise.valid:
                # Geometry is reliable but deliberately not accumulated while
                # a non-orientation gesture signal says the palm is deforming.
                self._last_valid_timestamp = frame.timestamp_s
                self._state = RelativeTrackingState.FROZEN
                self._frozen_count += 1
            else:
                elapsed = (
                    math.inf if self._last_valid_timestamp is None
                    else frame.timestamp_s - self._last_valid_timestamp
                )
                if elapsed > self.lost_after_s:
                    if self._state != RelativeTrackingState.LOST:
                        self._lost_count += 1
                    self._state = RelativeTrackingState.LOST
                    self._rotation = None
                    self._translation = None
                else:
                    self._state = RelativeTrackingState.FROZEN
                    self._frozen_count += 1
            return self._result(
                pairwise,
                False,
                clutch_required=self._state == RelativeTrackingState.LOST,
                freeze_reason=str(freeze_reason or "EXTERNAL_FREEZE"),
            )
        if pairwise.valid:
            self._rotation = pairwise.rotation_increment @ self._rotation
            self._translation = (
                pairwise.rotation_increment @ self._translation
                + pairwise.translation_increment
            )
            self._last_valid_timestamp = frame.timestamp_s
            self._state = RelativeTrackingState.TRACKING
            return self._result(pairwise, True, clutch_required=False)
        if self._last_valid_timestamp is None:
            # Startup may contain a device gap or one weak feature pair.  No
            # orientation has ever been published, so this remains an honest
            # INITIALIZING state rather than becoming LOST immediately.
            self._state = RelativeTrackingState.INITIALIZING
        else:
            elapsed = (
                math.inf
                if self._last_valid_timestamp is None
                else frame.timestamp_s - self._last_valid_timestamp
            )
            if elapsed > self.lost_after_s:
                if self._state != RelativeTrackingState.LOST:
                    self._lost_count += 1
                self._state = RelativeTrackingState.LOST
                self._rotation = None
                self._translation = None
            else:
                self._state = RelativeTrackingState.FROZEN
                self._frozen_count += 1
        return self._result(
            pairwise,
            False,
            clutch_required=self._state == RelativeTrackingState.LOST,
        )

    def _result(
        self, pairwise: RigidTrackResult, updated: bool, clutch_required: bool,
        freeze_reason: str = "NONE"
    ) -> RelativeOrientationResult:
        rotation = None if self._rotation is None else self._rotation.copy()
        translation = None if self._translation is None else self._translation.copy()
        return RelativeOrientationResult(
            state=self._state,
            pairwise=pairwise,
            accumulated_rotation=rotation,
            accumulated_translation=translation,
            orientation_updated=updated,
            clutch_required=clutch_required,
            frozen_count=self._frozen_count,
            lost_count=self._lost_count,
            reinitialization_count=self._reinitialization_count,
            freeze_reason=freeze_reason,
        )


__all__ = [
    "RGBDRigidTracker",
    "RGBDRigidTrackerConfig",
    "RGBDTrackerFrame",
    "RGBDRelativeOrientationTracker",
    "RelativeOrientationResult",
    "RelativeTrackingState",
    "RigidEstimate",
    "RigidTrackResult",
    "RigidTrackingError",
    "PixelTrackSet",
    "build_3d_correspondences",
    "build_rigid_palm_mask",
    "rgbd_tracker_frame_from_d455",
    "robust_palm_center_m",
    "deproject_pixels",
    "ransac_rigid_kabsch",
    "rigid_kabsch",
    "sample_aligned_depth_m",
    "track_shi_tomasi_klt_fb",
]
