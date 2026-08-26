#!/usr/bin/env python3
"""Causal RGB-D forearm anchor for the existing MANO wrist frame.

This is a compact Linux adaptation of the supplied Teleoperation Core V5
local-forearm route.  MediaPipe hand pixels only propose a broad 2-D fan.  The
accepted forearm axis is fitted from the current D455 aligned metric depth by
robust PCA and cross-section centres.  The forearm cannot observe twist about
its own longitudinal axis, so MANO remains responsible for roll and the full
wrist pose.

The fusion is deliberately low weight.  If the forearm measurement is absent
or fails a quality gate, the caller receives the original MANO rotation; a
forearm failure never invalidates or hides the hand mesh.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ForearmFusionConfig:
    nominal_length_m: float = 0.16
    minimum_search_px: float = 62.0
    maximum_search_px: float = 230.0
    fan_angles_deg: Tuple[float, ...] = (-35.0, -17.5, 0.0, 17.5, 35.0)
    depth_band_m: float = 0.085
    minimum_wrist_distance_m: float = 0.012
    maximum_wrist_distance_m: float = 0.19
    minimum_points: int = 120
    maximum_points: int = 6000
    cross_section_count: int = 8
    # The supplied V5 defaults require five sections.  On the live D455 view
    # only the distal 10--15 cm is often visible; three robust section centres
    # still define a line while keeping the PCA elongation/span gates active.
    minimum_cross_sections: int = 3
    minimum_points_per_section: int = 8
    minimum_span_m: float = 0.055
    maximum_span_m: float = 0.28
    minimum_axis_ratio: float = 1.45
    maximum_centerline_rms_m: float = 0.020
    minimum_confidence: float = 0.42
    axis_filter_alpha_min: float = 0.20
    axis_filter_alpha_max: float = 0.42
    maximum_axis_innovation_deg: float = 42.0
    hold_timeout_s: float = 0.12
    maximum_fusion_weight: float = 0.20


@dataclass(frozen=True)
class ForearmObservation:
    valid: bool
    axis: Optional[np.ndarray]
    center_m: Optional[np.ndarray]
    confidence: float
    reason: str
    status: str
    age_s: float
    span_m: float
    axis_ratio: float
    centerline_rms_m: float
    point_count: int
    cross_section_count: int
    wrist_pixel: Optional[np.ndarray]
    proximal_pixel: Optional[np.ndarray]
    processing_ms: float
    candidate_diagnostics: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "axis_elbow_to_wrist_camera": (
                None if self.axis is None else self.axis.tolist()
            ),
            "center_m": None if self.center_m is None else self.center_m.tolist(),
            "confidence": float(self.confidence),
            "reason": str(self.reason),
            "status": str(self.status),
            "age_s": float(self.age_s),
            "span_m": float(self.span_m),
            "axis_ratio": float(self.axis_ratio),
            "centerline_rms_m": float(self.centerline_rms_m),
            "point_count": int(self.point_count),
            "cross_section_count": int(self.cross_section_count),
            "wrist_pixel": (
                None if self.wrist_pixel is None else self.wrist_pixel.tolist()
            ),
            "proximal_pixel": (
                None if self.proximal_pixel is None else self.proximal_pixel.tolist()
            ),
            "processing_ms": float(self.processing_ms),
            "candidate_diagnostics": self.candidate_diagnostics,
            "measurement_semantics": (
                "D455_ALIGNED_DEPTH_ROBUST_LOCAL_FOREARM_AXIS_"
                "MEDIAPIPE_2D_ROI_SEED_ONLY"
            ),
            "twist_observable": False,
        }


@dataclass(frozen=True)
class _ForearmJob:
    aligned_depth_raw: np.ndarray
    depth_scale_m_per_unit: float
    color_intrinsics: Dict[str, Any]
    wrist_center_m: np.ndarray
    hand_detection: Optional[Dict[str, Any]]
    detection_age_s: float
    identity: Tuple[int, int, bool, int]
    source_capture_sequence: int
    source_monotonic_s: float
    submitted_monotonic_s: float


@dataclass(frozen=True)
class _CompletedForearmJob:
    observation: ForearmObservation
    identity: Tuple[int, int, bool, int]
    source_capture_sequence: int
    source_monotonic_s: float
    submitted_monotonic_s: float
    started_monotonic_s: float
    completed_monotonic_s: float
    input_version: int


class LatestOnlyForearmEstimator:
    """Run the optional RGB-D forearm anchor outside the HaMeR control path.

    The estimator is intentionally a capacity-one overwrite mailbox. A slow
    depth fit can therefore never build a latency queue behind the camera. A
    result is usable only by the exact hand/presence/context identity that
    submitted it, and its confidence decays with source-frame age before it is
    allowed to perturb the MANO wrist orientation.
    """

    def __init__(
        self,
        config: Optional["ForearmFusionConfig"] = None,
        estimator: Optional["CausalForearmEstimator"] = None,
        maximum_rate_hz: float = 0.0,
    ) -> None:
        self.config = config or ForearmFusionConfig()
        self._estimator = estimator or CausalForearmEstimator(self.config)
        rate = float(maximum_rate_hz)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("maximum_rate_hz must be finite and non-negative")
        self._minimum_period_s = 0.0 if rate == 0.0 else 1.0 / rate
        self._condition = threading.Condition()
        self._job: Optional[_ForearmJob] = None
        self._input_version = 0
        self._consumed_version = 0
        self._completed: Optional[_CompletedForearmJob] = None
        self._worker_identity: Optional[Tuple[int, int, bool, int]] = None
        self._published = 0
        self._completed_count = 0
        self._overwritten = 0
        self._errors = 0
        self._last_error = ""
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="rgbd-forearm-latest-only",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _identity(value: Any) -> Tuple[int, int, bool, int]:
        try:
            presence, active_hand, is_right, context = tuple(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "forearm identity must be "
                "(presence_generation, active_hand_generation, is_right, context)"
            ) from exc
        if not isinstance(is_right, (bool, np.bool_)):
            raise ValueError("forearm identity is_right must be boolean")
        parsed = (int(presence), int(active_hand), bool(is_right), int(context))
        if parsed[0] < 0 or parsed[1] < 0 or parsed[3] < 0:
            raise ValueError("forearm identity generations must be non-negative")
        return parsed

    def submit(
        self,
        aligned_depth_raw: Any,
        depth_scale_m_per_unit: float,
        color_intrinsics: Mapping[str, Any],
        wrist_center_m: Any,
        hand_detection: Optional[Mapping[str, Any]],
        detection_age_s: float,
        identity: Any,
        source_capture_sequence: int,
        source_monotonic_s: float,
        now_monotonic: Optional[float] = None,
    ) -> int:
        """Copy and publish one job without waiting for an older fit."""

        depth = np.asarray(aligned_depth_raw)
        wrist = np.asarray(wrist_center_m, dtype=np.float64)
        if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
            raise ValueError("async forearm depth must be a two-dimensional integer image")
        if wrist.shape != (3,) or not np.all(np.isfinite(wrist)):
            raise ValueError("async forearm wrist center must be a finite 3-vector")
        sequence = int(source_capture_sequence)
        source = float(source_monotonic_s)
        submitted = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        if sequence < 0:
            raise ValueError("source_capture_sequence must be non-negative")
        if (
            not math.isfinite(source)
            or not math.isfinite(submitted)
            or source <= 0.0
            or submitted < source
        ):
            raise ValueError("invalid forearm source/submission monotonic time")
        job = _ForearmJob(
            aligned_depth_raw=np.ascontiguousarray(depth).copy(),
            depth_scale_m_per_unit=float(depth_scale_m_per_unit),
            color_intrinsics=copy.deepcopy(dict(color_intrinsics)),
            wrist_center_m=wrist.copy(),
            hand_detection=(
                None
                if hand_detection is None
                else copy.deepcopy(dict(hand_detection))
            ),
            detection_age_s=float(detection_age_s),
            identity=self._identity(identity),
            source_capture_sequence=sequence,
            source_monotonic_s=source,
            submitted_monotonic_s=submitted,
        )
        with self._condition:
            if self._stopping:
                raise RuntimeError("async forearm estimator is closed")
            if self._input_version > self._consumed_version:
                self._overwritten += 1
            self._job = job
            self._input_version += 1
            self._published += 1
            self._condition.notify_all()
            return self._input_version

    def latest(
        self,
        expected_identity: Any,
        maximum_source_age_s: float,
        now_monotonic: Optional[float] = None,
    ) -> Tuple[Optional[ForearmObservation], Dict[str, Any]]:
        """Return a fresh identity-matched observation plus audit metadata."""

        expected = self._identity(expected_identity)
        maximum_age = float(maximum_source_age_s)
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not math.isfinite(maximum_age) or maximum_age <= 0.0:
            raise ValueError("maximum_source_age_s must be finite and positive")
        if not math.isfinite(now):
            raise ValueError("now_monotonic must be finite")
        with self._condition:
            completed = self._completed
            statistics = self._statistics_locked()
        if completed is None:
            return None, {
                "usable": False,
                "reason": "no_completed_forearm_measurement",
                "expected_identity": list(expected),
                "statistics": statistics,
            }
        source_age = max(0.0, now - completed.source_monotonic_s)
        identity_matches = completed.identity == expected
        within_age = source_age <= maximum_age
        observation = completed.observation
        freshness_gain = float(
            np.clip(1.0 - source_age / maximum_age, 0.0, 1.0)
        )
        usable = bool(identity_matches and within_age and observation.valid)
        reason = "ok"
        if not identity_matches:
            reason = "forearm_identity_mismatch"
        elif not within_age:
            reason = "forearm_source_stale"
        elif not observation.valid:
            reason = "forearm_measurement_invalid:" + str(observation.reason)
        adjusted = None
        if usable:
            adjusted_confidence = float(observation.confidence * freshness_gain)
            if adjusted_confidence >= self.config.minimum_confidence * 0.5:
                adjusted = replace(
                    observation,
                    confidence=adjusted_confidence,
                    status="async_" + str(observation.status),
                    age_s=max(float(observation.age_s), source_age),
                )
            else:
                usable = False
                reason = "forearm_freshness_confidence_below_gate"
        diagnostics = {
            "usable": bool(usable),
            "reason": reason,
            "expected_identity": list(expected),
            "result_identity": list(completed.identity),
            "source_capture_sequence": int(completed.source_capture_sequence),
            "source_age_s": source_age,
            "maximum_source_age_s": maximum_age,
            "freshness_gain": freshness_gain,
            "queue_delay_s": max(
                0.0,
                completed.started_monotonic_s
                - completed.submitted_monotonic_s,
            ),
            "worker_processing_s": max(
                0.0,
                completed.completed_monotonic_s
                - completed.started_monotonic_s,
            ),
            "input_version": int(completed.input_version),
            "statistics": statistics,
        }
        return adjusted, diagnostics

    def _run(self) -> None:
        next_start_monotonic = 0.0
        while True:
            with self._condition:
                while True:
                    if self._stopping:
                        return
                    has_input = self._input_version > self._consumed_version
                    delay = next_start_monotonic - time.monotonic()
                    if has_input and delay <= 0.0:
                        break
                    self._condition.wait(None if not has_input else delay)
                job = self._job
                version = self._input_version
                self._consumed_version = version
            if job is None:
                continue
            started = time.monotonic()
            next_start_monotonic = started + self._minimum_period_s
            if job.identity != self._worker_identity:
                self._estimator.reset()
                self._worker_identity = job.identity
            queued_age = max(0.0, started - job.submitted_monotonic_s)
            try:
                observation = self._estimator.update(
                    job.aligned_depth_raw,
                    job.depth_scale_m_per_unit,
                    job.color_intrinsics,
                    job.wrist_center_m,
                    job.hand_detection,
                    detection_age_s=job.detection_age_s + queued_age,
                    now_monotonic=started,
                )
            except Exception as exc:
                error = "{}:{}".format(type(exc).__name__, exc)
                observation = _invalid(
                    "async forearm worker:" + error,
                    started,
                    status="mano_only_fallback",
                )
                with self._condition:
                    self._errors += 1
                    self._last_error = error
            completed_at = time.monotonic()
            completed = _CompletedForearmJob(
                observation=observation,
                identity=job.identity,
                source_capture_sequence=job.source_capture_sequence,
                source_monotonic_s=job.source_monotonic_s,
                submitted_monotonic_s=job.submitted_monotonic_s,
                started_monotonic_s=started,
                completed_monotonic_s=completed_at,
                input_version=version,
            )
            with self._condition:
                self._completed = completed
                self._completed_count += 1

    def _statistics_locked(self) -> Dict[str, Any]:
        return {
            "published": int(self._published),
            "completed": int(self._completed_count),
            "overwritten_before_estimation": int(self._overwritten),
            "errors": int(self._errors),
            "last_error": str(self._last_error),
            "maximum_rate_hz": (
                0.0
                if self._minimum_period_s <= 0.0
                else 1.0 / self._minimum_period_s
            ),
            "capacity": 1,
            "policy": "overwrite_old_keep_latest",
        }

    @property
    def statistics(self) -> Dict[str, Any]:
        with self._condition:
            return self._statistics_locked()

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=3.0)


def _invalid(
    reason: str,
    started: Optional[float] = None,
    wrist_pixel: Optional[np.ndarray] = None,
    status: str = "invalid",
    candidate_diagnostics: Optional[Dict[str, Any]] = None,
) -> ForearmObservation:
    elapsed_ms = (
        0.0 if started is None else 1000.0 * (time.perf_counter() - started)
    )
    return ForearmObservation(
        False,
        None,
        None,
        0.0,
        str(reason),
        str(status),
        float("inf"),
        0.0,
        0.0,
        float("nan"),
        0,
        0,
        None if wrist_pixel is None else np.asarray(wrist_pixel).copy(),
        None,
        elapsed_ms,
        candidate_diagnostics,
    )


def _unit(vector: Any, dimensions: int = 3) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(dimensions)
    length = float(np.linalg.norm(value))
    if not np.all(np.isfinite(value)) or length < 1.0e-9:
        raise ValueError("cannot normalize a degenerate vector")
    return value / length


def _rotate_2d(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    ) @ np.asarray(vector, dtype=np.float64).reshape(2)


def _camera_parameters(intrinsics: Mapping[str, Any]) -> Tuple[float, ...]:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics.get("ppx", intrinsics.get("cx")))
    cy = float(intrinsics.get("ppy", intrinsics.get("cy")))
    values = (fx, fy, cx, cy)
    if not np.all(np.isfinite(values)) or fx <= 0.0 or fy <= 0.0:
        raise ValueError("invalid color-camera intrinsics")
    return values


def _deproject(
    columns: np.ndarray,
    rows: np.ndarray,
    depths_m: np.ndarray,
    intrinsics: Mapping[str, Any],
) -> np.ndarray:
    fx, fy, cx, cy = _camera_parameters(intrinsics)
    depths = np.asarray(depths_m, dtype=np.float64)
    return np.column_stack(
        (
            (np.asarray(columns, dtype=np.float64) - cx) * depths / fx,
            (np.asarray(rows, dtype=np.float64) - cy) * depths / fy,
            depths,
        )
    )


def _tube_polygon(
    start: np.ndarray,
    end: np.ndarray,
    start_half_width: float,
    end_half_width: float,
) -> np.ndarray:
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    direction = _unit(delta, 2)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    return np.asarray(
        [
            start + normal * start_half_width,
            start - normal * start_half_width,
            end - normal * end_half_width,
            end + normal * end_half_width,
        ],
        dtype=np.float64,
    )


def _robust_axis(points: np.ndarray, minimum_points: int) -> Tuple[np.ndarray, ...]:
    selected = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    for _index in range(3):
        if len(selected) < minimum_points:
            raise ValueError("too few robust forearm points")
        center = np.median(selected, axis=0)
        centered = selected - center
        covariance = centered.T @ centered / max(len(selected) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = _unit(eigenvectors[:, -1])
        axial = centered @ axis
        radial = np.linalg.norm(centered - np.outer(axial, axis), axis=1)
        radial_median = float(np.median(radial))
        radial_mad = float(np.median(np.abs(radial - radial_median)))
        low, high = np.percentile(axial, [2.0, 98.0])
        keep = (
            (axial >= low)
            & (axial <= high)
            & (
                radial
                <= radial_median + max(3.5 * 1.4826 * radial_mad, 0.009)
            )
        )
        if int(np.count_nonzero(keep)) < minimum_points or bool(np.all(keep)):
            break
        selected = selected[keep]
    center = np.median(selected, axis=0)
    centered = selected - center
    covariance = centered.T @ centered / max(len(selected) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return _unit(eigenvectors[:, -1]), center, eigenvalues, selected


def _cross_section_line(
    points: np.ndarray,
    initial_axis: np.ndarray,
    initial_center: np.ndarray,
    config: ForearmFusionConfig,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    axial = (points - initial_center) @ initial_axis
    low, high = np.percentile(axial, [3.0, 97.0])
    edges = np.linspace(low, high, config.cross_section_count + 1)
    centers = []
    for index in range(config.cross_section_count):
        include = (axial >= edges[index]) & (
            axial <= edges[index + 1]
            if index == config.cross_section_count - 1
            else axial < edges[index + 1]
        )
        section = points[include]
        if len(section) >= config.minimum_points_per_section:
            centers.append(np.median(section, axis=0))
    if len(centers) < config.minimum_cross_sections:
        raise ValueError("too few valid forearm cross-sections")
    center_array = np.asarray(centers, dtype=np.float64)
    line_origin = np.mean(center_array, axis=0)
    covariance = (center_array - line_origin).T @ (center_array - line_origin)
    _values, vectors = np.linalg.eigh(covariance)
    line_axis = _unit(vectors[:, -1])
    if float(np.dot(line_axis, initial_axis)) < 0.0:
        line_axis = -line_axis
    # The all-surface PCA is less noisy; the section line prevents an
    # asymmetric visible surface from moving the longitudinal direction.
    axis = _unit(0.85 * initial_axis + 0.15 * line_axis)
    section_delta = center_array - line_origin
    section_axial = section_delta @ axis
    section_residual = np.linalg.norm(
        section_delta - np.outer(section_axial, axis), axis=1
    )
    centerline_rms = float(np.sqrt(np.mean(np.square(section_residual))))
    point_axial = (points - line_origin) @ axis
    span = float(np.percentile(point_axial, 97.5) - np.percentile(point_axial, 2.5))
    return axis, line_origin, span, centerline_rms, len(center_array)


def _candidate_from_direction(
    depth_m: np.ndarray,
    intrinsics: Mapping[str, Any],
    wrist_center_m: np.ndarray,
    wrist_pixel: np.ndarray,
    proximal_direction: np.ndarray,
    search_length_px: float,
    detector_confidence: float,
    config: ForearmFusionConfig,
) -> Dict[str, Any]:
    proximal_pixel = wrist_pixel + search_length_px * proximal_direction
    image_delta = wrist_pixel - proximal_pixel
    image_length = float(np.linalg.norm(image_delta))
    if image_length < 48.0:
        return {"valid": False, "reason": "forearm ROI too short"}
    start = proximal_pixel + 0.12 * image_delta
    end = wrist_pixel - 0.18 * image_delta
    half_width = float(np.clip(0.21 * image_length, 12.0, 54.0))
    polygon = _tube_polygon(start, end, half_width, 0.62 * half_width)
    height, width = depth_m.shape
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, np.rint(polygon).astype(np.int32), 1)
    valid_depth = (
        np.isfinite(depth_m)
        & (depth_m > 0.12)
        & (depth_m < 3.0)
        & (np.abs(depth_m - float(wrist_center_m[2])) <= config.depth_band_m)
    )
    candidate_mask = ((roi_mask > 0) & valid_depth).astype(np.uint8)
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )
    if count <= 1:
        return {"valid": False, "reason": "no connected forearm depth"}
    seed = wrist_pixel + 0.28 * (proximal_pixel - wrist_pixel)
    seed_mask = np.zeros_like(candidate_mask)
    cv2.circle(
        seed_mask,
        tuple(np.rint(seed).astype(int)),
        max(8, int(round(0.20 * half_width))),
        1,
        -1,
    )
    labels_with_seed = [
        label
        for label in range(1, count)
        if int(np.count_nonzero((labels == label) & (seed_mask > 0))) > 0
    ]
    if not labels_with_seed:
        return {"valid": False, "reason": "no wrist-connected forearm depth"}
    selected_label = max(
        labels_with_seed,
        key=lambda label: (
            20 * int(np.count_nonzero((labels == label) & (seed_mask > 0)))
            + int(stats[label, cv2.CC_STAT_AREA])
        ),
    )
    rows, columns = np.nonzero(labels == selected_label)
    if len(rows) < config.minimum_points:
        return {"valid": False, "reason": "forearm component too small"}
    if len(rows) > config.maximum_points:
        selection = np.linspace(
            0, len(rows) - 1, config.maximum_points, dtype=np.int64
        )
        rows, columns = rows[selection], columns[selection]
    depths = depth_m[rows, columns]
    points = _deproject(columns, rows, depths, intrinsics)
    # A 2-D component can connect the distal forearm to the torso/face when
    # they have similar depth.  The archive calls this a *local* forearm ROI;
    # enforce that contract metrically around the independent MANO/D455 wrist
    # instead of accepting an implausible half-metre body component.
    wrist_distances = np.linalg.norm(points - wrist_center_m, axis=1)
    local = (
        (wrist_distances >= config.minimum_wrist_distance_m)
        & (wrist_distances <= config.maximum_wrist_distance_m)
    )
    points = points[local]
    if len(points) < config.minimum_points:
        return {
            "valid": False,
            "reason": "too few metric-local forearm points",
            "point_count": int(len(points)),
        }
    initial_axis = None
    initial_center = None
    eigenvalues = None
    robust_points = None
    try:
        initial_axis, initial_center, eigenvalues, robust_points = _robust_axis(
            points, config.minimum_points
        )
        axis, center, span, line_rms, section_count = _cross_section_line(
            robust_points, initial_axis, initial_center, config
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        diagnostic = {"valid": False, "reason": str(exc)}
        if (
            initial_axis is not None
            and initial_center is not None
            and eigenvalues is not None
            and robust_points is not None
        ):
            axial = (robust_points - initial_center) @ initial_axis
            diagnostic.update(
                span_m=float(
                    np.percentile(axial, 97.5) - np.percentile(axial, 2.5)
                ),
                axis_ratio=float(
                    eigenvalues[-1] / max(eigenvalues[-2], 1.0e-12)
                ),
                point_count=int(len(robust_points)),
            )
        return diagnostic
    # Positive axis is anatomical elbow -> wrist.  This sign convention is
    # exactly what the supplied transported-forearm formulation expects.
    if float(np.dot(axis, wrist_center_m - center)) < 0.0:
        axis = -axis
    ratio = float(eigenvalues[-1] / max(eigenvalues[-2], 1.0e-12))
    valid = bool(
        config.minimum_span_m <= span <= config.maximum_span_m
        and ratio >= config.minimum_axis_ratio
        and line_rms <= config.maximum_centerline_rms_m
    )
    span_score = float(np.clip(span / (1.7 * config.minimum_span_m), 0.0, 1.0))
    ratio_score = float(
        np.clip((ratio - 1.0) / max(2.5 * config.minimum_axis_ratio - 1.0, 1.0e-6), 0.0, 1.0)
    )
    residual_score = float(
        math.exp(-((line_rms / max(config.maximum_centerline_rms_m, 1.0e-6)) ** 2))
    )
    section_score = float(
        np.clip(section_count / max(config.cross_section_count, 1), 0.0, 1.0)
    )
    confidence = float(
        np.clip(detector_confidence, 0.0, 1.0)
        * (span_score * max(ratio_score, 0.10) * residual_score * section_score)
        ** 0.25
    )
    valid = bool(valid and confidence >= config.minimum_confidence)
    return {
        "valid": valid,
        "reason": "ok" if valid else "forearm depth quality gate failed",
        "axis": axis,
        "center": center,
        "confidence": confidence,
        "span_m": span,
        "axis_ratio": ratio,
        "centerline_rms_m": line_rms,
        "point_count": int(len(robust_points)),
        "cross_section_count": int(section_count),
        "proximal_pixel": proximal_pixel,
    }


def estimate_forearm_from_rgbd(
    aligned_depth_raw: Any,
    depth_scale_m_per_unit: float,
    color_intrinsics: Mapping[str, Any],
    wrist_center_m: Any,
    hand_detection: Mapping[str, Any],
    previous_axis: Optional[np.ndarray] = None,
    config: Optional[ForearmFusionConfig] = None,
) -> ForearmObservation:
    """Estimate one current-frame forearm axis; no historical hold is used."""

    started = time.perf_counter()
    settings = config or ForearmFusionConfig()
    try:
        depth_raw = np.asarray(aligned_depth_raw)
        scale = float(depth_scale_m_per_unit)
        wrist_center = np.asarray(wrist_center_m, dtype=np.float64).reshape(3)
        wrist_pixel = np.asarray(
            hand_detection["wrist_pixel"], dtype=np.float64
        ).reshape(2)
        palm_pixels = np.asarray(
            hand_detection["palm_mcp_pixels"], dtype=np.float64
        ).reshape(4, 2)
        detector_confidence = float(hand_detection.get("confidence", 0.0))
        if (
            depth_raw.ndim != 2
            or not np.issubdtype(depth_raw.dtype, np.integer)
            or not math.isfinite(scale)
            or scale <= 0.0
            or not np.all(np.isfinite(wrist_center))
            or not np.all(np.isfinite(wrist_pixel))
            or not np.all(np.isfinite(palm_pixels))
        ):
            raise ValueError("invalid RGB-D forearm inputs")
        fx, fy, _cx, _cy = _camera_parameters(color_intrinsics)
    except (KeyError, TypeError, ValueError) as exc:
        return _invalid(
            "forearm input:{}".format(exc), started,
            locals().get("wrist_pixel"),
        )

    palm_center = np.median(palm_pixels, axis=0)
    try:
        proximal_seed = _unit(wrist_pixel - palm_center, 2)
    except ValueError:
        return _invalid("palm-to-wrist seed too short", started, wrist_pixel)
    focal = 0.5 * (fx + fy)
    search_length = float(
        np.clip(
            focal * settings.nominal_length_m / max(float(wrist_center[2]), 0.12),
            settings.minimum_search_px,
            settings.maximum_search_px,
        )
    )
    depth_m = depth_raw.astype(np.float64) * scale
    candidates = []
    attempts = []
    for angle in settings.fan_angles_deg:
        direction = _unit(_rotate_2d(proximal_seed, angle), 2)
        candidate = _candidate_from_direction(
            depth_m,
            color_intrinsics,
            wrist_center,
            wrist_pixel,
            direction,
            search_length,
            detector_confidence,
            settings,
        )
        diagnostic = {
            "angle_deg": float(angle),
            "valid": bool(candidate.get("valid", False)),
            "reason": str(candidate.get("reason", "unknown")),
            "confidence": float(candidate.get("confidence", 0.0)),
            "span_m": float(candidate.get("span_m", 0.0)),
            "axis_ratio": float(candidate.get("axis_ratio", 0.0)),
            "centerline_rms_m": float(
                candidate.get("centerline_rms_m", float("nan"))
            ),
            "point_count": int(candidate.get("point_count", 0)),
            "cross_section_count": int(
                candidate.get("cross_section_count", 0)
            ),
        }
        attempts.append(diagnostic)
        if candidate.get("valid"):
            if previous_axis is not None:
                axis = np.asarray(candidate["axis"], dtype=np.float64)
                if float(np.dot(axis, previous_axis)) < 0.0:
                    candidate["axis"] = -axis
                cosine = float(
                    np.clip(np.dot(candidate["axis"], previous_axis), -1.0, 1.0)
                )
                candidate["continuity_deg"] = math.degrees(math.acos(cosine))
            else:
                candidate["continuity_deg"] = 0.0
            candidates.append(candidate)
    if not candidates:
        reason_counts: Dict[str, int] = {}
        for item in attempts:
            reason = str(item["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        reason_summary = ",".join(
            "{}x{}".format(count, reason)
            for reason, count in sorted(reason_counts.items())
        )
        return _invalid(
            "local RGB-D fan invalid:" + reason_summary,
            started,
            wrist_pixel,
            candidate_diagnostics={"attempts": attempts},
        )
    selected = max(
        candidates,
        key=lambda value: (
            float(value["confidence"])
            - 0.004 * float(value.get("continuity_deg", 0.0))
        ),
    )
    return ForearmObservation(
        True,
        np.asarray(selected["axis"], dtype=np.float64).copy(),
        np.asarray(selected["center"], dtype=np.float64).copy(),
        float(selected["confidence"]),
        "ok",
        "measurement",
        0.0,
        float(selected["span_m"]),
        float(selected["axis_ratio"]),
        float(selected["centerline_rms_m"]),
        int(selected["point_count"]),
        int(selected["cross_section_count"]),
        wrist_pixel.copy(),
        np.asarray(selected["proximal_pixel"], dtype=np.float64).copy(),
        1000.0 * (time.perf_counter() - started),
        {"attempts": attempts},
    )


class CausalForearmEstimator:
    """Quality-adaptive axis filter with a very short dropout hold."""

    def __init__(self, config: Optional[ForearmFusionConfig] = None) -> None:
        self.config = config or ForearmFusionConfig()
        self._axis: Optional[np.ndarray] = None
        self._center: Optional[np.ndarray] = None
        self._last_valid_monotonic: Optional[float] = None
        self._last_observation: Optional[ForearmObservation] = None

    def reset(self) -> None:
        self._axis = None
        self._center = None
        self._last_valid_monotonic = None
        self._last_observation = None

    def update(
        self,
        aligned_depth_raw: Any,
        depth_scale_m_per_unit: float,
        color_intrinsics: Mapping[str, Any],
        wrist_center_m: Any,
        hand_detection: Optional[Mapping[str, Any]],
        detection_age_s: float = 0.0,
        now_monotonic: Optional[float] = None,
    ) -> ForearmObservation:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if hand_detection is None:
            measurement = _invalid("no confirmed hand pixels")
        elif not math.isfinite(float(detection_age_s)) or float(detection_age_s) > 0.25:
            measurement = _invalid("forearm ROI seed is stale")
        else:
            measurement = estimate_forearm_from_rgbd(
                aligned_depth_raw,
                depth_scale_m_per_unit,
                color_intrinsics,
                wrist_center_m,
                hand_detection,
                self._axis,
                self.config,
            )
        if measurement.valid and measurement.axis is not None:
            candidate = measurement.axis.copy()
            if self._axis is not None:
                if float(np.dot(candidate, self._axis)) < 0.0:
                    candidate = -candidate
                angle = math.degrees(
                    math.acos(float(np.clip(np.dot(candidate, self._axis), -1.0, 1.0)))
                )
                if angle > self.config.maximum_axis_innovation_deg:
                    return self._held_or_invalid(
                        now,
                        "forearm axis innovation rejected ({:.1f} deg)".format(angle),
                        measurement,
                    )
                alpha = float(
                    self.config.axis_filter_alpha_min
                    + np.clip(measurement.confidence, 0.0, 1.0)
                    * (
                        self.config.axis_filter_alpha_max
                        - self.config.axis_filter_alpha_min
                    )
                )
                candidate = _unit((1.0 - alpha) * self._axis + alpha * candidate)
            self._axis = candidate
            self._center = (
                None if measurement.center_m is None else measurement.center_m.copy()
            )
            self._last_valid_monotonic = now
            filtered = ForearmObservation(
                True,
                candidate.copy(),
                None if self._center is None else self._center.copy(),
                measurement.confidence,
                "ok",
                "tracking" if self._last_observation is not None else "initialized",
                0.0,
                measurement.span_m,
                measurement.axis_ratio,
                measurement.centerline_rms_m,
                measurement.point_count,
                measurement.cross_section_count,
                None if measurement.wrist_pixel is None else measurement.wrist_pixel.copy(),
                None if measurement.proximal_pixel is None else measurement.proximal_pixel.copy(),
                measurement.processing_ms,
                measurement.candidate_diagnostics,
            )
            self._last_observation = filtered
            return filtered
        return self._held_or_invalid(now, measurement.reason, measurement)

    def _held_or_invalid(
        self,
        now: float,
        reason: str,
        measurement: Optional[ForearmObservation] = None,
    ) -> ForearmObservation:
        if (
            self._axis is not None
            and self._last_valid_monotonic is not None
            and now - self._last_valid_monotonic <= self.config.hold_timeout_s
            and self._last_observation is not None
        ):
            age = max(0.0, now - self._last_valid_monotonic)
            confidence = self._last_observation.confidence * max(
                0.0, 1.0 - age / self.config.hold_timeout_s
            )
            return ForearmObservation(
                confidence >= self.config.minimum_confidence * 0.5,
                self._axis.copy(),
                None if self._center is None else self._center.copy(),
                confidence,
                str(reason),
                "held_short_dropout",
                age,
                self._last_observation.span_m,
                self._last_observation.axis_ratio,
                self._last_observation.centerline_rms_m,
                self._last_observation.point_count,
                self._last_observation.cross_section_count,
                self._last_observation.wrist_pixel,
                self._last_observation.proximal_pixel,
                0.0,
                self._last_observation.candidate_diagnostics,
            )
        if measurement is not None:
            return ForearmObservation(
                False,
                None,
                None,
                0.0,
                str(reason),
                "mano_only_fallback",
                float("inf"),
                measurement.span_m,
                measurement.axis_ratio,
                measurement.centerline_rms_m,
                measurement.point_count,
                measurement.cross_section_count,
                measurement.wrist_pixel,
                measurement.proximal_pixel,
                measurement.processing_ms,
                measurement.candidate_diagnostics,
            )
        return _invalid(reason, status="mano_only_fallback")


def _project_to_so3(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(value)):
        raise ValueError("non-finite wrist rotation")
    u_value, _singular, vt_value = np.linalg.svd(value)
    rotation = u_value @ vt_value
    if float(np.linalg.det(rotation)) < 0.0:
        u_value[:, -1] *= -1.0
        rotation = u_value @ vt_value
    return rotation


def fuse_wrist_frame_with_forearm(
    mano_rotation: Any,
    observation: Optional[ForearmObservation],
    config: Optional[ForearmFusionConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Regularize MANO's longitudinal axis while preserving MANO roll."""

    settings = config or ForearmFusionConfig()
    raw = _project_to_so3(mano_rotation)
    diagnostics: Dict[str, Any] = {
        "applied": False,
        "fusion_weight": 0.0,
        "correction_deg": 0.0,
        "raw_mano_rotation_row_major": raw.reshape(-1).tolist(),
        "forearm": None if observation is None else observation.as_dict(),
        "roll_source": "MANO_WRIST_RING",
        "forearm_role": "LOW_WEIGHT_LONGITUDINAL_AXIS_ANCHOR",
    }
    if observation is None or not observation.valid or observation.axis is None:
        diagnostics["fallback"] = "MANO_ONLY_FOREARM_UNAVAILABLE"
        return raw, diagnostics
    forearm = _unit(observation.axis)
    mano_longitudinal = raw[:, 2]
    if float(np.dot(forearm, mano_longitudinal)) < 0.0:
        forearm = -forearm
    weight = float(
        settings.maximum_fusion_weight * np.clip(observation.confidence, 0.0, 1.0)
    )
    fused_longitudinal = _unit(
        (1.0 - weight) * mano_longitudinal + weight * forearm
    )
    mano_palm_normal = raw[:, 0]
    fused_normal = mano_palm_normal - fused_longitudinal * float(
        np.dot(mano_palm_normal, fused_longitudinal)
    )
    if float(np.linalg.norm(fused_normal)) < 1.0e-6:
        # This is geometrically rare; use the MANO width axis as a safe basis.
        fused_normal = np.cross(raw[:, 1], fused_longitudinal)
    fused_normal = _unit(fused_normal)
    fused_width = _unit(np.cross(fused_longitudinal, fused_normal))
    fused_normal = _unit(np.cross(fused_width, fused_longitudinal))
    fused = _project_to_so3(
        np.column_stack((fused_normal, fused_width, fused_longitudinal))
    )
    relative = fused @ raw.T
    correction = math.degrees(
        math.acos(float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    )
    diagnostics.update(
        applied=True,
        fusion_weight=weight,
        correction_deg=correction,
        fallback="",
    )
    return fused, diagnostics


def apply_forearm_fusion_to_packet(
    packet: Mapping[str, Any],
    observation: Optional[ForearmObservation],
    config: Optional[ForearmFusionConfig] = None,
) -> Dict[str, Any]:
    """Return a packet with fused rotation and explicit audit diagnostics."""

    output = dict(packet)
    if (packet.get("orientation_channel_valid") is False or
            bool(packet.get("orientation_held", False))):
        raw = np.asarray(
            packet["palm_rotation_row_major"], dtype=np.float64
        ).reshape(3, 3)
        _, diagnostics = fuse_wrist_frame_with_forearm(
            raw, None, config
        )
        diagnostics["forearm"] = (
            None if observation is None else observation.as_dict()
        )
        diagnostics["fallback"] = "ORIENTATION_HELD_NO_FOREARM_FUSION"
        output["forearm_fusion"] = diagnostics
        return output
    fused, diagnostics = fuse_wrist_frame_with_forearm(
        np.asarray(packet["palm_rotation_row_major"], dtype=np.float64).reshape(3, 3),
        observation,
        config,
    )
    output["forearm_fusion"] = diagnostics
    if diagnostics["applied"]:
        output["palm_rotation_row_major"] = fused.reshape(-1).tolist()
        output["orientation_source"] = (
            str(packet.get("orientation_source", "HAMER_MANO_WRIST"))
            + "+D455_RGBD_FOREARM_AXIS_LOW_WEIGHT"
        )
    return output


__all__ = [
    "CausalForearmEstimator",
    "ForearmFusionConfig",
    "ForearmObservation",
    "LatestOnlyForearmEstimator",
    "apply_forearm_fusion_to_packet",
    "estimate_forearm_from_rgbd",
    "fuse_wrist_frame_with_forearm",
]
