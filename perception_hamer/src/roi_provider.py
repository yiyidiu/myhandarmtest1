#!/usr/bin/env python3
"""Fail-closed 2-D hand ROI providers for the crop-only perception path.

All providers consume RGB ``uint8`` images and emit continuous half-open
``[x1, y1, x2, y2)`` boxes in the original image.  This module deliberately
contains no depth, 3-D landmark, palm-frame, or robot-orientation logic.

The MediaPipe adapter is restricted to presence, 2-D landmark bounds, and a
coarse handedness label.  Its detector is imported lazily so the manual and
KLT paths do not require MediaPipe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


EVIDENCE_SCOPE = "USB2_DEVELOPMENT_ONLY"
BBOX_CONVENTION = "continuous_xyxy_half_open_original_rgb"


class ROIError(RuntimeError):
    """Base error for invalid ROI input or provider state."""


class ROIInitializationError(ROIError):
    """Raised when a provider cannot establish a trackable initial ROI."""


@dataclass(frozen=True)
class ROIObservation:
    """One provider result.

    ``bbox`` is ``None`` whenever ``lost`` is true.  Consumers must gate on
    ``lost`` and must never reuse a stale box implicitly.
    """

    bbox: Optional[np.ndarray]
    source: str
    confidence: float
    age: int
    center_jump: float
    scale_change: float
    lost: bool
    reinitialized: bool
    is_right: Optional[bool]
    reason: str = ""
    evidence_scope: str = EVIDENCE_SCOPE

    def __post_init__(self) -> None:
        if self.source not in {"manual_roi", "tracker_roi", "mediapipe_bbox"}:
            raise ValueError("unsupported ROI source")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0,1]")
        if int(self.age) < 0:
            raise ValueError("age must be non-negative")
        if not math.isfinite(float(self.center_jump)) or self.center_jump < 0.0:
            raise ValueError("center_jump must be finite and non-negative")
        if not math.isfinite(float(self.scale_change)) or self.scale_change < 0.0:
            raise ValueError("scale_change must be finite and non-negative")
        if self.is_right is not None and not isinstance(self.is_right, (bool, np.bool_)):
            raise TypeError("is_right must be bool or None")
        if self.lost:
            if self.bbox is not None:
                raise ValueError("a lost observation must not expose a stale bbox")
        else:
            value = np.asarray(self.bbox, dtype=np.float32)
            if value.shape != (4,) or not np.all(np.isfinite(value)):
                raise ValueError("bbox must contain four finite values")
            if value[2] <= value[0] or value[3] <= value[1]:
                raise ValueError("bbox must have positive width and height")
            object.__setattr__(self, "bbox", value.copy())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bbox": None if self.bbox is None else self.bbox.astype(float).tolist(),
            "bbox_convention": BBOX_CONVENTION,
            "source": self.source,
            "confidence": float(self.confidence),
            "age": int(self.age),
            "center_jump": float(self.center_jump),
            "center_jump_units": "pixels",
            "scale_change": float(self.scale_change),
            "scale_change_definition": "sqrt(current_area/previous_area)",
            "lost": bool(self.lost),
            "valid": not bool(self.lost),
            "reinitialized": bool(self.reinitialized),
            "is_right": None if self.is_right is None else bool(self.is_right),
            "reason": self.reason,
            "evidence_scope": self.evidence_scope,
        }


def validate_rgb_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height,width,3)")
    if frame.dtype != np.uint8:
        raise ValueError("frame must have dtype uint8")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError("frame is too small")
    return frame


def clip_bbox(
    bbox: Sequence[float], image_shape: Sequence[int], minimum_size: float = 2.0
) -> Tuple[np.ndarray, float]:
    """Validate and clip a box, returning its visible-area fraction."""

    value = np.asarray(bbox, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("bbox must be finite [x1,y1,x2,y2]")
    x1, y1, x2, y2 = value.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")
    minimum_size = float(minimum_size)
    if not math.isfinite(minimum_size) or minimum_size <= 0.0:
        raise ValueError("minimum_size must be positive")
    height, width = int(image_shape[0]), int(image_shape[1])
    clipped = np.array(
        [
            np.clip(x1, 0.0, float(width)),
            np.clip(y1, 0.0, float(height)),
            np.clip(x2, 0.0, float(width)),
            np.clip(y2, 0.0, float(height)),
        ],
        dtype=np.float32,
    )
    clipped_w = float(clipped[2] - clipped[0])
    clipped_h = float(clipped[3] - clipped[1])
    if clipped_w < minimum_size or clipped_h < minimum_size:
        raise ValueError("bbox has insufficient visible extent")
    requested_area = (x2 - x1) * (y2 - y1)
    visible_fraction = clipped_w * clipped_h / requested_area
    return clipped, float(np.clip(visible_fraction, 0.0, 1.0))


def _motion_metrics(
    previous: Optional[np.ndarray], current: np.ndarray
) -> Tuple[float, float]:
    if previous is None:
        return 0.0, 1.0
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    previous_center = 0.5 * (previous[:2] + previous[2:])
    current_center = 0.5 * (current[:2] + current[2:])
    center_jump = float(np.linalg.norm(current_center - previous_center))
    previous_area = float(np.prod(previous[2:] - previous[:2]))
    current_area = float(np.prod(current[2:] - current[:2]))
    scale_change = math.sqrt(current_area / previous_area)
    return center_jump, scale_change


def _lost(source: str, is_right: Optional[bool], reason: str) -> ROIObservation:
    return ROIObservation(
        bbox=None,
        source=source,
        confidence=0.0,
        age=0,
        center_jump=0.0,
        scale_change=0.0,
        lost=True,
        reinitialized=False,
        is_right=is_right,
        reason=reason,
    )


class HandROIProvider(ABC):
    """Common P3 provider contract."""

    @abstractmethod
    def initialize(self, frame: np.ndarray) -> ROIObservation:
        raise NotImplementedError

    @abstractmethod
    def update(self, frame: np.ndarray) -> ROIObservation:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class ManualROIProvider(HandROIProvider):
    """Fixed user-supplied box used as the detector-independent baseline."""

    def __init__(self, bbox: Sequence[float], is_right: Optional[bool] = None) -> None:
        self._requested_bbox = np.asarray(bbox, dtype=np.float64)
        if self._requested_bbox.shape != (4,) or not np.all(
            np.isfinite(self._requested_bbox)
        ):
            raise ValueError("bbox must be finite [x1,y1,x2,y2]")
        if is_right is not None and not isinstance(is_right, (bool, np.bool_)):
            raise TypeError("is_right must be bool or None")
        self._is_right = None if is_right is None else bool(is_right)
        self.reset()

    def reset(self) -> None:
        self._age = -1
        self._previous_bbox: Optional[np.ndarray] = None

    def initialize(self, frame: np.ndarray) -> ROIObservation:
        self.reset()
        return self._observe(frame, reinitialized=True)

    def update(self, frame: np.ndarray) -> ROIObservation:
        return self._observe(frame, reinitialized=self._age < 0)

    def _observe(self, frame: np.ndarray, reinitialized: bool) -> ROIObservation:
        validate_rgb_frame(frame)
        try:
            bbox, visible_fraction = clip_bbox(self._requested_bbox, frame.shape)
        except ValueError as exc:
            self.reset()
            return _lost("manual_roi", self._is_right, str(exc))
        center_jump, scale_change = _motion_metrics(self._previous_bbox, bbox)
        self._age += 1
        self._previous_bbox = bbox
        return ROIObservation(
            bbox=bbox,
            source="manual_roi",
            confidence=visible_fraction,
            age=self._age,
            center_jump=center_jump,
            scale_change=scale_change,
            lost=False,
            reinitialized=reinitialized,
            is_right=self._is_right,
        )


class KLTTrackerROIProvider(HandROIProvider):
    """Sparse KLT + RANSAC affine ROI tracker.

    The tracker never performs an automatic detector fallback.  After loss, a
    caller must provide a fresh seed through :meth:`reinitialize`; this makes
    reacquisition explicit and prevents a stale box from reaching HaMeR.
    """

    def __init__(
        self,
        initial_bbox: Optional[Sequence[float]] = None,
        is_right: Optional[bool] = None,
        max_features: int = 160,
        min_tracked_points: int = 8,
        quality_level: float = 0.01,
        min_distance: float = 5.0,
        max_forward_backward_error: float = 1.5,
        ransac_reprojection_threshold: float = 2.5,
        minimum_visible_fraction: float = 0.50,
        minimum_scale_change: float = 0.50,
        maximum_scale_change: float = 2.00,
        bbox_smoothing_alpha: float = 1.0,
    ) -> None:
        self._seed_bbox = (
            None if initial_bbox is None else np.asarray(initial_bbox, dtype=np.float64)
        )
        if self._seed_bbox is not None and self._seed_bbox.shape != (4,):
            raise ValueError("initial_bbox must be [x1,y1,x2,y2]")
        if is_right is not None and not isinstance(is_right, (bool, np.bool_)):
            raise TypeError("is_right must be bool or None")
        self._is_right = None if is_right is None else bool(is_right)
        self.max_features = int(max_features)
        self.min_tracked_points = int(min_tracked_points)
        self.quality_level = float(quality_level)
        self.min_distance = float(min_distance)
        self.max_forward_backward_error = float(max_forward_backward_error)
        self.ransac_reprojection_threshold = float(ransac_reprojection_threshold)
        self.minimum_visible_fraction = float(minimum_visible_fraction)
        self.minimum_scale_change = float(minimum_scale_change)
        self.maximum_scale_change = float(maximum_scale_change)
        self.bbox_smoothing_alpha = float(bbox_smoothing_alpha)
        if self.max_features < self.min_tracked_points or self.min_tracked_points < 3:
            raise ValueError("KLT feature limits are inconsistent")
        if not 0.0 <= self.minimum_visible_fraction <= 1.0:
            raise ValueError("minimum_visible_fraction must be in [0,1]")
        if not 0.0 < self.minimum_scale_change <= self.maximum_scale_change:
            raise ValueError("invalid scale-change gate")
        if not 0.0 < self.bbox_smoothing_alpha <= 1.0:
            raise ValueError("bbox_smoothing_alpha must be in (0,1]")
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._previous_gray: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None
        self._bbox: Optional[np.ndarray] = None
        self._age = -1

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        validate_rgb_frame(frame)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def _detect_points(self, gray: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        left = max(0, int(math.floor(float(x1))))
        top = max(0, int(math.floor(float(y1))))
        right = min(gray.shape[1], int(math.ceil(float(x2))))
        bottom = min(gray.shape[0], int(math.ceil(float(y2))))
        mask[top:bottom, left:right] = 255
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=5,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    def initialize(self, frame: np.ndarray) -> ROIObservation:
        if self._seed_bbox is None:
            raise ROIInitializationError("tracker_roi requires an initial bbox")
        return self.reinitialize(frame, self._seed_bbox, self._is_right)

    def reinitialize(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        is_right: Optional[bool] = None,
    ) -> ROIObservation:
        gray = self._gray(frame)
        clipped, visible_fraction = clip_bbox(bbox, frame.shape)
        if visible_fraction < self.minimum_visible_fraction:
            raise ROIInitializationError("initial bbox visible fraction is too low")
        points = self._detect_points(gray, clipped)
        if len(points) < self.min_tracked_points:
            self.reset()
            raise ROIInitializationError(
                "initial bbox contains too few trackable KLT features"
            )
        if is_right is not None and not isinstance(is_right, (bool, np.bool_)):
            raise TypeError("is_right must be bool or None")
        self._is_right = None if is_right is None else bool(is_right)
        self._seed_bbox = np.asarray(bbox, dtype=np.float64).copy()
        self._previous_gray = gray
        self._points = points
        self._bbox = clipped
        self._initialized = True
        self._age = 0
        return ROIObservation(
            bbox=clipped,
            source="tracker_roi",
            confidence=float(visible_fraction),
            age=0,
            center_jump=0.0,
            scale_change=1.0,
            lost=False,
            reinitialized=True,
            is_right=self._is_right,
        )

    def _fail(self, reason: str) -> ROIObservation:
        is_right = self._is_right
        self.reset()
        return _lost("tracker_roi", is_right, reason)

    def update(self, frame: np.ndarray) -> ROIObservation:
        current_gray = self._gray(frame)
        if (
            not self._initialized
            or self._previous_gray is None
            or self._points is None
            or self._bbox is None
        ):
            return _lost("tracker_roi", self._is_right, "not_initialized")
        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            current_gray,
            self._points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or forward_status is None:
            return self._fail("forward_klt_failed")
        back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self._previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if back_points is None or backward_status is None:
            return self._fail("backward_klt_failed")
        previous_flat = self._points.reshape(-1, 2)
        next_flat = next_points.reshape(-1, 2)
        back_flat = back_points.reshape(-1, 2)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.all(np.isfinite(next_flat), axis=1)
            & np.all(np.isfinite(back_flat), axis=1)
        )
        fb_error = np.linalg.norm(previous_flat - back_flat, axis=1)
        valid &= fb_error <= self.max_forward_backward_error
        if int(np.count_nonzero(valid)) < self.min_tracked_points:
            return self._fail("insufficient_forward_backward_consistent_features")
        source = previous_flat[valid]
        target = next_flat[valid]
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reprojection_threshold,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inlier_mask is None or not np.all(np.isfinite(affine)):
            return self._fail("ransac_affine_failed")
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < self.min_tracked_points:
            return self._fail("insufficient_ransac_inliers")
        x1, y1, x2, y2 = self._bbox.astype(np.float64)
        corners = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float64,
        )
        warped = (affine @ corners.T).T
        requested = np.array(
            [
                np.min(warped[:, 0]),
                np.min(warped[:, 1]),
                np.max(warped[:, 0]),
                np.max(warped[:, 1]),
            ],
            dtype=np.float64,
        )
        try:
            current_bbox, visible_fraction = clip_bbox(requested, frame.shape)
        except ValueError:
            return self._fail("tracked_bbox_outside_image")
        if self.bbox_smoothing_alpha < 1.0:
            previous_center = 0.5 * (self._bbox[:2] + self._bbox[2:])
            current_center = 0.5 * (current_bbox[:2] + current_bbox[2:])
            previous_extent = self._bbox[2:] - self._bbox[:2]
            current_extent = current_bbox[2:] - current_bbox[:2]
            alpha = self.bbox_smoothing_alpha
            smoothed_center = (1.0 - alpha) * previous_center + alpha * current_center
            smoothed_extent = np.exp(
                (1.0 - alpha) * np.log(previous_extent)
                + alpha * np.log(current_extent)
            )
            current_bbox, visible_fraction = clip_bbox(
                np.concatenate(
                    [smoothed_center - 0.5 * smoothed_extent,
                     smoothed_center + 0.5 * smoothed_extent]
                ),
                frame.shape,
            )
        center_jump, scale_change = _motion_metrics(self._bbox, current_bbox)
        if visible_fraction < self.minimum_visible_fraction:
            return self._fail("tracked_bbox_visible_fraction_too_low")
        if not self.minimum_scale_change <= scale_change <= self.maximum_scale_change:
            return self._fail("tracked_bbox_scale_change_rejected")

        tracked_fraction = float(np.count_nonzero(valid)) / float(len(self._points))
        inlier_ratio = float(inlier_count) / float(len(source))
        confidence = float(
            np.clip(tracked_fraction * inlier_ratio * visible_fraction, 0.0, 1.0)
        )
        self._bbox = current_bbox
        self._previous_gray = current_gray
        self._age += 1
        refreshed = self._detect_points(current_gray, current_bbox)
        if len(refreshed) >= self.min_tracked_points:
            self._points = refreshed
        else:
            self._points = target[inliers].reshape(-1, 1, 2).astype(np.float32)
        return ROIObservation(
            bbox=current_bbox,
            source="tracker_roi",
            confidence=confidence,
            age=self._age,
            center_jump=center_jump,
            scale_change=scale_change,
            lost=False,
            reinitialized=False,
            is_right=self._is_right,
        )


class MediaPipeBBoxProvider(HandROIProvider):
    """MediaPipe presence/bbox/coarse-handedness adapter only."""

    def __init__(
        self,
        detector: Optional[Any] = None,
        maximum_hands: int = 1,
        minimum_detection_confidence: float = 0.5,
        bbox_margin_fraction: float = 0.15,
    ) -> None:
        self._detector = detector
        self._owns_detector = detector is None
        self.maximum_hands = int(maximum_hands)
        self.minimum_detection_confidence = float(minimum_detection_confidence)
        self.bbox_margin_fraction = float(bbox_margin_fraction)
        if self.maximum_hands < 1:
            raise ValueError("maximum_hands must be positive")
        if not 0.0 <= self.minimum_detection_confidence <= 1.0:
            raise ValueError("minimum_detection_confidence must be in [0,1]")
        if not math.isfinite(self.bbox_margin_fraction) or self.bbox_margin_fraction < 0:
            raise ValueError("bbox_margin_fraction must be finite and non-negative")
        self.reset()

    def _ensure_detector(self) -> Any:
        if self._detector is None:
            try:
                import mediapipe as mp
            except Exception as exc:
                raise ROIInitializationError(
                    "MediaPipe is unavailable; use manual_roi or tracker_roi"
                ) from exc
            self._detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.maximum_hands,
                min_detection_confidence=self.minimum_detection_confidence,
                min_tracking_confidence=self.minimum_detection_confidence,
            )
        return self._detector

    def reset(self) -> None:
        self._previous_bbox: Optional[np.ndarray] = None
        self._age = -1
        self._was_lost = True

    def close(self) -> None:
        if self._owns_detector and self._detector is not None:
            close = getattr(self._detector, "close", None)
            if callable(close):
                close()
            self._detector = None
        self.reset()

    def initialize(self, frame: np.ndarray) -> ROIObservation:
        self.reset()
        return self.update(frame)

    @staticmethod
    def _coarse_handedness(results: Any, index: int) -> Tuple[Optional[bool], float]:
        entries = getattr(results, "multi_handedness", None)
        if entries is None or index >= len(entries):
            return None, 0.5
        classifications = getattr(entries[index], "classification", None)
        if not classifications:
            return None, 0.5
        classification = classifications[0]
        label = str(getattr(classification, "label", "")).strip().lower()
        score = float(getattr(classification, "score", 0.5))
        score = float(np.clip(score if math.isfinite(score) else 0.0, 0.0, 1.0))
        if label == "right":
            return True, score
        if label == "left":
            return False, score
        return None, score

    def update(self, frame: np.ndarray) -> ROIObservation:
        validate_rgb_frame(frame)
        detector = self._ensure_detector()
        results = detector.process(frame)
        candidates = []
        hands = getattr(results, "multi_hand_landmarks", None)
        if hands:
            height, width = frame.shape[:2]
            for index, hand in enumerate(hands[: self.maximum_hands]):
                landmarks = getattr(hand, "landmark", None)
                if not landmarks:
                    continue
                # Only normalized image-plane x/y values are read here.  No
                # depth-like landmark component enters the contract.
                xy = np.asarray(
                    [[float(point.x), float(point.y)] for point in landmarks],
                    dtype=np.float64,
                )
                if xy.ndim != 2 or xy.shape[1] != 2 or not np.all(np.isfinite(xy)):
                    continue
                minimum = xy.min(axis=0) * np.array([width, height], dtype=np.float64)
                maximum = xy.max(axis=0) * np.array([width, height], dtype=np.float64)
                extent = maximum - minimum
                if np.any(extent < 2.0):
                    continue
                margin = self.bbox_margin_fraction * max(float(extent[0]), float(extent[1]))
                requested = np.array(
                    [
                        minimum[0] - margin,
                        minimum[1] - margin,
                        maximum[0] + margin,
                        maximum[1] + margin,
                    ],
                    dtype=np.float64,
                )
                try:
                    bbox, visible_fraction = clip_bbox(requested, frame.shape)
                except ValueError:
                    continue
                is_right, classification_score = self._coarse_handedness(results, index)
                confidence = classification_score * visible_fraction
                area = float(np.prod(bbox[2:] - bbox[:2]))
                candidates.append((confidence, area, bbox, is_right))
        if not candidates:
            self._previous_bbox = None
            self._age = -1
            self._was_lost = True
            return _lost("mediapipe_bbox", None, "hand_not_detected")
        confidence, _, bbox, is_right = max(candidates, key=lambda item: (item[0], item[1]))
        center_jump, scale_change = _motion_metrics(self._previous_bbox, bbox)
        reinitialized = self._was_lost or self._previous_bbox is None
        self._age += 1
        self._previous_bbox = bbox
        self._was_lost = False
        return ROIObservation(
            bbox=bbox,
            source="mediapipe_bbox",
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            age=self._age,
            center_jump=center_jump,
            scale_change=scale_change,
            lost=False,
            reinitialized=reinitialized,
            is_right=is_right,
        )


__all__ = [
    "BBOX_CONVENTION",
    "EVIDENCE_SCOPE",
    "HandROIProvider",
    "KLTTrackerROIProvider",
    "ManualROIProvider",
    "MediaPipeBBoxProvider",
    "ROIError",
    "ROIInitializationError",
    "ROIObservation",
    "clip_bbox",
    "validate_rgb_frame",
]
