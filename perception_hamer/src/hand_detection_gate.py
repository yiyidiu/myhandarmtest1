"""Fail-closed temporal confirmation for 2-D hand detections."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np


def bbox_iou(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    lower = np.maximum(a[:2], b[:2])
    upper = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(0.0, upper-lower)))
    area_a = float(np.prod(np.maximum(0.0, a[2:]-a[:2])))
    area_b = float(np.prod(np.maximum(0.0, b[2:]-b[:2])))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection/union


class ConsecutiveHandDetectionGate:
    """Require a stable handedness and overlapping bbox over several frames."""

    def __init__(self, required_frames: int = 3, minimum_iou: float = 0.35) -> None:
        self.required_frames = int(required_frames)
        self.minimum_iou = float(minimum_iou)
        if self.required_frames < 2:
            raise ValueError("hand confirmation requires at least two frames")
        if not 0.0 <= self.minimum_iou <= 1.0:
            raise ValueError("minimum_iou must be in [0,1]")
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self._previous: Optional[Dict[str, Any]] = None

    def observe(self, detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not detection.get("valid"):
            self.reset()
            return None
        try:
            current_is_right = bool(detection["is_right"])
            current_bbox = np.asarray(detection["bbox"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            self.reset()
            return None
        consistent = bool(
            self._previous is not None
            and bool(self._previous["is_right"]) == current_is_right
            and bbox_iou(self._previous["bbox"], current_bbox) >= self.minimum_iou
        )
        self.count = self.count + 1 if consistent else 1
        self._previous = dict(detection)
        self._previous["bbox"] = current_bbox.tolist()
        if self.count < self.required_frames:
            return None
        confirmed = dict(self._previous)
        confirmed["consecutive_confirmation_frames"] = self.count
        confirmed["minimum_confirmation_iou"] = self.minimum_iou
        return confirmed


class ContinuousHandPresenceGate:
    """Treat detector evidence, never an optical-flow ROI, as hand presence.

    By default one negative detector result fails closed immediately.  Live
    callers may explicitly allow a bounded negative-result grace interval.
    During that interval identity continuity is retained.  A caller may keep
    processing the current camera frame only if its live KLT crop still agrees
    with the last confirmed detector bbox; reusing an old pose is never
    allowed. Exceeding either the frame or time bound fails closed. A missing
    detector result also fails closed after ``timeout_s`` so a dead sidecar
    cannot leave the last MANO mesh or teleoperation packet alive forever.

    ``generation`` changes on every valid/invalid transition.  Display code
    can use it to prevent a mesh inferred before a disappearance from being
    reused after a later reacquisition.
    """

    def __init__(
        self,
        required_frames: int = 2,
        minimum_iou: float = 0.25,
        timeout_s: float = 0.25,
        negative_grace_frames: int = 0,
        negative_grace_s: float = 0.08,
    ) -> None:
        self._confirmation = ConsecutiveHandDetectionGate(
            required_frames=required_frames,
            minimum_iou=minimum_iou,
        )
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        self.negative_grace_frames = int(negative_grace_frames)
        self.negative_grace_s = float(negative_grace_s)
        if self.negative_grace_frames < 0:
            raise ValueError("negative_grace_frames must be non-negative")
        if not math.isfinite(self.negative_grace_s) or self.negative_grace_s < 0.0:
            raise ValueError("negative_grace_s must be finite and non-negative")
        self.valid = False
        self.reason = "no_detector_result"
        self.generation = 0
        self.confirmation_serial = 0
        self.last_result_monotonic: Optional[float] = None
        self.confirmed_detection: Optional[Dict[str, Any]] = None
        self.consecutive_negative_results = 0
        self.first_negative_monotonic: Optional[float] = None

    def _set_valid(self, valid: bool) -> None:
        value = bool(valid)
        if value != self.valid:
            self.valid = value
            self.generation += 1

    def observe(
        self, detection: Dict[str, Any], observed_monotonic: Optional[float] = None
    ) -> Dict[str, Any]:
        now = time.monotonic() if observed_monotonic is None else float(
            observed_monotonic
        )
        if not math.isfinite(now):
            raise ValueError("observed_monotonic must be finite")
        self.last_result_monotonic = now
        if not isinstance(detection, dict) or not detection.get("valid"):
            self.consecutive_negative_results += 1
            if self.first_negative_monotonic is None:
                self.first_negative_monotonic = now
            negative_age = max(0.0, now - self.first_negative_monotonic)
            if (
                self.valid
                and self.consecutive_negative_results
                <= self.negative_grace_frames
                and negative_age <= self.negative_grace_s
            ):
                # Short MediaPipe miss runs are common during fast motion.
                # Retain the confirmed identity/bbox for bounded continuity;
                # the live caller still validates its current KLT crop and
                # computes a new pose from the current camera frame.
                self.reason = "hand_detector_transient_miss_{}/{}".format(
                    self.consecutive_negative_results,
                    self.negative_grace_frames,
                )
                return self.snapshot(now)
            self._confirmation.reset()
            self.confirmed_detection = None
            self._set_valid(False)
            if isinstance(detection, dict):
                self.reason = str(detection.get("reason", "invalid_detection"))
            else:
                self.reason = "malformed_detection"
            return self.snapshot(now)

        self.consecutive_negative_results = 0
        self.first_negative_monotonic = None
        confirmed = self._confirmation.observe(detection)
        if confirmed is not None:
            self.confirmed_detection = confirmed
            self.confirmation_serial += 1
            self._set_valid(True)
            self.reason = "hand_confirmed"
        elif not self.valid:
            self.reason = "hand_reconfirming_{}/{}".format(
                self._confirmation.count,
                self._confirmation.required_frames,
            )
        return self.snapshot(now)

    def snapshot(self, now_monotonic: Optional[float] = None) -> Dict[str, Any]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not math.isfinite(now):
            raise ValueError("now_monotonic must be finite")
        age = (
            float("inf")
            if self.last_result_monotonic is None
            else max(0.0, now - self.last_result_monotonic)
        )
        if age > self.timeout_s:
            self._confirmation.reset()
            self.confirmed_detection = None
            self._set_valid(False)
            self.reason = "hand_detector_timeout"
            self.consecutive_negative_results = 0
            self.first_negative_monotonic = None
        return {
            "valid": bool(self.valid),
            "reason": str(self.reason),
            "generation": int(self.generation),
            "confirmation_serial": int(self.confirmation_serial),
            "detection_age_s": age,
            "confirmed_detection": (
                None
                if self.confirmed_detection is None
                else dict(self.confirmed_detection)
            ),
            "required_confirmation_frames": int(
                self._confirmation.required_frames
            ),
            "consecutive_negative_results": int(
                self.consecutive_negative_results
            ),
            "negative_grace_frames": int(self.negative_grace_frames),
            "negative_grace_s": float(self.negative_grace_s),
            "transient_miss": bool(
                self.valid and self.consecutive_negative_results > 0
            ),
        }


__all__ = [
    "ConsecutiveHandDetectionGate",
    "ContinuousHandPresenceGate",
    "bbox_iou",
]
