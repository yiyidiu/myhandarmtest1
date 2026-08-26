"""Automatic single active-hand selection from multi-hand MediaPipe boxes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .hand_detection_gate import bbox_iou


class AutomaticActiveHandSelector:
    """Keep one active hand while ignoring a simultaneously visible other hand.

    The first usable candidate becomes active automatically.  As long as that
    physical handedness remains visible, an opposite hand cannot steal the
    crop.  If only the opposite hand remains for ``switch_frames`` consecutive
    detector results, an explicit automatic hand-switch event is emitted.
    """

    def __init__(
        self, switch_frames: int = 3, handedness_flip_continuity_iou: float = 0.60
    ) -> None:
        self.switch_frames = int(switch_frames)
        if self.switch_frames < 2:
            raise ValueError("automatic hand switching requires at least two frames")
        self.handedness_flip_continuity_iou = float(
            handedness_flip_continuity_iou
        )
        if not 0.0 <= self.handedness_flip_continuity_iou <= 1.0:
            raise ValueError("handedness flip continuity IoU must be in [0,1]")
        self.active_is_right: Optional[bool] = None
        # Monotonic identity generation for the camera process lifetime.  This
        # is not a biometric hand identifier; it is a fail-closed session
        # identity that changes whenever automatic selection changes sides.
        self.active_hand_generation = 0
        self.previous_bbox: Optional[np.ndarray] = None
        self._switch_candidate_is_right: Optional[bool] = None
        self._switch_count = 0

    @staticmethod
    def _valid_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = payload.get("detections")
        if not isinstance(candidates, list):
            candidates = [payload] if payload.get("valid") else []
        return [
            dict(item) for item in candidates
            if isinstance(item, dict)
            and item.get("valid")
            and isinstance(item.get("is_right"), (bool, np.bool_))
        ]

    def _choose(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.previous_bbox is None:
            return max(
                candidates,
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    float(item.get("bbox_area_fraction", 0.0)),
                ),
            )
        return max(
            candidates,
            key=lambda item: (
                bbox_iou(self.previous_bbox, item.get("bbox")),
                float(item.get("confidence", 0.0)),
            ),
        )

    def _selected_payload(
        self,
        selected: Dict[str, Any],
        all_candidates: List[Dict[str, Any]],
        switched_from: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self.previous_bbox = np.asarray(selected["bbox"], dtype=np.float64).copy()
        result = dict(selected)
        result.update({
            "detections": all_candidates,
            "detected_hand_count": len(all_candidates),
            "active_hand_is_right": bool(self.active_is_right),
            "active_hand_generation": int(self.active_hand_generation),
            "ignored_non_active_hand_count": sum(
                bool(item["is_right"]) != bool(self.active_is_right)
                for item in all_candidates
            ),
            "automatic_hand_switch": switched_from is not None,
            "switched_from_is_right": switched_from,
            "active_hand_policy": "automatic_single_active_hand",
        })
        return result

    def select(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {
                "valid": False,
                "reason": "malformed_detector_payload",
                "active_hand_generation": int(self.active_hand_generation),
            }
        candidates = self._valid_candidates(payload)
        if self.active_is_right is None:
            if not candidates:
                result = dict(payload)
                result["active_hand_policy"] = "automatic_single_active_hand"
                result["active_hand_generation"] = int(
                    self.active_hand_generation
                )
                return result
            selected = self._choose(candidates)
            self.active_is_right = bool(selected["is_right"])
            self.active_hand_generation += 1
            self._switch_candidate_is_right = None
            self._switch_count = 0
            return self._selected_payload(selected, candidates)

        active_candidates = [
            item for item in candidates
            if bool(item["is_right"]) == bool(self.active_is_right)
        ]
        if active_candidates:
            self._switch_candidate_is_right = None
            self._switch_count = 0
            return self._selected_payload(
                self._choose(active_candidates), candidates
            )

        if not candidates:
            self._switch_candidate_is_right = None
            self._switch_count = 0
            return {
                "valid": False,
                "reason": str(payload.get("reason", "active_hand_not_detected")),
                "active_hand_is_right": bool(self.active_is_right),
                "active_hand_generation": int(self.active_hand_generation),
                "active_hand_policy": "automatic_single_active_hand",
                "detections": [],
            }

        # MediaPipe handedness is a coarse per-frame classifier, not a stable
        # physical identity.  A high-overlap box on the immediately preceding
        # track is the same hand with a flipped label; preserve the C-bound
        # identity and MANO reflection instead of manufacturing a hand switch.
        # A spatially distinct opposite hand still follows the bounded switch
        # confirmation path below.
        if self.previous_bbox is not None:
            continuity_candidate = max(
                candidates,
                key=lambda item: bbox_iou(
                    self.previous_bbox, item.get("bbox")
                ),
            )
            continuity_iou = bbox_iou(
                self.previous_bbox, continuity_candidate.get("bbox")
            )
            if continuity_iou >= self.handedness_flip_continuity_iou:
                reported_is_right = bool(continuity_candidate["is_right"])
                stabilized = dict(continuity_candidate)
                stabilized["detector_reported_is_right"] = reported_is_right
                stabilized["is_right"] = bool(self.active_is_right)
                stabilized["handedness_stabilized_by_spatial_continuity"] = True
                stabilized["handedness_continuity_iou"] = float(
                    continuity_iou
                )
                self._switch_candidate_is_right = None
                self._switch_count = 0
                selected = self._selected_payload(stabilized, candidates)
                selected["ignored_non_active_hand_count"] = max(
                    0, len(candidates) - 1
                )
                return selected

        # The active hand is absent, while only a spatially distinct opposite
        # hand is visible.
        candidate = self._choose(candidates)
        candidate_is_right = bool(candidate["is_right"])
        if self._switch_candidate_is_right == candidate_is_right:
            self._switch_count += 1
        else:
            self._switch_candidate_is_right = candidate_is_right
            self._switch_count = 1
        if self._switch_count < self.switch_frames:
            return {
                "valid": False,
                "reason": "confirming_automatic_hand_switch_{}/{}".format(
                    self._switch_count, self.switch_frames
                ),
                "active_hand_is_right": bool(self.active_is_right),
                "active_hand_generation": int(self.active_hand_generation),
                "switch_candidate_is_right": candidate_is_right,
                "active_hand_policy": "automatic_single_active_hand",
                "detections": candidates,
            }
        switched_from = bool(self.active_is_right)
        self.active_is_right = candidate_is_right
        self.active_hand_generation += 1
        self.previous_bbox = None
        self._switch_candidate_is_right = None
        self._switch_count = 0
        return self._selected_payload(candidate, candidates, switched_from)


__all__ = ["AutomaticActiveHandSelector"]
