#!/usr/bin/env python3
"""Morphology-invariant continuous finger observations from HaMeR joints.

HaMeR exposes its 21 joints in OpenPose hand order.  This module deliberately
does not interpret MANO joint rotation axes: those rotations remain in the
MANO_RIGHT canonical convention and are a poor robot-independent interface.
Instead, each digit is represented by the accumulated unsigned bend of its
four-segment wrist-to-tip chain.  Angles are invariant to camera translation,
rotation and hand scale; the robot-specific retargeting remains on the ROS
side of the UDP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Tuple

import numpy as np


FINGER_FEATURE_DEFINITION = "mano_openpose_chain_total_bend_over_pi_v1"
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


@dataclass(frozen=True)
class FingerObservation:
    """One robot-independent five-digit flexion observation."""

    valid: bool
    flexion: np.ndarray
    confidence: float
    invalid_reason: str
    quality: Mapping[str, Any]

    def as_packet(self) -> dict:
        return {
            "contract_version": 1,
            "feature_definition": FINGER_FEATURE_DEFINITION,
            "valid": bool(self.valid),
            "flexion": np.asarray(self.flexion, dtype=float).tolist(),
            "confidence": float(self.confidence),
            "invalid_reason": str(self.invalid_reason),
        }


def _finite_unit(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(parsed, 0.0, 1.0))


def _chain_flexion(points: np.ndarray, chain: Tuple[int, ...]) -> float:
    segments = np.diff(points[list(chain)], axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.min(lengths) <= 1.0e-8:
        raise ValueError("MANO finger chain contains a zero-length segment")
    if float(np.min(lengths) / np.max(lengths)) < 0.10:
        raise ValueError("MANO finger chain has implausible segment proportions")

    accumulated_bend = 0.0
    chain_points = points[list(chain)]
    for previous, joint, following in zip(
        chain_points[:-2], chain_points[1:-1], chain_points[2:]
    ):
        incoming = previous - joint
        outgoing = following - joint
        cosine = float(
            np.dot(incoming, outgoing)
            / (np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        )
        internal_angle = math.acos(float(np.clip(cosine, -1.0, 1.0)))
        accumulated_bend += math.pi - internal_angle

    # About 180 degrees of accumulated MCP/PIP/DIP bend represents a closed
    # digit for this low-dimensional interface.  Larger anatomical/model
    # excursions saturate instead of producing an out-of-contract command.
    return float(np.clip(accumulated_bend / math.pi, 0.0, 1.0))


def observe_mano_fingers(
    joints: Any,
    roi_confidence: Any,
    visible_fraction: Any,
    crop_quality: Any,
) -> FingerObservation:
    """Extract five continuous flexion features from one exact HaMeR frame."""

    confidence_terms = (
        _finite_unit(roi_confidence, "roi_confidence"),
        _finite_unit(visible_fraction, "visible_fraction"),
        _finite_unit(crop_quality, "crop_quality"),
    )
    confidence = float(np.prod(confidence_terms))
    try:
        points = np.asarray(joints, dtype=np.float64)
        if points.shape != (21, 3) or not np.all(np.isfinite(points)):
            raise ValueError("MANO joints must be a finite 21x3 array")
        palm_extent = float(
            np.median(
                np.linalg.norm(points[[5, 9, 13, 17]] - points[0], axis=1)
            )
        )
        if not math.isfinite(palm_extent) or palm_extent <= 1.0e-6:
            raise ValueError("MANO palm extent is degenerate")
        flexion = np.asarray(
            [_chain_flexion(points, chain) for chain in FINGER_CHAINS],
            dtype=np.float64,
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        return FingerObservation(
            valid=False,
            flexion=np.zeros(5, dtype=np.float64),
            confidence=0.0,
            invalid_reason="FINGER_GEOMETRY_INVALID:{}".format(exc),
            quality={"feature_definition": FINGER_FEATURE_DEFINITION},
        )

    return FingerObservation(
        valid=True,
        flexion=flexion,
        confidence=confidence,
        invalid_reason="",
        quality={
            "feature_definition": FINGER_FEATURE_DEFINITION,
            "palm_extent_model_units": palm_extent,
            "confidence_terms": list(confidence_terms),
        },
    )


def invalid_finger_observation(reason: str) -> dict:
    """Return a geometry-free nested UDP observation for invalid heartbeats."""

    return {
        "contract_version": 1,
        "feature_definition": FINGER_FEATURE_DEFINITION,
        "valid": False,
        "flexion": [0.0] * 5,
        "confidence": 0.0,
        "invalid_reason": str(reason or "FINGER_OBSERVATION_UNAVAILABLE"),
    }


__all__ = [
    "FINGER_CHAINS",
    "FINGER_FEATURE_DEFINITION",
    "FINGER_NAMES",
    "FingerObservation",
    "invalid_finger_observation",
    "observe_mano_fingers",
]
