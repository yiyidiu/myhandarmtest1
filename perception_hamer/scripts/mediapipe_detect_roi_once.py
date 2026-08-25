#!/usr/bin/env python3
"""One-shot MediaPipe hand presence/2-D bbox/coarse-handedness sidecar.

Raw RGB bytes are read from stdin.  Only the 2-D wrist and four palm-root
pixels are exported in addition to the bounding box.  They seed the local
RGB-D forearm search; MediaPipe z/world landmarks are deliberately never
exported and cannot become the MANO wrist orientation.
"""

from __future__ import annotations

import argparse
import json
import sys

import mediapipe as mp
import numpy as np


def _read_exact(size: int):
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _detect(
    detector, rgb: np.ndarray, width: int, height: int, margin: float,
    minimum_bbox_area_fraction: float, edge_margin_px: float,
) -> dict:
    result = detector.process(rgb)
    if not result.multi_hand_landmarks:
        return {"valid": False, "reason": "no_hand_detected"}
    candidates = []
    rejected = []
    handedness_results = result.multi_handedness or []
    for index, landmarks in enumerate(result.multi_hand_landmarks):
        if index >= len(handedness_results):
            rejected.append({"valid": False, "reason": "missing_handedness"})
            continue
        # Landmarks are used only as a set of 2-D points for their bounds.
        # Their individual identities, z and world coordinates are ignored.
        xy = np.asarray(
            [[point.x * width, point.y * height]
             for point in landmarks.landmark],
            dtype=np.float64,
        )
        lower, upper = xy.min(axis=0), xy.max(axis=0)
        extent = upper - lower
        expanded_lower = lower - margin * extent
        expanded_upper = upper + margin * extent
        clipped_lower = np.maximum(expanded_lower, [0.0, 0.0])
        clipped_upper = np.minimum(expanded_upper, [float(width), float(height)])
        area_fraction = float(
            np.prod(np.maximum(0.0, clipped_upper-clipped_lower))
        ) / float(width*height)
        observed_bbox = [
            float(clipped_lower[0]), float(clipped_lower[1]),
            float(clipped_upper[0]), float(clipped_upper[1]),
        ]
        candidate = {
            "valid": True,
            "bbox": observed_bbox,
            "bbox_area_fraction": area_fraction,
            # These five image points are an ROI proposal for the independent
            # aligned-depth forearm fit.  They are not a pose measurement.
            "wrist_pixel": xy[0].tolist(),
            "palm_mcp_pixels": xy[[5, 9, 13, 17]].tolist(),
        }
        if area_fraction < float(minimum_bbox_area_fraction):
            candidate.update(valid=False, reason="hand_bbox_too_small")
            rejected.append(candidate)
            continue
        edge = float(edge_margin_px)
        if (np.any(expanded_lower < edge)
                or expanded_upper[0] > float(width)-edge
                or expanded_upper[1] > float(height)-edge):
            candidate.update(valid=False, reason="hand_bbox_touches_image_edge")
            rejected.append(candidate)
            continue
        handedness = handedness_results[index].classification[0]
        # MediaPipe documents handedness for mirrored/selfie input.  D455 RGB
        # is not mirrored, so exchange Left/Right for the physical operator.
        media_pipe_label = handedness.label.lower()
        candidate.update({
            "is_right": media_pipe_label == "left",
            "confidence": float(handedness.score),
            "mediapipe_selfie_label": media_pipe_label,
            "input_mirrored": False,
            "handedness_label_exchanged_for_unmirrored_d455": True,
            "allowed_usage": (
                "presence_2d_bbox_coarse_handedness_and_"
                "rgbd_forearm_roi_seed_only"
            ),
        })
        candidates.append(candidate)
    if not candidates:
        reason = (
            rejected[0].get("reason", "no_usable_hand_detection")
            if rejected else "no_usable_hand_detection"
        )
        return {
            "valid": False,
            "reason": reason,
            "rejected_detections": rejected,
        }
    selected = dict(max(candidates, key=lambda item: item["confidence"]))
    selected["detections"] = candidates
    selected["detected_hand_count"] = len(candidates)
    selected["rejected_detections"] = rejected
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--margin", type=float, default=0.18)
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--min-bbox-area-fraction", type=float, default=0.012)
    parser.add_argument("--edge-margin-px", type=float, default=8.0)
    parser.add_argument("--stream", action="store_true",
                        help="read fixed-size RGB frames until stdin closes")
    args = parser.parse_args()
    expected = args.width * args.height * 3
    with mp.solutions.hands.Hands(
        # Presence must be decided independently for every D455 image.  The
        # MediaPipe tracking shortcut can otherwise propagate stale landmarks
        # for several frames after the real hand has left and create a ghost
        # crop/HaMeR mesh.
        static_image_mode=True,
        # Return both physical hands as 2-D candidates.  The parent process
        # automatically keeps one active hand; a simultaneously visible other
        # hand is ignored and is never sent to the single HaMeR/MANO model.
        max_num_hands=2,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=0.50,
    ) as detector:
        while True:
            payload = _read_exact(expected)
            if payload is None:
                return 0 if args.stream else 2
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(
                args.height, args.width, 3
            )
            result = _detect(
                detector, rgb, args.width, args.height, args.margin,
                args.min_bbox_area_fraction, args.edge_margin_px,
            )
            print(json.dumps(result, separators=(",", ":")), flush=True)
            if not args.stream:
                return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
