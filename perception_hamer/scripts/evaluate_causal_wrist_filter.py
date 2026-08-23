#!/usr/bin/env python3
"""Replay recorded HaMeR palm frames through the causal SO(3) filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from perception_hamer.src.causal_wrist_so3_filter import (  # noqa: E402
    CausalWristSO3Filter,
)
from perception_hamer.src.crop_quality import bbox_crop_quality  # noqa: E402
from perception_hamer.src.palm_frame import rotation_distance_rad  # noqa: E402


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def evaluate(path: Path) -> dict:
    active = CausalWristSO3Filter()
    previous_bbox = None
    previous_raw = None
    previous_filtered = None
    raw_steps_deg: list[float] = []
    filtered_steps_deg: list[float] = []
    processing_us: list[float] = []
    rows = 0
    source_valid = 0
    accepted = 0
    rejected = 0
    segment_resets = 0

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            rows += 1
            record = json.loads(line)
            palm_frames = record.get("palm_frames") or {}
            joint = (
                palm_frames.get("mano_joint_palm_frame_raw")
                or palm_frames.get("mano_joint_palm_frame")
                or {}
            )
            if not record.get("valid") or not joint.get("valid"):
                active.reset()
                previous_bbox = None
                previous_raw = None
                previous_filtered = None
                segment_resets += 1
                continue
            try:
                timestamp = float(record["timestamp"])
                raw_rotation = np.asarray(joint["rotation"], dtype=np.float64)
                bbox = record.get("bbox") or (record.get("roi") or {}).get("bbox")
                crop_quality = bbox_crop_quality(
                    bbox, previous_bbox, image_width=640, image_height=480
                )
                roi_quality = float(
                    np.clip((record.get("roi") or {}).get("confidence", 0.0), 0.0, 1.0)
                )
                visible_quality = float(
                    np.clip(
                        (record.get("hamer_quality") or {}).get(
                            "bbox_visible_fraction", 0.0
                        ),
                        0.0,
                        1.0,
                    )
                )
                measurement_quality = roi_quality * visible_quality * crop_quality
            except (KeyError, TypeError, ValueError):
                active.reset()
                previous_bbox = None
                previous_raw = None
                previous_filtered = None
                segment_resets += 1
                continue

            source_valid += 1
            if previous_raw is not None:
                raw_steps_deg.append(
                    math.degrees(rotation_distance_rad(previous_raw, raw_rotation))
                )
            began = time.perf_counter()
            result = active.update(timestamp, raw_rotation, measurement_quality)
            processing_us.append((time.perf_counter() - began) * 1.0e6)
            previous_raw = raw_rotation
            previous_bbox = np.asarray(bbox, dtype=np.float64)
            if result.valid:
                accepted += 1
                if previous_filtered is not None:
                    filtered_steps_deg.append(
                        math.degrees(
                            rotation_distance_rad(previous_filtered, result.rotation)
                        )
                    )
                previous_filtered = result.rotation
            else:
                rejected += 1
                previous_filtered = None

    raw_distribution = _distribution(raw_steps_deg)
    filtered_distribution = _distribution(filtered_steps_deg)

    def ratio(field: str):
        raw_value = raw_distribution[field]
        filtered_value = filtered_distribution[field]
        if raw_value is None or filtered_value is None or raw_value <= 0.0:
            return None
        return float(filtered_value / raw_value)

    return {
        "input": str(path.resolve()),
        "input_sha256": _sha256(path),
        "rows": rows,
        "source_valid_frames": source_valid,
        "filter_accepted_frames": accepted,
        "filter_rejected_frames": rejected,
        "segment_resets": segment_resets,
        "raw_interframe_rotation_deg": raw_distribution,
        "filtered_interframe_rotation_deg": filtered_distribution,
        "filtered_over_raw": {
            "median": ratio("median"),
            "p95": ratio("p95"),
            "max": ratio("max"),
        },
        "filter_processing_us": _distribution(processing_us),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing input file(s): " + ", ".join(missing))
    results = [evaluate(path) for path in args.inputs]
    payload = {
        "format": "causal_wrist_so3_filter_replay_v1",
        "method": "quality_adaptive_causal_so3",
        "results": results,
        "caveat": (
            "Inter-frame rotation reduction on existing recordings is not "
            "independent pose-ground-truth accuracy and is not a live USB3 result."
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if any(item["source_valid_frames"] > 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
