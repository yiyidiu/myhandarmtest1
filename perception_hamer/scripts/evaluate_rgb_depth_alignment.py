#!/usr/bin/env python3
"""Run a preliminary RGB/aligned-depth edge agreement evaluation."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.depth_evaluation import rgb_depth_edge_alignment_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_directory")
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--maximum-frames", type=int, default=30)
    parser.add_argument("--depth-edge-threshold-raw", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    session = Path(args.session_directory).resolve()
    records = [json.loads(line) for line in (session / "frames.jsonl").open()]
    per_frame = []
    failures = []
    for record in records[: args.maximum_frames]:
        rgb_bgr = cv2.imread(str(session / record["rgb_path"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(session / record["aligned_depth_path"]), cv2.IMREAD_UNCHANGED)
        try:
            per_frame.append(
                rgb_depth_edge_alignment_metrics(
                    rgb, depth, args.roi, args.depth_edge_threshold_raw
                )
            )
        except ValueError as exc:
            failures.append({"index": record["index"], "reason": str(exc)})
    result = {
        "schema_version": 1,
        "test": "PRELIMINARY_USB2_ALIGNMENT",
        "preliminary_only": True,
        "usb_type_descriptor": "2.1",
        "formal_result": False,
        "repeat_required_on_usb3": True,
        "roi_definition": "operator-selected half-open image rectangle",
        "frames_requested": min(len(records), args.maximum_frames),
        "frames_evaluated": len(per_frame),
        "failures": failures,
        "aggregate": {
            key: float(np.median([item[key] for item in per_frame]))
            for key in (
                "depth_to_nearest_rgb_edge_p50_px",
                "depth_to_nearest_rgb_edge_p95_px",
                "depth_to_nearest_rgb_edge_mean_px",
            )
        } if per_frame else None,
        "per_frame": per_frame,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if per_frame else 2


if __name__ == "__main__":
    raise SystemExit(main())
