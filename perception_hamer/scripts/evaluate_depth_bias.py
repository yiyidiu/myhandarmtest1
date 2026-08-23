#!/usr/bin/env python3
"""Evaluate preliminary depth bias for a known-distance planar ROI."""

import argparse
import json
from pathlib import Path
import sys

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.depth_evaluation import depth_bias_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_directory")
    parser.add_argument("--reference-distance-m", type=float, required=True)
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--maximum-frames", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    session = Path(args.session_directory).resolve()
    records = [json.loads(line) for line in (session / "frames.jsonl").open()]
    per_frame = []
    for record in records[: args.maximum_frames]:
        depth = cv2.imread(str(session / record["aligned_depth_path"]), cv2.IMREAD_UNCHANGED)
        per_frame.append(
            depth_bias_metrics(
                depth,
                record["depth_scale_m_per_unit"],
                args.reference_distance_m,
                args.roi,
            )
        )
    result = {
        "schema_version": 1,
        "test": "PRELIMINARY_USB2_DEPTH_BIAS",
        "preliminary_only": True,
        "usb_type_descriptor": "2.1",
        "formal_result": False,
        "repeat_required_on_usb3": True,
        "reference_distance_definition": "operator-measured camera optical center to planar target",
        "frames_evaluated": len(per_frame),
        "per_frame": per_frame,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
