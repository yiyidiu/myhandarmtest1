#!/usr/bin/env python3
"""Probe the live MediaPipe edge gate with one frozen recorded RGB frame.

The hand image is translated without rotation or rescaling.  Only detector
metadata are retained; translated RGB images are neither written nor emitted.
This isolates the first live-pipeline decision made as an otherwise unchanged
hand approaches each image boundary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path):
    """Keep repository-local provenance independent of one workstation root."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def load_jsonl_index(path, wanted_index):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if int(record.get("index", -1)) == wanted_index:
                return record
    raise ValueError(f"index {wanted_index} is absent from {path}")


def translate_image(rgb, dx, dy):
    transform = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        rgb,
        transform,
        (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def shifted_bbox_at_margin(base, direction, margin, width, height):
    x1, y1, x2, y2 = base
    if direction == "left":
        dx, dy = margin - x1, 0.0
    elif direction == "right":
        dx, dy = width - margin - x2, 0.0
    elif direction == "top":
        dx, dy = 0.0, margin - y1
    elif direction == "bottom":
        dx, dy = 0.0, height - margin - y2
    else:
        raise ValueError(f"unknown direction: {direction}")
    return dx, dy, [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def bbox_visible_fraction(bbox, width, height):
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    visible = (
        max(0.0, min(float(width), x2) - max(0.0, x1))
        * max(0.0, min(float(height), y2) - max(0.0, y1))
    )
    return visible / area if area > 0.0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--observer-records", type=Path, required=True)
    parser.add_argument("--index", type=int, default=145)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--helper-python",
        type=Path,
        default=Path(
            os.environ.get(
                "MEDIAPIPE_PYTHON",
                str(Path.home() / "anaconda3/envs/mediapipe_env/bin/python"),
            )
        ),
    )
    parser.add_argument(
        "--helper-script",
        type=Path,
        default=Path("perception_hamer/scripts/mediapipe_detect_roi_once.py"),
    )
    parser.add_argument(
        "--margins-px", type=float, nargs="+", default=[40.0, 20.0, 10.0, 5.0, 0.0, -20.0]
    )
    args = parser.parse_args()

    frame_record = load_jsonl_index(args.session / "frames.jsonl", args.index)
    observer_record = load_jsonl_index(args.observer_records, args.index)
    rgb_path = args.session / frame_record["rgb_path"]
    if sha256(rgb_path) != frame_record["rgb_sha256"]:
        raise ValueError("recorded RGB hash mismatch")
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode {rgb_path}")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    base_bbox = np.asarray(
        observer_record["absolute_detection"]["roi"]["hand_bbox_xyxy"],
        dtype=np.float64,
    )

    variants = [
        {
            "name": "nominal",
            "direction": "center",
            "requested_margin_px": None,
            "dx_px": 0.0,
            "dy_px": 0.0,
            "expected_shifted_bbox_xyxy": base_bbox.tolist(),
        }
    ]
    for direction in ("left", "right", "top", "bottom"):
        for margin in args.margins_px:
            dx, dy, shifted_bbox = shifted_bbox_at_margin(
                base_bbox, direction, float(margin), width, height
            )
            variants.append(
                {
                    "name": f"{direction}_margin_{margin:g}px",
                    "direction": direction,
                    "requested_margin_px": float(margin),
                    "dx_px": float(dx),
                    "dy_px": float(dy),
                    "expected_shifted_bbox_xyxy": shifted_bbox,
                }
            )

    command = [
        str(args.helper_python.resolve()),
        str(args.helper_script.resolve()),
        "--width",
        str(width),
        "--height",
        str(height),
        "--stream",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=False,
    )
    try:
        for variant in variants:
            shifted = translate_image(rgb, variant["dx_px"], variant["dy_px"])
            process.stdin.write(shifted.tobytes())
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("MediaPipe helper stopped before returning every result")
            detection = json.loads(line.decode("utf-8"))
            variant["expected_bbox_visible_fraction"] = bbox_visible_fraction(
                variant["expected_shifted_bbox_xyxy"], width, height
            )
            variant["detector"] = detection
        process.stdin.close()
        return_code = process.wait(timeout=30)
        if return_code != 0:
            raise RuntimeError(f"MediaPipe helper exited {return_code}")
    finally:
        if process.poll() is None:
            process.kill()

    by_direction = {}
    for direction in ("left", "right", "top", "bottom"):
        rows = [row for row in variants if row["direction"] == direction]
        by_direction[direction] = {
            "valid_requested_margins_px": [
                row["requested_margin_px"] for row in rows
                if row["detector"].get("valid", False)
            ],
            "invalid_requested_margins_px": [
                row["requested_margin_px"] for row in rows
                if not row["detector"].get("valid", False)
            ],
            "invalid_reasons": [
                row["detector"].get("reason") for row in rows
                if not row["detector"].get("valid", False)
            ],
        }

    result = {
        "schema": "handarm_m1_frame_edge_presence_probe_v1",
        "status": "CONTROLLED_SYNTHETIC_REPLAY_NOT_LIVE_EDGE_VALIDATION",
        "source_session": portable_path(args.session),
        "source_frame_index": args.index,
        "source_rgb_relative_path": frame_record["rgb_path"],
        "source_rgb_sha256": frame_record["rgb_sha256"],
        "source_observer_records": portable_path(args.observer_records),
        "source_observer_records_sha256": sha256(args.observer_records),
        "image_size_px": [width, height],
        "base_hand_bbox_xyxy": base_bbox.tolist(),
        "physical_pose_transform": "2D translation only; no rotation or rescaling",
        "retained_output": "detector metadata only; no translated RGB retained",
        "live_detector_command": [
            "${MEDIAPIPE_PYTHON}",
            portable_path(args.helper_script),
            "--width",
            str(width),
            "--height",
            str(height),
            "--stream",
        ],
        "live_detector_runtime_note": (
            "Set MEDIAPIPE_PYTHON to the Python interpreter containing the "
            "project's MediaPipe dependencies."
        ),
        "detector_edge_margin_px": 8.0,
        "summary_by_direction": by_direction,
        "variants": variants,
        "evidence_boundary": (
            "This isolates the current image-plane presence gate on one recorded "
            "pose. It does not reproduce real edge-view biomechanics, lighting, "
            "depth loss, temporal KLT behavior, or live control recovery."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
