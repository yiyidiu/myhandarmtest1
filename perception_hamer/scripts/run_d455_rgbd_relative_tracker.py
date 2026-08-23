#!/usr/bin/env python3
"""Record independent D455 RGB-D KLT/RANSAC-Kabsch relative orientation.

No HaMeR/MANO orientation is imported or accepted. A manual palm-only ROI is
selected once (or supplied explicitly); the central eroded, depth-continuous
region feeds sequential RGB-D tracking at the camera rate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Dict, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(ROOT))

from perception_hamer.src.d455_capture import D455Capture
from perception_hamer.src.rgbd_rigid_tracker import (
    RGBDRelativeOrientationTracker,
    RGBDRigidTrackerConfig,
    RelativeTrackingState,
    build_rigid_palm_mask,
    rgbd_tracker_frame_from_d455,
    robust_palm_center_m,
)


SCENARIOS = ("P5_STATIC", "P5_TRANSLATION", "P5_ROTATION", "P5_GESTURE")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def select_palm_roi(rgb: np.ndarray) -> np.ndarray:
    helper = SCRIPT_DIR / "manual_select_roi_once.py"
    executable = "/home/diu/anaconda3/envs/mediapipe_env/bin/python"
    completed = __import__("subprocess").run(
        [executable, str(helper), "--width", str(rgb.shape[1]),
         "--height", str(rgb.shape[0])],
        input=rgb.tobytes(), capture_output=True, timeout=120.0,
    )
    try:
        result = json.loads(completed.stdout.decode().strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError("manual palm ROI selector failed") from exc
    if completed.returncode or not result.get("valid"):
        raise RuntimeError("manual palm ROI selection cancelled")
    return np.asarray(result["bbox"], dtype=np.float64)


def draw_axes(image: np.ndarray, rotation: Any, center: Sequence[int]) -> None:
    if rotation is None: return
    matrix = np.asarray(rotation, dtype=np.float64)
    origin = np.asarray(center, dtype=np.float64)
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for index, color in enumerate(colors):
        direction = matrix[:2, index]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-8:
            end = tuple(np.rint(origin + 45.0 * direction / norm).astype(int))
            cv2.arrowedLine(image, tuple(origin.astype(int)), end, color, 2,
                            tipLength=0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--bbox", nargs=4, type=float,
                        help="manual palm-only ROI xyxy; otherwise mouse selection")
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--countdown-s", type=float, default=4.0)
    parser.add_argument("--gesture-freeze", action="store_true",
                        help="freeze accumulation for the P5_GESTURE recording")
    parser.add_argument("--output-root", default=str(
        ROOT / "datasets/development_usb2/p5_rgbd_relative_orientation"))
    parser.add_argument("--realsense-sdk-site-packages", default=(
        "/home/diu/anaconda3/envs/mediapipe_env/lib/python3.10/site-packages"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sdk = Path(args.realsense_sdk_site_packages)
    if sdk.is_dir() and str(sdk) not in sys.path: sys.path.append(str(sdk))
    config = RGBDRigidTrackerConfig(
        maximum_frame_gap=1,
        maximum_dt_s=0.12,
        maximum_rotation_increment_deg=30.0,
        palm_bbox_erosion_fraction=0.15,
    )
    capture = D455Capture(640, 480, 30, require_superspeed=False)
    tracker = RGBDRelativeOrientationTracker(config, lost_after_s=0.25)
    output_dir = None
    frames_file = None
    video = None
    records = []
    processing_ms = []
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        capture.start()
        frame = capture.wait_for_stable_frames(consecutive=8)
        roi = np.asarray(args.bbox, dtype=np.float64) if args.bbox else select_palm_roi(frame.rgb)
        output_dir = Path(args.output_root).resolve() / (
            args.scenario + "_" + time.strftime("%Y%m%dT%H%M%S")
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "rgb").mkdir(); (output_dir / "aligned_depth").mkdir()
        frames_file = (output_dir / "frames.jsonl").open("x", encoding="utf-8")
        video = cv2.VideoWriter(str(output_dir / "tracking_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
        if not video.isOpened(): raise RuntimeError("overlay video failed to open")
        print("Selected palm ROI:", roi.tolist(), flush=True)
        countdown = time.monotonic()
        while time.monotonic() - countdown < args.countdown_s:
            frame = capture.wait_for_frame()
        tracker.engage_clutch()
        start = time.monotonic()
        index = 0
        while time.monotonic() - start < args.duration_s:
            frame = capture.wait_for_frame()
            tracker_frame = rgbd_tracker_frame_from_d455(frame, roi)
            began = time.perf_counter()
            result = tracker.process(
                tracker_frame,
                externally_frozen=bool(args.gesture_freeze),
                freeze_reason=("GESTURE_TEST_MANUAL_FREEZE" if args.gesture_freeze else "NONE"),
            )
            try:
                center_m = robust_palm_center_m(tracker_frame, config)
            except Exception:
                center_m = None
            elapsed_ms = (time.perf_counter() - began) * 1000.0
            processing_ms.append(elapsed_ms)
            payload: Dict[str, Any] = {
                "index": index,
                "color_frame_number": frame.color_frame_number,
                "depth_frame_number": frame.depth_frame_number,
                "timestamp_s": tracker_frame.timestamp_s,
                "timestamp_domain": tracker_frame.timestamp_domain,
                "device_timestamp_skew_ms": frame.device_timestamp_skew_ms,
                "palm_roi_xyxy": roi,
                "palm_center_m": center_m,
                "processing_ms": elapsed_ms,
                "result": result.as_dict(),
                "rgb_path": f"rgb/{index:06d}.png",
                "aligned_depth_path": f"aligned_depth/{index:06d}.png",
            }
            canvas = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            x1, y1, x2, y2 = np.rint(roi).astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            mask = build_rigid_palm_mask(tracker_frame, config)
            contour = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(canvas, contour, -1, (255, 100, 0), 1)
            pixels = result.pairwise.tracked_pixels_current
            if pixels is not None:
                for pixel in pixels:
                    cv2.circle(canvas, tuple(np.rint(pixel).astype(int)), 2,
                               (0, 255, 0), -1)
            draw_axes(canvas, result.accumulated_rotation,
                      (int((x1+x2)/2), int((y1+y2)/2)))
            cv2.putText(canvas,
                f"{result.state.value} valid={result.pairwise.valid} "
                f"inliers={result.pairwise.ransac_inliers} {elapsed_ms:.1f}ms",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .50, (255,255,255), 1,
                cv2.LINE_AA)
            rgb_path = output_dir / payload["rgb_path"]
            depth_path = output_dir / payload["aligned_depth_path"]
            if not cv2.imwrite(str(rgb_path), canvas): raise RuntimeError("RGB write failed")
            if not cv2.imwrite(str(depth_path), frame.aligned_depth_raw):
                raise RuntimeError("depth write failed")
            video.write(canvas)
            frames_file.write(json.dumps(json_safe(payload), separators=(",", ":")) + "\n")
            records.append(payload); index += 1
        wall = max(time.monotonic() - start, 1e-9)
        valid = sum(r["result"]["pairwise"]["valid"] for r in records)
        states = [r["result"]["state"] for r in records]
        summary = {
            "schema_version": 1,
            "scenario": args.scenario,
            "profile": "D455 RGB8 + aligned Z16 640x480@30",
            "usb_type_descriptor": capture.device_metadata["usb_type_descriptor"],
            "frames": len(records),
            "duration_s": wall,
            "raw_capture_hz": len(records) / wall,
            "kabsch_valid_frames": valid,
            "kabsch_valid_coverage": valid / max(len(records), 1),
            "kabsch_processing_ms": {
                "mean": float(np.mean(processing_ms)),
                "p50": float(np.percentile(processing_ms, 50)),
                "p95": float(np.percentile(processing_ms, 95)),
                "maximum": float(np.max(processing_ms)),
            },
            "frozen_frames": states.count("FROZEN"),
            "lost_frames": states.count("LOST"),
            "reinitialization_count": (
                records[-1]["result"]["reinitialization_count"] if records else 0
            ),
            "cpu_time_s": {
                "user": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                "system": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            },
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "rss_growth_mib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss_start
            ) / 1024.0,
            "orientation_source": "RGBD_KLT_RANSAC_KABSCH_ONLY",
            "hamer_orientation_used": False,
            "hamer_global_orient_used": False,
            "mano_orientation_used": False,
            "gesture_freeze_source": (
                "manual_experiment_label" if args.gesture_freeze else "none"
            ),
            "config": json_safe(config.__dict__),
            "palm_roi_xyxy": roi.tolist(),
            "status": "COMPLETE",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False))
        return 0
    finally:
        if frames_file is not None: frames_file.close()
        if video is not None: video.release()
        capture.stop()


if __name__ == "__main__": raise SystemExit(main())
