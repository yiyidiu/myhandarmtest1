#!/usr/bin/env python3
"""Re-run P5 RGB-D KLT/RANSAC-Kabsch from one recorded session.

The replay accepts only the recorded RGB, aligned Z16, color intrinsics,
device timestamps/frame IDs, and palm ROI.  It does not load HaMeR and cannot
consume any MANO or HaMeR orientation field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_hamer.src.rgbd_rigid_tracker import (
    RGBDRelativeOrientationTracker,
    RGBDRigidTrackerConfig,
    RGBDTrackerFrame,
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not records:
        raise ValueError("recording contains no frames")
    return records


SCENARIO_MAP = {
    "DEV_HAMER_STATIC": "P5_STATIC",
    "DEV_HAMER_TRANSLATION": "P5_TRANSLATION",
    "DEV_HAMER_OPEN_CLOSE": "P5_GESTURE",
}


def legacy_palm_bbox(record: dict, image_shape: tuple[int, int]) -> list[float]:
    """Use only HaMeR's permitted 2-D palm-region projection, never pose."""

    normalized = np.asarray(record["mano_joints_2d_crop_normalized"], dtype=float)
    affine = np.asarray(record["hamer_quality"]["affine_original_to_crop"], dtype=float)
    if normalized.shape != (21, 2) or affine.shape != (2, 3):
        raise ValueError("legacy P3 record has invalid 2-D palm projection")
    crop_pixels = (normalized + 0.5) * 256.0
    points = (cv2.invertAffineTransform(affine) @
              np.column_stack((crop_pixels, np.ones(21))).T).T
    palm = points[[0, 5, 9, 13, 17]]
    low, high = palm.min(axis=0), palm.max(axis=0)
    extent = high - low; low -= 0.35 * extent; high += 0.35 * extent
    height, width = image_shape
    low = np.maximum(low, [0.0, 0.0]); high = np.minimum(high, [width, height])
    if np.any(high - low < 12.0):
        original = np.asarray(record.get("roi", {}).get("bbox"), dtype=float)
        if original.shape != (4,) or np.any(original[2:] <= original[:2]):
            raise ValueError("projected palm region is too small")
        center = (original[:2] + original[2:]) / 2.0
        extent = (original[2:] - original[:2]) * 0.55
        low, high = center - extent / 2.0, center + extent / 2.0
    return np.concatenate((low, high)).tolist()


def hand_pose_change_p75(previous: Any, current: Any) -> float:
    if previous is None or current is None:
        return 0.0
    first = np.asarray(previous, dtype=float); second = np.asarray(current, dtype=float)
    relative = np.swapaxes(first, -1, -2) @ second
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1.0)/2.0, -1, 1)
    return float(np.percentile(np.degrees(np.arccos(cosine)), 75))


def replay_session(session: Path, output: Path, overlay_video: Optional[Path] = None) -> dict:
    summary = json.loads((session / "summary.json").read_text(encoding="utf-8"))
    device = summary.get("device", {})
    intrinsics = device.get("color_intrinsics")
    depth_scale = device.get("depth_scale_m_per_unit")
    if not isinstance(intrinsics, dict) or depth_scale is None:
        raise ValueError("session does not contain color intrinsics/depth scale")
    legacy = "experiment" in summary
    scenario = SCENARIO_MAP.get(summary.get("experiment"), summary.get("scenario"))
    config_payload = dict(summary.get("config", {}))
    if legacy:
        config_payload.update({
            "maximum_frame_gap": 5,
            "maximum_dt_s": 0.18,
            "shi_tomasi_quality": 0.005,
            "minimum_corner_distance_px": 3.0,
            "minimum_fb_tracks": 6,
        })
    config = RGBDRigidTrackerConfig(**config_payload)
    tracker = RGBDRelativeOrientationTracker(config, lost_after_s=0.25)
    tracker.engage_clutch()
    source_records = load_jsonl(session / "frames.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    video = None
    if overlay_video is not None:
        video = cv2.VideoWriter(str(overlay_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                15.0 if legacy else 30.0, (640, 480))
        if not video.isOpened():
            raise RuntimeError("offline overlay video failed to open")
    valid = 0
    processing_ms = []
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.monotonic()
    previous_hand_pose = None
    output_records = []
    try:
      with output.open("x", encoding="utf-8") as handle:
        for record in source_records:
            rgb_path = (session / record["rgb_path"]).resolve()
            depth_path = (session / record["aligned_depth_path"]).resolve()
            if session.resolve() not in rgb_path.parents or session.resolve() not in depth_path.parents:
                raise ValueError("frame path escapes the session")
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth is None:
                raise ValueError(f"failed to decode frame {record['index']}")
            roi_payload = record.get("palm_roi", record.get("roi", {}))
            bbox = (legacy_palm_bbox(record, depth.shape) if legacy else
                    roi_payload.get("bbox", record.get("palm_roi_xyxy")))
            frame = RGBDTrackerFrame(
                rgb=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                aligned_depth_raw=np.ascontiguousarray(depth),
                color_intrinsics=intrinsics,
                depth_scale_m_per_unit=float(depth_scale),
                palm_bbox_xyxy=bbox,
                timestamp_s=float(record.get("timestamp_s", record.get("timestamp"))),
                frame_number=int(record.get("color_frame_number", record.get("frame_number"))),
                timestamp_domain=str(record["timestamp_domain"]),
            )
            began = time.perf_counter()
            score = hand_pose_change_p75(previous_hand_pose, record.get("hand_pose"))
            previous_hand_pose = record.get("hand_pose", previous_hand_pose)
            gesture_freeze = bool(scenario == "P5_GESTURE" and (
                score > 7.5 or record.get("hamer_context", {}).get("gesture_changing")))
            result = tracker.process(
                frame,
                externally_frozen=gesture_freeze,
                freeze_reason=("RECORDED_HAND_POSE_CHANGE" if gesture_freeze else "NONE"),
            )
            elapsed = (time.perf_counter() - began) * 1000.0
            processing_ms.append(elapsed)
            valid += int(result.pairwise.valid)
            payload = {
                "index": int(record["index"]),
                "color_frame_number": frame.frame_number,
                "timestamp_s": frame.timestamp_s,
                "timestamp_domain": frame.timestamp_domain,
                "palm_roi": {"bbox": list(map(float, bbox)), "source": (
                    "hamer_2d_palm_region_only" if legacy else roi_payload.get("source", "recorded"))},
                "processing_ms": elapsed,
                "result": result.as_dict(),
                "rgb_path": str(rgb_path), "aligned_depth_path": str(depth_path),
                "hand_pose_gesture_change_p75_deg": score,
                "orientation_source": "RGBD_KLT_RANSAC_KABSCH_ONLY",
                "hamer_orientation_used": False,
            }
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            output_records.append(payload)
            if video is not None:
                overlay = bgr.copy(); box = np.rint(bbox).astype(int)
                cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), (0,255,255), 2)
                pixels = result.pairwise.tracked_pixels_current
                if pixels is not None:
                    for point in pixels:
                        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 2, (0,255,0), -1)
                cv2.putText(overlay, f"{result.state.value} inliers={result.pairwise.ransac_inliers}",
                            (8,22), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                video.write(overlay)
    finally:
        if video is not None:
            video.release()
    state_values = [record["result"]["state"] for record in output_records]
    duration = source_records[-1].get("timestamp_s", source_records[-1].get("timestamp")) - source_records[0].get("timestamp_s", source_records[0].get("timestamp"))
    wall_elapsed = max(time.monotonic() - wall_start, 1e-9)
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (usage_end.ru_utime + usage_end.ru_stime
                   - usage_start.ru_utime - usage_start.ru_stime)
    replay_summary = {
        "schema_version": 1,
        "scenario": scenario,
        "profile": summary.get("profile"),
        "device": device,
        "usb_type_descriptor": summary.get("usb_type_descriptor"),
        "source_session": str(session.resolve()),
        "output": str(output.resolve()),
        "frames": len(source_records),
        "valid_frames": valid,
        "valid_coverage": valid / len(source_records),
        "kabsch_valid_frames": valid,
        "kabsch_valid_coverage": valid / len(source_records),
        "frozen_frames": state_values.count("FROZEN"),
        "lost_frames": state_values.count("LOST"),
        "reinitialization_count": output_records[-1]["result"]["reinitialization_count"],
        "raw_capture_hz": (len(source_records)-1)/duration,
        "kabsch_processing_hz": (len(source_records)-1)/duration,
        "offline_replay_wall_s": wall_elapsed,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_utilization_percent": 100.0 * cpu_seconds / wall_elapsed,
        "peak_rss_mib": usage_end.ru_maxrss / 1024.0,
        "kabsch_processing_ms": {
            "mean": float(np.mean(processing_ms)),
            "p50": float(np.percentile(processing_ms, 50)),
            "p95": float(np.percentile(processing_ms, 95)),
            "maximum": float(np.max(processing_ms)),
        },
        "orientation_source": "RGBD_KLT_RANSAC_KABSCH_ONLY",
        "config": config.__dict__,
        "offline_input_uses_hamer_2d_region_only": legacy,
        "source_hamer_orientation_fields_read": False,
        "hamer_loaded": False,
        "hamer_orientation_used": False,
        "status": "REPLAY_COMPLETE_NOT_ACCEPTANCE_DECISION",
    }
    summary_path = (output.parent / "summary.json" if output.name == "frames.jsonl"
                    else output.with_suffix(".summary.json"))
    summary_path.write_text(
        json.dumps(replay_summary, indent=2) + "\n", encoding="utf-8"
    )
    return replay_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overlay-video", type=Path)
    args = parser.parse_args()
    session = args.session.resolve()
    output = args.output or (Path.cwd() / "offline_replay_frames.jsonl")
    result = replay_session(session, output.resolve(), None if args.overlay_video is None else args.overlay_video.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
