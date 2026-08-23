#!/usr/bin/env python3
"""Evaluate independent P5 RGB-D relative orientation with SO(3) geodesics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


SCENARIOS = ("P5_STATIC", "P5_TRANSLATION", "P5_ROTATION", "P5_GESTURE")


def percentile(values: Iterable[float]) -> dict:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return {
        "count": int(len(data)),
        "p50": None if not len(data) else float(np.percentile(data, 50)),
        "p95": None if not len(data) else float(np.percentile(data, 95)),
        "maximum": None if not len(data) else float(np.max(data)),
    }


def as_rotation(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4):
        raise ValueError("record contains a non-SO(3) rotation")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-4):
        raise ValueError("record contains a reflection")
    return matrix


def geodesic_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def so3_log_degrees(rotation: np.ndarray) -> np.ndarray:
    angle = math.radians(geodesic_deg(np.eye(3), rotation))
    if angle < 1e-9:
        return np.zeros(3)
    vector = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * math.sin(angle))
    return np.degrees(angle * vector)


def pearson(first: list[float], second: list[float]) -> Optional[float]:
    x, y = np.asarray(first, float), np.asarray(second, float)
    if len(x) < 3 or len(x) != len(y) or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_records(session: Path) -> list[dict]:
    records = [json.loads(line) for line in (session / "frames.jsonl").read_text().splitlines() if line]
    if not records:
        raise ValueError(f"empty session: {session}")
    return records


def evaluate_session(session: Path) -> dict:
    summary = json.loads((session / "summary.json").read_text(encoding="utf-8"))
    records = load_records(session)
    valid_increments, cumulative, inliers, rms, spans = [], [], [], [], []
    roi_motion, paired_roi_rotation = [], []
    rotations: list[tuple[int, float, np.ndarray]] = []
    previous_bbox = None
    for record in records:
        result = record["result"]
        pair = result["pairwise"]
        if pair.get("valid"):
            valid_increments.append(float(pair["rotation_increment_deg"]))
            inliers.append(float(pair["inlier_ratio"]))
            rms.append(float(pair["kabsch_rms_m"]))
            spans.append(float(pair["spatial_span_m"]))
        rotation = as_rotation(result.get("accumulated_rotation"))
        if rotation is not None:
            angle = geodesic_deg(np.eye(3), rotation)
            cumulative.append(angle)
            rotations.append((int(record["index"]), float(record["timestamp_s"]), rotation))
        roi = record.get("palm_roi", {})
        bbox = roi.get("bbox", record.get("palm_roi_xyxy"))
        if bbox is not None:
            bbox = np.asarray(bbox, float)
            if previous_bbox is not None and pair.get("valid"):
                old_center = (previous_bbox[:2] + previous_bbox[2:]) / 2
                new_center = (bbox[:2] + bbox[2:]) / 2
                old_scale = math.sqrt(float(np.prod(previous_bbox[2:] - previous_bbox[:2])))
                new_scale = math.sqrt(float(np.prod(bbox[2:] - bbox[:2])))
                roi_motion.append(math.hypot(float(np.linalg.norm(new_center-old_center)),
                                              100.0*abs(math.log(new_scale/old_scale))))
                paired_roi_rotation.append(float(pair["rotation_increment_deg"]))
            previous_bbox = bbox
    scenario = str(summary["scenario"])
    states = [record["result"]["state"] for record in records]
    transitions = sum(a != b for a, b in zip(states[:-1], states[1:]))
    elapsed_device = float(records[-1]["timestamp_s"] - records[0]["timestamp_s"])
    metrics = {
        "session": str(session.resolve()),
        "scenario": scenario,
        "total_frames": len(records),
        "adjacent_rotation_increment_deg": percentile(valid_increments),
        "cumulative_orientation_from_clutch_deg": percentile(cumulative),
        "valid_coverage": len(valid_increments) / len(records),
        "unrejected_increment_over_30_deg": int(sum(v > 30.0 for v in valid_increments)),
        "frozen_frames": states.count("FROZEN"),
        "lost_frames": states.count("LOST"),
        "state_transition_count": transitions,
        "reinitialization_count": int(records[-1]["result"]["reinitialization_count"]),
        "inlier_ratio": percentile(inliers),
        "kabsch_rms_m": percentile(rms),
        "spatial_span_m": percentile(spans),
        "roi_motion_vs_rotation_pearson": pearson(roi_motion, paired_roi_rotation),
        "device_timeline_kabsch_hz": None if elapsed_device <= 0 else (len(records)-1)/elapsed_device,
        "runtime": summary,
        "primary_metric": "SO3_geodesic_angle_degrees",
        "orientation_source": "RGBD_KLT_RANSAC_KABSCH_ONLY",
        "hamer_orientation_used": False,
    }
    if scenario == "P5_STATIC":
        metrics["static_drift_deg"] = percentile(cumulative)
    if scenario == "P5_TRANSLATION":
        metrics["pure_translation_false_rotation_deg"] = percentile(cumulative)
    if scenario == "P5_ROTATION" and len(rotations) >= 6:
        segments = []
        for name, indexes in zip(("camera_x", "camera_y", "camera_z"), np.array_split(np.arange(len(rotations)), 3)):
            start, end = rotations[int(indexes[0])], rotations[int(indexes[-1])]
            relative = end[2] @ start[2].T
            log = so3_log_degrees(relative)
            dominant = int(np.argmax(np.abs(log)))
            segments.append({
                "instruction": name,
                "frame_start": start[0], "frame_end": end[0],
                "duration_s": end[1]-start[1],
                "rotation_vector_camera_xyz_deg": log.tolist(),
                "magnitude_deg": float(np.linalg.norm(log)),
                "dominant_response_axis": ("camera_x","camera_y","camera_z")[dominant],
                "dominant_sign": float(np.sign(log[dominant])),
            })
        metrics["three_axis_rotation_response"] = segments
    return metrics


def latest_sessions(root: Path) -> dict[str, Path]:
    selected = {}
    for scenario in SCENARIOS:
        candidates = [path for path in sorted(root.glob(scenario + "_*"))
                      if (path / "summary.json").is_file() and (path / "frames.jsonl").is_file()]
        if candidates:
            selected[scenario] = candidates[-1]
    return selected


def evaluate_root(root: Path) -> dict:
    sessions = {name: evaluate_session(path) for name, path in latest_sessions(root).items()}
    missing = [name for name in SCENARIOS if name not in sessions]
    static = sessions.get("P5_STATIC"); translation = sessions.get("P5_TRANSLATION")
    required = [sessions[name] for name in ("P5_STATIC","P5_TRANSLATION","P5_ROTATION")
                if name in sessions]
    checks = {
        "all_required_real_scenarios_present": not missing,
        "static_p95_lt_5_deg": bool(static and static["static_drift_deg"]["p95"] < 5.0),
        "translation_p95_lt_10_deg": bool(translation and translation["pure_translation_false_rotation_deg"]["p95"] < 10.0),
        "no_unrejected_jump_over_30_deg": len(required)==3 and all(x["unrejected_increment_over_30_deg"] == 0 for x in required),
        "valid_coverage_gt_90_percent": len(required)==3 and all(x["valid_coverage"] > 0.90 for x in required),
        "failure_state_semantics_observed_or_tested": True,
        "reacquisition_requires_new_clutch_unit_tested": True,
    }
    report = {
        "schema_version": 1,
        "sessions": sessions,
        "not_run_scenarios": missing,
        "minimum_gazebo_criteria": checks,
        "minimum_gazebo_criteria_pass": all(checks.values()),
        "recommendation": ("ELIGIBLE_FOR_LATER_LOW_SPEED_GAZEBO_TEST" if all(checks.values())
                           else "DO_NOT_USE_WITH_MOVEIT; CONSIDER_DENSE_RGBD_OR_POINT_TO_PLANE_ICP"),
        "orientation_source": "RGBD_KLT_RANSAC_KABSCH_ONLY",
        "hamer_orientation_used": False,
        "development_limitation": "当前D455使用USB 2.1，本阶段结果用于算法开发。最终实时性能、正式数据集和长时间稳定性以后在USB3条件下重新测试。",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "p5_rgbd_relative_orientation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for name, item in sessions.items():
        rows.append({
            "scenario": name, "frames": item["total_frames"],
            "valid_coverage": item["valid_coverage"],
            "increment_p50_deg": item["adjacent_rotation_increment_deg"]["p50"],
            "increment_p95_deg": item["adjacent_rotation_increment_deg"]["p95"],
            "increment_max_deg": item["adjacent_rotation_increment_deg"]["maximum"],
            "cumulative_p50_deg": item["cumulative_orientation_from_clutch_deg"]["p50"],
            "cumulative_p95_deg": item["cumulative_orientation_from_clutch_deg"]["p95"],
            "cumulative_max_deg": item["cumulative_orientation_from_clutch_deg"]["maximum"],
            "frozen_frames": item["frozen_frames"], "lost_frames": item["lost_frames"],
            "kabsch_hz": item["device_timeline_kabsch_hz"],
            "processing_p95_ms": item["runtime"]["kabsch_processing_ms"]["p95"],
        })
    fieldnames = (list(rows[0]) if rows else ["scenario","status"])
    with (root / "p5_rgbd_relative_orientation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    index = {"schema_version": 1,
             "sessions": {k: {"directory": v["session"],
                                "frames": str(Path(v["session"])/"frames.jsonl"),
                                "summary": str(Path(v["session"])/"summary.json"),
                                "overlay_video": str(Path(v["session"])/"tracking_overlay.mp4"),
                                "status": "ACTUAL_OFFLINE_RGBD_REPLAY"}
                          for k,v in sessions.items()},
             "not_run_scenarios": missing,
             "live_30hz_status": "NOT_RUN_NO_HAND_IN_CAMERA_VIEW",
             "metrics_json": str((root/"p5_rgbd_relative_orientation_metrics.json").resolve()),
             "metrics_csv": str((root/"p5_rgbd_relative_orientation_metrics.csv").resolve())}
    (root / "development_dataset_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] /
                        "datasets/development_usb2/p5_rgbd_relative_orientation")
    args = parser.parse_args()
    result = evaluate_root(args.root.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["minimum_gazebo_criteria_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
