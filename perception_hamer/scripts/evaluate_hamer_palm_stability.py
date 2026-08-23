#!/usr/bin/env python3
"""Evaluate A/B/C HaMeR palm stability using SO(3) geodesic angles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = PACKAGE_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from perception_hamer.src.palm_frame import require_so3
from perception_hamer.src.realtime_hamer_pipeline import so3_geodesic_degrees


EXPERIMENTS = (
    "DEV_HAMER_STATIC",
    "DEV_HAMER_TRANSLATION",
    "DEV_HAMER_OPEN_CLOSE",
)
METHODS = (
    "raw_global_orient",
    "mano_joint_palm_frame",
    "mano_rigid_vertex_palm_frame",
)


def percentile_summary(values: Iterable[float], include_maximum: bool = True) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    result = {
        "count": int(len(array)),
        "p50": None if len(array) == 0 else float(np.percentile(array, 50)),
        "p95": None if len(array) == 0 else float(np.percentile(array, 95)),
    }
    if include_maximum:
        result["maximum"] = None if len(array) == 0 else float(np.max(array))
    return result


def chordal_so3_reference(rotations: Sequence[Any]) -> np.ndarray:
    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) == 0:
        raise ValueError("rotations must be a nonempty Nx3x3 array")
    for value in values:
        require_so3(value, atol=5e-3)
    left, _, right_t = np.linalg.svd(np.mean(values, axis=0))
    reference = left @ right_t
    if np.linalg.det(reference) < 0.0:
        left[:, -1] *= -1.0
        reference = left @ right_t
    return require_so3(reference, atol=1e-6)


def pearson_or_none(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _load_records(session: Path) -> List[dict]:
    source = session / "frames.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
    if not records:
        raise ValueError(f"session contains no records: {session}")
    return records


def _latest_session(root: Path, experiment: str) -> Path:
    candidates = []
    for path in sorted(root.glob(experiment + "_*")):
        summary_path = path / "summary.json"
        if not path.is_dir() or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("roi_seed_hand_presence_validated") is True
            and summary.get("experiment_usable") is True
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"no usable hand-presence-validated {experiment} session below {root}"
        )
    return candidates[-1]


def evaluate_session(session: Path, experiment: str) -> Tuple[dict, List[dict]]:
    records = _load_records(session)
    valid_records = [record for record in records if record.get("valid")]
    # Shape calibration frames contain intentionally changing betas and must
    # not contaminate a palm-stability measurement.  Only geometry recomputed
    # after betas_user is frozen belongs to the A/B/C comparison window.
    analysis_records = [
        record for record in valid_records
        if record.get("betas_calibration", {}).get("frozen") is True
        and record.get("betas_user") is not None
    ]
    method_metrics: Dict[str, Any] = {}
    csv_rows: List[dict] = []
    for method in METHODS:
        rotations = []
        roi_motion = []
        method_records = []
        for record in analysis_records:
            estimate = record.get("palm_frames", {}).get(method, {})
            if not estimate.get("valid") or estimate.get("rotation") is None:
                continue
            rotation = require_so3(estimate["rotation"], atol=5e-3)
            rotations.append(rotation)
            method_records.append(record)
        # Use changes between adjacent *inferred* frames. Capture-thread KLT
        # steps are smaller and do not account for frames dropped by the
        # latest-only scheduler.
        roi_motion = [0.0]
        for previous, current in zip(method_records[:-1], method_records[1:]):
            previous_bbox = np.asarray(previous["roi"]["bbox"], dtype=np.float64)
            current_bbox = np.asarray(current["roi"]["bbox"], dtype=np.float64)
            previous_center = 0.5 * (previous_bbox[:2] + previous_bbox[2:])
            current_center = 0.5 * (current_bbox[:2] + current_bbox[2:])
            previous_scale = math.sqrt(float(np.prod(previous_bbox[2:] - previous_bbox[:2])))
            current_scale = math.sqrt(float(np.prod(current_bbox[2:] - current_bbox[:2])))
            roi_motion.append(math.hypot(
                float(np.linalg.norm(current_center - previous_center)),
                100.0 * abs(math.log(current_scale / previous_scale)),
            ))
        if rotations:
            reference = chordal_so3_reference(rotations)
            reference_change = [so3_geodesic_degrees(reference, value)
                                for value in rotations]
            consecutive_change = [0.0] + [
                so3_geodesic_degrees(rotations[index - 1], rotations[index])
                for index in range(1, len(rotations))
            ]
        else:
            reference_change, consecutive_change = [], []
        method_metrics[method] = {
            "orientation_change_from_chordal_reference_deg": percentile_summary(
                reference_change, include_maximum=True
            ),
            "consecutive_orientation_change_deg": percentile_summary(
                consecutive_change
            ),
            "roi_motion_vs_consecutive_orientation_pearson": pearson_or_none(
                roi_motion, consecutive_change
            ),
            "valid_frames": len(rotations),
            "valid_coverage": len(rotations) / len(records),
            "analysis_coverage_after_betas_freeze": (
                len(rotations) / len(analysis_records) if analysis_records else 0.0
            ),
            "primary_metric": "SO3_geodesic_angle_degrees",
        }
        summary = method_metrics[method]["orientation_change_from_chordal_reference_deg"]
        csv_rows.append({
            "experiment": experiment,
            "method": method,
            "valid_frames": len(rotations),
            "valid_coverage": len(rotations) / len(records),
            "orientation_p50_deg": summary["p50"],
            "orientation_p95_deg": summary["p95"],
            "orientation_max_deg": summary.get("maximum"),
            "roi_orientation_correlation": method_metrics[method][
                "roi_motion_vs_consecutive_orientation_pearson"
            ],
        })
    timestamps = np.asarray(
        [float(record["timestamp"]) for record in valid_records], dtype=np.float64
    )
    inference_ms = [float(record["inference_ms"]) for record in valid_records
                    if record.get("inference_ms") is not None]
    roi_centers, roi_scales = [], []
    for record in records:
        bbox = record.get("roi", {}).get("bbox")
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=np.float64)
            roi_centers.append(0.5 * (bbox[:2] + bbox[2:]))
            roi_scales.append(math.sqrt(float(np.prod(bbox[2:] - bbox[:2]))))
    center_steps = [] if len(roi_centers) < 2 else np.linalg.norm(
        np.diff(np.asarray(roi_centers), axis=0), axis=1
    ).tolist()
    scale_ratios = [] if len(roi_scales) < 2 else (
        np.asarray(roi_scales[1:]) / np.asarray(roi_scales[:-1])
    ).tolist()
    summary_payload = json.loads((session / "summary.json").read_text(encoding="utf-8"))
    session_metrics = {
        "session": str(session),
        "experiment": experiment,
        "total_frames": len(records),
        "valid_frames": len(valid_records),
        "valid_coverage": len(valid_records) / len(records),
        "betas_calibration_frames_excluded": len(valid_records) - len(analysis_records),
        "analysis_frames_after_betas_freeze": len(analysis_records),
        "actual_hamer_hz": (
            None if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]
            else (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        ),
        "inference_ms": {
            "mean": None if not inference_ms else float(np.mean(inference_ms)),
            **percentile_summary(inference_ms),
        },
        "roi_center_step_px": percentile_summary(center_steps),
        "roi_scale_ratio": percentile_summary(scale_ratios),
        "gpu_system_peak_used_mib": summary_payload.get("gpu_system_peak_used_mib"),
        "methods": method_metrics,
    }
    return session_metrics, csv_rows


def evaluate_root(root: Path) -> dict:
    sessions, rows = {}, []
    for experiment in EXPERIMENTS:
        session = _latest_session(root, experiment)
        metrics, method_rows = evaluate_session(session, experiment)
        sessions[experiment] = metrics
        rows.extend(method_rows)
    report = {
        "schema_version": 1,
        "metric_convention": "SO3 geodesic angle acos((trace(R_ref^T R)-1)/2), degrees",
        "sessions": sessions,
        "development_limitation": (
            "当前D455使用USB 2.1，本阶段结果用于算法开发。"
            "最终实时性能、正式数据集和长时间稳定性以后在USB3条件下重新测试。"
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "hamer_palm_stability_metrics.json"
    csv_path = root / "hamer_palm_stability_metrics.csv"
    index_path = root / "development_dataset_index.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = {
        "schema_version": 1,
        "sessions": {
            key: {
                "path": value["session"],
                "frames_jsonl": str(Path(value["session"]) / "frames.jsonl"),
                "overlay_video": str(Path(value["session"]) / "axes_overlay.mp4"),
                "summary": str(Path(value["session"]) / "summary.json"),
            }
            for key, value in sessions.items()
        },
        "metrics_json": str(json_path),
        "metrics_csv": str(csv_path),
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return {"metrics_json": str(json_path), "metrics_csv": str(csv_path),
            "dataset_index": str(index_path), **report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=str(
        REPOSITORY_ROOT / "datasets/development_usb2/hamer_palm_stability"))
    args = parser.parse_args()
    print(json.dumps(evaluate_root(Path(args.root).resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
