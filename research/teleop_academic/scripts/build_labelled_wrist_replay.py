#!/usr/bin/env python3
"""Build a pose-only ROS replay from labelled offline wrist observations.

The output deliberately contains no RGB, depth, MANO vertices, or identity
metadata.  It joins one frozen position observation and one frozen causal SO(3)
observation by source frame index so the same task-labelled wrist motion can be
propagated through the existing Gazebo teleoperation chain.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CSV_FIELDS = [
    "raw_hand_source_stamp",
    "raw_hand_frame",
    "raw_hand_valid",
    "raw_hand_x",
    "raw_hand_y",
    "raw_hand_z",
    "raw_hand_qx",
    "raw_hand_qy",
    "raw_hand_qz",
    "raw_hand_qw",
    "confidence_x",
    "confidence_y",
    "confidence_z",
    "confidence_roll",
    "confidence_pitch",
    "confidence_yaw",
    "gesture",
    "gesture_confidence",
    "invalid_reason",
    "source_index",
    "intent_label",
    "primary_axis",
    "cue_stage_index",
    "cue_sweep_subphase",
    "cue_direction_sign",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_by_index(path):
    records = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if "index" not in record:
                raise ValueError(f"{path}:{line_number} has no index")
            index = int(record["index"])
            if index in records:
                raise ValueError(f"{path} repeats index {index}")
            records[index] = record
    return records


def project_to_so3(value):
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("state_rotation must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def matrix_to_quaternion_xyzw(value):
    rotation = project_to_so3(value)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            quaternion = np.array([
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ])
        elif axis == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            quaternion = np.array([
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ])
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            quaternion = np.array([
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ])
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("state_rotation produced a zero quaternion")
    return quaternion / norm


def finite_vector(record, key, length):
    value = np.asarray(record.get(key), dtype=np.float64)
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain {length} finite values")
    return value


def build_rows(args, positions, orientations):
    selected_indices = sorted(
        index
        for index in set(positions).intersection(orientations)
        if args.start_index <= index <= args.end_index
    )
    if not selected_indices:
        raise ValueError("selected index range has no joined observations")

    rows = []
    rejected = {}
    first_timestamp = None
    previous_quaternion = None
    for index in selected_indices:
        position_record = positions[index]
        orientation_record = orientations[index]
        reasons = []
        if not bool(position_record.get("valid", False)):
            reasons.append("position_invalid")
        if not bool(orientation_record.get("state_valid", False)):
            reasons.append("orientation_state_invalid")
        if bool(orientation_record.get("state_held", False)):
            reasons.append("orientation_state_held")
        if not bool(orientation_record.get("measurement_admitted", False)):
            reasons.append("orientation_measurement_not_admitted")
        if reasons:
            rejected[str(index)] = reasons
            continue

        position = finite_vector(position_record, "position_m", 3)
        quaternion = matrix_to_quaternion_xyzw(
            orientation_record.get("state_rotation")
        )
        if previous_quaternion is not None and np.dot(
            quaternion, previous_quaternion
        ) < 0.0:
            quaternion *= -1.0
        previous_quaternion = quaternion.copy()

        position_timestamp = float(position_record["timestamp_s"])
        orientation_timestamp = float(orientation_record["timestamp_s"])
        if not math.isfinite(position_timestamp) or not math.isfinite(
            orientation_timestamp
        ):
            raise ValueError(f"index {index} has a non-finite timestamp")
        if abs(position_timestamp - orientation_timestamp) > 1.0e-6:
            raise ValueError(
                f"index {index} timestamp mismatch: "
                f"{position_timestamp} vs {orientation_timestamp}"
            )
        if first_timestamp is None:
            first_timestamp = position_timestamp
        replay_timestamp = 1.0 + position_timestamp - first_timestamp

        position_confidence = float(np.clip(
            position_record.get("inlier_fraction", 0.0), 0.0, 1.0
        ))
        rotation_confidence = float(np.clip(
            orientation_record.get("measurement", {}).get("quality", 0.0),
            0.0,
            1.0,
        ))
        rows.append({
            "raw_hand_source_stamp": f"{replay_timestamp:.9f}",
            "raw_hand_frame": "camera_color_optical_frame",
            "raw_hand_valid": "1",
            "raw_hand_x": f"{position[0]:.12f}",
            "raw_hand_y": f"{position[1]:.12f}",
            "raw_hand_z": f"{position[2]:.12f}",
            "raw_hand_qx": f"{quaternion[0]:.12f}",
            "raw_hand_qy": f"{quaternion[1]:.12f}",
            "raw_hand_qz": f"{quaternion[2]:.12f}",
            "raw_hand_qw": f"{quaternion[3]:.12f}",
            "confidence_x": f"{position_confidence:.9f}",
            "confidence_y": f"{position_confidence:.9f}",
            "confidence_z": f"{position_confidence:.9f}",
            "confidence_roll": f"{rotation_confidence:.9f}",
            "confidence_pitch": f"{rotation_confidence:.9f}",
            "confidence_yaw": f"{rotation_confidence:.9f}",
            "gesture": "0",
            "gesture_confidence": "0.0",
            "invalid_reason": "",
            "source_index": str(index),
            "intent_label": args.intent_label,
            "primary_axis": args.primary_axis,
            "cue_stage_index": str(
                orientation_record.get("cue_stage_index", "")
                if orientation_record.get("cue_stage_index") is not None
                else ""
            ),
            "cue_sweep_subphase": str(
                orientation_record.get("cue_sweep_subphase") or ""
            ),
            "cue_direction_sign": str(
                orientation_record.get("cue_direction_sign", "")
                if orientation_record.get("cue_direction_sign") is not None
                else ""
            ),
        })
    if len(rows) < 2:
        raise ValueError("fewer than two valid joined observations remain")
    return rows, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-records", type=Path, required=True)
    parser.add_argument("--orientation-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--intent-label", required=True)
    parser.add_argument("--primary-axis", required=True)
    args = parser.parse_args()
    if args.start_index < 0 or args.end_index < args.start_index:
        raise ValueError("index range must be nonnegative and ordered")
    if not args.position_records.is_file() or not args.orientation_records.is_file():
        raise ValueError("both source JSONL files must exist")

    positions = read_jsonl_by_index(args.position_records)
    orientations = read_jsonl_by_index(args.orientation_records)
    rows, rejected = build_rows(args, positions, orientations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "schema": "handarm_labelled_wrist_pose_replay_v1",
        "evidence_scope": "POSE_ONLY_DERIVATIVE_NO_RGB_DEPTH_OR_IDENTITY",
        "control_authorized": False,
        "task_label_semantics": (
            "The operator was instructed to primarily express the named axis; "
            "this is a categorical task label, not neural-intent ground truth."
        ),
        "intent_label": args.intent_label,
        "primary_axis": args.primary_axis,
        "selected_source_index_range_inclusive": [
            args.start_index,
            args.end_index,
        ],
        "output_record_count": len(rows),
        "output_source_index_range": [
            int(rows[0]["source_index"]),
            int(rows[-1]["source_index"]),
        ],
        "output_duration_s": (
            float(rows[-1]["raw_hand_source_stamp"])
            - float(rows[0]["raw_hand_source_stamp"])
        ),
        "rejected_indices": rejected,
        "sources": {
            "position_records": str(args.position_records),
            "position_records_sha256": file_sha256(args.position_records),
            "orientation_records": str(args.orientation_records),
            "orientation_records_sha256": file_sha256(args.orientation_records),
        },
        "output_csv": str(args.output),
        "output_csv_sha256": file_sha256(args.output),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
