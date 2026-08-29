#!/usr/bin/env python3
"""Build the frozen AMP-IT functional-reconstruction pose replay.

This is not an exact reproduction of Rodrigues et al. (ISMAR 2023): no
official code/data were found, and the printed exponential expression does
not meet its stated piecewise endpoints.  The implementation therefore uses
the endpoint-continuous exponential curve frozen in m2_frozen_protocol.yaml.
It remains causal, uses a 500 ms trailing window, and scales translation and
rotation independently along the current output pose's local axes.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import yaml


EPSILON = 1.0e-12


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_quaternion_xyzw(value):
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= EPSILON:
        raise ValueError("quaternion norm is zero")
    return quaternion / norm


def quaternion_to_matrix_xyzw(value):
    x, y, z, w = normalize_quaternion_xyzw(value)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def project_to_so3(value):
    u, _, vt = np.linalg.svd(np.asarray(value, dtype=np.float64))
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
    return normalize_quaternion_xyzw(quaternion)


def so3_log(rotation):
    rotation = project_to_so3(rotation)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1.0e-8:
        return 0.5 * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
    if math.pi - angle <= 1.0e-6:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    return angle / (2.0 * math.sin(angle)) * np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(vector):
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    if angle <= 1.0e-8:
        return project_to_so3(np.eye(3) + cross + 0.5 * cross @ cross)
    return (
        np.eye(3)
        + math.sin(angle) / angle * cross
        + (1.0 - math.cos(angle)) / (angle * angle) * cross @ cross
    )


def endpoint_anchored_gain(speed, minimum_speed, full_speed):
    speed = abs(float(speed))
    if speed <= minimum_speed:
        return 0.0
    if speed >= full_speed:
        return 1.0
    fraction = (speed - minimum_speed) / (full_speed - minimum_speed)
    return (4.0 ** fraction - 1.0) / 3.0


def find_window_start(timestamps, current_index, window_s):
    target = timestamps[current_index] - window_s
    candidate = None
    for index in range(current_index - 1, -1, -1):
        if timestamps[index] <= target:
            candidate = index
            break
    return candidate


def distribution(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "maximum": ordered[-1],
        "zero_fraction": sum(abs(value) <= EPSILON for value in ordered) / len(ordered),
        "one_fraction": sum(abs(value - 1.0) <= EPSILON for value in ordered) / len(ordered),
    }


def pose_path_summary(positions, rotations):
    translation_path = sum(
        float(np.linalg.norm(current - previous))
        for previous, current in zip(positions, positions[1:])
    )
    rotation_path = sum(
        float(np.linalg.norm(so3_log(previous.T @ current)))
        for previous, current in zip(rotations, rotations[1:])
    )
    return {
        "translation_path_m": translation_path,
        "rotation_path_deg": math.degrees(rotation_path),
        "first_to_last_translation_m": float(np.linalg.norm(positions[-1] - positions[0])),
        "first_to_last_rotation_deg": math.degrees(float(
            np.linalg.norm(so3_log(rotations[0].T @ rotations[-1]))
        )),
    }


def rotation_mean(rotations, indices):
    quaternions = [matrix_to_quaternion_xyzw(rotations[index]) for index in indices]
    reference = quaternions[0]
    aligned = [
        quaternion if np.dot(quaternion, reference) >= 0.0 else -quaternion
        for quaternion in quaternions
    ]
    return quaternion_to_matrix_xyzw(normalize_quaternion_xyzw(
        np.sum(np.asarray(aligned), axis=0)
    ))


def transition_summary(rows, positions, rotations, window_size=6):
    outbound = [
        index for index, row in enumerate(rows)
        if row.get("cue_sweep_subphase") == "OUTBOUND"
    ]
    if not outbound:
        return None
    before = [
        index for index in range(outbound[0])
        if not rows[index].get("cue_sweep_subphase")
    ][-window_size:]
    outbound_endpoint = outbound[-window_size:]
    if len(before) < window_size or len(outbound_endpoint) < window_size:
        return None
    baseline_position = np.median(np.asarray([positions[index] for index in before]), axis=0)
    outbound_position = np.median(
        np.asarray([positions[index] for index in outbound_endpoint]), axis=0
    )
    baseline_rotation = rotation_mean(rotations, before)
    outbound_rotation = rotation_mean(rotations, outbound_endpoint)
    translation_excursions = [
        float(np.linalg.norm(positions[index] - baseline_position)) for index in outbound
    ]
    rotation_excursions = [
        math.degrees(float(np.linalg.norm(so3_log(baseline_rotation.T @ rotations[index]))))
        for index in outbound
    ]
    return {
        "endpoint_window_size": window_size,
        "baseline_source_indices": [int(rows[index]["source_index"]) for index in before],
        "outbound_endpoint_source_indices": [
            int(rows[index]["source_index"]) for index in outbound_endpoint
        ],
        "endpoint_translation_magnitude_m": float(
            np.linalg.norm(outbound_position - baseline_position)
        ),
        "endpoint_rotation_magnitude_deg": math.degrees(float(
            np.linalg.norm(so3_log(baseline_rotation.T @ outbound_rotation))
        )),
        "outbound_max_translation_excursion_m": max(translation_excursions),
        "outbound_max_rotation_excursion_deg": max(rotation_excursions),
    }


def load_input(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows or not fieldnames:
        raise ValueError("input CSV is empty")
    required = {
        "raw_hand_source_stamp", "raw_hand_valid",
        "raw_hand_x", "raw_hand_y", "raw_hand_z",
        "raw_hand_qx", "raw_hand_qy", "raw_hand_qz", "raw_hand_qw",
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError("input CSV is missing fields: {}".format(missing))
    timestamps = np.asarray([
        float(row["raw_hand_source_stamp"]) for row in rows
    ], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("input timestamps must be finite and strictly increasing")
    if any(str(row["raw_hand_valid"]).strip().lower() not in ("1", "true", "yes")
           for row in rows):
        raise ValueError("this frozen development replay requires every input pose valid")
    positions = [np.asarray([
        float(row["raw_hand_x"]), float(row["raw_hand_y"]),
        float(row["raw_hand_z"]),
    ], dtype=np.float64) for row in rows]
    rotations = [quaternion_to_matrix_xyzw([
        float(row["raw_hand_qx"]), float(row["raw_hand_qy"]),
        float(row["raw_hand_qz"]), float(row["raw_hand_qw"]),
    ]) for row in rows]
    return rows, fieldnames, timestamps, positions, rotations


def build_replay(rows, timestamps, input_positions, input_rotations, baseline):
    window_s = float(baseline["causal_window_s"])
    translation_minimum = float(baseline["translation_minimum_speed_m_s"])
    translation_full = float(baseline["translation_full_speed_m_s"])
    rotation_minimum = float(baseline["rotation_minimum_speed_deg_s"])
    rotation_full = float(baseline["rotation_full_speed_deg_s"])
    if not (window_s > 0.0 and 0.0 <= translation_minimum < translation_full
            and 0.0 <= rotation_minimum < rotation_full):
        raise ValueError("published baseline thresholds are invalid")

    output_positions = [input_positions[0].copy()]
    output_rotations = [input_rotations[0].copy()]
    translation_gains = [[] for _ in range(3)]
    rotation_gains = [[] for _ in range(3)]
    window_start_indices = [None]
    for index in range(1, len(rows)):
        window_start = find_window_start(timestamps, index, window_s)
        output_rotation = output_rotations[-1]
        if window_start is None:
            translation_gain = np.zeros(3)
            rotation_gain = np.zeros(3)
        else:
            duration = timestamps[index] - timestamps[window_start]
            window_translation_world = (
                input_positions[index] - input_positions[window_start]
            )
            window_translation_local = output_rotation.T @ window_translation_world
            translation_speed = np.abs(window_translation_local) / duration

            window_rotation_world = (
                input_rotations[index] @ input_rotations[window_start].T
            )
            window_rotation_local = output_rotation.T @ so3_log(window_rotation_world)
            rotation_speed = np.degrees(np.abs(window_rotation_local)) / duration
            translation_gain = np.asarray([
                endpoint_anchored_gain(value, translation_minimum, translation_full)
                for value in translation_speed
            ])
            rotation_gain = np.asarray([
                endpoint_anchored_gain(value, rotation_minimum, rotation_full)
                for value in rotation_speed
            ])

        incremental_translation_world = input_positions[index] - input_positions[index - 1]
        incremental_translation_local = output_rotation.T @ incremental_translation_world
        next_position = (
            output_positions[-1]
            + output_rotation @ (translation_gain * incremental_translation_local)
        )

        incremental_rotation_world = (
            input_rotations[index] @ input_rotations[index - 1].T
        )
        incremental_rotation_local = output_rotation.T @ so3_log(
            incremental_rotation_world
        )
        next_rotation = output_rotation @ so3_exp(
            rotation_gain * incremental_rotation_local
        )
        output_positions.append(next_position)
        output_rotations.append(next_rotation)
        window_start_indices.append(window_start)
        for axis in range(3):
            translation_gains[axis].append(float(translation_gain[axis]))
            rotation_gains[axis].append(float(rotation_gain[axis]))

    output_rows = []
    previous_quaternion = None
    for row, position, rotation in zip(rows, output_positions, output_rotations):
        output = dict(row)
        quaternion = matrix_to_quaternion_xyzw(rotation)
        if previous_quaternion is not None and np.dot(quaternion, previous_quaternion) < 0.0:
            quaternion *= -1.0
        previous_quaternion = quaternion.copy()
        output.update({
            "raw_hand_x": "{:.12f}".format(position[0]),
            "raw_hand_y": "{:.12f}".format(position[1]),
            "raw_hand_z": "{:.12f}".format(position[2]),
            "raw_hand_qx": "{:.12f}".format(quaternion[0]),
            "raw_hand_qy": "{:.12f}".format(quaternion[1]),
            "raw_hand_qz": "{:.12f}".format(quaternion[2]),
            "raw_hand_qw": "{:.12f}".format(quaternion[3]),
        })
        output_rows.append(output)
    return (
        output_rows, output_positions, output_rotations,
        translation_gains, rotation_gains, window_start_indices,
    )


def main():
    default_protocol = Path(__file__).resolve().parents[1] / "m2_frozen_protocol.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=default_protocol)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    baseline = protocol["published_baseline"]
    if baseline.get("name") != "AMP_IT_ENDPOINT_ANCHORED_FUNCTIONAL_RECONSTRUCTION":
        raise ValueError("protocol does not select the expected AMP-IT baseline")
    if baseline.get("exact_reproduction_claimed") is not False:
        raise ValueError("this implementation must not claim exact reproduction")

    rows, fieldnames, timestamps, input_positions, input_rotations = load_input(args.input)
    (
        output_rows, output_positions, output_rotations,
        translation_gains, rotation_gains, window_start_indices,
    ) = build_replay(rows, timestamps, input_positions, input_rotations, baseline)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "schema": "handarm_m2_amp_it_functional_reconstruction_v1",
        "method": baseline["name"],
        "exact_reproduction_claimed": False,
        "dataset_role": "DEVELOPMENT_ONLY",
        "neural_intent_claimed": False,
        "input": {
            "path": str(args.input),
            "sha256": file_sha256(args.input),
            "records": len(rows),
            "duration_s": float(timestamps[-1] - timestamps[0]),
        },
        "protocol": {
            "path": str(args.protocol),
            "sha256": file_sha256(args.protocol),
            "published_baseline": baseline,
        },
        "implementation_choices_required_by_missing_reference_code": {
            "window_sampling": (
                "latest recorded sample at or before t-0.5s; output is held until "
                "a complete 0.5s history exists"
            ),
            "local_axes": "current reconstructed output-pose axes",
            "rotation_components": (
                "SO(3) logarithm components expressed in current output-pose axes"
            ),
            "integration": "scaled local pose increments integrated causally",
            "formula_difference": (
                "uses the frozen endpoint-continuous (4^xi-1)/3 curve because "
                "the printed paper expression is endpoint-inconsistent"
            ),
        },
        "causal_history": {
            "records_without_complete_window": sum(
                index is None for index in window_start_indices
            ),
            "future_frames_used": False,
        },
        "translation_gain_by_local_axis": [
            distribution(values) for values in translation_gains
        ],
        "rotation_gain_by_local_axis": [
            distribution(values) for values in rotation_gains
        ],
        "input_pose_motion": pose_path_summary(input_positions, input_rotations),
        "output_pose_motion": pose_path_summary(output_positions, output_rotations),
        "input_labelled_transition": transition_summary(
            rows, input_positions, input_rotations
        ),
        "output_labelled_transition": transition_summary(
            rows, output_positions, output_rotations
        ),
        "output": {
            "path": str(args.output),
            "sha256": file_sha256(args.output),
            "records": len(output_rows),
            "timestamps_source_indices_and_labels_preserved": True,
        },
        "claim_limit": (
            "This file reports only a pose-input functional reconstruction. "
            "A Gazebo run is required before making any robot end-effector claim."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
