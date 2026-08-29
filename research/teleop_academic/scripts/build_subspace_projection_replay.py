#!/usr/bin/env python3
"""Build the frozen causal P/O subspace-projection development replay.

The state estimator consumes only raw pose increments ending in the trailing
window.  Task labels are copied through for post-hoc evaluation but never used
to choose P, O, or undecided state.  This script is a development replay
preprocessor, not evidence of live-camera or independent-input performance.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


STATE_UNDECIDED = "UNDECIDED"
STATE_POSITION = "POSITION"
STATE_ORIENTATION = "ORIENTATION"
EPSILON = 1.0e-12
DIAGNOSTIC_FIELDS = [
    "m2_candidate_state",
    "m2_candidate_translation_energy",
    "m2_candidate_rotation_energy",
    "m2_candidate_translation_rotation_ratio",
    "m2_candidate_rotation_translation_ratio",
    "m2_candidate_window_full",
    "m2_candidate_reference_required",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def finite_pose(row):
    position = np.asarray([
        float(row["raw_hand_x"]),
        float(row["raw_hand_y"]),
        float(row["raw_hand_z"]),
    ], dtype=np.float64)
    quaternion = np.asarray([
        float(row["raw_hand_qx"]),
        float(row["raw_hand_qy"]),
        float(row["raw_hand_qz"]),
        float(row["raw_hand_qw"]),
    ], dtype=np.float64)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("valid input row contains a non-finite pose")
    norm = float(np.linalg.norm(quaternion))
    if norm <= EPSILON:
        raise ValueError("valid input row contains a zero quaternion")
    quaternion /= norm
    return position, Rotation.from_quat(quaternion)


def load_input(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    required = {
        "raw_hand_source_stamp", "raw_hand_valid",
        "raw_hand_x", "raw_hand_y", "raw_hand_z",
        "raw_hand_qx", "raw_hand_qy", "raw_hand_qz", "raw_hand_qw",
        "source_index", "intent_label", "cue_sweep_subphase",
    }
    missing = sorted(required.difference(fieldnames))
    if not rows or missing:
        raise ValueError("input is empty or missing fields: {}".format(missing))
    timestamps = [float(row["raw_hand_source_stamp"]) for row in rows]
    if any(not math.isfinite(value) for value in timestamps):
        raise ValueError("input timestamps must be finite")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("input timestamps must be strictly increasing")
    if not as_bool(rows[0]["raw_hand_valid"]):
        raise ValueError("development replay must begin with a valid C-reference pose")
    return rows, fieldnames, timestamps


def ratio_string(numerator, denominator):
    if denominator <= EPSILON:
        return "inf" if numerator > EPSILON else "undefined"
    return "{:.12f}".format(numerator / denominator)


def choose_state(previous, translation_energy, rotation_energy,
                 position_ratio, orientation_ratio):
    if translation_energy <= EPSILON and rotation_energy <= EPSILON:
        return STATE_UNDECIDED, "BOTH_ENERGIES_ZERO"
    if rotation_energy <= EPSILON or translation_energy / rotation_energy >= position_ratio:
        return STATE_POSITION, "POSITION_RATIO_ENTERED"
    if translation_energy <= EPSILON or rotation_energy / translation_energy >= orientation_ratio:
        return STATE_ORIENTATION, "ORIENTATION_RATIO_ENTERED"
    return previous, "AMBIGUOUS_RETAIN_PREVIOUS"


def mean_rotation(rotations):
    quaternions = [rotation.as_quat() for rotation in rotations]
    reference = quaternions[0]
    aligned = [
        quaternion if float(np.dot(quaternion, reference)) >= 0.0 else -quaternion
        for quaternion in quaternions
    ]
    mean = np.sum(np.asarray(aligned), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= EPSILON:
        raise ValueError("orientation endpoint mean is undefined")
    return Rotation.from_quat(mean / norm)


def transition_summary(rows, positions, rotations, endpoint_window_size=6):
    outbound = [
        index for index, row in enumerate(rows)
        if row.get("cue_sweep_subphase") == "OUTBOUND"
    ]
    if not outbound:
        return None
    baseline = [
        index for index in range(outbound[0])
        if not rows[index].get("cue_sweep_subphase")
    ][-endpoint_window_size:]
    endpoint = outbound[-endpoint_window_size:]
    if len(baseline) < endpoint_window_size or len(endpoint) < endpoint_window_size:
        raise ValueError("labelled transition has too few endpoint samples")
    baseline_position = np.median(
        np.asarray([positions[index] for index in baseline]), axis=0
    )
    endpoint_position = np.median(
        np.asarray([positions[index] for index in endpoint]), axis=0
    )
    baseline_rotation = mean_rotation([rotations[index] for index in baseline])
    endpoint_rotation = mean_rotation([rotations[index] for index in endpoint])
    translation_excursions = [
        float(np.linalg.norm(positions[index] - baseline_position))
        for index in outbound
    ]
    rotation_excursions = [
        math.degrees(float(np.linalg.norm(
            (baseline_rotation.inv() * rotations[index]).as_rotvec()
        )))
        for index in outbound
    ]
    return {
        "baseline_source_indices": [int(rows[index]["source_index"]) for index in baseline],
        "outbound_source_indices": [int(rows[index]["source_index"]) for index in outbound],
        "endpoint_source_indices": [int(rows[index]["source_index"]) for index in endpoint],
        "endpoint_translation_magnitude_m": float(np.linalg.norm(
            endpoint_position - baseline_position
        )),
        "endpoint_rotation_magnitude_deg": math.degrees(float(np.linalg.norm(
            (baseline_rotation.inv() * endpoint_rotation).as_rotvec()
        ))),
        "outbound_max_translation_excursion_m": max(translation_excursions),
        "outbound_max_rotation_excursion_deg": max(rotation_excursions),
    }


def evaluate_labelled_state(rows, timestamps, states):
    outbound = [
        index for index, row in enumerate(rows)
        if row.get("cue_sweep_subphase") == "OUTBOUND"
    ]
    if not outbound:
        return None
    intent_label = rows[outbound[0]].get("intent_label")
    expected = (
        STATE_ORIENTATION if intent_label == "ROTATION"
        else STATE_POSITION if intent_label == "POSITION"
        else None
    )
    if expected is None:
        raise ValueError("intent_label must be POSITION or ROTATION for evaluation")
    first_correct = next(
        (index for index in outbound if states[index] == expected), None
    )
    stable_correct = next((
        index for offset, index in enumerate(outbound)
        if states[index] == expected
        and all(states[later] == expected for later in outbound[offset:])
    ), None)
    return {
        "task_label_used_only_for_post_hoc_evaluation": intent_label,
        "expected_state": expected,
        "outbound_samples": len(outbound),
        "state_at_first_outbound_sample": states[outbound[0]],
        "correct_state_fraction": sum(states[index] == expected for index in outbound) / len(outbound),
        "undecided_fraction": sum(states[index] == STATE_UNDECIDED for index in outbound) / len(outbound),
        "first_correct_state_source_index": (
            int(rows[first_correct]["source_index"]) if first_correct is not None else None
        ),
        "first_correct_state_latency_from_outbound_start_s": (
            timestamps[first_correct] - timestamps[outbound[0]]
            if first_correct is not None else None
        ),
        "stable_correct_state_source_index": (
            int(rows[stable_correct]["source_index"]) if stable_correct is not None else None
        ),
        "stable_lock_latency_from_outbound_start_s": (
            timestamps[stable_correct] - timestamps[outbound[0]]
            if stable_correct is not None else None
        ),
        "stable_lock_definition": (
            "first expected-state sample after which every remaining labelled "
            "OUTBOUND sample stays in the expected state"
        ),
    }


def build(rows, timestamps, candidate):
    window_s = float(candidate["causal_window_s"])
    position_scales = np.asarray(candidate["translation_axis_scale_m"], dtype=np.float64)
    rotation_scales = np.radians(np.asarray(
        candidate["rotation_axis_scale_deg"], dtype=np.float64
    ))
    position_ratio = float(candidate["enter_position_energy_ratio"])
    orientation_ratio = float(candidate["enter_orientation_energy_ratio"])
    if (position_scales.shape != (3,) or rotation_scales.shape != (3,)
            or np.any(position_scales <= 0.0) or np.any(rotation_scales <= 0.0)
            or window_s <= 0.0 or position_ratio <= 1.0 or orientation_ratio <= 1.0):
        raise ValueError("candidate constants are invalid")

    input_positions = []
    input_rotations = []
    for row in rows:
        if as_bool(row["raw_hand_valid"]):
            position, rotation = finite_pose(row)
        else:
            try:
                position, rotation = finite_pose(row)
            except (KeyError, TypeError, ValueError):
                position = input_positions[-1].copy()
                rotation = input_rotations[-1]
        input_positions.append(position)
        input_rotations.append(rotation)

    output_positions = [input_positions[0].copy()]
    output_rotations = [input_rotations[0]]
    states = [STATE_UNDECIDED]
    energies = [(0.0, 0.0)]
    window_full_flags = [False]
    reference_required_flags = [False]
    state_reasons = ["INITIAL_C_REFERENCE"]
    increments = []
    reference_started_at = timestamps[0]
    reference_required = False
    previous_input_position = input_positions[0]
    previous_input_rotation = input_rotations[0]
    state = STATE_UNDECIDED
    transitions = []
    invalid_reset_count = 0

    for index in range(1, len(rows)):
        row = rows[index]
        input_valid = as_bool(row["raw_hand_valid"])
        explicit_reference = as_bool(
            row.get("candidate_reference_confirmed", "false")
        )
        if not input_valid:
            invalid_reset_count += 1
            reference_required = True
            state = STATE_UNDECIDED
            increments = []
            output_positions.append(output_positions[-1].copy())
            output_rotations.append(output_rotations[-1])
            states.append(state)
            energies.append((0.0, 0.0))
            window_full_flags.append(False)
            reference_required_flags.append(True)
            state_reasons.append("INVALID_INPUT_RESET_REQUIRE_NEW_REFERENCE")
            continue

        if reference_required and not explicit_reference:
            output_positions.append(output_positions[-1].copy())
            output_rotations.append(output_rotations[-1])
            states.append(STATE_UNDECIDED)
            energies.append((0.0, 0.0))
            window_full_flags.append(False)
            reference_required_flags.append(True)
            state_reasons.append("HOLD_AWAITING_NEW_REFERENCE")
            continue

        if reference_required and explicit_reference:
            reference_required = False
            reference_started_at = timestamps[index]
            previous_input_position = input_positions[index]
            previous_input_rotation = input_rotations[index]
            increments = []
            state = STATE_UNDECIDED
            output_positions.append(output_positions[-1].copy())
            output_rotations.append(output_rotations[-1])
            states.append(state)
            energies.append((0.0, 0.0))
            window_full_flags.append(False)
            reference_required_flags.append(False)
            state_reasons.append("NEW_REFERENCE_CONFIRMED")
            continue

        translation_increment = input_positions[index] - previous_input_position
        rotation_increment = (
            previous_input_rotation.inv() * input_rotations[index]
        ).as_rotvec()
        increments.append({
            "end_time": timestamps[index],
            "translation_squared": float(np.sum(
                (translation_increment / position_scales) ** 2
            )),
            "rotation_squared": float(np.sum(
                (rotation_increment / rotation_scales) ** 2
            )),
        })
        cutoff = timestamps[index] - window_s
        increments = [
            increment for increment in increments
            if increment["end_time"] > cutoff
        ]
        translation_energy = math.sqrt(sum(
            increment["translation_squared"] for increment in increments
        ))
        rotation_energy = math.sqrt(sum(
            increment["rotation_squared"] for increment in increments
        ))
        window_full = timestamps[index] - reference_started_at >= window_s
        previous_state = state
        if window_full:
            state, reason = choose_state(
                state, translation_energy, rotation_energy,
                position_ratio, orientation_ratio,
            )
        else:
            state = STATE_UNDECIDED
            reason = "WINDOW_NOT_FULL"

        next_position = output_positions[-1].copy()
        next_rotation = output_rotations[-1]
        if state == STATE_POSITION:
            next_position += translation_increment
        elif state == STATE_ORIENTATION:
            next_rotation = next_rotation * Rotation.from_rotvec(rotation_increment)
        output_positions.append(next_position)
        output_rotations.append(next_rotation)
        states.append(state)
        energies.append((translation_energy, rotation_energy))
        window_full_flags.append(window_full)
        reference_required_flags.append(False)
        state_reasons.append(reason)
        if state != previous_state:
            transitions.append({
                "source_index": int(row["source_index"]),
                "timestamp_s": timestamps[index],
                "from": previous_state,
                "to": state,
                "reason": reason,
                "translation_energy": translation_energy,
                "rotation_energy": rotation_energy,
            })
        previous_input_position = input_positions[index]
        previous_input_rotation = input_rotations[index]

    output_rows = []
    previous_output_quaternion = None
    for index, row in enumerate(rows):
        output = dict(row)
        quaternion = output_rotations[index].as_quat()
        if (previous_output_quaternion is not None
                and float(np.dot(quaternion, previous_output_quaternion)) < 0.0):
            quaternion *= -1.0
        previous_output_quaternion = quaternion.copy()
        output.update({
            "raw_hand_x": "{:.12f}".format(output_positions[index][0]),
            "raw_hand_y": "{:.12f}".format(output_positions[index][1]),
            "raw_hand_z": "{:.12f}".format(output_positions[index][2]),
            "raw_hand_qx": "{:.12f}".format(quaternion[0]),
            "raw_hand_qy": "{:.12f}".format(quaternion[1]),
            "raw_hand_qz": "{:.12f}".format(quaternion[2]),
            "raw_hand_qw": "{:.12f}".format(quaternion[3]),
            "m2_candidate_state": states[index],
            "m2_candidate_translation_energy": "{:.12f}".format(energies[index][0]),
            "m2_candidate_rotation_energy": "{:.12f}".format(energies[index][1]),
            "m2_candidate_translation_rotation_ratio": ratio_string(
                energies[index][0], energies[index][1]
            ),
            "m2_candidate_rotation_translation_ratio": ratio_string(
                energies[index][1], energies[index][0]
            ),
            "m2_candidate_window_full": "1" if window_full_flags[index] else "0",
            "m2_candidate_reference_required": (
                "1" if reference_required_flags[index] else "0"
            ),
        })
        if reference_required_flags[index]:
            output["raw_hand_valid"] = "0"
            existing = output.get("invalid_reason", "")
            output["invalid_reason"] = ";".join(filter(None, [
                existing, "candidate_requires_new_reference"
            ]))
        output_rows.append(output)
    return {
        "output_rows": output_rows,
        "input_positions": input_positions,
        "input_rotations": input_rotations,
        "output_positions": output_positions,
        "output_rotations": output_rotations,
        "states": states,
        "energies": energies,
        "window_full_flags": window_full_flags,
        "reference_required_flags": reference_required_flags,
        "state_reasons": state_reasons,
        "transitions": transitions,
        "invalid_reset_count": invalid_reset_count,
    }


def main():
    default_protocol = (
        Path(__file__).resolve().parents[1] / "m3_frozen_candidate_protocol.yaml"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=default_protocol)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    candidate = protocol["candidate"]
    if candidate.get("name") != "CAUSAL_NORMALIZED_SUBSPACE_ENERGY_SCHMITT_PROJECTION":
        raise ValueError("protocol does not select the frozen M2 candidate")
    if candidate.get("future_frames_allowed") is not False:
        raise ValueError("candidate must remain causal")
    if candidate.get("learned_coupling_matrix_allowed") is not False:
        raise ValueError("candidate must not load a learned coupling matrix")

    rows, fieldnames, timestamps = load_input(args.input)
    result = build(rows, timestamps, candidate)
    output_fieldnames = fieldnames + [
        field for field in DIAGNOSTIC_FIELDS if field not in fieldnames
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=output_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(result["output_rows"])

    report = {
        "schema": "handarm_m3_subspace_projection_development_replay_v1",
        "method": candidate["name"],
        "dataset_role": "DEVELOPMENT_ONLY",
        "neural_intent_claimed": False,
        "input": {
            "path": str(args.input),
            "sha256": file_sha256(args.input),
            "records": len(rows),
            "duration_s": timestamps[-1] - timestamps[0],
        },
        "protocol": {
            "path": str(args.protocol),
            "sha256": file_sha256(args.protocol),
            "candidate": candidate,
            "implementation_clarification": protocol[
                "pre_m3_implementation_clarification"
            ],
        },
        "causal_implementation": {
            "future_frames_used": False,
            "task_labels_used_for_state_decision": False,
            "increment_end_time_window": "(t-0.5s,t]",
            "window_not_full_policy": "HOLD_UNDECIDED",
            "invalid_or_edge_policy": (
                "hold pose, mark output invalid, clear state/history, and require "
                "a candidate_reference_confirmed marker before resuming"
            ),
            "first_pose_assumption": (
                "the external confirm_hand_reference service is armed before replay; "
                "the first valid pose is the common C reference"
            ),
        },
        "state_observation": {
            "histogram": dict(sorted(Counter(result["states"]).items())),
            "transitions": result["transitions"],
            "records_before_full_window": sum(
                not value for value in result["window_full_flags"]
            ),
            "invalid_resets": result["invalid_reset_count"],
            "reference_required_records": sum(result["reference_required_flags"]),
        },
        "labelled_state_evaluation": evaluate_labelled_state(
            rows, timestamps, result["states"]
        ),
        "input_labelled_transition": transition_summary(
            rows, result["input_positions"], result["input_rotations"]
        ),
        "output_labelled_transition": transition_summary(
            rows, result["output_positions"], result["output_rotations"]
        ),
        "output": {
            "path": str(args.output),
            "sha256": file_sha256(args.output),
            "records": len(result["output_rows"]),
            "timestamps_source_indices_and_task_labels_preserved": True,
        },
        "claim_limit": (
            "This is a causal transformation of one reused development input. "
            "Gazebo tool0 evidence is required for the frozen mechanism decision; "
            "live edge recovery and independent performance remain untested."
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
