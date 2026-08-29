#!/usr/bin/env python3
"""Summarize the observable input-to-object chain in one milestone-1 ROS bag."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median

import rosbag


ARM_JOINTS = [f"joint_{index}" for index in range(1, 7)]
HAND_ACTIVE_JOINTS = ["f1j1", "f1j2", "f2j1", "f3j2"]


def axis_span(samples):
    if not samples:
        return None
    dimensions = len(samples[0])
    return [
        max(sample[index] for sample in samples)
        - min(sample[index] for sample in samples)
        for index in range(dimensions)
    ]


def displacement(first, last):
    if first is None or last is None:
        return None
    return math.sqrt(sum((b - a) ** 2 for a, b in zip(first, last)))


def normalize_quaternion_xyzw(quaternion):
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1.0e-12:
        return None
    return [value / norm for value in quaternion]


def quaternion_mean_xyzw(quaternions):
    if not quaternions:
        return None
    reference = normalize_quaternion_xyzw(quaternions[0])
    if reference is None:
        return None
    total = [0.0, 0.0, 0.0, 0.0]
    for quaternion in quaternions:
        normalized = normalize_quaternion_xyzw(quaternion)
        if normalized is None:
            continue
        if sum(a * b for a, b in zip(reference, normalized)) < 0.0:
            normalized = [-value for value in normalized]
        total = [a + b for a, b in zip(total, normalized)]
    return normalize_quaternion_xyzw(total)


def quaternion_multiply_xyzw(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def quaternion_delta_rotation_vector_deg(first, last):
    first = normalize_quaternion_xyzw(first)
    last = normalize_quaternion_xyzw(last)
    if first is None or last is None:
        return None
    delta = quaternion_multiply_xyzw(
        last, [-first[0], -first[1], -first[2], first[3]]
    )
    delta = normalize_quaternion_xyzw(delta)
    if delta[3] < 0.0:
        delta = [-value for value in delta]
    vector_norm = math.sqrt(sum(value * value for value in delta[:3]))
    if vector_norm <= 1.0e-12:
        return [0.0, 0.0, 0.0]
    angle_deg = math.degrees(2.0 * math.atan2(vector_norm, delta[3]))
    return [value / vector_norm * angle_deg for value in delta[:3]]


def quaternion_angle_deg(first, last):
    rotation_vector = quaternion_delta_rotation_vector_deg(first, last)
    if rotation_vector is None:
        return None
    return math.sqrt(sum(value * value for value in rotation_vector))


def orientation_excursion_deg(quaternions):
    if not quaternions:
        return None
    return max(quaternion_angle_deg(quaternions[0], sample) for sample in quaternions)


def component_median(samples):
    if not samples:
        return None
    return [median(sample[index] for sample in samples) for index in range(len(samples[0]))]


def pose_transition_summary(
        positions, quaternions, baseline_indices, outbound_indices,
        outbound_excursion_indices=None):
    baseline_positions = [positions[index] for index in baseline_indices if index in positions]
    outbound_positions = [positions[index] for index in outbound_indices if index in positions]
    baseline_quaternions = [
        quaternions[index] for index in baseline_indices if index in quaternions
    ]
    outbound_quaternions = [
        quaternions[index] for index in outbound_indices if index in quaternions
    ]
    baseline_position = component_median(baseline_positions)
    outbound_position = component_median(outbound_positions)
    baseline_quaternion = quaternion_mean_xyzw(baseline_quaternions)
    outbound_quaternion = quaternion_mean_xyzw(outbound_quaternions)
    delta_position = (
        [last - first for first, last in zip(baseline_position, outbound_position)]
        if baseline_position is not None and outbound_position is not None
        else None
    )
    rotation_vector = (
        quaternion_delta_rotation_vector_deg(
            baseline_quaternion, outbound_quaternion
        )
        if baseline_quaternion is not None and outbound_quaternion is not None
        else None
    )
    excursion_indices = outbound_excursion_indices or outbound_indices
    outbound_translation_excursions = [
        (
            index,
            displacement(baseline_position, positions[index]),
        )
        for index in excursion_indices
        if baseline_position is not None and index in positions
    ]
    outbound_rotation_excursions = [
        (
            index,
            quaternion_angle_deg(baseline_quaternion, quaternions[index]),
        )
        for index in excursion_indices
        if baseline_quaternion is not None and index in quaternions
    ]
    maximum_translation = (
        max(outbound_translation_excursions, key=lambda item: item[1])
        if outbound_translation_excursions else None
    )
    maximum_rotation = (
        max(outbound_rotation_excursions, key=lambda item: item[1])
        if outbound_rotation_excursions else None
    )
    return {
        "baseline_position_samples": len(baseline_positions),
        "outbound_position_samples": len(outbound_positions),
        "baseline_orientation_samples": len(baseline_quaternions),
        "outbound_orientation_samples": len(outbound_quaternions),
        "baseline_position_m": baseline_position,
        "outbound_position_m": outbound_position,
        "delta_position_m": delta_position,
        "translation_magnitude_m": (
            math.sqrt(sum(value * value for value in delta_position))
            if delta_position is not None else None
        ),
        "baseline_quaternion_xyzw": baseline_quaternion,
        "outbound_quaternion_xyzw": outbound_quaternion,
        "delta_rotation_vector_deg": rotation_vector,
        "rotation_magnitude_deg": (
            math.sqrt(sum(value * value for value in rotation_vector))
            if rotation_vector is not None else None
        ),
        "outbound_max_translation_excursion_m": (
            maximum_translation[1] if maximum_translation is not None else None
        ),
        "outbound_max_translation_source_index": (
            maximum_translation[0] if maximum_translation is not None else None
        ),
        "outbound_max_rotation_excursion_deg": (
            maximum_rotation[1] if maximum_rotation is not None else None
        ),
        "outbound_max_rotation_source_index": (
            maximum_rotation[0] if maximum_rotation is not None else None
        ),
    }


def load_labelled_input(path, window_size):
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"labelled input is empty: {path}")
    for row in rows:
        row["source_index"] = int(row["source_index"])
        row["raw_hand_source_stamp"] = float(row["raw_hand_source_stamp"])
    outbound_offsets = [
        offset for offset, row in enumerate(rows)
        if row.get("cue_sweep_subphase") == "OUTBOUND"
    ]
    if not outbound_offsets:
        raise ValueError("labelled input has no OUTBOUND samples")
    first_outbound = outbound_offsets[0]
    baseline_rows = [
        row for row in rows[:first_outbound]
        if not row.get("cue_sweep_subphase")
    ][-window_size:]
    outbound_rows = [rows[offset] for offset in outbound_offsets][-window_size:]
    if len(baseline_rows) < window_size or len(outbound_rows) < window_size:
        raise ValueError(
            f"need {window_size} baseline and outbound rows for endpoint medians"
        )
    return {
        "rows": rows,
        "index_by_stamp": {
            round(row["raw_hand_source_stamp"], 6): row["source_index"]
            for row in rows
        },
        "baseline_indices": [row["source_index"] for row in baseline_rows],
        "outbound_indices": [row["source_index"] for row in outbound_rows],
        "outbound_excursion_indices": [
            rows[offset]["source_index"] for offset in outbound_offsets
        ],
        "intent_label": rows[0].get("intent_label"),
        "primary_axis": rows[0].get("primary_axis"),
        "cue_stage_index": outbound_rows[0].get("cue_stage_index"),
    }


def joint_summary(samples, names):
    result = {}
    for name in names:
        values = samples.get(name, [])
        if not values:
            result[name] = None
            continue
        result[name] = {
            "first_rad": values[0],
            "last_rad": values[-1],
            "last_minus_first_rad": values[-1] - values[0],
            "observed_span_rad": max(values) - min(values),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--object-model", default="target_object")
    parser.add_argument("--labelled-input-csv", type=Path)
    parser.add_argument("--endpoint-window-size", type=int, default=6)
    args = parser.parse_args()
    if args.endpoint_window_size <= 0:
        raise ValueError("endpoint window size must be positive")
    labelled_input = (
        load_labelled_input(args.labelled_input_csv, args.endpoint_window_size)
        if args.labelled_input_csv else None
    )

    input_positions = []
    input_quaternions = []
    input_source_times = []
    input_valid = 0
    gesture_counts = Counter()
    trend_reasons = Counter()
    trend_valid = 0
    relative_positions = []
    relative_quaternions = []
    target_positions = []
    target_quaternions = []
    actual_tool_positions = []
    actual_tool_quaternions = []
    input_positions_by_index = {}
    input_quaternions_by_index = {}
    relative_positions_by_index = {}
    relative_quaternions_by_index = {}
    target_positions_by_index = {}
    target_quaternions_by_index = {}
    actual_tool_positions_by_index = {}
    actual_tool_quaternions_by_index = {}
    active_source_index = None
    command_samples = []
    joint_samples = defaultdict(list)
    object_positions = []
    contact_pairs = Counter()
    hand_action_messages = 0
    hand_trajectory_commands = 0
    servo_status_counts = Counter()
    start_time = None
    end_time = None

    with rosbag.Bag(str(args.bag), "r") as bag:
        for topic, message, timestamp in bag.read_messages():
            seconds = timestamp.to_sec()
            start_time = seconds if start_time is None else min(start_time, seconds)
            end_time = seconds if end_time is None else max(end_time, seconds)

            if topic == "/shared_teleop/hamer_pose":
                input_valid += int(bool(message.valid))
                input_source_times.append(float(message.source_timestamp))
                input_position = [
                    message.wrist_pose.position.x,
                    message.wrist_pose.position.y,
                    message.wrist_pose.position.z,
                ]
                input_quaternion = [
                    message.wrist_pose.orientation.x,
                    message.wrist_pose.orientation.y,
                    message.wrist_pose.orientation.z,
                    message.wrist_pose.orientation.w,
                ]
                input_positions.append(input_position)
                input_quaternions.append(input_quaternion)
                if labelled_input is not None:
                    active_source_index = labelled_input["index_by_stamp"].get(
                        round(float(message.source_timestamp), 6)
                    )
                    if active_source_index is None:
                        raise ValueError(
                            "bag input timestamp is absent from labelled input: "
                            f"{message.source_timestamp:.9f}"
                        )
                    input_positions_by_index[active_source_index] = input_position
                    input_quaternions_by_index[active_source_index] = input_quaternion
                gesture_counts[str(int(message.gesture))] += 1
            elif topic == "/shared_teleop/trend_diagnostics":
                record = json.loads(message.data)
                trend_reasons[str(record.get("reason", ""))] += 1
                trend_valid += int(bool(record.get("valid", False)))
                if record.get("relative_position") is not None:
                    relative_positions.append(record["relative_position"])
                if record.get("relative_quaternion_xyzw") is not None:
                    relative_quaternions.append(record["relative_quaternion_xyzw"])
                if record.get("target_position") is not None:
                    target_positions.append(record["target_position"])
                if record.get("target_quaternion_xyzw") is not None:
                    target_quaternions.append(record["target_quaternion_xyzw"])
                if record.get("current_position") is not None:
                    actual_tool_positions.append(record["current_position"])
                if record.get("current_quaternion_xyzw") is not None:
                    actual_tool_quaternions.append(record["current_quaternion_xyzw"])
                if (
                    active_source_index is not None
                    and bool(record.get("valid", False))
                ):
                    if record.get("relative_position") is not None:
                        relative_positions_by_index[active_source_index] = record[
                            "relative_position"
                        ]
                    if record.get("relative_quaternion_xyzw") is not None:
                        relative_quaternions_by_index[active_source_index] = record[
                            "relative_quaternion_xyzw"
                        ]
                    if record.get("target_position") is not None:
                        target_positions_by_index[active_source_index] = record[
                            "target_position"
                        ]
                    if record.get("target_quaternion_xyzw") is not None:
                        target_quaternions_by_index[active_source_index] = record[
                            "target_quaternion_xyzw"
                        ]
                    if record.get("current_position") is not None:
                        actual_tool_positions_by_index[active_source_index] = record[
                            "current_position"
                        ]
                    if record.get("current_quaternion_xyzw") is not None:
                        actual_tool_quaternions_by_index[active_source_index] = record[
                            "current_quaternion_xyzw"
                        ]
            elif topic == "/shared_teleop/safe_twist":
                command_samples.append([
                    message.twist.linear.x,
                    message.twist.linear.y,
                    message.twist.linear.z,
                    message.twist.angular.x,
                    message.twist.angular.y,
                    message.twist.angular.z,
                ])
            elif topic == "/joint_states":
                for name, position in zip(message.name, message.position):
                    joint_samples[name].append(float(position))
            elif topic == "/gazebo/model_states" and args.object_model in message.name:
                pose = message.pose[message.name.index(args.object_model)]
                object_positions.append([
                    pose.position.x, pose.position.y, pose.position.z
                ])
            elif topic == "/handarm_sim_demo/target_contacts":
                for state in message.states:
                    pair = tuple(sorted((state.collision1_name, state.collision2_name)))
                    contact_pairs[pair] += 1
            elif topic == "/shared_teleop/hand_action":
                hand_action_messages += 1
            elif topic in (
                "/controller_gazebo_hand/command",
                "/controller_gazebo_hand/follow_joint_trajectory/goal",
            ):
                hand_trajectory_commands += 1
            elif topic == "/servo_server/status":
                servo_status_counts[str(int(message.data))] += 1

    input_duration = (
        input_source_times[-1] - input_source_times[0]
        if len(input_source_times) > 1 else None
    )
    input_rate = (
        (len(input_source_times) - 1) / input_duration
        if input_duration is not None and input_duration > 0 else None
    )
    nonzero_commands = sum(
        math.sqrt(sum(value * value for value in sample)) > 1.0e-8
        for sample in command_samples
    )
    robot_object_contacts = sum(
        count for pair, count in contact_pairs.items()
        if any("robot::" in item for item in pair)
        and any(f"{args.object_model}::" in item for item in pair)
    )
    object_ground_contacts = sum(
        count for pair, count in contact_pairs.items()
        if any("ground_plane::" in item for item in pair)
        and any(f"{args.object_model}::" in item for item in pair)
    )

    labelled_transition = None
    if labelled_input is not None:
        baseline_indices = labelled_input["baseline_indices"]
        outbound_indices = labelled_input["outbound_indices"]
        outbound_excursion_indices = labelled_input["outbound_excursion_indices"]
        labelled_transition = {
            "intent_label": labelled_input["intent_label"],
            "primary_axis": labelled_input["primary_axis"],
            "cue_stage_index": labelled_input["cue_stage_index"],
            "endpoint_definition": (
                "coordinate-wise position median and sign-aligned quaternion "
                "mean over the final labelled samples in the pre-cue baseline "
                "and OUTBOUND phase"
            ),
            "outbound_excursion_definition": (
                "maximum discrete pose-sample deviation from the pre-cue "
                "baseline pose over every labelled OUTBOUND source index"
            ),
            "endpoint_window_size": args.endpoint_window_size,
            "baseline_source_indices": baseline_indices,
            "outbound_source_indices": outbound_indices,
            "outbound_excursion_source_indices": outbound_excursion_indices,
            "input_observed": pose_transition_summary(
                input_positions_by_index,
                input_quaternions_by_index,
                baseline_indices,
                outbound_indices,
                outbound_excursion_indices,
            ),
            "mapping_relative": pose_transition_summary(
                relative_positions_by_index,
                relative_quaternions_by_index,
                baseline_indices,
                outbound_indices,
                outbound_excursion_indices,
            ),
            "mapping_target": pose_transition_summary(
                target_positions_by_index,
                target_quaternions_by_index,
                baseline_indices,
                outbound_indices,
                outbound_excursion_indices,
            ),
            "robot_actual": pose_transition_summary(
                actual_tool_positions_by_index,
                actual_tool_quaternions_by_index,
                baseline_indices,
                outbound_indices,
                outbound_excursion_indices,
            ),
        }

    result = {
        "schema_version": 4,
        "bag_duration_s": (
            end_time - start_time if start_time is not None and end_time is not None else None
        ),
        "input": {
            "messages": len(input_source_times),
            "valid_messages": input_valid,
            "source_duration_s": input_duration,
            "mean_rate_hz": input_rate,
            "position_span_m": axis_span(input_positions),
            "orientation_excursion_from_first_deg": orientation_excursion_deg(
                input_quaternions
            ),
            "gesture_histogram": dict(sorted(gesture_counts.items())),
        },
        "mapping": {
            "diagnostic_messages": sum(trend_reasons.values()),
            "valid_messages": trend_valid,
            "reason_histogram": dict(sorted(trend_reasons.items())),
            "relative_position_span_m": axis_span(relative_positions),
            "relative_orientation_excursion_from_first_deg": orientation_excursion_deg(
                relative_quaternions
            ),
            "target_position_span_m": axis_span(target_positions),
            "target_orientation_excursion_from_first_deg": orientation_excursion_deg(
                target_quaternions
            ),
        },
        "controller_command": {
            "samples": len(command_samples),
            "nonzero_samples": nonzero_commands,
            "component_span": axis_span(command_samples),
        },
        "servo_status_observation": {
            "histogram": dict(sorted(servo_status_counts.items())),
            "nonzero_samples": sum(
                count for status, count in servo_status_counts.items()
                if status != "0"
            ),
        },
        "robot_actual": {
            "tool_position_span_m": axis_span(actual_tool_positions),
            "tool_orientation_excursion_from_first_deg": orientation_excursion_deg(
                actual_tool_quaternions
            ),
            "arm_joints": joint_summary(joint_samples, ARM_JOINTS),
            "hand_active_joints": joint_summary(joint_samples, HAND_ACTIVE_JOINTS),
        },
        "hand_command_observation": {
            "hand_action_messages": hand_action_messages,
            "hand_trajectory_commands": hand_trajectory_commands,
        },
        "object_actual": {
            "model": args.object_model,
            "samples": len(object_positions),
            "first_position_m": object_positions[0] if object_positions else None,
            "last_position_m": object_positions[-1] if object_positions else None,
            "position_span_m": axis_span(object_positions),
            "first_to_last_displacement_m": displacement(
                object_positions[0] if object_positions else None,
                object_positions[-1] if object_positions else None,
            ),
            "robot_contact_state_samples": robot_object_contacts,
            "ground_contact_state_samples": object_ground_contacts,
            "contact_pairs": [
                {"collisions": list(pair), "state_samples": count}
                for pair, count in sorted(contact_pairs.items())
            ],
        },
    }
    if labelled_transition is not None:
        result["labelled_transition"] = labelled_transition

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
