#!/usr/bin/env python3
"""Offline six-direction acceptance for camera-range ground-sector mapping."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from handarm_moveit_demo.shared_teleop_core import (
    CameraRangeWorkspaceMapper, GroundSectorWorkspace, so3_exp, so3_log,
)


def parse_args():
    parser = argparse.ArgumentParser()
    package = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--config", default=str(package / "config/shared_teleop.yaml"))
    parser.add_argument(
        "--calibration",
        default=str(package / "config/camera_workspace_calibration.yaml"))
    parser.add_argument(
        "--robot-zero", nargs=3, type=float, default=[0.302, 0.0, 0.388])
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    calibration = yaml.safe_load(Path(args.calibration).read_text(encoding="utf-8"))
    profile = config["mapping_profiles"]["camera_ground_workspace"]
    robot = profile["robot_workspace"]
    workspace = GroundSectorWorkspace(
        robot["center_base_m"], robot["radii_m"],
        robot["minimum_forward_x_m"], robot["minimum_tool_z_m"],
        robot["utilization"], robot["boundary_margin_m"])
    human = calibration["human_workspace"]
    human_orientation = calibration["human_orientation"]
    normalized = profile["normalized_pose_mapping"]
    mapper = CameraRangeWorkspaceMapper(
        profile["translation_matrix"], profile["rotation_matrix"],
        profile["rotation_gain"],
        math.radians(config["control"]["maximum_relative_rotation_deg"]),
        human["negative_extent_m"], human["positive_extent_m"],
        workspace, profile.get("response_exponent", 1.0),
        np.radians(human_orientation["negative_extent_deg"]),
        np.radians(human_orientation["positive_extent_deg"]),
        np.radians(normalized[
            "robot_orientation_negative_extent_deg"]),
        np.radians(normalized[
            "robot_orientation_positive_extent_deg"]), True)
    zero = np.asarray(args.robot_zero, dtype=np.float64)
    if not workspace.contains(zero):
        raise SystemExit("robot zero lies outside configured ground sector")

    rows = []
    cases = [
        ("IMAGE_RIGHT_TO_BASE_NEGATIVE_Y", 0, +1),
        ("IMAGE_LEFT_TO_BASE_POSITIVE_Y", 0, -1),
        ("IMAGE_DOWN_TO_GROUND", 1, +1),
        ("IMAGE_UP_TO_BASE_POSITIVE_Z", 1, -1),
        ("AWAY_FROM_CAMERA_TO_INNER_FRONT", 2, +1),
        ("TOWARD_CAMERA_TO_OUTER_FRONT", 2, -1),
    ]
    negative = np.asarray(human["negative_extent_m"], dtype=np.float64)
    positive = np.asarray(human["positive_extent_m"], dtype=np.float64)
    for label, axis, sign in cases:
        hand = np.zeros(3)
        hand[axis] = sign * (positive[axis] if sign > 0 else negative[axis])
        target, _ = mapper.map(hand, np.eye(3), zero, np.eye(3))
        ellipsoid_gap = abs(1.0 - workspace.ellipsoid_value(target))
        boundary_gap = min(
            ellipsoid_gap,
            abs(target[0] - workspace.minimum_forward_x_m),
            abs(target[2] - workspace.minimum_tool_z_m),
        )
        diagnostics = mapper.mapping_diagnostics()
        rows.append({
            "case": label,
            "hand_delta_m": hand.tolist(),
            "target_position_m": target.tolist(),
            "inside_workspace": workspace.contains(target),
            "on_workspace_boundary": bool(boundary_gap <= 1.0e-7),
            "human_fraction": diagnostics["human_translation_fraction"],
            "robot_boundary_distance_m": diagnostics[
                "robot_boundary_distance_m"],
        })
    orientation_rows = []
    human_rotation_negative = np.radians(np.asarray(
        human_orientation["negative_extent_deg"], dtype=np.float64))
    human_rotation_positive = np.radians(np.asarray(
        human_orientation["positive_extent_deg"], dtype=np.float64))
    robot_rotation_negative = np.asarray(
        normalized["robot_orientation_negative_extent_deg"],
        dtype=np.float64)
    robot_rotation_positive = np.asarray(
        normalized["robot_orientation_positive_extent_deg"],
        dtype=np.float64)
    for axis in range(3):
        for sign in (-1, +1):
            hand_angle = sign * (
                human_rotation_positive[axis] if sign > 0
                else human_rotation_negative[axis])
            hand_vector = np.zeros(3)
            hand_vector[axis] = hand_angle
            target_position, target_rotation = mapper.map(
                np.zeros(3), so3_exp(hand_vector), zero, np.eye(3))
            target_vector_deg = np.degrees(so3_log(target_rotation))
            expected_angle_deg = sign * (
                robot_rotation_positive[axis] if sign > 0
                else robot_rotation_negative[axis])
            diagnostics = mapper.mapping_diagnostics()
            orientation_rows.append({
                "case": "LOCAL_{}_{}".format(
                    "XYZ"[axis], "POSITIVE" if sign > 0 else "NEGATIVE"),
                "hand_rotation_vector_deg": np.degrees(
                    hand_vector).tolist(),
                "target_rotation_vector_deg": target_vector_deg.tolist(),
                "expected_axis_angle_deg": float(expected_angle_deg),
                "position_unchanged": bool(np.allclose(
                    target_position, zero, atol=1.0e-12)),
                "human_rotation_fraction": diagnostics[
                    "human_rotation_fraction"],
                "angle_matches_configured_boundary": bool(np.allclose(
                    target_vector_deg,
                    np.eye(3)[axis] * expected_angle_deg,
                    atol=1.0e-7)),
            })

    returned, returned_rotation = mapper.map(
        np.zeros(3), np.eye(3), zero, np.eye(3))
    passed = bool(
        np.allclose(returned, zero, atol=1.0e-12) and
        np.allclose(returned_rotation, np.eye(3), atol=1.0e-12) and
        all(row["inside_workspace"] and row["on_workspace_boundary"] and
            abs(row["human_fraction"] - 1.0) <= 1.0e-9
            for row in rows) and
        all(row["position_unchanged"] and
            row["angle_matches_configured_boundary"] and
            abs(row["human_rotation_fraction"] - 1.0) <= 1.0e-9
            for row in orientation_rows))
    result = {
        "passed": passed,
        "mapping_profile": "camera_ground_workspace",
        "calibration_status": calibration.get("status", "UNKNOWN"),
        "robot_zero_position_m": zero.tolist(),
        "return_to_zero_exact": bool(
            np.allclose(returned, zero, atol=1.0e-12) and
            np.allclose(returned_rotation, np.eye(3), atol=1.0e-12)),
        "workspace": workspace.as_dict(),
        "directions": rows,
        "orientation_directions": orientation_rows,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
        print(str(destination))
    else:
        print(payload)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
