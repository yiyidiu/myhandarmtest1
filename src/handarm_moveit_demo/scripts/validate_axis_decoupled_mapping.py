#!/usr/bin/env python3
"""Offline acceptance for perspective and workspace-axis decoupling."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from handarm_moveit_demo.shared_teleop_core import (  # noqa: E402
    AxisDecoupledWorkspaceMapper,
    CameraRangeWorkspaceMapper,
    DirectionPreservingIncrementProjector,
    GroundSectorWorkspace,
    PerspectiveIntentDecoupler,
    apply_ground_sector_workspace_boundary,
)


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def make_workspace(profile):
    value = profile["robot_workspace"]
    return GroundSectorWorkspace(
        value["center_base_m"], value["radii_m"],
        value["minimum_forward_x_m"], value["minimum_tool_z_m"],
        value.get("utilization", 1.0), value.get("boundary_margin_m", 0.0))


def orientation_arguments(profile, calibration):
    normalized = profile["normalized_pose_mapping"]
    human = calibration["human_orientation"]
    return (
        np.radians(human["negative_extent_deg"]),
        np.radians(human["positive_extent_deg"]),
        np.radians(normalized["robot_orientation_negative_extent_deg"]),
        np.radians(normalized["robot_orientation_positive_extent_deg"]),
    )


def make_mapper(profile, calibration, decoupled):
    workspace = make_workspace(profile)
    human = calibration["human_workspace"]
    negative_extent = human["negative_extent_m"]
    positive_extent = human["positive_extent_m"]
    if decoupled:
        negative_extent = human.get(
            "perspective_decoupled_negative_extent_m", negative_extent)
        positive_extent = human.get(
            "perspective_decoupled_positive_extent_m", positive_extent)
    common = (
        profile["translation_matrix"], profile["rotation_matrix"],
        profile["rotation_gain"], math.radians(179.0),
        negative_extent, positive_extent,
        workspace, profile.get("response_exponent", 1.0),
        *orientation_arguments(profile, calibration),
    )
    if decoupled:
        axis = profile["axis_workspace_mapping"]
        return AxisDecoupledWorkspaceMapper(
            *common,
            workspace_projection_safety_factor=axis.get(
                "workspace_projection_safety_factor", 0.995))
    return CameraRangeWorkspaceMapper(*common, False)


def validate(config, calibration):
    old_profile = config["mapping_profiles"]["camera_ground_workspace"]
    new_profile = config["mapping_profiles"][
        "camera_ground_axis_decoupled"]
    zero_camera_depth = 0.60
    decoupler = PerspectiveIntentDecoupler(
        new_profile["perspective_decoupling"]["minimum_depth_m"])

    raw_metric_lateral = 0.0
    control_lateral = 0.0
    for ray_x in (-0.40, -0.20, 0.0, 0.20, 0.40):
        for ray_y in (-0.25, 0.0, 0.25):
            zero = np.array([
                ray_x * zero_camera_depth,
                ray_y * zero_camera_depth,
                zero_camera_depth])
            for depth in np.linspace(0.35, 0.95, 25):
                current = np.array([ray_x * depth, ray_y * depth, depth])
                relative = current - zero
                axial_speed = 0.20
                velocity = np.array([
                    ray_x * axial_speed, ray_y * axial_speed, axial_speed,
                    0.0, 0.0, 0.0])
                result = decoupler.transform(relative, velocity, zero)
                raw_metric_lateral = max(
                    raw_metric_lateral,
                    float(np.linalg.norm(relative[:2])))
                control_lateral = max(
                    control_lateral,
                    float(np.linalg.norm(result.relative_position[:2])),
                    float(np.linalg.norm(result.velocity[:2])))

    # Use the actual current C-zero tool position and calibrated camera limits.
    robot_zero = np.array([0.302, 0.0, 0.388])
    rotation = np.eye(3)
    human = calibration["human_workspace"]
    lateral_input = 0.80 * float(human["positive_extent_m"][0])
    depth_extent = float(human["negative_extent_m"][2])
    depth_sequence = np.concatenate((
        [0.0], np.linspace(0.0, -depth_extent, 21)[1:],
        np.linspace(-depth_extent, depth_extent, 41)[1:]))

    new_mapper = make_mapper(new_profile, calibration, True)
    new_targets = []
    for depth in depth_sequence:
        target, _ = new_mapper.map(
            [lateral_input, 0.0, depth], rotation,
            robot_zero, rotation)
        new_targets.append(target)
    new_targets = np.asarray(new_targets)
    new_lateral_range = float(np.ptp(new_targets[:, 1]))

    # Reproduce the exact edge case: first create a combined lateral target
    # outside the shell, then change only camera depth.  Neither the workspace
    # mapper nor IK may spend an old clipped lateral residual on that frame.
    edge_mapper = make_mapper(new_profile, calibration, True)
    decoupled_negative = human.get(
        "perspective_decoupled_negative_extent_m",
        human["negative_extent_m"])
    decoupled_positive = human.get(
        "perspective_decoupled_positive_extent_m",
        human["positive_extent_m"])
    edge_hand = np.array([
        decoupled_positive[0], decoupled_positive[1], 0.0])
    edge_target, _ = edge_mapper.map(
        edge_hand, rotation, robot_zero, rotation)
    edge_depth_hand = edge_hand.copy()
    edge_depth_hand[2] = 0.20 * decoupled_positive[2]
    edge_depth_target, _ = edge_mapper.map(
        edge_depth_hand, rotation, robot_zero, rotation)
    # Camera depth maps to robot X, so Y/Z are the unrequested components.
    edge_workspace_cross_axis_delta = float(np.linalg.norm(
        edge_depth_target[1:] - edge_target[1:]))

    projector = DirectionPreservingIncrementProjector(
        bisection_iterations=12, safety_factor=0.98)

    def synthetic_reachable(position, _rotation):
        return bool(position[0] ** 2 + position[1] ** 2 +
                    position[2] ** 2 <= 1.0)

    projector_zero = np.zeros(3)
    projector_first = projector.project(
        projector_zero, rotation, [1.0, 0.8, 0.0], rotation,
        synthetic_reachable)
    projector_depth = projector.project(
        projector_zero, rotation, [1.0, 0.8, 0.2], rotation,
        synthetic_reachable)
    edge_ik_cross_axis_delta = float(np.linalg.norm(
        projector_depth.position[:2] - projector_first.position[:2]))

    old_mapper = make_mapper(old_profile, calibration, False)
    old_targets = []
    for depth in depth_sequence:
        target, _ = old_mapper.map(
            [lateral_input, 0.0, depth], rotation,
            robot_zero, rotation)
        old_targets.append(target)
    old_targets = np.asarray(old_targets)
    old_lateral_range = float(np.ptp(old_targets[:, 1]))

    returned, _ = new_mapper.map(
        np.zeros(3), rotation, robot_zero, rotation)
    zero_return_error = float(np.linalg.norm(returned - robot_zero))

    workspace = make_workspace(new_profile)
    boundary_point = workspace.center + np.array([
        workspace.radii[0] / math.sqrt(2.0),
        workspace.radii[1] / math.sqrt(2.0), 0.0])
    legacy_boundary, _ = apply_ground_sector_workspace_boundary(
        boundary_point, [0.20, 0.0, 0.0], workspace, 0.0, False)
    protected_boundary, protected_reasons = (
        apply_ground_sector_workspace_boundary(
            boundary_point, [0.20, 0.0, 0.0], workspace, 0.0, True))

    checks = {
        "perspective_fixed_ray_cross_axis_below_1e-9": (
            control_lateral <= 1.0e-9),
        "axis_mapper_depth_sweep_cross_axis_below_1mm": (
            new_lateral_range <= 0.001),
        "edge_depth_increment_keeps_workspace_lateral_axes": (
            edge_workspace_cross_axis_delta <= 1.0e-9),
        "edge_depth_increment_keeps_ik_lateral_axes": (
            edge_ik_cross_axis_delta <= 1.0e-9),
        "legacy_radial_coupling_reproduced_above_10mm": (
            old_lateral_range >= 0.010),
        "boundary_does_not_create_lateral_velocity": bool(
            np.linalg.norm(protected_boundary) <= 1.0e-12 and
            "WORKSPACE_HARD_ELLIPSOID_DIRECTION_HOLD" in
            protected_reasons),
        "c_zero_return_below_1um": zero_return_error <= 1.0e-6,
    }
    return {
        "schema": "handarm_axis_decoupled_acceptance_v2",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": {
            "fixed_ray_raw_metric_lateral_coupling_m": raw_metric_lateral,
            "fixed_ray_decoupled_cross_axis_max": control_lateral,
            "new_depth_sweep_robot_y_range_m": new_lateral_range,
            "edge_depth_workspace_cross_axis_delta_m": (
                edge_workspace_cross_axis_delta),
            "edge_depth_ik_cross_axis_delta_m": edge_ik_cross_axis_delta,
            "old_depth_sweep_robot_y_range_m": old_lateral_range,
            "legacy_boundary_output_mps": legacy_boundary.tolist(),
            "direction_preserving_boundary_output_mps": (
                protected_boundary.tolist()),
            "c_zero_return_error_m": zero_return_error,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="",
        help="optional JSON report path")
    args = parser.parse_args()
    config = load_yaml(PACKAGE / "config/shared_teleop.yaml")
    calibration = load_yaml(
        PACKAGE / "config/camera_workspace_calibration.yaml")
    report = validate(config, calibration)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
