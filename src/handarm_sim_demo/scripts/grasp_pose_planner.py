#!/usr/bin/env python3
"""Object-relative three-finger grasp pose planner (plan-only baseline).

The planner reads the live Gazebo object pose, updates the exact MoveIt box,
performs bounded top/side/roll enclosure search, asks MoveIt for the complete
six-joint IK, validates the pregrasp trajectory and publishes RViz evidence.
It never changes joint_6 after IK and never commands the robot or hand.
"""

from collections import Counter
import copy
import datetime
import json
import math
import os
import sys
import time

import moveit_commander
import numpy as np
import rospkg
import rospy
import yaml
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from std_msgs.msg import Bool, String


PACKAGE_PATH = rospkg.RosPack().get_path("handarm_sim_demo")
sys.path.insert(0, os.path.join(PACKAGE_PATH, "scripts"))

from grasp_candidate_quality import evaluate_candidate_quality
from grasp_geometry import HandGeometry, transform, validate_rotation
from grasp_pose_visualizer import GraspPoseVisualizer, matrix_pose


def quaternion_to_rotation(quaternion):
    values = np.asarray(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("object quaternion is not finite")
    norm = float(np.linalg.norm(values))
    if norm < 1.0e-12:
        raise ValueError("object quaternion is degenerate")
    x, y, z, w = values / norm
    return validate_rotation(
        np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        ),
        "quaternion rotation",
    )


def pose_matrix(pose):
    return transform(
        quaternion_to_rotation(pose.orientation),
        [pose.position.x, pose.position.y, pose.position.z],
    )


def quaternion_distance_deg(first, second):
    a = np.asarray([first.x, first.y, first.z, first.w], dtype=float)
    b = np.asarray([second.x, second.y, second.z, second.w], dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = min(1.0, max(-1.0, abs(float(a @ b))))
    return math.degrees(2.0 * math.acos(dot))


def position_distance(first, second):
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def unpack_plan(result):
    if isinstance(result, tuple):
        success, trajectory, planning_time, _ = result
        return bool(success), trajectory, float(planning_time)
    return bool(result.joint_trajectory.points), result, None


def geometry_score(candidate, search_config=None):
    result = candidate.enclosure
    if not result.valid:
        return -1.0e9
    fractions = [value.closure_fraction for value in result.contacts.values()]
    spread = max(fractions) - min(fractions)
    clearance = min(result.table_clearance_m, 0.050)
    palm = min(result.palm_clearance_m, 0.050)
    offset = float(np.linalg.norm(candidate.center_offset_hand_m[:2]))
    # The real f1/f2-only counterexample showed that treating one 1/30-sweep
    # f3 lag as harmless is not robust to finite-effort contact.  Prefer a
    # pose where f3 is no later than the first opposing contact.  The penalty
    # is configuration data and never changes the hand's physical model.
    f3_lag = max(
        0.0,
        result.contacts["f3"].closure_fraction
        - min(
            result.contacts["f1"].closure_fraction,
            result.contacts["f2"].closure_fraction,
        ),
    )
    lag_penalty = float(
        (search_config or {}).get("f3_contact_lag_penalty_per_fraction", 0.0)
    )
    return (
        100.0
        + 10000.0 * result.projected_contact_area_m2
        + 100.0 * clearance
        + 50.0 * palm
        - 25.0 * spread
        - 100.0 * offset
        - lag_penalty * f3_lag
    )


def calibrated_roll_samples(window_deg, step_deg):
    """Return an inclusive, finite object-frame calibration roll grid."""
    if not isinstance(window_deg, (list, tuple)) or len(window_deg) != 2:
        raise ValueError("calibrated roll window must contain two values")
    lower, upper = (float(window_deg[0]), float(window_deg[1]))
    step = float(step_deg)
    if not all(math.isfinite(value) for value in (lower, upper, step)):
        raise ValueError("calibrated roll grid must be finite")
    if lower < 0.0 or upper > 360.0 or lower > upper or step <= 0.0:
        raise ValueError("calibrated roll window/step is invalid")
    values = []
    value = lower
    while value <= upper + 1.0e-9:
        values.append(round(value, 9))
        value += step
    if not values or abs(values[-1] - upper) > 1.0e-9:
        values.append(upper)
    return values


class ThreeFingerPosePlanner:
    def __init__(self):
        self.scene_path = rospy.get_param(
            "~scene_config",
            os.path.join(PACKAGE_PATH, "config", "physical_grasp_scene.yaml"),
        )
        self.geometry_path = rospy.get_param(
            "~geometry_config",
            os.path.join(PACKAGE_PATH, "config", "three_finger_grasp_geometry.yaml"),
        )
        self.grasp_family = rospy.get_param("~grasp_family", "auto")
        if self.grasp_family not in ("top_down", "top_oblique", "side", "auto"):
            raise ValueError(
                "grasp_family must be top_down, top_oblique, side or auto"
            )
        self.results_dir = rospy.get_param(
            "~results_dir",
            os.path.abspath(
                os.path.join(PACKAGE_PATH, "..", "..", "results", "sim_baseline")
            ),
        )
        with open(self.scene_path, "r", encoding="utf-8") as stream:
            self.scene_config = yaml.safe_load(stream)
        with open(self.geometry_path, "r", encoding="utf-8") as stream:
            self.geometry_config = yaml.safe_load(stream)
        target_key = rospy.get_param("~target_object_key", "target")
        self.target_spec = self.scene_config["objects"][target_key]
        self.target_name = self.target_spec["name"]
        support_surface_key = self.scene_config.get(
            "support_surface_key", "table")
        table_spec = self.scene_config["objects"][support_surface_key]
        self.table_z = float(table_spec["pose"]["position"][2]) + 0.5 * float(
            table_spec["size"][2]
        )
        urdf_path = rospy.get_param(
            "~urdf_path",
            os.path.join(
                rospkg.RosPack().get_path("abb120_moveit_config1"),
                "config",
                "gazebo_handarm.urdf",
            ),
        )
        self.geometry = HandGeometry(urdf_path, self.geometry_path)
        self.group_name = rospy.get_param("~planning_group", "abbarm")
        self.end_effector_link = self.geometry_config["frames"]["tool"]
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.group.set_end_effector_link(self.end_effector_link)
        self.group.set_planner_id("RRTConnect")
        self.group.set_planning_time(5.0)
        self.group.set_num_planning_attempts(5)
        runtime = self.geometry_config["runtime_acceptance"]
        self.group.set_max_velocity_scaling_factor(
            float(runtime["approach_velocity_scaling"])
        )
        self.group.set_max_acceleration_scaling_factor(
            float(runtime["approach_acceleration_scaling"])
        )
        self.compute_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        self.check_state = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity
        )
        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        self.visualizer = GraspPoseVisualizer()
        self.status = rospy.Publisher(
            "/handarm_sim_demo/three_finger_grasp_status",
            String,
            queue_size=1,
            latch=True,
        )

    def wait_ready(self):
        rospy.loginfo("[three-finger-plan] Waiting for startup and synchronized scene...")
        for topic in ("/handarm_sim_demo/startup_ready", "/handarm_sim_demo/scene_ready"):
            message = rospy.wait_for_message(topic, Bool, timeout=90.0)
            if not message.data:
                raise RuntimeError("{} reported false".format(topic))
        for service in (
            "/gazebo/get_model_state",
            "/compute_ik",
            "/check_state_validity",
        ):
            rospy.wait_for_service(service, timeout=30.0)

    def stable_object_pose(self):
        acceptance = self.geometry_config["runtime_acceptance"]
        count = int(acceptance["object_pose_stability_samples"])
        samples = []
        for _ in range(count):
            response = self.get_model_state(self.target_name, "world")
            if not response.success:
                raise RuntimeError(
                    "OBJECT_POSE_UNAVAILABLE: {}".format(response.status_message)
                )
            samples.append(copy.deepcopy(response.pose))
            rospy.sleep(0.05)
        position_range = max(
            position_distance(first.position, second.position)
            for first in samples
            for second in samples
        )
        orientation_range = max(
            quaternion_distance_deg(first.orientation, second.orientation)
            for first in samples
            for second in samples
        )
        if position_range > float(acceptance["object_pose_position_range_m"]):
            raise RuntimeError(
                "OBJECT_POSE_NOT_STABLE: position range {:.6f}m".format(
                    position_range
                )
            )
        if orientation_range > float(
            acceptance["object_pose_orientation_range_deg"]
        ):
            raise RuntimeError(
                "OBJECT_POSE_NOT_STABLE: orientation range {:.3f}deg".format(
                    orientation_range
                )
            )
        return samples[-1], position_range, orientation_range

    def _synchronize_object_box(self, pose, size):
        dimensions_expected = [float(value) for value in size]
        if len(dimensions_expected) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in dimensions_expected
        ):
            raise ValueError("object proxy size must contain three positive values")
        stamped = PoseStamped()
        stamped.header.frame_id = self.scene_config["frame_id"]
        stamped.header.stamp = rospy.Time.now()
        stamped.pose = pose
        self.scene.add_box(
            self.target_name, stamped, size=tuple(dimensions_expected)
        )
        collision = self.scene.get_objects([self.target_name]).get(self.target_name)
        if collision is None or len(collision.primitives) != 1:
            raise RuntimeError("OBJECT_POSE_SYNC_FAILED: object proxy missing")
        dimensions = list(collision.primitives[0].dimensions)
        if max(
            abs(actual - expected)
            for actual, expected in zip(dimensions, dimensions_expected)
        ) > 1.0e-9:
            raise RuntimeError("OBJECT_POSE_SYNC_FAILED: dimensions differ")
        planning_pose = collision.pose
        if collision.primitive_poses:
            local = pose_matrix(collision.primitive_poses[0])
            planning_matrix = pose_matrix(planning_pose) @ local
            planning_pose = matrix_pose(planning_matrix)
        position_error = position_distance(planning_pose.position, pose.position)
        orientation_error = quaternion_distance_deg(
            planning_pose.orientation, pose.orientation
        )
        return position_error, orientation_error

    def synchronize_exact_object(self, pose):
        position_error, orientation_error = self._synchronize_object_box(
            pose, self.target_spec["size"]
        )
        acceptance = self.geometry_config["runtime_acceptance"]
        if position_error > float(
            acceptance["object_scene_sync_position_tolerance_m"]
        ) or orientation_error > float(
            acceptance["object_scene_sync_orientation_tolerance_deg"]
        ):
            raise RuntimeError(
                "OBJECT_POSE_SYNC_FAILED: position={:.6f}m orientation={:.3f}deg".format(
                    position_error, orientation_error
                )
            )
        return position_error, orientation_error

    def synchronize_padded_object(self, pose):
        padding = self.target_spec.get("planning_padding_m", [0.0, 0.0, 0.0])
        if len(padding) != 3 or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in padding
        ):
            raise ValueError("planning_padding_m must contain three nonnegative values")
        padded_size = [
            float(size) + 2.0 * float(margin)
            for size, margin in zip(self.target_spec["size"], padding)
        ]
        position_error, orientation_error = self._synchronize_object_box(
            pose, padded_size
        )
        acceptance = self.geometry_config["runtime_acceptance"]
        if position_error > float(
            acceptance["object_scene_sync_position_tolerance_m"]
        ) or orientation_error > float(
            acceptance["object_scene_sync_orientation_tolerance_deg"]
        ):
            raise RuntimeError(
                "PADDED_OBJECT_POSE_SYNC_FAILED: position={:.6f}m "
                "orientation={:.3f}deg".format(
                    position_error, orientation_error
                )
            )
        return padded_size

    def _ik(self, matrix, seed_state=None, timeout_s=0.20):
        request = GetPositionIKRequest()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.end_effector_link
        request.ik_request.pose_stamped.header.frame_id = self.scene_config["frame_id"]
        request.ik_request.pose_stamped.header.stamp = rospy.Time.now()
        request.ik_request.pose_stamped.pose = matrix_pose(matrix)
        request.ik_request.robot_state = (
            copy.deepcopy(seed_state)
            if seed_state is not None
            else self.robot.get_current_state()
        )
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout = rospy.Duration(float(timeout_s))
        response = self.compute_ik(request)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, "IK_FAILED_{}".format(response.error_code.val)
        state = response.solution
        values = dict(zip(state.joint_state.name, state.joint_state.position))
        active = self.group.get_active_joints()
        if any(name not in values or not math.isfinite(values[name]) for name in active):
            return None, "IK_INVALID_SOLUTION"
        validity = self.check_state(
            GetStateValidityRequest(robot_state=state, group_name=self.group_name)
        )
        if not validity.valid:
            return None, "IK_STATE_COLLISION"
        return state, ""

    def _joint_metrics(self, state):
        values = dict(zip(state.joint_state.name, state.joint_state.position))
        active = self.group.get_active_joints()
        current = dict(zip(active, self.group.get_current_joint_values()))
        margins = []
        for name in active:
            lower, upper = self.robot.get_joint(name).bounds()
            if math.isfinite(lower) and math.isfinite(upper):
                margins.append(min(values[name] - lower, upper - values[name]))
        distance = math.sqrt(sum((values[name] - current[name]) ** 2 for name in active))
        return {
            "joint_positions": {name: values[name] for name in active},
            "joint_6_rad": values["joint_6"],
            "joint_limit_margin_rad": min(margins) if margins else None,
            "distance_from_current_state_rad": distance,
        }

    def evaluate_complete_ik(self, candidate):
        distance = float(self.geometry_config["search"]["pregrasp_distance_m"])
        approach = candidate.T_world_hand[:3, 2]
        pregrasp = candidate.T_world_tool0.copy()
        pregrasp[:3, 3] -= distance * approach
        # Reject an unreachable grasp immediately. This makes complete Roll
        # coverage cheap and prevents geometry score from hiding a reachable
        # lower-ranked wrist branch.
        grasp_state, reason = self._ik(candidate.T_world_tool0)
        if grasp_state is None:
            return None, reason
        pregrasp_state, reason = self._ik(pregrasp)
        if pregrasp_state is None:
            return None, reason
        pregrasp_metrics = self._joint_metrics(pregrasp_state)
        previous = pregrasp_state
        # Use actual Cartesian fractions and timestamps later during execution;
        # plan-only first requires collision-aware full IK at every pose sample.
        for fraction in np.linspace(0.0, 1.0, 9)[1:]:
            waypoint = pregrasp.copy()
            waypoint[:3, 3] += fraction * distance * approach
            state, reason = self._ik(waypoint, previous)
            if state is None:
                return None, "APPROACH_INCOMPLETE_{}".format(reason)
            previous_values = dict(
                zip(previous.joint_state.name, previous.joint_state.position)
            )
            values = dict(zip(state.joint_state.name, state.joint_state.position))
            if max(
                abs(values[name] - previous_values[name])
                for name in self.group.get_active_joints()
            ) > 0.75:
                return None, "APPROACH_IK_BRANCH_JUMP"
            previous = state
        metrics = self._joint_metrics(previous)
        metrics["pregrasp_joint_positions"] = pregrasp_metrics[
            "joint_positions"
        ]
        metrics["pregrasp_joint_6_rad"] = pregrasp_metrics["joint_6_rad"]
        metrics["pregrasp_joint_limit_margin_rad"] = pregrasp_metrics[
            "joint_limit_margin_rad"
        ]
        metrics["pregrasp_T_world_tool0"] = pregrasp.tolist()
        metrics["grasp_T_world_tool0"] = candidate.T_world_tool0.tolist()
        metrics["approach_ik_samples"] = 9
        metrics["ik_success"] = True
        return metrics, ""

    def plan_pregrasp(self, ik_metrics):
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()
        target = ik_metrics["pregrasp_joint_positions"]
        self.group.set_joint_value_target(target)
        if not hasattr(self, "_stable_object_pose"):
            raise RuntimeError("stable object pose is unavailable for padded transit")
        padded_size = self.synchronize_padded_object(self._stable_object_pose)
        try:
            success, trajectory, planning_time = unpack_plan(self.group.plan())
            if not success or not trajectory.joint_trajectory.points:
                return None, "PADDED_PREGRASP_PLAN_FAILED"
            state = self.robot.get_current_state()
            state_index = {
                name: index for index, name in enumerate(state.joint_state.name)
            }
            positions = list(state.joint_state.position)
            previous_time = -1.0
            for point in trajectory.joint_trajectory.points:
                seconds = point.time_from_start.to_sec()
                if not math.isfinite(seconds) or seconds <= previous_time:
                    return None, "PREGRASP_TRAJECTORY_TIME_INVALID"
                previous_time = seconds
                for name, value in zip(
                    trajectory.joint_trajectory.joint_names, point.positions
                ):
                    positions[state_index[name]] = value
                state.joint_state.position = positions
                validity = self.check_state(
                    GetStateValidityRequest(
                        robot_state=state, group_name=self.group_name
                    )
                )
                if not validity.valid:
                    return None, "PADDED_PREGRASP_TRAJECTORY_COLLISION"
        finally:
            # Exact geometry is mandatory for the subsequent approach and
            # contact reasoning.  There is no unpadded transit fallback.
            self.synchronize_exact_object(self._stable_object_pose)
        self._planned_pregrasp_trajectory = trajectory
        return {
            "pregrasp_plan_success": True,
            "pregrasp_planning_proxy_size_m": padded_size,
            "pregrasp_planning_padding_enforced": True,
            "pregrasp_planning_time_s": planning_time,
            "pregrasp_trajectory_points": len(
                trajectory.joint_trajectory.points
            ),
            "pregrasp_trajectory_duration_s": previous_time,
        }, ""

    def select_candidate(self, candidates):
        valid = sorted(
            (
                item
                for item in candidates
                if item.enclosure.valid
                and item.enclosure.table_clearance_m
                >= float(
                    self.geometry_config["search"]
                    ["minimum_coarse_table_clearance_m"]
                )
                and evaluate_candidate_quality(
                    item, self.geometry_config
                ).passed
            ),
            key=lambda item: geometry_score(
                item, self.geometry_config["search"]
            ),
            reverse=True,
        )
        if not valid:
            return None, None, {"failure_reason": "NO_THREE_FINGER_GEOMETRY_CANDIDATE"}
        # Evaluate one best offset for every family/direction/tilt/Roll.  The
        # former four-per-direction cap silently missed reachable wrist rolls.
        shortlist_by_key = {}
        for item in valid:
            key = (
                item.family,
                item.direction,
                item.tilt_deg,
                item.roll_deg,
            )
            if key not in shortlist_by_key:
                shortlist_by_key[key] = item
        shortlist = list(shortlist_by_key.values())
        ik_failures = Counter()
        accepted = []
        for item in shortlist:
            metrics, reason = self.evaluate_complete_ik(item)
            if metrics is None:
                ik_failures[reason] += 1
                continue
            combined = geometry_score(
                item, self.geometry_config["search"]
            ) - metrics["distance_from_current_state_rad"]
            accepted.append((combined, item, metrics))
        if not accepted:
            return None, None, {
                "failure_reason": "NO_COMPLETE_IK_CANDIDATE",
                "ik_failures": dict(ik_failures),
            }
        accepted.sort(key=lambda value: value[0], reverse=True)
        _, selected, metrics = accepted[0]
        fine_candidates = []
        fine_accepted = []
        if selected.family in ("top_down", "top_oblique"):
            half_width = int(
                self.geometry_config["search"]["fine_roll_half_width_deg"]
            )
            fine_step = int(self.geometry_config["search"]["fine_roll_step_deg"])
            if selected.family == "top_oblique":
                planar_offsets = self.geometry_config["search"][
                    "top_oblique_planar_offsets_m"
                ]
            else:
                planar_offsets = [selected.center_offset_hand_m[:2]]
            roll_values = {
                (selected.roll_deg + delta) % 360.0
                for delta in range(-half_width, half_width + 1, fine_step)
            }
            if selected.family == "top_oblique":
                roll_values.update(
                    calibrated_roll_samples(
                        self.geometry_config["search"]
                        ["contact_calibrated_top_oblique_roll_window_deg"],
                        fine_step,
                    )
                )
            for roll in sorted(roll_values):
                for planar_offset in planar_offsets:
                    fine = self.geometry.make_candidate(
                        self._T_world_object,
                        self.target_spec["size"],
                        self.table_z,
                        selected.family,
                        selected.direction,
                        roll,
                        planar_offset,
                        selected.side_height_m,
                        selected.tilt_deg,
                        selected.object_center_axial_offset_m,
                    )
                    fine_candidates.append(fine)
                    if (
                        not fine.enclosure.valid
                        or fine.enclosure.table_clearance_m
                        < float(
                            self.geometry_config["search"]
                            ["minimum_planned_table_clearance_m"]
                        )
                        or not evaluate_candidate_quality(
                            fine, self.geometry_config
                        ).passed
                    ):
                        continue
                    fine_metrics, _ = self.evaluate_complete_ik(fine)
                    if fine_metrics is None:
                        continue
                    score = geometry_score(
                        fine, self.geometry_config["search"]
                    ) - fine_metrics["distance_from_current_state_rad"]
                    fine_accepted.append((score, fine, fine_metrics))
            # A contact-calibrated roll must be evaluated for every configured
            # top-oblique tilt/axial pair, not only the best coarse pair.  A
            # coarse 15-degree winner can otherwise prevent a more liftable
            # tilt from ever being instantiated at the exact calibrated roll.
            if selected.family == "top_oblique":
                calibrated_rolls = calibrated_roll_samples(
                    self.geometry_config["search"]
                    ["contact_calibrated_top_oblique_roll_window_deg"],
                    fine_step,
                )
                for specification in self.geometry_config["search"][
                    "top_oblique_tilt_axial_pairs"
                ]:
                    tilt = float(specification["tilt_deg"])
                    axial = float(
                        specification["object_center_axial_offset_m"]
                    )
                    if (
                        abs(tilt - selected.tilt_deg) <= 1.0e-12
                        and abs(axial - selected.object_center_axial_offset_m)
                        <= 1.0e-12
                    ):
                        continue
                    for roll in calibrated_rolls:
                        for planar_offset in planar_offsets:
                            fine = self.geometry.make_candidate(
                                self._T_world_object,
                                self.target_spec["size"],
                                self.table_z,
                                "top_oblique",
                                "object_pos_z",
                                roll,
                                planar_offset,
                                0.0,
                                tilt,
                                axial,
                            )
                            fine_candidates.append(fine)
                            if (
                                not fine.enclosure.valid
                                or fine.enclosure.table_clearance_m
                                < float(
                                    self.geometry_config["search"]
                                    ["minimum_planned_table_clearance_m"]
                                )
                                or not evaluate_candidate_quality(
                                    fine, self.geometry_config
                                ).passed
                            ):
                                continue
                            fine_metrics, _ = self.evaluate_complete_ik(fine)
                            if fine_metrics is None:
                                continue
                            score = geometry_score(
                                fine, self.geometry_config["search"]
                            ) - fine_metrics["distance_from_current_state_rad"]
                            fine_accepted.append((score, fine, fine_metrics))
            accepted.extend(fine_accepted)
        final_clearance = float(
            self.geometry_config["search"]["minimum_planned_table_clearance_m"]
        )
        accepted = [
            value
            for value in accepted
            if value[1].enclosure.table_clearance_m >= final_clearance
            and evaluate_candidate_quality(
                value[1], self.geometry_config
            ).passed
            and (
                value[1].family != "top_oblique"
                or float(
                    self.geometry_config["search"]
                    ["contact_calibrated_top_oblique_roll_window_deg"][0]
                )
                <= value[1].roll_deg
                <= float(
                    self.geometry_config["search"]
                    ["contact_calibrated_top_oblique_roll_window_deg"][1]
                )
            )
        ]
        if not accepted:
            return None, None, {
                "failure_reason": "NO_FINAL_TABLE_CLEARANCE_CANDIDATE",
                "ik_failures": dict(ik_failures),
            }
        accepted.sort(key=lambda value: value[0], reverse=True)
        _, selected, metrics = accepted[0]
        plan, reason = self.plan_pregrasp(metrics)
        if plan is None:
            return None, None, {
                "failure_reason": reason,
                "ik_failures": dict(ik_failures),
            }
        metrics.update(plan)
        metrics["coarse_geometry_score"] = geometry_score(
            selected, self.geometry_config["search"]
        )
        metrics["coarse_ik_candidate_count"] = len(shortlist)
        metrics["fine_roll_candidate_count"] = len(fine_candidates)
        metrics["ik_failures"] = dict(ik_failures)
        return selected, metrics, {}

    def run(self):
        self.wait_ready()
        pose, position_range, orientation_range = self.stable_object_pose()
        self._stable_object_pose = copy.deepcopy(pose)
        sync_position, sync_orientation = self.synchronize_exact_object(pose)
        T_world_object = pose_matrix(pose)
        self._T_world_object = T_world_object
        rospy.loginfo(
            "[three-finger-plan] Live object pose stable; generating %s candidates...",
            self.grasp_family,
        )
        started = time.monotonic()
        candidates = self.geometry.coarse_geometry_candidates(
            T_world_object,
            self.target_spec["size"],
            self.table_z,
            self.grasp_family,
        )
        # Calibration rolls participate in the first complete-IK screen.  If
        # all 15-degree coarse rolls fail, waiting until the later fine pass
        # would incorrectly return NO_COMPLETE_IK_CANDIDATE without ever
        # testing the measured 268-degree enclosure.
        if self.grasp_family in ("top_oblique", "auto"):
            fine_step = int(
                self.geometry_config["search"]["fine_roll_step_deg"]
            )
            calibrated_rolls = calibrated_roll_samples(
                self.geometry_config["search"]
                ["contact_calibrated_top_oblique_roll_window_deg"],
                fine_step,
            )
            for specification in self.geometry_config["search"][
                "top_oblique_tilt_axial_pairs"
            ]:
                for roll in calibrated_rolls:
                    for planar_offset in self.geometry_config["search"][
                        "top_oblique_planar_offsets_m"
                    ]:
                        candidates.append(
                            self.geometry.make_candidate(
                                T_world_object,
                                self.target_spec["size"],
                                self.table_z,
                                "top_oblique",
                                "object_pos_z",
                                roll,
                                planar_offset,
                                0.0,
                                float(specification["tilt_deg"]),
                                float(
                                    specification[
                                        "object_center_axial_offset_m"
                                    ]
                                ),
                            )
                        )
        geometry_time = time.monotonic() - started
        selected, ik_metrics, failure = self.select_candidate(candidates)
        rejection_counts = Counter(
            reason
            for item in candidates
            if not item.enclosure.valid
            for reason in item.enclosure.failure_reasons
        )
        quality_rejection_counts = Counter()
        for item in candidates:
            if not item.enclosure.valid:
                continue
            quality = evaluate_candidate_quality(item, self.geometry_config)
            for reason in quality.failures:
                quality_rejection_counts[reason] += 1
        family_counts = Counter(
            "{}:{}".format(item.family, item.direction)
            for item in candidates
            if item.enclosure.valid
        )
        record = {
            "schema_version": 1,
            "mode": "three_finger_pose_plan_only",
            "plan_only": True,
            "robot_executed": False,
            "hand_closed": False,
            "object_pose_source": "/gazebo/get_model_state",
            "target_object": self.target_name,
            "T_world_object": T_world_object.tolist(),
            "object_size_m": list(self.target_spec["size"]),
            "table_z_m": self.table_z,
            "object_pose_position_range_m": position_range,
            "object_pose_orientation_range_deg": orientation_range,
            "planning_scene_position_error_m": sync_position,
            "planning_scene_orientation_error_deg": sync_orientation,
            "grasp_family_requested": self.grasp_family,
            "candidate_count": len(candidates),
            "geometry_valid_count": sum(
                item.enclosure.valid for item in candidates
            ),
            "quality_rejection_counts": dict(quality_rejection_counts),
            "geometry_valid_by_direction": dict(family_counts),
            "rejection_counts": dict(rejection_counts),
            "geometry_search_time_s": geometry_time,
            "grasp_center": self.geometry.transform_summary(),
            "success": selected is not None,
            "failure_reason": failure.get("failure_reason", ""),
            "failure_details": failure,
            "selected_candidate": selected.as_dict() if selected else None,
            "selected_quality": (
                evaluate_candidate_quality(
                    selected, self.geometry_config
                ).as_dict()
                if selected is not None
                else None
            ),
            "moveit": ik_metrics,
            "joint_6_source": "complete_moveit_ik" if selected else None,
        }
        self.visualizer.reset()
        self.visualizer.add_obb(T_world_object, self.target_spec["size"])
        self.visualizer.add_frame("object_frame", T_world_object, 0.09, 0.007)
        self.visualizer.add_candidates(
            candidates,
            float(self.geometry_config["search"]["pregrasp_distance_m"]),
        )
        if selected is not None:
            self.visualizer.add_selected(
                selected,
                float(self.geometry_config["search"]["pregrasp_distance_m"]),
                ik_metrics["joint_6_rad"],
            )
            self.visualizer.add_pad_sweeps(self.geometry, selected)
        self.visualizer.publish()
        os.makedirs(self.results_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        result_path = os.path.join(
            self.results_dir, "three_finger_pose_plan_only_{}.json".format(stamp)
        )
        with open(result_path, "x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
        self.status.publish(json.dumps(record, sort_keys=True))
        if selected is None:
            raise RuntimeError(record["failure_reason"])
        self.selected_candidate = selected
        self.selected_ik_metrics = ik_metrics
        self.plan_only_record = record
        self.plan_only_result_path = result_path
        rospy.loginfo(
            "[three-finger-plan] PLAN-ONLY selected %s/%s roll=%.1fdeg "
            "joint_6=%.3frad; results=%s",
            selected.family,
            selected.direction,
            selected.roll_deg,
            ik_metrics["joint_6_rad"],
            result_path,
        )
        return record, result_path


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("three_finger_grasp_pose_planner")
    try:
        planner = ThreeFingerPosePlanner()
        planner.run()
        rospy.loginfo(
            "[three-finger-plan] Markers are latched. Press Ctrl-C when observation is complete."
        )
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("Three-finger plan-only failed: %s", exc)
        raise SystemExit(8)
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
