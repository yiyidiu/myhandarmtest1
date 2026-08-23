#!/usr/bin/env python3
"""Physical three-finger pick, lift, table place, release and retreat.

The object is never attached to the robot.  This node first executes the
independently accepted contact-only sequence, then permits lift only while all
three finger families remain in fresh Gazebo contact with the target.
"""

import copy
import datetime
import json
import math
import os
import sys
import threading
import time

import moveit_commander
import numpy as np
import rospkg
import rospy
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetStateValidityRequest
from trajectory_msgs.msg import JointTrajectoryPoint


PACKAGE_PATH = rospkg.RosPack().get_path("handarm_sim_demo")
sys.path.insert(0, os.path.join(PACKAGE_PATH, "scripts"))

from grasp_candidate_quality import (
    evaluate_actual_lift_evidence,
    validate_planned_lift_vector,
)
from grasp_pose_planner import (
    pose_matrix,
    position_distance,
    quaternion_distance_deg,
)
from grasp_pose_visualizer import matrix_pose
from three_finger_grasp_demo import (
    ThreeFingerContactDemo,
    pose_as_dict,
)


def trajectory_duration_s(trajectory):
    points = trajectory.joint_trajectory.points
    return points[-1].time_from_start.to_sec() if points else 0.0


def opposite_approach_lift_vector(T_world_hand, distance_m):
    """Retract from an object-relative grasp while gaining world height."""
    matrix = np.asarray(T_world_hand, dtype=float)
    distance = float(distance_m)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("T_world_hand must be a finite 4x4 matrix")
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("lift distance must be positive")
    approach = matrix[:3, 2]
    norm = float(np.linalg.norm(approach))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError("grasp approach axis must be unit length")
    vector = -distance * approach
    if vector[2] <= 0.0:
        raise ValueError("opposite approach direction does not lift upward")
    return vector


def pivoted_grasp_center_lift_target(
    T_world_tool,
    T_tool_hand,
    T_hand_grasp_center,
    lift_vector_world_m,
    hand_local_y_tilt_deg,
):
    """Lift the grasp center while tilting the wrist away from self-collision.

    The requested translation belongs to the physical three-finger grasp
    center, not ``tool0``.  Rotating ``tool0`` in place would sweep the palm
    and the held object sideways because the grasp center is about 170 mm
    beyond the tool frame.
    """
    matrices = []
    for name, value in (
        ("T_world_tool", T_world_tool),
        ("T_tool_hand", T_tool_hand),
        ("T_hand_grasp_center", T_hand_grasp_center),
    ):
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("{} must be a finite 4x4 matrix".format(name))
        matrices.append(matrix)
    lift = np.asarray(lift_vector_world_m, dtype=float)
    if lift.shape != (3,) or not np.all(np.isfinite(lift)):
        raise ValueError("lift vector must be a finite 3-vector")
    tilt_deg = float(hand_local_y_tilt_deg)
    if not math.isfinite(tilt_deg) or abs(tilt_deg) > 45.0:
        raise ValueError("hand local-y lift tilt must be within 45 degrees")

    T_world_tool, T_tool_hand, T_hand_grasp_center = matrices
    T_world_hand = T_world_tool @ T_tool_hand
    T_world_grasp_center = T_world_hand @ T_hand_grasp_center
    tilt = math.radians(tilt_deg)
    R_tilt_y = np.array(
        [
            [math.cos(tilt), 0.0, math.sin(tilt)],
            [0.0, 1.0, 0.0],
            [-math.sin(tilt), 0.0, math.cos(tilt)],
        ]
    )
    R_world_hand_target = T_world_hand[:3, :3] @ R_tilt_y
    grasp_center_target = T_world_grasp_center[:3, 3] + lift
    T_world_hand_target = np.eye(4)
    T_world_hand_target[:3, :3] = R_world_hand_target
    T_world_hand_target[:3, 3] = (
        grasp_center_target
        - R_world_hand_target @ T_hand_grasp_center[:3, 3]
    )
    return T_world_hand_target @ np.linalg.inv(T_tool_hand)


def grasp_center_position_from_tool(
    T_world_tool, T_tool_hand, T_hand_grasp_center
):
    """Return the world position of the physical grasp center."""
    return np.asarray(
        T_world_tool @ T_tool_hand @ T_hand_grasp_center,
        dtype=float,
    )[:3, 3]


def enforce_minimum_trajectory_duration(trajectory, minimum_s):
    """Stretch time and consistently reduce velocity/acceleration."""
    minimum_s = float(minimum_s)
    if not math.isfinite(minimum_s) or minimum_s <= 0.0:
        raise ValueError("minimum trajectory duration must be positive")
    points = trajectory.joint_trajectory.points
    if not points:
        raise ValueError("cannot stretch an empty trajectory")
    original_s = points[-1].time_from_start.to_sec()
    if not math.isfinite(original_s) or original_s <= 0.0:
        raise ValueError("trajectory duration is invalid")
    if original_s >= minimum_s:
        return trajectory
    scale = minimum_s / original_s
    for point in points:
        point.time_from_start = rospy.Duration(
            point.time_from_start.to_sec() * scale
        )
        if point.velocities:
            point.velocities = [value / scale for value in point.velocities]
        if point.accelerations:
            point.accelerations = [
                value / (scale * scale) for value in point.accelerations
            ]
    return trajectory


class ThreeFingerPickPlaceDemo(ThreeFingerContactDemo):
    def __init__(self):
        super().__init__()
        self.runtime = self.planner.geometry_config["runtime_acceptance"]

    def validate_trajectory(self, trajectory):
        names = list(trajectory.joint_trajectory.joint_names)
        points = list(trajectory.joint_trajectory.points)
        if not names or not points:
            return False, "empty trajectory"
        state = self.planner.robot.get_current_state()
        indices = {
            name: index for index, name in enumerate(state.joint_state.name)
        }
        if any(name not in indices for name in names):
            return False, "trajectory contains an unknown joint"
        positions = list(state.joint_state.position)
        previous_s = -1.0
        for point in points:
            seconds = point.time_from_start.to_sec()
            if not math.isfinite(seconds) or seconds <= previous_s:
                return False, "trajectory time is not strictly increasing"
            previous_s = seconds
            if len(point.positions) != len(names) or not all(
                math.isfinite(value) for value in point.positions
            ):
                return False, "trajectory contains invalid joint values"
            for name, value in zip(names, point.positions):
                positions[indices[name]] = value
            state.joint_state.position = positions
            validity = self.planner.check_state(
                GetStateValidityRequest(
                    robot_state=state, group_name=self.planner.group_name
                )
            )
            if not validity.valid:
                return False, "trajectory waypoint is in collision"
        return True, ""

    def plan_cartesian(self, target, label, minimum_duration_s):
        trajectory, fraction = self.planner.group.compute_cartesian_path(
            [target], float(self.runtime["cartesian_eef_step_m"]), True
        )
        if fraction < float(self.runtime["cartesian_fraction_min"]):
            raise RuntimeError(
                "{} Cartesian fraction {:.6f}".format(label, fraction)
            )
        trajectory = self.planner.group.retime_trajectory(
            self.planner.robot.get_current_state(),
            trajectory,
            velocity_scaling_factor=float(
                self.runtime["approach_velocity_scaling"]
            ),
            acceleration_scaling_factor=float(
                self.runtime["approach_acceleration_scaling"]
            ),
            algorithm="iterative_time_parameterization",
        )
        trajectory = enforce_minimum_trajectory_duration(
            trajectory, minimum_duration_s
        )
        valid, reason = self.validate_trajectory(trajectory)
        if not valid:
            raise RuntimeError("{} {}".format(label, reason))
        return trajectory, float(fraction)

    def execute_trajectory(self, trajectory, label, require_contact):
        result = {}

        def worker():
            try:
                result["success"] = bool(
                    self.planner.group.execute(trajectory, wait=True)
                )
            except Exception as exc:
                result["exception"] = exc

        if require_contact:
            self.reset_contacts()
            self.wait_for_three_finger_contact()
        started = time.monotonic()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        bad_since = None
        maximum_loss_s = 0.0
        contact_lost = False
        last_bad_snapshot = None
        while thread.is_alive() and not rospy.is_shutdown():
            if require_contact:
                snapshot = self.contact_snapshot()
                now = time.monotonic()
                latest = snapshot["latest_monotonic"]
                fresh = (
                    latest is not None
                    and 0.0 <= now - latest
                    <= float(self.runtime["contact_message_timeout_s"])
                )
                good = (
                    fresh
                    and snapshot["latest_families"] == self.required
                    and not snapshot["unexpected"]
                )
                if good:
                    if bad_since is not None:
                        maximum_loss_s = max(maximum_loss_s, now - bad_since)
                    bad_since = None
                else:
                    last_bad_snapshot = snapshot
                    if bad_since is None:
                        bad_since = now
                    maximum_loss_s = max(maximum_loss_s, now - bad_since)
                    if maximum_loss_s > float(
                        self.runtime["contact_loss_grace_s"]
                    ):
                        contact_lost = True
                        self.planner.group.stop()
                        break
            rospy.sleep(0.01)
        thread.join(timeout=3.0)
        if thread.is_alive():
            self.planner.group.stop()
            raise RuntimeError("{} execution thread did not stop".format(label))
        if "exception" in result:
            self.planner.group.stop()
            raise RuntimeError("{} raised {}".format(label, result["exception"]))
        if contact_lost:
            self.planner.group.stop()
            object_pose = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            tool_pose = self.planner.group.get_current_pose(
                self.planner.end_effector_link
            ).pose
            hand_state = self.hand.joint_state()
            hand_joints = {
                name: hand_state[name] for name in self.hand.controller_names
            }
            raise RuntimeError(
                "{} lost three-finger contact for {:.3f}s; latest={} "
                "all={} unexpected={} object_xyz={} tool_xyz={} hand_joints={}".format(
                    label,
                    maximum_loss_s,
                    sorted((last_bad_snapshot or {}).get("latest_families", [])),
                    sorted((last_bad_snapshot or {}).get("all_families", [])),
                    sorted((last_bad_snapshot or {}).get("unexpected", [])),
                    [
                        object_pose.position.x,
                        object_pose.position.y,
                        object_pose.position.z,
                    ],
                    [tool_pose.position.x, tool_pose.position.y, tool_pose.position.z],
                    hand_joints,
                )
            )
        if not result.get("success"):
            self.planner.group.stop()
            raise RuntimeError("{} execution returned false".format(label))

        # A FollowJointTrajectory action may report success anywhere inside
        # its configured goal tolerance.  Do not immediately send STOP and
        # freeze that residual tracking error.  Keep the final reachable
        # setpoint active and, under the same contact guard, require a tighter
        # measured joint convergence before accepting task-space motion.
        names = list(trajectory.joint_trajectory.joint_names)
        targets = list(trajectory.joint_trajectory.points[-1].positions)
        tolerance = float(self.runtime["post_execution_joint_tolerance_rad"])
        stable_required = float(
            self.runtime["post_execution_settle_stability_s"]
        )
        deadline = time.monotonic() + float(
            self.runtime["post_execution_settle_timeout_s"]
        )
        settled_since = None
        settle_contact_bad_since = None
        final_joint_error = math.inf
        final_joint_errors = {}
        while time.monotonic() < deadline and not rospy.is_shutdown():
            state = self.planner.robot.get_current_state().joint_state
            actual = dict(zip(state.name, state.position))
            if any(name not in actual for name in names):
                self.planner.group.stop()
                raise RuntimeError("{} settle state is incomplete".format(label))
            final_joint_errors = {
                name: abs(actual[name] - target)
                for name, target in zip(names, targets)
            }
            final_joint_error = max(final_joint_errors.values())
            if require_contact:
                snapshot = self.contact_snapshot()
                now = time.monotonic()
                latest = snapshot["latest_monotonic"]
                fresh = (
                    latest is not None
                    and 0.0 <= now - latest
                    <= float(self.runtime["contact_message_timeout_s"])
                )
                contact_good = (
                    fresh
                    and snapshot["latest_families"] == self.required
                    and not snapshot["unexpected"]
                )
                if contact_good:
                    settle_contact_bad_since = None
                else:
                    if settle_contact_bad_since is None:
                        settle_contact_bad_since = now
                    if now - settle_contact_bad_since > float(
                        self.runtime["contact_loss_grace_s"]
                    ):
                        self.planner.group.stop()
                        raise RuntimeError(
                            "{} lost three-finger contact while settling".format(label)
                        )
            now = time.monotonic()
            if final_joint_error <= tolerance:
                if settled_since is None:
                    settled_since = now
                if now - settled_since >= stable_required:
                    break
            else:
                settled_since = None
            rospy.sleep(0.01)
        else:
            self.planner.group.stop()
            maximum_error_joint = (
                max(final_joint_errors, key=final_joint_errors.get)
                if final_joint_errors
                else "unknown"
            )
            raise RuntimeError(
                "{} endpoint did not settle: max_joint_error={:.6f}rad "
                "joint={} errors={}".format(
                    label,
                    final_joint_error,
                    maximum_error_joint,
                    final_joint_errors,
                )
            )
        if require_contact:
            snapshot = self.contact_snapshot()
            if (
                snapshot["latest_families"] != self.required
                or snapshot["unexpected"]
            ):
                raise RuntimeError("{} ended without three-finger contact".format(label))
        return {
            "execution_time_s": time.monotonic() - started,
            "maximum_contact_loss_s": maximum_loss_s,
            "settled_maximum_joint_error_rad": final_joint_error,
            "settle_duration_s": time.monotonic() - settled_since,
        }

    def hold_with_contact(self, duration_s, require_airborne=False):
        started = time.monotonic()
        bad_since = None
        maximum_loss_s = 0.0
        while time.monotonic() - started < duration_s and not rospy.is_shutdown():
            snapshot = self.contact_snapshot()
            now = time.monotonic()
            latest = snapshot["latest_monotonic"]
            good = (
                latest is not None
                and 0.0 <= now - latest
                <= float(self.runtime["contact_message_timeout_s"])
                and snapshot["latest_families"] == self.required
                and not snapshot["unexpected"]
                and (not require_airborne or not snapshot["target_table_support"])
            )
            if good:
                if bad_since is not None:
                    maximum_loss_s = max(maximum_loss_s, now - bad_since)
                bad_since = None
            else:
                if bad_since is None:
                    bad_since = now
                maximum_loss_s = max(maximum_loss_s, now - bad_since)
                if maximum_loss_s > float(self.runtime["contact_loss_grace_s"]):
                    raise RuntimeError(
                        "physical hold lost three-finger contact for {:.3f}s".format(
                            maximum_loss_s
                        )
                    )
            rospy.sleep(0.01)
        return time.monotonic() - started, maximum_loss_s

    def wait_for_airborne(self):
        stable_s = float(
            self.runtime["airborne_no_table_support_stability_s"]
        )
        deadline = time.monotonic() + float(
            self.runtime["airborne_no_table_support_timeout_s"]
        )
        since = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            snapshot = self.contact_snapshot()
            now = time.monotonic()
            latest = snapshot["latest_monotonic"]
            fresh = (
                latest is not None
                and 0.0 <= now - latest
                <= float(self.runtime["contact_message_timeout_s"])
            )
            airborne = (
                fresh
                and not snapshot["target_table_support"]
                and snapshot["latest_families"] == self.required
                and not snapshot["unexpected"]
            )
            if airborne:
                if since is None:
                    since = now
                if now - since >= stable_s:
                    return now - since
            else:
                since = None
            rospy.sleep(0.01)
        raise RuntimeError(
            "object did not establish stable three-finger airborne state"
        )

    def wait_for_table_support(self):
        stable_s = float(self.runtime["table_support_stability_s"])
        deadline = time.monotonic() + float(
            self.runtime["table_support_timeout_s"]
        )
        since = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            snapshot = self.contact_snapshot()
            now = time.monotonic()
            latest = snapshot["latest_monotonic"]
            fresh = (
                latest is not None
                and 0.0 <= now - latest
                <= float(self.runtime["contact_message_timeout_s"])
            )
            if fresh and snapshot["target_table_support"]:
                if since is None:
                    since = now
                if now - since >= stable_s:
                    return now - since
            else:
                since = None
            rospy.sleep(0.01)
        raise RuntimeError("target did not establish stable table support")

    def wait_for_finger_contact_clear(self):
        stable_s = float(self.runtime["release_contact_clear_stability_s"])
        deadline = time.monotonic() + float(
            self.runtime["release_contact_clear_timeout_s"]
        )
        since = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            snapshot = self.contact_snapshot()
            now = time.monotonic()
            latest = snapshot["latest_monotonic"]
            fresh = (
                latest is not None
                and 0.0 <= now - latest
                <= float(self.runtime["contact_message_timeout_s"])
            )
            clear = fresh and not snapshot["latest_families"]
            if clear and snapshot["target_table_support"]:
                if since is None:
                    since = now
                if now - since >= stable_s:
                    return now - since
            else:
                since = None
            rospy.sleep(0.01)
        raise RuntimeError(
            "finger contacts did not clear while target remained table-supported"
        )

    def actual_table_clearance(self):
        T_world_tool = pose_matrix(
            self.planner.group.get_current_pose(
                self.planner.end_effector_link
            ).pose
        )
        geometry = self.planner.geometry
        T_tool_hand = geometry.T_tool_grasp_center @ np.linalg.inv(
            geometry.T_hand_grasp_center
        )
        return geometry.minimum_table_clearance(
            T_world_tool @ T_tool_hand, self.planner.table_z
        )

    def remove_target_proxy(self):
        self.planner.scene.remove_world_object(self.planner.target_name)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if self.planner.target_name not in self.planner.scene.get_known_object_names():
                return
            rospy.sleep(0.05)
        raise RuntimeError("target planning proxy was not removed")

    def make_translation_target(self, source, vector):
        target = copy.deepcopy(source)
        target.position.x += float(vector[0])
        target.position.y += float(vector[1])
        target.position.z += float(vector[2])
        return target

    def apply_contact_preload(self):
        """Apply a bounded flexion-only preload without changing palm shape."""
        current = self.hand.joint_state()
        deltas = self.runtime["lift_preload_flexion_delta_rad"]
        flexion_names = set(self.hand.config["execution"]["flexion_joint_names"])
        if set(deltas) != flexion_names:
            raise RuntimeError("preload deltas must cover flexion joints exactly")
        targets = dict(current)
        close_targets = dict(
            zip(
                self.hand.names,
                self.hand.config["commands"]["CLOSE"]["positions"],
            )
        )
        for name in flexion_names:
            delta = float(deltas[name])
            if not math.isfinite(delta) or delta <= 0.0:
                raise RuntimeError("preload flexion delta must be positive")
            targets[name] = min(current[name] + delta, close_targets[name])
        for name in self.hand.config["execution"]["configuration_joint_names"]:
            targets[name] = current[name]

        duration_s = float(self.runtime["lift_preload_duration_s"])
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = list(self.hand.controller_names)
        point = JointTrajectoryPoint()
        point.positions = [targets[name] for name in self.hand.controller_names]
        point.time_from_start = rospy.Duration(duration_s)
        goal.trajectory.points = [point]
        goal.path_tolerance = [
            JointTolerance(
                name=name,
                position=float(self.hand.config["execution"]["path_tolerance_rad"]),
            )
            for name in self.hand.controller_names
        ]
        goal.goal_tolerance = [
            JointTolerance(
                name=name,
                position=float(self.runtime["lift_preload_goal_tolerance_rad"]),
            )
            for name in self.hand.controller_names
        ]
        goal.goal_time_tolerance = rospy.Duration(1.0)
        self.hand.client.send_goal(goal)
        if not self.hand.client.wait_for_result(rospy.Duration(duration_s + 2.0)):
            self.hand.client.cancel_goal()
            raise RuntimeError("contact preload trajectory timeout")
        result = self.hand.client.get_result()
        if result is None or result.error_code != result.SUCCESSFUL:
            code = None if result is None else int(result.error_code)
            raise RuntimeError("contact preload trajectory failed {}".format(code))
        actual = self.hand.joint_state()
        palm_names = self.hand.config["execution"]["configuration_joint_names"]
        palm_error = max(abs(actual[name] - current[name]) for name in palm_names)
        if palm_error > float(self.runtime["lift_preload_palm_tolerance_rad"]):
            raise RuntimeError("contact preload changed palm configuration")
        return {
            "start_joint_positions": current,
            "target_joint_positions": targets,
            "actual_joint_positions": actual,
            "maximum_palm_change_rad": palm_error,
        }

    def run_pick_place(self):
        contact_record, contact_path = super().run()
        candidate = self.planner.selected_candidate
        object_initial = copy.deepcopy(
            self.planner.get_model_state(self.planner.target_name, "world").pose
        )
        # Use the pre-grasp live pose as the table reference, not the slightly
        # displaced contact pose.
        initial_matrix = np.asarray(
            self.planner.plan_only_record["T_world_object"], dtype=float
        )
        object_table_reference = Pose()
        object_table_reference.position.x = initial_matrix[0, 3]
        object_table_reference.position.y = initial_matrix[1, 3]
        object_table_reference.position.z = initial_matrix[2, 3]
        object_table_reference.orientation.w = 1.0
        states = ["CONTACT_ONLY_GATE_PASS"]
        record = {
            "schema_version": 1,
            "mode": "three_finger_physical_pick_place",
            "contact_result": contact_path,
            "success": False,
            "attachment_used": False,
            "states": states,
            "failure_reason": "",
            "selected_candidate": candidate.as_dict(),
            "object_pose_before_lift": pose_as_dict(object_initial),
        }
        target_proxy_removed = False
        try:
            states.append("VERIFY_CONTACT_TABLE_CLEARANCE")
            record["contact_actual_table_clearance_m"] = (
                self.actual_table_clearance()
            )
            minimum_clearance = float(
                self.planner.geometry_config["contact_geometry"]
                ["minimum_table_clearance_m"]
            )
            if record["contact_actual_table_clearance_m"] < minimum_clearance:
                raise RuntimeError("hand/table clearance failed before lift")

            states.append("REMOVE_MOVEIT_TARGET_PROXY_ONLY")
            self.remove_target_proxy()
            target_proxy_removed = True
            record["planning_scene_target_removed_for_lift"] = True
            start_validity = self.planner.check_state(
                GetStateValidityRequest(
                    robot_state=self.planner.robot.get_current_state(),
                    group_name=self.planner.group_name,
                )
            )
            record["lift_start_state_valid"] = bool(start_validity.valid)
            if not start_validity.valid:
                raise RuntimeError("robot state is in collision before lift")

            states.append("LIFT_WITH_THREE_FINGER_GUARD")
            tool_before_lift = self.planner.group.get_current_pose(
                self.planner.end_effector_link
            ).pose
            object_before_lift = copy.deepcopy(
                self.planner.get_model_state(
                    self.planner.target_name, "world"
                ).pose
            )
            lift_vector = opposite_approach_lift_vector(
                candidate.T_world_hand,
                self.runtime["lift_distance_along_opposite_approach_m"],
            )
            record["lift_vector_world_m"] = lift_vector.tolist()
            record["planned_lift_vector_quality"] = validate_planned_lift_vector(
                lift_vector, self.planner.geometry_config
            )
            T_world_tool_before_lift = pose_matrix(tool_before_lift)
            T_tool_hand = (
                self.planner.geometry.T_tool_grasp_center
                @ np.linalg.inv(self.planner.geometry.T_hand_grasp_center)
            )
            lift_tilt_deg = float(
                self.runtime.get("lift_hand_local_y_tilt_deg", 0.0)
            )
            record["lift_hand_local_y_tilt_deg"] = lift_tilt_deg
            T_world_tool_lift_target = pivoted_grasp_center_lift_target(
                T_world_tool_before_lift,
                T_tool_hand,
                self.planner.geometry.T_hand_grasp_center,
                lift_vector,
                lift_tilt_deg,
            )
            lift_target = matrix_pose(T_world_tool_lift_target)
            record["lift_tool_target_pose"] = pose_as_dict(lift_target)
            lift_trajectory, lift_fraction = self.plan_cartesian(
                lift_target,
                "lift",
                float(self.runtime["minimum_lift_place_duration_s"]),
            )
            record["lift_fraction"] = lift_fraction
            record["lift_trajectory_points"] = len(
                lift_trajectory.joint_trajectory.points
            )
            record["lift_trajectory_duration_s"] = trajectory_duration_s(
                lift_trajectory
            )
            states.append("APPLY_BOUNDED_FLEXION_PRELOAD")
            object_before_preload = copy.deepcopy(
                self.planner.get_model_state(self.planner.target_name, "world").pose
            )
            record["preload"] = self.apply_contact_preload()
            object_after_preload = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["preload_object_displacement_m"] = position_distance(
                object_before_preload.position, object_after_preload.position
            )
            if record["preload_object_displacement_m"] > float(
                self.runtime["maximum_grasp_object_displacement_m"]
            ):
                raise RuntimeError("contact preload displaced the object")
            self.reset_contacts()
            self.wait_for_three_finger_contact()
            states.append("LIFT_WITH_BOUNDED_PRELOAD")
            lift_execution = self.execute_trajectory(
                lift_trajectory, "lift", True
            )
            record.update(
                {
                    "lift_{}".format(key): value
                    for key, value in lift_execution.items()
                }
            )
            tool_after_lift = self.planner.group.get_current_pose(
                self.planner.end_effector_link
            ).pose
            object_after_lift = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["object_pose_after_lift"] = pose_as_dict(object_after_lift)
            record["object_lift_m"] = (
                object_after_lift.position.z - object_before_lift.position.z
            )
            tool_delta = np.array(
                [
                    tool_after_lift.position.x - tool_before_lift.position.x,
                    tool_after_lift.position.y - tool_before_lift.position.y,
                    tool_after_lift.position.z - tool_before_lift.position.z,
                ]
            )
            grasp_center_before_lift = grasp_center_position_from_tool(
                T_world_tool_before_lift,
                T_tool_hand,
                self.planner.geometry.T_hand_grasp_center,
            )
            grasp_center_after_lift = grasp_center_position_from_tool(
                pose_matrix(tool_after_lift),
                T_tool_hand,
                self.planner.geometry.T_hand_grasp_center,
            )
            grasp_center_delta = (
                grasp_center_after_lift - grasp_center_before_lift
            )
            object_delta = np.array(
                [
                    object_after_lift.position.x - object_before_lift.position.x,
                    object_after_lift.position.y - object_before_lift.position.y,
                    object_after_lift.position.z - object_before_lift.position.z,
                ]
            )
            record["tool_lift_delta_m"] = tool_delta.tolist()
            record["grasp_center_lift_delta_m"] = grasp_center_delta.tolist()
            record["object_lift_delta_m"] = object_delta.tolist()
            record["object_tool_lift_disagreement_m"] = float(
                np.linalg.norm(tool_delta - object_delta)
            )
            record["object_grasp_center_lift_disagreement_m"] = float(
                np.linalg.norm(grasp_center_delta - object_delta)
            )
            lift_evidence = evaluate_actual_lift_evidence(
                object_delta, self.planner.geometry_config
            )
            record["actual_object_lift_evidence"] = lift_evidence
            record["actual_object_lift_world_z_fraction"] = lift_evidence[
                "world_z_fraction"
            ]
            if not lift_evidence["passed"]:
                raise RuntimeError(
                    "physical object lift evidence failed: {}".format(
                        ",".join(lift_evidence["failures"])
                    )
                )
            if record["object_lift_m"] < float(
                self.runtime["minimum_object_lift_m"]
            ):
                raise RuntimeError("object did not reach minimum physical lift")
            maximum_grasp_center_disagreement = float(
                self.runtime.get(
                    "maximum_object_grasp_center_disagreement_m",
                    self.runtime["maximum_object_tool_disagreement_m"],
                )
            )
            if record["object_grasp_center_lift_disagreement_m"] > (
                maximum_grasp_center_disagreement
            ):
                raise RuntimeError("object did not physically follow the hand")
            record["airborne_no_table_support_stability_s"] = (
                self.wait_for_airborne()
            )
            record["lift_executed"] = True

            states.append("PHYSICAL_HOLD")
            hold_s, hold_loss_s = self.hold_with_contact(
                float(self.runtime["physical_hold_duration_s"]),
                require_airborne=True,
            )
            record["physical_hold_duration_s"] = hold_s
            record["physical_hold_maximum_contact_loss_s"] = hold_loss_s
            held_pose = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["object_hold_lift_m"] = (
                held_pose.position.z - object_before_lift.position.z
            )
            if record["object_hold_lift_m"] < float(
                self.runtime["minimum_object_lift_m"]
            ):
                raise RuntimeError("object was not physically held above table")

            states.append("PLACE_WITH_THREE_FINGER_GUARD")
            place_trajectory, place_fraction = self.plan_cartesian(
                tool_before_lift,
                "place",
                float(self.runtime["minimum_lift_place_duration_s"]),
            )
            record["place_fraction"] = place_fraction
            record["place_trajectory_points"] = len(
                place_trajectory.joint_trajectory.points
            )
            record["place_trajectory_duration_s"] = trajectory_duration_s(
                place_trajectory
            )
            record.update(
                {
                    "place_{}".format(key): value
                    for key, value in self.execute_trajectory(
                        place_trajectory, "place", True
                    ).items()
                }
            )
            placed_pose = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["object_pose_before_release"] = pose_as_dict(placed_pose)
            record["object_place_error_m"] = position_distance(
                placed_pose.position, object_table_reference.position
            )
            if record["object_place_error_m"] > float(
                self.runtime["object_place_tolerance_m"]
            ):
                raise RuntimeError("object did not return to its table pose")
            record["place_actual_table_clearance_m"] = (
                self.actual_table_clearance()
            )
            if record["place_actual_table_clearance_m"] < minimum_clearance:
                raise RuntimeError("hand/table clearance failed before release")
            record["table_support_stability_s"] = self.wait_for_table_support()

            states.append("RELEASE_ON_SUPPORTED_TABLE")
            released = self.hand.command("RELEASE")
            record["release_result"] = released
            if not released["success"]:
                raise RuntimeError(
                    "RELEASE failed: {}".format(released["failure_reason"])
                )

            # Generate the retreat from the post-release live arm state.  The
            # hand command takes several seconds and contact unloading can
            # move arm joints slightly; planning before RELEASE therefore
            # creates a stale trajectory start.  The target proxy remains
            # removed, and this Cartesian segment is strictly upward and away
            # from the table-supported object.
            states.append("PLAN_OPEN_HAND_RETREAT")
            tool_before_retreat = self.planner.group.get_current_pose(
                self.planner.end_effector_link
            ).pose
            approach_axis = candidate.T_world_hand[:3, 2]
            retreat_vector = -float(self.runtime["retreat_distance_m"]) * approach_axis
            retreat_target = self.make_translation_target(
                tool_before_retreat, retreat_vector
            )
            retreat_trajectory, retreat_fraction = self.plan_cartesian(
                retreat_target,
                "release retreat",
                float(self.runtime["minimum_retreat_duration_s"]),
            )
            record["retreat_fraction"] = retreat_fraction
            record["retreat_trajectory_points"] = len(
                retreat_trajectory.joint_trajectory.points
            )
            record["retreat_trajectory_duration_s"] = trajectory_duration_s(
                retreat_trajectory
            )

            states.append("RETREAT_OPEN_HAND")
            record.update(
                {
                    "retreat_{}".format(key): value
                    for key, value in self.execute_trajectory(
                        retreat_trajectory, "release retreat", False
                    ).items()
                }
            )
            record["retreat_actual_table_clearance_m"] = (
                self.actual_table_clearance()
            )
            if record["retreat_actual_table_clearance_m"] < minimum_clearance:
                raise RuntimeError("hand/table clearance failed after retreat")
            record["release_contact_clear_stability_s"] = (
                self.wait_for_finger_contact_clear()
            )
            released_pose = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["object_pose_after_release"] = pose_as_dict(released_pose)
            record["release_object_displacement_m"] = position_distance(
                placed_pose.position, released_pose.position
            )
            if record["release_object_displacement_m"] > float(
                self.runtime["object_place_tolerance_m"]
            ):
                raise RuntimeError("release and retreat displaced the supported object")

            states.append("RESTORE_EXACT_TARGET_AFTER_CLEAR_RETREAT")
            sync_position, sync_orientation = (
                self.planner.synchronize_exact_object(released_pose)
            )
            target_proxy_removed = False
            record["release_target_proxy_sync_position_m"] = sync_position
            record["release_target_proxy_sync_orientation_deg"] = sync_orientation
            rospy.sleep(float(self.runtime["release_settle_duration_s"]))
            final_pose = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["object_pose_final"] = pose_as_dict(final_pose)
            record["object_final_table_error_m"] = position_distance(
                final_pose.position, object_table_reference.position
            )
            if record["object_final_table_error_m"] > float(
                self.runtime["object_place_tolerance_m"]
            ):
                raise RuntimeError("released object did not remain on table")
            record["final_table_support_stability_s"] = (
                self.wait_for_table_support()
            )
            states.append("THREE_FINGER_PICK_PLACE_PASS")
            record["place_executed"] = True
            record["release_executed"] = True
            record["success"] = True
        except Exception as exc:
            self.planner.group.stop()
            record["failure_reason"] = str(exc)
            states.append("FAILED")
            # Never open a possibly suspended object. Release is attempted
            # only when fresh target/table support is already present.
            try:
                snapshot = self.contact_snapshot()
                if snapshot["target_table_support"]:
                    self.hand.command("RELEASE")
                    self.hand.command("STOP")
                    record["fail_safe_release_on_supported_table"] = True
                else:
                    self.hand.command("STOP")
                    record["fail_safe_release_on_supported_table"] = False
            except Exception as cleanup_exc:
                record["cleanup_failure"] = str(cleanup_exc)
            if target_proxy_removed:
                record["planning_scene_target_left_removed_after_failure"] = True
        record["states"] = states
        os.makedirs(self.planner.results_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        path = os.path.join(
            self.planner.results_dir,
            "three_finger_pick_place_{}.json".format(stamp),
        )
        with open(path, "x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
        self.planner.status.publish(json.dumps(record, sort_keys=True))
        if not record["success"]:
            raise RuntimeError(record["failure_reason"])
        rospy.loginfo(
            "[three-finger-pick-place] PASS physical lift/place/release; results=%s",
            path,
        )
        return record, path


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("three_finger_pick_place_demo")
    try:
        demo = ThreeFingerPickPlaceDemo()
        demo.run_pick_place()
        rospy.loginfo(
            "[three-finger-pick-place] Final released scene held for observation; Ctrl-C to exit."
        )
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("Three-finger pick/place failed: %s", exc)
        raise SystemExit(8)
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
