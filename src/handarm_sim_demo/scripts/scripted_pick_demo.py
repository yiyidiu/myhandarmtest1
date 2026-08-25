#!/usr/bin/env python3
"""Known-pose physical grasp demo and isolated nonphysical legacy demo."""

import csv
import copy
import datetime
import json
import math
import os
import re
import sys
import threading
import time

import actionlib
import moveit_commander
import rospkg
import rospy
import yaml
from controller_manager_msgs.srv import ListControllers
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from moveit_msgs.msg import AllowedCollisionEntry, MoveGroupAction, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import (
    ApplyPlanningScene,
    ApplyPlanningSceneRequest,
    GetPlanningScene,
    GetPlanningSceneRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

PACKAGE_PATH = rospkg.RosPack().get_path("handarm_sim_demo")
sys.path.insert(0, os.path.join(PACKAGE_PATH, "scripts"))

from hand_commander import HandCommander
from multi_angle_avoidance_course import nearest_equivalent_within_bounds


EXPECTED_CONTROLLERS = {"controller_gazebo", "controller_gazebo_hand"}
CSV_FIELDS = [
    "run_id", "trial", "grasp_mode", "state", "success", "states",
    "object_pose_source", "target_object", "target_was_world_object",
    "target_touch_collision_links", "target_touch_acm_applied",
    "pregrasp_plan_success", "pregrasp_trajectory_points",
    "pregrasp_planning_time_s", "pregrasp_execution_time_s",
    "pregrasp_execution_success", "pregrasp_position_error_m",
    "pregrasp_orientation_error_deg", "approach_fraction",
    "pregrasp_raw_ik_joint_target", "pregrasp_joint_target",
    "pregrasp_max_abs_joint_delta_rad",
    "pregrasp_refinement_fraction", "pregrasp_refinement_trajectory_points",
    "pregrasp_refinement_trajectory_duration_s",
    "pregrasp_refinement_collision_free",
    "pregrasp_refinement_execution_time_s",
    "pregrasp_refinement_execution_success",
    "pregrasp_refinement_position_error_m",
    "pregrasp_refinement_orientation_error_deg",
    "approach_trajectory_points", "approach_trajectory_duration_s",
    "approach_collision_free",
    "approach_execution_time_s", "approach_execution_success",
    "approach_position_error_m", "approach_orientation_error_deg",
    "object_before_approach_xyz_m", "object_after_approach_xyz_m",
    "object_approach_displacement_m", "object_after_grasp_xyz_m",
    "object_grasp_displacement_m",
    "object_before_pregrasp_xyz_m", "object_pregrasp_displacement_m",
    "exact_target_proxy_restored",
    "hand_open_success", "hand_close_success",
    "hand_close_joint_verification_success", "hand_target_joint_positions",
    "hand_actual_joint_positions", "hand_shape_mode", "hand_hold_success",
    "contact_sensor_available", "grasp_contact_finger_families",
    "grasp_contact_pairs", "grasp_contact_duration_s",
    "grasp_multifinger_contact_success",
    "grasp_configuration_hold_success", "grasp_configuration_max_error_rad",
    "contact_continuity_success", "contact_max_loss_s",
    "attachment_used", "attachment_type", "attachment_pose_jump_m",
    "lift_start_state_valid", "lift_start_contacts",
    "lift_fraction", "lift_trajectory_points", "lift_trajectory_duration_s",
    "lift_collision_free",
    "lift_execution_time_s", "lift_execution_success", "lift_position_error_m",
    "object_initial_z_m", "object_final_z_m", "object_lift_m",
    "tool_lift_z_m",
    "object_tool_lift_disagreement_m", "physical_grasp_claimed",
    "planning_scene_target_removed_for_lift", "physical_hold_duration_s",
    "object_hold_z_m", "object_hold_drop_m", "physical_hold_success",
    "place_fraction", "place_trajectory_points", "place_trajectory_duration_s",
    "place_collision_free",
    "place_execution_time_s", "place_execution_success",
    "place_position_error_m", "object_place_error_m", "object_place_z_m",
    "object_placed_before_release_success",
    "hand_release_after_lift_success", "object_released_z_m",
    "release_attempt_count", "release_attempt_failures",
    "release_attempt_results", "release_joint_diagnostics",
    "release_contact_clear_success", "release_contact_clear_duration_s",
    "release_pre_retreat_object_displacement_m",
    "release_pre_retreat_initial_error_m",
    "release_target_proxy_restored_for_retreat",
    "release_acm_restored_for_retreat",
    "release_retreat_fraction", "release_retreat_trajectory_points",
    "release_retreat_trajectory_duration_s", "release_retreat_collision_free",
    "release_retreat_execution_time_s", "release_retreat_execution_success",
    "release_retreat_position_error_m", "release_retreat_orientation_error_deg",
    "release_post_retreat_object_displacement_m",
    "release_post_retreat_initial_error_m",
    "release_post_retreat_contact_clear",
    "release_settle_displacement_m", "release_on_table_success",
    "task_execution_time_s", "failure_reason",
]


FINGER_LINK_PATTERN = re.compile(r"(?:^|::)f([123])link[123](?:::|$)")


def set_acm_pairs(acm, first_name, other_names, allowed):
    """Set selected symmetric ACM pairs while preserving every other pair."""
    names = list(acm.entry_names)
    for name in [first_name] + list(other_names):
        if name in names:
            continue
        names.append(name)
        for entry in acm.entry_values:
            entry.enabled.append(False)
        acm.entry_values.append(AllowedCollisionEntry(enabled=[False] * len(names)))
    acm.entry_names = names
    size = len(names)
    if len(acm.entry_values) != size or any(
        len(entry.enabled) != size for entry in acm.entry_values
    ):
        raise ValueError("allowed collision matrix is not square")
    first_index = names.index(first_name)
    for name in other_names:
        other_index = names.index(name)
        acm.entry_values[first_index].enabled[other_index] = bool(allowed)
        acm.entry_values[other_index].enabled[first_index] = bool(allowed)
    return acm


def classify_target_contact_pairs(states, target_model="target_object"):
    """Return physical finger families and collision pairs touching target.

    Table/object support contact and any arm/palm collision are deliberately
    excluded.  A family is one fixed-preshape finger (f1, f2 or f3), not one
    collision shape, so several shapes on one finger cannot fake a
    multi-finger grasp.
    """
    families = set()
    pairs = set()
    target_prefix = target_model + "::"
    for state in states:
        first = str(state.collision1_name)
        second = str(state.collision2_name)
        if target_prefix not in first and target_prefix not in second:
            continue
        other = second if target_prefix in first else first
        match = FINGER_LINK_PATTERN.search(other)
        if match is None:
            continue
        families.add("f{}".format(match.group(1)))
        pairs.add("{} <-> {}".format(first, second))
    return families, pairs


def point_json(point):
    return json.dumps([point.x, point.y, point.z])


def quaternion_norm(q):
    return math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)


def normalize_quaternion(q):
    norm = quaternion_norm(q)
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("invalid quaternion")
    return Quaternion(x=q.x / norm, y=q.y / norm, z=q.z / norm, w=q.w / norm)


def quaternion_multiply(a, b):
    return normalize_quaternion(
        Quaternion(
            x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        )
    )


def rotate_vector(q, vector):
    q = normalize_quaternion(q)
    x, y, z = vector
    # Unit-quaternion rotation matrix.
    return [
        (1 - 2 * (q.y * q.y + q.z * q.z)) * x
        + 2 * (q.x * q.y - q.z * q.w) * y
        + 2 * (q.x * q.z + q.y * q.w) * z,
        2 * (q.x * q.y + q.z * q.w) * x
        + (1 - 2 * (q.x * q.x + q.z * q.z)) * y
        + 2 * (q.y * q.z - q.x * q.w) * z,
        2 * (q.x * q.z - q.y * q.w) * x
        + 2 * (q.y * q.z + q.x * q.w) * y
        + (1 - 2 * (q.x * q.x + q.y * q.y)) * z,
    ]


def rpy_quaternion(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return normalize_quaternion(
        Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy,
        )
    )


def compose_pregrasp(object_spec, relative_spec):
    object_q = rpy_quaternion(object_spec["pose"]["orientation_rpy"])
    values = relative_spec["orientation_xyzw"]
    relative_q = normalize_quaternion(
        Quaternion(x=values[0], y=values[1], z=values[2], w=values[3])
    )
    offset = rotate_vector(object_q, relative_spec["position_m"])
    pose = Pose()
    object_position = object_spec["pose"]["position"]
    pose.position.x = object_position[0] + offset[0]
    pose.position.y = object_position[1] + offset[1]
    pose.position.z = object_position[2] + offset[2]
    pose.orientation = quaternion_multiply(object_q, relative_q)
    return pose


def quaternion_angle_deg(a, b):
    na, nb = normalize_quaternion(a), normalize_quaternion(b)
    dot = abs(na.x * nb.x + na.y * nb.y + na.z * nb.z + na.w * nb.w)
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def pose_errors(actual, target):
    position = math.sqrt(
        (actual.position.x - target.position.x) ** 2
        + (actual.position.y - target.position.y) ** 2
        + (actual.position.z - target.position.z) ** 2
    )
    return position, quaternion_angle_deg(actual.orientation, target.orientation)


def position_distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    )


def validate_grasp_displacement(displacement_m, maximum_m):
    """Validate measured closing displacement against a fail-closed limit."""
    if not math.isfinite(maximum_m) or maximum_m < 0.0:
        raise ValueError(
            "maximum grasp displacement limit must be finite and non-negative"
        )
    return (
        math.isfinite(displacement_m)
        and displacement_m >= 0.0
        and displacement_m <= maximum_m
    )


def is_contact_obstruction_candidate(command_result):
    """Identify only joint-verification failures that contact may explain.

    The caller must still enforce palm-configuration, object-displacement and
    independent multi-finger-contact gates before accepting the grasp.
    """
    return (
        not bool(command_result.get("success"))
        and command_result.get("failure_reason")
        in {
            "active joint verification failed",
            "mimic joint relation verification failed",
        }
    )


def release_joint_diagnostics(command_result):
    """Select JSON-safe joint evidence from a RELEASE command result."""
    keys = (
        "target_joint_positions",
        "actual_joint_positions",
        "active_joint_errors_rad",
        "mimic_joint_errors_rad",
        "mimic_relation_pass",
        "mimic_stability_pass",
        "failure_diagnostics",
        "error_code",
        "error_string",
        "success",
        "failure_reason",
    )
    return {key: command_result.get(key) for key in keys}


def evaluate_contact_clear_sample(now_s, latest_s, families, message_timeout_s):
    """A contact-clear sample is valid only when its empty sample is fresh."""
    if (
        latest_s is None
        or not math.isfinite(now_s)
        or not math.isfinite(latest_s)
        or not math.isfinite(message_timeout_s)
        or message_timeout_s <= 0.0
    ):
        return False, False
    fresh = 0.0 <= now_s - latest_s <= message_timeout_s
    return fresh, fresh and not (families or set())


def validate_release_acceptance_config(config):
    acceptance = config.get("physical_acceptance")
    if acceptance is None:
        return True
    for key in (
        "release_contact_clear_duration_s",
        "release_contact_clear_timeout_s",
        "release_settle_duration_s",
        "release_settle_tolerance_m",
    ):
        value = acceptance.get(key)
        if value is None:
            raise ValueError("physical_acceptance is missing {}".format(key))
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                "physical_acceptance {} must be finite and positive".format(key)
            )
    if int(acceptance.get("release_attempts", 0)) != 1:
        raise ValueError("physical release acceptance requires exactly one attempt")
    return True


def trajectory_duration_s(trajectory):
    points = trajectory.joint_trajectory.points
    return points[-1].time_from_start.to_sec() if points else 0.0


def enforce_minimum_trajectory_duration(trajectory, minimum_s):
    """Uniformly stretch all waypoint times, velocities and accelerations."""
    if not math.isfinite(minimum_s) or minimum_s <= 0.0:
        raise ValueError("minimum trajectory duration must be finite and positive")
    points = trajectory.joint_trajectory.points
    if not points:
        raise ValueError("cannot enforce duration on an empty trajectory")
    original_s = _duration_seconds(points[-1].time_from_start)
    if not math.isfinite(original_s) or original_s <= 0.0:
        raise ValueError("trajectory final duration must be finite and positive")
    if original_s >= minimum_s:
        return trajectory
    scale = minimum_s / original_s
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("trajectory duration scale is not finite and positive")
    acceleration_scale = scale * scale
    if not math.isfinite(acceleration_scale):
        raise ValueError("trajectory acceleration scale is non-finite")
    for point in points:
        scaled_s = _duration_seconds(point.time_from_start) * scale
        if not math.isfinite(scaled_s):
            raise ValueError("scaled trajectory time is non-finite")
        point.time_from_start = _duration_from_seconds(
            point.time_from_start, scaled_s
        )
        if point.velocities:
            scaled_velocities = [value / scale for value in point.velocities]
            if not all(math.isfinite(value) for value in scaled_velocities):
                raise ValueError("scaled trajectory velocity is non-finite")
            point.velocities = scaled_velocities
        if point.accelerations:
            scaled_accelerations = [
                value / acceleration_scale for value in point.accelerations
            ]
            if not all(math.isfinite(value) for value in scaled_accelerations):
                raise ValueError("scaled trajectory acceleration is non-finite")
            point.accelerations = scaled_accelerations
    return trajectory


def _duration_seconds(value):
    return float(value.to_sec()) if hasattr(value, "to_sec") else float(value)


def _duration_from_seconds(original, seconds):
    if hasattr(original, "from_sec"):
        return original.from_sec(seconds)
    return seconds


def canonicalize_periodic_joint_goal(goal, current, bounds, periodic_names):
    """Choose a bounded, nearest 2*pi representation for periodic joints."""
    result = dict(goal)
    for name, value in result.items():
        if not math.isfinite(value):
            raise ValueError("non-finite joint target for {}".format(name))
        if name not in bounds:
            raise ValueError("missing joint bounds for {}".format(name))
        minimum, maximum = bounds[name]
        if name in periodic_names:
            result[name] = nearest_equivalent_within_bounds(
                value, current[name], minimum, maximum
            )
        elif not minimum <= value <= maximum:
            raise ValueError("joint target for {} is outside bounds".format(name))
    return result


class ScriptedPickDemo:
    def __init__(self):
        self.scene_path = rospy.get_param("~scene_config")
        self.grasp_path = rospy.get_param("~grasp_config")
        self.hand_path = rospy.get_param("~hand_config")
        self.startup_path = rospy.get_param("~startup_config")
        self.grasp_mode = rospy.get_param("~grasp_mode", "approach_only")
        self.allow_nonphysical_attachment = bool(
            rospy.get_param("~allow_nonphysical_attachment", False)
        )
        self.repetitions = int(rospy.get_param("~repetitions", 1))
        with open(self.scene_path, "r", encoding="utf-8") as stream:
            self.scene_config = yaml.safe_load(stream)
        with open(self.grasp_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        validate_release_acceptance_config(self.config)
        with open(self.startup_path, "r", encoding="utf-8") as stream:
            self.startup = yaml.safe_load(stream)["initial_configuration"]
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.grasp_mode == "fixed_attachment_demo_nonphysical":
            if not self.allow_nonphysical_attachment:
                raise ValueError(
                    "fixed_attachment_demo_nonphysical requires "
                    "allow_nonphysical_attachment:=true"
                )
            if self.repetitions != 1:
                raise ValueError(
                    "fixed_attachment_demo_nonphysical requires repetitions=1"
                )
        elif self.allow_nonphysical_attachment:
            raise ValueError(
                "allow_nonphysical_attachment is valid only for "
                "fixed_attachment_demo_nonphysical"
            )
        package = rospkg.RosPack().get_path("handarm_sim_demo")
        self.results_dir = rospy.get_param(
            "~results_dir",
            os.path.abspath(os.path.join(package, "..", "..", "results", "sim_baseline")),
        )
        self.status_pub = rospy.Publisher(
            "/handarm_sim_demo/pick_status", String, queue_size=1, latch=True
        )
        client = actionlib.SimpleActionClient("/move_group", MoveGroupAction)
        rospy.loginfo(
            "[pick] Waiting for MoveIt /move_group (cold GUI start can take 20-30 s)..."
        )
        if not client.wait_for_server(rospy.Duration(90.0)):
            raise RuntimeError("move_group action did not become ready")
        rospy.loginfo("[pick] MoveIt action is available.")
        self.group = moveit_commander.MoveGroupCommander(self.config["planning_group"])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        rospy.wait_for_service("/check_state_validity", timeout=30.0)
        self.check_state = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        rospy.wait_for_service("/get_planning_scene", timeout=30.0)
        self.get_planning_scene = rospy.ServiceProxy(
            "/get_planning_scene", GetPlanningScene
        )
        rospy.wait_for_service("/apply_planning_scene", timeout=30.0)
        self.apply_planning_scene = rospy.ServiceProxy(
            "/apply_planning_scene", ApplyPlanningScene
        )
        rospy.wait_for_service("/gazebo/get_model_state", timeout=30.0)
        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        self.set_attached = None
        if self.grasp_mode == "fixed_attachment_demo_nonphysical":
            service_name = self.config["attachment_service"]
            rospy.loginfo("[pick] Waiting for deterministic attachment service %s...", service_name)
            rospy.wait_for_service(service_name, timeout=30.0)
            self.set_attached = rospy.ServiceProxy(service_name, SetBool)
        self.group.set_end_effector_link(self.config["end_effector_link"])
        self.group.set_planner_id(self.config["planner_id"])
        self.group.set_planning_time(float(self.config["planning_time_s"]))
        self.group.set_num_planning_attempts(int(self.config["planning_attempts"]))
        self.group.set_max_velocity_scaling_factor(float(self.config["velocity_scaling"]))
        self.group.set_max_acceleration_scaling_factor(float(self.config["acceleration_scaling"]))
        self.group.set_goal_position_tolerance(float(self.config["goal_position_tolerance_m"]))
        self.group.set_goal_orientation_tolerance(float(self.config["goal_orientation_tolerance_rad"]))
        self.hand = HandCommander(self.hand_path)
        acceptance = self.config.get("physical_acceptance", {})
        self.contact_topic = acceptance.get(
            "contact_topic", "/handarm_sim_demo/target_contacts"
        )
        self.minimum_finger_families = int(
            acceptance.get("minimum_finger_families", 2)
        )
        self._contact_lock = threading.Lock()
        self._contact_message_seen = False
        self._contact_latest_monotonic = None
        self._contact_latest_families = set()
        self._contact_all_families = set()
        self._contact_all_pairs = set()
        self._contact_qualifying_since = None
        self._contact_best_duration_s = 0.0
        self._contact_guard_active = False
        self._contact_guard_bad_since = None
        self._contact_guard_max_loss_s = 0.0
        self._contact_guard_lost = False
        self.contact_subscriber = rospy.Subscriber(
            self.contact_topic,
            ContactsState,
            self._contact_callback,
            queue_size=10,
        )

    def _contact_callback(self, message):
        now = time.monotonic()
        families, pairs = classify_target_contact_pairs(
            message.states,
            self.scene_config["objects"][self.config["target_object_key"]]["name"],
        )
        with self._contact_lock:
            self._contact_message_seen = True
            self._contact_latest_monotonic = now
            self._contact_latest_families = families
            self._contact_all_families.update(families)
            self._contact_all_pairs.update(pairs)
            if len(families) >= self.minimum_finger_families:
                if self._contact_qualifying_since is None:
                    self._contact_qualifying_since = now
                self._contact_best_duration_s = max(
                    self._contact_best_duration_s,
                    now - self._contact_qualifying_since,
                )
            else:
                self._contact_qualifying_since = None
            if self._contact_guard_active:
                if len(families) >= self.minimum_finger_families:
                    if self._contact_guard_bad_since is not None:
                        self._contact_guard_max_loss_s = max(
                            self._contact_guard_max_loss_s,
                            now - self._contact_guard_bad_since,
                        )
                    self._contact_guard_bad_since = None
                else:
                    if self._contact_guard_bad_since is None:
                        self._contact_guard_bad_since = now
                    loss = now - self._contact_guard_bad_since
                    self._contact_guard_max_loss_s = max(
                        self._contact_guard_max_loss_s, loss
                    )
                    if loss > float(
                        self.config["physical_acceptance"]["contact_loss_grace_s"]
                    ):
                        self._contact_guard_lost = True

    def reset_contact_observation(self):
        with self._contact_lock:
            self._contact_all_families = set()
            self._contact_all_pairs = set()
            self._contact_qualifying_since = None
            self._contact_best_duration_s = 0.0
            self._contact_guard_active = False
            self._contact_guard_bad_since = None
            self._contact_guard_max_loss_s = 0.0
            self._contact_guard_lost = False

    def begin_contact_guard(self):
        with self._contact_lock:
            self._contact_guard_active = True
            self._contact_guard_bad_since = None
            self._contact_guard_max_loss_s = 0.0
            self._contact_guard_lost = False

    def _contact_snapshot(self):
        with self._contact_lock:
            return (
                self._contact_message_seen,
                self._contact_latest_monotonic,
                set(self._contact_latest_families),
            )

    def wait_for_target_finger_contact_clear(self, row):
        """Require a continuous interval of fresh, empty contact samples."""
        acceptance = self.config["physical_acceptance"]
        required_s = float(acceptance["release_contact_clear_duration_s"])
        timeout_s = float(acceptance["release_contact_clear_timeout_s"])
        message_timeout_s = float(acceptance["contact_message_timeout_s"])
        started = time.monotonic()
        clear_since = None
        duration_s = 0.0
        while time.monotonic() - started < timeout_s and not rospy.is_shutdown():
            now = time.monotonic()
            seen, latest, families = self._contact_snapshot()
            fresh, clear = evaluate_contact_clear_sample(
                now, latest, families, message_timeout_s
            )
            if seen and clear:
                if clear_since is None:
                    clear_since = now
                duration_s = now - clear_since
                if duration_s >= required_s:
                    row["release_contact_clear_success"] = True
                    row["release_contact_clear_duration_s"] = duration_s
                    return
            else:
                clear_since = None
                duration_s = 0.0
            time.sleep(0.02)
        _, latest, families = self._contact_snapshot()
        fresh, clear = evaluate_contact_clear_sample(
            time.monotonic(), latest, families, message_timeout_s
        )
        row["release_contact_clear_success"] = False
        row["release_contact_clear_duration_s"] = duration_s
        raise RuntimeError(
            "target/finger contacts did not clear for {:.3f}s: "
            "fresh={}, clear={}, families={}".format(
                required_s, fresh, clear, sorted(families)
            )
        )

    def assert_contact_continuity(self, row, label):
        now = time.monotonic()
        with self._contact_lock:
            max_loss = self._contact_guard_max_loss_s
            if self._contact_guard_bad_since is not None:
                max_loss = max(max_loss, now - self._contact_guard_bad_since)
            lost = self._contact_guard_lost or max_loss > float(
                self.config["physical_acceptance"]["contact_loss_grace_s"]
            )
        row["contact_max_loss_s"] = max_loss
        row["contact_continuity_success"] = not lost
        if lost:
            raise RuntimeError(
                "{} lost multi-finger contact for {:.3f}s".format(label, max_loss)
            )

    def verify_configuration_hold(self, closed, row):
        target = closed.get("target_joint_positions") or {}
        actual = closed.get("actual_joint_positions") or {}
        configured = self.hand.config["execution"]["configuration_joint_names"]
        errors = []
        for name in configured:
            errors.append(abs(actual[name] - target[name]))
            for mimic, relation in self.hand.config["mimic_joints"].items():
                if relation["source"] == name:
                    expected = (
                        target[name] * float(relation["multiplier"])
                        + float(relation["offset"])
                    )
                    errors.append(abs(actual[mimic] - expected))
        maximum = max(errors)
        tolerance = float(
            self.config["physical_acceptance"]["configuration_hold_tolerance_rad"]
        )
        row["grasp_configuration_max_error_rad"] = maximum
        row["grasp_configuration_hold_success"] = maximum <= tolerance
        if not row["grasp_configuration_hold_success"]:
            raise RuntimeError(
                "GRASP changed palm configuration: max error {:.4f}rad".format(
                    maximum
                )
            )

    def verify_multifinger_contact(self, row):
        acceptance = self.config["physical_acceptance"]
        stable_required = float(acceptance["contact_stability_s"])
        message_timeout = float(acceptance["contact_message_timeout_s"])
        deadline = time.monotonic() + float(acceptance["contact_wait_timeout_s"])
        while time.monotonic() < deadline and not rospy.is_shutdown():
            now = time.monotonic()
            with self._contact_lock:
                seen = self._contact_message_seen
                latest = self._contact_latest_monotonic
                families = set(self._contact_latest_families)
                all_families = set(self._contact_all_families)
                pairs = set(self._contact_all_pairs)
                qualifying_since = self._contact_qualifying_since
                best_duration = self._contact_best_duration_s
            fresh = latest is not None and now - latest <= message_timeout
            current_duration = (
                now - qualifying_since
                if fresh and qualifying_since is not None
                else 0.0
            )
            duration = max(best_duration, current_duration)
            if (
                fresh
                and len(families) >= self.minimum_finger_families
                and current_duration >= stable_required
            ):
                row["contact_sensor_available"] = seen
                row["grasp_contact_finger_families"] = json.dumps(
                    sorted(all_families)
                )
                row["grasp_contact_pairs"] = json.dumps(sorted(pairs))
                row["grasp_contact_duration_s"] = duration
                row["grasp_multifinger_contact_success"] = True
                return
            time.sleep(0.02)
        with self._contact_lock:
            seen = self._contact_message_seen
            families = sorted(self._contact_all_families)
            pairs = sorted(self._contact_all_pairs)
            duration = self._contact_best_duration_s
        row["contact_sensor_available"] = seen
        row["grasp_contact_finger_families"] = json.dumps(families)
        row["grasp_contact_pairs"] = json.dumps(pairs)
        row["grasp_contact_duration_s"] = duration
        row["grasp_multifinger_contact_success"] = False
        if not seen:
            raise RuntimeError("target contact sensor produced no messages")
        raise RuntimeError(
            "stable multi-finger contact not achieved: families={}, duration={:.3f}s".format(
                families, duration
            )
        )

    def allow_target_finger_contacts(self, target_object, row):
        """Temporarily allow only configured fingers to contact the target."""
        request = GetPlanningSceneRequest()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        response = self.get_planning_scene(request)
        original = copy.deepcopy(response.scene.allowed_collision_matrix)
        modified = copy.deepcopy(original)
        touch_links = list(self.config["target_touch_links"])
        set_acm_pairs(modified, target_object, touch_links, True)
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = modified
        apply_request = ApplyPlanningSceneRequest()
        apply_request.scene = scene
        if not self.apply_planning_scene(apply_request).success:
            raise RuntimeError("failed to apply target/finger collision allowance")
        row["target_touch_collision_links"] = json.dumps(touch_links)
        row["target_touch_acm_applied"] = True
        return original

    def restore_collision_matrix(self, original):
        """Restore the exact ACM snapshot taken before the physical approach."""
        if original is None:
            return
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = original
        request = ApplyPlanningSceneRequest()
        request.scene = scene
        if not self.apply_planning_scene(request).success:
            raise RuntimeError("failed to restore allowed collision matrix")

    def restore_exact_target_proxy(self, target_object, pose, row):
        """Replace the padded transit proxy with exact grasp geometry."""
        object_spec = self.scene_config["objects"][self.config["target_object_key"]]
        stamped = PoseStamped()
        stamped.header.frame_id = self.scene_config["frame_id"]
        stamped.header.stamp = rospy.Time.now()
        stamped.pose = pose
        self.scene.add_box(target_object, stamped, size=tuple(object_spec["size"]))
        collision = self.scene.get_objects([target_object]).get(target_object)
        if collision is None or len(collision.primitives) != 1:
            raise RuntimeError("exact target planning proxy did not appear")
        dimensions = list(collision.primitives[0].dimensions)
        if max(
            abs(actual - expected)
            for actual, expected in zip(dimensions, object_spec["size"])
        ) > 1.0e-9:
            raise RuntimeError("exact target planning proxy has wrong dimensions")
        row["exact_target_proxy_restored"] = True

    @staticmethod
    def unpack_plan(result):
        if isinstance(result, tuple):
            success, trajectory, planning_time, _ = result
            return bool(success), trajectory, float(planning_time)
        return bool(result.joint_trajectory.points), result, None

    def validate_trajectory(self, trajectory):
        names = trajectory.joint_trajectory.joint_names
        points = trajectory.joint_trajectory.points
        if not names or not points:
            return False, "empty trajectory"
        state = self.robot.get_current_state()
        index = {name: i for i, name in enumerate(state.joint_state.name)}
        if any(name not in index for name in names):
            return False, "trajectory contains unknown joint"
        positions = list(state.joint_state.position)
        previous_time = -1.0
        for point in points:
            seconds = point.time_from_start.to_sec()
            if not math.isfinite(seconds) or seconds <= previous_time:
                return False, "trajectory time is not strictly increasing"
            previous_time = seconds
            if len(point.positions) != len(names) or not all(
                math.isfinite(value) for value in point.positions
            ):
                return False, "trajectory contains invalid joint values"
            for name, value in zip(names, point.positions):
                positions[index[name]] = value
            state.joint_state.position = positions
            response = self.check_state(
                GetStateValidityRequest(robot_state=state, group_name=self.config["planning_group"])
            )
            if not response.valid:
                return False, "trajectory waypoint is in collision"
        return True, ""

    def wait_ready(self):
        rospy.loginfo("[pick] Waiting for startup and synchronized scene...")
        for topic in ("/handarm_sim_demo/startup_ready", "/handarm_sim_demo/scene_ready"):
            if not rospy.wait_for_message(topic, Bool, timeout=90.0).data:
                raise RuntimeError("{} reported false".format(topic))
        rospy.wait_for_service("/controller_manager/list_controllers", timeout=30.0)
        controllers = rospy.ServiceProxy(
            "/controller_manager/list_controllers", ListControllers
        )().controller
        states = {item.name: item.state for item in controllers}
        if not EXPECTED_CONTROLLERS.issubset(states) or any(
            states[name] != "running" for name in EXPECTED_CONTROLLERS
        ):
            raise RuntimeError("trajectory controllers are not running")
        scene_status = json.loads(
            rospy.wait_for_message("/handarm_sim_demo/scene_status", String, timeout=10.0).data
        )
        object_name = self.scene_config["objects"][self.config["target_object_key"]]["name"]
        known = {item["planning_scene_name"] for item in scene_status["objects"]}
        if not scene_status.get("ready") or object_name not in known:
            raise RuntimeError("target is not a synchronized world collision object")
        rospy.loginfo(
            "[pick] READY: target=%s, collision objects=%d.", object_name, len(known)
        )
        return object_name, len(known)

    def home_target(self):
        names = self.startup["arm"]["joint_names"]
        target = list(self.startup["arm"]["positions"])
        current = dict(zip(self.group.get_active_joints(), self.group.get_current_joint_values()))
        for index, name in enumerate(names):
            if name in self.startup.get("wraparound_joints", []):
                target[index] += round((current[name] - target[index]) / (2 * math.pi)) * (2 * math.pi)
        return names, target

    def reset_robot(self):
        names, target = self.home_target()
        current = dict(zip(self.group.get_active_joints(), self.group.get_current_joint_values()))
        if max(abs(current[name] - value) for name, value in zip(names, target)) <= float(
            self.config["home_joint_tolerance_rad"]
        ):
            return True
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(dict(zip(names, target)))
        success, trajectory, _ = self.unpack_plan(self.group.plan())
        valid, _ = self.validate_trajectory(trajectory) if success else (False, "plan failed")
        if not success or not valid:
            return False
        executed = bool(self.group.execute(trajectory, wait=True))
        self.group.stop()
        return executed

    def execute_pregrasp(self, target, row):
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()
        # Resolve IK once from the current state and pass a deterministic joint
        # goal to the planner. Pose-goal sampling may otherwise choose a joint_6
        # representation one full turn away even though the tool pose is equal.
        self.group.set_joint_value_target(
            target, self.config["end_effector_link"], False
        )
        active_names = self.group.get_active_joints()
        raw_target = self.group.get_joint_value_target()
        if not isinstance(raw_target, dict):
            raw_target = dict(zip(active_names, raw_target))
        current = dict(zip(active_names, self.group.get_current_joint_values()))
        bounds = {
            name: tuple(self.robot.get_joint(name).bounds())
            for name in active_names
        }
        joint_target = canonicalize_periodic_joint_goal(
            raw_target,
            current,
            bounds,
            set(self.startup.get("wraparound_joints", [])),
        )
        row["pregrasp_raw_ik_joint_target"] = json.dumps(
            raw_target, sort_keys=True
        )
        row["pregrasp_joint_target"] = json.dumps(joint_target, sort_keys=True)
        row["pregrasp_max_abs_joint_delta_rad"] = max(
            abs(joint_target[name] - current[name]) for name in joint_target
        )
        self.group.set_joint_value_target(joint_target)
        success, trajectory, planning_time = self.unpack_plan(self.group.plan())
        row["pregrasp_plan_success"] = success
        row["pregrasp_planning_time_s"] = planning_time if success else None
        row["pregrasp_trajectory_points"] = len(trajectory.joint_trajectory.points)
        valid, reason = self.validate_trajectory(trajectory) if success else (False, "planning failed")
        if not success or not valid:
            raise RuntimeError("pregrasp {}".format(reason))
        started = time.monotonic()
        row["pregrasp_execution_success"] = bool(self.group.execute(trajectory, wait=True))
        row["pregrasp_execution_time_s"] = time.monotonic() - started
        self.group.stop()
        if not row["pregrasp_execution_success"]:
            raise RuntimeError("pregrasp execution returned false")
        actual = self.group.get_current_pose(self.config["end_effector_link"]).pose
        row["pregrasp_position_error_m"], row["pregrasp_orientation_error_deg"] = pose_errors(actual, target)
        self.verify_pose_errors(row["pregrasp_position_error_m"], row["pregrasp_orientation_error_deg"], "pregrasp")

    def execute_cartesian_segment(
        self, target, row, prefix, label, minimum_duration_s=None
    ):
        trajectory, fraction = self.group.compute_cartesian_path(
            [target], float(self.config["cartesian_eef_step_m"]), True
        )
        row["{}_fraction".format(prefix)] = float(fraction)
        if fraction < float(self.config["cartesian_fraction_min"]):
            raise RuntimeError(
                "{} Cartesian fraction {:.6f} is below threshold".format(
                    label, fraction
                )
            )
        trajectory = self.retime_cartesian_trajectory(trajectory)
        if minimum_duration_s is not None:
            trajectory = enforce_minimum_trajectory_duration(
                trajectory, float(minimum_duration_s)
            )
        row["{}_trajectory_points".format(prefix)] = len(
            trajectory.joint_trajectory.points
        )
        row["{}_trajectory_duration_s".format(prefix)] = trajectory_duration_s(
            trajectory
        )
        valid, reason = self.validate_trajectory(trajectory)
        row["{}_collision_free".format(prefix)] = valid
        if not valid:
            raise RuntimeError("{} {}".format(label, reason))
        started = time.monotonic()
        row["{}_execution_success".format(prefix)] = bool(
            self.group.execute(trajectory, wait=True)
        )
        row["{}_execution_time_s".format(prefix)] = time.monotonic() - started
        self.group.stop()
        if not row["{}_execution_success".format(prefix)]:
            raise RuntimeError("{} execution returned false".format(label))
        actual = self.group.get_current_pose(self.config["end_effector_link"]).pose
        position_error, orientation_error = pose_errors(actual, target)
        row["{}_position_error_m".format(prefix)] = position_error
        row["{}_orientation_error_deg".format(prefix)] = orientation_error
        self.verify_pose_errors(position_error, orientation_error, label)

    def execute_approach(self, target, row):
        self.execute_cartesian_segment(target, row, "approach", "approach")

    def model_pose(self, model_name):
        response = self.get_model_state(model_name, "world")
        if not response.success:
            raise RuntimeError(
                "Gazebo model state failed for {}: {}".format(
                    model_name, response.status_message
                )
            )
        return response.pose

    def retime_cartesian_trajectory(self, trajectory):
        retimed = self.group.retime_trajectory(
            self.robot.get_current_state(),
            trajectory,
            velocity_scaling_factor=float(
                self.config["cartesian_velocity_scaling"]
            ),
            acceleration_scaling_factor=float(
                self.config["cartesian_acceleration_scaling"]
            ),
            algorithm="iterative_time_parameterization",
        )
        if not retimed.joint_trajectory.points:
            raise RuntimeError("Cartesian trajectory retiming returned no points")
        return retimed

    def attach_object(self, target_object, row):
        before = self.model_pose(target_object)
        response = self.set_attached(True)
        if not response.success:
            raise RuntimeError("fixed attachment failed: {}".format(response.message))
        after = self.model_pose(target_object)
        jump = position_distance(before.position, after.position)
        row["attachment_pose_jump_m"] = jump
        if jump > float(self.config["attachment_pose_jump_tolerance_m"]):
            self.set_attached(False)
            raise RuntimeError(
                "attachment moved object by {:.6f} m".format(jump)
            )

        touch_links = set(self.robot.get_link_names(group="hand"))
        touch_links.update([self.config["end_effector_link"], "link_6"])
        self.scene.attach_box(
            self.config["end_effector_link"],
            target_object,
            touch_links=sorted(touch_links),
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if target_object in self.scene.get_attached_objects([target_object]):
                row["attachment_used"] = True
                row["attachment_type"] = "gazebo_fixed_joint_simulation_only"
                row["object_initial_z_m"] = after.position.z
                return after
            time.sleep(0.05)
        self.set_attached(False)
        raise RuntimeError("MoveIt attached collision object did not appear")

    def detach_object(self, target_object):
        try:
            self.scene.remove_attached_object(
                self.config["end_effector_link"], name=target_object
            )
        except Exception as exc:
            rospy.logwarn("[pick] MoveIt detach cleanup failed: %s", exc)
        if self.set_attached is not None:
            try:
                response = self.set_attached(False)
                if not response.success:
                    rospy.logwarn("[pick] Gazebo detach cleanup failed: %s", response.message)
            except rospy.ServiceException as exc:
                rospy.logwarn("[pick] Gazebo detach service failed: %s", exc)

    def execute_lift(self, target_object, object_before, row):
        start_validity = self.check_state(
            GetStateValidityRequest(
                robot_state=self.robot.get_current_state(),
                group_name=self.config["planning_group"],
            )
        )
        row["lift_start_state_valid"] = bool(start_validity.valid)
        row["lift_start_contacts"] = json.dumps(
            sorted(
                {
                    "{} <-> {}".format(
                        contact.contact_body_1, contact.contact_body_2
                    )
                    for contact in start_validity.contacts
                }
            )
        )
        if not start_validity.valid:
            raise RuntimeError(
                "lift start state is in collision: {}".format(
                    row["lift_start_contacts"]
                )
            )
        tool_before = self.group.get_current_pose(
            self.config["end_effector_link"]
        ).pose
        target = Pose()
        target.position.x = tool_before.position.x + self.config["lift_vector_world_m"][0]
        target.position.y = tool_before.position.y + self.config["lift_vector_world_m"][1]
        target.position.z = tool_before.position.z + self.config["lift_vector_world_m"][2]
        target.orientation = tool_before.orientation
        trajectory, fraction = self.group.compute_cartesian_path(
            [target], float(self.config["cartesian_eef_step_m"]), True
        )
        row["lift_fraction"] = float(fraction)
        if fraction < float(self.config["cartesian_fraction_min"]):
            raise RuntimeError(
                "lift Cartesian fraction {:.6f} is below threshold".format(fraction)
            )
        trajectory = self.retime_cartesian_trajectory(trajectory)
        minimum_duration_s = self.config.get(
            "minimum_lift_place_trajectory_duration_s"
        )
        if minimum_duration_s is not None:
            trajectory = enforce_minimum_trajectory_duration(
                trajectory, float(minimum_duration_s)
            )
        row["lift_trajectory_points"] = len(trajectory.joint_trajectory.points)
        row["lift_trajectory_duration_s"] = trajectory_duration_s(trajectory)
        valid, reason = self.validate_trajectory(trajectory)
        row["lift_collision_free"] = valid
        if not valid:
            raise RuntimeError("lift {}".format(reason))
        started = time.monotonic()
        row["lift_execution_success"] = bool(
            self.group.execute(trajectory, wait=True)
        )
        row["lift_execution_time_s"] = time.monotonic() - started
        self.group.stop()
        if not row["lift_execution_success"]:
            raise RuntimeError("lift execution returned false")

        tool_after = self.group.get_current_pose(
            self.config["end_effector_link"]
        ).pose
        object_after = self.model_pose(target_object)
        row["lift_position_error_m"], _ = pose_errors(tool_after, target)
        row["object_final_z_m"] = object_after.position.z
        row["object_lift_m"] = object_after.position.z - object_before.position.z
        tool_delta = [
            tool_after.position.x - tool_before.position.x,
            tool_after.position.y - tool_before.position.y,
            tool_after.position.z - tool_before.position.z,
        ]
        object_delta = [
            object_after.position.x - object_before.position.x,
            object_after.position.y - object_before.position.y,
            object_after.position.z - object_before.position.z,
        ]
        row["object_tool_lift_disagreement_m"] = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(tool_delta, object_delta))
        )
        row["tool_lift_z_m"] = tool_delta[2]
        tool_tolerance = float(self.config["lift_tool_tolerance_m"])
        follow_tolerance = float(self.config["object_follow_tolerance_m"])
        if row["lift_position_error_m"] > tool_tolerance:
            raise RuntimeError("lift end-effector verification failed")
        if row["object_tool_lift_disagreement_m"] > follow_tolerance:
            raise RuntimeError("object did not physically follow the tool")

    def remove_target_from_planning_scene_for_lift(self, target_object):
        """Remove only the MoveIt proxy; the dynamic Gazebo body is untouched."""
        self.scene.remove_world_object(target_object)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if target_object not in self.scene.get_known_object_names():
                return True
            time.sleep(0.05)
        return False

    def verify_physical_hold(self, target_object, row):
        acceptance = self.config["physical_acceptance"]
        minimum_lift = float(acceptance["minimum_object_lift_m"])
        hold_duration = float(acceptance["hold_duration_s"])

        hold_started = time.monotonic()
        hold_reference_z = float(row["object_final_z_m"])
        while (
            time.monotonic() - hold_started < hold_duration
            and not rospy.is_shutdown()
        ):
            time.sleep(0.05)
        held = self.model_pose(target_object)
        row["physical_hold_duration_s"] = time.monotonic() - hold_started
        row["object_hold_z_m"] = held.position.z
        row["object_hold_drop_m"] = hold_reference_z - held.position.z
        row["physical_hold_success"] = (
            held.position.z - float(row["object_initial_z_m"]) >= minimum_lift
        )
        if not row["physical_hold_success"]:
            raise RuntimeError(
                "object was not held above the {:.3f} m lift threshold".format(
                    minimum_lift
                )
            )

    def execute_place(self, target_object, object_initial, row):
        """Return the closed hand along the inverse lift path to the table."""
        tool_before = self.group.get_current_pose(
            self.config["end_effector_link"]
        ).pose
        vector = self.config["lift_vector_world_m"]
        target = Pose()
        target.position.x = tool_before.position.x - vector[0]
        target.position.y = tool_before.position.y - vector[1]
        target.position.z = tool_before.position.z - vector[2]
        target.orientation = tool_before.orientation
        trajectory, fraction = self.group.compute_cartesian_path(
            [target], float(self.config["cartesian_eef_step_m"]), True
        )
        row["place_fraction"] = float(fraction)
        if fraction < float(self.config["cartesian_fraction_min"]):
            raise RuntimeError(
                "place Cartesian fraction {:.6f} is below threshold".format(fraction)
            )
        trajectory = self.retime_cartesian_trajectory(trajectory)
        minimum_duration_s = self.config.get(
            "minimum_lift_place_trajectory_duration_s"
        )
        if minimum_duration_s is not None:
            trajectory = enforce_minimum_trajectory_duration(
                trajectory, float(minimum_duration_s)
            )
        row["place_trajectory_points"] = len(trajectory.joint_trajectory.points)
        row["place_trajectory_duration_s"] = trajectory_duration_s(trajectory)
        valid, reason = self.validate_trajectory(trajectory)
        row["place_collision_free"] = valid
        if not valid:
            raise RuntimeError("place {}".format(reason))
        started = time.monotonic()
        row["place_execution_success"] = bool(
            self.group.execute(trajectory, wait=True)
        )
        row["place_execution_time_s"] = time.monotonic() - started
        self.group.stop()
        if not row["place_execution_success"]:
            raise RuntimeError("place execution returned false")
        tool_after = self.group.get_current_pose(
            self.config["end_effector_link"]
        ).pose
        row["place_position_error_m"], _ = pose_errors(tool_after, target)
        placed = self.model_pose(target_object)
        row["object_place_z_m"] = placed.position.z
        row["object_place_error_m"] = position_distance(
            placed.position, object_initial.position
        )
        tolerance = float(
            self.config["physical_acceptance"]["place_position_tolerance_m"]
        )
        row["object_placed_before_release_success"] = (
            row["place_position_error_m"] <= tolerance
            and row["object_place_error_m"] <= tolerance
        )
        if not row["object_placed_before_release_success"]:
            raise RuntimeError(
                "object did not return to table pose: tool_error={:.4f}m, "
                "object_error={:.4f}m".format(
                    row["place_position_error_m"], row["object_place_error_m"]
                )
            )
        return placed

    def release_on_table(
        self, target_object, placed, object_initial, original_acm, row
    ):
        """Open once, clear contact, restore collision truth, and retreat."""
        acceptance = self.config["physical_acceptance"]
        if int(acceptance["release_attempts"]) != 1:
            raise RuntimeError("physical release must use exactly one attempt")

        released = self.hand.command("RELEASE")
        row["release_attempt_count"] = 1
        diagnostics = release_joint_diagnostics(released)
        row["release_attempt_results"] = json.dumps([diagnostics], sort_keys=True)
        row["release_joint_diagnostics"] = json.dumps(diagnostics, sort_keys=True)
        row["release_attempt_failures"] = json.dumps(
            [] if released["success"] else [released["failure_reason"]]
        )
        row["hand_release_after_lift_success"] = bool(released["success"])
        if not released["success"]:
            raise RuntimeError(
                "RELEASE after lift failed: {}".format(released["failure_reason"])
            )

        # Joint arrival alone is not physical release evidence. Require fresh
        # empty bumper samples before the robot is allowed to retreat.
        self.wait_for_target_finger_contact_clear(row)
        tolerance = float(acceptance["release_settle_tolerance_m"])
        pre_retreat_pose = self.model_pose(target_object)
        pre_retreat_displacement = position_distance(
            placed.position, pre_retreat_pose.position
        )
        pre_retreat_initial_error = position_distance(
            object_initial.position, pre_retreat_pose.position
        )
        row["release_pre_retreat_object_displacement_m"] = (
            pre_retreat_displacement
        )
        row["release_pre_retreat_initial_error_m"] = pre_retreat_initial_error
        if (
            pre_retreat_displacement > tolerance
            or pre_retreat_initial_error > tolerance
        ):
            raise RuntimeError(
                "RELEASE moved object before retreat: placed_error={:.4f}m, "
                "initial_error={:.4f}m".format(
                    pre_retreat_displacement, pre_retreat_initial_error
                )
            )

        # The target was deliberately removed only while it was physically
        # grasped. Restore both exact target geometry and the pre-approach ACM
        # before planning the retreat, so collision validation includes it.
        self.restore_exact_target_proxy(target_object, pre_retreat_pose, row)
        row["release_target_proxy_restored_for_retreat"] = True
        self.restore_collision_matrix(original_acm)
        row["release_acm_restored_for_retreat"] = True

        tool_before = self.group.get_current_pose(
            self.config["end_effector_link"]
        ).pose
        retreat_vector = self.config["approach_vector_world_m"]
        retreat_target = Pose()
        retreat_target.position.x = tool_before.position.x - retreat_vector[0]
        retreat_target.position.y = tool_before.position.y - retreat_vector[1]
        retreat_target.position.z = tool_before.position.z - retreat_vector[2]
        retreat_target.orientation = tool_before.orientation
        self.execute_cartesian_segment(
            retreat_target,
            row,
            "release_retreat",
            "release retreat",
            minimum_duration_s=self.config.get(
                "minimum_lift_place_trajectory_duration_s"
            ),
        )

        time.sleep(float(acceptance["release_settle_duration_s"]))
        _, latest, families = self._contact_snapshot()
        fresh, clear = evaluate_contact_clear_sample(
            time.monotonic(),
            latest,
            families,
            float(acceptance["contact_message_timeout_s"]),
        )
        row["release_post_retreat_contact_clear"] = clear
        if not clear:
            raise RuntimeError(
                "finger contacts returned after release retreat: "
                "fresh={}, families={}".format(fresh, sorted(families))
            )

        released_pose = self.model_pose(target_object)
        row["object_released_z_m"] = released_pose.position.z
        placed_displacement = position_distance(
            placed.position, released_pose.position
        )
        initial_error = position_distance(
            object_initial.position, released_pose.position
        )
        row["release_post_retreat_object_displacement_m"] = placed_displacement
        row["release_post_retreat_initial_error_m"] = initial_error
        row["release_settle_displacement_m"] = placed_displacement
        row["release_on_table_success"] = (
            placed_displacement <= tolerance and initial_error <= tolerance
        )
        if not row["release_on_table_success"]:
            raise RuntimeError(
                "released object moved after retreat: placed_error={:.4f}m, "
                "initial_error={:.4f}m".format(
                    placed_displacement, initial_error
                )
            )

    def verify_pose_errors(self, position_error, orientation_error, label):
        if position_error > float(self.config["verification_position_tolerance_m"]):
            raise RuntimeError("{} position verification failed".format(label))
        if orientation_error > float(self.config["verification_orientation_tolerance_deg"]):
            raise RuntimeError("{} orientation verification failed".format(label))

    def empty_row(self, trial, target_object):
        row = {field: None for field in CSV_FIELDS}
        row.update(
            run_id=self.run_id, trial=trial, grasp_mode=self.grasp_mode,
            state="FAILED", success=False, object_pose_source="config_known_pose",
            target_object=target_object, target_was_world_object=False,
            target_touch_acm_applied=False,
            exact_target_proxy_restored=False,
            pregrasp_plan_success=False, pregrasp_execution_success=False,
            approach_collision_free=False, approach_execution_success=False,
            hand_open_success=None, hand_close_success=None,
            hand_close_joint_verification_success=None,
            hand_shape_mode="grasp_release_only",
            hand_hold_success=False, attachment_used=False,
            contact_sensor_available=False,
            grasp_multifinger_contact_success=False,
            grasp_configuration_hold_success=False,
            contact_continuity_success=False,
            attachment_type="none", lift_execution_success=False,
            lift_start_state_valid=None,
            lift_collision_free=False, physical_grasp_claimed=False,
            planning_scene_target_removed_for_lift=False,
            physical_hold_success=False,
            place_collision_free=False, place_execution_success=False,
            object_placed_before_release_success=False,
            hand_release_after_lift_success=False,
            release_attempt_count=0, release_attempt_failures="[]",
            release_attempt_results="[]",
            release_contact_clear_success=False,
            release_contact_clear_duration_s=0.0,
            release_target_proxy_restored_for_retreat=False,
            release_acm_restored_for_retreat=False,
            release_retreat_collision_free=False,
            release_retreat_execution_success=False,
            release_post_retreat_contact_clear=False,
            release_on_table_success=False,
            failure_reason="",
        )
        return row

    def run_trial(self, trial, target_object):
        states = ["INIT", "WAIT_FOR_ROBOT", "WAIT_FOR_SCENE"]
        row = self.empty_row(trial, target_object)
        started = time.monotonic()
        original_acm = None
        try:
            rospy.loginfo(
                "[pick] Trial %d/%d started: mode=%s, target=%s.",
                trial,
                self.repetitions,
                self.grasp_mode,
                target_object,
            )
            if self.grasp_mode not in self.config["supported_grasp_modes"]:
                if self.grasp_mode == "attachment_demo":
                    status = self.config.get("attachment_demo_status", "NOT_AVAILABLE")
                elif self.grasp_mode == "physical_contact":
                    status = self.config.get("physical_contact_status", "NOT_RUN")
                else:
                    status = "UNSUPPORTED_GRASP_MODE"
                raise RuntimeError("grasp_mode {}: {}".format(self.grasp_mode, status))
            row["target_was_world_object"] = True
            states.append("RESET_ROBOT")
            rospy.loginfo("[pick] Trial %d: resetting robot...", trial)
            if not self.reset_robot():
                raise RuntimeError("robot reset failed")
            if self.grasp_mode == "fixed_attachment_demo_nonphysical":
                states.append("HOLD_INITIAL_HAND")
                rospy.loginfo(
                    "[pick] Trial %d: keeping the initial hand shape unchanged.",
                    trial,
                )
                held = self.hand.command("HOLD")
                row["hand_hold_success"] = bool(held["success"])
                row["hand_target_joint_positions"] = json.dumps(
                    held.get("target_joint_positions"), sort_keys=True
                )
                row["hand_actual_joint_positions"] = json.dumps(
                    held.get("actual_joint_positions"), sort_keys=True
                )
                if not held["success"]:
                    raise RuntimeError(
                        "initial hand HOLD failed: {}".format(
                            held["failure_reason"]
                        )
                    )
            else:
                states.append("RELEASE_HAND")
                rospy.loginfo("[pick] Trial %d: releasing hand...", trial)
                opened = self.hand.command("RELEASE")
                row["hand_open_success"] = bool(opened["success"])
                if not opened["success"]:
                    raise RuntimeError(
                        "RELEASE failed: {}".format(opened["failure_reason"])
                    )
            object_before_pregrasp = self.model_pose(target_object)
            row["object_before_pregrasp_xyz_m"] = point_json(
                object_before_pregrasp.position
            )
            object_spec = self.scene_config["objects"][self.config["target_object_key"]]
            pregrasp = compose_pregrasp(object_spec, self.config["object_to_pregrasp"])
            transit_pregrasp = copy.deepcopy(pregrasp)
            transit_retreat = self.config["transit_retreat_vector_world_m"]
            transit_pregrasp.position.x += transit_retreat[0]
            transit_pregrasp.position.y += transit_retreat[1]
            transit_pregrasp.position.z += transit_retreat[2]
            states.append("PLAN_PREGRASP")
            rospy.loginfo(
                "[pick] Trial %d: planning padded-proxy transit pregrasp...",
                trial,
            )
            self.execute_pregrasp(transit_pregrasp, row)
            states.append("EXECUTE_PREGRASP")
            object_before_approach = self.model_pose(target_object)
            row["object_pregrasp_displacement_m"] = position_distance(
                object_before_pregrasp.position, object_before_approach.position
            )
            if row["object_pregrasp_displacement_m"] > float(
                self.config["maximum_pregrasp_object_displacement_m"]
            ):
                raise RuntimeError(
                    "pregrasp transit disturbed target by {:.4f}m".format(
                        row["object_pregrasp_displacement_m"]
                    )
                )
            self.restore_exact_target_proxy(
                target_object, object_before_approach, row
            )
            states.append("REFINE_TO_EXACT_PREGRASP")
            rospy.loginfo(
                "[pick] Trial %d: Cartesian refinement to exact pregrasp...",
                trial,
            )
            self.execute_cartesian_segment(
                pregrasp,
                row,
                "pregrasp_refinement",
                "pregrasp refinement",
            )
            object_before_approach = self.model_pose(target_object)
            row["object_before_approach_xyz_m"] = point_json(
                object_before_approach.position
            )
            cumulative_pregrasp_displacement = position_distance(
                object_before_pregrasp.position, object_before_approach.position
            )
            row["object_pregrasp_displacement_m"] = cumulative_pregrasp_displacement
            if cumulative_pregrasp_displacement > float(
                self.config["maximum_pregrasp_object_displacement_m"]
            ):
                raise RuntimeError(
                    "pregrasp refinement disturbed target by {:.4f}m".format(
                        cumulative_pregrasp_displacement
                    )
                )
            if self.grasp_mode in (
                "approach_only", "physical_grasp_only", "physical_contact"
            ):
                states.append("ALLOW_TARGET_FINGER_CONTACT")
                original_acm = self.allow_target_finger_contacts(
                    target_object, row
                )
            approach = Pose()
            approach.position.x = pregrasp.position.x + self.config["approach_vector_world_m"][0]
            approach.position.y = pregrasp.position.y + self.config["approach_vector_world_m"][1]
            approach.position.z = pregrasp.position.z + self.config["approach_vector_world_m"][2]
            approach.orientation = pregrasp.orientation
            states.append("APPROACH")
            rospy.loginfo("[pick] Trial %d: executing Cartesian approach...", trial)
            self.execute_approach(approach, row)
            object_after_approach = self.model_pose(target_object)
            row["object_after_approach_xyz_m"] = point_json(
                object_after_approach.position
            )
            row["object_approach_displacement_m"] = position_distance(
                object_before_approach.position, object_after_approach.position
            )
            if self.grasp_mode == "fixed_attachment_demo_nonphysical":
                states.append("ATTACH_OBJECT")
                rospy.loginfo(
                    "[pick] Trial %d: creating simulation-only fixed attachment...",
                    trial,
                )
                object_before = self.attach_object(target_object, row)
                states.append("PLAN_AND_EXECUTE_LIFT")
                rospy.loginfo("[pick] Trial %d: lifting object by 0.10 m...", trial)
                self.execute_lift(target_object, object_before, row)
                states.append("VERIFY_LIFT")
                rospy.loginfo(
                    "[pick] Trial %d: object lifted %.3f m and remains attached.",
                    trial,
                    row["object_lift_m"],
                )
            elif self.grasp_mode in ("physical_grasp_only", "physical_contact"):
                states.append("GRASP_HAND")
                rospy.loginfo("[pick] Trial %d: grasping...", trial)
                self.reset_contact_observation()
                closed = self.hand.command("GRASP")
                object_after_grasp = self.model_pose(target_object)
                row["object_after_grasp_xyz_m"] = point_json(
                    object_after_grasp.position
                )
                row["object_grasp_displacement_m"] = position_distance(
                    object_after_approach.position, object_after_grasp.position
                )
                row["hand_close_joint_verification_success"] = bool(
                    closed["success"]
                )
                row["hand_target_joint_positions"] = json.dumps(
                    closed.get("target_joint_positions"), sort_keys=True
                )
                row["hand_actual_joint_positions"] = json.dumps(
                    closed.get("actual_joint_positions"), sort_keys=True
                )
                states.append("VERIFY_GRASP_DISPLACEMENT")
                maximum_grasp_displacement_m = float(
                    self.config["physical_acceptance"][
                        "maximum_grasp_displacement_m"
                    ]
                )
                try:
                    grasp_displacement_accepted = validate_grasp_displacement(
                        row["object_grasp_displacement_m"],
                        maximum_grasp_displacement_m,
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "static grasp displacement limit {:.6f} m is invalid "
                        "({}); measured object displacement was {:.6f} m".format(
                            maximum_grasp_displacement_m,
                            exc,
                            row["object_grasp_displacement_m"],
                        )
                    )
                if not grasp_displacement_accepted:
                    raise RuntimeError(
                        "static grasp displaced object by {:.6f} m; "
                        "maximum allowed is {:.6f} m".format(
                            row["object_grasp_displacement_m"],
                            maximum_grasp_displacement_m,
                        )
                    )
                self.verify_configuration_hold(closed, row)
                states.append("VERIFY_HAND_COMMAND")
                # A physical object is allowed to stop one or more flexion
                # joints or their finite-effort distal mimics before the
                # empty-hand target. Only those relation failures may proceed
                # to the independent contact check. The palm pair was already
                # hard-checked above; timeout/controller/path/mimic-stability
                # failures remain fatal.
                contact_obstruction_candidate = is_contact_obstruction_candidate(
                    closed
                )
                if not closed["success"] and not contact_obstruction_candidate:
                    raise RuntimeError(
                        "GRASP failed: {}".format(closed["failure_reason"])
                    )
                states.append("VERIFY_MULTIFINGER_CONTACT")
                self.verify_multifinger_contact(row)
                self.begin_contact_guard()
                row["hand_close_success"] = True
                rospy.loginfo(
                    "[pick] Trial %d: stable physical contact from %s.",
                    trial,
                    row["grasp_contact_finger_families"],
                )
                if self.grasp_mode == "physical_grasp_only":
                    states.append("STATIC_GRASP_COMPLETE")
                    self.assert_contact_continuity(row, "static grasp")
                    row["physical_grasp_claimed"] = True
                else:
                    object_before = self.model_pose(target_object)
                    row["object_initial_z_m"] = object_before.position.z
                    states.append("REMOVE_TARGET_MOVEIT_PROXY")
                    row["planning_scene_target_removed_for_lift"] = (
                        self.remove_target_from_planning_scene_for_lift(target_object)
                    )
                    if not row["planning_scene_target_removed_for_lift"]:
                        raise RuntimeError(
                            "target MoveIt proxy could not be removed before lift"
                        )
                    states.append("PLAN_AND_EXECUTE_PHYSICAL_LIFT")
                    rospy.loginfo(
                        "[pick] Trial %d: lifting while GRASP remains active...", trial
                    )
                    self.execute_lift(target_object, object_before, row)
                    if row["object_lift_m"] < float(
                        self.config["physical_acceptance"]["minimum_object_lift_m"]
                    ):
                        raise RuntimeError("physical object lift is below threshold")
                    states.append("VERIFY_PHYSICAL_HOLD")
                    self.verify_physical_hold(target_object, row)
                    states.append("VERIFY_CONTACT_AFTER_LIFT")
                    self.assert_contact_continuity(row, "lift/hold")
                    self.verify_multifinger_contact(row)
                    states.append("PLACE_OBJECT_ON_TABLE")
                    placed = self.execute_place(target_object, object_before, row)
                    states.append("VERIFY_CONTACT_BEFORE_RELEASE")
                    self.assert_contact_continuity(row, "placement")
                    self.verify_multifinger_contact(row)
                    states.append("RELEASE_OBJECT_ON_TABLE")
                    self.release_on_table(
                        target_object, placed, object_before, original_acm, row
                    )
                    states.append("VERIFY_RELEASE_ON_TABLE")
                    row["physical_grasp_claimed"] = True
            states.append("LIFT_OR_STOP")
            stopped = self.hand.command("STOP")
            if not stopped["success"]:
                raise RuntimeError("STOP failed: {}".format(stopped["failure_reason"]))
            states.append("DONE")
            row["state"] = "DONE"
            row["success"] = True
            rospy.loginfo("[pick] Trial %d DONE.", trial)
        except Exception as exc:
            self.group.stop()
            self.group.clear_pose_targets()
            try:
                self.hand.command("RELEASE")
            except Exception as release_exc:
                rospy.logwarn("[pick] fail-safe RELEASE failed: %s", release_exc)
            self.hand.command("STOP")
            if row.get("attachment_used"):
                self.detach_object(target_object)
                row["attachment_used"] = False
                row["attachment_type"] = "detached_after_failure"
            states.append("FAILED")
            row["failure_reason"] = str(exc)
            rospy.logerr("[pick] Trial %d FAILED: %s", trial, exc)
        try:
            self.restore_collision_matrix(original_acm)
            if original_acm is not None:
                states.append("RESTORE_COLLISION_MATRIX")
        except Exception as restore_exc:
            self.group.stop()
            try:
                self.hand.command("RELEASE")
                self.hand.command("STOP")
            except Exception as release_exc:
                rospy.logwarn(
                    "[pick] fail-safe after ACM restore failure also failed: %s",
                    release_exc,
                )
            if states[-1] != "FAILED":
                states.append("FAILED")
            row["state"] = "FAILED"
            row["success"] = False
            prior = row.get("failure_reason") or ""
            row["failure_reason"] = "{}{}ACM restore failed: {}".format(
                prior, "; " if prior else "", restore_exc
            )
            rospy.logerr("[pick] %s", row["failure_reason"])
        row["states"] = json.dumps(states, separators=(",", ":"))
        row["task_execution_time_s"] = time.monotonic() - started
        self.status_pub.publish(json.dumps(row, sort_keys=True))
        return row

    def save(self, rows, target_object, collision_count):
        os.makedirs(self.results_dir, exist_ok=True)
        json_path = os.path.join(self.results_dir, "pick_{}_{}.json".format(self.grasp_mode, self.run_id))
        object_spec = self.scene_config["objects"][self.config["target_object_key"]]
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "run_id": self.run_id, "grasp_mode": self.grasp_mode,
                    "all_success": all(row["success"] for row in rows), "rows": rows,
                    "scene_config": self.scene_path, "grasp_config": self.grasp_path,
                    "startup_configuration": self.startup,
                    "target_object": target_object, "target_object_spec": object_spec,
                    "object_to_pregrasp": self.config["object_to_pregrasp"],
                    "approach_vector_world_m": self.config["approach_vector_world_m"],
                    "collision_object_count": collision_count,
                    "attachment_demo_status": self.config["attachment_demo_status"],
                    "physical_contact_status": self.config["physical_contact_status"],
                }, stream, indent=2, sort_keys=True,
            )
        # Version 11 records the padded-transit/exact-grasp proxy evidence.
        # Keep earlier schemas
        # immutable so rows never silently shift beneath an existing header.
        csv_path = os.path.join(self.results_dir, "pick_results_v11.csv")
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        return json_path, csv_path

    def run(self):
        self.run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target_object, collision_count = self.wait_ready()
        rows = []
        for trial in range(1, self.repetitions + 1):
            row = self.run_trial(trial, target_object)
            rows.append(row)
            if not row["success"]:
                break
        paths = self.save(rows, target_object, collision_count)
        rospy.loginfo("[pick] DONE. Results: %s and %s", *paths)
        rospy.loginfo(
            "[pick] Gazebo/RViz remain open. Press Ctrl-C in this terminal when finished."
        )
        return all(row["success"] for row in rows)


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("scripted_pick_demo")
    try:
        success = ScriptedPickDemo().run()
    except Exception as exc:
        rospy.logfatal("Scripted pick demo initialization failed: %s", exc)
        success = False
    finally:
        moveit_commander.roscpp_shutdown()
    raise SystemExit(0 if success else 8)


if __name__ == "__main__":
    main()
