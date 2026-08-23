#!/usr/bin/env python3
"""Unified fail-closed command interface for the simulated three-finger hand."""

import json
import math
import threading
import time

import actionlib
import rospy
import yaml
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTolerance,
)
from gazebo_msgs.srv import GetJointProperties
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectoryPoint


VALID_COMMANDS = {"OPEN", "CLOSE", "PRE_SHAPE_A", "PRE_SHAPE_B", "HOLD", "STOP"}
PUBLIC_COMMAND_ALIASES = {
    "GRASP": "CLOSE",
    "RELEASE": "OPEN",
}
PUBLIC_OPERATOR_COMMANDS = frozenset(PUBLIC_COMMAND_ALIASES)


def normalize_command(name):
    requested = str(name).strip().upper()
    if requested.startswith("SET_CONFIGURATION:"):
        configuration = requested.split(":", 1)[1].strip()
        requested = "SET_CONFIGURATION_{}".format(configuration)
    return requested, PUBLIC_COMMAND_ALIASES.get(requested, requested)


def build_command_trajectory_waypoints(
    config, command_name, current_positions, target_positions
):
    """Build ROS-free trajectory waypoints for one hand command."""
    names = config["joint_names"]
    duration_s = float(config["execution"]["duration_s"])
    first_time_s = 0.2
    first_positions = [current_positions[name] for name in names]
    if command_name != "OPEN":
        return [
            (first_time_s, first_positions),
            (first_time_s + duration_s, list(target_positions)),
        ]
    delayed = set(config["execution"]["open_delayed_joints"])
    stage_one = dict(zip(names, target_positions))
    for name in delayed:
        stage_one[name] = current_positions[name]
    return [
        (first_time_s, first_positions),
        (
            first_time_s + duration_s / 2.0,
            [stage_one[name] for name in names],
        ),
        (first_time_s + duration_s, list(target_positions)),
    ]


def collect_failure_diagnostics(
    verify, target, command_name, start, active_names, mimic_names
):
    """Capture one fail-closed joint sample after an action-level failure."""
    try:
        (
            verification_success,
            mimic_relation_pass,
            actual,
            active_errors,
            mimic_errors,
            grasp_progress,
        ) = verify(target, command_name=command_name, start=start)
        actual_positions = {
            joint: actual[joint] for joint in active_names + mimic_names
        }
        values = list(actual_positions.values())
        values += list(active_errors.values()) + list(mimic_errors.values())
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("non-finite post-failure joint diagnostic")
        return {
            "verification_success": bool(verification_success),
            "mimic_relation_pass": bool(mimic_relation_pass),
            "actual_joint_positions": actual_positions,
            "active_joint_errors_rad": active_errors,
            "mimic_joint_errors_rad": mimic_errors,
            "grasp_joint_progress_rad": grasp_progress,
            "sample_error": "",
        }
    except Exception as exc:
        return {
            "verification_success": False,
            "mimic_relation_pass": False,
            "actual_joint_positions": None,
            "active_joint_errors_rad": None,
            "mimic_joint_errors_rad": None,
            "grasp_joint_progress_rad": None,
            "sample_error": "{}: {}".format(type(exc).__name__, exc),
        }


def command_failure_result(
    command,
    internal_command,
    target_joint_positions,
    execution_time_s,
    failure_reason,
    error_code,
    error_string,
    failure_diagnostics,
):
    """Build a diagnostic action result without promoting it to success."""
    diagnostics = failure_diagnostics or {}
    return {
        "command": command,
        "internal_command": internal_command,
        "target_joint_positions": target_joint_positions,
        "actual_joint_positions": diagnostics.get("actual_joint_positions"),
        "active_joint_errors_rad": diagnostics.get("active_joint_errors_rad"),
        "mimic_joint_errors_rad": diagnostics.get("mimic_joint_errors_rad"),
        "mimic_relation_pass": diagnostics.get("mimic_relation_pass"),
        "grasp_joint_progress_rad": diagnostics.get("grasp_joint_progress_rad"),
        "failure_diagnostics": diagnostics,
        "error_code": error_code,
        "error_string": error_string,
        "execution_time_s": execution_time_s,
        "success": False,
        "failure_reason": failure_reason,
    }


def validate_hand_config(config):
    names = config.get("joint_names", [])
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("hand config must contain four unique active joints")
    limits = config.get("joint_limits", {})
    for command, spec in config.get("commands", {}).items():
        if command not in VALID_COMMANDS:
            raise ValueError("unknown hand command {}".format(command))
        positions = spec.get("positions")
        if positions is None:
            continue
        if len(positions) != len(names) or not all(math.isfinite(v) for v in positions):
            raise ValueError("invalid target for {}".format(command))
        for name, value in zip(names, positions):
            lower, upper = limits[name]
            if value < lower or value > upper:
                raise ValueError("{} target for {} exceeds URDF limits".format(command, name))
    if set(config.get("commands", {})) != VALID_COMMANDS:
        raise ValueError("hand command set is incomplete")
    execution = config.get("execution", {})
    configuration = set(execution.get("configuration_joint_names", []))
    flexion = set(execution.get("flexion_joint_names", []))
    if not configuration or configuration & flexion or configuration | flexion != set(names):
        raise ValueError("configuration and flexion joints must partition active joints")
    duration_s = execution.get("duration_s")
    if (
        duration_s is None
        or not math.isfinite(float(duration_s))
        or float(duration_s) <= 0.0
    ):
        raise ValueError("execution duration must be finite and positive")
    delayed = set(execution.get("open_delayed_joints", []))
    if not delayed:
        raise ValueError("OPEN delayed joints must be declared")
    if delayed & configuration:
        raise ValueError("OPEN delayed joints must not include configuration joints")
    if not delayed <= flexion:
        raise ValueError("OPEN delayed joints must be flexion joints")
    command_positions = {
        command: dict(zip(names, config["commands"][command]["positions"]))
        for command in ("OPEN", "CLOSE")
    }
    if any(
        command_positions["OPEN"][name] != command_positions["CLOSE"][name]
        for name in configuration
    ):
        raise ValueError("GRASP/RELEASE must not move configuration joints")
    if any(
        command_positions["CLOSE"][name] <= command_positions["OPEN"][name]
        for name in flexion
    ):
        raise ValueError("GRASP must increase each flexion joint")
    minimum = execution.get("grasp_minimum_progress_rad", {})
    if set(minimum) != flexion or any(
        not math.isfinite(value) or value < 0.0 for value in minimum.values()
    ):
        raise ValueError("GRASP progress thresholds must cover flexion joints only")
    return True


class HandCommander:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        validate_hand_config(self.config)
        self.names = list(self.config["joint_names"])
        self.mimic_names = list(self.config["mimic_joints"])
        self.controller_names = self.names
        self.action_name = "/controller_gazebo_hand/follow_joint_trajectory"
        self.client = actionlib.SimpleActionClient(
            self.action_name, FollowJointTrajectoryAction
        )
        timeout = float(self.config["execution"]["timeout_s"])
        if not self.client.wait_for_server(rospy.Duration(timeout)):
            raise RuntimeError("hand trajectory action did not become ready")
        rospy.wait_for_service("/gazebo/get_joint_properties", timeout=timeout)
        self.get_joint_properties = rospy.ServiceProxy(
            "/gazebo/get_joint_properties", GetJointProperties
        )
        self.lock = threading.Lock()

    def joint_state(self):
        required = set(self.controller_names)
        deadline = time.monotonic() + 3.0
        values = {}
        while time.monotonic() < deadline and not rospy.is_shutdown():
            try:
                message = rospy.wait_for_message(
                    "/joint_states", JointState, timeout=0.5
                )
            except rospy.ROSException:
                continue
            values = dict(zip(message.name, message.position))
            if required.issubset(values):
                break
        if not required.issubset(values):
            raise RuntimeError(
                "joint_states is missing joints: {}".format(
                    sorted(required - set(values))
                )
            )
        if not all(math.isfinite(values[name]) for name in required):
            raise RuntimeError("hand joint state is non-finite")
        return values

    def mimic_positions(self):
        positions = {}
        for name in self.config["mimic_joints"]:
            response = self.get_joint_properties("robot::{}".format(name))
            if not response.success or not response.position:
                raise RuntimeError(
                    "Gazebo mimic joint query failed for {}: {}".format(
                        name, response.status_message
                    )
                )
            value = response.position[0]
            if not math.isfinite(value):
                raise RuntimeError("Gazebo mimic joint {} is non-finite".format(name))
            positions[name] = value
        return positions

    def _verify(self, target, command_name=None, start=None):
        actual = self.joint_state()
        actual.update(self.mimic_positions())
        tolerance = float(self.config["execution"]["joint_tolerance_rad"])
        active_errors = {
            name: abs(actual[name] - value)
            for name, value in zip(self.names, target)
        }
        mimic_errors = {}
        for mimic, relation in self.config["mimic_joints"].items():
            expected = (
                actual[relation["source"]] * float(relation["multiplier"])
                + float(relation["offset"])
            )
            mimic_errors[mimic] = abs(actual[mimic] - expected)
        grasp_progress = {}
        if command_name == "CLOSE":
            if start is None:
                raise RuntimeError("CLOSE verification requires its start state")
            minimum = self.config["execution"]["grasp_minimum_progress_rad"]
            target_by_name = dict(zip(self.names, target))
            for name in minimum:
                value = target_by_name[name]
                delta = value - start[name]
                direction = 1.0 if delta >= 0.0 else -1.0
                grasp_progress[name] = (actual[name] - start[name]) * direction
            configuration = self.config["execution"]["configuration_joint_names"]
            active_success = all(
                grasp_progress[name] >= float(minimum[name])
                for name in minimum
            ) and all(active_errors[name] <= tolerance for name in configuration)
        else:
            active_success = max(active_errors.values()) <= tolerance
        mimic_relation_pass = max(mimic_errors.values()) <= float(
            self.config["execution"]["mimic_tolerance_rad"]
        )
        return (
            active_success,
            mimic_relation_pass,
            actual,
            active_errors,
            mimic_errors,
            grasp_progress,
        )

    def command(self, name):
        requested_name, name = normalize_command(name)
        if name not in VALID_COMMANDS:
            return {
                "command": requested_name,
                "internal_command": None,
                "success": False,
                "failure_reason": "unknown command",
            }
        with self.lock:
            started = time.monotonic()
            if name == "STOP":
                self.client.cancel_all_goals()
                return {
                    "command": requested_name,
                    "internal_command": name,
                    "target_joint_positions": None,
                    "actual_joint_positions": self.joint_state(),
                    "execution_time_s": time.monotonic() - started,
                    "success": True,
                    "failure_reason": "",
                }
            current = self.joint_state()
            spec = self.config["commands"][name]
            target = (
                [current[joint] for joint in self.names]
                if spec.get("dynamic_current_position")
                else list(spec["positions"])
            )
            goal = FollowJointTrajectoryGoal()
            controller_target = dict(zip(self.names, target))
            goal.trajectory.joint_names = self.controller_names
            goal.trajectory.points = []
            for time_from_start_s, positions in build_command_trajectory_waypoints(
                self.config, name, current, target
            ):
                point = JointTrajectoryPoint()
                point.positions = positions
                point.time_from_start = rospy.Duration(time_from_start_s)
                goal.trajectory.points.append(point)
            goal.path_tolerance = [
                JointTolerance(
                    name=joint,
                    position=float(self.config["execution"]["path_tolerance_rad"]),
                )
                for joint in self.controller_names
            ]
            goal_tolerance = float(
                self.config["execution"][
                    "grasp_goal_tolerance_rad"
                    if name == "CLOSE"
                    else "joint_tolerance_rad"
                ]
            )
            goal.goal_tolerance = [
                JointTolerance(
                    name=joint,
                    position=goal_tolerance,
                )
                for joint in self.controller_names
            ]
            goal.goal_time_tolerance = rospy.Duration(1.0)
            self.client.send_goal(goal)
            if not self.client.wait_for_result(
                rospy.Duration(float(self.config["execution"]["timeout_s"]))
            ):
                self.client.cancel_goal()
                diagnostics = collect_failure_diagnostics(
                    self._verify, target, name, current, self.names, self.mimic_names
                )
                return command_failure_result(
                    requested_name,
                    name,
                    controller_target,
                    time.monotonic() - started,
                    "trajectory timeout",
                    None,
                    "",
                    diagnostics,
                )
            result = self.client.get_result()
            if result is None or result.error_code != result.SUCCESSFUL:
                code = None if result is None else int(result.error_code)
                detail = "" if result is None else str(result.error_string)
                diagnostics = collect_failure_diagnostics(
                    self._verify, target, name, current, self.names, self.mimic_names
                )
                return command_failure_result(
                    requested_name,
                    name,
                    controller_target,
                    time.monotonic() - started,
                    "trajectory failed {} {}".format(code, detail).strip(),
                    code,
                    detail,
                    diagnostics,
                )
            (
                success,
                mimic_relation_pass,
                actual,
                active_errors,
                mimic_errors,
                grasp_progress,
            ) = self._verify(target, command_name=name, start=current)
            execution = self.config["execution"]
            verification_samples = int(execution["mimic_stability_samples"])
            verification_period = float(execution["mimic_stability_period_s"])
            if verification_samples < 2 or verification_period <= 0.0:
                raise RuntimeError("invalid mimic stability sampling configuration")
            mimic_history = {joint: [actual[joint]] for joint in self.mimic_names}
            active_results = [success]
            mimic_results = [mimic_relation_pass]
            for _ in range(verification_samples - 1):
                time.sleep(verification_period)
                (
                    sample_active,
                    sample_mimic,
                    actual,
                    active_errors,
                    mimic_errors,
                    grasp_progress,
                ) = self._verify(target, command_name=name, start=current)
                active_results.append(sample_active)
                mimic_results.append(sample_mimic)
                for joint in self.mimic_names:
                    mimic_history[joint].append(actual[joint])
            mimic_stability_range = {
                joint: max(values) - min(values)
                for joint, values in mimic_history.items()
            }
            mimic_stability_pass = max(mimic_stability_range.values()) <= float(
                execution["mimic_stability_range_rad"]
            )
            success = all(active_results) and all(mimic_results) and mimic_stability_pass
            if not all(active_results):
                failure_reason = "active joint verification failed"
            elif not all(mimic_results):
                failure_reason = "mimic joint relation verification failed"
            elif not mimic_stability_pass:
                failure_reason = "mimic joint stability verification failed"
            else:
                failure_reason = ""
            return {
                "command": requested_name,
                "internal_command": name,
                "target_joint_positions": controller_target,
                "actual_joint_positions": {
                    joint: actual[joint]
                    for joint in self.names + self.mimic_names
                },
                "active_joint_errors_rad": active_errors,
                "mimic_joint_errors_rad": mimic_errors,
                "mimic_relation_pass": mimic_relation_pass,
                "mimic_stability_range_rad": mimic_stability_range,
                "mimic_stability_pass": mimic_stability_pass,
                "grasp_joint_progress_rad": grasp_progress,
                "error_code": int(result.error_code),
                "error_string": str(result.error_string),
                "execution_time_s": time.monotonic() - started,
                "success": success,
                "failure_reason": failure_reason,
            }


class HandCommanderNode:
    def __init__(self):
        self.commander = HandCommander(rospy.get_param("~hand_config"))
        self.status = rospy.Publisher(
            "/handarm_sim_demo/hand_status", String, queue_size=10, latch=True
        )
        self.ready = rospy.Publisher(
            "/handarm_sim_demo/hand_ready", Bool, queue_size=1, latch=True
        )
        self.subscriber = rospy.Subscriber(
            "/handarm_sim_demo/hand_command", String, self.callback, queue_size=1
        )
        self.ready.publish(True)

    def callback(self, message):
        requested, _ = normalize_command(message.data)
        if requested not in PUBLIC_OPERATOR_COMMANDS:
            result = {
                "command": requested,
                "internal_command": None,
                "success": False,
                "failure_reason": "public interface accepts GRASP or RELEASE only",
            }
        else:
            result = self.commander.command(requested)
        self.status.publish(json.dumps(result, sort_keys=True))


def main():
    rospy.init_node("hand_commander")
    try:
        HandCommanderNode()
        rospy.loginfo("Simulated hand commander ready")
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("Hand commander failed: %s", exc)
        raise SystemExit(6)


if __name__ == "__main__":
    main()
