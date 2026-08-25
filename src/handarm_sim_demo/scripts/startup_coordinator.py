#!/usr/bin/env python3
"""Actively gate simulation startup on models, joint states, and controllers."""

import json
import math
import threading
import time

import actionlib
import rospy
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTolerance,
)
from controller_manager_msgs.srv import (
    ListControllers,
    LoadController,
    SwitchController,
    SwitchControllerRequest,
)
from gazebo_msgs.srv import GetWorldProperties, SetModelConfiguration
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectoryPoint


EXPECTED_CONTROLLERS = {
    "controller_gazebo",
    "controller_gazebo_hand",
    "joint_state_controller",
}
DEFAULT_EXPECTED_MODELS = {
    "robot",
    "work_table",
    "target_object",
    "obstacle_a",
    "obstacle_b",
}
HAND_DIAGNOSTIC_JOINTS = (
    "f1j1",
    "f3j1",
    "f1j2",
    "f1j3",
    "f2j1",
    "f2j2",
    "f3j2",
    "f3j3",
)


def joint_position_error(measured, target, wraparound=False):
    """Return absolute joint error, respecting equivalent revolutions."""
    difference = float(measured) - float(target)
    if wraparound:
        difference = math.atan2(math.sin(difference), math.cos(difference))
    return abs(difference)


def wait_wall(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except (rospy.ROSException, rospy.ServiceException) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise RuntimeError("timeout waiting for {}: {}".format(description, last_error))


def joint_state_diagnostic(message, joint_names):
    """Return finite, JSON-serializable position/velocity diagnostics."""
    positions = dict(zip(message.name, message.position))
    velocities = dict(zip(message.name, message.velocity))
    return {
        name: {
            "position": positions.get(name),
            "velocity": velocities.get(name),
        }
        for name in joint_names
    }


def main():
    rospy.init_node("startup_coordinator")
    timeout = float(rospy.get_param("~startup_timeout", 60.0))
    stay_paused = bool(rospy.get_param("~paused", False))
    initialize_arm_trajectory = bool(
        rospy.get_param("~initialize_arm_trajectory", True)
    )
    initial = rospy.get_param("~initial_configuration")
    expected_models = set(
        rospy.get_param("~expected_models", sorted(DEFAULT_EXPECTED_MODELS))
    )
    if not expected_models or not all(
        isinstance(name, str) and name for name in expected_models
    ):
        raise ValueError("expected_models must contain non-empty model names")
    ready_pub = rospy.Publisher(
        "/handarm_sim_demo/startup_ready", Bool, queue_size=1, latch=True
    )
    status_pub = rospy.Publisher(
        "/handarm_sim_demo/startup_status", String, queue_size=1, latch=True
    )

    try:
        guard = wait_wall(
            lambda: rospy.wait_for_message(
                "/handarm_sim_demo/simulation_guard_ready", Bool, timeout=0.5
            ),
            timeout,
            "simulation guard",
        )
        if not guard.data:
            raise RuntimeError("simulation guard did not authorize startup")

        rospy.wait_for_service("/gazebo/get_world_properties", timeout=timeout)
        rospy.wait_for_service("/controller_manager/list_controllers", timeout=timeout)
        rospy.wait_for_service("/controller_manager/load_controller", timeout=timeout)
        rospy.wait_for_service("/controller_manager/switch_controller", timeout=timeout)
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=timeout)
        rospy.wait_for_service("/gazebo/pause_physics", timeout=timeout)
        rospy.wait_for_service("/gazebo/set_model_configuration", timeout=timeout)
        get_world = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
        list_controllers = rospy.ServiceProxy(
            "/controller_manager/list_controllers", ListControllers
        )
        load_controller = rospy.ServiceProxy(
            "/controller_manager/load_controller", LoadController
        )
        switch_controller = rospy.ServiceProxy(
            "/controller_manager/switch_controller", SwitchController
        )
        set_configuration = rospy.ServiceProxy(
            "/gazebo/set_model_configuration", SetModelConfiguration
        )

        wait_wall(
            lambda: expected_models.issubset(set(get_world().model_names)),
            timeout,
            "Gazebo models",
        )
        loaded = {item.name for item in list_controllers().controller}
        for controller in sorted(EXPECTED_CONTROLLERS - loaded):
            if not load_controller(controller).ok:
                raise RuntimeError("failed to load controller {}".format(controller))

        # Set the initial configuration while controllers are still stopped.
        # Starting position controllers first lets their default zero command
        # race this reset on slower physics profiles.
        reset_names = []
        reset_positions = []
        for label in ("arm", "hand"):
            reset_names.extend(initial[label]["joint_names"])
            reset_positions.extend(initial[label]["positions"])
        reset = set_configuration(
            model_name="robot",
            urdf_param_name="robot_description",
            joint_names=reset_names,
            joint_positions=reset_positions,
        )
        if not reset.success:
            raise RuntimeError(
                "Gazebo initial configuration reset failed: {}".format(
                    reset.status_message
                )
            )

        # Queue controller activation while physics is paused, then release
        # physics so controller_manager can complete it on the first update.
        # A synchronous switch before unpause cannot complete; unpausing before
        # issuing it gives the low-inertia fingers an uncontrolled impulse.
        switch_outcome = {}
        switch_started = threading.Event()

        def switch_on_first_update():
            switch_started.set()
            try:
                switch_outcome["result"] = switch_controller(
                    start_controllers=sorted(EXPECTED_CONTROLLERS),
                    stop_controllers=[],
                    strictness=SwitchControllerRequest.STRICT,
                    start_asap=True,
                    timeout=5.0,
                )
            except Exception as exc:  # Propagate service failures on main thread.
                switch_outcome["error"] = exc

        switch_thread = threading.Thread(
            target=switch_on_first_update,
            name="controller-start-before-unpause",
            daemon=True,
        )
        switch_thread.start()
        if not switch_started.wait(1.0):
            raise RuntimeError("controller switch request thread did not start")
        time.sleep(0.05)
        rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
        switch_thread.join(5.0)
        if switch_thread.is_alive():
            raise RuntimeError("controller switch did not complete after unpause")
        if "error" in switch_outcome:
            raise RuntimeError(
                "controller switch service failed: {}".format(
                    switch_outcome["error"]
                )
            )
        switch_result = switch_outcome.get("result")
        if switch_result is None or not switch_result.ok:
            raise RuntimeError("controller switch service rejected startup")

        def controllers_running():
            states = {item.name: item.state for item in list_controllers().controller}
            return EXPECTED_CONTROLLERS.issubset(states) and all(
                states[name] == "running" for name in EXPECTED_CONTROLLERS
            )

        wait_wall(controllers_running, timeout, "running controllers")
        initial_joint_state = wait_wall(
            lambda: rospy.wait_for_message("/joint_states", JointState, timeout=0.5),
            timeout,
            "joint states after controller start",
        )
        initial_measured = dict(
            zip(initial_joint_state.name, initial_joint_state.position)
        )
        rospy.loginfo(
            "Hand state after controller start: %s",
            json.dumps(
                joint_state_diagnostic(
                    initial_joint_state, HAND_DIAGNOSTIC_JOINTS
                ),
                sort_keys=True,
            ),
        )

        clients = {}
        trajectory_labels = (
            ("arm", "hand") if initialize_arm_trajectory else ("hand",)
        )
        for label in trajectory_labels:
            client = actionlib.SimpleActionClient(
                initial[label]["action"], FollowJointTrajectoryAction
            )
            if not client.wait_for_server(rospy.Duration(timeout)):
                raise RuntimeError("{} trajectory action unavailable".format(label))
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = list(initial[label]["joint_names"])
            current_point = JointTrajectoryPoint()
            current_point.positions = [
                initial_measured[name] for name in goal.trajectory.joint_names
            ]
            current_point.time_from_start = rospy.Duration(0.05)
            target_point = JointTrajectoryPoint()
            target_point.positions = list(initial[label]["positions"])
            target_point.time_from_start = rospy.Duration(float(initial["duration"]))
            goal.trajectory.points = [current_point, target_point]
            rospy.loginfo(
                "%s initial trajectory: joints=%s first=%s target=%s duration_s=%.3f",
                label,
                list(goal.trajectory.joint_names),
                list(current_point.positions),
                list(target_point.positions),
                float(initial["duration"]),
            )
            goal.path_tolerance = [
                JointTolerance(
                    name=name, position=float(initial[label]["path_tolerance"])
                )
                for name in goal.trajectory.joint_names
            ]
            goal.goal_tolerance = [
                JointTolerance(
                    name=name,
                    position=float(initial["tolerance"]),
                    velocity=0.0,
                    acceleration=0.0,
                )
                for name in goal.trajectory.joint_names
            ]
            goal.goal_time_tolerance = rospy.Duration(1.0)
            client.send_goal(goal)
            clients[label] = client
        for label, client in clients.items():
            if not client.wait_for_result(rospy.Duration(timeout)):
                client.cancel_goal()
                raise RuntimeError("{} initial trajectory timed out".format(label))
            result = client.get_result()
            if result is None or result.error_code != result.SUCCESSFUL:
                code = None if result is None else result.error_code
                detail = "no result" if result is None else result.error_string
                try:
                    failure_joint_state = rospy.wait_for_message(
                        "/joint_states", JointState, timeout=1.0
                    )
                    rospy.logerr(
                        "Hand state at initial trajectory failure: %s",
                        json.dumps(
                            joint_state_diagnostic(
                                failure_joint_state, HAND_DIAGNOSTIC_JOINTS
                            ),
                            sort_keys=True,
                        ),
                    )
                except rospy.ROSException as exc:
                    rospy.logerr(
                        "No joint-state diagnostic at initial trajectory failure: %s",
                        exc,
                    )
                raise RuntimeError(
                    "{} initial trajectory failed with code {}: {}".format(
                        label, code, detail
                    )
                )

        joint_state = rospy.wait_for_message("/joint_states", JointState, timeout=2.0)
        measured = dict(zip(joint_state.name, joint_state.position))
        errors = {}
        wraparound = set(initial.get("wraparound_joints", []))
        for label in ("arm", "hand"):
            for name, target in zip(
                initial[label]["joint_names"], initial[label]["positions"]
            ):
                errors[name] = joint_position_error(
                    measured[name], target, name in wraparound
                )
        max_error_joint = max(errors, key=errors.get)
        max_initial_error = errors[max_error_joint]
        if max_initial_error > float(initial["tolerance"]):
            raise RuntimeError(
                "initial joint error {:.6f} at {} (measured {:.6f}, target "
                "{:.6f}, periodic={}) exceeds tolerance {:.6f}".format(
                    max_initial_error,
                    max_error_joint,
                    measured[max_error_joint],
                    dict(
                        (name, target)
                        for label in ("arm", "hand")
                        for name, target in zip(
                            initial[label]["joint_names"],
                            initial[label]["positions"],
                        )
                    )[max_error_joint],
                    max_error_joint in wraparound,
                    float(initial["tolerance"]),
                )
            )
        if stay_paused:
            rospy.ServiceProxy("/gazebo/pause_physics", Empty)()

        status = {
            "ready": True,
            "paused": stay_paused,
            "controllers": sorted(EXPECTED_CONTROLLERS),
            "models": sorted(expected_models),
            "max_initial_joint_error_rad": max_initial_error,
            "max_initial_joint_error_name": max_error_joint,
        }
        status_pub.publish(json.dumps(status, sort_keys=True))
        ready_pub.publish(True)
        rospy.loginfo("Simulation startup ready: %s", status)
        rospy.spin()
    except Exception as exc:
        status_pub.publish(
            json.dumps({"ready": False, "failure_reason": str(exc)}, sort_keys=True)
        )
        rospy.logfatal("Simulation startup failed: %s", exc)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
