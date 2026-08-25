#!/usr/bin/env python3
"""Latch controller-start permission only after spawn_model applies every -J."""

import math
import shlex
import time

import rospy
from gazebo_msgs.srv import GetJointProperties, GetWorldProperties
from std_msgs.msg import Bool
from std_srvs.srv import Empty


def parse_initial_joint_arguments(text):
    tokens = shlex.split(str(text))
    if len(tokens) == 0 or len(tokens) % 3 != 0:
        raise ValueError("initial_joint_arguments must contain -J name value triples")
    targets = {}
    for index in range(0, len(tokens), 3):
        flag, name, raw_value = tokens[index:index + 3]
        if flag != "-J" or not name or name in targets:
            raise ValueError("invalid or duplicate initial joint argument")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("initial joint positions must be finite")
        targets[name] = value
    return targets


def angular_error(measured, target):
    return abs(math.atan2(math.sin(measured - target),
                          math.cos(measured - target)))


class GazeboInitialStateGate:
    def __init__(self):
        self.model_name = str(rospy.get_param("~model_name", "robot"))
        self.targets = parse_initial_joint_arguments(rospy.get_param(
            "~initial_joint_arguments"))
        self.timeout_s = float(rospy.get_param("~timeout_s", 30.0))
        self.tolerance_rad = float(rospy.get_param(
            "~tolerance_rad", 5.0e-4))
        self.stay_paused = bool(rospy.get_param("~stay_paused", False))
        if (not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0 or
                not math.isfinite(self.tolerance_rad) or
                self.tolerance_rad <= 0.0):
            raise ValueError("gate timing/tolerance must be finite and positive")
        self.publisher = rospy.Publisher(
            "/handarm_sim_demo/gazebo_initial_state_ready", Bool,
            queue_size=1, latch=True)
        self.publisher.publish(Bool(data=False))

    def run(self):
        rospy.wait_for_service("/gazebo/get_world_properties",
                               timeout=self.timeout_s)
        rospy.wait_for_service("/gazebo/get_joint_properties",
                               timeout=self.timeout_s)
        rospy.wait_for_service("/gazebo/unpause_physics",
                               timeout=self.timeout_s)
        get_world = rospy.ServiceProxy(
            "/gazebo/get_world_properties", GetWorldProperties)
        get_joint = rospy.ServiceProxy(
            "/gazebo/get_joint_properties", GetJointProperties)
        deadline = time.monotonic() + self.timeout_s
        worst_error = float("inf")
        worst_joint = "unavailable"
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                if self.model_name not in get_world().model_names:
                    time.sleep(0.02)
                    continue
                errors = {}
                complete = True
                for name, target in self.targets.items():
                    response = get_joint("{}::{}".format(
                        self.model_name, name))
                    if not response.success or not response.position:
                        complete = False
                        break
                    errors[name] = angular_error(
                        float(response.position[0]), target)
                if complete and errors:
                    worst_joint = max(errors, key=errors.get)
                    worst_error = errors[worst_joint]
                    if worst_error <= self.tolerance_rad:
                        # spawn_model applies the requested configuration while
                        # physics is paused. Release physics only after that
                        # succeeds, immediately before waking the controller
                        # spawners. This removes the old controller-vs--J race.
                        rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
                        self.publisher.publish(Bool(data=True))
                        rospy.logwarn(
                            "Gazebo initial-state gate OPEN: %d joints set; "
                            "worst error %.9f rad at %s. Controllers may start.",
                            len(errors), worst_error, worst_joint)
                        if self.stay_paused:
                            rospy.wait_for_service("/gazebo/pause_physics",
                                                   timeout=self.timeout_s)
                            # Give controller_manager its first update before
                            # honoring a diagnostic paused:=true request.
                            time.sleep(0.5)
                            rospy.ServiceProxy("/gazebo/pause_physics", Empty)()
                        rospy.spin()
                        return
            except (rospy.ServiceException, rospy.ROSException) as exc:
                rospy.logwarn_throttle(
                    1.0, "Waiting for Gazebo initial configuration: %s", exc)
            time.sleep(0.02)
        raise RuntimeError(
            "Gazebo did not reach requested initial state; worst {} error "
            "{:.9f} rad".format(worst_joint, worst_error))


def main():
    rospy.init_node("gazebo_initial_state_gate")
    try:
        GazeboInitialStateGate().run()
    except Exception as exc:
        rospy.logfatal("Gazebo initial-state gate failed: %s", exc)
        raise SystemExit(4)


if __name__ == "__main__":
    main()
