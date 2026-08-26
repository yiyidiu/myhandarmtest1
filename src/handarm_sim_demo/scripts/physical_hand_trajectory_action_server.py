#!/usr/bin/env python3
"""FollowJointTrajectory compatibility layer for the physical hand plugin.

The physical-grasp Gazebo profile intentionally removes the ros_control hand
controller so that it cannot fight contact constraints.  Existing MoveIt and
HandCommander clients still use the standard FollowJointTrajectory action;
this node validates that action, forwards its trajectory to the finite-force
spring plugin, and reports completion from measured joint states.
"""

import math
import threading
import time

import actionlib
import rospy
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryResult,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_JOINTS = ["f1j1", "f1j2", "f2j1", "f3j2"]


def validate_trajectory(message, required_names):
    if len(message.joint_names) != len(set(message.joint_names)):
        return "trajectory contains duplicate joint names"
    if set(message.joint_names) != set(required_names):
        return "trajectory joints must be exactly {}".format(required_names)
    if not message.points:
        return "trajectory has no points"
    previous = 0.0
    for index, point in enumerate(message.points):
        stamp = point.time_from_start.to_sec()
        if not math.isfinite(stamp) or stamp <= previous:
            return "point times must be finite and strictly increase from zero"
        if len(point.positions) != len(message.joint_names):
            return "point {} has the wrong position count".format(index)
        if not all(math.isfinite(value) for value in point.positions):
            return "point {} contains a non-finite position".format(index)
        previous = stamp
    return ""


def tolerance_map(goal, names, fallback):
    values = {name: float(fallback) for name in names}
    for tolerance in goal.goal_tolerance:
        if tolerance.name not in values:
            continue
        # FollowJointTrajectory uses -1 to erase a tolerance and 0 to retain
        # the controller default.
        if tolerance.position < 0.0:
            values[tolerance.name] = math.inf
        elif tolerance.position > 0.0:
            values[tolerance.name] = float(tolerance.position)
    return values


class PhysicalHandTrajectoryActionServer:
    def __init__(self):
        self.joint_names = list(rospy.get_param("~joint_names", DEFAULT_JOINTS))
        if len(self.joint_names) != 4 or len(set(self.joint_names)) != 4:
            raise ValueError("physical hand requires four unique active joints")
        self.command_topic = rospy.get_param(
            "~command_topic", "/controller_gazebo_hand/command"
        )
        self.action_name = rospy.get_param(
            "~action_name", "/controller_gazebo_hand/follow_joint_trajectory"
        )
        self.default_goal_tolerance = float(
            rospy.get_param("~default_goal_tolerance_rad", 0.05)
        )
        self.state_timeout_s = float(rospy.get_param("~state_timeout_s", 0.5))
        if self.default_goal_tolerance <= 0.0 or self.state_timeout_s <= 0.0:
            raise ValueError("physical hand tolerances/timeouts must be positive")

        self.lock = threading.Lock()
        self.positions = None
        self.state_wall_time = 0.0
        self.publisher = rospy.Publisher(
            self.command_topic, JointTrajectory, queue_size=1
        )
        self.subscriber = rospy.Subscriber(
            "/joint_states", JointState, self._state_callback, queue_size=20
        )
        self.server = actionlib.SimpleActionServer(
            self.action_name,
            FollowJointTrajectoryAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self.server.start()
        rospy.logwarn(
            "Physical hand trajectory action ready: %s -> %s",
            self.action_name,
            self.command_topic,
        )

    def _state_callback(self, message):
        values = dict(zip(message.name, message.position))
        if not all(name in values for name in self.joint_names):
            return
        positions = [values[name] for name in self.joint_names]
        if not all(math.isfinite(value) for value in positions):
            return
        with self.lock:
            self.positions = positions
            self.state_wall_time = time.monotonic()

    def _current(self):
        with self.lock:
            if self.positions is None:
                return None
            if time.monotonic() - self.state_wall_time > self.state_timeout_s:
                return None
            return list(self.positions)

    def _result(self, code, message):
        result = FollowJointTrajectoryResult()
        result.error_code = int(code)
        result.error_string = str(message)
        return result

    def _hold_current(self):
        positions = self._current()
        if positions is None:
            return
        message = JointTrajectory()
        message.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(0.05)
        message.points = [point]
        self.publisher.publish(message)

    def _feedback(self, desired):
        actual = self._current()
        if actual is None:
            return
        feedback = FollowJointTrajectoryFeedback()
        feedback.header.stamp = rospy.Time.now()
        feedback.joint_names = list(self.joint_names)
        feedback.desired.positions = list(desired)
        feedback.actual.positions = actual
        feedback.error.positions = [
            target - measured for target, measured in zip(desired, actual)
        ]
        self.server.publish_feedback(feedback)

    def _wait_until_start(self, start):
        while not rospy.is_shutdown() and rospy.Time.now() < start:
            if self.server.is_preempt_requested():
                self._hold_current()
                self.server.set_preempted(
                    self._result(FollowJointTrajectoryResult.SUCCESSFUL, "preempted")
                )
                return False
            time.sleep(0.01)
        return not rospy.is_shutdown()

    def _execute(self, goal):
        error = validate_trajectory(goal.trajectory, self.joint_names)
        if error:
            code = (
                FollowJointTrajectoryResult.INVALID_JOINTS
                if "joints" in error
                else FollowJointTrajectoryResult.INVALID_GOAL
            )
            self.server.set_aborted(self._result(code, error))
            return
        if self._current() is None:
            self.server.set_aborted(
                self._result(
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "fresh physical hand joint state is unavailable",
                )
            )
            return
        deadline = time.monotonic() + 2.0
        while self.publisher.get_num_connections() < 1 and time.monotonic() < deadline:
            if self.server.is_preempt_requested():
                self.server.set_preempted()
                return
            time.sleep(0.01)
        if self.publisher.get_num_connections() < 1:
            self.server.set_aborted(
                self._result(
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "physical hand Gazebo plugin is not connected",
                )
            )
            return

        now = rospy.Time.now()
        start = goal.trajectory.header.stamp
        if start == rospy.Time():
            start = now
        elif start < now:
            self.server.set_aborted(
                self._result(
                    FollowJointTrajectoryResult.OLD_HEADER_TIMESTAMP,
                    "trajectory header timestamp is in the past",
                )
            )
            return
        if not self._wait_until_start(start):
            return

        self.publisher.publish(goal.trajectory)
        final_by_message = dict(
            zip(goal.trajectory.joint_names, goal.trajectory.points[-1].positions)
        )
        desired = [final_by_message[name] for name in self.joint_names]
        finish = start + goal.trajectory.points[-1].time_from_start
        last_feedback = 0.0
        while not rospy.is_shutdown() and rospy.Time.now() < finish:
            if self.server.is_preempt_requested():
                self._hold_current()
                self.server.set_preempted(
                    self._result(FollowJointTrajectoryResult.SUCCESSFUL, "preempted")
                )
                return
            if time.monotonic() - last_feedback >= 0.05:
                self._feedback(desired)
                last_feedback = time.monotonic()
            time.sleep(0.01)

        tolerances = tolerance_map(
            goal, self.joint_names, self.default_goal_tolerance
        )
        settle_deadline = finish + goal.goal_time_tolerance
        # A zero goal_time_tolerance still receives one measured evaluation.
        while not rospy.is_shutdown():
            if self.server.is_preempt_requested():
                self._hold_current()
                self.server.set_preempted(
                    self._result(FollowJointTrajectoryResult.SUCCESSFUL, "preempted")
                )
                return
            actual = self._current()
            if actual is not None:
                errors = [
                    abs(target - measured)
                    for target, measured in zip(desired, actual)
                ]
                if all(
                    error <= tolerances[name]
                    for name, error in zip(self.joint_names, errors)
                ):
                    self.server.set_succeeded(
                        self._result(
                            FollowJointTrajectoryResult.SUCCESSFUL,
                            "physical hand trajectory completed",
                        )
                    )
                    return
            self._feedback(desired)
            if rospy.Time.now() >= settle_deadline:
                detail = "goal tolerance violated; target={} actual={}".format(
                    [round(value, 5) for value in desired],
                    None if actual is None else [round(value, 5) for value in actual],
                )
                self.server.set_aborted(
                    self._result(
                        FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED,
                        detail,
                    )
                )
                return
            time.sleep(0.01)


def main():
    rospy.init_node("physical_hand_trajectory_action_server")
    try:
        PhysicalHandTrajectoryActionServer()
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("Physical hand action server failed: %s", exc)
        raise SystemExit(6)


if __name__ == "__main__":
    main()
