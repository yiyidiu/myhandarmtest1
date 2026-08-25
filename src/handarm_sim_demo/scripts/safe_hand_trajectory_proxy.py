#!/usr/bin/env python3
"""Fail-closed collision-validating proxy for the Gazebo hand controller."""

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
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from urdf_parser_py.urdf import URDF


DEFAULT_JOINTS = ["f1j1", "f1j2", "f2j1", "f3j2"]


def validate_goal_trajectory(trajectory, required_names, limits):
    if len(trajectory.joint_names) != len(set(trajectory.joint_names)):
        return "trajectory contains duplicate joint names"
    if set(trajectory.joint_names) != set(required_names):
        return "trajectory joints must be exactly {}".format(required_names)
    if not trajectory.points:
        return "trajectory has no points"
    previous_time = 0.0
    for index, point in enumerate(trajectory.points):
        stamp = point.time_from_start.to_sec()
        if not math.isfinite(stamp) or stamp <= previous_time:
            return "point times must be finite and strictly increase from zero"
        if len(point.positions) != len(trajectory.joint_names):
            return "point {} has the wrong position count".format(index)
        for name, value in zip(trajectory.joint_names, point.positions):
            if not math.isfinite(value):
                return "point {} contains a non-finite position".format(index)
            lower, upper = limits[name]
            if value < lower or value > upper:
                return "point {} violates {} limits [{}, {}]".format(
                    index, name, lower, upper
                )
        previous_time = stamp
    return ""


def interpolate_segment(start, finish, maximum_step_rad):
    if len(start) != len(finish):
        raise ValueError("trajectory segment dimensions differ")
    largest_delta = max(abs(a - b) for a, b in zip(start, finish))
    steps = max(1, int(math.ceil(largest_delta / maximum_step_rad)))
    return [
        [a + (b - a) * float(index) / float(steps)
         for a, b in zip(start, finish)]
        for index in range(1, steps + 1)
    ]


def sampled_path(current, trajectory, required_names, maximum_step_rad):
    indices = [trajectory.joint_names.index(name) for name in required_names]
    previous = list(current)
    samples = []
    for point in trajectory.points:
        target = [point.positions[index] for index in indices]
        samples.extend(interpolate_segment(previous, target, maximum_step_rad))
        previous = target
    return samples


class SafeHandTrajectoryProxy:
    def __init__(self):
        self.joint_names = list(rospy.get_param("~joint_names", DEFAULT_JOINTS))
        if len(self.joint_names) != 4 or len(set(self.joint_names)) != 4:
            raise ValueError("safe hand proxy requires four unique active joints")
        self.public_action = str(rospy.get_param(
            "~public_action",
            "/controller_gazebo_hand/follow_joint_trajectory"))
        self.internal_action = str(rospy.get_param(
            "~internal_action",
            "/controller_gazebo_hand_internal/follow_joint_trajectory"))
        self.strict_service_name = str(rospy.get_param(
            "~strict_service",
            "/full_robot_self_collision_guard/check_state_validity"))
        self.motion_interlock_service_name = str(rospy.get_param(
            "~motion_interlock_service",
            "/full_robot_self_collision_guard/set_hand_motion_active"))
        self.maximum_collision_step_rad = float(rospy.get_param(
            "~maximum_collision_step_rad", 0.02))
        self.state_timeout_s = float(rospy.get_param("~state_timeout_s", 0.25))
        self.startup_timeout_s = float(rospy.get_param(
            "~startup_timeout_s", 45.0))
        if (not math.isfinite(self.maximum_collision_step_rad) or
                self.maximum_collision_step_rad <= 0.0 or
                self.maximum_collision_step_rad > 0.02 or
                not math.isfinite(self.state_timeout_s) or
                self.state_timeout_s <= 0.0 or
                not math.isfinite(self.startup_timeout_s) or
                self.startup_timeout_s <= 0.0):
            raise ValueError("invalid safe hand proxy timing/sampling parameters")

        robot = URDF.from_parameter_server("/robot_description")
        joints = {joint.name: joint for joint in robot.joints}
        self.limits = {}
        for name in self.joint_names:
            joint = joints.get(name)
            if joint is None or joint.limit is None:
                raise ValueError("missing URDF limit for {}".format(name))
            lower = float(joint.limit.lower)
            upper = float(joint.limit.upper)
            if not (math.isfinite(lower) and math.isfinite(upper) and
                    lower < upper):
                raise ValueError("invalid URDF limit for {}".format(name))
            self.limits[name] = (lower, upper)

        self.lock = threading.Lock()
        self.positions = None
        self.state_wall_time = 0.0
        rospy.Subscriber(
            "/joint_states", JointState, self._joint_state_callback,
            queue_size=2, tcp_nodelay=True)

        self.internal_client = actionlib.SimpleActionClient(
            self.internal_action, FollowJointTrajectoryAction)
        if not self.internal_client.wait_for_server(
                rospy.Duration(self.startup_timeout_s)):
            raise RuntimeError("internal hand trajectory controller unavailable")
        rospy.wait_for_service(
            self.strict_service_name, timeout=self.startup_timeout_s)
        rospy.wait_for_service(
            self.motion_interlock_service_name,
            timeout=self.startup_timeout_s)
        self.strict_service = rospy.ServiceProxy(
            self.strict_service_name, GetStateValidity, persistent=True)
        self.motion_interlock = rospy.ServiceProxy(
            self.motion_interlock_service_name, SetBool, persistent=True)

        self.server = actionlib.SimpleActionServer(
            self.public_action, FollowJointTrajectoryAction,
            execute_cb=self._execute, auto_start=False)
        self.server.start()
        rospy.logwarn(
            "Collision-checked hand action ACTIVE: %s -> %s; sample step <= %.4f rad",
            self.public_action, self.internal_action,
            self.maximum_collision_step_rad)

    def _joint_state_callback(self, message):
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
            if (self.positions is None or
                    time.monotonic() - self.state_wall_time >
                    self.state_timeout_s):
                return None
            return list(self.positions)

    @staticmethod
    def _result(code, message):
        result = FollowJointTrajectoryResult()
        result.error_code = int(code)
        result.error_string = str(message)
        return result

    def _candidate_is_valid(self, positions):
        request = GetStateValidityRequest()
        request.robot_state = RobotState()
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(self.joint_names)
        request.robot_state.joint_state.position = list(positions)
        try:
            response = self.strict_service(request)
        except (rospy.ServiceException, rospy.ROSException) as exc:
            # Recreate a broken persistent connection once, but never forward
            # the hand command unless the strict service answers valid.
            rospy.logerr("Strict hand collision query failed: %s", exc)
            self.strict_service.close()
            self.strict_service = rospy.ServiceProxy(
                self.strict_service_name, GetStateValidity, persistent=True)
            return False
        return bool(response.valid)

    def _validate_collision_path(self, trajectory, current):
        samples = sampled_path(
            current, trajectory, self.joint_names,
            self.maximum_collision_step_rad)
        if not samples:
            return "empty sampled collision path"
        for index, positions in enumerate(samples):
            if not self._candidate_is_valid(positions):
                return (
                    "strict full-robot self-collision check rejected "
                    "sample {}/{} at {}".format(
                        index + 1, len(samples),
                        [round(value, 5) for value in positions]))
        return ""

    def _feedback(self, message):
        feedback = FollowJointTrajectoryFeedback()
        feedback.header = message.header
        feedback.joint_names = list(message.joint_names)
        feedback.desired = message.desired
        feedback.actual = message.actual
        feedback.error = message.error
        self.server.publish_feedback(feedback)

    def _set_motion_interlock(self, active):
        try:
            response = self.motion_interlock(bool(active))
        except (rospy.ServiceException, rospy.ROSException) as exc:
            rospy.logerr("Full-robot hand motion interlock failed: %s", exc)
            self.motion_interlock.close()
            self.motion_interlock = rospy.ServiceProxy(
                self.motion_interlock_service_name, SetBool, persistent=True)
            return False
        if not response.success:
            rospy.logerr(
                "Full-robot hand motion interlock rejected request: %s",
                response.message)
            return False
        return True

    def _execute(self, goal):
        error = validate_goal_trajectory(
            goal.trajectory, self.joint_names, self.limits)
        if error:
            code = (FollowJointTrajectoryResult.INVALID_JOINTS
                    if "joints" in error else
                    FollowJointTrajectoryResult.INVALID_GOAL)
            self.server.set_aborted(self._result(code, error))
            return
        # The arm and hand must not independently execute two trajectories that
        # were each validated against a different frozen state. Acquire the
        # guard-owned interlock first; it synchronously commands zero arm
        # velocity. The complete hand path is then checked by MoveIt/FCL with
        # that arm configuration held fixed.
        if not self._set_motion_interlock(True):
            self.server.set_aborted(self._result(
                FollowJointTrajectoryResult.INVALID_GOAL,
                "could not acquire full-robot hand motion interlock"))
            return
        try:
            time.sleep(0.04)
            current = self._current()
            if current is None:
                self.server.set_aborted(self._result(
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "fresh complete hand JointState is unavailable"))
                return
            error = self._validate_collision_path(goal.trajectory, current)
            if error:
                rospy.logerr("HAND TRAJECTORY BLOCKED: %s", error)
                self.server.set_aborted(self._result(
                    FollowJointTrajectoryResult.INVALID_GOAL, error))
                return

            self.internal_client.send_goal(goal, feedback_cb=self._feedback)
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    self.internal_client.cancel_goal()
                    self.server.set_preempted(self._result(
                        FollowJointTrajectoryResult.SUCCESSFUL, "preempted"))
                    return
                if self.internal_client.wait_for_result(rospy.Duration(0.02)):
                    break
            if rospy.is_shutdown():
                self.internal_client.cancel_goal()
                return
            result = self.internal_client.get_result()
            if result is None:
                self.server.set_aborted(self._result(
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "internal hand controller returned no result"))
            elif result.error_code == FollowJointTrajectoryResult.SUCCESSFUL:
                self.server.set_succeeded(result)
            else:
                self.server.set_aborted(result)
        finally:
            # A failed release leaves the guard fail-safe (arm stopped). Never
            # bypass the interlock merely to recover motion.
            if not self._set_motion_interlock(False):
                rospy.logfatal(
                    "Hand motion interlock could not be released; arm remains stopped")


def main():
    rospy.init_node("safe_hand_trajectory_proxy")
    try:
        SafeHandTrajectoryProxy()
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("Safe hand trajectory proxy failed: %s", exc)
        raise SystemExit(9)


if __name__ == "__main__":
    main()
