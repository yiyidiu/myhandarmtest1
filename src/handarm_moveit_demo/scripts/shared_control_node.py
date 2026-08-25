#!/usr/bin/env python3
"""Blend human 6-D velocity with cancellable minimum orientation assistance."""

import json
import threading
import time

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from std_msgs.msg import String

from handarm_moveit_demo.msg import HandCommand
from handarm_moveit_demo.shared_teleop_core import (
    MinimumInterventionOrientationAssist, OrientationCandidate, compose_pose,
    flange_pose_for_fixed_center, matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix, select_nearest_candidate,
    side_grasp_candidates, top_grasp_candidate,
)


class SharedControlNode:
    def __init__(self):
        self.config = rospy.get_param("/shared_teleop", {})
        frames = self.config.get("frames", {})
        transforms = self.config.get("transforms", {})
        assist_config = self.config.get("assistance", {})
        topics = self.config.get("topics", {})
        self.base_frame = frames.get("base", "base_link")
        self.control_frame = frames.get("servo_control", "tool0")
        transform = transforms.get("servo_control_to_grasp_center", {})
        self.p_control_center = np.asarray(transform.get("translation_m", [0, 0, 0]), dtype=float)
        self.r_control_center = quaternion_xyzw_to_matrix(
            transform.get("quaternion_xyzw", [0, 0, 0, 1]))
        self.assist = MinimumInterventionOrientationAssist(
            self.p_control_center, self.r_control_center,
            angular_gain=assist_config.get("angular_gain", 1.5),
            position_gain=assist_config.get("position_gain", 2.0),
            maximum_assist_angular_speed=assist_config.get("maximum_angular_speed_radps", 0.45),
            maximum_assist_linear_speed=assist_config.get("maximum_linear_speed_mps", 0.08),
            rise_rate_per_s=assist_config.get("strength_rise_per_s", 1.5),
            fall_rate_per_s=assist_config.get("strength_fall_per_s", 3.0),
            opposition_dot_threshold=assist_config.get("opposition_dot_threshold", -0.002),
            opposition_duration_s=assist_config.get("opposition_duration_s", 0.20),
        )
        self.approach_axis = np.asarray(assist_config.get("approach_axis_grasp_center", [0, 0, 1]), dtype=float)
        self.table_normal = np.asarray(assist_config.get("table_normal_base", [0, 0, 1]), dtype=float)
        self.side_directions = assist_config.get("side_directions_base", {
            "left": [-1, 0, 0], "right": [1, 0, 0],
            "front": [0, 1, 0], "back": [0, -1, 0],
        })
        self.default_strength = float(assist_config.get("default_strength", 0.65))
        self.ik_enabled = bool(assist_config.get("require_moveit_ik", True))
        self.ik_service_name = assist_config.get("ik_service", "/compute_ik")
        self.ik_group = assist_config.get("move_group", "abbarm")
        self.ik_timeout = float(assist_config.get("ik_timeout_s", 0.08))
        self.listener = tf.TransformListener()
        self.ik = rospy.ServiceProxy(self.ik_service_name, GetPositionIK, persistent=True)
        self.lock = threading.Lock()
        self.latest = None
        self.last_tick = None
        self.last_candidates = []
        self.last_activation_error = "NONE"
        self.shutting_down = False
        self.publisher = rospy.Publisher(topics.get("assisted_command", "/shared_teleop/assisted_command"),
                                         HandCommand, queue_size=1)
        self.diagnostics = rospy.Publisher(topics.get("assist_diagnostics", "/shared_teleop/assist_diagnostics"),
                                           String, queue_size=1)
        rospy.Subscriber(topics.get("operator_command", "/shared_teleop/operator_command"),
                         HandCommand, self.command_callback, queue_size=1)
        rospy.Subscriber(topics.get("assist_request", "/shared_teleop/assist_request"),
                         String, self.assist_callback, queue_size=1)
        rate = float(self.config.get("control", {}).get("rate_hz", 50.0))
        self.timer = rospy.Timer(rospy.Duration(1.0/rate), self.tick)
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        self.shutting_down = True
        self.timer.shutdown()

    def command_callback(self, message):
        with self.lock:
            self.latest = message

    def current_control_pose(self):
        translation, quaternion = self.listener.lookupTransform(
            self.base_frame, self.control_frame, rospy.Time(0))
        return np.asarray(translation, dtype=float), quaternion_xyzw_to_matrix(quaternion)

    def ik_feasible(self, label, center_position, center_rotation):
        if not self.ik_enabled:
            return True
        control_position, control_rotation = flange_pose_for_fixed_center(
            center_position, center_rotation, self.p_control_center, self.r_control_center)
        request = GetPositionIKRequest()
        request.ik_request.group_name = self.ik_group
        request.ik_request.ik_link_name = self.control_frame
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout = rospy.Duration(self.ik_timeout)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.base_frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = control_position
        quaternion = matrix_to_quaternion_xyzw(control_rotation)
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = quaternion
        request.ik_request.pose_stamped = pose
        try:
            response = self.ik(request)
            return response.error_code.val == MoveItErrorCodes.SUCCESS
        except Exception as exc:
            rospy.logwarn("IK check failed for %s: %s", label, exc)
            return False

    def assist_callback(self, message):
        try:
            payload = json.loads(message.data) if message.data.strip().startswith("{") else {"mode": message.data.strip()}
            mode = str(payload.get("mode", "cancel")).lower()
            strength = float(np.clip(payload.get("strength", self.default_strength), 0.0, 1.0))
            if mode in ("cancel", "none", "off"):
                with self.lock:
                    self.assist.cancel()
                    self.last_candidates = []
                    self.last_activation_error = "NONE"
                rospy.loginfo("Orientation assistance cancelled by operator")
                return
            p_control, r_control = self.current_control_pose()
            p_center, r_center = compose_pose(p_control, r_control,
                                              self.p_control_center, self.r_control_center)
            if mode == "top":
                candidate = top_grasp_candidate(r_center, self.approach_axis, self.table_normal)
                candidate = OrientationCandidate(candidate.label, candidate.rotation,
                                                 candidate.distance_rad,
                                                 self.ik_feasible(candidate.label, p_center,
                                                                  candidate.rotation))
                candidates = [candidate]
            elif mode == "side":
                candidates = side_grasp_candidates(
                    r_center, self.approach_axis, self.side_directions,
                    feasibility=lambda label, rotation: self.ik_feasible(
                        label, p_center, rotation))
            else:
                raise ValueError("assist mode must be top, side, or cancel")
            selected = select_nearest_candidate(candidates)
            with self.lock:
                self.assist.activate(selected, strength)
                self.last_candidates = candidates
                self.last_activation_error = "NONE"
            rospy.loginfo("Orientation assistance selected %s (%.1f deg, strength %.2f)",
                          selected.label, np.degrees(selected.distance_rad), strength)
        except Exception as exc:
            with self.lock:
                self.last_activation_error = str(exc)
            rospy.logwarn("Orientation assistance request rejected: %s", exc)

    @staticmethod
    def command_vector(message):
        return np.array([message.twist.linear.x, message.twist.linear.y,
                         message.twist.linear.z, message.twist.angular.x,
                         message.twist.angular.y, message.twist.angular.z], dtype=float)

    def tick(self, event):
        if self.shutting_down or rospy.is_shutdown():
            return
        began = time.perf_counter()
        now = event.current_real.to_sec()
        dt = 0.0 if self.last_tick is None else max(0.0, now-self.last_tick)
        self.last_tick = now
        with self.lock:
            message = self.latest
        if message is None:
            return
        operator = self.command_vector(message) if message.valid else np.zeros(6)
        try:
            p_control, r_control = self.current_control_pose()
            with self.lock:
                result = self.assist.compute(dt, p_control, r_control, operator)
                candidates = list(self.last_candidates)
                activation_error = self.last_activation_error
            output_velocity = result.velocity
            valid = message.valid
            tf_reason = "NONE"
        except Exception as exc:
            output_velocity = np.zeros(6)
            valid = False
            result = None
            candidates = []
            activation_error = "TF_ERROR:{}".format(exc)
            tf_reason = activation_error
            rospy.logwarn_throttle(1.0, "Shared control waiting for TF %s -> %s: %s",
                                   self.base_frame, self.control_frame, exc)
        output = HandCommand()
        output.header = message.header  # preserve source time for the downstream watchdog
        output.confidence = message.confidence
        output.valid = valid
        output.gesture = message.gesture
        output.gesture_confidence = message.gesture_confidence
        (output.twist.linear.x, output.twist.linear.y, output.twist.linear.z,
         output.twist.angular.x, output.twist.angular.y, output.twist.angular.z) = output_velocity
        if self.shutting_down or rospy.is_shutdown():
            return
        self.publisher.publish(output)
        diagnostics = {
            "stamp": now,
            "source_stamp": message.header.stamp.to_sec(),
            "processing_ms": (time.perf_counter()-began)*1000.0,
            "actual_loop_hz": 0.0 if dt <= 0.0 else 1.0/dt,
            "mode": "none" if result is None else result.selected_label,
            "strength": 0.0 if result is None else result.strength,
            "opposing": False if result is None else result.opposing,
            "assist_velocity": [0.0]*6 if result is None else result.assist_velocity.tolist(),
            "selected": "none" if result is None else result.selected_label,
            "target_center_quaternion_xyzw": (None if result is None or result.target_center_rotation is None
                                                else matrix_to_quaternion_xyzw(result.target_center_rotation).tolist()),
            "target_control_position": (None if result is None or result.target_flange_position is None
                                          else result.target_flange_position.tolist()),
            "candidates": [{"label": candidate.label,
                             "distance_deg": float(np.degrees(candidate.distance_rad)),
                             "ik_feasible": candidate.feasible} for candidate in candidates],
            "kinematic_check": "MOVEIT_IK" if self.ik_enabled else "SKIPPED_NOT_VALIDATED",
            "activation_error": activation_error,
            "reason": tf_reason,
        }
        self.diagnostics.publish(String(data=json.dumps(diagnostics, separators=(",", ":"))))


def main():
    rospy.init_node("shared_control")
    SharedControlNode()
    rospy.spin()


if __name__ == "__main__":
    main()
