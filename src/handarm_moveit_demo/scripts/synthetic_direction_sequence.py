#!/usr/bin/env python3
"""Deterministic 30 Hz wrist/palm sequence for Gazebo direction checks.

The node holds a fixed wrist pose until ``start_direction_test`` is called.
Every positive phase is followed by an equal return phase, so the test remains
near the collision-free starting pose.  Values are expressed in the same fixed
D455 optical frame as the live HaMeR packet.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from handarm_moveit_demo.msg import HamerHandPose
from handarm_moveit_demo.shared_teleop_core import matrix_to_quaternion_xyzw, so3_exp


@dataclass(frozen=True)
class Stage:
    label: str
    duration_s: float
    linear_camera_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_camera_radps: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def direction_stages():
    linear = 0.025
    angular = 0.22
    motion_s = 1.0
    target_hold_s = 0.60
    # The AprilTag V3 rotational feedback gain is deliberately only 0.5/s.
    # Give the simulated arm time to settle back to the shared C-zero before
    # exciting the next axis, otherwise the previous-axis residual pollutes
    # the independent-axis measurement.
    return_settle_s = 3.00
    stages = [Stage("settle", 0.80)]
    axes = [
        # AprilTag V3 relation ported to the MANO wrist:
        # camera -Z -> base +X; camera -X -> base +Y; camera -Y -> base +Z.
        ("translation_base_x", (0.0, 0.0, -linear), (0.0, 0.0, 0.0)),
        ("translation_base_y", (-linear, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("translation_base_z", (0.0, -linear, 0.0), (0.0, 0.0, 0.0)),
        # q_target = q_robot_zero * q_hand_delta, so these are captured
        # MANO/tool-local axes rather than fixed base axes.
        ("rotation_tool_x", (0.0, 0.0, 0.0), (angular, 0.0, 0.0)),
        ("rotation_tool_y", (0.0, 0.0, 0.0), (0.0, angular, 0.0)),
        ("rotation_tool_z", (0.0, 0.0, 0.0), (0.0, 0.0, angular)),
    ]
    for label, linear_velocity, angular_velocity in axes:
        stages.extend([
            Stage(label, motion_s, linear_velocity, angular_velocity),
            # Keep the positive phase label while holding the hand target so
            # the relative-pose controller can converge before measurement.
            Stage(label, target_hold_s),
            Stage(label + "_return", motion_s,
                  tuple(-value for value in linear_velocity),
                  tuple(-value for value in angular_velocity)),
            Stage(label + "_pause", return_settle_s),
        ])
    return stages


class SyntheticDirectionSequence:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        topics = config.get("topics", {})
        frames = config.get("frames", {})
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.frame_id = frames.get("camera", "camera_color_optical_frame")
        self.pose_topic = topics.get("hamer_pose", "/shared_teleop/hamer_pose")
        self.stages = direction_stages()
        self.origin = np.asarray([0.0, 0.0, 0.55], dtype=float)
        self.start_time = None
        self.sequence = 0
        self.last_label = None
        self.pose_publisher = rospy.Publisher(
            self.pose_topic, HamerHandPose, queue_size=1)
        self.phase_publisher = rospy.Publisher(
            "/shared_teleop/direction_test_phase", String, queue_size=1, latch=True)
        rospy.Service("/shared_teleop/start_direction_test", Trigger, self.start)
        rospy.Service("/shared_teleop/stop_direction_test", Trigger, self.stop)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.tick)

    def start(self, _request):
        self.start_time = rospy.get_time() + 0.50
        self.last_label = None
        return TriggerResponse(True, "direction sequence starts in 0.5 s")

    def stop(self, _request):
        self.start_time = None
        self.last_label = None
        return TriggerResponse(True, "direction sequence stopped at neutral hand pose")

    def state_at(self, elapsed_s):
        if elapsed_s < 0.0:
            return np.zeros(3), np.zeros(3), "armed"
        position = np.zeros(3)
        rotation_vector = np.zeros(3)
        remaining = elapsed_s
        for stage in self.stages:
            local = min(max(remaining, 0.0), stage.duration_s)
            position += np.asarray(stage.linear_camera_mps) * local
            rotation_vector += np.asarray(stage.angular_camera_radps) * local
            if remaining < stage.duration_s:
                return position, rotation_vector, stage.label
            remaining -= stage.duration_s
        return position, rotation_vector, "complete"

    def tick(self, event):
        if self.start_time is None:
            offset = np.zeros(3); rotation_vector = np.zeros(3)
            label = "waiting_for_start"
        else:
            offset, rotation_vector, label = self.state_at(
                event.current_real.to_sec() - self.start_time)
        quaternion = matrix_to_quaternion_xyzw(so3_exp(rotation_vector))
        message = HamerHandPose()
        message.header.seq = self.sequence
        message.header.stamp = event.current_real
        message.header.frame_id = self.frame_id
        message.source_timestamp = event.current_real.to_sec()
        position = self.origin + offset
        (message.wrist_pose.position.x, message.wrist_pose.position.y,
         message.wrist_pose.position.z) = position
        (message.wrist_pose.orientation.x, message.wrist_pose.orientation.y,
         message.wrist_pose.orientation.z, message.wrist_pose.orientation.w) = quaternion
        message.confidence = [0.98] * 6
        message.valid = True
        message.gesture = 0
        message.gesture_confidence = 0.0
        message.invalid_reason = ""
        self.pose_publisher.publish(message)
        if label != self.last_label:
            self.phase_publisher.publish(String(data=label))
            rospy.loginfo("Direction test phase: %s", label)
            self.last_label = label
        self.sequence += 1


def main():
    rospy.init_node("synthetic_direction_sequence")
    SyntheticDirectionSequence()
    rospy.spin()


if __name__ == "__main__":
    main()
