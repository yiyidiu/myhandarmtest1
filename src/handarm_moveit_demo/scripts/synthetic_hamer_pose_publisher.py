#!/usr/bin/env python3
"""30 Hz synthetic HaMeR wrist pose for hardware-free shared-teleop testing."""

import math

import numpy as np
import rospy

from handarm_moveit_demo.msg import HamerHandPose
from handarm_moveit_demo.shared_teleop_core import (
    GESTURE_CLOSE, GESTURE_NONE, matrix_to_quaternion_xyzw, so3_exp,
)


class SyntheticHamerPublisher:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        topics = config.get("topics", {})
        frames = config.get("frames", {})
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.enable_gesture = bool(rospy.get_param("~enable_gesture", True))
        self.publisher = rospy.Publisher(topics.get("hamer_pose", "/shared_teleop/hamer_pose"),
                                         HamerHandPose, queue_size=1)
        self.frame_id = frames.get("camera", "camera_color_optical_frame")
        self.started = rospy.get_time()
        self.sequence = 0
        self.timer = rospy.Timer(rospy.Duration(1.0/self.rate_hz), self.tick)

    def tick(self, event):
        elapsed = event.current_real.to_sec()-self.started
        # All six components vary together with different frequencies/phases.
        position = np.array([
            0.04*math.sin(0.55*elapsed),
            0.03*math.sin(0.73*elapsed+0.4),
            0.025*math.sin(0.41*elapsed+0.8),
        ]) + np.array([0.0, 0.0, 0.55])
        rotation_vector = np.array([
            0.30*math.sin(0.49*elapsed),
            0.25*math.sin(0.61*elapsed+0.5),
            0.35*math.sin(0.37*elapsed+0.9),
        ])
        quaternion = matrix_to_quaternion_xyzw(so3_exp(rotation_vector))
        phase = elapsed % 10.0
        gesture = GESTURE_CLOSE if self.enable_gesture and 6.0 <= phase < 6.7 else GESTURE_NONE
        message = HamerHandPose()
        message.header.seq = self.sequence
        message.header.stamp = event.current_real
        message.source_timestamp = event.current_real.to_sec()
        message.header.frame_id = self.frame_id
        message.wrist_pose.position.x, message.wrist_pose.position.y, message.wrist_pose.position.z = position
        (message.wrist_pose.orientation.x, message.wrist_pose.orientation.y,
         message.wrist_pose.orientation.z, message.wrist_pose.orientation.w) = quaternion
        message.confidence = [0.92, 0.88, 0.85, 0.83, 0.80, 0.86]
        message.valid = True
        message.gesture = gesture
        message.gesture_confidence = 0.92 if gesture else 0.0
        message.invalid_reason = ""
        self.publisher.publish(message)
        self.sequence += 1


def main():
    rospy.init_node("synthetic_hamer_pose_publisher")
    SyntheticHamerPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
