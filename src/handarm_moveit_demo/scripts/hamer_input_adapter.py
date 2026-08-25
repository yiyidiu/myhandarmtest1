#!/usr/bin/env python3
"""Receive the explicit HaMeR/D455 metric wrist-pose UDP contract and publish ROS.

This adapter intentionally does not accept the legacy MediaPipe ``delta`` UDP
packet. Input schema ``handarm_hamer_pose_v1`` carries a D455-depth metric wrist
position and an SO(3) palm orientation from the existing HaMeR output adapter.
"""

import json
import math
import socket

import numpy as np
import rospy

from handarm_moveit_demo.msg import HamerHandPose
from handarm_moveit_demo.shared_teleop_core import (
    matrix_to_quaternion_xyzw, project_to_so3,
)


class HamerInputAdapter:
    def __init__(self):
        self.bind_ip = rospy.get_param("~bind_ip", "127.0.0.1")
        self.port = int(rospy.get_param("~port", 5010))
        self.topic = rospy.get_param("~output_topic", "/shared_teleop/hamer_pose")
        self.default_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        self.maximum_packet_bytes = int(rospy.get_param("~maximum_packet_bytes", 65535))
        self.publisher = rospy.Publisher(self.topic, HamerHandPose, queue_size=1)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.bind_ip, self.port))
        self.socket.setblocking(False)
        self.last_sequence = None
        self.session_id = None
        self.timer = rospy.Timer(rospy.Duration(0.005), self.poll)
        rospy.loginfo("HaMeR input adapter listening on udp://%s:%d -> %s",
                      self.bind_ip, self.port, self.topic)

    @staticmethod
    def _stamp(value):
        stamp = float(value)
        if not math.isfinite(stamp) or stamp <= 0.0:
            raise ValueError("stamp must be a positive wall-clock second value")
        return rospy.Time.from_sec(stamp)

    def convert(self, packet):
        if packet.get("schema") != "handarm_hamer_pose_v1":
            raise ValueError("unsupported schema")
        session = str(packet.get("session_id", ""))
        sequence = int(packet["sequence"])
        if not session:
            raise ValueError("session_id is required")
        if self.session_id == session and self.last_sequence is not None and sequence <= self.last_sequence:
            raise ValueError("duplicate_or_out_of_order_sequence")
        if self.session_id != session:
            self.session_id = session
            self.last_sequence = None
        position = np.asarray(packet["wrist_position_m"], dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("wrist_position_m must be a finite 3-vector")
        if "palm_rotation_row_major" in packet:
            rotation = project_to_so3(np.asarray(packet["palm_rotation_row_major"], dtype=float).reshape(3, 3))
            quaternion = matrix_to_quaternion_xyzw(rotation)
        else:
            quaternion = np.asarray(packet["palm_quaternion_xyzw"], dtype=float)
            if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
                raise ValueError("palm_quaternion_xyzw must be finite xyzw")
            norm = np.linalg.norm(quaternion)
            if norm < 1.0e-10:
                raise ValueError("zero quaternion")
            quaternion = quaternion / norm
        confidence = np.asarray(packet.get("confidence", [1.0] * 6), dtype=float)
        if confidence.shape != (6,) or not np.all(np.isfinite(confidence)):
            raise ValueError("confidence must be a finite six-vector")
        source_stamp = self._stamp(packet["stamp"]).to_sec()
        frame_id = str(packet.get("frame_id", self.default_frame))
        if frame_id != self.default_frame:
            raise ValueError("unexpected_camera_frame:{}".format(frame_id))
        # Fail closed for old camera processes: the live UDP adapter never
        # treats a pose as controllable unless it carries the explicit C-key
        # gate contract introduced with relative-pose tracking.
        control_enabled = packet.get("control_enabled") is True
        reference_epoch = int(packet.get("control_reference_epoch", 0))
        reference_token = str(packet.get("control_reference_token", ""))
        expected_token = "{}:{}".format(session, reference_epoch)
        if control_enabled and (
                reference_epoch <= 0 or reference_token != expected_token):
            raise ValueError("invalid_control_reference_token")
        message = HamerHandPose()
        # Watchdog age must stay in the same clock domain as rospy.Time.now(),
        # especially when Gazebo /use_sim_time is active.
        message.header.stamp = rospy.Time.now()
        message.source_timestamp = source_stamp
        message.header.seq = sequence
        message.header.frame_id = frame_id
        message.wrist_pose.position.x, message.wrist_pose.position.y, message.wrist_pose.position.z = position
        (message.wrist_pose.orientation.x, message.wrist_pose.orientation.y,
         message.wrist_pose.orientation.z, message.wrist_pose.orientation.w) = quaternion
        message.confidence = np.clip(confidence, 0.0, 1.0).tolist()
        message.control_gate_present = True
        message.control_enabled = control_enabled
        message.control_reference_epoch = max(0, reference_epoch)
        message.control_reference_token = reference_token
        message.valid = bool(packet.get("valid", True)) and control_enabled
        message.gesture = int(packet.get("gesture", 0))
        message.gesture_confidence = float(np.clip(packet.get("gesture_confidence", 0.0), 0.0, 1.0))
        message.invalid_reason = (
            str(packet.get("invalid_reason", ""))
            if control_enabled else "WAITING_FOR_OPERATOR_C_REFERENCE"
        )
        self.last_sequence = sequence
        return message

    def poll(self, _event):
        newest = None
        while True:
            try:
                payload, _ = self.socket.recvfrom(self.maximum_packet_bytes)
                newest = json.loads(payload.decode("utf-8"))
            except BlockingIOError:
                break
            except Exception as exc:
                rospy.logwarn_throttle(1.0, "HaMeR UDP receive error: %s", exc)
                break
        if newest is None:
            return
        try:
            message = self.convert(newest)
            self.publisher.publish(message)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "HaMeR UDP packet rejected: %s", exc)


def main():
    rospy.init_node("hamer_input_adapter")
    HamerInputAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
