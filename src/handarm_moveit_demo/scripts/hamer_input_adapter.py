#!/usr/bin/env python3
"""Receive the explicit HaMeR/D455 metric wrist-pose UDP contract and publish ROS.

This adapter intentionally does not accept the legacy MediaPipe ``delta`` UDP
packet. Valid pose packets and no-pose INVALID heartbeats share the
``handarm_hamer_pose_v1`` schema. A watchdog publishes an explicit locked ROS
message if the UDP producer disappears or sends a malformed packet.
"""

import json
import socket
import time

import rospy

from handarm_moveit_demo.msg import HamerHandPose
from handarm_moveit_demo.hamer_input_contract import (
    HamerPacketContract,
    InputWatchdog,
    ReferenceTokenInterlock,
)


class HamerInputAdapter:
    def __init__(self):
        self.bind_ip = rospy.get_param("~bind_ip", "127.0.0.1")
        self.port = int(rospy.get_param("~port", 5010))
        self.topic = rospy.get_param("~output_topic", "/shared_teleop/hamer_pose")
        self.default_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        self.maximum_packet_bytes = int(rospy.get_param("~maximum_packet_bytes", 65535))
        self.input_timeout_s = float(rospy.get_param("~input_timeout_s", 0.40))
        self.maximum_pipeline_latency_s = float(
            rospy.get_param("~maximum_pipeline_latency_s", 0.20)
        )
        self.require_timing_contract = bool(
            rospy.get_param("~require_timing_contract", True)
        )
        self.watchdog_publish_period_s = float(
            rospy.get_param("~watchdog_publish_period_s", 0.10)
        )
        self.publisher = rospy.Publisher(self.topic, HamerHandPose, queue_size=1)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.bind_ip, self.port))
        self.socket.setblocking(False)
        self.contract = HamerPacketContract(
            self.default_frame,
            maximum_pipeline_latency_s=self.maximum_pipeline_latency_s,
            require_timing_contract=self.require_timing_contract,
        )
        self.reference_interlock = ReferenceTokenInterlock()
        self.watchdog = InputWatchdog(
            self.input_timeout_s, self.watchdog_publish_period_s
        )
        self.fail_closed_sequence = 0
        self.timer = rospy.Timer(rospy.Duration(0.005), self.poll)
        rospy.loginfo(
            "HaMeR input adapter listening on udp://%s:%d -> %s "
            "(fail-closed timeout %.3f s, producer latency %.3f s, "
            "timing required=%s)",
            self.bind_ip,
            self.port,
            self.topic,
            self.input_timeout_s,
            self.maximum_pipeline_latency_s,
            self.require_timing_contract,
        )

    def convert(self, packet):
        normalized = self.contract.validate(packet)
        self.reference_interlock.accept(normalized)
        message = HamerHandPose()
        # Watchdog age must stay in the same clock domain as rospy.Time.now(),
        # especially when Gazebo /use_sim_time is active.
        message.header.stamp = rospy.Time.now()
        message.source_timestamp = normalized["source_stamp"]
        message.timing_contract_present = normalized["timing_contract_present"]
        message.source_capture_sequence = normalized["source_capture_sequence"]
        message.dropped_capture_frames = normalized["dropped_capture_frames"]
        message.capture_to_publish_s = normalized["capture_to_publish_s"]
        message.inference_executed = normalized["inference_executed"]
        message.inference_call_s = normalized["inference_call_s"]
        message.model_inference_s = normalized["model_inference_s"]
        message.postprocess_s = normalized["postprocess_s"]
        message.header.seq = normalized["sequence"]
        message.header.frame_id = normalized["frame_id"]
        (message.wrist_pose.position.x, message.wrist_pose.position.y,
         message.wrist_pose.position.z) = normalized["position"]
        (message.wrist_pose.orientation.x, message.wrist_pose.orientation.y,
         message.wrist_pose.orientation.z,
         message.wrist_pose.orientation.w) = normalized["quaternion"]
        message.confidence = normalized["confidence"].tolist()
        message.hand_identity_present = normalized["hand_identity_present"]
        message.hand_is_right = normalized["hand_is_right"]
        message.presence_generation = normalized["presence_generation"]
        message.active_hand_generation = normalized["active_hand_generation"]
        message.control_gate_present = True
        message.control_enabled = normalized["control_enabled"]
        message.control_reference_epoch = normalized["control_reference_epoch"]
        message.control_reference_token = normalized["control_reference_token"]
        message.valid = bool(
            normalized["observation_valid"] and normalized["control_enabled"]
        )
        message.gesture = normalized["gesture"]
        message.gesture_confidence = normalized["gesture_confidence"]
        message.invalid_reason = (
            normalized["invalid_reason"]
            if normalized["control_enabled"]
            else (
                normalized["invalid_reason"]
                or "WAITING_FOR_OPERATOR_C_REFERENCE"
            )
        )
        return message

    def fail_closed_message(self, reason):
        """Return a geometry-free ROS status that clears every active C token."""

        self.fail_closed_sequence += 1
        message = HamerHandPose()
        message.header.stamp = rospy.Time.now()
        message.header.seq = self.fail_closed_sequence
        message.header.frame_id = self.default_frame
        message.source_timestamp = 0.0
        message.timing_contract_present = False
        message.source_capture_sequence = 0
        message.dropped_capture_frames = 0
        message.capture_to_publish_s = 0.0
        message.inference_executed = False
        message.inference_call_s = 0.0
        message.model_inference_s = 0.0
        message.postprocess_s = 0.0
        message.wrist_pose.orientation.w = 1.0
        message.confidence = [0.0] * 6
        message.valid = False
        message.invalid_reason = str(reason or "HAMER_INPUT_FAIL_CLOSED")
        message.hand_identity_present = False
        message.hand_is_right = False
        message.presence_generation = 0
        message.active_hand_generation = 0
        message.control_gate_present = True
        message.control_enabled = False
        message.control_reference_epoch = 0
        message.control_reference_token = ""
        message.gesture = 0
        message.gesture_confidence = 0.0
        return message

    def publish_fail_closed(self, reason, rejected_packet=None):
        rejected_token = ""
        if isinstance(rejected_packet, dict) and (
            rejected_packet.get("control_enabled") is True
        ):
            rejected_token = str(
                rejected_packet.get("control_reference_token", "")
            )
        self.reference_interlock.require_new_reference(rejected_token)
        self.publisher.publish(self.fail_closed_message(reason))

    def poll(self, _event):
        newest = None
        receive_error = None
        while True:
            try:
                payload, _ = self.socket.recvfrom(self.maximum_packet_bytes)
                newest = json.loads(payload.decode("utf-8"))
            except BlockingIOError:
                break
            except Exception as exc:
                receive_error = exc
                break
        if newest is not None:
            try:
                message = self.convert(newest)
                self.publisher.publish(message)
                self.watchdog.mark_accepted(time.monotonic())
                return
            except Exception as exc:
                self.publish_fail_closed(
                    "HAMER_UDP_PACKET_REJECTED:{}:{}".format(
                        type(exc).__name__, exc
                    ),
                    rejected_packet=newest,
                )
                rospy.logwarn_throttle(
                    1.0, "HaMeR UDP packet rejected and control locked: %s", exc
                )
                return
        if receive_error is not None:
            self.publish_fail_closed(
                "HAMER_UDP_RECEIVE_ERROR:{}:{}".format(
                    type(receive_error).__name__, receive_error
                )
            )
            rospy.logwarn_throttle(
                1.0, "HaMeR UDP receive error and control locked: %s",
                receive_error,
            )
            return
        if self.watchdog.timeout_due(time.monotonic()):
            self.publish_fail_closed("HAMER_UDP_INPUT_TIMEOUT_REQUIRES_NEW_C")
            rospy.logwarn_throttle(
                1.0,
                "No accepted HaMeR UDP packet for %.3f s; control locked and "
                "a new C reference is required",
                self.input_timeout_s,
            )


def main():
    rospy.init_node("hamer_input_adapter")
    HamerInputAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
