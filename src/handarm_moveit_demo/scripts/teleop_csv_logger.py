#!/usr/bin/env python3
"""Record synchronized stage-one shared-teleoperation state to CSV."""

import csv
import json
from pathlib import Path
import threading
import time

import rospy
import tf
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String

from handarm_moveit_demo.msg import HamerHandPose, HandCommand


FIELDS = [
    "timestamp_ros", "raw_hand_stamp", "raw_hand_source_stamp", "raw_hand_frame", "raw_hand_valid",
    "timing_contract_present", "source_capture_sequence", "dropped_capture_frames",
    "capture_to_publish_s", "inference_executed", "inference_call_s",
    "model_inference_s", "postprocess_s",
    "finger_tracking_present", "finger_tracking_valid",
    "human_finger_flexion", "finger_tracking_confidence",
    "finger_invalid_reason", "finger_retargeting_status",
    "finger_retargeting_calibrated", "finger_hold_required",
    "robot_finger_closure", "finger_desired_joint_target_rad",
    "finger_command_joint_target_rad", "finger_actual_joint_position_rad",
    "raw_hand_x", "raw_hand_y", "raw_hand_z", "raw_hand_qx", "raw_hand_qy",
    "raw_hand_qz", "raw_hand_qw", "relative_hand_x", "relative_hand_y",
    "relative_hand_z", "relative_hand_quaternion_xyzw", "raw_vx", "raw_vy",
    "raw_vz", "raw_wx", "raw_wy", "raw_wz", "processed_vx", "processed_vy",
    "processed_vz", "processed_wx", "processed_wy", "processed_wz", "confidence_x",
    "confidence_y", "confidence_z", "confidence_roll", "confidence_pitch",
    "confidence_yaw", "assist_strength", "assist_candidates_json",
    "selected_correction", "selected_correction_quaternion_xyzw", "actual_ee_pose",
    "input_output_latency_s", "control_loop_hz", "trend_processing_ms",
    "assist_processing_ms", "output_processing_ms", "gesture", "gesture_confidence",
    "hand_identity_present", "hand_is_right", "presence_generation",
    "active_hand_generation", "control_gate_present", "control_enabled",
    "control_reference_epoch", "control_reference_token",
    "gesture_status", "timeout_reason", "limit_reasons", "jump_reason",
    "invalid_reason", "safety_reasons", "robot_output_allowed",
]


def vector_from_command(message):
    return [message.twist.linear.x, message.twist.linear.y, message.twist.linear.z,
            message.twist.angular.x, message.twist.angular.y, message.twist.angular.z]


class TeleopCsvLogger:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        topics = config.get("topics", {})
        frames = config.get("frames", {})
        output_directory = Path(rospy.get_param(
            "~output_directory", "/tmp/handarm_shared_teleop_logs")).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        requested = str(rospy.get_param("~output_file", "")).strip()
        self.path = (Path(requested).expanduser().resolve() if requested else
                     output_directory / ("shared_teleop_" + time.strftime("%Y%m%dT%H%M%S") + ".csv"))
        self.handle = self.path.open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDS)
        self.writer.writeheader(); self.handle.flush()
        self.lock = threading.Lock()
        self.io_lock = threading.Lock()
        self.shutting_down = False
        self.state = {"hamer": None, "raw_command": None, "safe_twist": None,
                      "trend": {}, "assist": {}, "output": {}, "gesture": {},
                      "finger": {}}
        self.rows = 0
        self.base_frame = frames.get("base", "base_link")
        self.control_frame = frames.get("servo_control", "tool0")
        self.listener = tf.TransformListener()
        rospy.Subscriber(topics.get("hamer_pose", "/shared_teleop/hamer_pose"),
                         HamerHandPose, lambda msg: self.set_state("hamer", msg), queue_size=1)
        rospy.Subscriber(topics.get("raw_command", "/shared_teleop/raw_hand_command"),
                         HandCommand, lambda msg: self.set_state("raw_command", msg), queue_size=1)
        rospy.Subscriber(topics.get("safe_twist", "/shared_teleop/safe_twist"),
                         TwistStamped, lambda msg: self.set_state("safe_twist", msg), queue_size=1)
        rospy.Subscriber(topics.get("trend_diagnostics", "/shared_teleop/trend_diagnostics"),
                         String, lambda msg: self.set_json("trend", msg), queue_size=1)
        rospy.Subscriber(topics.get("assist_diagnostics", "/shared_teleop/assist_diagnostics"),
                         String, lambda msg: self.set_json("assist", msg), queue_size=1)
        rospy.Subscriber(topics.get("output_diagnostics", "/shared_teleop/output_diagnostics"),
                         String, lambda msg: self.set_json("output", msg), queue_size=1)
        rospy.Subscriber(topics.get("gesture_diagnostics", "/shared_teleop/gesture_diagnostics"),
                         String, lambda msg: self.set_json("gesture", msg), queue_size=1)
        rospy.Subscriber(topics.get("finger_diagnostics", "/shared_teleop/finger_diagnostics"),
                         String, lambda msg: self.set_json("finger", msg), queue_size=1)
        rate = float(config.get("logging", {}).get("rate_hz", 50.0))
        self.timer = rospy.Timer(rospy.Duration(1.0/rate), self.tick)
        rospy.on_shutdown(self.close)
        rospy.loginfo("Shared teleoperation CSV log: %s", self.path)

    def set_state(self, name, value):
        with self.lock:
            self.state[name] = value

    def set_json(self, name, message):
        try:
            value = json.loads(message.data)
        except Exception:
            return
        self.set_state(name, value)

    @staticmethod
    def json_value(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def tick(self, event):
        if self.shutting_down or rospy.is_shutdown():
            return
        with self.lock:
            state = dict(self.state)
        hamer = state["hamer"]
        if hamer is None:
            return
        raw = state["raw_command"]
        safe = state["safe_twist"]
        trend = state["trend"]; assist = state["assist"]
        output = state["output"]; gesture = state["gesture"]
        finger = state["finger"]
        row = dict.fromkeys(FIELDS, "")
        row.update({
            "timestamp_ros": event.current_real.to_sec(),
            "raw_hand_stamp": hamer.header.stamp.to_sec(),
            "raw_hand_source_stamp": hamer.source_timestamp,
            "raw_hand_frame": hamer.header.frame_id, "raw_hand_valid": hamer.valid,
            "timing_contract_present": hamer.timing_contract_present,
            "source_capture_sequence": hamer.source_capture_sequence,
            "dropped_capture_frames": hamer.dropped_capture_frames,
            "capture_to_publish_s": hamer.capture_to_publish_s,
            "inference_executed": hamer.inference_executed,
            "inference_call_s": hamer.inference_call_s,
            "model_inference_s": hamer.model_inference_s,
            "postprocess_s": hamer.postprocess_s,
            "finger_tracking_present": hamer.finger_tracking_present,
            "finger_tracking_valid": hamer.finger_tracking_valid,
            "human_finger_flexion": self.json_value(list(hamer.finger_flexion)),
            "finger_tracking_confidence": hamer.finger_tracking_confidence,
            "finger_invalid_reason": hamer.finger_invalid_reason,
            "finger_retargeting_status": finger.get("status"),
            "finger_retargeting_calibrated": finger.get("calibrated"),
            "finger_hold_required": finger.get("hold_required"),
            "robot_finger_closure": self.json_value(
                finger.get("normalized_robot_closure")),
            "finger_desired_joint_target_rad": self.json_value(
                finger.get("desired_joint_target_rad")),
            "finger_command_joint_target_rad": self.json_value(
                finger.get("command_joint_target_rad")),
            "finger_actual_joint_position_rad": self.json_value(
                finger.get("actual_joint_position_rad")),
            "raw_hand_x": hamer.wrist_pose.position.x,
            "raw_hand_y": hamer.wrist_pose.position.y,
            "raw_hand_z": hamer.wrist_pose.position.z,
            "raw_hand_qx": hamer.wrist_pose.orientation.x,
            "raw_hand_qy": hamer.wrist_pose.orientation.y,
            "raw_hand_qz": hamer.wrist_pose.orientation.z,
            "raw_hand_qw": hamer.wrist_pose.orientation.w,
            "gesture": hamer.gesture, "gesture_confidence": hamer.gesture_confidence,
            "hand_identity_present": hamer.hand_identity_present,
            "hand_is_right": hamer.hand_is_right,
            "presence_generation": hamer.presence_generation,
            "active_hand_generation": hamer.active_hand_generation,
            "control_gate_present": hamer.control_gate_present,
            "control_enabled": hamer.control_enabled,
            "control_reference_epoch": hamer.control_reference_epoch,
            "control_reference_token": hamer.control_reference_token,
            "invalid_reason": hamer.invalid_reason,
            "relative_hand_quaternion_xyzw": self.json_value(trend.get("relative_quaternion_xyzw")),
            "assist_strength": assist.get("strength", 0.0),
            "assist_candidates_json": self.json_value(assist.get("candidates", [])),
            "selected_correction": assist.get("selected", "none"),
            "selected_correction_quaternion_xyzw": self.json_value(
                assist.get("target_center_quaternion_xyzw")),
            "input_output_latency_s": output.get("source_input_age_s"),
            "control_loop_hz": output.get("actual_loop_hz"),
            "trend_processing_ms": trend.get("processing_ms"),
            "assist_processing_ms": assist.get("processing_ms"),
            "output_processing_ms": output.get("processing_ms"),
            "gesture_status": gesture.get("reason"),
            "timeout_reason": next((reason for reason in output.get("reasons", [])
                                    if "TIMEOUT" in reason), ""),
            "limit_reasons": self.json_value([reason for reason in output.get("reasons", [])
                                               if "WORKSPACE" in reason or "LIMIT" in reason]),
            "jump_reason": trend.get("reason") if "REJECTED" in str(trend.get("reason")) else "",
            "safety_reasons": self.json_value(output.get("reasons", [])),
            "robot_output_allowed": output.get("output_allowed"),
        })
        relative = trend.get("relative_position", [None]*3)
        for field, value in zip(("relative_hand_x", "relative_hand_y", "relative_hand_z"), relative):
            row[field] = value
        raw_velocity = trend.get("raw_velocity", [None]*6)
        for field, value in zip(("raw_vx","raw_vy","raw_vz","raw_wx","raw_wy","raw_wz"), raw_velocity):
            row[field] = value
        if safe is not None:
            safe_velocity = [safe.twist.linear.x, safe.twist.linear.y, safe.twist.linear.z,
                             safe.twist.angular.x, safe.twist.angular.y, safe.twist.angular.z]
            for field, value in zip(("processed_vx","processed_vy","processed_vz",
                                     "processed_wx","processed_wy","processed_wz"), safe_velocity):
                row[field] = value
        confidence = list(raw.confidence) if raw is not None else list(hamer.confidence)
        for field, value in zip(("confidence_x","confidence_y","confidence_z",
                                 "confidence_roll","confidence_pitch","confidence_yaw"), confidence):
            row[field] = value
        try:
            translation, quaternion = self.listener.lookupTransform(
                self.base_frame, self.control_frame, rospy.Time(0))
            row["actual_ee_pose"] = self.json_value(list(translation)+list(quaternion))
        except Exception:
            row["actual_ee_pose"] = ""
        with self.io_lock:
            if self.shutting_down or self.handle.closed:
                return
            self.writer.writerow(row)
            self.rows += 1
            if self.rows % 50 == 0:
                self.handle.flush()

    def close(self):
        self.shutting_down = True
        self.timer.shutdown()
        with self.io_lock:
            if getattr(self, "handle", None) is not None and not self.handle.closed:
                self.handle.flush(); self.handle.close()
                rospy.loginfo("Closed shared teleoperation CSV log (%d rows): %s", self.rows, self.path)


def main():
    rospy.init_node("teleop_csv_logger")
    TeleopCsvLogger()
    rospy.spin()


if __name__ == "__main__":
    main()
