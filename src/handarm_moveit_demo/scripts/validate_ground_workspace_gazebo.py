#!/usr/bin/env python3
"""Deterministic UDP -> normalized workspace -> Servo -> Gazebo acceptance.

This tool is simulation-only.  It creates the network equivalent of one C
press, exercises the positive camera-X translation, the ground direction and
the positive local-X wrist rotation, and verifies that every motion returns to
the captured tool0 zero.  The normal camera process must not own UDP port 5010
while this validator is running.
"""

import argparse
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import rospy
import tf2_ros
import yaml
from std_msgs.msg import Int8, String

from handarm_moveit_demo.shared_teleop_core import (
    quaternion_xyzw_to_matrix, so3_log,
)


DANGEROUS_SERVO_STATUSES = (2, 4, 5)


def parse_args():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument(
        "--calibration",
        default=str(package / "config/camera_workspace_calibration.yaml"))
    parser.add_argument(
        "--output", default="/tmp/handarm_ground_workspace_gazebo.json")
    parser.add_argument("--rate-hz", type=float, default=25.0)
    parser.add_argument("--ramp-s", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=1.2)
    parser.add_argument("--return-hold-s", type=float, default=1.5)
    parser.add_argument("--return-position-tolerance-m", type=float,
                        default=0.003)
    parser.add_argument("--return-rotation-tolerance-deg", type=float,
                        default=1.0)
    parser.add_argument("--peak-position-error-tolerance-m", type=float,
                        default=0.010)
    parser.add_argument("--peak-rotation-error-tolerance-deg", type=float,
                        default=3.0)
    return parser.parse_args(rospy.myargv()[1:])


def axis_rotation(axis, angle_rad):
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    if axis == 0:
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ])
    if axis == 1:
        return np.array([
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ])
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


class GroundWorkspaceGazeboValidator:
    def __init__(self, args):
        self.args = args
        calibration = yaml.safe_load(
            Path(args.calibration).expanduser().read_text(encoding="utf-8"))
        self.translation_positive = np.asarray(
            calibration["human_workspace"]["positive_extent_m"],
            dtype=np.float64)
        self.rotation_positive = np.radians(np.asarray(
            calibration["human_orientation"]["positive_extent_deg"],
            dtype=np.float64))
        if (self.translation_positive.shape != (3,) or
                self.rotation_positive.shape != (3,) or
                not np.all(self.translation_positive > 0.0) or
                not np.all(self.rotation_positive > 0.0)):
            raise ValueError("calibration extents must be positive 3-vectors")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session = "ground-workspace-validation-{}".format(
            int(time.time() * 1000.0))
        self.sequence = 0
        self.latest_diagnostic = {}
        self.servo_statuses = []
        self.buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buffer)
        rospy.Subscriber(
            "/shared_teleop/trend_diagnostics", String,
            self.diagnostic_callback, queue_size=10)
        rospy.Subscriber(
            "/servo_server/status", Int8,
            lambda message: self.servo_statuses.append(int(message.data)),
            queue_size=20)

    def diagnostic_callback(self, message):
        try:
            self.latest_diagnostic = json.loads(message.data)
        except (TypeError, ValueError):
            pass

    def tool_pose(self):
        transform = self.buffer.lookup_transform(
            "base_link", "tool0", rospy.Time(0), rospy.Duration(0.10))
        position = transform.transform.translation
        quaternion = transform.transform.rotation
        return (
            np.asarray([position.x, position.y, position.z], dtype=np.float64),
            quaternion_xyzw_to_matrix([
                quaternion.x, quaternion.y, quaternion.z, quaternion.w,
            ]),
        )

    def send(self, hand_position, hand_rotation):
        self.sequence += 1
        packet = {
            "schema": "handarm_hamer_pose_v1",
            "session_id": self.session,
            "sequence": self.sequence,
            "stamp": time.time(),
            "frame_id": "camera_color_optical_frame",
            "wrist_position_m": np.asarray(
                hand_position, dtype=np.float64).tolist(),
            "palm_rotation_row_major": np.asarray(
                hand_rotation, dtype=np.float64).reshape(-1).tolist(),
            "confidence": [1.0] * 6,
            "valid": True,
            "gesture": 0,
            "gesture_confidence": 0.0,
            "invalid_reason": "",
            "control_enabled": True,
            "control_reference_epoch": 1,
            "control_reference_token": self.session + ":1",
        }
        self.socket.sendto(
            json.dumps(packet, separators=(",", ":")).encode("utf-8"),
            (self.args.host, self.args.port))

    def publish_stage(self, start_position, end_position,
                      rotation_axis, start_angle, end_angle,
                      duration_s, hold_s):
        began = time.monotonic()
        period = 1.0 / self.args.rate_hz
        while not rospy.is_shutdown() and time.monotonic() - began < duration_s:
            fraction = min(1.0, (time.monotonic() - began) / duration_s)
            position = ((1.0 - fraction) * start_position +
                        fraction * end_position)
            angle = (1.0 - fraction) * start_angle + fraction * end_angle
            self.send(position, axis_rotation(rotation_axis, angle))
            time.sleep(period)
        began = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() - began < hold_s:
            self.send(end_position, axis_rotation(rotation_axis, end_angle))
            time.sleep(period)
        return self.tool_pose(), dict(self.latest_diagnostic)

    @staticmethod
    def rotation_error_deg(reference, current):
        return math.degrees(float(np.linalg.norm(so3_log(
            reference.T @ current))))

    @staticmethod
    def diagnostic_vector_norm(diagnostic, key):
        value = diagnostic.get(key)
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float64)
        return float(np.linalg.norm(vector))

    def run(self):
        if not bool(rospy.get_param("/use_sim_time", False)):
            raise RuntimeError("refusing to move: /use_sim_time is not true")
        published = {name for name, _type in rospy.get_published_topics()}
        if "/gazebo/model_states" not in published:
            raise RuntimeError("refusing to move: Gazebo model_states is absent")

        deadline = time.monotonic() + 10.0
        while True:
            try:
                self.tool_pose()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("base_link -> tool0 TF was not ready")
                time.sleep(0.10)

        neutral = np.array([0.0, 0.0, 0.55], dtype=np.float64)
        identity = np.eye(3)
        records = []
        (zero_position, zero_rotation), diagnostic = self.publish_stage(
            neutral, neutral, 0, 0.0, 0.0, 1.5, 1.0)
        if diagnostic.get("mapping_profile") != "camera_ground_workspace":
            raise RuntimeError(
                "camera_ground_workspace mapping profile is not active")

        def exercise(name, end_position, axis, end_angle):
            (peak_position, peak_rotation), peak_diagnostic = self.publish_stage(
                neutral, end_position, axis, 0.0, end_angle,
                self.args.ramp_s, self.args.hold_s)
            (return_position, return_rotation), return_diagnostic = (
                self.publish_stage(
                    end_position, neutral, axis, end_angle, 0.0,
                    self.args.ramp_s, self.args.return_hold_s))
            record = {
                "name": name,
                "tool_translation_from_zero_m": (
                    peak_position - zero_position).tolist(),
                "tool_rotation_from_zero_deg": self.rotation_error_deg(
                    zero_rotation, peak_rotation),
                "return_position_error_m": float(np.linalg.norm(
                    return_position - zero_position)),
                "return_rotation_error_deg": self.rotation_error_deg(
                    zero_rotation, return_rotation),
                "peak_target_position_error_m": self.diagnostic_vector_norm(
                    peak_diagnostic, "position_error_m"),
                "peak_target_rotation_error_deg": (
                    None if peak_diagnostic.get("rotation_error_rad") is None
                    else math.degrees(self.diagnostic_vector_norm(
                        peak_diagnostic, "rotation_error_rad"))),
                "peak_diagnostic": peak_diagnostic,
                "return_diagnostic": return_diagnostic,
            }
            records.append(record)
            return record

        right = exercise(
            "CAMERA_POSITIVE_X_TO_BASE_NEGATIVE_Y",
            neutral + np.array([self.translation_positive[0], 0.0, 0.0]),
            0, 0.0)
        ground = exercise(
            "CAMERA_POSITIVE_Y_TO_BASE_NEGATIVE_Z_GROUND_LIMIT",
            neutral + np.array([0.0, self.translation_positive[1], 0.0]),
            0, 0.0)
        roll = exercise(
            "CAMERA_LOCAL_POSITIVE_X_ROTATION",
            neutral, 0, self.rotation_positive[0])

        return_ok = all(
            record["return_position_error_m"] <=
            self.args.return_position_tolerance_m and
            record["return_rotation_error_deg"] <=
            self.args.return_rotation_tolerance_deg
            for record in records)
        target_tracking_ok = all(
            record["peak_target_position_error_m"] is not None and
            record["peak_target_position_error_m"] <=
            self.args.peak_position_error_tolerance_m and
            record["peak_target_rotation_error_deg"] is not None and
            record["peak_target_rotation_error_deg"] <=
            self.args.peak_rotation_error_tolerance_deg
            for record in records)
        dangerous = sorted(set(
            code for code in self.servo_statuses
            if code in DANGEROUS_SERVO_STATUSES))
        passed = bool(
            right["tool_translation_from_zero_m"][1] < -0.10 and
            ground["tool_translation_from_zero_m"][2] < -0.10 and
            roll["tool_rotation_from_zero_deg"] >= 0.80 * math.degrees(
                self.rotation_positive[0]) and
            target_tracking_ok and return_ok and not dangerous)
        return {
            "passed": passed,
            "mapping_profile": "camera_ground_workspace",
            "calibration_file": str(Path(
                self.args.calibration).expanduser().resolve()),
            "robot_zero_position_m": zero_position.tolist(),
            "camera_positive_translation_extent_m": (
                self.translation_positive.tolist()),
            "camera_positive_rotation_extent_deg": np.degrees(
                self.rotation_positive).tolist(),
            "records": records,
            "return_position_tolerance_m": (
                self.args.return_position_tolerance_m),
            "return_rotation_tolerance_deg": (
                self.args.return_rotation_tolerance_deg),
            "peak_position_error_tolerance_m": (
                self.args.peak_position_error_tolerance_m),
            "peak_rotation_error_tolerance_deg": (
                self.args.peak_rotation_error_tolerance_deg),
            "servo_statuses_observed": sorted(set(self.servo_statuses)),
            "dangerous_servo_statuses": dangerous,
        }


def main():
    args = parse_args()
    if (args.rate_hz <= 0.0 or args.ramp_s <= 0.0 or args.hold_s <= 0.0 or
            args.return_hold_s <= 0.0):
        raise SystemExit("rate and durations must be positive")
    rospy.init_node("ground_workspace_gazebo_validator")
    validator = GroundWorkspaceGazeboValidator(args)
    result = validator.run()
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "output": str(destination),
        "records": [{
            "name": record["name"],
            "tool_translation_from_zero_m": record[
                "tool_translation_from_zero_m"],
            "tool_rotation_from_zero_deg": record[
                "tool_rotation_from_zero_deg"],
            "return_position_error_m": record[
                "return_position_error_m"],
            "return_rotation_error_deg": record[
                "return_rotation_error_deg"],
            "peak_target_position_error_m": record[
                "peak_target_position_error_m"],
            "peak_target_rotation_error_deg": record[
                "peak_target_rotation_error_deg"],
        } for record in result["records"]],
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
