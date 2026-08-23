#!/usr/bin/env python3
"""Drive a deterministic 90-degree local-X hand target and measure tool0."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
from std_msgs.msg import Float64, Int8, String
from std_srvs.srv import Trigger

from handarm_moveit_demo.msg import HamerHandPose
from handarm_moveit_demo.shared_teleop_core import (
    matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix, so3_exp, so3_log,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/handarm_gazebo_x90_response.json")
    parser.add_argument("--target-deg", type=float, default=90.0)
    parser.add_argument("--cross-y-deg", type=float, default=0.0)
    parser.add_argument("--cross-z-deg", type=float, default=0.0)
    parser.add_argument("--hand-translation-x-m", type=float, default=0.0)
    parser.add_argument("--abrupt-return", action="store_true")
    parser.add_argument("--ramp-s", type=float, default=0.50)
    parser.add_argument("--hold-s", type=float, default=2.00)
    parser.add_argument("--settle-s", type=float, default=2.00)
    return parser.parse_args(rospy.myargv()[1:])


class X90Validator:
    def __init__(self, args):
        self.args = args
        self.publisher = rospy.Publisher(
            "/shared_teleop/hamer_pose", HamerHandPose, queue_size=1)
        self.buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.sequence = 0
        self.statuses = []
        self.collision_scales = []
        self.continuity_reasons = []
        rospy.Subscriber(
            "/servo_server/status", Int8,
            lambda message: self.statuses.append(int(message.data)), queue_size=1)
        rospy.Subscriber(
            "/servo_server/internal/collision_velocity_scale", Float64,
            lambda message: self.collision_scales.append(float(message.data)),
            queue_size=1)
        rospy.Subscriber(
            "/shared_teleop/trend_diagnostics", String,
            self.diagnostic_callback, queue_size=10)

    def diagnostic_callback(self, message):
        try:
            reason = str(json.loads(message.data).get(
                "pose_continuity_reason", ""))
            if reason:
                self.continuity_reasons.append(reason)
        except (TypeError, ValueError):
            pass

    def publish(self, angle_rad):
        now = rospy.Time.now()
        message = HamerHandPose()
        message.header.seq = self.sequence
        message.header.stamp = now
        message.header.frame_id = "camera_color_optical_frame"
        message.source_timestamp = now.to_sec()
        target_rad = math.radians(self.args.target_deg)
        fraction = 0.0 if abs(target_rad) <= 1.0e-12 else angle_rad / target_rad
        message.wrist_pose.position.x = (
            fraction * self.args.hand_translation_x_m)
        message.wrist_pose.position.y = 0.0
        message.wrist_pose.position.z = 0.55
        quaternion = matrix_to_quaternion_xyzw(so3_exp([
            angle_rad,
            fraction * math.radians(self.args.cross_y_deg),
            fraction * math.radians(self.args.cross_z_deg),
        ]))
        (message.wrist_pose.orientation.x,
         message.wrist_pose.orientation.y,
         message.wrist_pose.orientation.z,
         message.wrist_pose.orientation.w) = quaternion
        message.confidence = [1.0] * 6
        message.valid = True
        message.gesture = 0
        message.gesture_confidence = 0.0
        message.invalid_reason = ""
        message.control_gate_present = False
        message.control_enabled = False
        message.control_reference_epoch = 0
        message.control_reference_token = ""
        self.publisher.publish(message)
        self.sequence += 1

    def pose(self):
        transform = self.buffer.lookup_transform(
            "base_link", "tool0", rospy.Time(0), rospy.Duration(0.10))
        p = transform.transform.translation
        q = transform.transform.rotation
        return (
            np.asarray([p.x, p.y, p.z]),
            quaternion_xyzw_to_matrix([q.x, q.y, q.z, q.w]),
        )

    def publish_for(self, duration_s, angle_function, sample=None):
        began = rospy.Time.now().to_sec()
        rate = rospy.Rate(30.0)
        while not rospy.is_shutdown():
            elapsed = rospy.Time.now().to_sec() - began
            if elapsed >= duration_s:
                break
            angle = float(angle_function(elapsed))
            self.publish(angle)
            if sample is not None:
                try:
                    sample(elapsed, angle, *self.pose())
                except Exception:
                    pass
            rate.sleep()

    def run(self):
        rospy.wait_for_service("/shared_teleop/confirm_hand_reference", timeout=10.0)
        self.publish_for(1.5, lambda _elapsed: 0.0)
        response = rospy.ServiceProxy(
            "/shared_teleop/confirm_hand_reference", Trigger)()
        if not response.success:
            raise RuntimeError("reference confirmation failed: " + response.message)
        self.publish_for(1.0, lambda _elapsed: 0.0)
        initial_position, initial_rotation = self.pose()

        target_rad = math.radians(self.args.target_deg)
        direction = 1.0 if target_rad >= 0.0 else -1.0
        target_magnitude_deg = abs(self.args.target_deg)
        records = []
        first_5_deg_s = None
        first_80_deg_s = None

        def sample(elapsed, hand_angle, position, rotation):
            nonlocal first_5_deg_s, first_80_deg_s
            local_rotation = initial_rotation.T @ rotation
            vector = so3_log(local_rotation)
            records.append({
                "elapsed_s": elapsed,
                "hand_angle_rad": hand_angle,
                "tool_local_rotation_vector_rad": vector,
                "tool_position": position,
            })
            directional_x_degrees = direction * math.degrees(float(vector[0]))
            if first_5_deg_s is None and directional_x_degrees >= 5.0:
                first_5_deg_s = elapsed
            if first_80_deg_s is None and directional_x_degrees >= 80.0:
                first_80_deg_s = elapsed

        self.publish_for(
            self.args.ramp_s,
            lambda elapsed: target_rad * min(1.0, elapsed / self.args.ramp_s),
            sample)
        ramp_finished_at = self.args.ramp_s

        def hold_sample(elapsed, hand_angle, position, rotation):
            sample(ramp_finished_at + elapsed, hand_angle, position, rotation)

        self.publish_for(
            self.args.hold_s, lambda _elapsed: target_rad, hold_sample)
        target_records = list(records)
        peak_record = max(
            records,
            key=lambda item: direction *
            item["tool_local_rotation_vector_rad"][0])
        peak_vector = np.asarray(peak_record["tool_local_rotation_vector_rad"])
        peak_x_deg = math.degrees(float(peak_vector[0]))
        peak_directional_x_deg = direction * peak_x_deg
        peak_cross_deg = math.degrees(float(np.linalg.norm(peak_vector[1:])))
        steady_start = self.args.ramp_s + max(0.0, self.args.hold_s - 0.40)
        steady_vectors = [
            np.asarray(record["tool_local_rotation_vector_rad"])
            for record in target_records
            if record["elapsed_s"] >= steady_start]
        if not steady_vectors:
            steady_vectors = [peak_vector]
        steady_vector = np.mean(steady_vectors, axis=0)
        steady_x_deg = math.degrees(float(steady_vector[0]))
        steady_directional_x_deg = direction * steady_x_deg
        steady_cross_deg = math.degrees(float(np.linalg.norm(steady_vector[1:])))
        peak_overshoot_deg = max(
            0.0, peak_directional_x_deg - target_magnitude_deg)

        def return_sample(elapsed, hand_angle, position, rotation):
            sample(
                ramp_finished_at + self.args.hold_s + elapsed,
                hand_angle, position, rotation)

        if self.args.abrupt_return:
            self.publish_for(
                self.args.ramp_s, lambda _elapsed: 0.0, return_sample)
        else:
            self.publish_for(
                self.args.ramp_s,
                lambda elapsed: target_rad * max(
                    0.0, 1.0 - elapsed / self.args.ramp_s),
                return_sample)
        self.publish_for(self.args.settle_s, lambda _elapsed: 0.0)
        final_position, final_rotation = self.pose()
        return_vector = so3_log(initial_rotation.T @ final_rotation)
        return_rotation_deg = math.degrees(float(np.linalg.norm(return_vector)))
        return_translation_m = float(np.linalg.norm(final_position - initial_position))
        dangerous = sorted(set(code for code in self.statuses if code in (2, 4, 5)))
        passed = bool(
            target_magnitude_deg - 5.0 <= steady_directional_x_deg <=
            target_magnitude_deg + 5.0 and
            peak_overshoot_deg <= 15.0 and
            peak_cross_deg <= 10.0 and
            first_5_deg_s is not None and first_5_deg_s <= 0.40 and
            first_80_deg_s is not None and first_80_deg_s <= 1.50 and
            return_rotation_deg <= 2.0 and
            return_translation_m <= 0.010 and
            (not self.args.abrupt_return or
             "C_ZERO_RETREAT_OVERRIDE" in self.continuity_reasons) and
            not dangerous)
        return {
            "passed": passed,
            "target_hand_x_deg": self.args.target_deg,
            "input_cross_y_deg": self.args.cross_y_deg,
            "input_cross_z_deg": self.args.cross_z_deg,
            "input_hand_translation_x_m": self.args.hand_translation_x_m,
            "abrupt_return": self.args.abrupt_return,
            "peak_tool_local_x_deg": peak_x_deg,
            "peak_tool_directional_x_deg": peak_directional_x_deg,
            "peak_tool_cross_axis_deg": peak_cross_deg,
            "peak_overshoot_deg": peak_overshoot_deg,
            "steady_tool_local_x_deg": steady_x_deg,
            "steady_tool_directional_x_deg": steady_directional_x_deg,
            "steady_tool_cross_axis_deg": steady_cross_deg,
            "steady_tool_to_hand_angle_ratio": (
                steady_x_deg / self.args.target_deg
                if abs(self.args.target_deg) > 1.0e-12 else None),
            "response_time_to_5_deg_s": first_5_deg_s,
            "response_time_to_80_deg_s": first_80_deg_s,
            "return_rotation_error_deg": return_rotation_deg,
            "return_translation_error_m": return_translation_m,
            "servo_statuses_observed": sorted(set(self.statuses)),
            "servo_danger_statuses": dangerous,
            "collision_scale_min": (
                None if not self.collision_scales else min(self.collision_scales)),
            "pose_continuity_reasons_observed": sorted(set(
                self.continuity_reasons)),
            "c_zero_retreat_override_seen": (
                "C_ZERO_RETREAT_OVERRIDE" in self.continuity_reasons),
        }


def main():
    args = parse_args()
    rospy.init_node("gazebo_x90_response_validator")
    try:
        result = X90Validator(args).run()
    except Exception as exc:
        result = {"passed": False, "error": type(exc).__name__ + ":" + str(exc)}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("result_file={}".format(output))
    if not result.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
