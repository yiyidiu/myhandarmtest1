#!/usr/bin/env python3
"""Measure the six deterministic hand mappings from base_link -> tool0 TF."""

import argparse
import json
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
from std_msgs.msg import Float64, Int8, String
from std_srvs.srv import Trigger

from handarm_moveit_demo.shared_teleop_core import (
    quaternion_xyzw_to_matrix, so3_log,
)


TARGETS = {
    "translation_base_x": ("translation", np.array([1.0, 0.0, 0.0])),
    "translation_base_y": ("translation", np.array([0.0, 1.0, 0.0])),
    "translation_base_z": ("translation", np.array([0.0, 0.0, 1.0])),
    "rotation_tool_x": ("rotation", np.array([1.0, 0.0, 0.0])),
    "rotation_tool_y": ("rotation", np.array([0.0, 1.0, 0.0])),
    "rotation_tool_z": ("rotation", np.array([0.0, 0.0, 1.0])),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/handarm_gazebo_direction_validation.json")
    parser.add_argument("--timeout", type=float, default=48.0)
    return parser.parse_args(rospy.myargv()[1:])


class Validator:
    def __init__(self, timeout_s):
        self.timeout_s = timeout_s
        self.phase = None
        self.collision_scale = None
        self.servo_status = None
        self.statuses = []
        self.phase_statuses = {}
        self.records = {}
        self.initial_pose = None
        self.final_pose = None
        rospy.Subscriber("/shared_teleop/direction_test_phase", String,
                         lambda message: setattr(self, "phase", message.data), queue_size=1)
        rospy.Subscriber("/servo_server/internal/collision_velocity_scale", Float64,
                         lambda message: setattr(self, "collision_scale", message.data), queue_size=1)
        rospy.Subscriber("/servo_server/status", Int8,
                         self.status_callback, queue_size=1)
        self.buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buffer)

    def status_callback(self, message):
        self.servo_status = int(message.data)
        self.statuses.append(int(message.data))
        if self.phase in TARGETS:
            self.phase_statuses.setdefault(self.phase, []).append(
                int(message.data))

    def pose(self):
        transform = self.buffer.lookup_transform(
            "base_link", "tool0", rospy.Time(0), rospy.Duration(0.08))
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        position = np.array([translation.x, translation.y, translation.z])
        matrix = quaternion_xyzw_to_matrix(
            [rotation.x, rotation.y, rotation.z, rotation.w])
        return position, matrix

    def run(self):
        for name in ["/shared_teleop/confirm_hand_reference",
                     "/shared_teleop/start_direction_test"]:
            rospy.wait_for_service(name, timeout=8.0)
        rospy.sleep(2.0)  # allow joint TF and collision monitor to become current
        confirm = rospy.ServiceProxy(
            "/shared_teleop/confirm_hand_reference", Trigger)()
        if not confirm.success:
            raise RuntimeError("hand reference confirmation failed: " + confirm.message)
        rospy.sleep(0.40)
        start = rospy.ServiceProxy("/shared_teleop/start_direction_test", Trigger)()
        if not start.success:
            raise RuntimeError("direction sequence failed to start: " + start.message)

        began = rospy.Time.now()
        sequence_started = False
        rate = rospy.Rate(80.0)
        while not rospy.is_shutdown():
            if (rospy.Time.now() - began).to_sec() > self.timeout_s:
                raise RuntimeError("direction sequence timed out")
            try:
                position, rotation = self.pose()
            except Exception:
                rate.sleep()
                continue
            if self.phase in ("armed", "settle"):
                sequence_started = True
            if self.phase == "settle" and self.initial_pose is None:
                self.initial_pose = (position.copy(), rotation.copy())
            if self.phase in TARGETS:
                record = self.records.setdefault(
                    self.phase,
                    [position.copy(), rotation.copy(), position.copy(),
                     rotation.copy(), 0, []])
                record[2] = position.copy()
                record[3] = rotation.copy()
                record[4] += 1
                if self.collision_scale is not None:
                    record[5].append(float(self.collision_scale))
            if self.phase == "complete" and sequence_started:
                self.final_pose = (position.copy(), rotation.copy())
                break
            rate.sleep()
        return self.evaluate()

    def evaluate(self):
        results = {}
        overall = True
        for label, (kind, configured_axis) in TARGETS.items():
            if label not in self.records:
                results[label] = {"passed": False, "reason": "phase_not_observed"}
                overall = False
                continue
            p0, r0, p1, r1, samples, scales = self.records[label]
            translation = p1 - p0
            rotation_vector = so3_log(r1 @ r0.T)
            vector = translation if kind == "translation" else rotation_vector
            expected_axis = configured_axis
            if kind == "rotation" and self.initial_pose is not None:
                # AprilTag V3 composes q_zero * q_delta.  A positive local
                # rotation therefore appears in base axes through R_zero.
                expected_axis = self.initial_pose[1] @ configured_axis
            projection = float(np.dot(vector, expected_axis))
            cross_axis = float(np.linalg.norm(vector - projection * expected_axis))
            hand_excursion = 0.025 if kind == "translation" else 0.22
            if kind == "rotation":
                # The ported relation is 1:1 and the tracker is deliberately
                # speed/acceleration limited, so require visible convergence.
                minimum = 0.10
            else:
                minimum = 0.010
            passed = (projection >= minimum and
                      cross_axis <= max(0.001, 0.10 * projection))
            results[label] = {
                "passed": passed,
                "samples": samples,
                "expected_base_axis": expected_axis.tolist(),
                "translation_m": translation.tolist(),
                "rotation_vector_rad": rotation_vector.tolist(),
                "axis_projection": projection,
                "known_hand_input_excursion": hand_excursion,
                "tool_to_hand_response_ratio": projection / hand_excursion,
                "cross_axis_norm": cross_axis,
                "servo_statuses_observed": sorted(set(
                    self.phase_statuses.get(label, []))),
                "collision_scale_min": None if not scales else min(scales),
                "collision_scale_mean": None if not scales else float(np.mean(scales)),
            }
            overall = overall and passed
        if self.initial_pose is not None and self.final_pose is not None:
            return_translation = self.final_pose[0] - self.initial_pose[0]
            return_rotation = so3_log(self.final_pose[1] @ self.initial_pose[1].T)
            return_translation_norm = float(np.linalg.norm(return_translation))
            return_rotation_norm = float(np.linalg.norm(return_rotation))
            return_to_zero_passed = bool(
                return_translation_norm <= 0.010 and
                return_rotation_norm <= np.deg2rad(2.0))
            overall = overall and return_to_zero_passed
        else:
            return_translation = np.full(3, np.nan)
            return_rotation = np.full(3, np.nan)
            return_translation_norm = float("nan")
            return_rotation_norm = float("nan")
            return_to_zero_passed = False
            overall = False
        dangerous_statuses = sorted(set(code for code in self.statuses if code in (2, 4, 5)))
        overall = overall and not dangerous_statuses
        return {
            "passed": overall,
            "frame": "base_link",
            "control_point": "tool0",
            "servo_statuses_observed": sorted(set(self.statuses)),
            "servo_danger_statuses": dangerous_statuses,
            "return_to_zero_passed": return_to_zero_passed,
            "return_translation_error_m": return_translation.tolist(),
            "return_rotation_error_rad": return_rotation.tolist(),
            "return_translation_error_norm_m": return_translation_norm,
            "return_rotation_error_norm_rad": return_rotation_norm,
            "return_translation_threshold_m": 0.010,
            "return_rotation_threshold_rad": float(np.deg2rad(2.0)),
            "axes": results,
        }


def main():
    args = parse_args()
    rospy.init_node("gazebo_direction_validator")
    try:
        result = Validator(args.timeout).run()
    except Exception as exc:
        result = {"passed": False, "error": str(exc)}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("result_file={}".format(output))
    if not result.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
