#!/usr/bin/env python3
"""Validate live human-pose control of Gazebo ``base_link -> tool0``.

The validator deliberately consumes the normal UDP teleoperation topics.  It
does not publish a synthetic pose or a Servo command.  During the measurement
window the operator must translate the hand and rotate the wrist.  The result
is a machine-readable JSON record proving that valid human poses reached ROS,
non-zero safe commands were produced, and Gazebo tool motion followed those
commands with the same sign after allowing for controller latency.
"""

import argparse
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int8, String
from std_srvs.srv import Trigger

from handarm_moveit_demo.msg import HamerHandPose, HandCommand
from handarm_moveit_demo.shared_teleop_core import (
    quaternion_xyzw_to_matrix,
    so3_log,
)


DANGEROUS_SERVO_STATUSES = (2, 4, 5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/handarm_live_gazebo_follow.json")
    parser.add_argument("--wait-for-input-s", type=float, default=45.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--countdown-s", type=float, default=3.0)
    parser.add_argument("--minimum-input-hz", type=float, default=3.0)
    parser.add_argument("--minimum-hand-translation-m", type=float, default=0.015)
    parser.add_argument("--minimum-hand-rotation-rad", type=float, default=0.08)
    parser.add_argument("--minimum-tool-translation-m", type=float, default=0.005)
    parser.add_argument("--minimum-tool-rotation-rad", type=float, default=0.03)
    parser.add_argument("--minimum-tracking-cosine", type=float, default=0.55)
    parser.add_argument("--allow-translation-only", action="store_true")
    parser.add_argument("--allow-rotation-only", action="store_true")
    parser.add_argument(
        "--confirm-reference", action="store_true",
        help=(
            "explicitly arm a new ROS reference before measuring; disabled "
            "by default because a live camera C reference must never be "
            "silently replaced by an acceptance tool"
        ),
    )
    return parser.parse_args(rospy.myargv()[1:])


def _finite_vector(values, size):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError("expected a finite {}-vector".format(size))
    return vector


def _best_lag_cosine(commands, actual, maximum_lag=8):
    """Return the best flattened vector cosine over a small causal lag range."""

    command_array = np.asarray(commands, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    if command_array.ndim != 2 or command_array.shape[1] != 3:
        return None, None
    if actual_array.shape != command_array.shape or len(command_array) < 5:
        return None, None
    best_score = None
    best_lag = None
    for lag in range(maximum_lag + 1):
        if lag == 0:
            left, right = command_array, actual_array
        else:
            left, right = command_array[:-lag], actual_array[lag:]
        active = np.linalg.norm(left, axis=1) > 1.0e-5
        if int(np.count_nonzero(active)) < 4:
            continue
        left = left[active]
        right = right[active]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1.0e-12:
            continue
        score = float(np.sum(left * right) / denominator)
        if best_score is None or score > best_score:
            best_score = score
            best_lag = lag
    return best_score, best_lag


class LiveGazeboFollowValidator:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.hamer = None
        self.raw_command = None
        self.safe_twist = None
        self.trend = {}
        self.output = {}
        self.servo_statuses = []
        self.hamer_messages = 0
        self.valid_hamer_messages = 0
        self.unique_sequences = set()
        self.hamer_receive_times = []
        self.buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buffer)

        rospy.Subscriber(
            "/shared_teleop/hamer_pose", HamerHandPose,
            self.hamer_callback, queue_size=1)
        rospy.Subscriber(
            "/shared_teleop/raw_hand_command", HandCommand,
            self.raw_callback, queue_size=1)
        rospy.Subscriber(
            "/shared_teleop/safe_twist", TwistStamped,
            self.safe_callback, queue_size=1)
        rospy.Subscriber(
            "/shared_teleop/trend_diagnostics", String,
            lambda message: self.json_callback("trend", message), queue_size=1)
        rospy.Subscriber(
            "/shared_teleop/output_diagnostics", String,
            lambda message: self.json_callback("output", message), queue_size=1)
        rospy.Subscriber(
            "/servo_server/status", Int8,
            self.servo_status_callback, queue_size=10)

    def hamer_callback(self, message):
        now = time.monotonic()
        with self.lock:
            self.hamer = message
            self.hamer_messages += 1
            self.unique_sequences.add(int(message.header.seq))
            self.hamer_receive_times.append(now)
            if message.valid:
                self.valid_hamer_messages += 1

    def raw_callback(self, message):
        with self.lock:
            self.raw_command = message

    def safe_callback(self, message):
        with self.lock:
            self.safe_twist = message

    def json_callback(self, name, message):
        try:
            payload = json.loads(message.data)
        except Exception:
            return
        with self.lock:
            setattr(self, name, payload)

    def servo_status_callback(self, message):
        with self.lock:
            self.servo_statuses.append(int(message.data))

    def snapshot(self):
        with self.lock:
            return {
                "hamer": self.hamer,
                "raw": self.raw_command,
                "safe": self.safe_twist,
                "trend": dict(self.trend),
                "output": dict(self.output),
            }

    def tool_pose(self):
        transform = self.buffer.lookup_transform(
            "base_link", "tool0", rospy.Time(0), rospy.Duration(0.08))
        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        return (
            np.array([translation.x, translation.y, translation.z], dtype=np.float64),
            quaternion_xyzw_to_matrix(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
        )

    @staticmethod
    def hand_pose(message):
        position = message.wrist_pose.position
        quaternion = message.wrist_pose.orientation
        return (
            np.array([position.x, position.y, position.z], dtype=np.float64),
            quaternion_xyzw_to_matrix(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
        )

    @staticmethod
    def command_vector(message):
        if message is None:
            return np.zeros(6)
        twist = message.twist
        return _finite_vector([
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z,
        ], 6)

    def wait_for_valid_input(self):
        deadline = time.monotonic() + self.args.wait_for_input_s
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            state = self.snapshot()
            try:
                self.tool_pose()
            except Exception:
                rate.sleep()
                continue
            if state["hamer"] is not None and state["hamer"].valid:
                return
            rate.sleep()
        raise RuntimeError(
            "no valid /shared_teleop/hamer_pose and base_link->tool0 TF within "
            "{:.1f} s; check the D455/HaMeR sender and keep one complete hand visible".format(
                self.args.wait_for_input_s))

    def wait_for_reference_ready(self):
        deadline = time.monotonic() + self.args.wait_for_input_s
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            state = self.snapshot()
            if (
                state["hamer"] is not None
                and state["hamer"].valid
                and bool(state["trend"].get("reference_ready", False))
                and bool(state["trend"].get("active_reference_token"))
            ):
                return
            rate.sleep()
        raise RuntimeError(
            "camera C reference is not ready; hold a neutral hand and press "
            "C in the camera window before starting live acceptance"
        )

    def run(self):
        self.wait_for_valid_input()
        if self.args.confirm_reference:
            rospy.wait_for_service(
                "/shared_teleop/confirm_hand_reference",
                timeout=self.args.wait_for_input_s,
            )
            confirm = rospy.ServiceProxy(
                "/shared_teleop/confirm_hand_reference", Trigger)()
            if not confirm.success:
                raise RuntimeError(
                    "reference confirmation failed: " + confirm.message
                )
            reference_message = "A new ROS reference was explicitly armed."
        else:
            self.wait_for_reference_ready()
            reference_message = (
                "Using the existing camera C reference without changing it."
            )
        if self.args.allow_rotation_only:
            motion_instruction = (
                "Rotate the wrist at least 15 degrees and return to C; do "
                "not deliberately translate the hand."
            )
        elif self.args.allow_translation_only:
            motion_instruction = (
                "Translate at least 3 cm and return to C; keep the wrist "
                "orientation neutral."
            )
        else:
            motion_instruction = (
                "Translate at least 3 cm, rotate the wrist at least 15 "
                "degrees, and then return both to C."
            )
        print(
            reference_message + " Keep the hand neutral for the countdown. "
            + motion_instruction,
            flush=True,
        )
        countdown_deadline = time.monotonic() + max(0.0, self.args.countdown_s)
        while not rospy.is_shutdown() and time.monotonic() < countdown_deadline:
            remaining = int(math.ceil(countdown_deadline - time.monotonic()))
            print("measurement starts in {}...".format(max(0, remaining)), flush=True)
            rospy.sleep(min(1.0, max(0.01, countdown_deadline-time.monotonic())))

        baseline_state = self.snapshot()
        if baseline_state["hamer"] is None or not baseline_state["hamer"].valid:
            self.wait_for_valid_input()
            baseline_state = self.snapshot()
        hand_p0, hand_r0 = self.hand_pose(baseline_state["hamer"])
        tool_p0, tool_r0 = self.tool_pose()
        started = time.monotonic()
        previous_time = None
        previous_tool_position = None
        previous_tool_rotation = None
        samples = []
        maximum_hand_translation = 0.0
        maximum_hand_rotation = 0.0
        maximum_tool_translation = 0.0
        maximum_tool_rotation = 0.0
        maximum_linear_command = 0.0
        maximum_angular_command = 0.0
        raw_valid_samples = 0
        output_allowed_samples = 0
        unsafe_reasons = set()
        rate = rospy.Rate(50.0)

        while not rospy.is_shutdown() and time.monotonic() - started < self.args.duration_s:
            now = time.monotonic()
            state = self.snapshot()
            try:
                tool_position, tool_rotation = self.tool_pose()
            except Exception:
                rate.sleep()
                continue
            safe_command = self.command_vector(state["safe"])
            raw = state["raw"]
            if raw is not None and raw.valid:
                raw_valid_samples += 1
            if state["output"].get("output_allowed"):
                output_allowed_samples += 1
            for reason in state["output"].get("reasons", []):
                if reason in (
                    "WORKSPACE_TF_UNAVAILABLE", "EMERGENCY_STOP_LATCHED",
                    "INPUT_CLOCK_MISMATCH",
                ):
                    unsafe_reasons.add(str(reason))

            maximum_linear_command = max(
                maximum_linear_command, float(np.linalg.norm(safe_command[:3])))
            maximum_angular_command = max(
                maximum_angular_command, float(np.linalg.norm(safe_command[3:])))
            maximum_tool_translation = max(
                maximum_tool_translation, float(np.linalg.norm(tool_position-tool_p0)))
            maximum_tool_rotation = max(
                maximum_tool_rotation,
                float(np.linalg.norm(so3_log(tool_rotation @ tool_r0.T))),
            )
            if state["hamer"] is not None and state["hamer"].valid:
                hand_position, hand_rotation = self.hand_pose(state["hamer"])
                maximum_hand_translation = max(
                    maximum_hand_translation,
                    float(np.linalg.norm(hand_position-hand_p0)),
                )
                maximum_hand_rotation = max(
                    maximum_hand_rotation,
                    float(np.linalg.norm(so3_log(hand_rotation @ hand_r0.T))),
                )

            if previous_time is not None and now > previous_time:
                dt = now - previous_time
                actual_linear = (tool_position - previous_tool_position) / dt
                actual_angular = so3_log(
                    tool_rotation @ previous_tool_rotation.T) / dt
                samples.append((safe_command.copy(), actual_linear, actual_angular))
            previous_time = now
            previous_tool_position = tool_position.copy()
            previous_tool_rotation = tool_rotation.copy()
            rate.sleep()

        elapsed = max(1.0e-9, time.monotonic()-started)
        commands = [entry[0] for entry in samples]
        linear_score, linear_lag = _best_lag_cosine(
            [entry[:3] for entry in commands],
            [entry[1] for entry in samples],
        )
        angular_score, angular_lag = _best_lag_cosine(
            [entry[3:] for entry in commands],
            [entry[2] for entry in samples],
        )

        with self.lock:
            receive_times = list(self.hamer_receive_times)
            statuses = list(self.servo_statuses)
            message_count = self.hamer_messages
            valid_message_count = self.valid_hamer_messages
            unique_sequences = len(self.unique_sequences)
        window_receive_times = [stamp for stamp in receive_times if stamp >= started]
        input_hz = len(window_receive_times) / elapsed
        dangerous = sorted(set(code for code in statuses if code in DANGEROUS_SERVO_STATUSES))

        translation_required = not self.args.allow_rotation_only
        rotation_required = not self.args.allow_translation_only
        translation_checks = {
            "human_motion": maximum_hand_translation >= self.args.minimum_hand_translation_m,
            "safe_command": maximum_linear_command >= 0.003,
            "tool_motion": maximum_tool_translation >= self.args.minimum_tool_translation_m,
            "command_tool_direction": (
                linear_score is not None
                and linear_score >= self.args.minimum_tracking_cosine
            ),
        }
        rotation_checks = {
            "human_motion": maximum_hand_rotation >= self.args.minimum_hand_rotation_rad,
            "safe_command": maximum_angular_command >= 0.02,
            "tool_motion": maximum_tool_rotation >= self.args.minimum_tool_rotation_rad,
            "command_tool_direction": (
                angular_score is not None
                and angular_score >= self.args.minimum_tracking_cosine
            ),
        }
        common_checks = {
            "input_rate": input_hz >= self.args.minimum_input_hz,
            "valid_hamer_received": valid_message_count > 0 and unique_sequences > 1,
            "raw_command_valid": raw_valid_samples > 0,
            "simulation_output_enabled": output_allowed_samples > 0,
            "no_dangerous_servo_status": not dangerous,
            "no_fail_closed_runtime_fault": not unsafe_reasons,
        }
        passed = all(common_checks.values())
        if translation_required:
            passed = passed and all(translation_checks.values())
        if rotation_required:
            passed = passed and all(rotation_checks.values())

        return {
            "passed": bool(passed),
            "mode": "LIVE_HUMAN_POSE_TO_GAZEBO_SERVO",
            "source_frame": "camera_color_optical_frame",
            "command_frame": "base_link",
            "control_point": "tool0",
            "duration_s": elapsed,
            "input": {
                "messages_total_process": message_count,
                "valid_messages_total_process": valid_message_count,
                "unique_sequences": unique_sequences,
                "measurement_rate_hz": input_hz,
            },
            "translation": {
                "required": translation_required,
                "checks": translation_checks,
                "maximum_human_excursion_m": maximum_hand_translation,
                "maximum_safe_command_mps": maximum_linear_command,
                "maximum_tool_excursion_m": maximum_tool_translation,
                "best_command_tool_velocity_cosine": linear_score,
                "best_lag_samples": linear_lag,
            },
            "rotation": {
                "required": rotation_required,
                "checks": rotation_checks,
                "maximum_human_excursion_rad": maximum_hand_rotation,
                "maximum_safe_command_radps": maximum_angular_command,
                "maximum_tool_excursion_rad": maximum_tool_rotation,
                "best_command_tool_velocity_cosine": angular_score,
                "best_lag_samples": angular_lag,
            },
            "safety": {
                "checks": common_checks,
                "servo_danger_statuses": dangerous,
                "fail_closed_runtime_faults": sorted(unsafe_reasons),
            },
            "acceptance_instruction": (
                "PASS proves live valid human pose packets drove Gazebo tool0 "
                "translation and rotation through the normal safe Servo chain."
            ),
        }


def main():
    args = parse_args()
    if args.allow_translation_only and args.allow_rotation_only:
        raise SystemExit("translation-only and rotation-only cannot both be selected")
    rospy.init_node("live_gazebo_follow_validator")
    try:
        result = LiveGazeboFollowValidator(args).run()
    except Exception as exc:
        result = {
            "passed": False,
            "mode": "LIVE_HUMAN_POSE_TO_GAZEBO_SERVO",
            "error": "{}:{}".format(type(exc).__name__, exc),
        }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("result_file={}".format(output))
    if not result.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
