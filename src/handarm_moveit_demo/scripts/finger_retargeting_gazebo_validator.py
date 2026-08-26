#!/usr/bin/env python3
"""End-to-end UDP/ROS/Gazebo acceptance for continuous finger retargeting."""

import argparse
import json
import socket
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from handarm_moveit_demo.hamer_input_contract import identity_reference_token


JOINTS = ["f1j1", "f1j2", "f2j1", "f3j2"]
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
OPEN = np.asarray([0.18, 0.20, 0.20, 0.20])


class FingerRetargetingGazeboValidator:
    def __init__(self, host, port):
        self.host = str(host)
        self.port = int(port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session = "finger-validation-{}".format(int(time.time() * 1000.0))
        self.sequence = 0
        self.positions = None
        self.position_stamp = 0.0
        self.latest_diagnostic = {}
        self.diagnostic_statuses = []
        self.collect_arm = False
        self.arm_samples = []
        self.last_stage_sample_times = []
        rospy.Subscriber("/joint_states", JointState, self.state_callback, queue_size=50)
        rospy.Subscriber(
            "/shared_teleop/finger_diagnostics",
            String,
            self.diagnostic_callback,
            queue_size=50,
        )

    def state_callback(self, message):
        values = dict(zip(message.name, message.position))
        if all(name in values for name in ARM_JOINTS):
            arm = np.asarray([values[name] for name in ARM_JOINTS], dtype=float)
            if np.all(np.isfinite(arm)) and self.collect_arm:
                self.arm_samples.append(arm)
        if not all(name in values for name in JOINTS):
            return
        position = np.asarray([values[name] for name in JOINTS], dtype=float)
        if np.all(np.isfinite(position)):
            self.positions = position
            self.position_stamp = time.monotonic()

    def diagnostic_callback(self, message):
        try:
            value = json.loads(message.data)
        except (TypeError, ValueError):
            return
        self.latest_diagnostic = value
        self.diagnostic_statuses.append(str(value.get("status", "")))

    def current(self):
        if self.positions is None or time.monotonic() - self.position_stamp > 0.30:
            raise RuntimeError("fresh four-joint hand state is unavailable")
        return self.positions.copy()

    def wait_ready(self, timeout_s=25.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            try:
                return self.current()
            except RuntimeError:
                time.sleep(0.05)
        raise RuntimeError("Gazebo hand state did not become ready")

    def packet(
        self,
        flexion,
        enabled,
        epoch,
        pose_valid=True,
        finger_valid=True,
        wrist_position=(0.0, 0.0, 0.55),
    ):
        self.sequence += 1
        token = (
            identity_reference_token(self.session, epoch, 1, 1, True)
            if enabled
            else ""
        )
        valid = bool(pose_valid)
        message = {
            "schema": "handarm_hamer_pose_v1",
            "session_id": self.session,
            "sequence": self.sequence,
            "stamp": time.time(),
            "frame_id": "camera_color_optical_frame",
            "valid": valid,
            "invalid_reason": "" if valid else "SYNTHETIC_OCCLUSION",
            "confidence": [1.0] * 6 if valid else [0.0] * 6,
            "gesture": 0,
            "gesture_confidence": 0.0,
            "finger_observation": {
                "contract_version": 1,
                "feature_definition": (
                    "mano_openpose_chain_total_bend_over_pi_v1"
                ),
                "valid": bool(valid and finger_valid),
                "flexion": list(flexion) if valid and finger_valid else [0.0] * 5,
                "confidence": 1.0 if valid and finger_valid else 0.0,
                "invalid_reason": "" if valid and finger_valid else "SYNTHETIC_OCCLUSION",
            },
            "hand_identity_present": True,
            "hand_is_right": True,
            "presence_generation": 1,
            "active_hand_generation": 1,
            "control_enabled": bool(enabled),
            "control_reference_epoch": int(epoch),
            "control_reference_token": token,
            "control_identity_present": bool(enabled),
            "control_presence_generation": 1 if enabled else 0,
            "control_active_hand_generation": 1 if enabled else 0,
            "control_hand_is_right": bool(enabled),
            "timing": {
                "contract_version": 1,
                "capture_sequence": self.sequence,
                "dropped_capture_frames": 0,
                "capture_to_loop_start_s": 0.001,
                "capture_to_publish_s": 0.002,
                "inference_executed": bool(valid),
                "inference_call_s": 0.005 if valid else 0.0,
                "model_inference_s": 0.004 if valid else 0.0,
                "postprocess_s": 0.001 if valid else 0.0,
            },
        }
        if valid:
            message.update({
                "wrist_position_m": list(wrist_position),
                "palm_rotation_row_major": np.eye(3).reshape(-1).tolist(),
            })
        return message

    def send(
        self,
        flexion,
        enabled=True,
        epoch=1,
        pose_valid=True,
        finger_valid=True,
        wrist_position=(0.0, 0.0, 0.55),
    ):
        message = self.packet(
            flexion,
            enabled,
            epoch,
            pose_valid=pose_valid,
            finger_valid=finger_valid,
            wrist_position=wrist_position,
        )
        self.socket.sendto(
            json.dumps(message, separators=(",", ":")).encode("utf-8"),
            (self.host, self.port),
        )

    def stage(
        self,
        duration_s,
        flexion_fn,
        enabled=True,
        epoch=1,
        pose_valid=True,
        finger_valid=True,
        wrist_position_fn=None,
    ):
        samples = []
        sample_times = []
        started = time.monotonic()
        while time.monotonic() - started < duration_s and not rospy.is_shutdown():
            elapsed = time.monotonic() - started
            self.send(
                flexion_fn(elapsed),
                enabled=enabled,
                epoch=epoch,
                pose_valid=pose_valid,
                finger_valid=finger_valid,
                wrist_position=(
                    [0.0, 0.0, 0.55]
                    if wrist_position_fn is None
                    else wrist_position_fn(elapsed)
                ),
            )
            try:
                samples.append(self.current())
                sample_times.append(time.monotonic())
            except RuntimeError:
                pass
            time.sleep(0.05)
        self.last_stage_sample_times = sample_times
        return samples

    @staticmethod
    def tail_mean(samples, count=8):
        if len(samples) < count:
            raise RuntimeError("too few Gazebo hand samples")
        return np.mean(np.asarray(samples[-count:]), axis=0)

    def move_digit(self, indices, epoch=1):
        baseline = np.full(5, 0.10)

        def trajectory(elapsed):
            amount = 0.65 * min(1.0, elapsed / 1.0)
            value = baseline.copy()
            value[list(indices)] += amount
            return value.tolist()

        samples = self.stage(2.0, trajectory, epoch=epoch)
        closed = self.tail_mean(samples)
        def reopen(elapsed):
            amount = 0.65 * max(0.0, 1.0 - elapsed / 1.0)
            value = baseline.copy()
            value[list(indices)] += amount
            return value.tolist()

        reopened_samples = self.stage(2.0, reopen, epoch=epoch)
        reopened = self.tail_mean(reopened_samples)
        return closed, reopened

    def run(self):
        startup = self.wait_ready()
        baseline = [0.10] * 5
        # This producer-origin disabled packet is the camera-up edge required
        # by the existing C-token interlock.
        self.stage(0.6, lambda _elapsed: baseline, enabled=False, epoch=0)
        open_samples = self.stage(2.0, lambda _elapsed: baseline, epoch=1)
        opened = self.tail_mean(open_samples)
        calibrated = bool(self.latest_diagnostic.get("calibrated", False))

        # Move the arm through the normal wrist-pose chain while the human
        # finger vector remains exactly at its C baseline.  This reproduces
        # the coupling that made the old hand oscillate in the user's video.
        self.arm_samples = []
        self.collect_arm = True
        transport_samples = self.stage(
            6.0,
            lambda _elapsed: baseline,
            epoch=1,
            wrist_position_fn=lambda elapsed: [
                0.05 * np.sin(2.0 * np.pi * elapsed / 3.0),
                0.0,
                0.55,
            ],
        )
        self.collect_arm = False
        transport_hand = np.asarray(transport_samples)
        transport_times = np.asarray(self.last_stage_sample_times)
        transport_arm = np.asarray(self.arm_samples)
        transport_hand_range = np.ptp(transport_hand, axis=0)
        transport_hand_velocity = np.abs(
            np.diff(transport_hand, axis=0)
            / np.diff(transport_times).reshape(-1, 1)
        )
        transport_hand_velocity_p95 = np.percentile(
            transport_hand_velocity, 95.0, axis=0
        )
        transport_arm_range = (
            np.ptp(transport_arm, axis=0)
            if transport_arm.ndim == 2 and len(transport_arm) >= 2
            else np.zeros(6)
        )
        # Return the wrist to its C pose before digit-isolation tests.
        self.stage(2.0, lambda _elapsed: baseline, epoch=1)

        index_closed, index_reopened = self.move_digit([1], epoch=1)
        thumb_closed, thumb_reopened = self.move_digit([0], epoch=1)
        remaining_closed, remaining_reopened = self.move_digit([2, 3, 4], epoch=1)

        # Continuous invalid heartbeats must hold immediately without
        # cancelling C. The same token must resume deterministic retargeting
        # when valid finger evidence returns.
        status_start = len(self.diagnostic_statuses)
        before_invalid = self.current()
        invalid_samples = self.stage(
            0.9,
            lambda _elapsed: baseline,
            epoch=1,
            pose_valid=False,
            finger_valid=False,
        )
        after_invalid = self.tail_mean(invalid_samples)
        timeout_seen = "FINGER_INPUT_TIMEOUT_HOLDING_C_REFERENCE" in (
            self.diagnostic_statuses[status_start:]
        ) or (
            "FINGER_INPUT_TIMEOUT_HOLDING_C_REFERENCE"
            in self.diagnostic_statuses
        )
        old_token_samples = self.stage(
            1.2,
            lambda elapsed: [
                0.10,
                0.10 + 0.65 * min(1.0, elapsed / 0.6),
                0.10,
                0.10,
                0.10,
            ],
            epoch=1,
        )
        old_token_after = self.tail_mean(old_token_samples)
        old_token_blocked = (
            "BLOCKED_REFERENCE_REQUIRES_NEW_C" in self.diagnostic_statuses
        )
        old_token_delta = old_token_after - after_invalid

        # A new C token recalibrates and restores the same deterministic map.
        new_c_open = self.stage(1.5, lambda _elapsed: baseline, epoch=2)
        new_c_opened = self.tail_mean(new_c_open)
        renewed_closed, renewed_reopened = self.move_digit([1], epoch=2)

        def selected_delta(closed, opened_state, selected):
            delta = closed - opened_state
            return float(delta[selected]), [float(value) for value in delta]

        index_selected, index_delta = selected_delta(index_closed, opened, 1)
        thumb_selected, thumb_delta = selected_delta(thumb_closed, index_reopened, 2)
        remaining_selected, remaining_delta = selected_delta(
            remaining_closed, thumb_reopened, 3
        )
        renewed_selected, renewed_delta = selected_delta(
            renewed_closed, new_c_opened, 1
        )
        reopen_errors = {
            "index": float(np.max(np.abs(index_reopened - OPEN))),
            "thumb": float(np.max(np.abs(thumb_reopened - OPEN))),
            "remaining": float(np.max(np.abs(remaining_reopened - OPEN))),
            "renewed": float(np.max(np.abs(renewed_reopened - OPEN))),
        }
        cross_limit = 0.08
        passed = bool(
            calibrated
            and np.max(np.abs(opened - OPEN)) <= 0.06
            and np.max(transport_hand_range) <= 0.01
            and np.max(transport_hand_velocity_p95) <= 0.20
            and np.max(transport_arm_range) >= 0.03
            and index_selected >= 0.30
            and abs(index_delta[2]) <= cross_limit
            and abs(index_delta[3]) <= cross_limit
            and thumb_selected >= 0.30
            and abs(thumb_delta[1]) <= cross_limit
            and abs(thumb_delta[3]) <= cross_limit
            and remaining_selected >= 0.30
            and abs(remaining_delta[1]) <= cross_limit
            and abs(remaining_delta[2]) <= cross_limit
            and timeout_seen
            and not old_token_blocked
            and np.max(np.abs(after_invalid - before_invalid)) <= 0.06
            and old_token_delta[1] >= 0.30
            and abs(old_token_delta[2]) <= cross_limit
            and abs(old_token_delta[3]) <= cross_limit
            and renewed_selected >= 0.30
            and max(reopen_errors.values()) <= 0.06
        )
        return {
            "passed": passed,
            "joint_names": JOINTS,
            "startup_joint_position_rad": startup.tolist(),
            "open_joint_position_rad": opened.tolist(),
            "open_reference_calibrated": calibrated,
            "fixed_fingers_during_arm_motion": {
                "hand_position_peak_to_peak_rad": transport_hand_range.tolist(),
                "hand_velocity_p95_rad_s": transport_hand_velocity_p95.tolist(),
                "arm_joint_peak_to_peak_rad": transport_arm_range.tolist(),
            },
            "index_to_f1_delta_rad": index_delta,
            "thumb_to_f2_delta_rad": thumb_delta,
            "middle_ring_pinky_to_f3_delta_rad": remaining_delta,
            "invalid_heartbeat_timeout_seen": timeout_seen,
            "invalid_hold_delta_rad": (after_invalid - before_invalid).tolist(),
            "same_c_token_blocked": old_token_blocked,
            "same_c_token_resume_delta_rad": old_token_delta.tolist(),
            "new_c_index_to_f1_delta_rad": renewed_delta,
            "reopen_max_error_rad": reopen_errors,
            "diagnostic_statuses": sorted(set(self.diagnostic_statuses)),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--output", default="")
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("finger_retargeting_gazebo_validator", anonymous=True)
    result = FingerRetargetingGazeboValidator(args.host, args.port).run()
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
