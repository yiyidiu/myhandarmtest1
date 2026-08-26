#!/usr/bin/env python3
"""Validate startup and timeout C-token interlocks in the live UDP chain."""

import argparse
import json
import socket
import time

import numpy as np
import rospy
import tf
from std_msgs.msg import Int8, String

from handarm_moveit_demo.hamer_input_contract import identity_reference_token
from handarm_moveit_demo.msg import HamerHandPose


class UdpCGateValidator:
    def __init__(self, host, port):
        self.host = host
        self.port = int(port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listener = tf.TransformListener()
        self.session = "c-gate-validation-{}".format(int(time.time()*1000.0))
        self.sequence = 0
        self.latest_diagnostic = {}
        self.reasons = []
        self.hamer_reasons = []
        self.statuses = []
        self.collecting_gap = False
        self.gap_trend_diagnostics = []
        self.gap_output_diagnostics = []
        rospy.Subscriber(
            "/shared_teleop/trend_diagnostics", String,
            self.diagnostic_callback, queue_size=20)
        rospy.Subscriber(
            "/shared_teleop/output_diagnostics", String,
            self.output_diagnostic_callback, queue_size=20)
        rospy.Subscriber(
            "/servo_server/status", Int8,
            self.status_callback, queue_size=20)
        rospy.Subscriber(
            "/shared_teleop/hamer_pose", HamerHandPose,
            self.hamer_callback, queue_size=20)

    def diagnostic_callback(self, message):
        try:
            self.latest_diagnostic = json.loads(message.data)
            self.reasons.append(str(self.latest_diagnostic.get("reason", "")))
            if self.collecting_gap:
                self.gap_trend_diagnostics.append(
                    dict(self.latest_diagnostic))
        except (TypeError, ValueError):
            pass

    def output_diagnostic_callback(self, message):
        if not self.collecting_gap:
            return
        try:
            self.gap_output_diagnostics.append(json.loads(message.data))
        except (TypeError, ValueError):
            pass

    def status_callback(self, message):
        self.statuses.append(int(message.data))

    def hamer_callback(self, message):
        reason = str(message.invalid_reason)
        if reason:
            self.hamer_reasons.append(reason)

    def tool_position(self):
        translation, _ = self.listener.lookupTransform(
            "base_link", "tool0", rospy.Time(0))
        return np.asarray(translation, dtype=float)

    def wait_ready(self, timeout_s=20.0):
        deadline = time.monotonic()+timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            try:
                position = self.tool_position()
                if np.all(np.isfinite(position)):
                    return position
            except Exception:
                pass
            time.sleep(0.05)
        raise RuntimeError("base_link -> tool0 TF was not ready")

    def send(self, position, enabled, epoch, valid=True):
        self.sequence += 1
        token = (
            identity_reference_token(self.session, epoch, 1, 1, True)
            if enabled else ""
        )
        packet = {
            "schema": "handarm_hamer_pose_v1",
            "session_id": self.session,
            "sequence": self.sequence,
            "stamp": time.time(),
            "frame_id": "camera_color_optical_frame",
            "wrist_position_m": np.asarray(position, dtype=float).tolist(),
            "palm_rotation_row_major": np.eye(3).reshape(-1).tolist(),
            "confidence": [1.0]*6,
            "valid": bool(valid),
            "gesture": 0,
            "gesture_confidence": 0.0,
            "finger_observation": {
                "contract_version": 1,
                "feature_definition": (
                    "mano_openpose_chain_total_bend_over_pi_v1"
                ),
                "valid": bool(valid),
                "flexion": [0.10] * 5 if valid else [0.0] * 5,
                "confidence": 1.0 if valid else 0.0,
                "invalid_reason": "" if valid else "SYNTHETIC_TRANSIENT_MISS",
            },
            "invalid_reason": "" if valid else "SYNTHETIC_TRANSIENT_MISS",
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
            # Synthetic samples still exercise the same strict producer
            # timing contract as the live camera process.
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
        self.socket.sendto(
            json.dumps(packet, separators=(",", ":")).encode("utf-8"),
            (self.host, self.port))

    def stage(self, duration_s, position_fn, enabled, epoch, valid=True):
        samples = []
        started = time.monotonic()
        while (time.monotonic()-started < duration_s and
               not rospy.is_shutdown()):
            elapsed = time.monotonic()-started
            self.send(position_fn(elapsed), enabled, epoch, valid=valid)
            try:
                samples.append(self.tool_position())
            except Exception:
                pass
            time.sleep(0.05)
        return samples

    def silent_stage(self, duration_s):
        """Stop UDP input while observing fail-closed timeout behavior."""
        samples = []
        self.gap_trend_diagnostics = []
        self.gap_output_diagnostics = []
        self.collecting_gap = True
        started = time.monotonic()
        try:
            while (time.monotonic()-started < duration_s and
                   not rospy.is_shutdown()):
                try:
                    samples.append(self.tool_position())
                except Exception:
                    pass
                time.sleep(0.02)
        finally:
            self.collecting_gap = False
        return samples

    def run(self):
        initial = self.wait_ready()
        hand_zero = np.array([0.0, 0.0, 0.55])

        # Simulate a camera process that was already enabled by an old C press
        # before Gazebo/this receiver restarted.  The old token must not move
        # the freshly spawned arm, even though it is otherwise well formed.
        locked_samples = self.stage(
            2.0,
            lambda elapsed: hand_zero + np.array([
                0.08 if int(elapsed*5.0) % 2 == 0 else -0.08, 0.0, 0.0]),
            enabled=True, epoch=1)
        locked_max_motion = max(
            [float(np.linalg.norm(value-initial)) for value in locked_samples]
            or [float("inf")])

        # A changed epoch is the network equivalent of pressing C again after
        # the receiver is ready.
        self.stage(1.0, lambda _elapsed: hand_zero, enabled=True, epoch=2)
        reference_ready = bool(
            self.latest_diagnostic.get("reference_ready", False))
        accepted_token = self.latest_diagnostic.get("active_reference_token")

        # A bounded detector miss publishes explicit INVALID heartbeats while
        # retaining the same camera C token.  Output must hold during the
        # suspension and resume from the same reference when current evidence
        # returns; asking the operator to press C for this short gap is a bug.
        transient_origin = self.tool_position()
        transient_samples = self.stage(
            0.25,
            lambda _elapsed: hand_zero,
            enabled=True,
            epoch=2,
            valid=False,
        )
        transient_hold_motion = max(
            [float(np.linalg.norm(value-transient_origin))
             for value in transient_samples]
            or [float("inf")]
        )
        resumed_samples = self.stage(
            2.0,
            lambda elapsed: hand_zero + np.array([
                0.025 * min(1.0, elapsed/1.0), 0.0, 0.0]),
            enabled=True,
            epoch=2,
        )
        resumed_position = (
            resumed_samples[-1] if resumed_samples else self.tool_position())
        same_c_resume_base_y_motion = float(
            resumed_position[1]-transient_origin[1])
        resumed_token = self.latest_diagnostic.get("active_reference_token")
        self.stage(
            2.0,
            lambda _elapsed: hand_zero,
            enabled=True,
            epoch=2,
        )

        # Start a reachable target, then deliberately stop all UDP packets.
        # The adapter must zero motion after its watchdog, while epoch 2 stays
        # captured. Resuming that same token must continue from the original
        # C reference rather than silently re-zeroing or demanding another C.
        right_samples = self.stage(
            2.0, lambda elapsed: hand_zero + np.array([
                0.025 * min(1.0, elapsed/1.0), 0.0, 0.0]),
            enabled=True, epoch=2)
        before_gap = (
            right_samples[-1] if right_samples else self.tool_position())
        first_right_base_y_motion = float(before_gap[1]-initial[1])
        hamer_reason_start = len(self.hamer_reasons)
        gap_samples = self.silent_stage(1.0)
        after_gap = gap_samples[-1] if gap_samples else self.tool_position()
        gap_base_y_motion = float(after_gap[1]-before_gap[1])
        timeout_reasons = self.hamer_reasons[hamer_reason_start:]
        timeout_hold_seen = any(
            "HAMER_UDP_INPUT_TIMEOUT_HOLDING_C" in reason
            for reason in timeout_reasons)

        same_token_origin = self.tool_position()
        same_token_reason_start = len(self.hamer_reasons)
        same_token_samples = self.stage(
            1.5,
            lambda elapsed: hand_zero + np.array([
                0.08 if int(elapsed*5.0) % 2 == 0 else -0.08,
                0.0,
                0.0,
            ]),
            enabled=True,
            epoch=2,
        )
        same_token_max_motion = max(
            [float(np.linalg.norm(value-same_token_origin))
             for value in same_token_samples]
            or [float("inf")])
        same_token_reasons = self.hamer_reasons[same_token_reason_start:]
        same_token_block_seen = any(
            "blocked_reference_token_requires_new_c" in reason
            for reason in same_token_reasons)
        same_token_resumed_token = self.latest_diagnostic.get(
            "active_reference_token")

        # Only a genuinely newer C edge may recapture the current robot pose
        # and restore control.
        self.stage(1.0, lambda _elapsed: hand_zero, enabled=True, epoch=3)
        recaptured_zero = self.tool_position()
        reference_ready_after_new_c = bool(
            self.latest_diagnostic.get("reference_ready", False))
        accepted_new_token = self.latest_diagnostic.get(
            "active_reference_token")
        renewed_samples = self.stage(
            2.0, lambda elapsed: hand_zero + np.array([
                0.025 * min(1.0, elapsed/1.0), 0.0, 0.0]),
            enabled=True, epoch=3)
        renewed_position = (
            renewed_samples[-1] if renewed_samples else self.tool_position())
        renewed_base_y_motion = float(
            renewed_position[1]-recaptured_zero[1])
        return_samples = self.stage(
            3.0, lambda _elapsed: hand_zero, enabled=True, epoch=3)
        returned = return_samples[-1] if return_samples else self.tool_position()
        return_error = float(np.linalg.norm(returned-recaptured_zero))
        dangerous = sorted(set(value for value in self.statuses
                               if value in (2, 4, 5)))
        waiting_seen = (
            "WAITING_FOR_NEW_C_REFERENCE_AFTER_STARTUP" in self.reasons)
        passed = bool(
            locked_max_motion <= 0.003
            and waiting_seen
            and reference_ready
            and accepted_token == identity_reference_token(
                self.session, 2, 1, 1, True)
            and transient_hold_motion <= 0.003
            and same_c_resume_base_y_motion <= -0.010
            and resumed_token == accepted_token
            # Ported AprilTag V3 wrist relation: image-right is base -Y and
            # 25 mm of hand motion targets 15 mm of tool motion.
            and first_right_base_y_motion <= -0.010
            and timeout_hold_seen
            and abs(gap_base_y_motion) <= 0.003
            and not same_token_block_seen
            and same_token_max_motion >= 0.010
            and same_token_resumed_token == accepted_token
            and reference_ready_after_new_c
            and accepted_new_token == identity_reference_token(
                self.session, 3, 1, 1, True)
            and renewed_base_y_motion <= -0.010
            and return_error <= 0.010
            and not dangerous)
        return {
            "passed": passed,
            "old_c_token_locked_after_receiver_restart": (
                locked_max_motion <= 0.003),
            "locked_max_tool_motion_m": locked_max_motion,
            "waiting_for_new_c_after_startup_diagnostic_seen": waiting_seen,
            "reference_ready_after_c": reference_ready,
            "accepted_reference_token": accepted_token,
            "transient_invalid_same_c_hold_max_tool_motion_m": (
                transient_hold_motion),
            "transient_invalid_same_c_resume_base_y_motion_m": (
                same_c_resume_base_y_motion),
            "transient_invalid_same_c_reference_token_preserved": (
                resumed_token == accepted_token),
            "image_right_base_negative_y_motion_before_timeout_m": (
                first_right_base_y_motion),
            "udp_timeout_holding_c_seen": timeout_hold_seen,
            "udp_silence_duration_s": 1.0,
            "tool_base_y_motion_during_silence_m": gap_base_y_motion,
            "same_c_token_rejected": same_token_block_seen,
            "same_c_token_resume_max_tool_motion_m": same_token_max_motion,
            "same_c_token_reference_preserved": (
                same_token_resumed_token == accepted_token),
            "reference_ready_after_new_c": reference_ready_after_new_c,
            "accepted_new_reference_token": accepted_new_token,
            "image_right_base_negative_y_motion_after_new_c_m": (
                renewed_base_y_motion),
            "returned_to_c_zero": return_error <= 0.010,
            "return_translation_error_m": return_error,
            "servo_statuses_observed": sorted(set(self.statuses)),
            "servo_danger_statuses": dangerous,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--output", default="")
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("udp_c_gate_validator", anonymous=True)
    result = UdpCGateValidator(args.host, args.port).run()
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text+"\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
