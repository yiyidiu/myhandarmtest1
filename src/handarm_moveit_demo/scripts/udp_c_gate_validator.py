#!/usr/bin/env python3
"""Validate that live UDP motion requires a new C edge after receiver start."""

import argparse
import json
import socket
import time

import numpy as np
import rospy
import tf
from std_msgs.msg import Int8, String

from handarm_moveit_demo.hamer_input_contract import identity_reference_token


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

    def send(self, position, enabled, epoch):
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
            "valid": True,
            "gesture": 0,
            "gesture_confidence": 0.0,
            "invalid_reason": "",
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
        }
        self.socket.sendto(
            json.dumps(packet, separators=(",", ":")).encode("utf-8"),
            (self.host, self.port))

    def stage(self, duration_s, position_fn, enabled, epoch):
        samples = []
        started = time.monotonic()
        while (time.monotonic()-started < duration_s and
               not rospy.is_shutdown()):
            elapsed = time.monotonic()-started
            self.send(position_fn(elapsed), enabled, epoch)
            try:
                samples.append(self.tool_position())
            except Exception:
                pass
            time.sleep(0.05)
        return samples

    def silent_stage(self, duration_s):
        """Stop UDP input while observing V3 HOLD_LAST behavior."""
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

        # Start a reachable target, then deliberately stop all UDP packets.
        # The migrated V3 receiver behavior must continue republishing the
        # last target instead of letting the downstream watchdog brake.
        right_samples = self.stage(
            0.8, lambda elapsed: hand_zero + np.array([
                0.025 * min(1.0, elapsed/0.75), 0.0, 0.0]),
            enabled=True, epoch=2)
        before_gap = (
            right_samples[-1] if right_samples else self.tool_position())
        gap_samples = self.silent_stage(1.2)
        after_gap = gap_samples[-1] if gap_samples else self.tool_position()
        right_samples.extend(self.stage(
            2.0, lambda _elapsed: hand_zero + np.array([0.025, 0.0, 0.0]),
            enabled=True, epoch=2))
        moved = right_samples[-1] if right_samples else self.tool_position()
        base_y_motion = float(moved[1]-initial[1])
        gap_base_y_motion = float(after_gap[1]-before_gap[1])
        after_gap_base_y_offset = float(after_gap[1]-initial[1])
        minimum_gap_base_y_motion = min(
            [float(value[1]-initial[1]) for value in gap_samples]
            or [float("-inf")])

        target_ages = [
            float(value.get("target_input_age_s", 0.0))
            for value in self.gap_trend_diagnostics
            if value.get("target_input_age_s") is not None]
        output_ages = [
            float(value.get("source_input_age_s", float("inf")))
            for value in self.gap_output_diagnostics]
        gap_output_reasons = sorted(set(
            str(reason)
            for value in self.gap_output_diagnostics
            for reason in value.get("reasons", [])))
        hold_last_passed = bool(
            any(value.get("target_hold_active", False)
                for value in self.gap_trend_diagnostics)
            and max(target_ages or [0.0]) >= 0.8
            and max(output_ages or [float("inf")]) <= 0.15
            and "INPUT_TIMEOUT_ZERO" not in gap_output_reasons
            # A fast controller may already be at the target before silence,
            # so requiring another millimetre of travel during the gap gives
            # a false failure.  It must instead remain at/approach the held
            # target and must not retreat toward C-zero.
            and after_gap_base_y_offset <= -0.010
            and gap_base_y_motion <= 0.001
            # A 25 mm hand target maps to -15 mm in base Y.  While UDP is
            # silent the last pose may still be approached, but a stale hand
            # feed-forward velocity must not drive through it.
            and minimum_gap_base_y_motion >= -0.018)

        return_samples = self.stage(
            3.0, lambda _elapsed: hand_zero, enabled=True, epoch=2)
        returned = return_samples[-1] if return_samples else self.tool_position()
        return_error = float(np.linalg.norm(returned-initial))
        dangerous = sorted(set(value for value in self.statuses
                               if value in (2, 4, 5)))
        waiting_seen = (
            "WAITING_FOR_NEW_C_REFERENCE_AFTER_STARTUP" in self.reasons)
        passed = bool(
            locked_max_motion <= 0.003
            and waiting_seen
            and reference_ready
            and accepted_token == "{}:2".format(self.session)
            # Ported AprilTag V3 wrist relation: image-right is base -Y and
            # 25 mm of hand motion targets 15 mm of tool motion.
            and base_y_motion <= -0.010
            and hold_last_passed
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
            "image_right_base_negative_y_motion_m": base_y_motion,
            "v3_hold_last_passed": hold_last_passed,
            "udp_silence_duration_s": 1.2,
            "tool_base_y_motion_during_silence_m": gap_base_y_motion,
            "tool_base_y_offset_after_silence_m": after_gap_base_y_offset,
            "minimum_tool_base_y_offset_during_silence_m": (
                minimum_gap_base_y_motion),
            "maximum_target_input_age_during_silence_s": max(
                target_ages or [0.0]),
            "maximum_downstream_input_age_during_silence_s": max(
                output_ages or [float("inf")]),
            "downstream_reasons_during_silence": gap_output_reasons,
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
