#!/usr/bin/env python3
"""250 Hz, queue-free EGM-style position reference for Gazebo only."""

import json
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, Int8, String

from handarm_moveit_demo.egm_position_reference import (
    EgmPositionReferenceModel, collision_proximity_hold_required)


DEFAULT_JOINTS = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DEFAULT_LOWER = [-2.87979, -1.91986, -1.91986, -2.79253, -2.094395, -6.981317]
DEFAULT_UPPER = [2.87979, 1.91986, 1.22173, 2.79253, 2.094395, 6.981317]
DEFAULT_MAXIMUM_VELOCITY = [4.36332, 4.36332, 4.36332, 5.58505, 5.58505, 7.33038]


class EgmPositionReferenceAdapter:
    def __init__(self):
        self.joints = list(rospy.get_param("~arm_joints", DEFAULT_JOINTS))
        self.rate_hz = float(rospy.get_param("~rate_hz", 250.0))
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        self.model = EgmPositionReferenceModel(
            joint_names=self.joints,
            initial_reference=rospy.get_param(
                "~initial_reference", [0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0]),
            lower_limits=rospy.get_param("~lower_limits", DEFAULT_LOWER),
            upper_limits=rospy.get_param("~upper_limits", DEFAULT_UPPER),
            maximum_velocity=rospy.get_param(
                "~maximum_velocity", DEFAULT_MAXIMUM_VELOCITY),
            maximum_acceleration=rospy.get_param(
                "~maximum_acceleration", [8.0, 8.0, 8.0, 12.0, 12.0, 16.0]),
            command_timeout_s=float(rospy.get_param("~command_timeout_s", 0.10)),
            joint_limit_margin_rad=float(rospy.get_param(
                "~joint_limit_margin_rad", 0.01)),
            maximum_step_dt_s=float(rospy.get_param(
                "~maximum_step_dt_s", 0.02)),
            maximum_following_error=rospy.get_param(
                "~maximum_following_error_rad",
                [0.04, 0.04, 0.04, 0.06, 0.06, 0.08]),
        )
        self.lock = threading.Lock()
        self.latest_actual = None
        self.latest_servo_status = 0
        self.latest_collision_scale = 1.0
        self.last_collision_scale_monotonic = None
        self.collision_monitor_seen_once = False
        self.retreat_authorized = False
        self.last_retreat_authorization_monotonic = None
        self.hard_stop_collision_scale = float(rospy.get_param(
            "~hard_stop_collision_scale", 0.20))
        self.retreat_authorization_timeout_s = float(rospy.get_param(
            "~retreat_authorization_timeout_s", 0.12))
        self.collision_scale_timeout_s = float(rospy.get_param(
            "~collision_scale_timeout_s", 0.25))
        if not 0.0 <= self.hard_stop_collision_scale < 1.0:
            raise ValueError("hard_stop_collision_scale must be in [0, 1)")
        if self.retreat_authorization_timeout_s <= 0.0:
            raise ValueError("retreat_authorization_timeout_s must be positive")
        if self.collision_scale_timeout_s <= 0.0:
            raise ValueError("collision_scale_timeout_s must be positive")
        self.last_state_key = None
        self.diagnostic_divisor = max(1, int(round(self.rate_hz / 20.0)))
        self.tick_count = 0

        raw_topic = rospy.get_param(
            "~raw_velocity_topic", "/egm_position_reference/raw_joint_velocity")
        controller_topic = rospy.get_param(
            "~controller_topic", "/abbarm_egm_position_controller/command")
        diagnostic_topic = rospy.get_param(
            "~diagnostic_topic", "/egm_position_reference/diagnostics")
        self.reference_publisher = rospy.Publisher(
            controller_topic, Float64MultiArray, queue_size=1)
        self.diagnostic_publisher = rospy.Publisher(
            diagnostic_topic, String, queue_size=1)
        rospy.Subscriber(
            "/joint_states", JointState, self.joint_state_callback, queue_size=1)
        rospy.Subscriber(
            raw_topic, Float64MultiArray, self.velocity_callback, queue_size=1)
        rospy.Subscriber(
            rospy.get_param("~servo_status_topic", "/servo_server/status"),
            Int8, self.servo_status_callback, queue_size=1)
        rospy.Subscriber(
            rospy.get_param(
                "~collision_scale_topic",
                "/servo_server/internal/collision_velocity_scale"),
            Float64, self.collision_scale_callback, queue_size=1)
        rospy.Subscriber(
            rospy.get_param(
                "~retreat_authorization_topic",
                "/shared_teleop/collision_retreat_authorized"),
            Bool, self.retreat_authorization_callback, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.rate_hz), self.tick)
        rospy.on_shutdown(self.shutdown)
        rospy.logwarn(
            "Gazebo EGM position-reference emulation enabled: latest-only %.1f Hz, "
            "raw=%s, output=%s", self.rate_hz, raw_topic, controller_topic)

    def joint_state_callback(self, message):
        by_name = dict(zip(message.name, message.position))
        if not all(name in by_name for name in self.joints):
            return
        actual = [by_name[name] for name in self.joints]
        if not np.all(np.isfinite(actual)):
            rospy.logwarn_throttle(1.0, "Ignoring non-finite arm joint state")
            return
        with self.lock:
            self.latest_actual = actual

    def velocity_callback(self, message):
        if len(message.data) != len(self.joints):
            rospy.logwarn_throttle(
                1.0, "EGM reference rejected velocity array of size %d; expected %d",
                len(message.data), len(self.joints))
            return
        try:
            with self.lock:
                self.model.update_velocity(
                    message.data, rospy.Time.now().to_sec())
        except ValueError as exc:
            rospy.logwarn_throttle(1.0, "EGM reference rejected velocity: %s", exc)

    def servo_status_callback(self, message):
        with self.lock:
            self.latest_servo_status = int(message.data)

    def collision_scale_callback(self, message):
        scale = float(message.data)
        if not np.isfinite(scale):
            rospy.logwarn_throttle(
                1.0, "Ignoring non-finite Servo collision scale")
            return
        with self.lock:
            self.latest_collision_scale = float(np.clip(scale, 0.0, 1.0))
            self.last_collision_scale_monotonic = time.monotonic()
            self.collision_monitor_seen_once = True

    def retreat_authorization_callback(self, message):
        with self.lock:
            self.retreat_authorized = bool(message.data)
            self.last_retreat_authorization_monotonic = time.monotonic()

    def tick(self, _event):
        now = rospy.Time.now().to_sec()
        now_monotonic = time.monotonic()
        with self.lock:
            if self.latest_actual is not None:
                self.model.update_actual(self.latest_actual)
            retreat_age_s = (
                float("inf")
                if self.last_retreat_authorization_monotonic is None else
                max(0.0, now_monotonic -
                    self.last_retreat_authorization_monotonic))
            collision_scale_age_s = (
                float("inf")
                if self.last_collision_scale_monotonic is None else
                max(0.0, now_monotonic -
                    self.last_collision_scale_monotonic))
            collision_monitor_stale = bool(
                collision_scale_age_s > self.collision_scale_timeout_s)
            proximity_hold = collision_proximity_hold_required(
                self.latest_collision_scale,
                self.hard_stop_collision_scale,
                self.retreat_authorized,
                retreat_age_s,
                self.retreat_authorization_timeout_s)
            hard_safety_hold = bool(
                self.latest_servo_status in (2, 4, 5) or proximity_hold or
                collision_monitor_stale)
            if hard_safety_hold and self.latest_actual is not None:
                # A zero qdot alone is insufficient for a position plant if an
                # older reference is still ahead of feedback.  Re-anchor once
                # per safety tick so the controller cannot continue pulling
                # toward a collision, singularity or joint bound.
                if (not collision_monitor_stale or
                        self.collision_monitor_seen_once):
                    self.model.synchronize_reference(self.latest_actual)
                else:
                    # Before the collision monitor publishes its first sample,
                    # keep the configured industrial-arm start reference.  The
                    # robot is still settling under gravity at this point, so
                    # capturing feedback here would permanently preserve sag.
                    self.model.hold_reference(self.latest_actual)
            output = self.model.step(now)
        if output is None:
            rospy.logwarn_throttle(
                1.0, "EGM reference waiting for complete /joint_states")
            return

        self.reference_publisher.publish(
            Float64MultiArray(data=output.reference.tolist()))
        state_key = (
            output.command_fresh, output.limit_clamped, output.time_reset,
            hard_safety_hold, collision_monitor_stale,
            self.latest_servo_status)
        if state_key != self.last_state_key:
            rospy.loginfo(
                "Hybrid reference state: command=%s limit_clamped=%s "
                "time_reset=%s servo_status=%d collision_scale=%.3f",
                ("SAFETY_HOLD" if hard_safety_hold else
                 "TRACKING" if output.command_fresh else "HOLD_POSITION"),
                output.limit_clamped, output.time_reset,
                self.latest_servo_status, self.latest_collision_scale)
            self.last_state_key = state_key

        self.tick_count += 1
        if self.tick_count % self.diagnostic_divisor == 0:
            finite_age = (
                output.command_age_s if np.isfinite(output.command_age_s) else None)
            self.diagnostic_publisher.publish(String(data=json.dumps({
                "mode": "MOVEIT_SERVO_SAFE_POSITION_REFERENCE",
                "rate_hz": self.rate_hz,
                "command_state": (
                    "SAFETY_HOLD" if hard_safety_hold else
                    "TRACKING" if output.command_fresh else "HOLD_POSITION"),
                "command_age_s": finite_age,
                "reference": output.reference.tolist(),
                "actual": output.actual.tolist(),
                "following_error": output.following_error.tolist(),
                "following_error_norm_rad": float(np.linalg.norm(
                    output.following_error)),
                "feedforward_velocity": output.feedforward_velocity.tolist(),
                "limit_clamped": output.limit_clamped,
                "following_error_clamped": output.following_error_clamped,
                "servo_status": self.latest_servo_status,
                "collision_velocity_scale": self.latest_collision_scale,
                "hard_safety_hold": hard_safety_hold,
                "collision_proximity_hold": proximity_hold,
                "collision_monitor_stale": collision_monitor_stale,
                "collision_scale_age_s": (
                    collision_scale_age_s
                    if np.isfinite(collision_scale_age_s) else None),
                "retreat_authorized": self.retreat_authorized,
                "retreat_authorization_age_s": (
                    retreat_age_s if np.isfinite(retreat_age_s) else None),
            }, separators=(",", ":"))))

    def shutdown(self):
        self.timer.shutdown()
        # The direct position controller retains the last published reference;
        # publishing a velocity zero is unnecessary and cannot move the target.


def main():
    rospy.init_node("egm_position_reference_adapter")
    EgmPositionReferenceAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
