#!/usr/bin/env python3
"""Safety gate and 50 Hz MoveIt Servo TwistStamped output adapter."""

import json
import threading
import time

import numpy as np
import rospy
import tf
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from handarm_moveit_demo.msg import HandCommand
from handarm_moveit_demo.shared_teleop_core import (
    GroundSectorWorkspace, LatestCommandShaper,
    REAL_ROBOT_AUTHORIZATION_TOKEN,
    apply_ground_sector_workspace_boundary, apply_workspace_boundary,
    robot_output_allowed,
)


def three_axis_private_parameter(name, fallback):
    value = rospy.get_param("~{}".format(name), fallback)
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    return value


class MoveItServoOutputAdapter:
    def __init__(self):
        self.config = rospy.get_param("/shared_teleop", {})
        limits = self.config.get("limits", {})
        safety = self.config.get("safety", {})
        frames = self.config.get("frames", {})
        topics = self.config.get("topics", {})
        workspace = self.config.get("workspace", {})
        self.mapping_profile_name = str(rospy.get_param(
            "~mapping_profile", "current_linear"))
        self.base_frame = frames.get("base", "base_link")
        self.workspace_mode = "AXIS_ALIGNED_BOX"
        self.ground_sector_workspace = None
        self.direction_preserving_workspace = False
        if self.mapping_profile_name == "current_linear":
            selected_workspace = workspace
        else:
            profile = self.config.get("mapping_profiles", {}).get(
                self.mapping_profile_name)
            if not isinstance(profile, dict):
                raise ValueError(
                    "unknown mapping profile {}".format(
                        self.mapping_profile_name))
            selected_workspace = profile.get("robot_workspace", {})
            if selected_workspace.get(
                    "model") != "FRONT_GROUND_CLIPPED_ELLIPSOID":
                raise ValueError("unsupported robot workspace model")
            self.workspace_mode = "FRONT_GROUND_CLIPPED_ELLIPSOID"
            self.ground_sector_workspace = GroundSectorWorkspace(
                selected_workspace.get("center_base_m"),
                selected_workspace.get("radii_m"),
                selected_workspace.get("minimum_forward_x_m"),
                selected_workspace.get("minimum_tool_z_m"),
                selected_workspace.get("utilization", 1.0),
                selected_workspace.get("boundary_margin_m", 0.0),
            )
            self.direction_preserving_workspace = bool(
                str(selected_workspace.get("boundary_policy", "")).upper()
                == "DIRECTION_PRESERVING_STOP")
        self.workspace_frame = selected_workspace.get(
            "reference_link", frames.get("servo_control", "tool0"))
        maximum_velocity = three_axis_private_parameter(
            "maximum_linear_velocity_mps",
            limits.get("maximum_linear_velocity_mps", [0.1]*3)) + \
            three_axis_private_parameter(
                "maximum_angular_velocity_radps",
                limits.get("maximum_angular_velocity_radps", [0.6]*3))
        maximum_acceleration = three_axis_private_parameter(
            "maximum_linear_acceleration_mps2",
            limits.get("maximum_linear_acceleration_mps2", [2.0]*3)) + \
            three_axis_private_parameter(
                "maximum_angular_acceleration_radps2",
                limits.get("maximum_angular_acceleration_radps2", [12.0]*3))
        self.shaper = LatestCommandShaper(
            maximum_velocity, maximum_acceleration,
            safety.get("input_timeout_s", 0.09),
            safety.get("timeout_zero_deadline_s", 0.15),
        )
        self.workspace_lower = selected_workspace.get(
            "minimum_base_m", [0.15, -0.55, 0.05])
        self.workspace_upper = selected_workspace.get(
            "maximum_base_m", [0.75, 0.55, 0.85])
        self.workspace_margin = float(selected_workspace.get(
            "soft_margin_m", 0.05))
        self.simulation = bool(rospy.get_param("~simulation", True))
        self.enable_robot = bool(rospy.get_param("~enable_robot", False))
        self.authorization = str(rospy.get_param("~robot_authorization", ""))
        self.calibration_confirmed = bool(rospy.get_param(
            "~calibration_confirmed", safety.get("real_calibration_confirmed", False)))
        self.output_allowed = robot_output_allowed(
            self.simulation, self.enable_robot,
            self.calibration_confirmed, self.authorization)
        self.listener = tf.TransformListener()
        self.lock = threading.Lock()
        self.estop_signal = False
        self.estop_latched = bool(safety.get("emergency_stop_latched", False))
        self.last_tick = None
        self.last_logged_reasons = None
        self.shutting_down = False
        self.require_full_robot_safety = bool(rospy.get_param(
            "~require_full_robot_safety", False))
        self.strict_safety_timeout_s = float(rospy.get_param(
            "~strict_safety_timeout_s", 0.20))
        if (not np.isfinite(self.strict_safety_timeout_s) or
                self.strict_safety_timeout_s <= 0.0):
            raise ValueError("strict_safety_timeout_s must be finite and positive")
        self.strict_self_collision_safe = False
        self.strict_self_collision_wall_time = 0.0
        self.safety_contract_ok = False
        self.safety_contract_received = False
        self.actual_publisher = rospy.Publisher(
            topics.get("servo_twist", "/servo_server/delta_twist_cmds"),
            TwistStamped, queue_size=1)
        self.monitor_publisher = rospy.Publisher(
            topics.get("safe_twist", "/shared_teleop/safe_twist"),
            TwistStamped, queue_size=1)
        self.diagnostics = rospy.Publisher(
            topics.get("output_diagnostics", "/shared_teleop/output_diagnostics"),
            String, queue_size=1)
        rospy.Subscriber(topics.get("assisted_command", "/shared_teleop/assisted_command"),
                         HandCommand, self.command_callback, queue_size=1)
        rospy.Subscriber(topics.get("emergency_stop", "/shared_teleop/emergency_stop"),
                         Bool, self.estop_callback, queue_size=1)
        if self.require_full_robot_safety:
            rospy.Subscriber(
                "/full_robot_self_collision_guard/safe", Bool,
                self.strict_safety_callback, queue_size=1)
            rospy.Subscriber(
                "/shared_teleop/safety_contract_ok", Bool,
                self.safety_contract_callback, queue_size=1)
        rospy.Service("/shared_teleop/reset_emergency_stop", Trigger, self.reset_estop)
        rate = float(self.config.get("control", {}).get("rate_hz", 50.0))
        self.timer = rospy.Timer(rospy.Duration(1.0/rate), self.tick)
        rospy.on_shutdown(self.halt)
        if self.simulation:
            rospy.logwarn("Servo output enabled for simulation only; enable_robot=%s is ignored", self.enable_robot)
        elif not self.output_allowed:
            rospy.logwarn("REAL ROBOT OUTPUT HARD-DISABLED: enable_robot, measured calibration, and exact authorization token are required")
        else:
            rospy.logwarn("REAL ABB OUTPUT ENABLED BY EXPLICIT AUTHORIZATION")

    @staticmethod
    def vector(message):
        return [message.twist.linear.x, message.twist.linear.y, message.twist.linear.z,
                message.twist.angular.x, message.twist.angular.y, message.twist.angular.z]

    def command_callback(self, message):
        try:
            with self.lock:
                self.shaper.update(self.vector(message), message.header.stamp.to_sec(), message.valid)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "Servo command rejected: %s", exc)

    def estop_callback(self, message):
        with self.lock:
            self.estop_signal = bool(message.data)
            if self.estop_signal:
                self.estop_latched = True
        if message.data:
            rospy.logerr("Emergency stop latched; Servo command forced to zero")

    def strict_safety_callback(self, message):
        with self.lock:
            self.strict_self_collision_safe = bool(message.data)
            self.strict_self_collision_wall_time = time.monotonic()

    def safety_contract_callback(self, message):
        with self.lock:
            self.safety_contract_ok = bool(message.data)
            self.safety_contract_received = True

    def reset_estop(self, _request):
        with self.lock:
            if self.estop_signal:
                return TriggerResponse(False, "cannot reset while emergency_stop input is true")
            self.estop_latched = False
        return TriggerResponse(True, "emergency stop latch reset")

    def make_message(self, velocity, stamp=None):
        message = TwistStamped()
        message.header.stamp = rospy.Time.now() if stamp is None else stamp
        message.header.frame_id = self.base_frame
        (message.twist.linear.x, message.twist.linear.y, message.twist.linear.z,
         message.twist.angular.x, message.twist.angular.y, message.twist.angular.z) = velocity
        return message

    def tick(self, event):
        if self.shutting_down or rospy.is_shutdown():
            return
        began = time.perf_counter()
        now = event.current_real.to_sec()
        dt = 0.0 if self.last_tick is None else max(0.0, now-self.last_tick)
        self.last_tick = now
        with self.lock:
            shaped = self.shaper.tick(now)
            estopped = self.estop_latched
            strict_safe = self.strict_self_collision_safe
            strict_age = (time.monotonic() -
                          self.strict_self_collision_wall_time)
            contract_ok = (self.safety_contract_received and
                           self.safety_contract_ok)
        velocity = shaped.velocity.copy()
        reasons = [] if shaped.reason == "NONE" else [shaped.reason]
        try:
            position, _ = self.listener.lookupTransform(
                self.base_frame, self.workspace_frame, rospy.Time(0))
            if self.ground_sector_workspace is None:
                velocity[:3], workspace_reasons = apply_workspace_boundary(
                    position, velocity[:3], self.workspace_lower,
                    self.workspace_upper, self.workspace_margin)
            else:
                velocity[:3], workspace_reasons = (
                    apply_ground_sector_workspace_boundary(
                        position, velocity[:3],
                        self.ground_sector_workspace,
                        self.workspace_margin,
                        self.direction_preserving_workspace))
            reasons.extend(workspace_reasons)
        except Exception as exc:
            velocity[:] = 0.0
            reasons.append("WORKSPACE_TF_UNAVAILABLE")
            rospy.logwarn_throttle(1.0, "Workspace TF unavailable, fail-closed: %s", exc)
        if estopped:
            velocity[:] = 0.0
            reasons.append("EMERGENCY_STOP_LATCHED")
        if self.require_full_robot_safety:
            if not contract_ok:
                velocity[:] = 0.0
                reasons.append("SAFETY_CONTRACT_NOT_READY")
            if (not strict_safe or
                    strict_age > self.strict_safety_timeout_s):
                velocity[:] = 0.0
                reasons.append("STRICT_SELF_COLLISION_GUARD_UNSAFE")
        message = self.make_message(velocity, event.current_real)
        self.monitor_publisher.publish(message)
        if self.output_allowed:
            self.actual_publisher.publish(message)
        process_ms = (time.perf_counter()-began)*1000.0
        self.diagnostics.publish(String(data=json.dumps({
            "stamp": now, "source_input_age_s": shaped.input_age_s,
            "actual_loop_hz": 0.0 if dt <= 0.0 else 1.0/dt,
            "processing_ms": process_ms,
            "valid": (shaped.valid and not estopped and
                      (not self.require_full_robot_safety or
                       (contract_ok and strict_safe and
                        strict_age <= self.strict_safety_timeout_s))),
            "output_allowed": self.output_allowed, "simulation": self.simulation,
            "enable_robot": self.enable_robot, "reasons": reasons or ["NONE"],
            "mapping_profile": self.mapping_profile_name,
            "workspace_mode": self.workspace_mode,
            "direction_preserving_workspace": bool(
                self.direction_preserving_workspace),
            "require_full_robot_safety": self.require_full_robot_safety,
            "strict_self_collision_safe": strict_safe,
            "strict_self_collision_status_age_s": strict_age,
            "safety_contract_ok": contract_ok,
            "velocity": velocity.tolist(),
        }, separators=(",", ":"))))
        reason_key = tuple(reasons)
        if reason_key != self.last_logged_reasons:
            if reasons:
                rospy.logwarn("Servo safety status changed: %s",
                              ",".join(reasons))
            elif self.last_logged_reasons:
                rospy.loginfo("Servo safety status recovered: NONE")
            self.last_logged_reasons = reason_key

    def halt(self):
        self.shutting_down = True
        self.timer.shutdown()
        if not self.output_allowed:
            return
        zero = self.make_message(np.zeros(6))
        for _ in range(4):
            self.actual_publisher.publish(zero)
            rospy.sleep(0.01)


def main():
    rospy.init_node("moveit_servo_output_adapter")
    MoveItServoOutputAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
