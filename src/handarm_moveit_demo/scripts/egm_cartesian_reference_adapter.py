#!/usr/bin/env python3
"""250 Hz EGM-style position reference with continuous singular recovery.

The node consumes the latest base-frame Cartesian Twist produced by the
existing hand mapping/safety chain.  It replaces MoveIt Servo only in the new
Gazebo EGM profile; the established velocity profile is untouched.
"""

import json
import threading

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, Int8, String

from handarm_moveit_demo.egm_position_reference import EgmPositionReferenceModel
from handarm_moveit_demo.egm_singularity_recovery import (
    DirectionalSingularityRecovery, UrdfSerialChain)


DEFAULT_JOINTS = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DEFAULT_LOWER = [-2.87979, -1.91986, -1.91986, -2.79253, -2.094395, -6.981317]
DEFAULT_UPPER = [2.87979, 1.91986, 1.22173, 2.79253, 2.094395, 6.981317]
DEFAULT_VELOCITY = [4.36332, 4.36332, 4.36332, 5.58505, 5.58505, 7.33038]
DEFAULT_START = [0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0]


class EgmCartesianReferenceAdapter:
    def __init__(self):
        self.joints = list(rospy.get_param("~arm_joints", DEFAULT_JOINTS))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.tip_frame = str(rospy.get_param("~tip_frame", "tool0"))
        self.rate_hz = float(rospy.get_param("~rate_hz", 250.0))
        self.command_timeout_s = float(rospy.get_param(
            "~command_timeout_s", 0.10))
        self.latch_on_twist_timeout = bool(rospy.get_param(
            "~latch_on_twist_timeout", False))
        if self.rate_hz <= 0.0 or self.command_timeout_s <= 0.0:
            raise ValueError("rate_hz and command_timeout_s must be positive")

        initial = rospy.get_param("~initial_reference", DEFAULT_START)
        lower = rospy.get_param("~lower_limits", DEFAULT_LOWER)
        upper = rospy.get_param("~upper_limits", DEFAULT_UPPER)
        maximum_velocity = rospy.get_param(
            "~maximum_velocity", DEFAULT_VELOCITY)
        urdf_xml = rospy.get_param("/robot_description")
        self.chain = UrdfSerialChain.from_urdf_xml(
            urdf_xml, self.base_frame, self.tip_frame)
        if list(self.chain.joint_names) != self.joints:
            raise ValueError(
                "URDF chain joints {} do not match configured joints {}".format(
                    list(self.chain.joint_names), self.joints))

        self.resolver = DirectionalSingularityRecovery(
            chain=self.chain,
            preferred_configuration=initial,
            lower_limits=lower,
            upper_limits=upper,
            maximum_velocity=maximum_velocity,
            damping_start_condition=float(rospy.get_param(
                "~damping_start_condition", 60.0)),
            hard_condition=float(rospy.get_param(
                "~hard_condition", 180.0)),
            release_condition=float(rospy.get_param(
                "~release_condition", 45.0)),
            minimum_damping=float(rospy.get_param(
                "~minimum_damping", 1.0e-4)),
            maximum_damping=float(rospy.get_param(
                "~maximum_damping", 0.12)),
            posture_gain_per_s=float(rospy.get_param(
                "~posture_gain_per_s", 0.35)),
            recovery_gain_per_s=float(rospy.get_param(
                "~recovery_gain_per_s", 2.0)),
            recovery_velocity_utilization=float(rospy.get_param(
                "~recovery_velocity_utilization", 0.45)),
            prediction_horizon_s=float(rospy.get_param(
                "~prediction_horizon_s", 0.04)),
            joint_soft_margin_rad=float(rospy.get_param(
                "~joint_soft_margin_rad", 0.12)),
            release_cycles=int(rospy.get_param("~release_cycles", 8)),
        )
        self.reference_model = EgmPositionReferenceModel(
            joint_names=self.joints,
            initial_reference=initial,
            lower_limits=lower,
            upper_limits=upper,
            maximum_velocity=maximum_velocity,
            maximum_acceleration=rospy.get_param(
                "~maximum_acceleration",
                [20.0, 20.0, 20.0, 30.0, 30.0, 40.0]),
            command_timeout_s=self.command_timeout_s,
            joint_limit_margin_rad=float(rospy.get_param(
                "~joint_limit_margin_rad", 0.01)),
            maximum_step_dt_s=float(rospy.get_param(
                "~maximum_step_dt_s", 0.02)),
            maximum_following_error=rospy.get_param(
                "~maximum_following_error_rad",
                [0.08, 0.08, 0.08, 0.12, 0.12, 0.16]),
        )

        self.lock = threading.Lock()
        self.actual = None
        self.latest_twist = np.zeros(6, dtype=float)
        self.latest_twist_time = None
        self.reference_synchronized = False
        self.external_position_hold = False
        self.position_hold_active = False
        self.last_mode = None
        self.last_status = None
        self.tick_count = 0
        self.diagnostic_divisor = max(1, int(round(self.rate_hz / 20.0)))

        input_topic = str(rospy.get_param(
            "~cartesian_twist_topic", "/servo_server/delta_twist_cmds"))
        output_topic = str(rospy.get_param(
            "~controller_topic", "/abbarm_egm_position_controller/command"))
        diagnostics_topic = str(rospy.get_param(
            "~diagnostic_topic", "/egm_position_reference/diagnostics"))
        status_topic = str(rospy.get_param(
            "~status_topic", "/servo_server/status"))
        position_hold_topic = str(rospy.get_param(
            "~position_hold_topic",
            "/shared_teleop/egm_position_hold")).strip()
        self.reference_publisher = rospy.Publisher(
            output_topic, Float64MultiArray, queue_size=1)
        self.diagnostic_publisher = rospy.Publisher(
            diagnostics_topic, String, queue_size=1)
        self.status_publisher = rospy.Publisher(
            status_topic, Int8, queue_size=1, latch=True)
        rospy.Subscriber(
            "/joint_states", JointState, self.joint_state_callback,
            queue_size=1)
        rospy.Subscriber(
            input_topic, TwistStamped, self.twist_callback, queue_size=1)
        if position_hold_topic:
            rospy.Subscriber(
                position_hold_topic, Bool, self.position_hold_callback,
                queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.rate_hz), self.tick)
        rospy.on_shutdown(self.shutdown)
        self.status_publisher.publish(Int8(data=0))
        rospy.logwarn(
            "Gazebo EGM Cartesian position-reference enabled: %.1f Hz, "
            "adaptive damping %.1f -> hard %.1f, directional recovery to %.1f",
            self.rate_hz, self.resolver.damping_start_condition,
            self.resolver.hard_condition, self.resolver.release_condition)

    def joint_state_callback(self, message):
        values = dict(zip(message.name, message.position))
        if not all(name in values for name in self.joints):
            return
        actual = np.asarray([values[name] for name in self.joints], dtype=float)
        if not np.all(np.isfinite(actual)):
            rospy.logwarn_throttle(1.0, "Ignoring non-finite arm joint state")
            return
        with self.lock:
            self.actual = actual
            if not self.reference_synchronized:
                self.reference_model.synchronize_reference(actual)
                self.reference_synchronized = True

    def twist_callback(self, message):
        if message.header.frame_id and message.header.frame_id != self.base_frame:
            rospy.logwarn_throttle(
                1.0, "EGM Cartesian reference rejected frame %s; expected %s",
                message.header.frame_id, self.base_frame)
            return
        values = np.asarray([
            message.twist.linear.x, message.twist.linear.y,
            message.twist.linear.z, message.twist.angular.x,
            message.twist.angular.y, message.twist.angular.z], dtype=float)
        if not np.all(np.isfinite(values)):
            rospy.logwarn_throttle(1.0, "Ignoring non-finite Cartesian Twist")
            return
        with self.lock:
            self.latest_twist = values
            self.latest_twist_time = rospy.Time.now().to_sec()

    def position_hold_callback(self, message):
        with self.lock:
            self.external_position_hold = bool(message.data)

    def tick(self, _event):
        now = rospy.Time.now().to_sec()
        with self.lock:
            if self.actual is None or not self.reference_synchronized:
                rospy.logwarn_throttle(
                    1.0, "EGM Cartesian reference waiting for /joint_states")
                return
            actual = self.actual.copy()
            twist_age = (
                float("inf") if self.latest_twist_time is None else
                max(0.0, now - self.latest_twist_time))
            input_fresh = twist_age <= self.command_timeout_s
            requested_twist = (
                self.latest_twist.copy() if input_fresh else
                np.zeros(6, dtype=float))
            external_hold = self.external_position_hold
            hold_requested = bool(
                external_hold or
                (self.latch_on_twist_timeout and not input_fresh))
            if hold_requested:
                if not self.position_hold_active:
                    self.reference_model.synchronize_reference(actual)
                    self.resolver.reset(actual)
                else:
                    self.reference_model.update_actual(actual)
                self.position_hold_active = True
                requested_twist = np.zeros(6, dtype=float)
                resolution = None
            else:
                self.position_hold_active = False
                resolution = self.resolver.resolve(actual, requested_twist)
                self.reference_model.update_actual(actual)
                self.reference_model.update_velocity(
                    resolution.joint_velocity, now)
            output = self.reference_model.step(now)
        if output is None:
            return

        self.reference_publisher.publish(Float64MultiArray(
            data=output.reference.tolist()))
        status = 0 if resolution is None else (1 if (
            resolution.recovery_active or
            resolution.mode == "DAMPED") else 0)
        if status != self.last_status:
            self.status_publisher.publish(Int8(data=status))
            self.last_status = status
        solver_mode = (
            "POSITION_HOLD" if resolution is None else resolution.mode)
        if solver_mode != self.last_mode:
            if resolution is None:
                rospy.loginfo(
                    "EGM joint reference latched for stiff position hold (%s)",
                    "confirmed target loss" if external_hold else
                    "Cartesian input timeout")
            elif resolution.recovery_active:
                rospy.logwarn(
                    "EGM singularity recovery active: condition=%.1f; "
                    "holding the unsafe component and retaining retreat motion",
                    resolution.condition_number)
            elif resolution.mode == "RECOVERY_RELEASED":
                rospy.loginfo(
                    "EGM singularity recovery released: condition=%.1f",
                    resolution.condition_number)
            else:
                rospy.loginfo(
                    "EGM Cartesian reference mode=%s condition=%.1f",
                    resolution.mode, resolution.condition_number)
            self.last_mode = solver_mode

        self.tick_count += 1
        if self.tick_count % self.diagnostic_divisor == 0:
            self.status_publisher.publish(Int8(data=status))
            finite_age = twist_age if np.isfinite(twist_age) else None
            finite_condition = (None if resolution is None else (
                resolution.condition_number
                if np.isfinite(resolution.condition_number) else None))
            finite_predicted = (None if resolution is None else (
                resolution.predicted_condition_number
                if np.isfinite(resolution.predicted_condition_number) else None))
            projected_twist = (
                np.zeros(6) if resolution is None else
                resolution.projected_twist)
            joint_velocity_reference = (
                np.zeros(len(self.joints)) if resolution is None else
                resolution.joint_velocity)
            self.diagnostic_publisher.publish(String(data=json.dumps({
                "mode": "EGM_CARTESIAN_POSITION_REFERENCE",
                "solver_mode": solver_mode,
                "input_state": (
                    "POSITION_HOLD" if resolution is None else
                    "TRACKING" if input_fresh else "TWIST_TIMEOUT_ZERO"),
                "input_age_s": finite_age,
                "rate_hz": self.rate_hz,
                "condition_number": finite_condition,
                "predicted_condition_number": finite_predicted,
                "minimum_singular_value": (
                    None if resolution is None else
                    resolution.minimum_singular_value),
                "damping": None if resolution is None else resolution.damping,
                "recovery_active": bool(
                    resolution is not None and resolution.recovery_active),
                "position_hold_active": resolution is None,
                "position_hold_source": (
                    "TARGET_LOSS" if external_hold else
                    "TWIST_TIMEOUT" if resolution is None else "NONE"),
                "blocked_twist_component": (
                    0.0 if resolution is None else
                    resolution.blocked_twist_component),
                "requested_twist": requested_twist.tolist(),
                "projected_twist": projected_twist.tolist(),
                "joint_velocity_reference": joint_velocity_reference.tolist(),
                "position_reference": output.reference.tolist(),
                "actual": output.actual.tolist(),
                "following_error": output.following_error.tolist(),
                "following_error_norm_rad": float(np.linalg.norm(
                    output.following_error)),
                "following_error_clamped": output.following_error_clamped,
                "last_safe_configuration": (
                    self.resolver.last_safe.tolist() if resolution is None else
                    resolution.last_safe_configuration.tolist()),
            }, separators=(",", ":"))))

    def shutdown(self):
        self.timer.shutdown()
        # JointGroupPositionController retains the last reference, so shutdown
        # cannot turn into an uncommanded gravity fall.


def main():
    rospy.init_node("egm_cartesian_reference_adapter")
    EgmCartesianReferenceAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
