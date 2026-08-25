#!/usr/bin/env python3
"""Drive toward a known invalid arm state and verify pre-contact stopping."""

import math
import threading
import time

import rospy
from control_msgs.msg import JointJog
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Int8, String


ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DEFAULT_INVALID_TARGET = [
    -1.99227410, 1.33092623, 0.98126174,
    -2.34297608, 0.60349443, -3.67728288,
]
COLLISION_STATUSES = {3, 4}


class DynamicSelfCollisionAcceptance:
    def __init__(self):
        self.target = [float(value) for value in rospy.get_param(
            "~invalid_target_rad", DEFAULT_INVALID_TARGET)]
        self.rate_hz = float(rospy.get_param("~rate_hz", 50.0))
        self.timeout_s = float(rospy.get_param("~timeout_s", 24.0))
        self.maximum_speed = float(rospy.get_param(
            "~maximum_joint_speed_radps", 0.45))
        self.maximum_acceleration = float(rospy.get_param(
            "~maximum_joint_acceleration_radps2", 0.80))
        self.gain = float(rospy.get_param("~joint_position_gain", 1.5))
        self.minimum_motion_rad = float(rospy.get_param(
            "~minimum_motion_rad", 0.30))
        self.retreat_timeout_s = float(rospy.get_param(
            "~retreat_timeout_s", 25.0))
        self.maximum_return_error_rad = float(rospy.get_param(
            "~maximum_return_error_rad", 0.01))
        if len(self.target) != len(ARM_JOINTS):
            raise ValueError("invalid_target_rad must contain six values")
        numeric = (self.target + [self.rate_hz, self.timeout_s,
                                  self.maximum_speed, self.maximum_acceleration,
                                  self.gain,
                                  self.minimum_motion_rad,
                                  self.retreat_timeout_s,
                                  self.maximum_return_error_rad])
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("acceptance parameters must be finite")
        if (self.rate_hz < 20.0 or self.timeout_s <= 0.0 or
                self.maximum_speed <= 0.0 or
                self.maximum_acceleration <= 0.0 or self.gain <= 0.0 or
                self.retreat_timeout_s <= 0.0 or
                self.maximum_return_error_rad <= 0.0):
            raise ValueError("invalid acceptance timing or gain")

        self.lock = threading.Lock()
        self.positions = None
        self.strict_safe = None
        self.servo_status = None
        self.collision_scale = None
        self.minimum_scale = 1.0
        self.collision_status_armed = False
        self.fresh_collision_status = False
        self.acceptance_active = False
        self.predicted_collision_block = False
        self.command_diagnostic = None
        self.strict_ever_unsafe = False
        self.publisher = rospy.Publisher(
            "/servo_server/delta_joint_cmds", JointJog, queue_size=1)
        rospy.Subscriber("/joint_states", JointState,
                         self.joint_state_callback, queue_size=1)
        rospy.Subscriber("/full_robot_self_collision_guard/safe", Bool,
                         self.strict_callback, queue_size=1)
        rospy.Subscriber("/servo_server/status", Int8,
                         self.status_callback, queue_size=1)
        rospy.Subscriber(
            "/servo_server/internal/collision_velocity_scale", Float64,
            self.scale_callback, queue_size=1)
        rospy.Subscriber(
            "/full_robot_self_collision_guard/command_diagnostic", String,
            self.command_diagnostic_callback, queue_size=1)
        rospy.wait_for_service(
            "/full_robot_self_collision_guard/check_state_validity", 20.0)
        self.check_state = rospy.ServiceProxy(
            "/full_robot_self_collision_guard/check_state_validity",
            GetStateValidity)

    def joint_state_callback(self, message):
        values = dict(zip(message.name, message.position))
        if all(name in values for name in ARM_JOINTS):
            with self.lock:
                self.positions = [values[name] for name in ARM_JOINTS]

    def strict_callback(self, message):
        with self.lock:
            self.strict_safe = bool(message.data)
            if not message.data:
                self.strict_ever_unsafe = True

    def status_callback(self, message):
        with self.lock:
            self.servo_status = int(message.data)
            if not self.acceptance_active:
                return
            if self.servo_status not in COLLISION_STATUSES:
                self.collision_status_armed = True
            elif self.collision_status_armed:
                # Servo status is not latched, but the last published value can
                # remain visible between separate validator runs. Count only a
                # collision status that follows a fresh non-collision status in
                # this run; distance scaling and the strict predictive gate are
                # tracked independently below.
                self.fresh_collision_status = True

    def scale_callback(self, message):
        with self.lock:
            self.collision_scale = float(message.data)
            if self.acceptance_active:
                self.minimum_scale = min(self.minimum_scale,
                                         self.collision_scale)

    def command_diagnostic_callback(self, message):
        with self.lock:
            self.command_diagnostic = str(message.data)
            if (self.acceptance_active and
                    self.command_diagnostic.startswith(
                        "PREDICTED_SELF_COLLISION:")):
                self.predicted_collision_block = True

    def snapshot(self):
        with self.lock:
            return (None if self.positions is None else list(self.positions),
                    self.strict_safe, self.servo_status,
                    self.collision_scale, self.minimum_scale,
                    self.fresh_collision_status,
                    self.predicted_collision_block,
                    self.strict_ever_unsafe,
                    self.command_diagnostic)

    def wait_ready(self):
        deadline = time.monotonic() + 20.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            positions, strict, status, scale, _, _, _, _, _ = self.snapshot()
            if (positions is not None and strict is True and
                    status is not None and scale is not None and
                    self.publisher.get_num_connections() > 0):
                return positions
            rospy.sleep(0.05)
        raise RuntimeError("collision acceptance interfaces did not become ready")

    def candidate_valid(self, values):
        request_state = RobotState()
        request_state.joint_state.name = list(ARM_JOINTS)
        request_state.joint_state.position = list(values)
        return bool(self.check_state(robot_state=request_state,
                                     group_name="abbarm").valid)

    def publish(self, velocities):
        command = JointJog()
        command.header.stamp = rospy.Time.now()
        command.joint_names = list(ARM_JOINTS)
        command.velocities = list(velocities)
        command.duration = 1.0 / self.rate_hz
        self.publisher.publish(command)

    def halt(self):
        for _ in range(8):
            self.publish([0.0] * len(ARM_JOINTS))
            rospy.sleep(1.0 / self.rate_hz)

    def retreat_to(self, target):
        """Command away from the collision boundary and verify recovery."""
        rate = rospy.Rate(self.rate_hz)
        deadline = time.monotonic() + self.retreat_timeout_s
        commanded_velocities = [0.0] * len(ARM_JOINTS)
        maximum_delta_velocity = self.maximum_acceleration / self.rate_hz
        return_error = math.inf
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                (positions, strict, _, _, _, _, _, strict_unsafe,
                 _) = self.snapshot()
                if strict_unsafe or strict is False:
                    raise RuntimeError(
                        "strict guard became unsafe during recovery")
                if positions is None:
                    raise RuntimeError(
                        "joint state disappeared during recovery")
                errors = [desired - current
                          for desired, current in zip(target, positions)]
                return_error = max(abs(value) for value in errors)
                if return_error <= self.maximum_return_error_rad:
                    break
                desired_velocities = [
                    max(-self.maximum_speed,
                        min(self.maximum_speed, self.gain * error))
                    for error in errors
                ]
                commanded_velocities = [
                    previous + max(-maximum_delta_velocity,
                                   min(maximum_delta_velocity,
                                       desired - previous))
                    for previous, desired in zip(
                        commanded_velocities, desired_velocities)
                ]
                self.publish(commanded_velocities)
                rate.sleep()
        finally:
            self.halt()
        recovered, strict, _, _, _, _, _, strict_unsafe, _ = self.snapshot()
        if recovered is None:
            raise RuntimeError("joint state unavailable after recovery")
        return_error = max(abs(desired - current)
                           for desired, current in zip(target, recovered))
        recovered_valid = self.candidate_valid(recovered)
        recovery_passed = (
            return_error <= self.maximum_return_error_rad and
            not strict_unsafe and strict is True and recovered_valid)
        rospy.logwarn(
            "DYNAMIC_SELF_COLLISION_RECOVERY %s return_error=%.6f "
            "strict=%s final_valid=%s recovered=%s",
            "PASS" if recovery_passed else "FAIL", return_error, strict,
            recovered_valid, [round(value, 5) for value in recovered])
        return recovery_passed

    def run(self):
        initial = self.wait_ready()
        # The strict guard intentionally publishes false while it is waiting
        # for the first complete JointState.  Acceptance begins only after a
        # fresh true status, so discard that expected startup history.
        with self.lock:
            self.strict_ever_unsafe = False
            self.minimum_scale = (1.0 if self.collision_scale is None else
                                  float(self.collision_scale))
            self.collision_status_armed = (
                self.servo_status not in COLLISION_STATUSES)
            self.fresh_collision_status = False
            self.predicted_collision_block = False
            self.acceptance_active = True
        if self.candidate_valid(self.target):
            raise RuntimeError("configured target is not self-colliding")
        rospy.logwarn(
            "Acceptance target confirmed invalid; commanding toward it from %s",
            [round(value, 5) for value in initial])

        rate = rospy.Rate(self.rate_hz)
        deadline = time.monotonic() + self.timeout_s
        collision_observed_at = None
        commanded_velocities = [0.0] * len(ARM_JOINTS)
        maximum_delta_velocity = self.maximum_acceleration / self.rate_hz
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                (positions, strict, status, scale, minimum_scale,
                 fresh_collision_status, predicted_block, strict_unsafe,
                 command_diagnostic) = self.snapshot()
                if strict_unsafe or strict is False:
                    raise RuntimeError(
                        "strict guard became unsafe: contact was not prevented")
                if positions is None:
                    raise RuntimeError("joint state disappeared")
                desired_velocities = [
                    max(-self.maximum_speed,
                        min(self.maximum_speed,
                            self.gain * (target - current)))
                    for target, current in zip(self.target, positions)
                ]
                commanded_velocities = [
                    previous + max(-maximum_delta_velocity,
                                   min(maximum_delta_velocity,
                                       desired - previous))
                    for previous, desired in zip(
                        commanded_velocities, desired_velocities)
                ]
                self.publish(commanded_velocities)
                collision_response = (
                    fresh_collision_status or predicted_block or
                    (scale is not None and scale < 0.98))
                if collision_response:
                    if collision_observed_at is None:
                        collision_observed_at = time.monotonic()
                        rospy.logwarn(
                            "Pre-contact response observed: status=%s scale=%.6f "
                            "predicted_block=%s diagnostic=%s",
                            status, scale, predicted_block,
                            command_diagnostic)
                    # Keep the command applied briefly: a working collision
                    # monitor must hold/decelerate rather than crossing contact.
                    if time.monotonic() - collision_observed_at >= 1.0:
                        break
                rate.sleep()
        finally:
            self.halt()
            with self.lock:
                self.acceptance_active = False

        (approach_final, strict, status, scale, minimum_scale,
         fresh_collision_status, predicted_block, strict_unsafe,
         command_diagnostic) = self.snapshot()
        maximum_motion = max(abs(end - start)
                             for start, end in zip(initial, approach_final))
        final_valid = self.candidate_valid(approach_final)
        target_error = max(abs(target - current)
                           for target, current in zip(
                               self.target, approach_final))
        approach_passed = (
            maximum_motion >= self.minimum_motion_rad and
            collision_observed_at is not None and
            (fresh_collision_status or predicted_block or
             minimum_scale < 0.98) and
            not strict_unsafe and strict is True and final_valid and
            target_error > 0.02)
        rospy.logwarn(
            "DYNAMIC_SELF_COLLISION_APPROACH %s motion=%.6f min_scale=%.6f "
            "status=%s fresh_status=%s predicted_block=%s strict=%s "
            "final_valid=%s target_error=%.6f final=%s",
            "PASS" if approach_passed else "FAIL", maximum_motion,
            minimum_scale,
            status, fresh_collision_status, predicted_block, strict,
            final_valid, target_error,
            [round(value, 5) for value in approach_final])

        recovery_passed = self.retreat_to(initial)
        passed = approach_passed and recovery_passed
        rospy.logwarn(
            "DYNAMIC_SELF_COLLISION_ACCEPTANCE %s approach=%s recovery=%s",
            "PASS" if passed else "FAIL", approach_passed, recovery_passed)
        if not passed:
            raise RuntimeError("dynamic self-collision acceptance failed")


def main():
    rospy.init_node("validate_dynamic_self_collision_stop")
    DynamicSelfCollisionAcceptance().run()


if __name__ == "__main__":
    main()
