#!/usr/bin/env python3
"""Measure fixed-target finger motion while the simulated ABB arm sweeps.

This is an A/B test for the hand plant, not a perception or teleoperation
quality test. The same constant four-joint hand target is used for every
profile while three arm joints follow bounded sinusoidal position references.
"""

import json
import math
from pathlib import Path
import threading
import time

import rospy
from gazebo_msgs.srv import GetJointProperties
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = (
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
)
ACTIVE_HAND_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
MIMIC_HAND_JOINTS = ("f3j1", "f1j3", "f2j2", "f3j3")
ALL_HAND_JOINTS = ACTIVE_HAND_JOINTS + MIMIC_HAND_JOINTS
MIMIC_SOURCE = dict(zip(MIMIC_HAND_JOINTS, ACTIVE_HAND_JOINTS))
ARM_ZERO = (0.0, 0.0, 0.0, 0.0, math.pi / 2.0, 0.0)
HAND_TARGET = (0.051, 0.0317, 0.0227, 0.0363)


def percentile(values, percent):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires data")
    index = int(math.ceil(percent * len(ordered) / 100.0)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def hand_metrics(samples):
    """Return position/rate metrics from common Gazebo service samples."""

    if len(samples) < 20:
        raise ValueError("at least 20 complete Gazebo samples are required")
    result = {}
    active_targets = dict(zip(ACTIVE_HAND_JOINTS, HAND_TARGET))
    for joint in ALL_HAND_JOINTS:
        positions = [sample["position"][joint] for sample in samples]
        gazebo_rates = [abs(sample["rate"][joint]) for sample in samples]
        derived_rates = []
        for previous, current in zip(samples, samples[1:]):
            dt = current["ros_time_s"] - previous["ros_time_s"]
            if math.isfinite(dt) and dt > 1.0e-6:
                derived_rates.append(abs(
                    current["position"][joint]
                    - previous["position"][joint]
                ) / dt)
        if not derived_rates:
            raise ValueError("no causal velocity samples for " + joint)
        metric = {
            "position_min_rad": min(positions),
            "position_max_rad": max(positions),
            "position_peak_to_peak_rad": max(positions) - min(positions),
            # Match the formal evidence methodology: numerically differentiate
            # sampled joint positions. Gazebo's instantaneous service rate is
            # retained separately so sub-sample physics motion is not hidden.
            "absolute_velocity_p95_rad_s": percentile(derived_rates, 95.0),
            "absolute_velocity_max_rad_s": max(derived_rates),
            "gazebo_rate_absolute_p95_rad_s": percentile(
                gazebo_rates, 95.0
            ),
            "gazebo_rate_absolute_max_rad_s": max(gazebo_rates),
        }
        if joint in active_targets:
            errors = [abs(value - active_targets[joint]) for value in positions]
            metric["fixed_target_error_p95_rad"] = percentile(errors, 95.0)
            metric["fixed_target_error_max_rad"] = max(errors)
        else:
            source = MIMIC_SOURCE[joint]
            errors = [
                abs(sample["position"][joint] - sample["position"][source])
                for sample in samples
            ]
            metric["mimic_relation_error_p95_rad"] = percentile(errors, 95.0)
            metric["mimic_relation_error_max_rad"] = max(errors)
        result[joint] = metric
    return result


def evaluate(samples, profile, range_limit_rad, velocity_p95_limit_rad_s):
    metrics = hand_metrics(samples)
    reasons = []
    for joint, values in metrics.items():
        if values["position_peak_to_peak_rad"] > range_limit_rad:
            reasons.append("{} position range".format(joint))
        if values["absolute_velocity_p95_rad_s"] > velocity_p95_limit_rad_s:
            reasons.append("{} velocity p95".format(joint))
    active_error_limit = 0.01
    mimic_error_limit = 0.03
    for joint in ACTIVE_HAND_JOINTS:
        if metrics[joint]["fixed_target_error_p95_rad"] > active_error_limit:
            reasons.append("{} fixed target error".format(joint))
    for joint in MIMIC_HAND_JOINTS:
        if metrics[joint]["mimic_relation_error_p95_rad"] > mimic_error_limit:
            reasons.append("{} mimic relation error".format(joint))

    arm_positions = [sample["position"]["joint_1"] for sample in samples]
    arm_range = max(arm_positions) - min(arm_positions)
    if arm_range < 0.40:
        reasons.append("arm excitation range too small")
    return {
        "schema_version": 1,
        "profile": str(profile),
        "test_scope": "FIXED_HAND_TARGET_DURING_ARM_TRANSPORT",
        "sample_count": len(samples),
        "sample_duration_wall_s": samples[-1]["elapsed_wall_s"],
        "joint_1_measured_range_rad": arm_range,
        "constant_active_hand_target_rad": dict(
            zip(ACTIVE_HAND_JOINTS, HAND_TARGET)
        ),
        "limits": {
            "position_peak_to_peak_rad": float(range_limit_rad),
            "absolute_velocity_p95_rad_s": float(
                velocity_p95_limit_rad_s
            ),
            "active_target_error_p95_rad": active_error_limit,
            "mimic_relation_error_p95_rad": mimic_error_limit,
        },
        "joint_metrics": metrics,
        "failure_reasons": sorted(set(reasons)),
        "passed": not reasons,
    }


class HandTransportValidator:
    def __init__(self):
        self.profile = rospy.get_param("~profile", "unknown")
        self.output_file = Path(
            rospy.get_param(
                "~output_file",
                "/tmp/hand_transport_stability_{}.json".format(self.profile),
            )
        ).expanduser().resolve()
        self.duration_s = float(rospy.get_param("~duration_s", 12.0))
        self.settle_s = float(rospy.get_param("~settle_s", 2.0))
        self.sample_period_s = float(rospy.get_param("~sample_period_s", 0.05))
        self.range_limit_rad = float(
            rospy.get_param("~range_limit_rad", 0.01)
        )
        self.velocity_p95_limit_rad_s = float(
            rospy.get_param("~velocity_p95_limit_rad_s", 0.20)
        )
        if (
            self.duration_s <= 2.0
            or self.settle_s <= 0.0
            or self.sample_period_s <= 0.0
        ):
            raise ValueError("invalid validation durations")
        self.arm_publisher = rospy.Publisher(
            "/abbarm_egm_position_controller/command",
            Float64MultiArray,
            queue_size=1,
        )
        self.hand_publisher = rospy.Publisher(
            "/controller_gazebo_hand/command",
            JointTrajectory,
            queue_size=1,
        )
        rospy.wait_for_service("/gazebo/get_joint_properties", timeout=45.0)
        self.get_joint = rospy.ServiceProxy(
            "/gazebo/get_joint_properties", GetJointProperties
        )

    def wait_for_connections(self):
        deadline = time.monotonic() + 30.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if (
                self.arm_publisher.get_num_connections() >= 1
                and self.hand_publisher.get_num_connections() >= 1
            ):
                return
            time.sleep(0.05)
        raise RuntimeError("arm or hand command consumer is unavailable")

    def publish_arm(self, positions):
        message = Float64MultiArray()
        message.data = list(positions)
        self.arm_publisher.publish(message)

    def publish_hand_target(self):
        message = JointTrajectory()
        message.joint_names = list(ACTIVE_HAND_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = list(HAND_TARGET)
        point.time_from_start = rospy.Duration(0.5)
        message.points = [point]
        self.hand_publisher.publish(message)

    @staticmethod
    def arm_target(elapsed_s):
        return (
            0.30 * math.sin(2.0 * math.pi * elapsed_s / 4.0),
            0.0,
            0.0,
            0.40 * math.sin(2.0 * math.pi * elapsed_s / 3.2),
            math.pi / 2.0,
            0.50 * math.sin(2.0 * math.pi * elapsed_s / 2.4),
        )

    def sample(self, elapsed_wall_s):
        position = {}
        rate = {}
        for joint in ARM_JOINTS + ALL_HAND_JOINTS:
            response = self.get_joint("robot::{}".format(joint))
            if not response.success or not response.position or not response.rate:
                raise RuntimeError(
                    "Gazebo joint query failed for {}: {}".format(
                        joint, response.status_message
                    )
                )
            value = float(response.position[0])
            velocity = float(response.rate[0])
            if not math.isfinite(value) or not math.isfinite(velocity):
                raise RuntimeError("non-finite Gazebo joint state: " + joint)
            position[joint] = value
            rate[joint] = velocity
        return {
            "elapsed_wall_s": float(elapsed_wall_s),
            "ros_time_s": rospy.Time.now().to_sec(),
            "position": position,
            "rate": rate,
        }

    def hold_for(self, duration_s):
        deadline = time.monotonic() + duration_s
        last_hand_command = -float("inf")
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_hand_command >= 0.20:
                # Controller subscribers can connect before their lifecycle
                # state reaches RUNNING. Repeating one identical target during
                # pre/post hold removes that launch race without changing the
                # fixed-target experiment.
                self.publish_hand_target()
                last_hand_command = now
            self.publish_arm(ARM_ZERO)
            time.sleep(0.01)

    def run(self):
        self.wait_for_connections()
        self.hold_for(self.settle_s)

        stop = threading.Event()
        began = time.monotonic()

        def publish_sweep():
            while not stop.is_set() and not rospy.is_shutdown():
                elapsed = time.monotonic() - began
                if elapsed >= self.duration_s:
                    break
                self.publish_arm(self.arm_target(elapsed))
                time.sleep(0.01)

        publisher = threading.Thread(target=publish_sweep, daemon=True)
        publisher.start()
        samples = []
        next_sample = began
        try:
            while not rospy.is_shutdown():
                now = time.monotonic()
                if now - began >= self.duration_s:
                    break
                if now < next_sample:
                    time.sleep(next_sample - now)
                samples.append(self.sample(time.monotonic() - began))
                next_sample += self.sample_period_s
        finally:
            stop.set()
            publisher.join(timeout=1.0)
            self.hold_for(self.settle_s)

        result = evaluate(
            samples,
            self.profile,
            self.range_limit_rad,
            self.velocity_p95_limit_rad_s,
        )
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return result


def main():
    rospy.init_node("validate_hand_transport_stability")
    try:
        result = HandTransportValidator().run()
    except Exception as exc:
        rospy.logfatal("hand transport validation failed: %s", exc)
        raise SystemExit(6)
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
