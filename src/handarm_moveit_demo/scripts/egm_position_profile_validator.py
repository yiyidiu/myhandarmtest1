#!/usr/bin/env python3
"""Validate motion, hold stiffness, disturbance recovery and zero return."""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import rospy
from gazebo_msgs.srv import ApplyJointEffort, ApplyJointEffortRequest
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
TEST_AXIS = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="/tmp/handarm_egm_position_validation.json")
    parser.add_argument("--startup-timeout-s", type=float, default=15.0)
    return parser.parse_args(rospy.myargv()[1:])


class EgmPositionProfileValidator:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.diagnostic = None
        self.actual = None
        self.actual_velocity = None
        self.reference_receive_times = []
        self.velocity_publisher = rospy.Publisher(
            "/egm_position_reference/raw_joint_velocity",
            Float64MultiArray,
            queue_size=1,
        )
        rospy.Subscriber(
            "/egm_position_reference/diagnostics",
            String,
            self.diagnostic_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/abbarm_egm_position_controller/command",
            Float64MultiArray,
            self.reference_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/joint_states", JointState, self.joint_state_callback, queue_size=1)

    def diagnostic_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.diagnostic = payload

    def reference_callback(self, _message):
        with self.lock:
            self.reference_receive_times.append(time.monotonic())

    def joint_state_callback(self, message):
        positions = dict(zip(message.name, message.position))
        velocities = dict(zip(message.name, message.velocity))
        if not all(name in positions and name in velocities for name in JOINTS):
            return
        actual = np.asarray([positions[name] for name in JOINTS], dtype=float)
        actual_velocity = np.asarray(
            [velocities[name] for name in JOINTS], dtype=float)
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(actual_velocity)):
            return
        with self.lock:
            self.actual = actual
            self.actual_velocity = actual_velocity

    def snapshot(self):
        with self.lock:
            diagnostic = (
                None if self.diagnostic is None else dict(self.diagnostic))
            actual = None if self.actual is None else self.actual.copy()
            velocity = (
                None if self.actual_velocity is None
                else self.actual_velocity.copy())
        return diagnostic, actual, velocity

    def wait_until(self, predicate, timeout_s, description):
        deadline = time.monotonic() + timeout_s
        rate = rospy.Rate(100.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            state = self.snapshot()
            if predicate(*state):
                return state
            rate.sleep()
        raise RuntimeError("timed out waiting for " + description)

    def publish_velocity_for(self, velocity, duration_s):
        message = Float64MultiArray(data=list(velocity))
        deadline = time.monotonic() + duration_s
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.velocity_publisher.publish(message)
            rate.sleep()

    def return_reference(self, target):
        rate = rospy.Rate(50.0)
        deadline = time.monotonic() + 4.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            diagnostic, _actual, _velocity = self.snapshot()
            if diagnostic is None:
                rate.sleep()
                continue
            reference = np.asarray(diagnostic["reference"], dtype=float)
            feedforward = np.asarray(
                diagnostic["feedforward_velocity"], dtype=float)
            error = float(target - reference[TEST_AXIS])
            if abs(error) <= 0.0015 and abs(feedforward[TEST_AXIS]) <= 0.015:
                break
            command = np.zeros(6)
            command[TEST_AXIS] = float(np.clip(4.0 * error, -0.35, 0.35))
            self.velocity_publisher.publish(Float64MultiArray(data=command.tolist()))
            rate.sleep()
        self.publish_velocity_for(np.zeros(6), 0.20)

    def reference_rate_hz(self):
        with self.lock:
            times = list(self.reference_receive_times)
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return float((len(times) - 1) / (times[-1] - times[0]))

    def run(self):
        self.wait_until(
            lambda diagnostic, actual, _velocity: (
                diagnostic is not None and actual is not None and
                diagnostic.get("mode") == "EGM_POSITION_REFERENCE_EMULATION"),
            self.args.startup_timeout_s,
            "EGM position diagnostics and complete joint state",
        )
        self.wait_until(
            lambda diagnostic, _actual, velocity: (
                diagnostic is not None and velocity is not None and
                diagnostic.get("command_state") == "HOLD_POSITION" and
                float(diagnostic.get("following_error_norm_rad", 1.0)) < 0.005 and
                float(np.linalg.norm(velocity)) < 0.01),
            self.args.startup_timeout_s,
            "stiff, stationary initial hold",
        )
        baseline, baseline_actual, baseline_velocity = self.snapshot()
        baseline_reference = np.asarray(baseline["reference"], dtype=float)

        command = np.zeros(6)
        command[TEST_AXIS] = 0.35
        self.publish_velocity_for(command, 0.60)
        self.publish_velocity_for(np.zeros(6), 0.16)
        moved, moved_actual, _moved_velocity = self.wait_until(
            lambda diagnostic, actual, velocity: (
                diagnostic is not None and actual is not None and velocity is not None and
                diagnostic.get("command_state") == "HOLD_POSITION" and
                abs(float(diagnostic["following_error"][TEST_AXIS])) < 0.015 and
                abs(float(velocity[TEST_AXIS])) < 0.02),
            3.0,
            "motion completion and position hold",
        )
        moved_reference = np.asarray(moved["reference"], dtype=float)
        reference_motion = float(
            moved_reference[TEST_AXIS] - baseline_reference[TEST_AXIS])

        hold_references = []
        hold_actual_errors = []
        hold_deadline = time.monotonic() + 0.60
        rate = rospy.Rate(100.0)
        while not rospy.is_shutdown() and time.monotonic() < hold_deadline:
            diagnostic, actual, _velocity = self.snapshot()
            if diagnostic is not None and actual is not None:
                reference = float(diagnostic["reference"][TEST_AXIS])
                hold_references.append(reference)
                hold_actual_errors.append(abs(reference - float(actual[TEST_AXIS])))
            rate.sleep()
        reference_drift = (
            float(max(hold_references) - min(hold_references))
            if hold_references else float("inf"))
        hold_error_max = (
            float(max(hold_actual_errors))
            if hold_actual_errors else float("inf"))

        rospy.wait_for_service("/gazebo/apply_joint_effort", timeout=5.0)
        effort_request = ApplyJointEffortRequest()
        effort_request.joint_name = JOINTS[TEST_AXIS]
        effort_request.effort = 35.0
        effort_request.start_time = rospy.Time(0)
        effort_request.duration = rospy.Duration.from_sec(0.15)
        effort_response = rospy.ServiceProxy(
            "/gazebo/apply_joint_effort", ApplyJointEffort)(effort_request)
        if not effort_response.success:
            raise RuntimeError("Gazebo rejected test disturbance")

        disturbance_errors = []
        disturbance_deadline = time.monotonic() + 0.60
        while not rospy.is_shutdown() and time.monotonic() < disturbance_deadline:
            _diagnostic, actual, _velocity = self.snapshot()
            if actual is not None:
                disturbance_errors.append(abs(
                    moved_reference[TEST_AXIS] - float(actual[TEST_AXIS])))
            rate.sleep()
        peak_disturbance_error = (
            float(max(disturbance_errors))
            if disturbance_errors else 0.0)

        recovered, recovered_actual, recovered_velocity = self.wait_until(
            lambda diagnostic, actual, velocity: (
                diagnostic is not None and actual is not None and velocity is not None and
                abs(float(diagnostic["reference"][TEST_AXIS]) -
                    float(actual[TEST_AXIS])) < 0.010 and
                abs(float(velocity[TEST_AXIS])) < 0.02),
            3.0,
            "post-disturbance recovery",
        )
        recovered_error = abs(
            float(recovered["reference"][TEST_AXIS]) -
            float(recovered_actual[TEST_AXIS]))

        self.return_reference(baseline_reference[TEST_AXIS])
        returned, returned_actual, returned_velocity = self.wait_until(
            lambda diagnostic, actual, velocity: (
                diagnostic is not None and actual is not None and velocity is not None and
                diagnostic.get("command_state") == "HOLD_POSITION" and
                abs(float(diagnostic["reference"][TEST_AXIS]) -
                    float(baseline_reference[TEST_AXIS])) < 0.004 and
                abs(float(actual[TEST_AXIS]) -
                    float(baseline_reference[TEST_AXIS])) < 0.012 and
                abs(float(velocity[TEST_AXIS])) < 0.02),
            4.0,
            "return to initial reference",
        )

        output_rate = self.reference_rate_hz()
        checks = {
            "initial_hold_error_norm_below_0_005_rad": (
                float(baseline["following_error_norm_rad"]) < 0.005),
            "reference_output_rate_above_200_hz": output_rate >= 200.0,
            "commanded_reference_motion_above_0_12_rad": reference_motion >= 0.12,
            "stopped_reference_drift_below_0_001_rad": reference_drift < 0.001,
            "stopped_tracking_error_below_0_015_rad": hold_error_max < 0.015,
            "disturbance_was_measurable": peak_disturbance_error > 0.001,
            "disturbance_recovered_below_0_010_rad": recovered_error < 0.010,
            "returned_to_initial_reference": (
                abs(float(returned["reference"][TEST_AXIS]) -
                    float(baseline_reference[TEST_AXIS])) < 0.004),
            "returned_actual_to_initial": (
                abs(float(returned_actual[TEST_AXIS]) -
                    float(baseline_reference[TEST_AXIS])) < 0.012),
        }
        return {
            "passed": bool(all(checks.values())),
            "profile": "EGM_POSITION_REFERENCE_EMULATION",
            "test_joint": JOINTS[TEST_AXIS],
            "checks": checks,
            "reference_output_rate_hz": output_rate,
            "initial_reference_rad": baseline_reference.tolist(),
            "initial_actual_rad": baseline_actual.tolist(),
            "initial_actual_velocity_rad_s": baseline_velocity.tolist(),
            "initial_following_error_norm_rad": float(
                baseline["following_error_norm_rad"]),
            "commanded_reference_motion_rad": reference_motion,
            "moved_actual_rad": moved_actual.tolist(),
            "hold_reference_drift_rad": reference_drift,
            "hold_actual_error_max_rad": hold_error_max,
            "applied_disturbance_nm": effort_request.effort,
            "applied_disturbance_duration_s": effort_request.duration.to_sec(),
            "peak_disturbance_error_rad": peak_disturbance_error,
            "recovered_error_rad": recovered_error,
            "recovered_actual_velocity_rad_s": recovered_velocity.tolist(),
            "returned_reference_rad": returned["reference"],
            "returned_actual_rad": returned_actual.tolist(),
            "returned_actual_velocity_rad_s": returned_velocity.tolist(),
        }


def main():
    args = parse_args()
    rospy.init_node("egm_position_profile_validator")
    try:
        result = EgmPositionProfileValidator(args).run()
    except Exception as exc:
        result = {"passed": False, "error": str(exc)}
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
