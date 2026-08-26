#!/usr/bin/env python3
"""Fail-closed continuous HaMeR-to-three-finger Gazebo command adapter."""

import json
import math
import threading
import time

import numpy as np
import rospy
import yaml
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from handarm_moveit_demo.finger_retargeting import ThreeFingerRetargeter
from handarm_moveit_demo.msg import HamerHandPose


DEFAULT_JOINTS = ["f1j1", "f1j2", "f2j1", "f3j2"]


class ThreeFingerRetargetingNode:
    def __init__(self):
        shared = rospy.get_param("/shared_teleop", {})
        config = shared.get("finger_retargeting", {})
        topics = shared.get("topics", {})
        self.joint_names = list(config.get("joint_names", DEFAULT_JOINTS))
        if self.joint_names != DEFAULT_JOINTS:
            raise ValueError("teleoperation hand joint order must be {}".format(DEFAULT_JOINTS))

        hand_config_path = rospy.get_param("~hand_config")
        with open(hand_config_path, "r", encoding="utf-8") as stream:
            hand_config = yaml.safe_load(stream)
        if hand_config.get("joint_names") != self.joint_names:
            raise ValueError("hand command config joint order does not match teleoperation")
        limits = hand_config.get("joint_limits", {})
        lower = [limits[name][0] for name in self.joint_names]
        upper = [limits[name][1] for name in self.joint_names]
        open_target = hand_config["commands"]["OPEN"]["positions"]
        close_target = hand_config["commands"]["CLOSE"]["positions"]
        self.retargeter = ThreeFingerRetargeter(
            open_target,
            close_target,
            lower,
            upper,
            config.get("source_mixing_matrix", [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.3, 0.2],
            ]),
            config,
        )
        self.state_timeout_s = float(config.get("joint_state_timeout_s", 0.25))
        self.input_timeout_s = float(config.get("input_timeout_s", 0.45))
        if self.state_timeout_s <= 0.0 or self.input_timeout_s <= 0.0:
            raise ValueError("finger state/input timeouts must be positive")
        self.command_topic = rospy.get_param(
            "~command_topic", "/controller_gazebo_hand/command"
        )
        self.diagnostic_topic = topics.get(
            "finger_diagnostics", "/shared_teleop/finger_diagnostics"
        )
        self.publisher = rospy.Publisher(
            self.command_topic, JointTrajectory, queue_size=1
        )
        self.diagnostics = rospy.Publisher(
            self.diagnostic_topic, String, queue_size=10
        )
        self.lock = threading.Lock()
        self.retarget_lock = threading.Lock()
        self.positions = None
        self.joint_state_monotonic = -float("inf")
        # Invalid heartbeats prove that the process is alive, but they must
        # never keep an old C token alive.  This clock advances only after a
        # structurally valid finger observation passes the pure controller.
        self.last_valid_finger_monotonic = time.monotonic()
        self.watchdog_latched = False
        self.last_command_target = None
        self.command_count = 0
        rospy.Subscriber(
            "/joint_states", JointState, self.joint_state_callback, queue_size=20
        )
        rospy.Subscriber(
            topics.get("hamer_pose", "/shared_teleop/hamer_pose"),
            HamerHandPose,
            self.hamer_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(0.02), self.watchdog_callback)
        rospy.on_shutdown(self.shutdown)
        rospy.logwarn(
            "Continuous three-finger retargeting armed: %s -> %s; "
            "an open, steady hand and a fresh camera C token are mandatory",
            topics.get("hamer_pose", "/shared_teleop/hamer_pose"),
            self.command_topic,
        )

    def shutdown(self):
        self.timer.shutdown()
        with self.retarget_lock:
            self.retargeter.block_active_reference()
        self.publish_hold("NODE_SHUTDOWN")

    def joint_state_callback(self, message):
        values = dict(zip(message.name, message.position))
        if not all(name in values for name in self.joint_names):
            return
        positions = np.asarray([values[name] for name in self.joint_names], dtype=float)
        if not np.all(np.isfinite(positions)):
            return
        with self.lock:
            self.positions = positions
            self.joint_state_monotonic = time.monotonic()

    def current_joints(self):
        with self.lock:
            if self.positions is None:
                return None
            if time.monotonic() - self.joint_state_monotonic > self.state_timeout_s:
                return None
            return self.positions.copy()

    def publish_trajectory(self, positions, duration_s=None):
        target = np.asarray(positions, dtype=float)
        if target.shape != (4,) or not np.all(np.isfinite(target)):
            raise ValueError("finger command target must be a finite four-vector")
        message = JointTrajectory()
        message.header.stamp = rospy.Time.now()
        message.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = target.tolist()
        point.time_from_start = rospy.Duration(
            self.retargeter.command_duration_s
            if duration_s is None else float(duration_s)
        )
        message.points = [point]
        self.publisher.publish(message)
        self.last_command_target = target.copy()
        self.command_count += 1

    def publish_hold(self, reason):
        current = self.current_joints()
        if current is not None:
            self.publish_trajectory(current, duration_s=0.05)
        self.publish_diagnostic(
            status=reason,
            calibrated=False,
            hold_required=True,
            actual=current,
        )

    def publish_diagnostic(self, **values):
        defaults = {
            "stamp_ros": rospy.Time.now().to_sec(),
            "status": "UNKNOWN",
            "calibrated": self.retargeter.calibrated,
            "reference_token": self.retargeter.active_token,
            "blocked_reference_tokens": sorted(self.retargeter.blocked_tokens),
            "hold_required": False,
            "human_flexion_raw": None,
            "human_flexion_filtered": None,
            "normalized_robot_closure": None,
            "desired_joint_target_rad": None,
            "command_joint_target_rad": (
                None
                if self.last_command_target is None
                else self.last_command_target.tolist()
            ),
            "actual_joint_position_rad": None,
            "joint_names": list(self.joint_names),
            "command_count": self.command_count,
        }
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            defaults[key] = value
        self.diagnostics.publish(
            String(data=json.dumps(defaults, separators=(",", ":")))
        )

    def hamer_callback(self, message):
        with self.lock:
            callback_monotonic = time.monotonic()
        current = self.current_joints()
        if current is None:
            self.publish_diagnostic(
                status="FRESH_HAND_JOINT_STATE_UNAVAILABLE",
                calibrated=False,
                hold_required=True,
            )
            return
        if not message.control_gate_present:
            with self.retarget_lock:
                self.retargeter.block_active_reference()
            self.publish_hold("CAMERA_C_GATE_ABSENT")
            return
        if not message.control_enabled or not message.control_reference_token:
            with self.retarget_lock:
                self.retargeter.block_active_reference()
            self.publish_hold("WAITING_FOR_NEW_CAMERA_C")
            return
        if not message.valid:
            self.publish_hold(message.invalid_reason or "HAND_POSE_INVALID")
            return
        if not message.finger_tracking_present or not message.finger_tracking_valid:
            self.publish_hold(
                message.finger_invalid_reason or "FINGER_OBSERVATION_INVALID"
            )
            return
        try:
            with self.retarget_lock:
                result = self.retargeter.update(
                    message.source_timestamp,
                    message.control_reference_token,
                    list(message.finger_flexion),
                    message.finger_tracking_confidence,
                    current,
                )
        except Exception as exc:
            with self.retarget_lock:
                self.retargeter.block_active_reference()
            self.publish_hold("FINGER_RETARGETING_EXCEPTION:{}".format(exc))
            rospy.logerr_throttle(1.0, "Finger retargeting locked: %s", exc)
            return
        if result.command_target is not None:
            self.publish_trajectory(result.command_target)
        elif result.hold_required:
            self.publish_trajectory(current, duration_s=0.05)
        rejected_statuses = {
            "INVALID_FINGER_INPUT",
            "LOW_FINGER_CONFIDENCE",
            "FINGER_INNOVATION_REJECTED_PENDING_CONFIRMATION",
            "NON_MONOTONIC_FINGER_TIMESTAMP",
            "MEASURED_HAND_JOINT_OUT_OF_BOUNDS",
            "BLOCKED_REFERENCE_REQUIRES_NEW_C",
        }
        if result.status not in rejected_statuses:
            with self.lock:
                self.last_valid_finger_monotonic = callback_monotonic
                self.watchdog_latched = False
        self.publish_diagnostic(
            status=result.status,
            calibrated=result.calibrated,
            reference_token=result.reference_token,
            hold_required=result.hold_required,
            human_flexion_raw=result.human_flexion_raw,
            human_flexion_filtered=result.human_flexion_filtered,
            normalized_robot_closure=result.normalized_robot_closure,
            desired_joint_target_rad=result.desired_target,
            command_joint_target_rad=result.command_target,
            actual_joint_position_rad=current,
            source_timestamp=message.source_timestamp,
            source_capture_sequence=message.source_capture_sequence,
            finger_tracking_confidence=message.finger_tracking_confidence,
        )
        if result.status not in ("TRACKING", "OPEN_BASELINE"):
            rospy.logwarn_throttle(
                1.0, "Three-finger retargeting hold/status: %s", result.status
            )

    def watchdog_callback(self, _event):
        with self.lock:
            age = time.monotonic() - self.last_valid_finger_monotonic
            latched = self.watchdog_latched
            if age > self.input_timeout_s and not latched:
                self.watchdog_latched = True
        if age <= self.input_timeout_s or latched:
            return
        with self.retarget_lock:
            self.retargeter.block_active_reference()
        self.publish_hold("FINGER_INPUT_TIMEOUT_REQUIRES_NEW_C")
        rospy.logerr(
            "No finger input for %.3f s; hand held and old C token blocked", age
        )


def main():
    rospy.init_node("three_finger_retargeting")
    try:
        ThreeFingerRetargetingNode()
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("Three-finger retargeting failed: %s", exc)
        raise SystemExit(6)


if __name__ == "__main__":
    main()
