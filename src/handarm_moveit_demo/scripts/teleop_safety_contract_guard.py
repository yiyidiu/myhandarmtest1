#!/usr/bin/env python3
"""Fail-closed runtime contract for the public Gazebo teleoperation chain."""

import math
import time
import xml.etree.ElementTree as ET

import rosgraph
import rospy
from std_msgs.msg import Bool


EXPECTED_DISABLED_ADJACENT_PAIRS = {
    frozenset(pair) for pair in (
        ("base_link", "link_1"),
        ("link_1", "link_2"),
        ("link_2", "link_3"),
        ("link_3", "link_4"),
        ("link_4", "link_5"),
        ("link_5", "link_6"),
        ("link_6", "handbase_link"),
        ("handbase_link", "f1link1"),
        ("f1link1", "f1link2"),
        ("f1link2", "f1link3"),
        ("handbase_link", "f2link1"),
        ("f2link1", "f2link2"),
        ("handbase_link", "f3link1"),
        ("f3link1", "f3link2"),
        ("f3link2", "f3link3"),
    )
}

EXPECTED_STRUCTURAL_PROXIMITY_PAIRS = {
    frozenset(pair) for pair in (
        ("link_5", "handbase_link"),
        ("handbase_link", "f1link2"),
        ("handbase_link", "f2link2"),
        ("handbase_link", "f3link2"),
    )
}

EXPECTED_GAZEBO_SELF_COLLISION_LINKS = {
    "base_link", "link_1", "link_2", "link_3", "link_4", "link_5",
    "link_6", "handbase_link", "f1link1", "f1link2", "f1link3",
    "f2link1", "f2link2", "f3link1", "f3link2", "f3link3",
}


class SafetyContractGuard:
    def __init__(self):
        self.servo_namespace = str(rospy.get_param(
            "~servo_namespace", "/servo_server")).rstrip("/")
        self.startup_timeout_s = float(rospy.get_param(
            "~startup_timeout_s", 30.0))
        self.audit_period_s = float(rospy.get_param(
            "~audit_period_s", 0.50))
        self.minimum_collision_check_rate = float(rospy.get_param(
            "~minimum_collision_check_rate", 60.0))
        self.minimum_self_collision_distance = float(rospy.get_param(
            "~minimum_self_collision_distance_m", 0.01))
        self.minimum_joint_limit_margin = float(rospy.get_param(
            "~minimum_joint_limit_margin_rad", 0.08))
        self.strict_status_timeout_s = float(rospy.get_param(
            "~strict_status_timeout_s", 0.20))
        if (not math.isfinite(self.startup_timeout_s) or
                self.startup_timeout_s <= 0.0 or
                not math.isfinite(self.audit_period_s) or
                self.audit_period_s <= 0.0 or
                not math.isfinite(self.strict_status_timeout_s) or
                self.strict_status_timeout_s <= 0.0):
            raise ValueError("guard time parameters must be finite and positive")
        self.master = rosgraph.Master(rospy.get_name())
        self.status = rospy.Publisher(
            "/shared_teleop/safety_contract_ok", Bool,
            queue_size=1, latch=True)
        self.strict_safe = None
        self.strict_safe_wall_time = 0.0
        self.strict_ever_safe = False
        rospy.Subscriber(
            "/full_robot_self_collision_guard/safe", Bool,
            self.strict_status_callback, queue_size=1)

    def strict_status_callback(self, message):
        self.strict_safe = bool(message.data)
        self.strict_safe_wall_time = time.monotonic()
        if self.strict_safe:
            self.strict_ever_safe = True

    def parameter(self, suffix, missing, unsafe):
        name = "{}{}".format(self.servo_namespace, suffix)
        if not rospy.has_param(name):
            missing.append("missing parameter {}".format(name))
            return None
        try:
            return rospy.get_param(name)
        except Exception as exc:
            unsafe.append("cannot read {}: {}".format(name, exc))
            return None

    @staticmethod
    def finite_float(value, name, unsafe):
        try:
            converted = float(value)
        except (TypeError, ValueError):
            unsafe.append("{} is not numeric".format(name))
            return None
        if not math.isfinite(converted):
            unsafe.append("{} is not finite".format(name))
            return None
        return converted

    def audit_servo_parameters(self, missing, unsafe):
        enabled = self.parameter("/check_collisions", missing, unsafe)
        if enabled is not None and enabled is not True:
            unsafe.append("MoveIt Servo self-collision checking is not true")

        rate = self.finite_float(
            self.parameter("/collision_check_rate", missing, unsafe),
            "collision_check_rate", unsafe)
        if rate is not None and rate < self.minimum_collision_check_rate:
            unsafe.append("collision_check_rate {:.3f} < {:.3f}".format(
                rate, self.minimum_collision_check_rate))

        distance = self.finite_float(
            self.parameter(
                "/self_collision_proximity_threshold", missing, unsafe),
            "self_collision_proximity_threshold", unsafe)
        if (distance is not None and
                distance < self.minimum_self_collision_distance):
            unsafe.append(
                "self-collision distance {:.6f} < {:.6f} m".format(
                    distance, self.minimum_self_collision_distance))

        margin = self.finite_float(
            self.parameter("/joint_limit_margin", missing, unsafe),
            "joint_limit_margin", unsafe)
        if margin is not None and margin < self.minimum_joint_limit_margin:
            unsafe.append("joint_limit_margin {:.6f} < {:.6f} rad".format(
                margin, self.minimum_joint_limit_margin))

        lower = self.finite_float(
            self.parameter("/lower_singularity_threshold", missing, unsafe),
            "lower_singularity_threshold", unsafe)
        hard = self.finite_float(
            self.parameter(
                "/hard_stop_singularity_threshold", missing, unsafe),
            "hard_stop_singularity_threshold", unsafe)
        if lower is not None and not (0.0 < lower <= 17.0):
            unsafe.append("lower singularity threshold must be in (0, 17]")
        if hard is not None and not (0.0 < hard <= 30.0):
            unsafe.append("hard singularity threshold must be in (0, 30]")
        if lower is not None and hard is not None and hard <= lower:
            unsafe.append("hard singularity threshold must exceed lower threshold")

        output_topic = self.parameter("/command_out_topic", missing, unsafe)
        if (output_topic is not None and
                output_topic !=
                "/full_robot_self_collision_guard/raw_arm_velocity"):
            unsafe.append(
                "Servo output must pass through the strict MoveIt velocity gate")

        adapter_gate = rospy.get_param(
            "/moveit_servo_output_adapter/require_full_robot_safety", None)
        if adapter_gate is None:
            missing.append("Servo input adapter safety gate parameter is unavailable")
        elif adapter_gate is not True:
            unsafe.append(
                "Servo input adapter is not gated by full-robot safety")
        if rospy.has_param("/controller_gazebo_hand/type"):
            unsafe.append(
                "public hand controller bypasses the collision proxy")
        internal_type = rospy.get_param(
            "/controller_gazebo_hand_internal/type", None)
        if internal_type != "position_controllers/JointTrajectoryController":
            missing.append(
                "private hand trajectory controller is unavailable")

    @staticmethod
    def audit_semantic_model(missing, unsafe):
        if not rospy.has_param("/robot_description_semantic"):
            missing.append("missing /robot_description_semantic")
            return
        try:
            root = ET.fromstring(rospy.get_param(
                "/robot_description_semantic"))
        except Exception as exc:
            unsafe.append("invalid robot_description_semantic: {}".format(exc))
            return
        entries = root.findall("disable_collisions")
        adjacent_pairs = set()
        structural_pairs = set()
        for entry in entries:
            reason = entry.attrib.get("reason")
            pair = frozenset((
                entry.attrib.get("link1", ""),
                entry.attrib.get("link2", "")))
            if reason == "Adjacent":
                adjacent_pairs.add(pair)
            elif reason == "StructuralAdjacent":
                structural_pairs.add(pair)
            else:
                unsafe.append(
                    "unapproved collision pair is disabled: {} ({})".format(
                        sorted(pair), reason))
        if adjacent_pairs != EXPECTED_DISABLED_ADJACENT_PAIRS:
            missing_pairs = EXPECTED_DISABLED_ADJACENT_PAIRS - adjacent_pairs
            extra_pairs = adjacent_pairs - EXPECTED_DISABLED_ADJACENT_PAIRS
            unsafe.append(
                "SRDF adjacent-pair contract mismatch; missing={} extra={}".format(
                    sorted(map(sorted, missing_pairs)),
                    sorted(map(sorted, extra_pairs))))
        if structural_pairs != EXPECTED_STRUCTURAL_PROXIMITY_PAIRS:
            missing_pairs = EXPECTED_STRUCTURAL_PROXIMITY_PAIRS - structural_pairs
            extra_pairs = structural_pairs - EXPECTED_STRUCTURAL_PROXIMITY_PAIRS
            unsafe.append(
                "SRDF structural-pair contract mismatch; missing={} extra={}".format(
                    sorted(map(sorted, missing_pairs)),
                    sorted(map(sorted, extra_pairs))))

    @staticmethod
    def audit_collision_geometry(missing, unsafe):
        if not rospy.has_param("/robot_description"):
            missing.append("missing /robot_description")
            return
        try:
            root = ET.fromstring(rospy.get_param("/robot_description"))
        except Exception as exc:
            unsafe.append("invalid robot_description: {}".format(exc))
            return
        handbase = next((link for link in root.findall("link")
                         if link.attrib.get("name") == "handbase_link"), None)
        if handbase is None:
            unsafe.append("handbase_link is missing from robot_description")
            return
        meshes = [
            geometry.find("mesh").attrib.get("filename", "")
            for collision in handbase.findall("collision")
            for geometry in collision.findall("geometry")
            if geometry.find("mesh") is not None
        ]
        expected_suffix = "/handbase_link_collision_8mm.STL"
        if len(meshes) != 1 or not meshes[0].endswith(expected_suffix):
            unsafe.append(
                "hand base must use the validated cut-out collision mesh")
        self_collision_links = {
            element.attrib.get("reference")
            for element in root.findall("gazebo")
            if (element.find("selfCollide") is not None and
                str(element.find("selfCollide").text).strip().lower()
                in ("true", "1"))
        }
        if not EXPECTED_GAZEBO_SELF_COLLISION_LINKS.issubset(
                self_collision_links):
            unsafe.append(
                "Gazebo selfCollide is missing for links {}".format(sorted(
                    EXPECTED_GAZEBO_SELF_COLLISION_LINKS -
                    self_collision_links)))

    def audit_ros_interfaces(self, missing, unsafe):
        try:
            publishers, _, services = self.master.getSystemState()
        except Exception as exc:
            missing.append("cannot query ROS master: {}".format(exc))
            return
        publisher_map = dict(publishers)
        service_map = dict(services)
        scale_topic = "{}/internal/collision_velocity_scale".format(
            self.servo_namespace)
        if self.servo_namespace not in publisher_map.get(scale_topic, []):
            missing.append(
                "Servo collision monitor is not publishing {}".format(
                    scale_topic))
        if "/check_state_validity" not in service_map:
            missing.append("MoveIt /check_state_validity service is unavailable")
        strict_topic = "/full_robot_self_collision_guard/safe"
        strict_node = "/full_robot_self_collision_guard"
        if strict_node not in publisher_map.get(strict_topic, []):
            missing.append("strict full-robot collision status publisher is unavailable")
        strict_service = (
            "/full_robot_self_collision_guard/check_state_validity")
        if strict_node not in service_map.get(strict_service, []):
            missing.append("strict full-robot candidate collision service is unavailable")
        hand_interlock_service = (
            "/full_robot_self_collision_guard/set_hand_motion_active")
        if strict_node not in service_map.get(hand_interlock_service, []):
            missing.append("arm/hand motion interlock service is unavailable")
        hand_status_topic = (
            "/controller_gazebo_hand/follow_joint_trajectory/status")
        if "/safe_hand_trajectory_proxy" not in publisher_map.get(
                hand_status_topic, []):
            missing.append("collision-checked public hand action is unavailable")
        elif set(publisher_map.get(hand_status_topic, [])) != {
                "/safe_hand_trajectory_proxy"}:
            unsafe.append("an unapproved node publishes the public hand action status")

        # ROS transport is not an authenticated safety bus, so the runtime
        # contract also enforces a single-writer topology. Any node attempting
        # to bypass either safety proxy is detected and shuts down this launch.
        twist_topic = "{}/delta_twist_cmds".format(self.servo_namespace)
        twist_publishers = set(publisher_map.get(twist_topic, []))
        expected_twist_publishers = {"/moveit_servo_output_adapter"}
        if not twist_publishers:
            missing.append("safety-gated Servo Twist publisher is unavailable")
        elif twist_publishers != expected_twist_publishers:
            unsafe.append(
                "Servo Twist has unapproved publishers: {}".format(sorted(
                    twist_publishers - expected_twist_publishers)))
        joint_topic = "{}/delta_joint_cmds".format(self.servo_namespace)
        if publisher_map.get(joint_topic, []):
            unsafe.append(
                "direct Servo JointJog publishers are forbidden in public teleop")

        raw_arm_topic = (
            "/full_robot_self_collision_guard/raw_arm_velocity")
        raw_arm_publishers = set(publisher_map.get(raw_arm_topic, []))
        if raw_arm_publishers != {self.servo_namespace}:
            if not raw_arm_publishers:
                missing.append("MoveIt Servo is not connected to the strict velocity gate")
            else:
                unsafe.append(
                    "strict velocity-gate input has unapproved publishers: {}".format(
                        sorted(raw_arm_publishers - {self.servo_namespace})))
        controller_topic = "/abbarm_velocity_controller/command"
        controller_publishers = set(publisher_map.get(controller_topic, []))
        if controller_publishers != {strict_node}:
            if not controller_publishers:
                missing.append("strict guard is not connected to the arm controller")
            else:
                unsafe.append(
                    "arm controller has unapproved publishers: {}".format(
                        sorted(controller_publishers - {strict_node})))

        internal_goal_topic = (
            "/controller_gazebo_hand_internal/follow_joint_trajectory/goal")
        internal_goal_publishers = set(
            publisher_map.get(internal_goal_topic, []))
        expected_internal_goal_publishers = {"/safe_hand_trajectory_proxy"}
        if not internal_goal_publishers:
            missing.append("safe hand proxy is not connected to the private action")
        elif internal_goal_publishers != expected_internal_goal_publishers:
            unsafe.append(
                "private hand action has unapproved goal publishers: {}".format(
                    sorted(internal_goal_publishers -
                           expected_internal_goal_publishers)))
        internal_command_topic = "/controller_gazebo_hand_internal/command"
        if publisher_map.get(internal_command_topic, []):
            unsafe.append(
                "direct private hand command publishers are forbidden")
        age = time.monotonic() - self.strict_safe_wall_time
        if self.strict_safe is None or age > self.strict_status_timeout_s:
            missing.append("strict full-robot collision status is stale")
        elif not self.strict_safe:
            message = "strict full-robot collision guard reports unsafe"
            if self.strict_ever_safe:
                unsafe.append(message)
            else:
                missing.append(message)

    def audit(self):
        missing = []
        unsafe = []
        self.audit_servo_parameters(missing, unsafe)
        self.audit_semantic_model(missing, unsafe)
        self.audit_collision_geometry(missing, unsafe)
        self.audit_ros_interfaces(missing, unsafe)
        return missing, unsafe

    def fail(self, reasons):
        self.status.publish(Bool(data=False))
        message = "FULL-ROBOT SELF-COLLISION SAFETY CONTRACT FAILED: {}".format(
            "; ".join(reasons))
        rospy.logfatal(message)
        return message

    def run(self):
        deadline = time.monotonic() + self.startup_timeout_s
        while not rospy.is_shutdown():
            missing, unsafe = self.audit()
            if unsafe:
                raise RuntimeError(self.fail(unsafe))
            if not missing:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(self.fail(missing))
            rospy.logwarn_throttle(
                2.0, "Waiting for self-collision safety contract: %s",
                "; ".join(missing))
            time.sleep(min(self.audit_period_s, 0.25))
        if rospy.is_shutdown():
            return
        self.status.publish(Bool(data=True))
        rospy.logwarn(
            "Full-robot collision contract ACTIVE: all non-adjacent "
            "arm-arm, hand-hand and hand-arm pairs are checked")
        while not rospy.is_shutdown():
            time.sleep(self.audit_period_s)
            # roslaunch stops sibling nodes in parallel.  Do not turn that
            # orderly teardown into a false safety failure merely because a
            # sibling disappeared while this process was sleeping.
            if rospy.is_shutdown():
                return
            missing, unsafe = self.audit()
            if rospy.is_shutdown():
                return
            if missing or unsafe:
                raise RuntimeError(self.fail(unsafe + missing))


def main():
    rospy.init_node("teleop_safety_contract_guard")
    try:
        SafetyContractGuard().run()
    except Exception as exc:
        if not rospy.is_shutdown():
            rospy.logfatal("Teleoperation safety guard terminating: %s", exc)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
