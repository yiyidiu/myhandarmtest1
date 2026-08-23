#!/usr/bin/env python3
"""ROS-free decision regressions for physical three-finger task evidence."""

import os
import sys
import unittest

import numpy as np
import rospy
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


PACKAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PACKAGE, "scripts"))

from three_finger_grasp_demo import (
    classify_contacts,
    has_target_table_support,
)
from three_finger_pick_place_demo import (
    ThreeFingerPickPlaceDemo,
    enforce_minimum_trajectory_duration,
    grasp_center_position_from_tool,
    opposite_approach_lift_vector,
    pivoted_grasp_center_lift_target,
    trajectory_duration_s,
)
from grasp_pose_planner import calibrated_roll_samples


class FakeContact:
    def __init__(self, first, second):
        self.collision1_name = first
        self.collision2_name = second


class ThreeFingerPickPlaceLogicTest(unittest.TestCase):
    def test_exact_contact_calibration_roll_is_always_generated(self):
        self.assertEqual(calibrated_roll_samples([268.0, 268.0], 2.0), [268.0])

    def test_oblique_lift_retracts_opposite_approach_and_gains_height(self):
        matrix = np.eye(4)
        matrix[:3, 2] = [0.5, 0.0, -(3.0 ** 0.5) / 2.0]
        vector = opposite_approach_lift_vector(matrix, 0.045)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 0.045)
        self.assertAlmostEqual(vector[0], -0.0225)
        self.assertGreater(vector[2], 0.038)

    def test_pivoted_lift_keeps_grasp_center_on_requested_translation(self):
        T_world_tool = np.eye(4)
        T_tool_hand = np.eye(4)
        T_tool_hand[:3, 3] = [0.01, -0.02, 0.08]
        T_hand_grasp_center = np.eye(4)
        T_hand_grasp_center[:3, 3] = [0.02, 0.01, 0.17]
        lift = np.array([0.0, 0.0, 0.085])
        target = pivoted_grasp_center_lift_target(
            T_world_tool,
            T_tool_hand,
            T_hand_grasp_center,
            lift,
            -10.0,
        )
        before = grasp_center_position_from_tool(
            T_world_tool, T_tool_hand, T_hand_grasp_center
        )
        after = grasp_center_position_from_tool(
            target, T_tool_hand, T_hand_grasp_center
        )
        np.testing.assert_allclose(after - before, lift, atol=1.0e-12)
        # Tool0 must compensate for the remote-center rotation rather than
        # merely moving straight up.
        self.assertGreater(abs(target[0, 3]), 0.001)
        self.assertFalse(np.allclose(target[:3, :3], np.eye(3)))

    def test_pivoted_lift_rejects_nonfinite_or_excessive_tilt(self):
        identity = np.eye(4)
        with self.assertRaises(ValueError):
            pivoted_grasp_center_lift_target(
                identity, identity, identity, [0.0, 0.0, np.nan], -10.0
            )
        with self.assertRaises(ValueError):
            pivoted_grasp_center_lift_target(
                identity, identity, identity, [0.0, 0.0, 0.085], -50.0
            )

    def test_three_distinct_families_are_not_faked_by_two_links(self):
        states = [
            FakeContact(
                "target_object::object_link::collision",
                "robot::f1link2::collision",
            ),
            FakeContact(
                "target_object::object_link::collision",
                "robot::f1link3::collision",
            ),
            FakeContact(
                "target_object::object_link::collision",
                "robot::f2link2::collision",
            ),
        ]
        families, _, unexpected = classify_contacts(states, "target_object")
        self.assertEqual(families, {"f1", "f2"})
        self.assertFalse(unexpected)

    def test_target_table_support_is_independent_of_finger_contact(self):
        states = [
            FakeContact(
                "work_table::table_link::collision",
                "target_object::object_link::collision",
            )
        ]
        self.assertTrue(has_target_table_support(states, "target_object"))
        families, _, unexpected = classify_contacts(states, "target_object")
        self.assertFalse(families)
        self.assertFalse(unexpected)

    def test_palm_target_contact_is_unexpected(self):
        states = [
            FakeContact(
                "target_object::object_link::collision",
                "robot::handbase_link::collision",
            )
        ]
        families, _, unexpected = classify_contacts(states, "target_object")
        self.assertFalse(families)
        self.assertTrue(unexpected)

    def test_duration_stretch_scales_dynamics_consistently(self):
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ["joint_1"]
        first = JointTrajectoryPoint()
        first.positions = [0.0]
        first.velocities = [2.0]
        first.accelerations = [4.0]
        first.time_from_start = rospy.Duration(1.0)
        second = JointTrajectoryPoint()
        second.positions = [1.0]
        second.velocities = [2.0]
        second.accelerations = [4.0]
        second.time_from_start = rospy.Duration(2.0)
        trajectory.joint_trajectory.points = [first, second]
        enforce_minimum_trajectory_duration(trajectory, 8.0)
        self.assertAlmostEqual(trajectory_duration_s(trajectory), 8.0)
        self.assertAlmostEqual(trajectory.joint_trajectory.points[-1].velocities[0], 0.5)
        self.assertAlmostEqual(trajectory.joint_trajectory.points[-1].accelerations[0], 0.25)

    def test_release_opens_then_retreats_before_requiring_contact_clear(self):
        source_path = os.path.join(
            PACKAGE, "scripts", "three_finger_pick_place_demo.py"
        )
        with open(source_path, encoding="utf-8") as stream:
            run_source = stream.read().split("def run_pick_place(self):", 1)[1]
        release = run_source.index('states.append("RELEASE_ON_SUPPORTED_TABLE")')
        plan = run_source.index('states.append("PLAN_OPEN_HAND_RETREAT")')
        retreat = run_source.index('states.append("RETREAT_OPEN_HAND")')
        clear = run_source.index("self.wait_for_finger_contact_clear()")
        restore = run_source.index(
            'states.append("RESTORE_EXACT_TARGET_AFTER_CLEAR_RETREAT")'
        )
        self.assertLess(release, plan)
        self.assertLess(plan, retreat)
        self.assertLess(retreat, clear)
        self.assertLess(clear, restore)


if __name__ == "__main__":
    unittest.main()
