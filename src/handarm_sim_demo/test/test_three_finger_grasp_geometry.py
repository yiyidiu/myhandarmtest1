#!/usr/bin/env python3
"""Regression tests for the fixed-palm three-finger enclosure geometry."""

import math
import os
import sys
import unittest

import numpy as np


PACKAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_SRC = os.path.abspath(os.path.join(PACKAGE, ".."))
sys.path.insert(0, os.path.join(PACKAGE, "scripts"))

from grasp_geometry import HandGeometry, OBB, _rotation_axis_angle, transform


class ThreeFingerGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = HandGeometry(
            os.path.join(
                WORKSPACE_SRC,
                "abb120_moveit_config1",
                "config",
                "gazebo_handarm.urdf",
            ),
            os.path.join(PACKAGE, "config", "three_finger_grasp_geometry.yaml"),
        )
        cls.object_size = np.array([0.05, 0.06, 0.10])
        cls.T_world_object = transform(np.eye(3), [0.34, 0.18, 0.42])
        cls.table_z = 0.37

    def test_grasp_center_is_urdf_derived_and_tool0_is_not_the_center(self):
        summary = self.geometry.transform_summary()
        hand_center = np.asarray(summary["T_handbase_grasp_center"])[:3, 3]
        tool_center = np.asarray(summary["T_tool0_grasp_center"])[:3, 3]
        np.testing.assert_allclose(
            hand_center, [0.008705, 0.010090, 0.250669], atol=0.002
        )
        np.testing.assert_allclose(
            tool_center, [0.008506, 0.010090, 0.080676], atol=0.002
        )
        self.assertGreater(np.linalg.norm(tool_center), 0.075)

    def test_handbase_mesh_has_explicit_fail_closed_box_proxy(self):
        _, boxes = self.geometry._collision_proxies(
            ("handbase_link",), self.geometry.open_joints
        )
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].link, "handbase_link")
        np.testing.assert_allclose(
            boxes[0].half_extents, [0.069, 0.069, 0.0895]
        )

    def test_centered_top_down_has_three_distinct_contacts_and_table_clearance(self):
        candidate = self.geometry.make_candidate(
            self.T_world_object,
            self.object_size,
            self.table_z,
            "top_down",
            "object_pos_z",
            0.0,
        )
        result = candidate.enclosure
        self.assertTrue(result.valid, result.failure_reasons)
        self.assertEqual(set(result.contacts), {"f1", "f2", "f3"})
        self.assertTrue(result.object_inside_three_finger_envelope)
        self.assertGreaterEqual(result.table_clearance_m, 0.008)
        self.assertGreaterEqual(result.palm_clearance_m, 0.003)
        self.assertAlmostEqual(np.linalg.det(candidate.T_world_tool0[:3, :3]), 1.0)

    def test_candidate_tracks_object_yaw_instead_of_static_world_quaternion(self):
        relative_tool_rotations = []
        for yaw_deg in (0.0, 30.0, 60.0, 90.0):
            T_world_object = transform(
                _rotation_axis_angle([0.0, 0.0, 1.0], math.radians(yaw_deg)),
                [0.34, 0.18, 0.42],
            )
            candidate = self.geometry.make_candidate(
                T_world_object,
                self.object_size,
                self.table_z,
                "top_down",
                "object_pos_z",
                0.0,
            )
            self.assertTrue(candidate.enclosure.valid)
            relative_tool_rotations.append(
                T_world_object[:3, :3].T @ candidate.T_world_tool0[:3, :3]
            )
        for relative in relative_tool_rotations[1:]:
            np.testing.assert_allclose(relative, relative_tool_rotations[0], atol=1e-8)

    def test_legacy_f1_f2_biased_center_is_rejected_for_missing_f3(self):
        center = self.geometry.T_hand_grasp_center[:3, 3] + np.array(
            [-0.076, 0.0, 0.005]
        )
        result = self.geometry.evaluate_enclosure(
            OBB(center, np.diag([1.0, -1.0, -1.0]), 0.5 * self.object_size)
        )
        self.assertFalse(result.valid)
        self.assertIn("F3_NO_CONTACT", result.failure_reasons)
        self.assertEqual(set(result.contacts), {"f1", "f2"})

    def test_lift_corrected_offset_preloads_f3_before_opposing_pair(self):
        candidate = self.geometry.make_candidate(
            self.T_world_object,
            self.object_size,
            self.table_z,
            "top_oblique",
            "object_pos_z",
            268.0,
            [0.006, -0.009],
            0.0,
            -30.0,
            0.052,
        )
        result = candidate.enclosure
        self.assertTrue(result.valid, result.failure_reasons)
        fractions = {
            name: contact.closure_fraction
            for name, contact in result.contacts.items()
        }
        self.assertLessEqual(
            fractions["f3"], min(fractions["f1"], fractions["f2"])
        )
        self.assertGreaterEqual(result.table_clearance_m, 0.024)

    def test_deliberate_center_error_is_rejected(self):
        center = self.geometry.T_hand_grasp_center[:3, 3] + np.array(
            [0.010, 0.0, 0.005]
        )
        result = self.geometry.evaluate_enclosure(
            OBB(center, np.diag([1.0, -1.0, -1.0]), 0.5 * self.object_size)
        )
        self.assertFalse(result.valid)
        self.assertIn("F1_NO_CONTACT", result.failure_reasons)

    def test_table_clearance_is_a_hard_gate_not_a_score(self):
        grasp_center = self.geometry.T_hand_grasp_center[:3, 3]
        R_world_hand = np.diag([1.0, -1.0, -1.0])
        T_world_hand = transform(
            R_world_hand,
            self.T_world_object[:3, 3] - R_world_hand @ grasp_center,
        )
        result = self.geometry.evaluate_enclosure(
            OBB(
                grasp_center,
                np.diag([1.0, -1.0, -1.0]),
                0.5 * self.object_size,
            ),
            T_world_hand,
            self.table_z,
        )
        self.assertFalse(result.valid)
        self.assertIn("TABLE_COLLISION", result.failure_reasons)

    def test_side_candidate_matching_old_low_sweep_is_rejected(self):
        candidate = self.geometry.make_candidate(
            self.T_world_object,
            self.object_size,
            self.table_z,
            "side",
            "object_pos_x",
            0.0,
            side_height_m=0.020,
        )
        self.assertFalse(candidate.enclosure.valid)
        self.assertIn("TABLE_COLLISION", candidate.enclosure.failure_reasons)

    def test_invalid_reflection_never_enters_candidate_geometry(self):
        with self.assertRaises(ValueError):
            OBB(
                np.zeros(3),
                np.diag([-1.0, 1.0, 1.0]),
                np.array([0.025, 0.030, 0.050]),
            )


if __name__ == "__main__":
    unittest.main()
