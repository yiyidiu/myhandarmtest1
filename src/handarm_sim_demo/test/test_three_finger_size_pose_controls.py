#!/usr/bin/env python3
"""ROS-free regressions for grasp-quality and the size-only control."""

import os
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import yaml


PACKAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_SRC = os.path.abspath(os.path.join(PACKAGE, ".."))
sys.path.insert(0, os.path.join(PACKAGE, "scripts"))

from grasp_candidate_quality import (
    evaluate_actual_lift_evidence,
    evaluate_candidate_quality,
    validate_planned_lift_vector,
)
from grasp_geometry import HandGeometry, transform
from grasp_size_pose_diagnostic import box_inertia_kg_m2, run_size_pose_diagnostic
from grasp_xy_ik_probe import validate_probes


URDF = os.path.join(
    WORKSPACE_SRC, "abb120_moveit_config1", "config", "gazebo_handarm.urdf"
)
GEOMETRY = os.path.join(PACKAGE, "config", "three_finger_grasp_geometry.yaml")
SIZE_GEOMETRY = os.path.join(
    PACKAGE, "config", "three_finger_size_control_geometry.yaml"
)
SIZE_SCENE = os.path.join(PACKAGE, "config", "three_finger_size_control_scene.yaml")
SIZE_WORLD = os.path.join(PACKAGE, "worlds", "handarm_three_finger_size_control.world")
PROBE_SCENE = os.path.join(
    PACKAGE, "config", "three_finger_large_xy_probe_01_scene.yaml"
)
PROBE_WORLD = os.path.join(
    PACKAGE, "worlds", "handarm_three_finger_large_xy_probe_01.world"
)


class ThreeFingerSizePoseControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = HandGeometry(URDF, GEOMETRY)
        with open(GEOMETRY, encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)
        cls.T = transform(np.eye(3), [0.34, 0.18, 0.42])

    def test_historical_oblique_edge_grip_is_rejected(self):
        candidate = self.geometry.make_candidate(
            self.T,
            [0.05, 0.06, 0.10],
            0.37,
            "top_oblique",
            "object_pos_z",
            268.0,
            (0.006, -0.009),
            0.0,
            -30.0,
            0.052,
        )
        self.assertTrue(candidate.enclosure.valid)
        quality = evaluate_candidate_quality(candidate, self.config)
        self.assertFalse(quality.passed)
        self.assertIn("AXIAL_OFFSET_TOO_LARGE", quality.failures)
        self.assertIn("CONTACT_TOO_CLOSE_TO_OBJECT_EDGE", quality.failures)

    def test_centered_top_down_passes_quality(self):
        candidate = self.geometry.make_candidate(
            self.T,
            [0.05, 0.06, 0.10],
            0.37,
            "top_down",
            "object_pos_z",
            90.0,
        )
        self.assertTrue(evaluate_candidate_quality(candidate, self.config).passed)

    def test_explicit_family_never_expands_to_other_families(self):
        candidates = self.geometry.coarse_geometry_candidates(
            self.T, [0.05, 0.06, 0.10], 0.37, "top_down"
        )
        self.assertEqual({item.family for item in candidates}, {"top_down"})

    def test_auto_expands_only_configured_concrete_families(self):
        size_geometry = HandGeometry(URDF, SIZE_GEOMETRY)
        T = transform(np.eye(3), [0.34, 0.18, 0.44])
        candidates = size_geometry.coarse_geometry_candidates(
            T, [0.07, 0.08, 0.14], 0.37, "auto"
        )
        self.assertTrue(candidates)
        self.assertEqual({item.family for item in candidates}, {"top_down"})
        with self.assertRaises(ValueError):
            size_geometry.coarse_geometry_candidates(
                T, [0.07, 0.08, 0.14], 0.37, "top_oblique"
            )

    def test_large_object_inertia_and_scene_world_match(self):
        expected = box_inertia_kg_m2(0.10, [0.07, 0.08, 0.14])
        self.assertAlmostEqual(expected["ixx_kg_m2"], 0.0002166666666666667)
        self.assertAlmostEqual(expected["iyy_kg_m2"], 0.0002041666666666667)
        self.assertAlmostEqual(expected["izz_kg_m2"], 0.00009416666666666667)
        with open(SIZE_SCENE, encoding="utf-8") as stream:
            target = yaml.safe_load(stream)["objects"]["target"]
        root = ET.parse(SIZE_WORLD).getroot()
        model = next(item for item in root.findall(".//model") if item.get("name") == "target_object")
        world_size = [
            float(value)
            for value in model.find("link/collision/geometry/box/size").text.split()
        ]
        self.assertEqual(target["size"], [0.07, 0.08, 0.14])
        self.assertEqual(world_size, target["size"])
        self.assertEqual(target["pose"]["position"][:2], [0.34, 0.18])

    def test_control_has_lift_request_margin_and_strict_gates(self):
        with open(SIZE_GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        runtime = config["runtime_acceptance"]
        self.assertGreater(
            runtime["lift_distance_along_opposite_approach_m"],
            runtime["minimum_object_lift_m"],
        )
        self.assertGreaterEqual(runtime["minimum_object_lift_m"], 0.060)
        self.assertGreaterEqual(runtime["physical_hold_duration_s"], 5.0)
        self.assertGreaterEqual(
            validate_planned_lift_vector([0.0, 0.0, 0.085], config)[
                "world_z_fraction"
            ],
            0.99,
        )
        with self.assertRaises(ValueError):
            validate_planned_lift_vector([0.05, 0.0, 0.03], config)

    def test_probe_01_changes_only_large_object_xy(self):
        with open(SIZE_SCENE, encoding="utf-8") as stream:
            baseline = yaml.safe_load(stream)["objects"]["target"]
        with open(PROBE_SCENE, encoding="utf-8") as stream:
            probe = yaml.safe_load(stream)["objects"]["target"]
        self.assertEqual(probe["size"], baseline["size"])
        self.assertEqual(probe["mass"], baseline["mass"])
        self.assertEqual(probe["pose"]["position"][2], baseline["pose"]["position"][2])
        self.assertEqual(probe["pose"]["position"][:2], [0.30, 0.0])
        root = ET.parse(PROBE_WORLD).getroot()
        model = next(item for item in root.findall(".//model") if item.get("name") == "target_object")
        self.assertEqual(
            [float(value) for value in model.find("pose").text.split()[:3]],
            [0.30, 0.0, 0.44],
        )

    def test_virtual_probe_table_bounds_are_fail_closed(self):
        self.assertEqual(
            validate_probes([[0.30, 0.10], [0.38, -0.10]]),
            [[0.30, 0.10], [0.38, -0.10]],
        )
        for invalid in ([], [[0.28, 0.0]], [[0.30, 0.42]], [[0.30, 0.0], [0.30, 0.0]]):
            with self.assertRaises(ValueError):
                validate_probes(invalid)

    def test_size_diagnostic_keeps_xy_constant_and_rejects_old_pose(self):
        record = run_size_pose_diagnostic(URDF, GEOMETRY)
        for row in record["rows"].values():
            self.assertEqual(row["xy_m"], [0.34, 0.18])
            rejected = row["historical_oblique_268_m30_52mm"]
            self.assertFalse(rejected["quality_passed"])
            self.assertIn("AXIAL_OFFSET_TOO_LARGE", rejected["quality_failures"])

    def test_historical_twenty_one_mm_slide_is_not_success(self):
        object_delta = np.array([-0.020251, -0.000721, 0.021665])
        with open(SIZE_GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        evidence = evaluate_actual_lift_evidence(object_delta, config)
        self.assertFalse(evidence["passed"])
        self.assertIn("OBJECT_WORLD_Z_LIFT_TOO_SMALL", evidence["failures"])
        self.assertIn("OBJECT_LIFT_NOT_WORLD_Z_DOMINANT", evidence["failures"])
        self.assertTrue(
            evaluate_actual_lift_evidence([0.0, 0.0, 0.060], config)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
