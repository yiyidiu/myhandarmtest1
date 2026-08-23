#!/usr/bin/env python3
"""Regression tests for the doubled-size strict-top-down pose experiment."""

import copy
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
    evaluate_candidate_quality,
    validate_quality_gate_config,
)
from grasp_geometry import HandGeometry, transform
from grasp_size_pose_diagnostic import box_inertia_kg_m2


URDF = os.path.join(
    WORKSPACE_SRC, "abb120_moveit_config1", "config", "gazebo_handarm.urdf"
)
SCENE = os.path.join(PACKAGE, "config", "double_size_top_grasp_scene.yaml")
GEOMETRY = os.path.join(
    PACKAGE, "config", "double_size_top_grasp_geometry.yaml"
)
WORLD = os.path.join(PACKAGE, "worlds", "double_size_top_grasp.world")
LAUNCH = os.path.join(
    PACKAGE, "launch", "double_size_top_grasp_pose_demo.launch"
)
POSE_LAUNCH = os.path.join(
    PACKAGE, "launch", "three_finger_grasp_pose_demo.launch"
)
LEGACY_GEOMETRY = os.path.join(
    PACKAGE, "config", "three_finger_grasp_geometry.yaml"
)
BASELINE_SCENE = os.path.join(
    PACKAGE, "config", "physical_grasp_scene.yaml"
)
TABLE_Z = 0.37


class DoubleSizeTopGraspTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doubled_size = [0.10, 0.12, 0.20]
        cls.T_world_object = transform(
            np.eye(3), [0.34, 0.18, 0.47]
        )

    def _world_target_model(self):
        root = ET.parse(WORLD).getroot()
        return next(
            model
            for model in root.findall(".//model")
            if model.get("name") == "target_object"
        )

    def test_exact_doubled_dimensions_table_supported_z_and_consistent_world(self):
        with open(BASELINE_SCENE, encoding="utf-8") as stream:
            baseline = yaml.safe_load(stream)["objects"]["target"]
        with open(SCENE, encoding="utf-8") as stream:
            target = yaml.safe_load(stream)["objects"]["target"]

        self.assertEqual(
            target["size"], [2.0 * value for value in baseline["size"]]
        )
        self.assertEqual(target["size"], self.doubled_size)
        self.assertEqual(target["mass"], baseline["mass"])
        self.assertEqual(target["mass"], 0.10)
        self.assertEqual(target["pose"]["position"][:2], baseline["pose"]["position"][:2])
        self.assertEqual(target["pose"]["position"][:2], [0.34, 0.18])
        # Bottom remains on the table: z - half_height == table_top.
        self.assertAlmostEqual(
            target["pose"]["position"][2] - 0.5 * target["size"][2],
            TABLE_Z,
        )
        self.assertEqual(target["pose"]["position"][2], 0.47)

        model = self._world_target_model()
        pose = [float(value) for value in model.find("pose").text.split()[:3]]
        self.assertEqual(pose, [0.34, 0.18, 0.47])
        size = [
            float(value)
            for value in model.find("link/collision/geometry/box/size").text.split()
        ]
        self.assertEqual(size, self.doubled_size)
        self.assertAlmostEqual(float(model.findtext("link/inertial/mass")), 0.10)
        expected_inertia = box_inertia_kg_m2(0.10, self.doubled_size)
        for tag in ("ixx", "iyy", "izz"):
            self.assertAlmostEqual(
                float(model.findtext("link/inertial/inertia/{}".format(tag))),
                expected_inertia["{}_kg_m2".format(tag)],
                places=7,
            )

    def test_world_has_no_attachment_fixed_joint_or_lift_plugin(self):
        model = self._world_target_model()
        self.assertEqual(model.findall("joint"), [])
        plugins = [
            plugin.get("filename", "")
            for plugin in model.findall("link/sensor/plugin")
        ]
        self.assertEqual(plugins, ["libgazebo_ros_bumper.so"])
        model_text = ET.tostring(model, encoding="unicode")
        self.assertNotIn("attach", model_text.lower())
        self.assertNotIn("fixed_joint", model_text.lower())

    def test_launch_is_pose_only_and_points_to_new_files(self):
        root = ET.parse(LAUNCH).getroot()
        includes = root.findall("include")
        self.assertEqual(len(includes), 1)
        pose_demo = includes[0].get("file")
        self.assertEqual(
            pose_demo, "$(find handarm_sim_demo)/launch/three_finger_grasp_pose_demo.launch"
        )
        args = {arg.get("name"): arg.get("value") for arg in includes[0].findall("arg")}
        self.assertEqual(args["grasp_family"], "top_down")
        self.assertEqual(
            args["scene_config"],
            "$(find handarm_sim_demo)/config/double_size_top_grasp_scene.yaml",
        )
        self.assertEqual(
            args["geometry_config"],
            "$(find handarm_sim_demo)/config/double_size_top_grasp_geometry.yaml",
        )
        self.assertEqual(
            args["world_name"],
            "$(find handarm_sim_demo)/worlds/double_size_top_grasp.world",
        )
        # The wrapper owns no ROS node itself and delegates to the proven
        # pose-only planner launch; comments may use the words they negate.
        self.assertEqual(root.findall("node"), [])
        pose_root = ET.parse(POSE_LAUNCH).getroot()
        node_names = [
            node.get("pkg") + "/" + node.get("type") for node in pose_root.findall("node")
        ]
        self.assertEqual(
            node_names,
            ["handarm_sim_demo/grasp_pose_planner.py"],
        )
        with open(POSE_LAUNCH, encoding="utf-8") as stream:
            pose_text = stream.read().lower()
        for forbidden in ("contact_demo", "pick_place", "attach"):
            self.assertNotIn(forbidden, pose_text)

    def test_legacy_centered_object_gate_remains_unchanged(self):
        with open(LEGACY_GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        gates = validate_quality_gate_config(config)
        self.assertEqual(gates["quality_reference_mode"], "centered_object")
        self.assertEqual(
            gates["maximum_centered_object_center_axial_offset_m"], 0.025
        )
        geometry = HandGeometry(URDF, LEGACY_GEOMETRY)
        candidate = geometry.make_candidate(
            transform(np.eye(3), [0.34, 0.18, 0.42]),
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
        quality = evaluate_candidate_quality(candidate, config)
        self.assertFalse(quality.passed)
        self.assertIn("AXIAL_OFFSET_TOO_LARGE", quality.failures)

    def test_geometry_config_is_strict_top_down_without_oblique_fallback(self):
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        search = config["search"]
        self.assertEqual(search["expected_object_size_m"], self.doubled_size)
        self.assertEqual(search["center_offsets_m"], [0.0])
        self.assertEqual(search["top_object_center_in_grasp_frame_z_m"], 0.085)
        self.assertEqual(search["grasp_families"], ["top_down"])
        self.assertEqual(search["top_oblique_tilt_axial_pairs"], [])
        self.assertEqual(search["top_oblique_planar_offsets_m"], [[0.0, 0.0]])
        self.assertNotIn("-30", yaml.safe_dump(config))
        self.assertNotIn("0.052", yaml.safe_dump(config))

    def test_top_precision_band_mode_rejects_missing_or_invalid_bounds(self):
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            validate_quality_gate_config(config)["quality_reference_mode"],
            "top_precision_band",
        )

        missing_mode = copy.deepcopy(config)
        del missing_mode["quality_reference_mode"]
        with self.assertRaises(ValueError):
            validate_quality_gate_config(missing_mode)

        unknown_mode = copy.deepcopy(config)
        unknown_mode["quality_reference_mode"] = "optimizer_drift"
        with self.assertRaises(ValueError):
            validate_quality_gate_config(unknown_mode)

        missing_band = copy.deepcopy(config)
        del missing_band["top_precision_band"]
        with self.assertRaises(ValueError):
            validate_quality_gate_config(missing_band)

        invalid_band = copy.deepcopy(config)
        invalid_band["top_precision_band"] = {
            "minimum_contact_depth_below_top_m": 0.050,
            "maximum_contact_depth_below_top_m": 0.040,
        }
        with self.assertRaises(ValueError):
            validate_quality_gate_config(invalid_band)

    def test_real_urdf_axial_0085_has_quality_qualified_top_down_candidates(self):
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        geometry = HandGeometry(URDF, GEOMETRY)
        qualified = []
        for roll_deg in range(0, 360, 15):
            candidate = geometry.make_candidate(
                self.T_world_object,
                self.doubled_size,
                TABLE_Z,
                "top_down",
                "object_pos_z",
                float(roll_deg),
                (0.0, 0.0),
                0.0,
                0.0,
                0.085,
            )
            quality = evaluate_candidate_quality(candidate, config)
            if quality.passed:
                qualified.append((candidate, quality))
        self.assertTrue(qualified, "no axial-0.085 candidate passed quality")
        for candidate, quality in qualified:
            self.assertTrue(candidate.enclosure.valid)
            self.assertEqual(set(candidate.enclosure.contacts), {"f1", "f2", "f3"})
            faces = {
                (contact.face_axis, contact.face_sign)
                for contact in candidate.enclosure.contacts.values()
            }
            self.assertEqual(len(faces), 3)
            self.assertNotIn("AXIAL_OFFSET_TOO_LARGE", quality.failures)
            depths = quality.metrics["contact_depth_below_top_m"]
            for depth in depths.values():
                self.assertGreaterEqual(depth, 0.035)
                self.assertLessEqual(depth, 0.055)

    def test_axial_0035_does_not_qualify(self):
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        geometry = HandGeometry(URDF, GEOMETRY)
        for roll_deg in range(0, 360, 15):
            candidate = geometry.make_candidate(
                self.T_world_object,
                self.doubled_size,
                TABLE_Z,
                "top_down",
                "object_pos_z",
                float(roll_deg),
                (0.0, 0.0),
                0.0,
                0.0,
                0.035,
            )
            self.assertFalse(
                evaluate_candidate_quality(candidate, config).passed,
                "roll {} unexpectedly qualified at axial 0.035".format(roll_deg),
            )


if __name__ == "__main__":
    unittest.main()
