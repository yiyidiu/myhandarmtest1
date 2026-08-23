#!/usr/bin/env python3
"""Regression tests for the reachable strict-top-down plan-only experiment."""

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
    _object_vertical_reference,
    evaluate_candidate_quality,
    validate_quality_gate_config,
)
from grasp_geometry import HandGeometry, transform
from grasp_size_pose_diagnostic import box_inertia_kg_m2


URDF = os.path.join(
    WORKSPACE_SRC, "abb120_moveit_config1", "config", "gazebo_handarm.urdf"
)
SCENE = os.path.join(PACKAGE, "config", "reachable_top_grasp_scene.yaml")
GEOMETRY = os.path.join(PACKAGE, "config", "reachable_top_grasp_geometry.yaml")
WORLD = os.path.join(PACKAGE, "worlds", "reachable_top_grasp.world")
POSITION_PID = os.path.join(PACKAGE, "config", "gazebo_hand_position_pid.yaml")
LAUNCH = os.path.join(PACKAGE, "launch", "reachable_top_grasp_pose_demo.launch")
PICK_PLACE_LAUNCH = os.path.join(
    PACKAGE, "launch", "reachable_top_grasp_pick_place_demo.launch"
)
POSE_LAUNCH = os.path.join(
    PACKAGE, "launch", "three_finger_grasp_pose_demo.launch"
)
LEGACY_GEOMETRY = os.path.join(
    PACKAGE, "config", "three_finger_grasp_geometry.yaml"
)
DOUBLE_SIZE_SCENE = os.path.join(
    PACKAGE, "config", "double_size_top_grasp_scene.yaml"
)
DOUBLE_SIZE_GEOMETRY = os.path.join(
    PACKAGE, "config", "double_size_top_grasp_geometry.yaml"
)
TABLE_Z = 0.21


class ReachableTopGraspTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Codex TRAC-IK read-only probe inputs: target size/mass/pose, table
        # top and strict top_down axial offset.
        cls.object_size = [0.08, 0.09, 0.16]
        cls.T_world_object = transform(np.eye(3), [0.30, 0.10, 0.29])

    def _world_target_model(self):
        root = ET.parse(WORLD).getroot()
        return next(
            model
            for model in root.findall(".//model")
            if model.get("name") == "target_object"
        )

    def test_exact_reachable_dimensions_pose_mass_and_consistent_world(self):
        with open(SCENE, encoding="utf-8") as stream:
            scene = yaml.safe_load(stream)
        target = scene["objects"]["target"]
        table = scene["objects"]["table"]

        # Exact size and mass gate.
        self.assertEqual(target["size"], self.object_size)
        self.assertEqual(target["size"], [0.08, 0.09, 0.16])
        self.assertEqual(target["mass"], 0.10)
        # Exact object centre and strict zero RPY.
        self.assertEqual(
            target["pose"]["position"], [0.30, 0.10, 0.29]
        )
        self.assertEqual(target["pose"]["orientation_rpy"], [0.0, 0.0, 0.0])
        # Table top at z=0.21 and object bottom resting on it:
        # z - half_height == table_top.
        table_top = (
            table["pose"]["position"][2] + 0.5 * table["size"][2]
        )
        self.assertAlmostEqual(table_top, TABLE_Z)
        self.assertAlmostEqual(
            target["pose"]["position"][2] - 0.5 * target["size"][2],
            TABLE_Z,
        )

        model = self._world_target_model()
        pose = [float(value) for value in model.find("pose").text.split()[:3]]
        self.assertEqual(pose, [0.30, 0.10, 0.29])
        size = [
            float(value)
            for value in model.find("link/collision/geometry/box/size").text.split()
        ]
        self.assertEqual(size, self.object_size)
        self.assertAlmostEqual(float(model.findtext("link/inertial/mass")), 0.10)
        # Solid-box inertia consistent with the scene size and mass.
        expected_inertia = box_inertia_kg_m2(0.10, self.object_size)
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

    def test_joint_2_position_pid_removes_observed_gravity_error_without_integral(self):
        with open(POSITION_PID, encoding="utf-8") as stream:
            gains = yaml.safe_load(stream)
        self.assertEqual(gains["joint_2"]["p"], 700.0)
        self.assertEqual(gains["joint_2"]["d"], 10.0)
        self.assertEqual(gains["joint_2"]["i"], 0.0)
        self.assertEqual(gains["joint_2"]["i_clamp"], 0.0)

    def test_launch_is_pose_only_and_points_to_new_files(self):
        root = ET.parse(LAUNCH).getroot()
        includes = root.findall("include")
        self.assertEqual(len(includes), 1)
        pose_demo = includes[0].get("file")
        self.assertEqual(
            pose_demo,
            "$(find handarm_sim_demo)/launch/three_finger_grasp_pose_demo.launch",
        )
        args = {
            arg.get("name"): arg.get("value")
            for arg in includes[0].findall("arg")
        }
        self.assertEqual(args["grasp_family"], "top_down")
        self.assertEqual(
            args["scene_config"],
            "$(find handarm_sim_demo)/config/reachable_top_grasp_scene.yaml",
        )
        self.assertEqual(
            args["geometry_config"],
            "$(find handarm_sim_demo)/config/reachable_top_grasp_geometry.yaml",
        )
        self.assertEqual(
            args["world_name"],
            "$(find handarm_sim_demo)/worlds/reachable_top_grasp.world",
        )
        # The wrapper owns no ROS node itself; comments may use the words
        # they negate.
        self.assertEqual(root.findall("node"), [])
        pose_root = ET.parse(POSE_LAUNCH).getroot()
        node_names = [
            node.get("pkg") + "/" + node.get("type")
            for node in pose_root.findall("node")
        ]
        self.assertEqual(node_names, ["handarm_sim_demo/grasp_pose_planner.py"])
        with open(POSE_LAUNCH, encoding="utf-8") as stream:
            pose_text = stream.read().lower()
        for forbidden in (
            "contact_demo",
            "pick_place",
            "attach",
            "hand_commander",
            "three_finger_grasp_demo",
            "execute",
        ):
            self.assertNotIn(forbidden, pose_text)

    def test_pick_place_wrapper_uses_only_validated_reachable_files(self):
        root = ET.parse(PICK_PLACE_LAUNCH).getroot()
        self.assertEqual(root.findall("node"), [])
        includes = root.findall("include")
        self.assertEqual(len(includes), 1)
        self.assertEqual(
            includes[0].get("file"),
            "$(find handarm_sim_demo)/launch/three_finger_pick_place_demo.launch",
        )
        args = {
            arg.get("name"): arg.get("value")
            for arg in includes[0].findall("arg")
        }
        self.assertEqual(args["grasp_family"], "top_down")
        self.assertEqual(
            args["scene_config"],
            "$(find handarm_sim_demo)/config/reachable_top_grasp_scene.yaml",
        )
        self.assertEqual(
            args["geometry_config"],
            "$(find handarm_sim_demo)/config/reachable_top_grasp_geometry.yaml",
        )
        self.assertEqual(
            args["world_name"],
            "$(find handarm_sim_demo)/worlds/reachable_top_grasp.world",
        )

    def test_geometry_config_is_strict_top_down_without_oblique_or_side_fallback(self):
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        search = config["search"]
        self.assertEqual(search["expected_object_size_m"], self.object_size)
        self.assertEqual(search["center_offsets_m"], [0.0])
        # Axial probe value from Codex TRAC-IK evidence.
        self.assertEqual(search["top_object_center_in_grasp_frame_z_m"], 0.035)
        self.assertEqual(search["grasp_families"], ["top_down"])
        self.assertEqual(search["top_oblique_tilt_axial_pairs"], [])
        self.assertEqual(search["top_oblique_planar_offsets_m"], [[0.0, 0.0]])
        dumped = yaml.safe_dump(config)
        self.assertNotIn("-30", dumped)
        self.assertNotIn("0.052", dumped)

        geometry = HandGeometry(URDF, GEOMETRY)
        candidates = geometry.coarse_geometry_candidates(
            self.T_world_object, self.object_size, TABLE_Z, "top_down"
        )
        self.assertTrue(candidates)
        self.assertEqual({item.family for item in candidates}, {"top_down"})
        for family in ("top_oblique", "side"):
            with self.assertRaises(ValueError):
                geometry.coarse_geometry_candidates(
                    self.T_world_object, self.object_size, TABLE_Z, family
                )

    def test_top_precision_band_mode_and_depth_gate(self):
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        gates = validate_quality_gate_config(config)
        self.assertEqual(gates["quality_reference_mode"], "top_precision_band")
        self.assertEqual(
            gates["top_precision_band"],
            {
                "minimum_contact_depth_below_top_m": 0.015,
                "maximum_contact_depth_below_top_m": 0.145,
            },
        )

        missing_mode = copy.deepcopy(config)
        del missing_mode["quality_reference_mode"]
        with self.assertRaises(ValueError):
            validate_quality_gate_config(missing_mode)

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

    def test_joint_6_roll_search_accepts_interior_contacts_and_rejects_edges(self):
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        geometry = HandGeometry(URDF, GEOMETRY)
        qualified_rolls = []
        for roll_deg in range(0, 360, 15):
            candidate = geometry.make_candidate(
                self.T_world_object,
                self.object_size,
                TABLE_Z,
                "top_down",
                "object_pos_z",
                float(roll_deg),
            )
            quality = evaluate_candidate_quality(candidate, config)
            if quality.passed:
                qualified_rolls.append(roll_deg)
        # Joint 6 is a bounded contact-placement variable.  The accepted
        # directions place all three fingers on interior side faces, while
        # edge-contact directions are rejected before IK and execution.
        self.assertEqual(
            qualified_rolls, [15, 75, 105, 165, 195, 255, 285, 345]
        )
        for roll_deg in (105, 165):
            candidate = geometry.make_candidate(
                self.T_world_object,
                self.object_size,
                TABLE_Z,
                "top_down",
                "object_pos_z",
                float(roll_deg),
            )
            quality = evaluate_candidate_quality(candidate, config)
            self.assertTrue(candidate.enclosure.valid)
            self.assertTrue(quality.passed, quality.failures)
            self.assertEqual(
                set(candidate.enclosure.contacts), {"f1", "f2", "f3"}
            )
            faces = {
                (contact.face_axis, contact.face_sign)
                for contact in candidate.enclosure.contacts.values()
            }
            self.assertEqual(len(faces), 3)
            self.assertNotIn("AXIAL_OFFSET_TOO_LARGE", quality.failures)
            # Hard contact gate and the reachable-experiment planning gates.
            self.assertGreaterEqual(
                candidate.enclosure.table_clearance_m, 0.008
            )
            self.assertGreaterEqual(
                candidate.enclosure.table_clearance_m, 0.030
            )
            depths = quality.metrics["contact_depth_below_top_m"]
            for depth in depths.values():
                self.assertGreaterEqual(depth, 0.015)
                self.assertLessEqual(depth, 0.145)
            self.assertLess(
                quality.metrics["contact_closure_spread"], 0.034
            )
            self.assertLess(
                quality.metrics["maximum_contact_height_ratio"], 0.25
            )

        edge_candidate = geometry.make_candidate(
            self.T_world_object,
            self.object_size,
            TABLE_Z,
            "top_down",
            "object_pos_z",
            0.0,
        )
        edge_quality = evaluate_candidate_quality(edge_candidate, config)
        self.assertFalse(edge_quality.passed)
        self.assertIn(
            "CONTACT_DEPTH_OUTSIDE_TOP_PRECISION_BAND",
            edge_quality.failures,
        )
        self.assertAlmostEqual(
            edge_quality.metrics["contact_depth_below_top_m"]["f2"], 0.0
        )

    def test_roll_105_candidate_matches_codex_trac_ik_probe(self):
        # Codex real TRAC-IK read-only probe: roll=105 deg solves in
        # collision-free IK.  Verify the geometric precondition of that
        # finding: enclosure + quality pass, the grasp centre sits 35 mm
        # above the object centre and the hand points straight down.
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        geometry = HandGeometry(URDF, GEOMETRY)
        candidate = geometry.make_candidate(
            self.T_world_object,
            self.object_size,
            TABLE_Z,
            "top_down",
            "object_pos_z",
            105.0,
        )
        self.assertTrue(candidate.enclosure.valid)
        self.assertEqual(
            set(candidate.enclosure.contacts), {"f1", "f2", "f3"}
        )
        quality = evaluate_candidate_quality(candidate, config)
        self.assertTrue(quality.passed, quality.failures)
        for depth in quality.metrics["contact_depth_below_top_m"].values():
            self.assertGreaterEqual(depth, 0.015)
            self.assertLessEqual(depth, 0.145)
        self.assertEqual(candidate.family, "top_down")
        self.assertEqual(candidate.object_center_axial_offset_m, 0.035)
        np.testing.assert_allclose(
            candidate.T_world_grasp_center[:3, 3],
            [0.30, 0.10, 0.325],
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            candidate.T_world_hand[:3, 2], [0.0, 0.0, -1.0], atol=1.0e-9
        )
        # The fine-roll refinement window (half width 15 deg, step 2 deg)
        # must include the IK-proven 105-degree roll.
        with open(GEOMETRY, encoding="utf-8") as stream:
            search = yaml.safe_load(stream)["search"]
        self.assertLessEqual(
            search["coarse_roll_step_deg"] * (105 // search["coarse_roll_step_deg"]),
            105,
        )
        self.assertEqual(105 % search["coarse_roll_step_deg"], 0)
        self.assertEqual(search["fine_roll_step_deg"], 2)
        self.assertEqual(search["fine_roll_half_width_deg"], 15)

    def test_search_clearance_gates_admit_reachable_pose_family(self):
        with open(GEOMETRY, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        search = config["search"]
        # Lowering the support surface while keeping the target bottom on it
        # gives the whole hand about 72.8 mm planned clearance at the strict
        # top-down pose, while the independent measured runtime gate remains.
        self.assertEqual(search["minimum_coarse_table_clearance_m"], 0.030)
        self.assertEqual(search["minimum_planned_table_clearance_m"], 0.030)
        self.assertLess(
            search["minimum_coarse_table_clearance_m"], 0.07282
        )
        self.assertLess(
            search["minimum_planned_table_clearance_m"], 0.07282
        )
        # Still above the hard contact-geometry gate.
        hard_gate = config["contact_geometry"]["minimum_table_clearance_m"]
        self.assertGreater(search["minimum_coarse_table_clearance_m"], hard_gate)
        self.assertGreater(search["minimum_planned_table_clearance_m"], hard_gate)
        self.assertEqual(
            config["runtime_acceptance"][
                "post_execution_joint_tolerance_rad"
            ],
            0.025,
        )
        self.assertEqual(
            config["runtime_acceptance"]["minimum_retreat_duration_s"],
            8.0,
        )
        self.assertEqual(
            config["runtime_acceptance"]
            ["maximum_pregrasp_object_displacement_m"],
            0.005,
        )

    def test_oblique_vertical_reference_composes_hand_rotation_once(self):
        # Regression for the former double-rotation bug.  Even when an
        # oblique candidate is rolled and tilted, the upright object's world
        # vertical remains its local +z axis.
        geometry = HandGeometry(URDF, GEOMETRY)
        candidate = geometry.make_candidate(
            self.T_world_object,
            self.object_size,
            TABLE_Z,
            "top_oblique",
            "object_pos_z",
            350.0,
            planar_offset_hand_m=[0.006, -0.009],
            tilt_deg=-30.0,
            object_center_axial_offset_m=0.040,
        )
        self.assertEqual(_object_vertical_reference(candidate), (2, 1))

    def test_existing_double_size_and_legacy_files_are_unchanged(self):
        with open(DOUBLE_SIZE_SCENE, encoding="utf-8") as stream:
            double_target = yaml.safe_load(stream)["objects"]["target"]
        self.assertEqual(double_target["size"], [0.10, 0.12, 0.20])
        self.assertEqual(
            double_target["pose"]["position"], [0.34, 0.18, 0.47]
        )
        self.assertEqual(double_target["mass"], 0.10)
        with open(DOUBLE_SIZE_GEOMETRY, encoding="utf-8") as stream:
            double_config = yaml.safe_load(stream)
        self.assertEqual(
            double_config["quality_reference_mode"], "top_precision_band"
        )
        self.assertEqual(
            double_config["search"]["top_object_center_in_grasp_frame_z_m"],
            0.085,
        )
        self.assertEqual(double_config["search"]["grasp_families"], ["top_down"])
        with open(LEGACY_GEOMETRY, encoding="utf-8") as stream:
            legacy = yaml.safe_load(stream)
        gates = validate_quality_gate_config(legacy)
        self.assertEqual(gates["quality_reference_mode"], "centered_object")
        self.assertEqual(
            gates["maximum_centered_object_center_axial_offset_m"], 0.025
        )


if __name__ == "__main__":
    unittest.main()
