#!/usr/bin/env python3

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def model_map(world_path):
    root = ET.parse(str(world_path)).getroot()
    world = root.find("world")
    return {model.attrib["name"]: model for model in world.findall("model")}


class GroundSceneTest(unittest.TestCase):
    def test_workspace_world_contains_ground_without_table_or_objects(self):
        path = PACKAGE / "worlds/handarm_ground_workspace.world"
        root = ET.parse(str(path)).getroot()
        world = root.find("world")
        includes = [entry.findtext("uri") for entry in world.findall("include")]
        self.assertIn("model://ground_plane", includes)
        self.assertEqual(world.findall("model"), [])
        self.assertNotIn("work_table", path.read_text(encoding="utf-8"))

    def test_ground_grasp_objects_rest_on_ground_by_half_height(self):
        path = PACKAGE / "worlds/handarm_ground_grasp.world"
        models = model_map(path)
        self.assertNotIn("work_table", models)
        expected = {
            "target_object": (0.10, 0.051),
            "ground_object_left": (0.08, 0.041),
            "ground_object_right": (0.12, 0.061),
        }
        self.assertEqual(set(models), set(expected))
        for name, (height, expected_center_z) in expected.items():
            pose = [float(value) for value in models[name].findtext("pose").split()]
            size = [float(value) for value in models[name].findtext(
                "link/collision/geometry/box/size").split()]
            self.assertAlmostEqual(size[2], height)
            self.assertAlmostEqual(pose[2], expected_center_z)
            self.assertAlmostEqual(pose[2] - 0.5 * size[2], 0.001)

    def test_ground_scene_config_uses_logical_z_zero_support(self):
        config = yaml.safe_load((PACKAGE / "config/ground_grasp_scene.yaml").read_text(
            encoding="utf-8"))
        self.assertEqual(config["support_surface_key"], "ground")
        ground = config["objects"]["ground"]
        top = ground["pose"]["position"][2] + 0.5 * ground["size"][2]
        self.assertAlmostEqual(top, 0.0)
        self.assertFalse(ground["scene_manager_enabled"])
        self.assertNotIn("ground", config["scenario_object_sets"]["no_obstacle"])
        self.assertEqual(
            set(config["scenario_object_sets"]["no_obstacle"]),
            {"target", "left_object", "right_object"})

    def test_ground_launch_is_well_formed_and_uses_new_files(self):
        path = PACKAGE / "launch/ground_grasp_pose_demo.launch"
        root = ET.parse(str(path)).getroot()
        include = root.find("include")
        forwarded = {
            entry.attrib["name"]: entry.attrib.get("value")
            for entry in include.findall("arg")
        }
        self.assertIn("ground_grasp_scene.yaml", forwarded["scene_config"])
        self.assertIn("handarm_ground_grasp.world", forwarded["world_name"])
        self.assertNotIn("work_table", forwarded["expected_models"])


if __name__ == "__main__":
    unittest.main()
