#!/usr/bin/env python3

from pathlib import Path
import unittest
import math
import xml.etree.ElementTree as ET

import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def model_map(world_path):
    root = ET.parse(str(world_path)).getroot()
    world = root.find("world")
    return {model.attrib["name"]: model for model in world.findall("model")}


class GroundSceneTest(unittest.TestCase):
    def test_workspace_world_contains_table_without_grasp_objects(self):
        path = PACKAGE / "worlds/handarm_ground_workspace.world"
        root = ET.parse(str(path)).getroot()
        world = root.find("world")
        includes = [entry.findtext("uri") for entry in world.findall("include")]
        self.assertIn("model://ground_plane", includes)
        self.assertEqual({m.get("name") for m in world.findall("model")},
                         {"work_table", "left_work_table", "right_work_table"})

    def test_grasp_objects_rest_on_platform_by_half_height(self):
        path = PACKAGE / "worlds/handarm_ground_grasp.world"
        models = model_map(path)
        config = yaml.safe_load((PACKAGE / "config/ground_grasp_scene.yaml").read_text())
        expected = {item["name"]: item for item in config["objects"].values()
                    if item.get("scene_manager_enabled", True)}
        self.assertEqual(set(models), set(expected))
        for name, item in expected.items():
            pose = [float(value) for value in models[name].findtext("pose").split()]
            geometry = models[name].find("link/collision/geometry")
            if item["type"] == "compound":
                parts = models[name].findall("link/collision")
                self.assertEqual(len(parts), len(item["collision_parts"]))
                for actual, expected_part in zip(parts, item["collision_parts"]):
                    kind = expected_part["type"]
                    actual_xyz = [float(v) for v in actual.findtext("pose").split()[:3]]
                    for a, b in zip(actual_xyz, expected_part["position"]):
                        self.assertAlmostEqual(a, b, places=9)
                    geom = actual.find("geometry/" + kind)
                    self.assertIsNotNone(geom)
                    dims = ([float(v) for v in geom.findtext("size").split()] if kind == "box" else
                            [float(geom.findtext("length")), float(geom.findtext("radius"))]
                            if kind == "cylinder" else [float(geom.findtext("radius"))])
                    for a, b in zip(dims, expected_part["dimensions"]):
                        self.assertAlmostEqual(a, b, places=9)
                size = item["size"]
            elif item["type"] == "cylinder":
                radius = float(geometry.findtext("cylinder/radius"))
                size = [2 * radius, 2 * radius, float(geometry.findtext("cylinder/length"))]
            else:
                size = [float(value) for value in geometry.findtext("box/size").split()]
            for a, b in zip(size, item["size"]):
                self.assertAlmostEqual(a, b, places=9)
            self.assertEqual(pose[:3], item["pose"]["position"])
            for actual, expected_angle in zip(pose[3:], item["pose"]["orientation_rpy"]):
                self.assertAlmostEqual(actual, expected_angle, places=8)
            if item.get("graspable"):
                self.assertEqual(float(models[name].findtext("link/inertial/mass")), item["mass"])
                roll, pitch = pose[3:5]
                height = (abs(math.sin(pitch))*size[0] +
                          abs(math.cos(pitch)*math.sin(roll))*size[1] +
                          abs(math.cos(pitch)*math.cos(roll))*size[2])
                support_key = {"target_object": "left_table",
                               "ground_object_left": "right_table",
                               "ground_object_right": "table"}[name]
                support = config["objects"][support_key]
                top = support["pose"]["position"][2] + support["size"][2]/2
                self.assertAlmostEqual(pose[2] - 0.5 * height, top + 0.001)

    def test_platform_is_above_floor_without_raising_robot_base(self):
        config = yaml.safe_load((PACKAGE / "config/ground_grasp_scene.yaml").read_text(
            encoding="utf-8"))
        self.assertEqual(config["support_surface_key"], "table")
        table = config["objects"]["table"]
        self.assertAlmostEqual(table["pose"]["position"][2] + table["size"][2]/2, 0.12)
        self.assertAlmostEqual(table["pose"]["position"][0], 0.53)
        for key in ("left_table", "right_table"):
            side = config["objects"][key]
            self.assertAlmostEqual(side["pose"]["position"][2] + side["size"][2]/2, 0.09)
            for leg in side["collision_parts"][1:]:
                foot_z = (side["pose"]["position"][2] + leg["position"][2]
                          - leg["dimensions"][2]/2)
                self.assertAlmostEqual(foot_z, 0.0)
        ground = config["objects"]["ground"]
        top = ground["pose"]["position"][2] + 0.5 * ground["size"][2]
        self.assertAlmostEqual(top, -0.003)
        self.assertFalse(ground["scene_manager_enabled"])
        self.assertTrue(ground["planning_scene_enabled"])
        self.assertIn("ground", config["scenario_object_sets"]["no_obstacle"])
        self.assertEqual(
            set(config["scenario_object_sets"]["no_obstacle"]),
            {"ground", "table", "left_table", "right_table", "target", "left_object", "right_object"})
        self.assertEqual(set(config["scenario_object_sets"]["ground_only"]),
                         {"ground", "table", "left_table", "right_table"})

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
