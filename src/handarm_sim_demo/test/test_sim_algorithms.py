#!/usr/bin/env python3

import json
import inspect
import math
import os
import sys
import unittest
import xml.etree.ElementTree as ET

import yaml
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix


PACKAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOVEIT_CONFIG = os.path.abspath(
    os.path.join(PACKAGE, "..", "abb120_moveit_config1")
)
HAND_PACKAGE = os.path.abspath(os.path.join(PACKAGE, "..", "handarmtest1"))
sys.path.insert(0, os.path.join(PACKAGE, "scripts"))

from scene_manager import compose_pose, quaternion_angle, validate_config
from scripted_avoidance_demo import make_pose, quaternion_angle_deg
from hand_commander import (
    PUBLIC_OPERATOR_COMMANDS,
    VALID_COMMANDS,
    build_command_trajectory_waypoints,
    collect_failure_diagnostics,
    command_failure_result,
    normalize_command,
    validate_hand_config,
)
from safe_hand_trajectory_proxy import (
    interpolate_segment,
    sampled_path,
    validate_goal_trajectory,
)
from run_hand_stability_tests import (
    all_joint_limits,
    classify_service_rate_outliers,
    evaluate_heartbeat_coverage,
    evaluate_stability,
)
from multi_angle_avoidance_course import (
    nearest_equivalent_within_bounds,
    validate_course_config,
)
from startup_coordinator import joint_position_error
from scripted_pick_demo import (
    canonicalize_periodic_joint_goal,
    classify_target_contact_pairs,
    compose_pregrasp,
    enforce_minimum_trajectory_duration,
    evaluate_contact_clear_sample,
    is_contact_obstruction_candidate,
    position_distance,
    quaternion_angle_deg as pick_angle_deg,
    release_joint_diagnostics,
    set_acm_pairs,
    trajectory_duration_s,
    validate_grasp_displacement,
    validate_release_acceptance_config,
)


class _FakeDuration:
    def __init__(self, seconds):
        self._seconds = float(seconds)

    def to_sec(self):
        return self._seconds

    @classmethod
    def from_sec(cls, seconds):
        return cls(seconds)


class _FakeTrajectoryPoint:
    def __init__(
        self,
        time_s,
        positions=None,
        velocities=None,
        accelerations=None,
        effort=None,
    ):
        self.time_from_start = _FakeDuration(time_s)
        self.positions = [] if positions is None else positions
        self.velocities = [] if velocities is None else velocities
        self.accelerations = [] if accelerations is None else accelerations
        self.effort = [] if effort is None else effort


class _FakeJointTrajectory:
    def __init__(self, points, joint_names=None):
        self.points = list(points)
        self.joint_names = [] if joint_names is None else list(joint_names)


class _FakeTrajectory:
    def __init__(self, points):
        self.joint_trajectory = _FakeJointTrajectory(points)


class SceneAlgorithmsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(PACKAGE, "config", "demo_scene.yaml")) as stream:
            cls.scene = yaml.safe_load(stream)

    def test_all_scenarios_validate(self):
        for scenario in self.scene["scenario_object_sets"]:
            self.assertGreater(len(validate_config(self.scene, scenario)), 0)

    def test_unknown_scenario_rejected(self):
        with self.assertRaises(ValueError):
            validate_config(self.scene, "does_not_exist")

    def test_invalid_size_rejected(self):
        data = yaml.safe_load(yaml.safe_dump(self.scene))
        data["objects"]["table"]["size"][0] = 0.0
        with self.assertRaises(ValueError):
            validate_config(data, "no_obstacle")

    def test_quaternion_sign_equivalence(self):
        a = Quaternion(x=0.1, y=0.2, z=0.3, w=math.sqrt(0.86))
        b = Quaternion(x=-a.x, y=-a.y, z=-a.z, w=-a.w)
        self.assertAlmostEqual(quaternion_angle(a, b), 0.0, places=12)
        self.assertAlmostEqual(quaternion_angle_deg(a, b), 0.0, places=10)

    def test_pose_composition(self):
        parent = Pose()
        parent.position.x = 1.0
        parent.orientation.z = math.sqrt(0.5)
        parent.orientation.w = math.sqrt(0.5)
        child = Pose()
        child.position.x = 2.0
        child.orientation.w = 1.0
        result = compose_pose(parent, child)
        self.assertAlmostEqual(result.position.x, 1.0, places=12)
        self.assertAlmostEqual(result.position.y, 2.0, places=12)

    def test_avoidance_goal_contract(self):
        for name, spec in self.scene["avoidance_goals"].items():
            pose = make_pose(spec)
            self.assertTrue(math.isfinite(pose.position.x), name)
            norm = math.sqrt(
                pose.orientation.x ** 2
                + pose.orientation.y ** 2
                + pose.orientation.z ** 2
                + pose.orientation.w ** 2
            )
            self.assertAlmostEqual(norm, 1.0, places=5, msg=name)
            self.assertIn(spec["expected_outcome"], ("success", "safe_failure"))

    def test_hand_command_contract(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        self.assertTrue(validate_hand_config(config))
        self.assertEqual(set(config["commands"]), VALID_COMMANDS)
        names = config["joint_names"]
        opened = dict(zip(names, config["commands"]["OPEN"]["positions"]))
        closed = dict(zip(names, config["commands"]["CLOSE"]["positions"]))
        for name in config["execution"]["configuration_joint_names"]:
            self.assertEqual(opened[name], closed[name])
        for name in config["execution"]["flexion_joint_names"]:
            self.assertGreater(closed[name], opened[name])

    def test_safe_hand_proxy_samples_every_segment_at_20_milliradians(self):
        names = ["f1j1", "f1j2", "f2j1", "f3j2"]
        trajectory = _FakeJointTrajectory([
            _FakeTrajectoryPoint(1.0, [0.18, 0.20, 0.20, 0.20]),
            _FakeTrajectoryPoint(2.0, [0.18, 0.85, 0.85, 0.90]),
        ], names)
        samples = sampled_path(
            [0.10, 0.10, 0.10, 0.10], trajectory, names, 0.02)
        self.assertGreater(len(samples), 30)
        previous = [0.10, 0.10, 0.10, 0.10]
        for sample in samples:
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(previous, sample)),
                0.020000000001)
            previous = sample
        for actual, expected in zip(
                samples[-1], [0.18, 0.85, 0.85, 0.90]):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_safe_hand_proxy_rejects_limit_violation_before_collision_query(self):
        names = ["f1j1", "f1j2", "f2j1", "f3j2"]
        limits = {name: (0.0, 1.0) for name in names}
        trajectory = _FakeJointTrajectory([
            _FakeTrajectoryPoint(1.0, [0.2, 1.01, 0.2, 0.2]),
        ], names)
        self.assertIn(
            "violates f1j2 limits",
            validate_goal_trajectory(trajectory, names, limits))

    def test_segment_interpolator_includes_target_without_oversized_step(self):
        samples = interpolate_segment([0.0, 0.0], [0.051, -0.001], 0.02)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[-1], [0.051, -0.001])

    def test_all_runtime_urdfs_share_safe_hand_dynamics(self):
        hand_joints = {
            "f1j1", "f1j2", "f1j3", "f2j1",
            "f2j2", "f3j1", "f3j2", "f3j3",
        }
        hand_links = {
            "handbase_link", "f1link1", "f1link2", "f1link3",
            "f2link1", "f2link2", "f3link1", "f3link2", "f3link3",
        }
        signatures = []
        for filename in ("gazebo_handarm.urdf", "gazebo_handarm_velocity.urdf"):
            root = ET.parse(os.path.join(MOVEIT_CONFIG, "config", filename)).getroot()
            joint_signature = {}
            for joint in root.findall("joint"):
                name = joint.get("name")
                if name not in hand_joints:
                    continue
                self.assertEqual(joint.get("type"), "revolute", (filename, name))
                limit = joint.find("limit")
                dynamics = joint.find("dynamics")
                self.assertIsNotNone(limit, (filename, name))
                self.assertIsNotNone(dynamics, (filename, name))
                self.assertLessEqual(float(limit.get("effort")), 4.0)
                self.assertLessEqual(float(limit.get("velocity")), 0.8)
                self.assertGreater(float(dynamics.get("damping")), 0.0)
                self.assertEqual(float(dynamics.get("friction")), 0.0)
                joint_signature[name] = (
                    joint.get("type"),
                    tuple(sorted(limit.attrib.items())),
                    tuple(sorted(dynamics.attrib.items())),
                )
            self.assertEqual(set(joint_signature), hand_joints)
            for link in root.findall("link"):
                if link.get("name") not in hand_links:
                    continue
                collisions = link.findall("collision")
                self.assertTrue(collisions, (filename, link.get("name")))
                if link.get("name") == "handbase_link":
                    meshes = [collision.find("geometry/mesh")
                              for collision in collisions]
                    self.assertEqual(len(meshes), 1)
                    self.assertTrue(meshes[0].get("filename").endswith(
                        "handbase_link_collision_8mm.STL"))
                    continue
                for collision in collisions:
                    self.assertIsNone(
                        collision.find("geometry/mesh"),
                        (filename, link.get("name")),
                    )
            plugins = {
                plugin.findtext("mimicJoint"): plugin
                for plugin in root.findall("gazebo/plugin")
                if plugin.get("filename")
                == "libroboticsgroup_upatras_gazebo_mimic_joint_plugin.so"
            }
            self.assertEqual(set(plugins), {"f3j1", "f1j3", "f2j2", "f3j3"})
            friction = {
                gazebo.get("reference"): (
                    gazebo.findtext("mu1"), gazebo.findtext("mu2")
                )
                for gazebo in root.findall("gazebo")
                if gazebo.get("reference") in hand_links - {"handbase_link"}
                and gazebo.find("mu1") is not None
            }
            self.assertEqual(set(friction), hand_links - {"handbase_link"})
            self.assertTrue(
                all(values == ("1.5", "1.5") for values in friction.values())
            )
            for name, plugin in plugins.items():
                self.assertTrue(plugin.findtext("hasPID"), (filename, name))
                self.assertLessEqual(float(plugin.findtext("maxEffort")), 4.0)
                self.assertEqual(
                    float(plugin.findtext("diagnosticVelocityThreshold")), 0.05
                )
            signatures.append(joint_signature)
        self.assertEqual(signatures[0], signatures[1])

    def test_legacy_xacro_cannot_restore_unstable_hand_physics(self):
        root = ET.parse(os.path.join(HAND_PACKAGE, "xacro", "hand.xacro")).getroot()
        hand_joints = {
            "f1j1", "f1j2", "f1j3", "f2j1",
            "f2j2", "f3j1", "f3j2", "f3j3",
        }
        for joint in root.findall("joint"):
            if joint.get("name") not in hand_joints:
                continue
            self.assertEqual(joint.get("type"), "revolute")
            self.assertLessEqual(float(joint.find("limit").get("effort")), 4.0)
            self.assertLessEqual(float(joint.find("limit").get("velocity")), 0.8)
            self.assertGreater(float(joint.find("dynamics").get("damping")), 0.0)
            self.assertEqual(float(joint.find("dynamics").get("friction")), 0.0)
        self.assertEqual(
            {joint.get("name") for joint in root.findall("joint")}
            & hand_joints,
            hand_joints,
        )
        for link in root.findall("link"):
            collisions = link.findall("collision")
            if link.get("name") == "handbase_link":
                self.assertEqual(len(collisions), 1)
                mesh = collisions[0].find("geometry/mesh")
                self.assertIsNotNone(mesh)
                self.assertTrue(mesh.get("filename").endswith(
                    "handbase_link_collision_8mm.STL"))
                continue
            for collision in collisions:
                self.assertIsNone(collision.find("geometry/mesh"))
        with open(
            os.path.join(HAND_PACKAGE, "xacro", "hand_g.xacro"),
            encoding="utf-8",
        ) as stream:
            gazebo_source = stream.read()
        self.assertNotIn("<turnGravityOff>true</turnGravityOff>", gazebo_source)
        with open(
            os.path.join(HAND_PACKAGE, "launch", "irb120_gazebo.launch"),
            encoding="utf-8",
        ) as stream:
            legacy_launch = stream.read()
        self.assertIn("simulation_baseline.launch", legacy_launch)
        self.assertNotIn("spawn_model", legacy_launch)

    def test_grasp_cannot_move_configuration_joint(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        config["commands"]["CLOSE"]["positions"][0] = 3.10
        with self.assertRaisesRegex(ValueError, "must not move configuration"):
            validate_hand_config(config)

    def test_hand_stability_evaluator_accepts_quiet_followers(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        positions = dict(
            zip(config["joint_names"], config["commands"]["OPEN"]["positions"])
        )
        for mimic, relation in config["mimic_joints"].items():
            positions[mimic] = positions[relation["source"]]
        samples = []
        for index in range(101):
            samples.append(
                {
                    "elapsed_s": index * 0.05,
                    "joints": {
                        name: {"position": value, "velocity": 0.001}
                        for name, value in positions.items()
                    },
                }
            )
        result = evaluate_stability(samples, config, "RELEASE")
        self.assertTrue(result["success"], result["failure_reasons"])
        self.assertEqual(result["sample_count"], 101)
        self.assertAlmostEqual(
            result["joint_metrics"]["f3j1"]["tail_velocity_at_abs_max_rad_s"],
            0.001,
        )

    def test_hand_stability_evaluator_rejects_one_bad_distal_joint(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        positions = dict(
            zip(config["joint_names"], config["commands"]["OPEN"]["positions"])
        )
        for mimic, relation in config["mimic_joints"].items():
            positions[mimic] = positions[relation["source"]]
        samples = []
        for index in range(101):
            joints = {
                name: {"position": value, "velocity": 0.001}
                for name, value in positions.items()
            }
            if index >= 80:
                joints["f2j2"] = {"position": 0.30, "velocity": 0.20}
            samples.append({"elapsed_s": index * 0.05, "joints": joints})
        result = evaluate_stability(samples, config, "RELEASE")
        self.assertFalse(result["success"])
        self.assertIn("f2j2 relation error", result["failure_reasons"])
        self.assertIn("f2j2 settled velocity p95", result["failure_reasons"])
        self.assertIn("f2j2 position range", result["failure_reasons"])
        self.assertAlmostEqual(
            result["joint_metrics"]["f2j2"]["tail_velocity_abs_max_elapsed_s"],
            4.0,
        )

    def test_hand_stability_evaluator_records_one_async_rate_outlier(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        positions = dict(
            zip(config["joint_names"], config["commands"]["OPEN"]["positions"])
        )
        for mimic, relation in config["mimic_joints"].items():
            positions[mimic] = positions[relation["source"]]
        samples = []
        for index in range(101):
            joints = {
                name: {"position": value, "velocity": 0.001}
                for name, value in positions.items()
            }
            if index == 80:
                joints["f2j2"]["velocity"] = 0.5
            samples.append({"elapsed_s": index * 0.05, "joints": joints})
        result = evaluate_stability(samples, config, "RELEASE")
        self.assertTrue(result["success"], result["failure_reasons"])
        self.assertEqual(
            result["joint_metrics"]["f2j2"][
                "tail_velocity_threshold_exceedance_count"
            ],
            1,
        )
        self.assertAlmostEqual(
            result["joint_metrics"]["f2j2"]["tail_velocity_abs_max_rad_s"],
            0.5,
        )

    def test_post_step_heartbeat_coverage_is_fail_closed(self):
        mimic_names = ("f3j1", "f1j3", "f2j2", "f3j3")
        rows = []
        for name in mimic_names:
            for index in range(25):
                rows.append(
                    {
                        "joint": "source_" + name,
                        "mimic_joint": name,
                        "window_start_sim_time_s": 10.0 + 0.1 * index,
                        "sim_time_s": 10.1 + 0.1 * index,
                        "window_update_count": 100,
                        "max_abs_source_velocity_rad_s": 0.001,
                        "max_abs_mimic_velocity_rad_s": 0.001,
                    }
                )
        result = evaluate_heartbeat_coverage(rows, mimic_names, 0.05)
        self.assertTrue(result["success"], result["failure_reasons"])
        rows = [row for row in rows if row["mimic_joint"] != "f2j2"]
        result = evaluate_heartbeat_coverage(rows, mimic_names, 0.05)
        self.assertFalse(result["success"])
        self.assertIn("f2j2 diagnostic heartbeat count", result["failure_reasons"])

    def test_service_rate_outlier_requires_covering_post_step_window(self):
        metrics = {
            "f2j2": {
                "service_rate_outliers": [
                    {
                        "sim_time_before_s": 12.020,
                        "sim_time_after_s": 12.021,
                        "velocity_rad_s": 0.5,
                    }
                ]
            }
        }
        heartbeat = {
            "joint": "f2j1",
            "mimic_joint": "f2j2",
            "window_start_sim_time_s": 12.0,
            "sim_time_s": 12.1,
            "max_abs_mimic_velocity_rad_s": 0.001,
        }
        self.assertEqual(
            classify_service_rate_outliers(metrics, [heartbeat], 0.05), []
        )
        self.assertEqual(
            metrics["f2j2"]["service_rate_outliers"][0]["classification"],
            "ASYNC_SERVICE_MID_UPDATE_ARTIFACT_SUPPORTED",
        )
        heartbeat["max_abs_mimic_velocity_rad_s"] = 0.5
        self.assertEqual(
            classify_service_rate_outliers(metrics, [heartbeat], 0.05), ["f2j2"]
        )

    def test_service_rate_outlier_count_is_fail_closed(self):
        metrics = {
            "f2j2": {
                "service_rate_outliers": [
                    {
                        "sim_time_before_s": 12.020 + 0.01 * index,
                        "sim_time_after_s": 12.021 + 0.01 * index,
                        "velocity_rad_s": 0.5,
                    }
                    for index in range(2)
                ]
            }
        }
        heartbeat = {
            "joint": "f2j1",
            "mimic_joint": "f2j2",
            "window_start_sim_time_s": 12.0,
            "sim_time_s": 12.1,
            "max_abs_mimic_velocity_rad_s": 0.001,
        }
        self.assertEqual(
            classify_service_rate_outliers(
                metrics, [heartbeat], 0.05, max_supported_per_joint=1
            ),
            ["f2j2"],
        )
        self.assertEqual(
            metrics["f2j2"]["supported_service_rate_outlier_count"], 2
        )

    def test_hand_stability_limits_include_passive_joints(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        limits = all_joint_limits(config)
        self.assertEqual(
            set(limits),
            set(config["joint_names"]) | set(config["mimic_joints"]),
        )
        self.assertEqual(limits["f2j2"], limits["f2j1"])

    def test_hand_limit_violation_rejected(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        config["commands"]["CLOSE"]["positions"][0] = 3.5
        with self.assertRaises(ValueError):
            validate_hand_config(config)

    def test_public_hand_command_aliases(self):
        self.assertEqual(PUBLIC_OPERATOR_COMMANDS, {"GRASP", "RELEASE"})
        self.assertEqual(normalize_command("grasp"), ("GRASP", "CLOSE"))
        self.assertEqual(normalize_command("release"), ("RELEASE", "OPEN"))
        self.assertEqual(
            normalize_command("set_configuration:PRE_SHAPE_A"),
            ("SET_CONFIGURATION_PRE_SHAPE_A", "SET_CONFIGURATION_PRE_SHAPE_A"),
        )
        self.assertEqual(
            normalize_command("set_configuration_b"),
            ("SET_CONFIGURATION_B", "SET_CONFIGURATION_B"),
        )

    def test_wraparound_joint_error_uses_shortest_angle(self):
        target = -0.4533
        measured = target + 2.0 * math.pi + 2.0e-6
        self.assertAlmostEqual(
            joint_position_error(measured, target, wraparound=True), 2.0e-6, places=10
        )
        self.assertGreater(
            joint_position_error(measured, target, wraparound=False), 6.0
        )

    def test_all_controlled_revolute_joints_use_periodic_startup_error(self):
        with open(
            os.path.join(PACKAGE, "config", "startup_configuration.yaml")
        ) as stream:
            initial = yaml.safe_load(stream)["initial_configuration"]
        controlled = set(initial["arm"]["joint_names"])
        controlled.update(initial["hand"]["joint_names"])
        self.assertEqual(controlled, set(initial["wraparound_joints"]))

    def test_object_relative_pregrasp_composition(self):
        object_spec = {
            "pose": {
                "position": [1.0, 2.0, 3.0],
                "orientation_rpy": [0.0, 0.0, math.pi / 2.0],
            }
        }
        relative = {
            "position_m": [1.0, 0.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        pose = compose_pregrasp(object_spec, relative)
        self.assertAlmostEqual(pose.position.x, 1.0, places=10)
        self.assertAlmostEqual(pose.position.y, 3.0, places=10)
        self.assertAlmostEqual(pose.position.z, 3.5, places=10)

    def test_pick_quaternion_sign_equivalence(self):
        a = Quaternion(x=0.1, y=0.2, z=0.3, w=math.sqrt(0.86))
        b = Quaternion(x=-a.x, y=-a.y, z=-a.z, w=-a.w)
        self.assertAlmostEqual(pick_angle_deg(a, b), 0.0, places=10)

    def test_nonphysical_attachment_is_explicit_and_isolated(self):
        with open(os.path.join(PACKAGE, "config", "grasp_demo.yaml")) as stream:
            config = yaml.safe_load(stream)
        self.assertNotIn("deterministic_lift", config["supported_grasp_modes"])
        self.assertIn(
            "fixed_attachment_demo_nonphysical", config["supported_grasp_modes"]
        )
        self.assertEqual(config["lift_vector_world_m"], [0.0, 0.0, 0.10])
        self.assertEqual(
            config["attachment_service"],
            "/handarm_sim_demo/set_object_attached",
        )
        with open(
            os.path.join(PACKAGE, "worlds", "handarm_pick_obstacle.world")
        ) as stream:
            physical_world = stream.read()
        with open(
            os.path.join(
                PACKAGE,
                "worlds",
                "handarm_pick_obstacle_nonphysical_attachment.world",
            )
        ) as stream:
            nonphysical_world = stream.read()
        self.assertNotIn("handarm_deterministic_attach", physical_world)
        self.assertIn("handarm_deterministic_attach", nonphysical_world)

    def test_physical_grasp_world_has_no_unmodeled_guide_fixture(self):
        with open(
            os.path.join(PACKAGE, "worlds", "handarm_physical_grasp.world")
        ) as stream:
            world = stream.read()
        self.assertNotIn("grasp_guide_", world)
        self.assertNotIn("grasp_pedestal", world)
        with open(
            os.path.join(PACKAGE, "config", "physical_grasp_scene.yaml")
        ) as stream:
            scene = yaml.safe_load(stream)
        self.assertNotIn("gazebo_only_grasp_fixture", scene)
        self.assertEqual(scene["objects"]["target"]["size"], [0.05, 0.06, 0.10])
        self.assertEqual(
            scene["objects"]["target"]["planning_padding_m"],
            [0.02, 0.02, 0.02],
        )
        self.assertGreater(
            min(
                physical + 2.0 * padding - physical
                for physical, padding in zip(
                    scene["objects"]["target"]["size"],
                    scene["objects"]["target"]["planning_padding_m"],
                )
            ),
            0.0,
        )

    def test_physical_grasp_world_reports_contacts_without_attachment(self):
        root = ET.parse(
            os.path.join(PACKAGE, "worlds", "handarm_physical_grasp.world")
        ).getroot()
        plugin = root.find(
            ".//model[@name='target_object']/link/sensor/"
            "plugin[@filename='libgazebo_ros_bumper.so']"
        )
        self.assertIsNotNone(plugin)
        self.assertEqual(
            plugin.findtext("bumperTopicName"),
            "/handarm_sim_demo/target_contacts",
        )
        self.assertIsNone(root.find(".//plugin[@filename='libgazebo_ros_link_attacher.so']"))

    def test_physical_grasp_flow_places_before_release(self):
        with open(
            os.path.join(PACKAGE, "scripts", "scripted_pick_demo.py"),
            encoding="utf-8",
        ) as stream:
            source = stream.read()
        place_call = "placed = self.execute_place(target_object, object_before, row)"
        release_call = "self.release_on_table("
        self.assertIn(place_call, source)
        self.assertIn(release_call, source)
        self.assertLess(source.index(place_call), source.index(release_call))
        self.assertNotIn("SetModelState", source)
        with open(
            os.path.join(PACKAGE, "config", "physical_grasp_demo.yaml")
        ) as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["lift_vector_world_m"][:2], [0.0, 0.0])
        self.assertGreater(config["lift_vector_world_m"][2], 0.0)
        self.assertLessEqual(config["object_follow_tolerance_m"], 0.010)
        self.assertLess(config["transit_retreat_vector_world_m"][0], 0.0)
        self.assertLessEqual(
            config["maximum_pregrasp_object_displacement_m"], 0.005
        )

    def test_contact_classifier_requires_distinct_finger_families(self):
        class State:
            def __init__(self, first, second):
                self.collision1_name = first
                self.collision2_name = second

        states = [
            State("target_object::object_link::collision", "work_table::table_link::collision"),
            State("target_object::object_link::collision", "robot::f1link1::collision"),
            State("robot::f1link2::collision_1", "target_object::object_link::collision"),
            State("target_object::object_link::collision", "robot::f2link2::collision"),
            State("target_object::object_link::collision", "robot::handbase_link::collision"),
        ]
        families, pairs = classify_target_contact_pairs(states)
        self.assertEqual(families, {"f1", "f2"})
        self.assertEqual(len(pairs), 3)

    def test_target_touch_allowance_is_scoped_and_symmetric(self):
        matrix = AllowedCollisionMatrix(
            entry_names=["target_object", "work_table"],
            entry_values=[
                AllowedCollisionEntry(enabled=[False, False]),
                AllowedCollisionEntry(enabled=[False, False]),
            ],
        )
        set_acm_pairs(matrix, "target_object", ["f1link1", "f2link1"], True)
        indices = {name: index for index, name in enumerate(matrix.entry_names)}
        target = indices["target_object"]
        table = indices["work_table"]
        for link in ("f1link1", "f2link1"):
            finger = indices[link]
            self.assertTrue(matrix.entry_values[target].enabled[finger])
            self.assertTrue(matrix.entry_values[finger].enabled[target])
        self.assertFalse(matrix.entry_values[target].enabled[table])
        self.assertFalse(matrix.entry_values[table].enabled[target])
        self.assertTrue(
            all(
                len(entry.enabled) == len(matrix.entry_names)
                for entry in matrix.entry_values
            )
        )

    def test_spawn_initialization_has_single_owner(self):
        """Delayed spawn_model -J resets must not race running controllers."""
        with open(
            os.path.join(PACKAGE, "launch", "simulation_baseline.launch")
        ) as stream:
            launch = stream.read()
        spawn_line = next(
            line for line in launch.splitlines() if "args=\"-urdf" in line
        )
        self.assertNotIn("initial_joint_positions", spawn_line)
        self.assertNotIn(" -J ", spawn_line)
        with open(
            os.path.join(PACKAGE, "scripts", "startup_coordinator.py")
        ) as stream:
            coordinator = stream.read()
        self.assertIn("set_configuration(", coordinator)
        self.assertLess(
            coordinator.index("switch_thread.start()"),
            coordinator.index('rospy.ServiceProxy("/gazebo/unpause_physics"'),
        )

    def test_joint6_preserves_the_full_abb_axis6_working_range(self):
        for filename in ("gazebo_handarm.urdf", "gazebo_handarm_velocity.urdf"):
            urdf_path = os.path.join(
                os.path.dirname(PACKAGE),
                "abb120_moveit_config1",
                "config",
                filename,
            )
            root = ET.parse(urdf_path).getroot()
            joint = root.find("./joint[@name='joint_6']")
            self.assertIsNotNone(joint, filename)
            self.assertEqual(joint.attrib["type"], "revolute", filename)
            limit = joint.find("limit")
            self.assertAlmostEqual(float(limit.attrib["lower"]), -6.981317)
            self.assertAlmostEqual(float(limit.attrib["upper"]), 6.981317)
            folded_stop = 6.98132 - 2.0 * math.pi
            self.assertGreater(abs(float(limit.attrib["lower"])), folded_stop)

    def test_all_handarm_worlds_bound_contact_correction_velocity(self):
        for filename in (
            "handarm_pick_obstacle.world",
            "handarm_pick_obstacle_nonphysical_attachment.world",
            "handarm_physical_grasp.world",
        ):
            root = ET.parse(os.path.join(PACKAGE, "worlds", filename)).getroot()
            value = root.findtext(
                "world/physics/ode/constraints/contact_max_correcting_vel"
            )
            self.assertIsNotNone(value, filename)
            self.assertLessEqual(float(value), 0.1, filename)

    def test_position_distance(self):
        a = Pose().position
        b = Pose().position
        b.x, b.y, b.z = 1.0, 2.0, 2.0
        self.assertAlmostEqual(position_distance(a, b), 3.0, places=12)

    def test_grasp_displacement_validator_accepts_valid_and_boundary_values(self):
        self.assertTrue(validate_grasp_displacement(0.0, 0.010))
        self.assertTrue(validate_grasp_displacement(0.005, 0.010))
        self.assertTrue(validate_grasp_displacement(0.010, 0.010))

    def test_grasp_displacement_validator_rejects_motion_over_limit(self):
        self.assertFalse(validate_grasp_displacement(0.0100001, 0.010))
        self.assertFalse(validate_grasp_displacement(0.011, 0.010))

    def test_grasp_displacement_validator_rejects_nonfinite_and_negative(self):
        for displacement in (float("nan"), float("inf"), float("-inf"), -1.0e-9):
            self.assertFalse(
                validate_grasp_displacement(displacement, 0.010),
                displacement,
            )

    def test_grasp_displacement_validator_rejects_invalid_limits(self):
        for maximum in (float("nan"), float("inf"), float("-inf"), -1.0e-9):
            with self.assertRaises(ValueError, msg=maximum):
                validate_grasp_displacement(0.0, maximum)

    def test_only_contact_explainable_hand_failures_may_reach_contact_gate(self):
        for reason in (
            "active joint verification failed",
            "mimic joint relation verification failed",
        ):
            self.assertTrue(
                is_contact_obstruction_candidate(
                    {"success": False, "failure_reason": reason}
                )
            )
        for reason in (
            "trajectory timeout",
            "trajectory failed -4 path tolerance",
            "mimic joint stability verification failed",
            "",
        ):
            self.assertFalse(
                is_contact_obstruction_candidate(
                    {"success": False, "failure_reason": reason}
                )
            )
        self.assertFalse(
            is_contact_obstruction_candidate(
                {
                    "success": True,
                    "failure_reason": "mimic joint relation verification failed",
                }
            )
        )

    def test_physical_acceptance_config_sets_grasp_displacement_limit(self):
        with open(
            os.path.join(PACKAGE, "config", "physical_grasp_demo.yaml")
        ) as stream:
            config = yaml.safe_load(stream)
        maximum = config["physical_acceptance"]["maximum_grasp_displacement_m"]
        self.assertEqual(maximum, 0.010)
        self.assertTrue(validate_grasp_displacement(0.010, maximum))

    def test_public_hand_command_execution_duration_is_eight_seconds(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["execution"]["duration_s"], 8.0)

    def test_open_command_retracts_f2_before_f1_and_f3(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        names = config["joint_names"]
        current = dict(zip(names, config["commands"]["CLOSE"]["positions"]))
        opened = config["commands"]["OPEN"]["positions"]
        waypoints = build_command_trajectory_waypoints(
            config, "OPEN", current, opened
        )
        self.assertEqual([item[0] for item in waypoints], [0.2, 4.2, 8.2])
        stage_one = dict(zip(names, waypoints[1][1]))
        self.assertEqual(stage_one["f1j1"], 0.18)
        self.assertEqual(stage_one["f2j1"], 0.20)
        self.assertEqual(stage_one["f1j2"], current["f1j2"])
        self.assertEqual(stage_one["f3j2"], current["f3j2"])
        self.assertEqual(waypoints[-1][1], opened)

    def test_close_command_remains_one_synchronized_segment(self):
        with open(os.path.join(PACKAGE, "config", "hand_commands.yaml")) as stream:
            config = yaml.safe_load(stream)
        names = config["joint_names"]
        current = dict(zip(names, config["commands"]["OPEN"]["positions"]))
        closed = config["commands"]["CLOSE"]["positions"]
        waypoints = build_command_trajectory_waypoints(
            config, "CLOSE", current, closed
        )
        self.assertEqual([item[0] for item in waypoints], [0.2, 8.2])
        self.assertEqual(waypoints[-1][1], closed)

    def test_open_delayed_joint_must_be_flexion_only(self):
        path = os.path.join(PACKAGE, "config", "hand_commands.yaml")
        for delayed in ([], ["f1j1"], ["f3j1"]):
            with open(path) as stream:
                config = yaml.safe_load(stream)
            config["execution"]["open_delayed_joints"] = delayed
            with self.assertRaises(ValueError):
                validate_hand_config(config)

    def test_contact_clear_requires_fresh_empty_sample(self):
        self.assertEqual(
            evaluate_contact_clear_sample(10.0, 9.9, set(), 0.2),
            (True, True),
        )
        self.assertEqual(
            evaluate_contact_clear_sample(10.0, 9.9, {"f2"}, 0.2),
            (True, False),
        )
        self.assertEqual(
            evaluate_contact_clear_sample(10.0, 9.6, set(), 0.2),
            (False, False),
        )
        self.assertEqual(
            evaluate_contact_clear_sample(10.0, None, set(), 0.2),
            (False, False),
        )

    def test_release_acceptance_is_single_attempt_and_positive(self):
        path = os.path.join(PACKAGE, "config", "physical_grasp_demo.yaml")
        with open(path) as stream:
            config = yaml.safe_load(stream)
        self.assertTrue(validate_release_acceptance_config(config))
        config["physical_acceptance"]["release_attempts"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_release_acceptance_config(config)

    def test_release_restores_target_and_acm_before_retreat_planning(self):
        source = inspect.getsource(
            __import__("scripted_pick_demo").ScriptedPickDemo.release_on_table
        )
        proxy = source.index("restore_exact_target_proxy")
        acm = source.index("restore_collision_matrix")
        retreat = source.index("self.execute_cartesian_segment")
        self.assertLess(proxy, acm)
        self.assertLess(acm, retreat)

    def test_action_failure_diagnostics_never_promote_success(self):
        active = ["f1j1", "f1j2", "f2j1", "f3j2"]
        mimic = ["f3j1", "f1j3", "f2j2", "f3j3"]

        def verify(target, command_name, start):
            actual = {name: 0.20 for name in active + mimic}
            errors = {name: 0.0 for name in active}
            return True, True, actual, errors, {}, {}

        diagnostics = collect_failure_diagnostics(
            verify, [0.18, 0.20, 0.20, 0.20], "OPEN", {}, active, mimic
        )
        result = command_failure_result(
            "RELEASE",
            "OPEN",
            dict(zip(active, [0.18, 0.20, 0.20, 0.20])),
            8.2,
            "trajectory failed -5 goal tolerance",
            -5,
            "goal tolerance",
            diagnostics,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], -5)
        self.assertEqual(result["actual_joint_positions"], diagnostics["actual_joint_positions"])
        json.dumps(result)

    def test_action_failure_diagnostic_sampling_error_is_structured(self):
        def verify(target, command_name, start):
            raise RuntimeError("joint state unavailable")

        diagnostics = collect_failure_diagnostics(
            verify, [0.18, 0.20, 0.20, 0.20], "OPEN", {}, [], []
        )
        self.assertFalse(diagnostics["verification_success"])
        self.assertIn("joint state unavailable", diagnostics["sample_error"])
        self.assertIsNone(diagnostics["actual_joint_positions"])

    def test_release_joint_diagnostics_is_json_ready(self):
        result = release_joint_diagnostics(
            {
                "success": False,
                "failure_reason": "active joint verification failed",
                "target_joint_positions": {"f2j1": 0.20},
                "actual_joint_positions": {"f2j1": 0.265},
                "active_joint_errors_rad": {"f2j1": 0.065},
                "failure_diagnostics": {"sample_error": ""},
                "error_code": -5,
                "error_string": "goal tolerance",
            }
        )
        self.assertEqual(result["active_joint_errors_rad"]["f2j1"], 0.065)
        json.dumps(result)

    def test_trajectory_duration_reads_last_waypoint(self):
        class Duration:
            @staticmethod
            def to_sec():
                return 4.25

        class Point:
            time_from_start = Duration()

        class JointTrajectory:
            points = [Point()]

        class Trajectory:
            joint_trajectory = JointTrajectory()

        self.assertEqual(trajectory_duration_s(Trajectory()), 4.25)

    def test_minimum_trajectory_duration_keeps_long_trajectory_unchanged(self):
        first = _FakeTrajectoryPoint(
            0.0,
            positions=[1.0],
            velocities=[0.10],
            accelerations=[0.30],
            effort=[0.50],
        )
        last = _FakeTrajectoryPoint(
            9.0,
            positions=[1.1],
            velocities=[0.11],
            accelerations=[0.31],
            effort=[0.51],
        )
        original = [
            (
                point.time_from_start.to_sec(),
                list(point.positions),
                list(point.velocities),
                list(point.accelerations),
                list(point.effort),
            )
            for point in (first, last)
        ]
        trajectory = _FakeTrajectory([first, last])
        self.assertIs(
            enforce_minimum_trajectory_duration(trajectory, 8.0), trajectory
        )
        self.assertEqual(
            [
                (
                    point.time_from_start.to_sec(),
                    point.positions,
                    point.velocities,
                    point.accelerations,
                    point.effort,
                )
                for point in (first, last)
            ],
            original,
        )

    def test_minimum_trajectory_duration_stretches_every_waypoint(self):
        trajectory = _FakeTrajectory(
            [
                _FakeTrajectoryPoint(
                    0.0,
                    positions=[1.0],
                    velocities=[0.8],
                    accelerations=[0.16],
                    effort=[0.25],
                ),
                _FakeTrajectoryPoint(
                    0.5,
                    positions=[1.1],
                    velocities=[0.4],
                    accelerations=[0.08],
                    effort=[0.15],
                ),
                _FakeTrajectoryPoint(
                    2.0,
                    positions=[1.2],
                    velocities=[0.1],
                    accelerations=[0.02],
                    effort=[0.05],
                ),
            ]
        )
        enforce_minimum_trajectory_duration(trajectory, 8.0)
        points = trajectory.joint_trajectory.points
        self.assertEqual(
            [point.time_from_start.to_sec() for point in points],
            [0.0, 2.0, 8.0],
        )
        expected = [
            ([0.2], [0.01], [1.0], [0.25]),
            ([0.1], [0.005], [1.1], [0.15]),
            ([0.025], [0.00125], [1.2], [0.05]),
        ]
        for point, values in zip(points, expected):
            for actual, wanted in zip(point.velocities, values[0]):
                self.assertAlmostEqual(actual, wanted, places=12)
            for actual, wanted in zip(point.accelerations, values[1]):
                self.assertAlmostEqual(actual, wanted, places=12)
            self.assertEqual(point.positions, values[2])
            self.assertEqual(point.effort, values[3])

    def test_minimum_trajectory_duration_keeps_empty_derivatives_empty(self):
        point = _FakeTrajectoryPoint(2.0, positions=[1.0])
        trajectory = _FakeTrajectory([_FakeTrajectoryPoint(0.0), point])
        enforce_minimum_trajectory_duration(trajectory, 8.0)
        self.assertEqual(point.time_from_start.to_sec(), 8.0)
        self.assertEqual(point.velocities, [])
        self.assertEqual(point.accelerations, [])

    def test_minimum_trajectory_duration_rejects_invalid_inputs(self):
        trajectory = _FakeTrajectory([_FakeTrajectoryPoint(2.0)])
        for minimum_s in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=minimum_s):
                enforce_minimum_trajectory_duration(trajectory, minimum_s)
        for final_s in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=final_s):
                enforce_minimum_trajectory_duration(
                    _FakeTrajectory([_FakeTrajectoryPoint(final_s)]), 8.0
                )
        with self.assertRaises(ValueError):
            enforce_minimum_trajectory_duration(_FakeTrajectory([]), 8.0)

    def test_physical_grasp_config_sets_minimum_lift_place_duration(self):
        with open(
            os.path.join(PACKAGE, "config", "physical_grasp_demo.yaml")
        ) as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["minimum_lift_place_trajectory_duration_s"], 8.0
        )

    def test_multi_angle_course_contract(self):
        with open(
            os.path.join(PACKAGE, "config", "multi_angle_obstacle_course.yaml")
        ) as stream:
            course = yaml.safe_load(stream)
        self.assertTrue(validate_course_config(course, self.scene))
        self.assertGreaterEqual(len(course["waypoints"]), 5)
        self.assertGreaterEqual(course["minimum_collision_proven_direct_segments"], 3)

    def test_multi_angle_rviz_subscribes_all_evidence_topics(self):
        with open(
            os.path.join(PACKAGE, "config", "multi_angle_avoidance.rviz")
        ) as stream:
            config = yaml.safe_load(stream)
        displays = config["Visualization Manager"]["Displays"]
        topics = {
            display.get("Marker Topic")
            for display in displays
            if display.get("Class") == "rviz/MarkerArray"
        }
        self.assertEqual(
            topics,
            {
                "/handarm_sim_demo/course_waypoints",
                "/handarm_sim_demo/planned_ee_path",
                "/handarm_sim_demo/executed_ee_path",
            },
        )

    def test_multi_angle_course_rejects_false_clearance(self):
        with open(
            os.path.join(PACKAGE, "config", "multi_angle_obstacle_course.yaml")
        ) as stream:
            course = yaml.safe_load(stream)
        course["waypoints"][0]["position"][1] = 0.15
        with self.assertRaises(ValueError):
            validate_course_config(course, self.scene)

    def test_multi_angle_course_counts_only_distinct_orientations(self):
        with open(
            os.path.join(PACKAGE, "config", "multi_angle_obstacle_course.yaml")
        ) as stream:
            course = yaml.safe_load(stream)
        first = course["waypoints"][0]["orientation_xyzw"]
        second = course["waypoints"][1]["orientation_xyzw"]
        for index, waypoint in enumerate(course["waypoints"]):
            waypoint["orientation_xyzw"] = list(first if index % 2 == 0 else second)
        with self.assertRaises(ValueError):
            validate_course_config(course, self.scene)

    def test_periodic_home_target_never_escapes_finite_joint_bounds(self):
        self.assertAlmostEqual(
            nearest_equivalent_within_bounds(-0.70, 5.60, -2.79, 2.79),
            -0.70,
        )

    def test_periodic_home_target_uses_valid_nearest_equivalent(self):
        expected = -0.4533 + 2.0 * math.pi
        self.assertAlmostEqual(
            nearest_equivalent_within_bounds(-0.4533, 5.8, -6.98132, 6.98132),
            expected,
        )

    def test_periodic_home_target_rejects_impossible_bounds(self):
        with self.assertRaises(ValueError):
            nearest_equivalent_within_bounds(0.0, 0.0, 1.0, 2.0)

    def test_pick_joint_goal_uses_nearest_periodic_representation(self):
        result = canonicalize_periodic_joint_goal(
            {"joint_5": 0.2, "joint_6": 5.9},
            {"joint_5": 0.0, "joint_6": -0.45},
            {"joint_5": (-2.0, 2.0), "joint_6": (-6.98132, 6.98132)},
            {"joint_6"},
        )
        self.assertAlmostEqual(result["joint_5"], 0.2)
        self.assertAlmostEqual(result["joint_6"], 5.9 - 2.0 * math.pi)
        self.assertLess(abs(result["joint_6"] + 0.45), 0.1)


if __name__ == "__main__":
    unittest.main()
