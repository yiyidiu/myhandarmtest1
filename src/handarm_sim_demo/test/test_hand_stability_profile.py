#!/usr/bin/env python3

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = PACKAGE.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

from physical_hand_trajectory_action_server import validate_trajectory  # noqa: E402
from render_teleop_hand_urdf import (  # noqa: E402
    ACTIVE_JOINTS,
    ALL_HAND_JOINTS,
    MIMIC_JOINTS,
    render_physical_grasp_hand,
    render_rigid_transport_hand,
)
from validate_hand_transport_stability import (  # noqa: E402
    ACTIVE_HAND_JOINTS as VALIDATOR_ACTIVE_JOINTS,
    ALL_HAND_JOINTS as VALIDATOR_HAND_JOINTS,
    HAND_TARGET,
    MIMIC_SOURCE,
    evaluate,
)


URDF = (
    WORKSPACE_SRC
    / "abb120_moveit_config1"
    / "config"
    / "gazebo_handarm.urdf"
)


class _Duration:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds


def trajectory(names, points):
    return SimpleNamespace(
        joint_names=list(names),
        points=[
            SimpleNamespace(
                positions=list(positions),
                time_from_start=_Duration(seconds),
            )
            for seconds, positions in points
        ],
    )


class HandStabilityProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = URDF.read_text(encoding="utf-8")

    def test_physical_profile_has_one_owner_for_all_hand_joints(self):
        root = ET.fromstring(render_physical_grasp_hand(self.original))
        transmitted = {
            joint.get("name")
            for transmission in root.findall("transmission")
            for joint in [transmission.find("joint")]
            if joint is not None
        }
        self.assertFalse(set(ACTIVE_JOINTS) & transmitted)
        physical_plugins = [
            plugin
            for plugin in root.findall("./gazebo/plugin")
            if plugin.get("filename")
            == "libhandarm_stable_hand_spring_plugin.so"
        ]
        self.assertEqual(len(physical_plugins), 1)
        plugin = physical_plugins[0]
        expected_tuning = {
            "activeStiffness": "25.0 12.0 12.0 12.0",
            "activeDamping": "1.50 0.60 0.60 0.60",
            "activeMaxEffort": "3.0 0.60 0.60 0.60",
            "mimicStiffness": "25.0 10.0 10.0 10.0",
            "mimicDamping": "1.50 0.50 0.50 0.50",
            "mimicMaxEffort": "3.0 0.40 0.40 0.40",
        }
        self.assertEqual(
            {name: plugin.findtext(name) for name in expected_tuning},
            expected_tuning,
        )
        self.assertFalse(any(
            plugin.find("mimicJoint") is not None
            for plugin in root.findall("./gazebo/plugin")
        ))
        implicit = {
            gazebo.get("reference")
            for gazebo in root.findall("gazebo")
            if gazebo.findtext("implicitSpringDamper") == "true"
        }
        self.assertEqual(implicit, set(ALL_HAND_JOINTS))

    def test_physical_profile_softens_all_finger_contacts(self):
        root = ET.fromstring(render_physical_grasp_hand(self.original))
        contacts = {
            gazebo.get("reference"): (
                gazebo.findtext("kp"),
                gazebo.findtext("kd"),
                gazebo.findtext("maxVel"),
                gazebo.findtext("minDepth"),
            )
            for gazebo in root.findall("gazebo")
            if gazebo.find("kp") is not None
        }
        self.assertEqual(len(contacts), 8)
        self.assertTrue(all(
            values == ("100000.0", "100.0", "0.02", "0.001")
            for values in contacts.values()
        ))

    def test_rigid_rollback_removes_only_mimic_pid_selection(self):
        root = ET.fromstring(render_rigid_transport_hand(self.original))
        mimics = {
            plugin.findtext("mimicJoint"): plugin
            for plugin in root.findall("./gazebo/plugin")
            if plugin.findtext("mimicJoint") in MIMIC_JOINTS
        }
        self.assertEqual(set(mimics), set(MIMIC_JOINTS))
        self.assertTrue(all(plugin.find("hasPID") is None
                            for plugin in mimics.values()))

    def test_action_contract_rejects_bad_time_and_nonfinite_position(self):
        valid = trajectory(
            ACTIVE_JOINTS,
            [(1.0, [0.1, 0.2, 0.3, 0.4])],
        )
        self.assertEqual(validate_trajectory(valid, ACTIVE_JOINTS), "")
        bad_time = trajectory(
            ACTIVE_JOINTS,
            [(0.0, [0.1, 0.2, 0.3, 0.4])],
        )
        self.assertIn(
            "strictly increase", validate_trajectory(bad_time, ACTIVE_JOINTS)
        )
        bad_value = trajectory(
            ACTIVE_JOINTS,
            [(1.0, [0.1, float("nan"), 0.3, 0.4])],
        )
        self.assertIn(
            "non-finite", validate_trajectory(bad_value, ACTIVE_JOINTS)
        )

    def test_action_contract_requires_exact_unique_joint_set(self):
        duplicate = trajectory(
            ["f1j1", "f1j1", "f2j1", "f3j2"],
            [(1.0, [0.1, 0.2, 0.3, 0.4])],
        )
        self.assertIn(
            "duplicate", validate_trajectory(duplicate, ACTIVE_JOINTS)
        )
        missing = trajectory(
            ACTIVE_JOINTS[:-1],
            [(1.0, [0.1, 0.2, 0.3])],
        )
        self.assertIn("exactly", validate_trajectory(missing, ACTIVE_JOINTS))

    @staticmethod
    def _transport_samples(extra_f1_motion=0.0):
        targets = dict(zip(VALIDATOR_ACTIVE_JOINTS, HAND_TARGET))
        samples = []
        for index in range(25):
            phase = index / 24.0
            positions = {
                "joint_1": -0.30 + 0.60 * phase,
            }
            for joint in VALIDATOR_ACTIVE_JOINTS:
                positions[joint] = targets[joint]
            positions["f1j1"] += extra_f1_motion * phase
            for joint, source in MIMIC_SOURCE.items():
                positions[joint] = positions[source]
            samples.append({
                "elapsed_wall_s": 0.05 * index,
                "ros_time_s": 0.05 * index,
                "position": positions,
                "rate": {joint: 0.01 for joint in VALIDATOR_HAND_JOINTS},
            })
        return samples

    def test_transport_acceptance_requires_arm_excitation_and_quiet_fingers(self):
        result = evaluate(
            self._transport_samples(), "physical_grasp", 0.01, 0.20
        )
        self.assertTrue(result["passed"])
        self.assertGreater(result["joint_1_measured_range_rad"], 0.40)

    def test_transport_acceptance_rejects_fixed_target_finger_motion(self):
        result = evaluate(
            self._transport_samples(extra_f1_motion=0.02),
            "original",
            0.01,
            0.20,
        )
        self.assertFalse(result["passed"])
        self.assertIn("f1j1 position range", result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
