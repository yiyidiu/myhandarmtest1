#!/usr/bin/env python3

import math
import pathlib
import subprocess
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from handarm_moveit_demo.egm_position_reference import (
    EgmPositionHoldGate, EgmPositionReferenceModel,
    collision_proximity_hold_required)


PACKAGE = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_SRC = PACKAGE.parent


def make_model(**overrides):
    values = dict(
        joint_names=["j1", "j2"],
        initial_reference=[0.0, 0.5],
        lower_limits=[-1.0, -1.0],
        upper_limits=[1.0, 1.0],
        maximum_velocity=[2.0, 2.0],
        maximum_acceleration=[100.0, 100.0],
        command_timeout_s=0.10,
        joint_limit_margin_rad=0.0,
        maximum_step_dt_s=0.02,
    )
    values.update(overrides)
    return EgmPositionReferenceModel(**values)


class CollisionProximityHoldTest(unittest.TestCase):
    def test_low_scale_holds_without_a_fresh_explicit_retreat(self):
        self.assertTrue(collision_proximity_hold_required(
            0.19, 0.20, False, float("inf"), 0.12))
        self.assertTrue(collision_proximity_hold_required(
            0.19, 0.20, True, 0.13, 0.12))

    def test_fresh_retreat_is_the_only_low_scale_exception(self):
        self.assertFalse(collision_proximity_hold_required(
            0.19, 0.20, True, 0.02, 0.12))
        self.assertFalse(collision_proximity_hold_required(
            0.21, 0.20, False, float("inf"), 0.12))

    def test_invalid_monitor_values_fail_validation(self):
        with self.assertRaises(ValueError):
            collision_proximity_hold_required(
                math.nan, 0.20, False, float("inf"), 0.12)


class EgmPositionReferenceModelTest(unittest.TestCase):
    def test_waits_for_feedback_before_output(self):
        model = make_model()
        self.assertIsNone(model.step(0.0))

    def test_holds_persistent_reference_when_actual_is_disturbed(self):
        model = make_model()
        model.update_actual([0.0, 0.5])
        initial = model.step(0.0)
        np.testing.assert_allclose(initial.reference, [0.0, 0.5])
        model.update_actual([-0.2, 0.3])
        disturbed = model.step(0.02)
        np.testing.assert_allclose(disturbed.reference, [0.0, 0.5])
        np.testing.assert_allclose(disturbed.following_error, [0.2, 0.2])
        self.assertFalse(disturbed.command_fresh)

    def test_integrates_latest_velocity_without_a_trajectory_queue(self):
        model = make_model()
        model.update_actual([0.0, 0.5])
        model.step(0.0)
        model.update_velocity([1.0, 0.0], 0.0)
        positive = model.step(0.02)
        self.assertAlmostEqual(positive.reference[0], 0.02)
        model.update_velocity([-1.0, 0.0], 0.02)
        negative = model.step(0.04)
        self.assertAlmostEqual(negative.reference[0], 0.0)
        self.assertLess(negative.feedforward_velocity[0], 0.0)

    def test_stale_velocity_ramps_to_zero_and_then_holds(self):
        model = make_model(maximum_acceleration=[10.0, 10.0])
        model.update_actual([0.0, 0.5])
        model.step(0.0)
        model.update_velocity([1.0, 0.0], 0.0)
        for tick in range(1, 6):
            moving = model.step(tick * 0.02)
        self.assertTrue(moving.command_fresh)
        for tick in range(6, 12):
            stopped = model.step(tick * 0.02)
        self.assertFalse(stopped.command_fresh)
        self.assertAlmostEqual(stopped.feedforward_velocity[0], 0.0)
        held_reference = stopped.reference.copy()
        for tick in range(12, 20):
            stopped = model.step(tick * 0.02)
        np.testing.assert_allclose(stopped.reference, held_reference)

    def test_reference_never_crosses_joint_margin(self):
        model = make_model(
            initial_reference=[0.94, 0.5], joint_limit_margin_rad=0.05)
        model.update_actual([0.94, 0.5])
        model.step(0.0)
        model.update_velocity([2.0, 0.0], 0.0)
        output = model.step(0.02)
        self.assertAlmostEqual(output.reference[0], 0.95)
        self.assertTrue(output.limit_clamped)
        self.assertAlmostEqual(output.feedforward_velocity[0], 0.0)

    def test_time_reset_clears_feedforward_without_reanchoring(self):
        model = make_model()
        model.update_actual([0.0, 0.5])
        model.step(1.0)
        model.update_velocity([1.0, 0.0], 1.0)
        moved = model.step(1.02)
        reset = model.step(0.10)
        self.assertTrue(reset.time_reset)
        self.assertAlmostEqual(reset.feedforward_velocity[0], 0.0)
        np.testing.assert_allclose(reset.reference, moved.reference)

    def test_rejects_nonfinite_command(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.update_velocity([math.nan, 0.0], 0.0)

    def test_synchronize_reference_is_bumpless_and_clears_old_motion(self):
        model = make_model()
        model.update_actual([0.0, 0.5])
        model.step(0.0)
        model.update_velocity([1.0, 0.0], 0.0)
        model.step(0.02)
        model.synchronize_reference([-0.2, 0.3])
        output = model.step(1.0)
        np.testing.assert_allclose(output.reference, [-0.2, 0.3])
        np.testing.assert_allclose(output.feedforward_velocity, [0.0, 0.0])
        self.assertFalse(output.command_fresh)

    def test_settled_hold_preserves_reference_instead_of_capturing_gravity_sag(self):
        model = make_model()
        model.update_actual([0.0, 0.5])
        model.step(0.0)
        model.update_velocity([1.0, 0.0], 0.0)
        moved = model.step(0.02)
        self.assertAlmostEqual(moved.reference[0], 0.02)
        model.hold_reference([-0.01, 0.48])
        held = model.step(0.04)
        self.assertAlmostEqual(held.reference[0], 0.02)
        self.assertAlmostEqual(held.reference[1], 0.5)
        np.testing.assert_allclose(held.feedforward_velocity, [0.0, 0.0])
        np.testing.assert_allclose(held.actual, [-0.01, 0.48])

    def test_wide_revolute_reference_reanchors_to_nearest_equivalent_turn(self):
        model = make_model(
            initial_reference=[0.0, 5.97],
            lower_limits=[-1.0, -7.0],
            upper_limits=[1.0, 7.0])
        model.update_actual([0.0, -0.36])
        output = model.step(0.0)
        self.assertLess(abs(output.following_error[1]), 0.05)
        self.assertAlmostEqual(
            output.reference[1], 5.97 - 2.0 * math.pi, places=8)

    def test_following_error_leash_prevents_open_loop_reference_runaway(self):
        model = make_model(maximum_following_error=[0.05, 0.10])
        model.update_actual([0.0, 0.5])
        model.step(0.0)
        model.update_velocity([2.0, 0.0], 0.0)
        model.step(0.02)
        limited = model.step(0.04)
        self.assertTrue(limited.following_error_clamped)
        self.assertAlmostEqual(limited.reference[0], 0.05)
        self.assertAlmostEqual(limited.following_error[0], 0.05)

        # The leash is feedback-relative, so it advances without a jump when
        # the position servo catches up.
        model.update_actual([0.03, 0.5])
        advanced = model.step(0.06)
        self.assertTrue(advanced.following_error_clamped)
        self.assertAlmostEqual(advanced.reference[0], 0.08)
        self.assertAlmostEqual(advanced.following_error[0], 0.05)

    def test_following_error_leash_requires_positive_finite_values(self):
        with self.assertRaises(ValueError):
            make_model(maximum_following_error=[0.0, 0.1])
        with self.assertRaises(ValueError):
            make_model(maximum_following_error=[math.inf, 0.1])


class EgmPositionHoldGateTest(unittest.TestCase):
    def make_gate(self, **overrides):
        values = dict(
            latch_on_twist_timeout=True,
            settled_hold_enabled=True,
            settled_delay_s=0.25,
            linear_enter_mps=0.008,
            angular_enter_radps=0.06,
            linear_release_mps=0.025,
            angular_release_radps=0.18,
        )
        values.update(overrides)
        return EgmPositionHoldGate(**values)

    def test_quiet_dwell_enters_hold_and_hysteresis_ignores_residual_noise(self):
        gate = self.make_gate()
        quiet = [0.004, 0.0, 0.0, 0.0, 0.03, 0.0]
        first = gate.update(0.0, True, False, quiet)
        almost = gate.update(0.24, True, False, quiet)
        entered = gate.update(0.25, True, False, quiet)
        self.assertFalse(first.active)
        self.assertFalse(almost.active)
        self.assertTrue(entered.active)
        self.assertTrue(entered.entered)
        self.assertEqual(entered.source, "SETTLED_TARGET")

        residual = gate.update(
            0.30, True, False,
            [0.020, 0.0, 0.0, 0.0, 0.12, 0.0])
        self.assertTrue(residual.active)
        self.assertFalse(residual.entered)

    def test_deliberate_translation_or_rotation_releases_settled_hold(self):
        gate = self.make_gate(settled_delay_s=0.0)
        entered = gate.update(0.0, True, False, np.zeros(6))
        self.assertTrue(entered.active)
        translated = gate.update(
            0.1, True, False, [0.026, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertFalse(translated.active)
        self.assertTrue(translated.released)

        gate.reset()
        gate.update(0.2, True, False, np.zeros(6))
        rotated = gate.update(
            0.3, True, False, [0.0, 0.0, 0.0, 0.0, 0.181, 0.0])
        self.assertFalse(rotated.active)
        self.assertTrue(rotated.released)

    def test_target_loss_and_timeout_have_priority(self):
        gate = self.make_gate(settled_hold_enabled=False)
        target_loss = gate.update(
            0.0, True, True, [0.5, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertTrue(target_loss.active)
        self.assertEqual(target_loss.source, "TARGET_LOSS")
        timeout = gate.update(0.1, False, False, np.zeros(6))
        self.assertTrue(timeout.active)
        self.assertEqual(timeout.source, "TWIST_TIMEOUT")
        tracking = gate.update(0.2, True, False, np.zeros(6))
        self.assertFalse(tracking.active)
        self.assertTrue(tracking.released)

    def test_hold_gate_rejects_invalid_threshold_order(self):
        with self.assertRaises(ValueError):
            self.make_gate(linear_enter_mps=0.03,
                           linear_release_mps=0.02)


class EgmPositionProfileWiringTest(unittest.TestCase):
    def test_established_velocity_entry_is_unchanged_and_independent(self):
        legacy = (PACKAGE / "launch/live_human_ground_gazebo_teleop.launch").read_text()
        self.assertIn("live_human_gazebo_teleop.launch", legacy)
        self.assertNotIn("egm_position", legacy)

    def test_new_entry_selects_only_the_egm_demo(self):
        path = PACKAGE / "launch/live_human_ground_gazebo_egm_teleop.launch"
        root = ET.parse(str(path)).getroot()
        includes = root.findall("include")
        self.assertEqual(len(includes), 1)
        self.assertIn("shared_teleop_egm_position_demo.launch", includes[0].get("file"))
        forwarded = {
            item.get("name"): item.get("value") for item in includes[0].findall("arg")}
        self.assertEqual(forwarded["mapping_profile"], "$(arg mapping_profile)")
        arguments = {
            item.get("name"): item.get("default")
            for item in root.findall("arg")}
        self.assertEqual(
            arguments["mapping_profile"],
            "camera_ground_axis_decoupled")

    def test_egm_demo_uses_new_adapter_and_keeps_explicit_legacy_fallback(self):
        path = PACKAGE / "launch/shared_teleop_egm_position_demo.launch"
        root = ET.parse(str(path)).getroot()
        include_files = [item.get("file") for item in root.findall("include")]
        self.assertTrue(any("gazebo_egm_position.launch" in item for item in include_files))
        servo_include = next(item for item in root.findall("include") if
                             "abbarm_servo_egm_position_gazebo.launch" in
                             item.get("file"))
        self.assertEqual(
            servo_include.get("if"), "$(arg legacy_moveit_servo_reference)")
        nodes = {item.get("type"): item for item in root.findall("node")}
        self.assertIn("egm_position_reference_adapter.py", nodes)
        self.assertIn("egm_cartesian_reference_adapter.py", nodes)
        self.assertEqual(
            nodes["egm_position_reference_adapter.py"].get("if"),
            "$(arg legacy_moveit_servo_reference)")
        self.assertEqual(
            nodes["egm_cartesian_reference_adapter.py"].get("unless"),
            "$(arg legacy_moveit_servo_reference)")
        self.assertEqual(nodes["egm_position_reference_adapter.py"].get("required"), "true")

        direct_node = nodes["egm_cartesian_reference_adapter.py"]
        parameters = {
            item.get("name"): item for item in direct_node.findall("param")}
        self.assertEqual(
            parameters["latch_on_twist_timeout"].get("if"),
            "$(arg stable_position_reference_profile)")
        self.assertEqual(
            parameters["synchronize_reference_on_start"].get("value"),
            "false")
        self.assertEqual(
            parameters["synchronize_reference_on_start"].get("if"),
            "$(arg stable_position_reference_profile)")
        self.assertEqual(
            parameters["position_hold_topic"].get("value"),
            "/shared_teleop/egm_position_hold")
        leash = next(item for item in direct_node.findall("rosparam")
                     if item.get("param") == "maximum_following_error_rad")
        self.assertEqual(
            [float(value) for value in leash.text.strip("[]").split(",")],
            [0.08, 0.08, 0.08, 0.12, 0.12, 0.16])

    def test_public_egm_entry_defaults_to_safe_hybrid_servo(self):
        path = PACKAGE / "launch/live_human_ground_gazebo_egm_teleop.launch"
        root = ET.parse(str(path)).getroot()
        arguments = {item.get("name"): item.get("default")
                     for item in root.findall("arg")}
        self.assertEqual(arguments["use_moveit_servo_safety"], "true")
        self.assertEqual(
            arguments["legacy_moveit_servo_reference"],
            "$(arg use_moveit_servo_safety)")
        self.assertEqual(
            arguments["stable_position_reference_profile"], "true")
        self.assertEqual(
            arguments["hand_stability_profile"], "physical_grasp")
        include = root.find("include")
        forwarded = {item.get("name"): item.get("value")
                     for item in include.findall("arg")}
        self.assertEqual(
            forwarded["legacy_moveit_servo_reference"],
            "$(arg legacy_moveit_servo_reference)")
        self.assertEqual(
            forwarded["stable_position_reference_profile"],
            "$(arg stable_position_reference_profile)")
        self.assertEqual(
            forwarded["hand_stability_profile"],
            "$(arg hand_stability_profile)")
        scene = next(node for node in root.findall("node")
                     if node.get("type") == "scene_manager.py")
        self.assertEqual(scene.get("required"), "true")

    def test_safe_hybrid_adapter_has_short_feedback_leash(self):
        path = PACKAGE / "launch/shared_teleop_egm_position_demo.launch"
        root = ET.parse(str(path)).getroot()
        node = next(item for item in root.findall("node")
                    if item.get("type") == "egm_position_reference_adapter.py")
        leash = next(item for item in node.findall("rosparam")
                     if item.get("param") == "maximum_following_error_rad")
        self.assertEqual(
            [float(value) for value in leash.text.strip("[]").split(",")],
            [0.04, 0.04, 0.04, 0.06, 0.06, 0.08])
        parameters = {
            item.get("name"): item.get("value")
            for item in node.findall("param")}
        self.assertAlmostEqual(
            float(parameters["hard_stop_collision_scale"]), 0.20)
        self.assertLessEqual(
            float(parameters["collision_scale_timeout_s"]), 0.25)
        self.assertLessEqual(
            float(parameters["retreat_authorization_timeout_s"]), 0.12)

    def test_stable_response_is_egm_opt_in_and_keeps_global_profile(self):
        path = PACKAGE / "launch/shared_teleop_core.launch"
        root = ET.parse(str(path)).getroot()
        arguments = {item.get("name"): item.get("default")
                     for item in root.findall("arg")}
        self.assertEqual(
            arguments["stable_position_reference_profile"], "false")
        nodes = {item.get("type"): item for item in root.findall("node")}
        trend = nodes["six_dof_trend_node.py"]
        trend_parameters = {
            item.get("name"): item for item in trend.findall("param")}
        expected = {
            "translation_error_gain_per_s": 3.0,
            "rotation_error_gain_per_s": 3.0,
            "translation_feedforward_gain": 0.20,
            "rotation_feedforward_gain": 0.15,
            "maximum_linear_velocity_mps": 0.30,
            "maximum_angular_velocity_radps": 2.0,
            "position_hold_after_target_loss_s": 0.40,
        }
        for name, value in expected.items():
            self.assertEqual(
                trend_parameters[name].get("if"),
                "$(arg stable_position_reference_profile)")
            self.assertAlmostEqual(
                float(trend_parameters[name].get("value")), value)
        output = nodes["moveit_servo_output_adapter.py"]
        output_parameters = {
            item.get("name"): item for item in output.findall("param")}
        self.assertAlmostEqual(
            float(output_parameters[
                "maximum_linear_acceleration_mps2"].get("value")), 1.5)
        self.assertAlmostEqual(
            float(output_parameters[
                "maximum_angular_acceleration_radps2"].get("value")), 10.0)

    def test_controller_is_latest_position_array_not_trajectory(self):
        path = WORKSPACE_SRC / "abb120_moveit_config1/config/ros_controllers_egm_position.yaml"
        config = yaml.safe_load(path.read_text())
        arm = config["abbarm_egm_position_controller"]
        self.assertEqual(
            arm["type"], "position_controllers/JointGroupPositionController")
        self.assertNotIn("constraints", arm)
        self.assertEqual(len(arm["joints"]), 6)

    def test_servo_output_is_velocity_feedforward_not_controller_topic(self):
        path = WORKSPACE_SRC / "abb120_moveit_config1/config/servo_abbarm_egm_position_gazebo.yaml"
        config = yaml.safe_load(path.read_text())
        self.assertEqual(
            config["command_out_topic"],
            "/egm_position_reference/raw_joint_velocity")
        self.assertTrue(config["publish_joint_velocities"])
        self.assertFalse(config["publish_joint_positions"])
        self.assertNotEqual(
            config["command_out_topic"],
            "/abbarm_egm_position_controller/command")

    def test_egm_plant_enables_gravity_and_loads_finite_effort_pid(self):
        path = WORKSPACE_SRC / "abb120_moveit_config1/launch/gazebo_egm_position.launch"
        text = path.read_text()
        self.assertIn("gazebo_handarm.urdf", text)
        self.assertNotIn("gazebo_handarm_velocity.urdf", text)
        self.assertIn("gazebo_arm_egm_position_pid.yaml", text)
        self.assertIn("gazebo_hand_only_pid.yaml", text)
        self.assertNotIn("gazebo_hand_position_pid.yaml", text)
        root = ET.parse(str(path)).getroot()
        arguments = {item.get("name"): item.get("default")
                     for item in root.findall("arg")}
        self.assertEqual(
            arguments["hand_stability_profile"], "physical_grasp")
        hand_pid = next(
            item for item in root.findall("rosparam")
            if "gazebo_hand_only_pid.yaml" in item.get("file", ""))
        self.assertEqual(
            hand_pid.get("if"),
            "$(eval arg('hand_stability_profile') == 'original')")
        action_node = next(
            item for item in root.findall("node")
            if item.get("type") == "physical_hand_trajectory_action_server.py")
        self.assertEqual(
            action_node.get("if"),
            "$(eval arg('hand_stability_profile') == 'physical_grasp')")

    def test_rigid_transport_renderer_is_reversible_and_removes_mimic_pid(self):
        script = (WORKSPACE_SRC /
                  "handarm_sim_demo/scripts/render_teleop_hand_urdf.py")
        source = (WORKSPACE_SRC /
                  "abb120_moveit_config1/config/gazebo_handarm.urdf")
        original = source.read_text()
        rollback = subprocess.check_output([
            str(script), "--input", str(source), "--profile", "original"],
            text=True)
        self.assertEqual(rollback, original)

        rendered = subprocess.check_output([
            str(script), "--input", str(source),
            "--profile", "rigid_transport"], text=True)
        root = ET.fromstring(rendered)
        expected_damping = {
            "f1j1": "0.05", "f3j1": "0.05",
            "f1j2": "0.04", "f2j1": "0.04", "f3j2": "0.04",
            "f1j3": "0.01", "f2j2": "0.01", "f3j3": "0.01",
        }
        for joint in root.findall("joint"):
            name = joint.get("name")
            if name in expected_damping:
                self.assertEqual(
                    joint.find("dynamics").get("damping"),
                    expected_damping[name])
        mimic_plugins = {
            plugin.findtext("mimicJoint"): plugin
            for plugin in root.findall("./gazebo/plugin")
            if plugin.find("mimicJoint") is not None}
        self.assertEqual(
            set(mimic_plugins), {"f3j1", "f1j3", "f2j2", "f3j3"})
        for plugin in mimic_plugins.values():
            self.assertIsNone(plugin.find("hasPID"))
            self.assertIsNone(plugin.find("maxVelocity"))

    def test_physical_grasp_renderer_has_one_finite_compliance_owner(self):
        script = (WORKSPACE_SRC /
                  "handarm_sim_demo/scripts/render_teleop_hand_urdf.py")
        source = (WORKSPACE_SRC /
                  "abb120_moveit_config1/config/gazebo_handarm.urdf")
        rendered = subprocess.check_output([
            str(script), "--input", str(source),
            "--profile", "physical_grasp"], text=True)
        root = ET.fromstring(rendered)
        active = {"f1j1", "f1j2", "f2j1", "f3j2"}
        transmissions = {
            item.find("joint").get("name")
            for item in root.findall("transmission")}
        self.assertTrue(active.isdisjoint(transmissions))
        self.assertEqual(
            [plugin for plugin in root.findall("./gazebo/plugin")
             if plugin.find("mimicJoint") is not None], [])
        implicit = {
            item.get("reference") for item in root.findall("gazebo")
            if item.findtext("implicitSpringDamper") == "true"}
        self.assertEqual(
            implicit,
            active | {"f3j1", "f1j3", "f2j2", "f3j3"})
        physical = next(
            plugin for plugin in root.findall("./gazebo/plugin")
            if plugin.get("name") == "stable_physical_grasp_hand")
        self.assertEqual(
            physical.get("filename"),
            "libhandarm_stable_hand_spring_plugin.so")
        self.assertEqual(physical.findtext("activeMaxEffort"),
                         "3.0 0.60 0.60 0.60")
        self.assertEqual(physical.findtext("mimicMaxEffort"),
                         "3.0 0.40 0.40 0.40")
        finger_surfaces = {
            item.get("reference"): item for item in root.findall("gazebo")
            if item.get("reference", "").startswith("f")
            and item.find("mu1") is not None}
        self.assertEqual(len(finger_surfaces), 8)
        for surface in finger_surfaces.values():
            self.assertEqual(float(surface.findtext("kp")), 100000.0)
            self.assertEqual(float(surface.findtext("kd")), 100.0)
            self.assertEqual(float(surface.findtext("maxVel")), 0.02)

    def test_egm_arm_pid_is_independent_bounded_and_gravity_capable(self):
        path = WORKSPACE_SRC / "handarm_sim_demo/config/gazebo_arm_egm_position_pid.yaml"
        config = yaml.safe_load(path.read_text())
        self.assertEqual(set(config), {
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"})
        for gains in config.values():
            self.assertGreater(gains["p"], 0.0)
            self.assertGreater(gains["d"], 0.0)
            self.assertGreater(gains["i"], 0.0)
            self.assertGreater(gains["i_clamp"], 0.0)
            self.assertTrue(gains["antiwindup"])


if __name__ == "__main__":
    unittest.main()
