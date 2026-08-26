#!/usr/bin/env python3

import math
import pathlib
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from handarm_moveit_demo.egm_position_reference import (
    EgmPositionReferenceModel, collision_proximity_hold_required)


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

    def test_invalid_monitor_values_are_rejected(self):
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

    def test_hold_reference_stops_feedforward_without_capturing_sag(self):
        model = make_model()
        model.update_actual([0.0, 0.5])
        model.step(0.0)
        model.update_velocity([1.0, 0.0], 0.0)
        moved = model.step(0.02)
        model.hold_reference([-0.01, 0.48])
        held = model.step(0.04)
        np.testing.assert_allclose(held.reference, moved.reference)
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
        self.assertEqual(forwarded["input_source"], "$(arg input_source)")
        arguments = {
            item.get("name"): item.get("default") for item in root.findall("arg")}
        self.assertEqual(arguments["mapping_profile"], "current_linear")
        self.assertEqual(arguments["input_source"], "udp")

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

        strict_guard = nodes["full_robot_self_collision_guard"]
        self.assertEqual(
            strict_guard.get("if"), "$(arg legacy_moveit_servo_reference)")
        self.assertEqual(strict_guard.get("required"), "true")
        guard_parameters = {
            item.get("name"): item.get("value")
            for item in strict_guard.findall("param")}
        self.assertEqual(
            guard_parameters["raw_velocity_topic"],
            "/egm_position_reference/raw_joint_velocity")
        self.assertEqual(
            guard_parameters["safe_velocity_topic"],
            "/egm_position_reference/collision_checked_joint_velocity")
        self.assertEqual(
            guard_parameters["enable_swept_command_gate"], "true")
        self.assertAlmostEqual(
            float(guard_parameters["prediction_horizon_s"]), 0.08)
        self.assertEqual(
            int(guard_parameters["maximum_prediction_samples"]), 3)
        self.assertEqual(
            guard_parameters["hand_command_topic"],
            "/controller_gazebo_hand/command")
        guard_rosparams = {
            item.get("param"): yaml.safe_load(item.text)
            for item in strict_guard.findall("rosparam")}
        self.assertEqual(
            guard_rosparams["hand_reference_joint_names"],
            ["f1j1", "f1j2", "f2j1", "f3j2"])
        self.assertEqual(
            guard_rosparams["initial_hand_reference"],
            [0.051, 0.0317, 0.0227, 0.0363])
        adapter_parameters = {
            item.get("name"): item.get("value")
            for item in nodes["egm_position_reference_adapter.py"].findall("param")}
        self.assertEqual(
            adapter_parameters["raw_velocity_topic"],
            "/egm_position_reference/collision_checked_joint_velocity")

        direct_node = nodes["egm_cartesian_reference_adapter.py"]
        parameters = {
            item.get("name"): item for item in direct_node.findall("param")}
        self.assertEqual(
            parameters["latch_on_twist_timeout"].get("if"),
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
        include = root.find("include")
        forwarded = {item.get("name"): item.get("value")
                     for item in include.findall("arg")}
        self.assertEqual(
            forwarded["legacy_moveit_servo_reference"],
            "$(arg legacy_moveit_servo_reference)")
        self.assertEqual(
            forwarded["stable_position_reference_profile"],
            "$(arg stable_position_reference_profile)")
        self.assertEqual(forwarded["require_scene_ready"], "true")
        scene = next(node for node in root.findall("node")
                     if node.get("type") == "scene_manager.py")
        self.assertEqual(scene.get("required"), "true")

    def test_safe_hybrid_has_feedback_leash_and_fail_closed_monitors(self):
        root = ET.parse(str(
            PACKAGE / "launch/shared_teleop_egm_position_demo.launch")).getroot()
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
            float(parameters["hard_stop_collision_scale"]), 0.0)
        self.assertLessEqual(
            float(parameters["collision_scale_timeout_s"]), 0.25)
        self.assertLessEqual(
            float(parameters["retreat_authorization_timeout_s"]), 0.12)
        self.assertLessEqual(
            float(parameters["strict_command_safe_timeout_s"]), 0.25)
        self.assertEqual(
            parameters["strict_command_safe_topic"],
            "/full_robot_self_collision_guard/command_safe")

    def test_safety_hold_captures_feedback_once_then_freezes_reference(self):
        source = (PACKAGE / "scripts/egm_position_reference_adapter.py").read_text()
        hold_start = source.index(
            "if hard_safety_hold and self.latest_actual is not None:")
        hold_end = source.index("output = self.model.step(now)", hold_start)
        hold_block = source[hold_start:hold_end]
        self.assertIn("if not self.hard_safety_hold_active:", hold_block)
        self.assertEqual(
            hold_block.count(
                "self.model.synchronize_reference(self.latest_actual)"), 1)
        self.assertIn("self.model.hold_reference(self.latest_actual)", hold_block)
        self.assertIn("self.hard_safety_hold_active = True", hold_block)

    def test_strict_guard_uses_bounded_combined_arm_hand_prediction(self):
        source = (PACKAGE / "src/full_robot_self_collision_guard.cpp").read_text()
        self.assertIn('"prediction_horizon_s", prediction_horizon_s_, 0.40', source)
        self.assertIn("maximum_prediction_step_rad_, 0.01", source)
        self.assertIn("maximum_prediction_samples_, 256", source)
        self.assertIn('"STRICT_ARM_HAND_FUTURE"', source)
        self.assertNotIn('"MEASURED_COAST"', source)
        self.assertIn(
            "if (enable_swept_command_gate_ && !predictedVelocityIsSafe(",
            source)
        self.assertIn("positionsSatisfyBounds(candidate, 0.0", source)
        self.assertNotIn("candidate.satisfiesBounds(", source)

        acceptance = ET.parse(str(
            PACKAGE / "launch/egm_servo_safety_acceptance.launch")).getroot()
        guard = next(node for node in acceptance.findall("node")
                     if node.get("type") == "full_robot_self_collision_guard")
        parameters = {
            item.get("name"): item.get("value")
            for item in guard.findall("param")}
        self.assertEqual(parameters["enable_swept_command_gate"], "true")
        self.assertEqual(parameters["maximum_prediction_samples"], "256")

    def test_transient_joint_state_timeout_holds_without_latched_estop(self):
        source = (PACKAGE / "src/full_robot_self_collision_guard.cpp").read_text()
        timer_start = source.index("void statusTimer(const ros::TimerEvent&)")
        timer_end = source.index("ros::NodeHandle nh_", timer_start)
        timer = source[timer_start:timer_end]
        self.assertNotIn('latchFault("JOINT_STATE_TIMEOUT")', timer)
        self.assertIn("if (fault)", timer)
        self.assertIn("if (!safe || command_stale)", timer)
        self.assertIn(
            'publishCommandStatus(false, detail.empty() ?', timer)

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
        self.assertTrue(config["check_collisions"])
        self.assertGreaterEqual(config["collision_check_rate"], 50.0)
        self.assertLessEqual(config["lower_singularity_threshold"], 17.0)
        self.assertLessEqual(config["hard_stop_singularity_threshold"], 30.0)
        self.assertGreaterEqual(config["joint_limit_margin"], 0.08)

    def test_semantic_model_exempts_only_true_structural_neighbors(self):
        path = WORKSPACE_SRC / "abb120_moveit_config1/config/handarm.srdf"
        root = ET.parse(str(path)).getroot()
        disabled = root.findall("disable_collisions")
        reasons = [item.get("reason") for item in disabled]
        self.assertNotIn("Never", reasons)
        self.assertNotIn("Default", reasons)
        self.assertEqual(reasons.count("Adjacent"), 15)
        self.assertEqual(reasons.count("StructuralAdjacent"), 4)

    def test_complete_robot_has_collision_mesh_and_physics_backstop(self):
        path = WORKSPACE_SRC / "abb120_moveit_config1/config/gazebo_handarm.urdf"
        root = ET.parse(str(path)).getroot()
        palm = root.find("link[@name='handbase_link']/collision/geometry/mesh")
        self.assertIsNotNone(palm)
        self.assertIn("handbase_link_collision_8mm.STL", palm.get("filename"))
        protected = {
            item.get("reference") for item in root.findall("gazebo")
            if item.findtext("selfCollide") == "true"}
        self.assertEqual(protected, {
            "base_link", "link_1", "link_2", "link_3", "link_4",
            "link_5", "link_6", "handbase_link", "f1link1", "f1link2",
            "f1link3", "f2link1", "f2link2", "f3link1", "f3link2",
            "f3link3"})

    def test_egm_plant_enables_gravity_and_loads_finite_effort_pid(self):
        path = WORKSPACE_SRC / "abb120_moveit_config1/launch/gazebo_egm_position.launch"
        text = path.read_text()
        self.assertIn("gazebo_handarm.urdf", text)
        self.assertNotIn("gazebo_handarm_velocity.urdf", text)
        self.assertIn("gazebo_arm_egm_position_pid.yaml", text)
        self.assertIn("gazebo_hand_only_pid.yaml", text)
        self.assertNotIn("gazebo_hand_position_pid.yaml", text)

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
