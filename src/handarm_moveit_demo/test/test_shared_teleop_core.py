#!/usr/bin/env python3

import math
from pathlib import Path
import sys
import time
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from handarm_moveit_demo.shared_teleop_core import (  # noqa: E402
    AprilTagV3PoseContinuityFilter,
    CollisionRetreatGuard,
    CoordinateVelocityMapper,
    CameraRangeWorkspaceMapper,
    GESTURE_CLOSE,
    GESTURE_NONE,
    GestureIsolationGate,
    GroundSectorWorkspace,
    LatestCommandShaper,
    MinimumInterventionOrientationAssist,
    PoseSample,
    RelativePoseMapper,
    RelativePoseServoController,
    SixDofTrendEstimator,
    StationaryFeedforwardGate,
    SymmetricSideGraspProjector,
    compose_pose,
    confirmed_position_reference_hold,
    flange_pose_for_fixed_center,
    interpolate_pose_ray,
    quaternion_xyzw_to_matrix,
    rotation_distance,
    robot_output_allowed,
    select_nearest_candidate,
    side_grasp_candidates,
    so3_exp,
    so3_log,
    top_grasp_candidate,
    apply_ground_sector_workspace_boundary,
)


def sample(stamp, velocity, quaternion=None, confidence=None):
    velocity = np.asarray(velocity, dtype=float)
    rotation = (so3_exp(velocity[3:] * stamp) if quaternion is None else
                quaternion_xyzw_to_matrix(quaternion))
    return PoseSample(stamp, velocity[:3] * stamp, rotation,
                      np.ones(6) if confidence is None else confidence)


class AprilTagV3PoseContinuityTest(unittest.TestCase):
    def test_step_clamp_and_lowpass_match_v3_sender_constants(self):
        continuity = AprilTagV3PoseContinuityFilter(
            maximum_position_step_m=0.035,
            maximum_rotation_step_rad=math.radians(15.0),
            position_alpha=0.30,
            rotation_alpha=0.28,
        )
        continuity.update(PoseSample(
            0.0, np.zeros(3), np.eye(3), np.ones(6)))
        filtered = continuity.update(PoseSample(
            0.1, [0.20, 0.0, 0.0],
            so3_exp([0.0, math.radians(90.0), 0.0]), np.ones(6)))
        np.testing.assert_allclose(
            filtered.position, [0.035 * 0.30, 0.0, 0.0], atol=1.0e-12)
        self.assertAlmostEqual(
            rotation_distance(np.eye(3), filtered.rotation),
            math.radians(15.0) * 0.28, places=12)

    def test_invalid_frame_holds_last_filtered_mano_pose(self):
        continuity = AprilTagV3PoseContinuityFilter()
        initial = continuity.update(PoseSample(
            0.0, [0.01, -0.02, 0.50], so3_exp([0.1, 0.0, 0.0]),
            np.ones(6)))
        held = continuity.update(PoseSample(
            0.1, [0.40, 0.20, 0.10], so3_exp([1.0, 0.5, -0.2]),
            np.ones(6), valid=False))
        self.assertFalse(held.valid)
        np.testing.assert_allclose(held.position, initial.position)
        np.testing.assert_allclose(held.rotation, initial.rotation)

    def test_persistent_mano_flip_never_creeps_into_target(self):
        continuity = AprilTagV3PoseContinuityFilter(
            maximum_rotation_innovation_rad=math.radians(35.0))
        initial = continuity.update(PoseSample(
            0.0, np.zeros(3), np.eye(3), np.ones(6)))
        flipped = so3_exp([0.0, math.radians(105.0), 0.0])
        for index in range(1, 8):
            held = continuity.update(PoseSample(
                index * 0.1, [0.10, 0.0, 0.0], flipped, np.ones(6)))
            self.assertFalse(held.valid)
            self.assertEqual(
                continuity.last_reason, "ORIENTATION_INNOVATION_REJECTED")
            np.testing.assert_allclose(held.position, initial.position)
            np.testing.assert_allclose(held.rotation, initial.rotation)

    def test_adaptive_gain_smooths_rest_but_tracks_deliberate_motion(self):
        continuity = AprilTagV3PoseContinuityFilter(
            maximum_position_step_m=0.05,
            maximum_rotation_step_rad=math.radians(45.0),
            position_alpha=0.25, position_alpha_max=0.95,
            rotation_alpha=0.30, rotation_alpha_max=0.97,
            position_quiet_step_m=0.0025,
            position_responsive_step_m=0.015,
            rotation_quiet_step_rad=math.radians(1.2),
            rotation_responsive_step_rad=math.radians(8.0))
        continuity.update(PoseSample(
            0.0, np.zeros(3), np.eye(3), np.ones(6)))
        quiet = continuity.update(PoseSample(
            0.1, [0.001, 0.0, 0.0],
            so3_exp([math.radians(0.8), 0.0, 0.0]), np.ones(6)))
        self.assertAlmostEqual(continuity.last_position_alpha, 0.25)
        self.assertAlmostEqual(continuity.last_rotation_alpha, 0.30)
        self.assertAlmostEqual(quiet.position[0], 0.00025)

        moving = continuity.update(PoseSample(
            0.2, [0.030, 0.0, 0.0],
            so3_exp([math.radians(20.0), 0.0, 0.0]), np.ones(6)))
        self.assertAlmostEqual(continuity.last_position_alpha, 0.95)
        self.assertAlmostEqual(continuity.last_rotation_alpha, 0.97)
        self.assertGreater(moving.position[0], 0.028)

    def test_position_outlier_is_held_without_target_jump(self):
        continuity = AprilTagV3PoseContinuityFilter(
            maximum_position_innovation_m=0.08)
        initial = continuity.update(PoseSample(
            0.0, [0.0, 0.0, 0.55], np.eye(3), np.ones(6)))
        held = continuity.update(PoseSample(
            0.1, [0.0, 0.0, 0.80], np.eye(3), np.ones(6)))
        self.assertFalse(held.valid)
        self.assertEqual(
            continuity.last_reason, "POSITION_INNOVATION_REJECTED")
        np.testing.assert_allclose(held.position, initial.position)

    def test_large_innovation_toward_c_zero_is_clamped_but_never_locked(self):
        continuity = AprilTagV3PoseContinuityFilter(
            maximum_position_step_m=0.04,
            maximum_position_innovation_m=0.08,
            maximum_rotation_step_rad=math.radians(45.0),
            maximum_rotation_innovation_rad=math.radians(70.0),
            position_alpha=1.0, rotation_alpha=1.0)
        zero_position = np.zeros(3)
        zero_rotation = np.eye(3)
        continuity.update(PoseSample(
            0.0, zero_position, zero_rotation, np.ones(6)))
        # Reach a distant pose through individually valid observations.
        continuity.update(PoseSample(
            0.1, [0.07, 0.0, 0.0],
            so3_exp([math.radians(60.0), 0.0, 0.0]), np.ones(6)))
        continuity.update(PoseSample(
            0.2, [0.11, 0.0, 0.0],
            so3_exp([math.radians(90.0), 0.0, 0.0]), np.ones(6)))
        before_position = continuity.last_position.copy()
        before_rotation = continuity.last_rotation.copy()

        returned = continuity.update(
            PoseSample(0.3, zero_position, zero_rotation, np.ones(6)),
            return_reference_position=zero_position,
            return_reference_rotation=zero_rotation)
        self.assertTrue(returned.valid)
        self.assertEqual(
            continuity.last_reason, "C_ZERO_RETREAT_OVERRIDE")
        self.assertLess(
            np.linalg.norm(returned.position),
            np.linalg.norm(before_position))
        self.assertLess(
            rotation_distance(returned.rotation, zero_rotation),
            rotation_distance(before_rotation, zero_rotation))
        self.assertLessEqual(
            np.linalg.norm(returned.position - before_position),
            0.04 + 1.0e-12)
        self.assertLessEqual(
            rotation_distance(returned.rotation, before_rotation),
            math.radians(45.0) + 1.0e-12)

        # Repeated C-zero observations finish the retreat instead of holding
        # the stale side target forever.
        for index in range(4, 9):
            returned = continuity.update(
                PoseSample(
                    index / 10.0, zero_position, zero_rotation, np.ones(6)),
                return_reference_position=zero_position,
                return_reference_rotation=zero_rotation)
        np.testing.assert_allclose(returned.position, zero_position, atol=1.0e-12)
        np.testing.assert_allclose(returned.rotation, zero_rotation, atol=1.0e-12)


class CollisionRetreatGuardTest(unittest.TestCase):
    def test_guard_stops_deeper_motion_and_passes_c_zero_retreat(self):
        guard = CollisionRetreatGuard(
            enter_scale=0.20, release_scale=0.80,
            translation_progress_m=0.001,
            rotation_progress_rad=math.radians(1.0))
        robot_zero_position = np.zeros(3)
        robot_zero_rotation = np.eye(3)
        current_position = np.asarray([0.10, 0.0, 0.0])
        current_rotation = so3_exp([0.0, 0.0, 0.5])
        # +X and +Z rotation both move farther from the measured robot C-zero.
        deeper_command = np.asarray([0.02, 0.0, 0.0, 0.0, 0.0, 0.10])
        normal = guard.apply(
            1.0, current_position, current_rotation,
            robot_zero_position, robot_zero_rotation, deeper_command)
        self.assertFalse(normal.active)
        np.testing.assert_allclose(normal.velocity, deeper_command)

        blocked = guard.apply(
            0.19, current_position, current_rotation,
            robot_zero_position, robot_zero_rotation, deeper_command)
        self.assertTrue(blocked.active)
        np.testing.assert_allclose(blocked.velocity, np.zeros(6))

        # -X and -Z rotation reduce the measured error to C-zero directly.
        retreat_command = np.asarray([-0.02, 0.0, 0.0, 0.0, 0.0, -0.10])
        retreat = guard.apply(
            0.19, current_position, current_rotation,
            robot_zero_position, robot_zero_rotation, retreat_command)
        self.assertTrue(retreat.linear_retreat_allowed)
        self.assertTrue(retreat.angular_retreat_allowed)
        np.testing.assert_allclose(retreat.velocity, retreat_command)

        mixed_command = np.asarray([-0.02, 0.0, 0.0, 0.0, 0.0, 0.10])
        mixed = guard.apply(
            0.19, current_position, current_rotation,
            robot_zero_position, robot_zero_rotation, mixed_command)
        self.assertTrue(mixed.linear_retreat_allowed)
        self.assertFalse(mixed.angular_retreat_allowed)
        np.testing.assert_allclose(
            mixed.velocity, [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0])

        released = guard.apply(
            0.85, current_position, current_rotation,
            robot_zero_position, robot_zero_rotation, deeper_command)
        self.assertFalse(released.active)
        np.testing.assert_allclose(released.velocity, deeper_command)

    def test_guard_reports_servo_recovery_reason(self):
        guard = CollisionRetreatGuard()
        result = guard.apply(
            0.0, [0.1, 0.0, 0.0], np.eye(3), np.zeros(3), np.eye(3),
            [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            active_reason="SERVO_STATUS_5_RETURN_TOWARD_C_ZERO")
        self.assertTrue(result.active)
        self.assertEqual(
            result.reason, "SERVO_STATUS_5_RETURN_TOWARD_C_ZERO")


class SymmetricSideGraspProjectorTest(unittest.TestCase):
    def make_projector(self):
        return SymmetricSideGraspProjector(
            enabled=True, axis="x",
            blend_start_rad=math.radians(30.0),
            blend_full_rad=math.radians(55.0),
            dominance_start_ratio=0.90,
            dominance_full_ratio=1.15)

    def test_dominant_local_x_keeps_angle_and_removes_coupled_axes(self):
        projector = self.make_projector()
        raw_vector = np.radians([90.0, 24.0, -12.0])
        result = projector.project(so3_exp(raw_vector))
        self.assertTrue(result.active)
        self.assertAlmostEqual(result.weight, 1.0)
        np.testing.assert_allclose(
            result.projected_rotation_vector,
            np.radians([90.0, 0.0, 0.0]), atol=1.0e-10)
        np.testing.assert_allclose(
            projector.project_local_angular_velocity(
                [1.2, -0.4, 0.3], result),
            [1.2, 0.0, 0.0], atol=1.0e-12)

    def test_positive_and_negative_side_targets_are_exact_inverses(self):
        projector = self.make_projector()
        raw_vector = np.radians([82.0, 19.0, -8.0])
        positive = projector.project(so3_exp(raw_vector))
        negative = projector.project(so3_exp(-raw_vector))
        np.testing.assert_allclose(
            negative.projected_rotation_vector,
            -positive.projected_rotation_vector, atol=1.0e-10)
        np.testing.assert_allclose(
            negative.rotation, positive.rotation.T, atol=1.0e-10)
        self.assertEqual(positive.side_sign, 1)
        self.assertEqual(negative.side_sign, -1)

    def test_yaw_dominant_motion_is_not_modified(self):
        projector = self.make_projector()
        raw_vector = np.radians([20.0, 5.0, 65.0])
        result = projector.project(so3_exp(raw_vector))
        self.assertFalse(result.active)
        self.assertEqual(result.weight, 0.0)
        np.testing.assert_allclose(
            result.projected_rotation_vector, raw_vector, atol=1.0e-10)


class StationaryFeedforwardGateTest(unittest.TestCase):
    def test_stationary_derivative_is_zero_but_deliberate_motion_is_unchanged(self):
        gate = StationaryFeedforwardGate(
            linear_quiet_mps=0.012, linear_full_mps=0.040,
            angular_quiet_radps=0.18, angular_full_radps=0.60)
        quiet = gate.apply([0.008, 0.0, 0.0, 0.0, 0.10, 0.0])
        np.testing.assert_allclose(quiet.velocity, np.zeros(6))
        self.assertEqual(quiet.linear_weight, 0.0)
        self.assertEqual(quiet.angular_weight, 0.0)

        moving = gate.apply([0.08, 0.0, 0.0, 0.0, 1.20, 0.0])
        np.testing.assert_allclose(
            moving.velocity, [0.08, 0.0, 0.0, 0.0, 1.20, 0.0])
        self.assertEqual(moving.linear_weight, 1.0)
        self.assertEqual(moving.angular_weight, 1.0)


class RobotDescriptionLimitTest(unittest.TestCase):
    def test_joint_6_uses_official_plus_minus_400_degree_range_everywhere(self):
        sources = [
            PACKAGE.parent / "handarmtest1" / "xacro" / "arm_macro.xacro",
            PACKAGE.parent / "handarmtest1" / "urdf" / "arm.urdf",
            PACKAGE.parent / "abb120_moveit_config1" / "config" /
            "gazebo_handarm.urdf",
            PACKAGE.parent / "abb120_moveit_config1" / "config" /
            "gazebo_handarm_velocity.urdf",
        ]
        expected = math.radians(400.0)
        for source in sources:
            root = ET.parse(str(source)).getroot()
            matches = [
                joint for joint in root.iter("joint")
                if str(joint.get("name", "")).endswith("joint_6") and
                joint.get("type") == "revolute"]
            self.assertEqual(len(matches), 1, str(source))
            limit = matches[0].find("limit")
            self.assertIsNotNone(limit, str(source))
            self.assertAlmostEqual(
                float(limit.get("lower")), -expected, places=5,
                msg=str(source))
            self.assertAlmostEqual(
                float(limit.get("upper")), expected, places=5,
                msg=str(source))


class SixDofCommandTest(unittest.TestCase):
    def make_estimator(self):
        return SixDofTrendEstimator(
            window_size=4,
            translation_deadband_m=(0.0, 0.0, 0.0),
            rotation_deadband_rad=(0.0, 0.0, 0.0),
            smoothing_alpha=1.0,
        )

    def test_all_translation_and_rotation_axes_can_coexist(self):
        velocity = np.array([0.03, -0.02, 0.015, 0.20, -0.15, 0.10])
        estimator = self.make_estimator()
        result = None
        for index in range(4):
            result = estimator.update(sample(index / 30.0, velocity))
        self.assertTrue(result.valid)
        self.assertTrue(np.all(np.abs(result.raw_velocity) > 1.0e-4))
        np.testing.assert_allclose(result.raw_velocity, velocity, atol=2.0e-3)

    def test_non_dominant_axes_are_not_zeroed(self):
        mapper = CoordinateVelocityMapper(np.eye(3), np.eye(3),
                                          np.ones(3), np.ones(3),
                                          np.ones(3), np.ones(3))
        result = mapper.map([0.8, 0.04, -0.02, 0.5, -0.03, 0.01], np.ones(6))
        self.assertEqual(np.count_nonzero(result), 6)

    def test_quaternion_sign_flip_is_same_orientation(self):
        q = np.array([0.2, -0.1, 0.3, 0.92]); q /= np.linalg.norm(q)
        estimator = self.make_estimator()
        estimator.update(sample(0.0, np.zeros(6), q))
        result = estimator.update(sample(1.0 / 30.0, np.zeros(6), -q))
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.raw_velocity[3:], 0.0, atol=1.0e-10)

    def test_relative_rotation_is_expressed_in_c_zero_local_frame(self):
        estimator = self.make_estimator()
        hand_zero = so3_exp([0.30, -0.20, 0.10])
        hand_delta = so3_exp([0.08, 0.04, -0.03])
        estimator.update(PoseSample(
            0.0, np.zeros(3), hand_zero, np.ones(6)))
        result = estimator.update(PoseSample(
            1.0 / 30.0, np.zeros(3), hand_zero @ hand_delta,
            np.ones(6)))
        np.testing.assert_allclose(
            result.relative_rotation, hand_delta, atol=1.0e-10)

    def test_pose_jump_is_rejected(self):
        estimator = self.make_estimator()
        estimator.update(sample(0.0, np.zeros(6)))
        normal = estimator.update(sample(1.0 / 30.0, [0.02, 0, 0, 0, 0, 0]))
        jumped = PoseSample(2.0 / 30.0, np.array([0.5, 0, 0]), np.eye(3), np.ones(6))
        result = estimator.update(jumped)
        self.assertTrue(normal.valid)
        self.assertFalse(result.valid)
        self.assertIn("JUMP", result.reason)

    def test_tracking_gap_recovery_never_pays_back_unobserved_motion(self):
        estimator = self.make_estimator()
        estimator.update(sample(0.0, np.zeros(6)))
        estimator.update(sample(1.0 / 30.0, [0.03, 0, 0, 0, 0, 0]))
        invalid = PoseSample(
            2.0 / 30.0, np.asarray([0.002, 0.0, 0.0]), np.eye(3),
            np.ones(6), valid=False,
        )
        stopped = estimator.update(invalid)
        recovered = PoseSample(
            3.0 / 30.0, np.asarray([0.20, 0.0, 0.0]), np.eye(3),
            np.ones(6), valid=True,
        )
        first_after_gap = estimator.update(recovered)
        self.assertFalse(stopped.valid)
        np.testing.assert_allclose(stopped.raw_velocity, 0.0)
        self.assertTrue(first_after_gap.valid)
        self.assertEqual(
            first_after_gap.reason, "TRACKING_REACQUIRED_NO_PAYBACK"
        )
        np.testing.assert_allclose(first_after_gap.raw_velocity, 0.0)
        # The explicit user zero remains fixed even though the derivative
        # window was rebuilt at the recovered pose.
        np.testing.assert_allclose(first_after_gap.relative_position[0], 0.20)

    def test_long_timestamp_gap_reanchors_derivative_window(self):
        estimator = self.make_estimator()
        estimator.update(sample(0.0, np.zeros(6)))
        estimator.update(sample(1.0 / 30.0, [0.03, 0, 0, 0, 0, 0]))
        after_gap = PoseSample(
            0.50, np.asarray([0.25, 0.0, 0.0]), np.eye(3), np.ones(6)
        )
        result = estimator.update(after_gap)
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "TIMING_GAP_REANCHORED_NO_PAYBACK")
        np.testing.assert_allclose(result.raw_velocity, 0.0)

    def test_timeout_reaches_zero_by_deadline(self):
        shaper = LatestCommandShaper(
            [0.1, 0.1, 0.1, 0.6, 0.6, 0.6],
            [2.0, 2.0, 2.0, 12.0, 12.0, 12.0], 0.09, 0.15)
        shaper.update([0.1, -0.05, 0.02, 0.6, 0.2, -0.1], 0.0, True)
        for index in range(1, 9):
            result = shaper.tick(index * 0.02)
        np.testing.assert_allclose(result.velocity, 0.0, atol=1.0e-12)
        self.assertEqual(result.reason, "INPUT_TIMEOUT_ZERO")

    def test_future_clock_domain_is_rejected(self):
        shaper = LatestCommandShaper(
            [0.08, 0.08, 0.08, 0.5, 0.5, 0.5],
            [1.6, 1.6, 1.6, 10.0, 10.0, 10.0], 0.09, 0.15)
        shaper.update([0.02] * 6, 1000.0, True)
        result = shaper.tick(3.0)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "INPUT_CLOCK_MISMATCH")
        np.testing.assert_allclose(result.velocity, 0.0, atol=1.0e-12)

    def test_confidence_scales_each_axis_independently(self):
        mapper = CoordinateVelocityMapper(np.eye(3), np.eye(3),
                                          np.ones(3), np.ones(3),
                                          np.ones(3), np.ones(3))
        result = mapper.map(np.ones(6) * 0.1,
                            [0.05, 1.0, 0.2, 1.0, 0.1, 0.5])
        np.testing.assert_allclose(result, [0.005, 0.1, 0.02, 0.1, 0.01, 0.05])

    def test_confidence_follows_axis_mapping(self):
        matrix = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        mapper = CoordinateVelocityMapper(matrix, matrix, np.ones(3), np.ones(3),
                                          np.ones(3), np.ones(3))
        confidence = mapper.map_confidence([0.1, 0.2, 0.9, 0.3, 0.4, 0.8])
        np.testing.assert_allclose(confidence, [0.9, 0.1, 0.2, 0.8, 0.3, 0.4])

    def test_translation_mapping_can_intentionally_reflect_left_right(self):
        translation = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]
        mapper = CoordinateVelocityMapper(
            translation, np.eye(3), np.ones(3), np.ones(3),
            np.ones(3), np.ones(3))
        image_right = mapper.map([0.1, 0, 0, 0, 0, 0], np.ones(6))
        image_left = mapper.map([-0.1, 0, 0, 0, 0, 0], np.ones(6))
        np.testing.assert_allclose(image_right[:3], [0, 0.1, 0])
        np.testing.assert_allclose(image_left[:3], [0, -0.1, 0])

    def test_relative_pose_uses_apriltag_v3_translation_relation(self):
        mapper = RelativePoseMapper(
            [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
            np.eye(3), [0.6, 0.6, 0.6], [1.0, 1.0, 1.0],
            [0.25, 0.25, 0.25], math.radians(140.0))
        robot_zero_position = np.array([0.31, 0.01, 0.47])
        robot_zero_rotation = so3_exp([0.1, -0.2, 0.05])
        target_position, target_rotation = mapper.map(
            [0, 0, 0], np.eye(3), robot_zero_position,
            robot_zero_rotation)
        np.testing.assert_allclose(target_position, robot_zero_position)
        np.testing.assert_allclose(target_rotation, robot_zero_rotation)

        image_right, _ = mapper.map(
            [0.05, 0, 0], np.eye(3), robot_zero_position,
            robot_zero_rotation)
        image_up, _ = mapper.map(
            [0, -0.05, 0], np.eye(3), robot_zero_position,
            robot_zero_rotation)
        toward_camera, _ = mapper.map(
            [0, 0, -0.05], np.eye(3), robot_zero_position,
            robot_zero_rotation)
        np.testing.assert_allclose(
            image_right-robot_zero_position, [0, -0.03, 0])
        np.testing.assert_allclose(
            image_up-robot_zero_position, [0, 0, 0.03])
        np.testing.assert_allclose(
            toward_camera-robot_zero_position, [0.03, 0, 0])

    def test_relative_pose_uses_apriltag_v3_one_to_one_local_rotation(self):
        mapper = RelativePoseMapper(
            [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
            np.eye(3), [0.6, 0.6, 0.6], [1.0, 1.0, 1.0],
            [0.25, 0.25, 0.25], math.radians(140.0))
        hand_rotation = so3_exp([0, 0, math.radians(20.0)])
        robot_zero_rotation = so3_exp([0.3, -0.2, 0.1])
        _, target_rotation = mapper.map(
            [0, 0, 0], hand_rotation, [0, 0, 0],
            robot_zero_rotation)
        np.testing.assert_allclose(
            robot_zero_rotation.T @ target_rotation,
            hand_rotation, atol=1.0e-9)
        self.assertAlmostEqual(
            np.linalg.norm(so3_log(robot_zero_rotation.T @ target_rotation)),
            math.radians(20.0), places=9)

    def test_camera_range_maps_each_human_extreme_to_ground_sector_boundary(self):
        workspace = GroundSectorWorkspace(
            [0.0, 0.0, 0.5], [1.0, 1.0, 1.0],
            minimum_forward_x_m=0.0, minimum_tool_z_m=0.0)
        mapper = CameraRangeWorkspaceMapper(
            [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
            np.eye(3), [1.0, 1.0, 1.0], math.radians(179.0),
            [0.10, 0.20, 0.30], [0.10, 0.20, 0.30], workspace)
        zero = np.array([0.25, 0.0, 0.5])

        image_right, _ = mapper.map(
            [0.10, 0.0, 0.0], np.eye(3), zero, np.eye(3))
        np.testing.assert_allclose(
            image_right, [0.25, -math.sqrt(1.0 - 0.25 ** 2), 0.5],
            atol=1.0e-9)

        image_down, _ = mapper.map(
            [0.0, 0.20, 0.0], np.eye(3), zero, np.eye(3))
        np.testing.assert_allclose(image_down, [0.25, 0.0, 0.0])

        toward_camera, _ = mapper.map(
            [0.0, 0.0, -0.30], np.eye(3), zero, np.eye(3))
        np.testing.assert_allclose(toward_camera, [1.0, 0.0, 0.5])
        self.assertTrue(workspace.contains(toward_camera))

    def test_camera_range_saturates_beyond_visible_boundary_and_returns_zero(self):
        workspace = GroundSectorWorkspace(
            [0.0, 0.0, 0.5], [1.0, 1.0, 1.0], 0.0, 0.0)
        mapper = CameraRangeWorkspaceMapper(
            np.eye(3), np.eye(3), np.ones(3), math.radians(179.0),
            [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], workspace)
        zero = np.array([0.25, 0.0, 0.5])
        boundary, _ = mapper.map(
            [0.2, 0.0, 0.0], np.eye(3), zero, np.eye(3))
        beyond, _ = mapper.map(
            [0.5, 0.0, 0.0], np.eye(3), zero, np.eye(3))
        returned, returned_rotation = mapper.map(
            [0.0, 0.0, 0.0], np.eye(3), zero, np.eye(3))
        np.testing.assert_allclose(boundary, beyond)
        np.testing.assert_allclose(returned, zero)
        np.testing.assert_allclose(returned_rotation, np.eye(3))
        self.assertTrue(mapper.mapping_diagnostics()["translation_saturated"] is False)

    def test_camera_range_keeps_one_to_one_so3_rotation(self):
        workspace = GroundSectorWorkspace(
            [0.0, 0.0, 0.5], [1.0, 1.0, 1.0], 0.0, 0.0)
        mapper = CameraRangeWorkspaceMapper(
            np.eye(3), np.eye(3), np.ones(3), math.radians(179.0),
            [0.2] * 3, [0.2] * 3, workspace)
        relative = so3_exp([math.radians(35.0), 0.0, 0.0])
        zero_rotation = so3_exp([0.0, 0.2, 0.0])
        _, target = mapper.map(
            [0.0, 0.0, 0.0], relative,
            [0.25, 0.0, 0.5], zero_rotation)
        np.testing.assert_allclose(
            zero_rotation.T @ target, relative, atol=1.0e-9)

    def test_normalized_camera_pose_maps_asymmetric_orientation_extents(self):
        workspace = GroundSectorWorkspace(
            [0.0, 0.0, 0.5], [1.0, 1.0, 1.0], 0.0, 0.0)
        mapper = CameraRangeWorkspaceMapper(
            np.eye(3), np.eye(3), np.ones(3), math.radians(179.0),
            [0.1] * 3, [0.1] * 3, workspace, 1.0,
            np.radians([90.0] * 3), np.radians([90.0] * 3),
            np.radians([60.0, 60.0, 120.0]),
            np.radians([60.0, 30.0, 120.0]), True)
        zero_position = np.array([0.25, 0.0, 0.5])
        zero_rotation = so3_exp([0.1, -0.2, 0.05])

        _, positive = mapper.map(
            np.zeros(3), so3_exp([0.0, math.radians(90.0), 0.0]),
            zero_position, zero_rotation)
        _, negative = mapper.map(
            np.zeros(3), so3_exp([0.0, math.radians(-90.0), 0.0]),
            zero_position, zero_rotation)
        positive_vector = so3_log(zero_rotation.T @ positive)
        negative_vector = so3_log(zero_rotation.T @ negative)
        np.testing.assert_allclose(
            positive_vector, [0.0, math.radians(30.0), 0.0], atol=1.0e-9)
        np.testing.assert_allclose(
            negative_vector, [0.0, math.radians(-60.0), 0.0], atol=1.0e-9)

    def test_normalized_camera_pose_shares_radius_for_mixed_6d_extremes(self):
        workspace = GroundSectorWorkspace(
            [0.0, 0.0, 0.5], [1.0, 1.0, 1.0], 0.0, 0.0)
        mapper = CameraRangeWorkspaceMapper(
            np.eye(3), np.eye(3), np.ones(3), math.radians(179.0),
            [0.1] * 3, [0.1] * 3, workspace, 1.0,
            np.radians([90.0] * 3), np.radians([90.0] * 3),
            np.radians([60.0] * 3), np.radians([60.0] * 3), True)
        zero = np.array([0.25, 0.0, 0.5])
        target_position, target_rotation = mapper.map(
            [0.1, 0.0, 0.0],
            so3_exp([0.0, math.radians(90.0), 0.0]),
            zero, np.eye(3))
        share = 1.0 / math.sqrt(2.0)
        np.testing.assert_allclose(
            target_position, zero + [0.75 * share, 0.0, 0.0], atol=1.0e-9)
        np.testing.assert_allclose(
            so3_log(target_rotation),
            [0.0, math.radians(60.0) * share, 0.0], atol=1.0e-9)
        boundary_position, boundary_rotation, fraction, direction = (
            mapper.reachability_boundary())
        np.testing.assert_allclose(boundary_position, target_position)
        np.testing.assert_allclose(boundary_rotation, target_rotation)
        self.assertAlmostEqual(fraction, 1.0)
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0)
        self.assertAlmostEqual(
            mapper.mapping_diagnostics()["human_pose_fraction"],
            math.sqrt(2.0))

    def test_independent_camera_pose_keeps_full_position_and_y_rotation(self):
        workspace = GroundSectorWorkspace(
            [0.0, 0.0, 0.5], [1.0, 1.0, 1.0], 0.0, 0.0)
        mapper = CameraRangeWorkspaceMapper(
            np.eye(3), np.eye(3), np.ones(3), math.radians(179.0),
            [0.1] * 3, [0.1] * 3, workspace, 1.0,
            np.radians([90.0] * 3), np.radians([90.0] * 3),
            np.radians([60.0, 76.2, 120.0]),
            np.radians([60.0, 30.8, 120.0]), False)
        zero = np.array([0.25, 0.0, 0.5])

        target_position, target_rotation = mapper.map(
            [0.1, 0.0, 0.0],
            so3_exp([0.0, math.radians(20.0), 0.0]),
            zero, np.eye(3))
        np.testing.assert_allclose(
            target_position, [1.0, 0.0, 0.5], atol=1.0e-9)
        np.testing.assert_allclose(
            so3_log(target_rotation),
            [0.0, math.radians(20.0), 0.0], atol=1.0e-9)

        _, positive_limit = mapper.map(
            np.zeros(3), so3_exp([0.0, math.radians(90.0), 0.0]),
            zero, np.eye(3))
        _, negative_inside = mapper.map(
            np.zeros(3), so3_exp([0.0, math.radians(-50.0), 0.0]),
            zero, np.eye(3))
        np.testing.assert_allclose(
            so3_log(positive_limit),
            [0.0, math.radians(30.8), 0.0], atol=1.0e-9)
        np.testing.assert_allclose(
            so3_log(negative_inside),
            [0.0, math.radians(-50.0), 0.0], atol=1.0e-9)
        diagnostics = mapper.mapping_diagnostics()
        self.assertEqual(
            diagnostics["orientation_mapping"],
            "INDEPENDENT_1_TO_1_DIRECTIONAL_CAP")

    def test_pose_ray_interpolation_preserves_axis_and_zero_return(self):
        start_position = np.array([0.3, 0.0, 0.4])
        start_rotation = so3_exp([0.2, -0.1, 0.05])
        end_position = np.array([0.6, -0.2, 0.1])
        end_rotation = start_rotation @ so3_exp([0.8, 0.0, 0.0])
        returned_position, returned_rotation = interpolate_pose_ray(
            start_position, start_rotation, end_position, end_rotation, 0.0)
        np.testing.assert_allclose(returned_position, start_position)
        np.testing.assert_allclose(returned_rotation, start_rotation)
        middle_position, middle_rotation = interpolate_pose_ray(
            start_position, start_rotation, end_position, end_rotation, 0.5)
        np.testing.assert_allclose(
            middle_position, 0.5 * (start_position + end_position))
        np.testing.assert_allclose(
            so3_log(start_rotation.T @ middle_rotation), [0.4, 0.0, 0.0],
            atol=1.0e-9)

    def test_ground_sector_limiter_blocks_outward_but_allows_retreat(self):
        workspace = GroundSectorWorkspace(
            [0.0, 0.0, 0.5], [1.0, 1.0, 1.0], 0.0, 0.0)
        blocked, reasons = apply_ground_sector_workspace_boundary(
            [0.25, 0.0, 0.0], [0.0, 0.0, -0.2], workspace, 0.05)
        np.testing.assert_allclose(blocked, [0.0, 0.0, 0.0])
        self.assertIn("WORKSPACE_HARD_GROUND", reasons)
        retreat, retreat_reasons = apply_ground_sector_workspace_boundary(
            [0.25, 0.0, 0.0], [0.0, 0.0, 0.2], workspace, 0.05)
        np.testing.assert_allclose(retreat, [0.0, 0.0, 0.2])
        self.assertNotIn("WORKSPACE_HARD_GROUND", retreat_reasons)

    def test_apriltag_v3_feedforward_and_feedback_speed_relation(self):
        controller = RelativePoseServoController(
            [0.8] * 3, [0.5] * 3, [0.04] * 3, [0.25] * 3,
            maximum_linear_speed_norm=0.04,
            maximum_angular_speed_norm=0.25,
            translation_feedforward_gain=[0.8] * 3,
            rotation_feedforward_gain=[0.5] * 3,
        )
        command = controller.command(
            np.zeros(3), np.eye(3), np.zeros(3), np.eye(3),
            [0.02, -0.01, 0.0, 0.20, 0.0, -0.10])
        np.testing.assert_allclose(
            command, [0.016, -0.008, 0.0, 0.10, 0.0, -0.05])

    def test_relative_pose_servo_drives_back_to_captured_zero(self):
        controller = RelativePoseServoController(
            [4, 4, 4], [5, 5, 5], [0.08]*3, [1.5]*3)
        command = controller.command(
            [0.31, 0.06, 0.47], so3_exp([0.20, 0, 0]),
            [0.31, 0.01, 0.47], np.eye(3))
        np.testing.assert_allclose(command[:3], [0, -0.08, 0])
        np.testing.assert_allclose(command[3:], [-1.0, 0, 0], atol=1.0e-9)

    def test_relative_pose_servo_limits_combined_xyz_and_rpy_norms(self):
        controller = RelativePoseServoController(
            [4, 4, 4], [5, 5, 5], [0.08]*3, [1.5]*3,
            maximum_linear_speed_norm=0.10,
            maximum_angular_speed_norm=1.20,
        )
        command = controller.command(
            [0, 0, 0], np.eye(3), [1, 1, 1],
            so3_exp([1, 1, 1]))
        self.assertLessEqual(np.linalg.norm(command[:3]), 0.10+1.0e-12)
        self.assertLessEqual(np.linalg.norm(command[3:]), 1.20+1.0e-12)


class OrientationAssistTest(unittest.TestCase):
    def test_top_grasp_is_nearest_axis_alignment(self):
        current = so3_exp([0.25, -0.35, 0.4])
        candidate = top_grasp_candidate(current, [0, 0, 1], [0, 0, 1])
        np.testing.assert_allclose(candidate.rotation @ [0, 0, 1], [0, 0, -1], atol=1.0e-7)
        spun = candidate.rotation @ so3_exp([0, 0, math.pi])
        self.assertLessEqual(candidate.distance_rad,
                             rotation_distance(current, spun) + 1.0e-8)

    def test_side_grasp_selects_smallest_feasible_rotation(self):
        current = so3_exp([0.0, math.pi / 2.0 - 0.08, 0.0])
        directions = {
            "left": [-1, 0, 0], "right": [1, 0, 0],
            "front": [0, 1, 0], "back": [0, -1, 0],
        }
        candidates = side_grasp_candidates(current, [0, 0, 1], directions)
        selected = select_nearest_candidate(candidates)
        expected = min(candidates, key=lambda item: item.distance_rad)
        self.assertEqual(selected.label, expected.label)
        self.assertEqual(selected.label, "right")

    def test_grasp_center_is_preserved_during_orientation_correction(self):
        p_flange = np.array([0.45, 0.1, 0.55])
        r_flange = so3_exp([0.1, -0.2, 0.3])
        p_fc = np.array([0.17, 0.0, 0.0])
        r_fc = so3_exp([0.0, math.pi / 2.0, 0.0])
        center_position, _ = compose_pose(p_flange, r_flange, p_fc, r_fc)
        target_center_rotation = so3_exp([0.4, 0.1, -0.2])
        target_position, target_rotation = flange_pose_for_fixed_center(
            center_position, target_center_rotation, p_fc, r_fc)
        corrected_center, corrected_rotation = compose_pose(
            target_position, target_rotation, p_fc, r_fc)
        np.testing.assert_allclose(corrected_center, center_position, atol=1.0e-10)
        np.testing.assert_allclose(corrected_rotation, target_center_rotation, atol=1.0e-10)

    def test_sustained_opposition_reduces_assistance_strength(self):
        assist = MinimumInterventionOrientationAssist(
            [0.17, 0, 0], np.eye(3), rise_rate_per_s=5.0,
            fall_rate_per_s=4.0, opposition_duration_s=0.10)
        candidate = top_grasp_candidate(np.eye(3), [0, 0, 1], [0, 1, 0])
        assist.activate(candidate)
        for _ in range(5):
            before = assist.compute(0.02, np.zeros(3), np.eye(3), np.zeros(6))
        initial = before.strength
        for _ in range(15):
            result = assist.compute(0.02, np.zeros(3), np.eye(3),
                                    [0, 0, 0, -0.6, 0, 0])
        self.assertTrue(result.opposing)
        self.assertLess(result.strength, initial)


class GestureIsolationTest(unittest.TestCase):
    def test_short_false_positive_does_not_trigger(self):
        gate = GestureIsolationGate(stable_duration_s=0.3)
        self.assertIsNone(gate.update(0.0, GESTURE_CLOSE, 0.9).action)
        self.assertIsNone(gate.update(0.1, GESTURE_CLOSE, 0.9).action)
        result = gate.update(0.15, GESTURE_NONE, 0.0)
        self.assertIsNone(result.action)
        self.assertFalse(result.hold_arm)

    def test_stable_gesture_holds_arm_and_emits_once(self):
        gate = GestureIsolationGate(stable_duration_s=0.3)
        gate.update(0.0, GESTURE_CLOSE, 0.9)
        gate.update(0.2, GESTURE_CLOSE, 0.9)
        result = gate.update(0.31, GESTURE_CLOSE, 0.9)
        self.assertEqual(result.action, GESTURE_CLOSE)
        self.assertTrue(result.hold_arm)
        repeated = gate.update(0.5, GESTURE_CLOSE, 0.9)
        self.assertIsNone(repeated.action)
        self.assertTrue(repeated.hold_arm)


class ControlLoopTest(unittest.TestCase):
    def test_nominal_50hz_loop_statistics_and_processing_budget(self):
        shaper = LatestCommandShaper(
            [0.1, 0.1, 0.1, 0.6, 0.6, 0.6],
            [2, 2, 2, 12, 12, 12], 0.09, 0.15)
        processing = []
        ages = []
        for tick in range(250):
            now = tick / 50.0
            if tick % 5 != 4:  # irregular 30-ish Hz input; never pretend it is 50 Hz.
                source = math.floor(now * 30.0) / 30.0
                shaper.update(np.ones(6) * 0.01, source, True)
            began = time.perf_counter()
            result = shaper.tick(now)
            processing.append((time.perf_counter() - began) * 1000.0)
            ages.append(result.input_age_s)
        actual_hz = 249 / (249 / 50.0)
        self.assertAlmostEqual(actual_hz, 50.0, places=8)
        self.assertLess(np.percentile(processing, 99), 2.0)
        self.assertLess(max(age for age in ages if math.isfinite(age)), 0.09)

    def test_real_robot_output_is_fail_closed(self):
        token = "I_CONFIRM_REAL_ABB_IRB120"
        self.assertFalse(robot_output_allowed(False, False, False, ""))
        self.assertFalse(robot_output_allowed(False, True, False, token))
        self.assertFalse(robot_output_allowed(False, True, True, "wrong"))
        self.assertTrue(robot_output_allowed(False, True, True, token))
        self.assertTrue(robot_output_allowed(True, False, False, ""))


class SafetyConfigurationTest(unittest.TestCase):
    def test_position_reference_hold_requires_confirmed_target_loss(self):
        self.assertFalse(confirmed_position_reference_hold(True, 0.399, 0.40))
        self.assertTrue(confirmed_position_reference_hold(True, 0.400, 0.40))
        self.assertFalse(confirmed_position_reference_hold(False, 3.0, 0.40))
        self.assertFalse(confirmed_position_reference_hold(True, 3.0, 0.0))
        self.assertFalse(
            confirmed_position_reference_hold(True, float("inf"), 0.40))
        with self.assertRaises(ValueError):
            confirmed_position_reference_hold(True, 1.0, -0.1)

    def test_configured_camera_axes_map_to_expected_base_directions(self):
        config = yaml.safe_load((PACKAGE / "config/shared_teleop.yaml").read_text(
            encoding="utf-8"))
        mapping = config["mapping"]
        translation = np.asarray(mapping["translation_matrix"], dtype=float)
        rotation = np.asarray(mapping["rotation_matrix"], dtype=float)
        expected_translation = np.array(
            [[0, 0, -1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        expected_rotation = np.eye(3)
        np.testing.assert_array_equal(translation, expected_translation)
        np.testing.assert_array_equal(rotation, expected_rotation)
        self.assertAlmostEqual(np.linalg.det(translation), -1.0)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_live_mapping_uses_v3_pose_with_response_first_limits(self):
        config = yaml.safe_load((PACKAGE / "config/shared_teleop.yaml").read_text(
            encoding="utf-8"))
        mapping = config["mapping"]
        limits = config["limits"]
        np.testing.assert_allclose(mapping["translation_gain"], [0.6, 0.6, 1.0])
        np.testing.assert_allclose(mapping["rotation_gain"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            limits["maximum_linear_velocity_mps"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            limits["maximum_angular_velocity_radps"], [10.0, 10.0, 10.0])
        np.testing.assert_allclose(
            limits["maximum_linear_acceleration_mps2"], [50.0, 50.0, 50.0])
        np.testing.assert_allclose(
            limits["maximum_angular_acceleration_radps2"], [500.0, 500.0, 500.0])
        self.assertEqual(config["control"]["mode"], "RELATIVE_POSE_TRACKING")
        np.testing.assert_allclose(
            config["control"]["translation_error_gain_per_s"], [8.0] * 3)
        np.testing.assert_allclose(
            config["control"]["rotation_error_gain_per_s"], [8.0] * 3)
        np.testing.assert_allclose(
            config["control"]["translation_feedforward_gain"], [1.0] * 3)
        np.testing.assert_allclose(
            config["control"]["rotation_feedforward_gain"], [0.75] * 3)
        self.assertAlmostEqual(config["safety"]["input_timeout_s"], 0.40)
        self.assertAlmostEqual(config["safety"]["timeout_zero_deadline_s"], 0.65)
        self.assertAlmostEqual(config["trend"]["maximum_dt_s"], 0.45)
        self.assertAlmostEqual(config["trend"]["maximum_pose_step_m"], 0.040)
        self.assertAlmostEqual(
            config["trend"]["maximum_pose_position_innovation_m"], 1.000)
        self.assertAlmostEqual(
            config["trend"]["maximum_pose_rotation_step_deg"], 45.0)
        self.assertAlmostEqual(
            config["trend"]["maximum_pose_rotation_innovation_deg"], 180.0)
        self.assertAlmostEqual(
            config["trend"]["pose_position_lowpass_alpha"], 0.28)
        self.assertAlmostEqual(
            config["trend"]["pose_position_lowpass_alpha_max"], 0.95)
        self.assertAlmostEqual(
            config["trend"]["pose_rotation_lowpass_alpha"], 0.30)
        self.assertAlmostEqual(
            config["trend"]["pose_rotation_lowpass_alpha_max"], 0.97)
        self.assertAlmostEqual(
            config["trend"]["causal_smoothing_alpha"], 0.65)
        self.assertAlmostEqual(
            config["control"]["maximum_linear_speed_norm_mps"], 1.00)
        self.assertAlmostEqual(
            config["control"]["maximum_angular_speed_norm_radps"], 10.00)
        self.assertAlmostEqual(
            config["control"]["maximum_relative_rotation_deg"], 179.0)
        self.assertAlmostEqual(
            config["control"]["collision_disarm_scale"], 0.0)
        self.assertEqual(
            config["control"]["servo_interlock_statuses"], [])
        self.assertEqual(
            config["control"]["servo_retreat_statuses"], [2, 5])
        self.assertEqual(
            config["control"]["servo_auto_reset_statuses"], [2, 5])
        self.assertEqual(
            config["control"]["servo_reset_service"],
            "/servo_server/reset_servo_status")
        self.assertAlmostEqual(
            config["control"]["servo_reset_min_interval_s"], 0.25)
        self.assertAlmostEqual(
            config["control"]["servo_reset_fresh_target_s"], 0.40)
        self.assertTrue(config["control"]["repeat_last_target_pose"])
        self.assertAlmostEqual(
            config["control"]["target_hold_timeout_s"], 0.40)
        self.assertAlmostEqual(
            config["control"]["collision_guard_enter_scale"], 0.20)
        self.assertAlmostEqual(
            config["control"]["collision_guard_release_scale"], 0.80)
        self.assertFalse(
            config["control"]["require_collision_free_target_ik"])

    def test_reference_is_explicit_fixed_camera_frame_without_auto_rezero(self):
        config = yaml.safe_load((PACKAGE / "config/shared_teleop.yaml").read_text(
            encoding="utf-8"))
        reference = config["reference"]
        self.assertEqual(reference["frame"], "camera_color_optical_frame")
        self.assertTrue(reference["require_confirmation"])
        self.assertTrue(reference["require_new_c_after_receiver_start"])
        self.assertFalse(reference["allow_automatic_rezero"])
        self.assertEqual(
            reference["direction_basis"],
            "FIXED_CAMERA_TRANSLATION_AND_C_ZERO_LOCAL_ROTATION")

    def test_ground_workspace_profile_is_additive_and_legacy_remains_default(self):
        config = yaml.safe_load((PACKAGE / "config/shared_teleop.yaml").read_text(
            encoding="utf-8"))
        profile = config["mapping_profiles"]["camera_ground_workspace"]
        self.assertEqual(profile["mode"], "CAMERA_RANGE_TO_GROUND_SECTOR")
        np.testing.assert_array_equal(
            profile["translation_matrix"], config["mapping"]["translation_matrix"])
        self.assertEqual(
            profile["robot_workspace"]["model"],
            "FRONT_GROUND_CLIPPED_ELLIPSOID")
        self.assertGreaterEqual(
            profile["robot_workspace"]["minimum_forward_x_m"], 0.0)
        self.assertGreater(
            profile["robot_workspace"]["minimum_tool_z_m"], 0.0)
        self.assertGreaterEqual(
            profile["robot_workspace"]["minimum_tool_z_m"], 0.10)
        self.assertFalse(
            profile["normalized_pose_mapping"][
                "combine_translation_rotation"])
        self.assertFalse(
            profile["normalized_pose_mapping"][
                "reachability_projection"]["enabled"])

        calibration = yaml.safe_load((
            PACKAGE / "config/camera_workspace_calibration.yaml").read_text(
                encoding="utf-8"))
        self.assertEqual(calibration["schema_version"], 1)
        self.assertEqual(
            calibration["frame_id"], "camera_color_optical_frame")
        self.assertTrue(all(
            value > 0.0 for value in
            calibration["human_workspace"]["negative_extent_m"]))
        self.assertTrue(all(
            value > 0.0 for value in
            calibration["human_workspace"]["positive_extent_m"]))

    def test_new_launch_files_default_real_robot_output_off(self):
        launches = [PACKAGE / "launch/shared_teleop_core.launch",
                    PACKAGE / "launch/shared_teleop_safe_demo.launch"]
        for launch in launches:
            root = ET.parse(str(launch)).getroot()
            arguments = {entry.attrib["name"]: entry.attrib.get("default")
                         for entry in root.findall("arg")}
            self.assertEqual(arguments.get("enable_robot"), "false", str(launch))

    def test_safe_demo_starts_at_requested_joint_target(self):
        launch = PACKAGE / "launch/shared_teleop_safe_demo.launch"
        root = ET.parse(str(launch)).getroot()
        arguments = {entry.attrib["name"]: entry.attrib.get("default")
                     for entry in root.findall("arg")}
        tokens = arguments["safe_initial_joint_positions"].split()
        positions = {
            tokens[index + 1]: float(tokens[index + 2])
            for index in range(0, len(tokens), 3)
            if tokens[index] == "-J"
        }
        np.testing.assert_allclose(
            [positions["joint_{}".format(index)] for index in range(1, 7)],
            [0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0],
            atol=1.0e-12)

    def test_live_human_launch_forces_udp_safe_simulation_entry(self):
        launch = PACKAGE / "launch/live_human_gazebo_teleop.launch"
        root = ET.parse(str(launch)).getroot()
        includes = root.findall("include")
        self.assertEqual(len(includes), 1)
        self.assertIn("shared_teleop_safe_demo.launch", includes[0].attrib["file"])
        forwarded = {
            entry.attrib["name"]: entry.attrib.get("value")
            for entry in includes[0].findall("arg")
        }
        self.assertEqual(forwarded.get("input_source"), "udp")
        self.assertEqual(forwarded.get("response_first"), "true")
        text = launch.read_text(encoding="utf-8")
        self.assertNotIn("enable_robot", text)

    def test_ground_live_launch_selects_new_profile_without_changing_legacy(self):
        launch = PACKAGE / "launch/live_human_ground_gazebo_teleop.launch"
        root = ET.parse(str(launch)).getroot()
        include = root.find("include")
        forwarded = {
            entry.attrib["name"]: entry.attrib.get("value")
            for entry in include.findall("arg")
        }
        self.assertEqual(
            forwarded["mapping_profile"], "camera_ground_workspace")
        self.assertIn("world_name", forwarded)
        self.assertEqual(
            forwarded["safe_initial_joint_positions"],
            "$(arg safe_initial_joint_positions)")

        arguments = {
            entry.attrib["name"]: entry.attrib.get("default")
            for entry in root.findall("arg")
        }
        tokens = arguments["safe_initial_joint_positions"].split()
        positions = {
            tokens[index + 1]: float(tokens[index + 2])
            for index in range(0, len(tokens), 3)
            if tokens[index] == "-J"
        }
        np.testing.assert_allclose(
            [positions["joint_{}".format(index)] for index in range(1, 7)],
            [0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0],
            atol=1.0e-12)

        legacy = ET.parse(str(
            PACKAGE / "launch/live_human_gazebo_teleop.launch")).getroot()
        arguments = {
            entry.attrib["name"]: entry.attrib.get("default")
            for entry in legacy.findall("arg")
        }
        self.assertEqual(arguments["mapping_profile"], "current_linear")

    def test_safe_servo_checks_known_model_collisions(self):
        path = PACKAGE.parent / "abb120_moveit_config1/config/servo_abbarm_velocity_safe.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertTrue(config["check_collisions"])
        self.assertLessEqual(config["incoming_command_timeout"], 0.10)
        self.assertEqual(config["ee_frame_name"], "tool0")
        self.assertGreaterEqual(config["self_collision_proximity_threshold"], 0.01)
        self.assertGreaterEqual(config["collision_check_rate"], 60.0)

    def test_realtime_gazebo_servo_is_unfiltered_and_collision_unscaled(self):
        path = (PACKAGE.parent /
                "abb120_moveit_config1/config/servo_abbarm_velocity_realtime_gazebo.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertFalse(config["check_collisions"])
        self.assertTrue(config["low_latency_mode"])
        self.assertEqual(config["low_pass_filter_coeff"], 0.0)
        self.assertEqual(config["joint_limit_margin"], 0.0)
        self.assertEqual(config["ee_frame_name"], "tool0")

    def test_hardware_transforms_are_explicitly_uncalibrated(self):
        config = yaml.safe_load((PACKAGE / "config/shared_teleop.yaml").read_text(encoding="utf-8"))
        self.assertFalse(config["calibration"]["real_robot_use_allowed"])
        self.assertIn("TEMPORARY", config["calibration"]["status"])
        self.assertEqual(config["frames"]["servo_control"], "tool0")
        self.assertEqual(config["workspace"]["reference_link"], "tool0")

    def test_velocity_gazebo_loads_dedicated_arm_velocity_pids(self):
        source_space = PACKAGE.parent
        launch = (source_space / "abb120_moveit_config1/launch/gazebo_velocity.launch").read_text(
            encoding="utf-8")
        arm_gains = yaml.safe_load((
            source_space /
            "handarm_sim_demo/config/gazebo_arm_velocity_pid.yaml").read_text(
                encoding="utf-8"))
        hand_gains = (source_space /
                      "handarm_sim_demo/config/gazebo_hand_only_pid.yaml").read_text(
                          encoding="utf-8")
        self.assertIn("gazebo_arm_velocity_pid.yaml", launch)
        self.assertIn("gazebo_hand_only_pid.yaml", launch)
        for joint in range(1, 7):
            name = "joint_{}".format(joint)
            self.assertIn(name, arm_gains)
            self.assertGreater(arm_gains[name]["p"], 0.0)
            self.assertNotRegex(
                hand_gains, r"(?m)^{}\s*:".format(name))

        velocity_urdf = (source_space /
                         "abb120_moveit_config1/config/gazebo_handarm_velocity.urdf").read_text(
                             encoding="utf-8")
        for link in ["link_{}".format(i) for i in range(1, 7)]:
            self.assertRegex(
                velocity_urdf,
                r'<gazebo reference="{}"><gravity>false</gravity></gazebo>'.format(link))


if __name__ == "__main__":
    unittest.main()
