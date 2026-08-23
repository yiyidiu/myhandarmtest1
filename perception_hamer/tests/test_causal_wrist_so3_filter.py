#!/usr/bin/env python3

import math
import unittest

import numpy as np

from perception_hamer.src.causal_wrist_so3_filter import (
    CausalWristSO3Filter,
    CausalWristSO3FilterConfig,
)
from perception_hamer.src.crop_quality import bbox_crop_quality
from perception_hamer.src.palm_frame import rotation_distance_rad


def so3_exp(rotation_vector):
    value = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle < 1.0e-12:
        return np.eye(3)
    axis = value / angle
    x_value, y_value, z_value = axis
    skew = np.asarray([
        [0.0, -z_value, y_value],
        [z_value, 0.0, -x_value],
        [-y_value, x_value, 0.0],
    ])
    return (
        np.eye(3) + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


class CropQualityTest(unittest.TestCase):
    def test_centered_stable_crop_scores_above_border_and_jump(self):
        centered = [240, 160, 400, 320]
        stable = bbox_crop_quality(centered, centered, 640, 480)
        border = bbox_crop_quality([0, 160, 160, 320], centered, 640, 480)
        jumped = bbox_crop_quality([430, 250, 590, 410], centered, 640, 480)
        self.assertGreater(stable, border)
        self.assertGreater(stable, jumped)
        self.assertEqual(bbox_crop_quality(None, centered, 640, 480), 0.0)


class CausalWristSO3FilterTest(unittest.TestCase):
    def test_static_noise_is_reduced_on_so3(self):
        generator = np.random.default_rng(20260820)
        active = CausalWristSO3Filter()
        raw = []
        filtered = []
        for index in range(240):
            measurement = so3_exp(np.radians(generator.normal(0.0, 1.2, 3)))
            result = active.update(index / 30.0, measurement, 0.95)
            self.assertTrue(result.valid, result)
            raw.append(measurement)
            filtered.append(result.rotation)
        raw_steps = np.asarray([
            rotation_distance_rad(first, second)
            for first, second in zip(raw[20:-1], raw[21:])
        ])
        filtered_steps = np.asarray([
            rotation_distance_rad(first, second)
            for first, second in zip(filtered[20:-1], filtered[21:])
        ])
        self.assertLess(np.median(filtered_steps), 0.70 * np.median(raw_steps))

    def test_uses_actual_irregular_dt_and_quality(self):
        high = CausalWristSO3Filter()
        low = CausalWristSO3Filter()
        high.update(0.0, np.eye(3), 1.0)
        low.update(0.0, np.eye(3), 1.0)
        # Stay below the unambiguous-motion threshold: small corrections keep
        # quality-adaptive smoothing, while >=8 deg is intentionally direct.
        target = so3_exp([0.0, math.radians(5.0), 0.0])
        high_result = high.update(0.12, target, 0.95)
        low_result = low.update(0.12, target, 0.20)
        self.assertGreater(high_result.gain, low_result.gain)
        self.assertGreater(high_result.confidence, low_result.confidence)

    def test_constant_slow_rotation_has_bounded_lag(self):
        active = CausalWristSO3Filter()
        errors_deg = []
        for index in range(181):
            expected = so3_exp([0.0, 0.0, math.radians(0.2 * index)])
            result = active.update(index / 30.0, expected, 0.95)
            self.assertTrue(result.valid, result)
            errors_deg.append(
                math.degrees(rotation_distance_rad(result.rotation, expected))
            )
        self.assertLess(float(np.median(errors_deg[30:])), 1.0)
        self.assertLess(float(np.percentile(errors_deg[30:], 95)), 1.5)

    def test_large_positive_and_negative_rotations_follow_in_one_update(self):
        for angle_deg in (-150.0, -90.0, 90.0, 150.0):
            active = CausalWristSO3Filter()
            active.update(0.0, np.eye(3), 1.0)
            target = so3_exp([math.radians(angle_deg), 0.0, 0.0])
            result = active.update(0.1, target, 0.20)
            self.assertTrue(result.valid)
            self.assertEqual(
                result.status, "tracking_large_angle_passthrough"
            )
            self.assertTrue(result.large_angle_passthrough)
            self.assertAlmostEqual(result.gain, 1.0)
            self.assertLess(
                rotation_distance_rad(result.rotation, target), 1.0e-7
            )

    def test_reject_mode_remains_available_as_explicit_rollback(self):
        active = CausalWristSO3Filter(
            CausalWristSO3FilterConfig(large_angle_mode="reject")
        )
        active.update(0.0, np.eye(3), 1.0)
        target = so3_exp([0.0, 0.0, math.radians(90.0)])
        result = active.update(0.1, target, 0.95)
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "jump_rejected")
        self.assertIsNone(result.rotation)
        self.assertFalse(result.large_angle_passthrough)

    def test_nonmonotonic_timestamp_never_exposes_stale_rotation(self):
        active = CausalWristSO3Filter()
        active.update(1.0, np.eye(3), 1.0)
        result = active.update(1.0, np.eye(3), 1.0)
        self.assertFalse(result.valid)
        self.assertIsNone(result.rotation)
        self.assertEqual(result.reason, "nonmonotonic_timestamp")


if __name__ == "__main__":
    unittest.main()
