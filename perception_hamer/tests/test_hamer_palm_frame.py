#!/usr/bin/env python3

import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.hamer_palm_frame import (  # noqa: E402
    RobustBetasCalibrator,
    build_hamer_joint_palm_frame,
)


def joints() -> np.ndarray:
    value = np.zeros((21, 3), dtype=np.float64)
    value[0] = [0.0, 0.0, 0.0]
    value[5] = [1.0, 2.0, 0.0]
    value[9] = [0.0, 2.5, 0.0]
    value[17] = [-1.0, 2.0, 0.0]
    return value


class HamerPalmFrameTest(unittest.TestCase):
    def test_requested_joint_definition_is_identity_for_canonical_points(self):
        result = build_hamer_joint_palm_frame(joints(), True)
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(result.origin, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.det(result.rotation), 1.0, places=12)

    def test_rigid_transform_covariance(self):
        angle = 0.7
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([0.1, -0.2, 1.0])
        transformed = joints() @ rotation.T + translation
        result = build_hamer_joint_palm_frame(transformed, True)
        np.testing.assert_allclose(result.rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(result.origin, translation, atol=1e-12)

    def test_left_source_axes_still_produce_so3(self):
        mirrored = joints().copy()
        mirrored[:, 0] *= -1.0
        result = build_hamer_joint_palm_frame(mirrored, False)
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.rotation.T @ result.rotation, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(result.rotation), 1.0)

    def test_invalid_never_returns_identity_substitute(self):
        result = build_hamer_joint_palm_frame(np.zeros((21, 3)), True)
        self.assertFalse(result.valid)
        self.assertIsNone(result.rotation)
        self.assertIsNone(result.quaternion_xyzw)
        self.assertTrue(result.failure_reason)

    def test_quaternion_sign_continuity(self):
        first = build_hamer_joint_palm_frame(joints(), True)
        second = build_hamer_joint_palm_frame(
            joints(), True, previous_quaternion_xyzw=-first.quaternion_xyzw
        )
        self.assertGreater(
            float(np.dot(second.quaternion_xyzw, -first.quaternion_xyzw)), 0.0
        )


class RobustBetasTest(unittest.TestCase):
    def test_requires_thirty_and_uses_median(self):
        calibrator = RobustBetasCalibrator(30, 60)
        for index in range(29):
            self.assertFalse(calibrator.add(np.full(10, index * 1e-3), index))
        self.assertTrue(calibrator.add(np.full(10, 100.0), 29.0))
        self.assertTrue(calibrator.frozen)
        np.testing.assert_allclose(calibrator.betas_user, np.full(10, 0.0145))
        report = calibrator.as_dict()
        self.assertEqual(report["collected_samples"], 30)
        self.assertEqual(report["estimator"], "coordinate_wise_median")

    def test_invalid_beta_rejected(self):
        calibrator = RobustBetasCalibrator()
        with self.assertRaises(ValueError):
            calibrator.add(np.zeros(9), 1.0)
        with self.assertRaises(ValueError):
            calibrator.add(np.full(10, np.nan), 1.0)


if __name__ == "__main__":
    unittest.main()
