#!/usr/bin/env python3

import unittest

import numpy as np

from perception_hamer.src.finger_observation import observe_mano_fingers


CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


def open_hand():
    points = np.zeros((21, 3), dtype=float)
    angles = np.deg2rad([-45.0, -20.0, 0.0, 20.0, 40.0])
    for chain, angle in zip(CHAINS, angles):
        direction = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        for distance, index in enumerate(chain[1:], 1):
            points[index] = direction * distance
    return points


class FingerObservationTest(unittest.TestCase):
    def test_open_chains_have_zero_flexion(self):
        result = observe_mano_fingers(open_hand(), 0.9, 0.8, 0.75)
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.flexion, np.zeros(5), atol=1.0e-12)
        self.assertAlmostEqual(result.confidence, 0.54)

    def test_flexion_is_rigid_transform_and_scale_invariant(self):
        points = open_hand()
        # Bend the index chain through three right-angle joints.
        points[5:9] = np.asarray([
            [1.0, -0.35, 0.0],
            [1.0, 0.65, 0.0],
            [0.0, 0.65, 0.0],
            [0.0, -0.35, 0.0],
        ])
        angle = np.deg2rad(37.0)
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        transformed = 2.7 * (points @ rotation.T) + [4.0, -2.0, 0.5]
        original = observe_mano_fingers(points, 1.0, 1.0, 1.0)
        moved = observe_mano_fingers(transformed, 1.0, 1.0, 1.0)
        self.assertTrue(original.valid)
        self.assertGreater(original.flexion[1], 0.99)
        np.testing.assert_allclose(moved.flexion, original.flexion, atol=1.0e-8)

    def test_degenerate_or_nonfinite_geometry_fails_closed(self):
        points = open_hand()
        points[8] = points[7]
        result = observe_mano_fingers(points, 1.0, 1.0, 1.0)
        self.assertFalse(result.valid)
        np.testing.assert_array_equal(result.flexion, np.zeros(5))
        self.assertIn("FINGER_GEOMETRY_INVALID", result.invalid_reason)

        points = open_hand()
        points[3, 1] = np.nan
        result = observe_mano_fingers(points, 1.0, 1.0, 1.0)
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
