#!/usr/bin/env python3

import unittest
import time

import numpy as np

from perception_hamer.src.finger_observation import finger_shape_signature, finger_surface_depths, observe_mano_fingers


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
    def test_normalized_shapes_ignore_wrist_pose_and_size_but_capture_thumb_opposition(self):
        hand = open_hand()
        signature = finger_shape_signature(hand)
        rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(finger_shape_signature(2.4*hand @ rotation.T + [19, -7, 3]),
                                   signature, atol=1e-12)
        thumb = hand.copy()
        thumb[3:5] = [.5, .25, .6]
        displaced = finger_shape_signature(thumb) - signature
        np.testing.assert_allclose(displaced[[0, 1, 2, 4]], 0)
        self.assertGreater(abs(displaced[3,2]), .5)
        self.assertIsNone(finger_shape_signature(np.zeros((21,3))))

    def test_surface_depth_uses_local_consistent_hand_surface(self):
        pixels = np.tile([7, 7], (21, 1))
        pixels[1:5] = [[1, 1], [16, 7], [7, 16], [16, 16]]
        z = np.zeros((25, 25), dtype=np.uint16)
        z[5:10, 5:10] = 682
        z[5:10, 14:19] = 1050
        z[14:19, 5:10] = 650
        z[14:16, 5:10] = 750
        z[14:16, 14:19] = 688
        values = finger_surface_depths(pixels, z, .001, .70)
        self.assertAlmostEqual(values[0], .682)
        self.assertEqual(values[1:5], [None]*4)
        self.assertIsNone(finger_surface_depths(None, z, .001, .70))

    def test_detector_pairs_only_completed_matching_exposures(self):
        from perception_hamer.scripts.run_d455_hamer_crop import AsyncMediaPipeDetection

        class Sidecar:
            def detect(self, rgb):
                return dict(valid=True, bbox=[0,0,8,8], is_right=False, confidence=.9,
                            exposure_value=int(rgb[0,0,0]))

        worker = AsyncMediaPipeDetection(Sidecar(), fixed_left_hand=True)
        try:
            for frame in (21, 27):
                worker.submit(np.full((12,12,3),frame,dtype=np.uint8), frame)
                end = time.monotonic() + 2
                while worker.detection_for_frame(frame) is None and time.monotonic()<end:
                    time.sleep(.002)
                self.assertEqual(worker.detection_for_frame(frame)["exposure_value"], frame)
            self.assertEqual(worker.detection_for_frame(21)["exposure_value"], 21)
            self.assertIsNone(worker.detection_for_frame(22))
        finally:
            worker.close()

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
