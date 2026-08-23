#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.roi_provider import (  # noqa: E402
    EVIDENCE_SCOPE,
    KLTTrackerROIProvider,
    ManualROIProvider,
    MediaPipeBBoxProvider,
    ROIInitializationError,
    clip_bbox,
)


def textured_frame(height=180, width=240):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(45, 135, 12):
        for x in range(65, 165, 12):
            colour = ((3 * x) % 255, (5 * y) % 255, (x + y) % 255)
            cv2.circle(frame, (x, y), 3, colour, -1)
            cv2.line(frame, (x - 4, y), (x + 4, y), (255, 255, 255), 1)
    return frame


class BoxContractTest(unittest.TestCase):
    def test_clip_and_visible_fraction(self):
        bbox, visible = clip_bbox([-10, 20, 50, 80], (100, 100, 3))
        np.testing.assert_allclose(bbox, [0, 20, 50, 80])
        self.assertAlmostEqual(visible, 5.0 / 6.0)

    def test_invalid_boxes_rejected(self):
        for bbox in ([0, 0, 0, 10], [10, 10, 2, 20], [0, 0, np.nan, 2]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                clip_bbox(bbox, (100, 100, 3))


class ManualProviderTest(unittest.TestCase):
    def test_fixed_baseline_contract(self):
        frame = textured_frame()
        provider = ManualROIProvider([60, 40, 170, 140], is_right=True)
        first = provider.initialize(frame)
        second = provider.update(frame)
        self.assertFalse(first.lost)
        self.assertTrue(first.reinitialized)
        self.assertEqual(first.age, 0)
        self.assertEqual(second.age, 1)
        self.assertFalse(second.reinitialized)
        self.assertEqual(second.center_jump, 0.0)
        self.assertEqual(second.scale_change, 1.0)
        self.assertEqual(second.evidence_scope, EVIDENCE_SCOPE)
        self.assertEqual(second.as_dict()["source"], "manual_roi")
        self.assertTrue(second.as_dict()["valid"])

    def test_outside_box_fails_closed(self):
        provider = ManualROIProvider([300, 300, 400, 400])
        result = provider.initialize(textured_frame())
        self.assertTrue(result.lost)
        self.assertIsNone(result.bbox)


class KLTProviderTest(unittest.TestCase):
    def test_translation_is_tracked_with_motion_metrics(self):
        first_frame = textured_frame()
        dx, dy = 8.0, -5.0
        affine = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        second_frame = cv2.warpAffine(
            first_frame, affine, (first_frame.shape[1], first_frame.shape[0])
        )
        provider = KLTTrackerROIProvider(
            [55, 35, 175, 145],
            is_right=False,
            min_tracked_points=6,
            max_forward_backward_error=0.8,
        )
        initial = provider.initialize(first_frame)
        tracked = provider.update(second_frame)
        self.assertTrue(initial.reinitialized)
        self.assertFalse(tracked.lost, tracked.reason)
        np.testing.assert_allclose(
            tracked.bbox, np.array([55, 35, 175, 145]) + [dx, dy, dx, dy], atol=1.5
        )
        self.assertAlmostEqual(tracked.center_jump, np.hypot(dx, dy), delta=1.5)
        self.assertAlmostEqual(tracked.scale_change, 1.0, delta=0.03)
        self.assertFalse(tracked.is_right)

    def test_loss_exposes_no_stale_box_and_reinitialize_is_explicit(self):
        first_frame = textured_frame()
        provider = KLTTrackerROIProvider(
            [55, 35, 175, 145], min_tracked_points=6
        )
        provider.initialize(first_frame)
        lost = provider.update(np.zeros_like(first_frame))
        self.assertTrue(lost.lost)
        self.assertIsNone(lost.bbox)
        again = provider.update(first_frame)
        self.assertTrue(again.lost)
        reacquired = provider.reinitialize(first_frame, [55, 35, 175, 145], True)
        self.assertTrue(reacquired.reinitialized)
        self.assertFalse(reacquired.lost)

    def test_textureless_initialization_rejected(self):
        provider = KLTTrackerROIProvider([20, 20, 100, 100])
        with self.assertRaises(ROIInitializationError):
            provider.initialize(np.zeros((120, 160, 3), dtype=np.uint8))

    def test_bbox_log_scale_smoothing(self):
        first_frame = textured_frame()
        affine = np.array([[1.2, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float32)
        second_frame = cv2.warpAffine(first_frame, affine, (240, 180))
        raw = KLTTrackerROIProvider(
            [55, 35, 175, 145], min_tracked_points=6, bbox_smoothing_alpha=1.0
        )
        smooth = KLTTrackerROIProvider(
            [55, 35, 175, 145], min_tracked_points=6, bbox_smoothing_alpha=0.25
        )
        raw.initialize(first_frame)
        smooth.initialize(first_frame)
        raw_result = raw.update(second_frame)
        smooth_result = smooth.update(second_frame)
        self.assertFalse(raw_result.lost, raw_result.reason)
        self.assertFalse(smooth_result.lost, smooth_result.reason)
        self.assertLess(abs(smooth_result.scale_change - 1.0), abs(raw_result.scale_change - 1.0))

    def test_reference_alpha_reduces_lag_relative_to_old_live_value(self):
        first_frame = textured_frame()
        affine = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -6.0]], dtype=np.float32)
        second_frame = cv2.warpAffine(first_frame, affine, (240, 180))
        old = KLTTrackerROIProvider(
            [55, 35, 175, 145], min_tracked_points=6,
            bbox_smoothing_alpha=0.35,
        )
        migrated = KLTTrackerROIProvider(
            [55, 35, 175, 145], min_tracked_points=6,
            bbox_smoothing_alpha=0.55,
        )
        old.initialize(first_frame)
        migrated.initialize(first_frame)
        old_result = old.update(second_frame)
        migrated_result = migrated.update(second_frame)
        expected_center = np.array([127.0, 84.0])
        old_center = 0.5 * (old_result.bbox[:2] + old_result.bbox[2:])
        migrated_center = 0.5 * (
            migrated_result.bbox[:2] + migrated_result.bbox[2:]
        )
        self.assertLess(
            np.linalg.norm(migrated_center - expected_center),
            np.linalg.norm(old_center - expected_center),
        )


class _NoDepthLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def z(self):
        raise AssertionError("MediaPipe landmark depth must not be read")


class _Hand:
    def __init__(self, points):
        self.landmark = [_NoDepthLandmark(x, y) for x, y in points]


class _Classification:
    label = "Left"
    score = 0.8


class _Handedness:
    classification = [_Classification()]


class _Result:
    def __init__(self, detected=True):
        if detected:
            self.multi_hand_landmarks = [
                _Hand([(0.25, 0.20), (0.45, 0.20), (0.45, 0.60), (0.25, 0.60)])
            ]
            self.multi_handedness = [_Handedness()]
        else:
            self.multi_hand_landmarks = None
            self.multi_handedness = None

    @property
    def multi_hand_world_landmarks(self):
        raise AssertionError("MediaPipe world landmarks must not be read")


class _Detector:
    def __init__(self, sequence):
        self.sequence = list(sequence)

    def process(self, _frame):
        return self.sequence.pop(0)


class MediaPipeBBoxProviderTest(unittest.TestCase):
    def test_only_xy_bbox_and_coarse_handedness_are_consumed(self):
        provider = MediaPipeBBoxProvider(
            detector=_Detector([_Result(True)]), bbox_margin_fraction=0.0
        )
        result = provider.initialize(np.zeros((100, 200, 3), dtype=np.uint8))
        np.testing.assert_allclose(result.bbox, [50, 20, 90, 60])
        self.assertFalse(result.is_right)
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertTrue(result.reinitialized)

    def test_detection_loss_and_reacquisition_are_explicit(self):
        provider = MediaPipeBBoxProvider(
            detector=_Detector([_Result(True), _Result(False), _Result(True)]),
            bbox_margin_fraction=0.0,
        )
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.assertFalse(provider.initialize(frame).lost)
        lost = provider.update(frame)
        self.assertTrue(lost.lost)
        self.assertIsNone(lost.bbox)
        reacquired = provider.update(frame)
        self.assertFalse(reacquired.lost)
        self.assertTrue(reacquired.reinitialized)
        self.assertEqual(reacquired.age, 0)


if __name__ == "__main__":
    unittest.main()
