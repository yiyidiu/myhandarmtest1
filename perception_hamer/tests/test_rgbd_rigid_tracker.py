#!/usr/bin/env python3

import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.rgbd_rigid_tracker import (  # noqa: E402
    RGBDRigidTracker,
    RGBDRigidTrackerConfig,
    RGBDTrackerFrame,
    RGBDRelativeOrientationTracker,
    RelativeTrackingState,
    RigidTrackResult,
    RigidTrackingError,
    build_3d_correspondences,
    build_rigid_palm_mask,
    deproject_pixels,
    ransac_rigid_kabsch,
    rigid_kabsch,
    rgbd_tracker_frame_from_d455,
    track_shi_tomasi_klt_fb,
)


WIDTH = 160
HEIGHT = 120
FX = 120.0
FY = 120.0
PPX = 80.0
PPY = 60.0
ROI = (10.0, 10.0, 150.0, 110.0)


def intrinsics(coeffs=None, model="distortion.none"):
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "fx": FX,
        "fy": FY,
        "ppx": PPX,
        "ppy": PPY,
        "distortion_model": model,
        "coeffs": [0.0] * 5 if coeffs is None else list(coeffs),
    }


def feature_image(shift_x=0.0, shift_y=0.0):
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(20, 105, 14):
        for x in range(20, 145, 14):
            color = (
                80 + (3 * x + y) % 170,
                60 + (x + 5 * y) % 190,
                70 + (7 * x + 2 * y) % 180,
            )
            cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), color, -1)
    if shift_x == 0.0 and shift_y == 0.0:
        return image
    matrix = np.asarray([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
    return cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def frame(
    *,
    shift_x=0.0,
    shift_y=0.0,
    timestamp_s=1.0,
    frame_number=10,
    depth_raw=None,
    timestamp_domain="global_time",
):
    if depth_raw is None:
        depth_raw = np.full((HEIGHT, WIDTH), 1000, dtype=np.uint16)
    return RGBDTrackerFrame(
        rgb=feature_image(shift_x, shift_y),
        aligned_depth_raw=np.ascontiguousarray(depth_raw),
        color_intrinsics=intrinsics(),
        depth_scale_m_per_unit=0.001,
        palm_bbox_xyxy=ROI,
        timestamp_s=timestamp_s,
        frame_number=frame_number,
        timestamp_domain=timestamp_domain,
    )


def rotation_xyz(rx, ry, rz):
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rx_matrix = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_matrix = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_matrix = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_matrix @ ry_matrix @ rx_matrix


def synthetic_points(count=80, seed=19):
    random = np.random.default_rng(seed)
    points = random.uniform([-0.12, -0.09, 0.55], [0.12, 0.09, 0.95], size=(count, 3))
    return points


class KabschGeometryTest(unittest.TestCase):
    def test_pure_translation_has_identity_rotation(self):
        source = synthetic_points()
        translation = np.asarray([0.035, -0.012, 0.021])
        estimate = rigid_kabsch(source, source + translation)
        np.testing.assert_allclose(estimate.rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(estimate.translation, translation, atol=1e-12)
        self.assertLess(estimate.rms_m, 1e-12)
        self.assertAlmostEqual(np.linalg.det(estimate.rotation), 1.0)

    def test_pure_rotation_and_translation(self):
        source = synthetic_points()
        rotation = rotation_xyz(math.radians(8), math.radians(-11), math.radians(17))
        translation = np.asarray([-0.02, 0.014, 0.006])
        target = (rotation @ source.T).T + translation
        estimate = rigid_kabsch(source, target)
        np.testing.assert_allclose(estimate.rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(estimate.translation, translation, atol=1e-12)
        self.assertLess(estimate.rms_m, 1e-12)

    def test_ransac_rejects_thirty_percent_outliers(self):
        source = synthetic_points(100)
        rotation = rotation_xyz(0.04, -0.08, 0.12)
        translation = np.asarray([0.018, -0.009, 0.011])
        target = (rotation @ source.T).T + translation
        random = np.random.default_rng(23)
        target[70:] = random.uniform([-0.3, -0.3, 0.3], [0.3, 0.3, 1.3], (30, 3))
        config = RGBDRigidTrackerConfig(
            ransac_threshold_m=0.003,
            maximum_kabsch_rms_m=0.002,
            minimum_ransac_inliers=50,
            minimum_inlier_ratio=0.50,
        )
        estimate = ransac_rigid_kabsch(source, target, config)
        self.assertEqual(estimate.inlier_count, 70)
        np.testing.assert_allclose(estimate.rotation, rotation, atol=1e-10)
        np.testing.assert_allclose(estimate.translation, translation, atol=1e-10)

    def test_gaussian_noise_remains_rigid(self):
        source = synthetic_points(120)
        rotation = rotation_xyz(0.03, 0.06, -0.09)
        translation = np.asarray([0.008, 0.012, -0.004])
        random = np.random.default_rng(29)
        target = (rotation @ source.T).T + translation
        target += random.normal(0.0, 0.0008, target.shape)
        config = RGBDRigidTrackerConfig(
            ransac_threshold_m=0.004,
            maximum_kabsch_rms_m=0.002,
            minimum_ransac_inliers=80,
        )
        estimate = ransac_rigid_kabsch(source, target, config)
        self.assertGreaterEqual(estimate.inlier_count, 115)
        np.testing.assert_allclose(estimate.rotation, rotation, atol=0.002)
        np.testing.assert_allclose(estimate.translation, translation, atol=0.0015)

    def test_collinear_and_near_collinear_points_are_rejected(self):
        x_values = np.linspace(-0.1, 0.1, 20)
        source = np.column_stack((x_values, 1e-9 * x_values, np.full(20, 0.8)))
        target = source + np.asarray([0.01, 0.0, 0.0])
        config = RGBDRigidTrackerConfig()
        with self.assertRaisesRegex(RigidTrackingError, "DEGENERATE_3D_GEOMETRY"):
            ransac_rigid_kabsch(source, target, config)

    def test_similarity_scale_is_not_estimated(self):
        source = synthetic_points(60)
        target = 1.15 * source
        config = RGBDRigidTrackerConfig(
            ransac_threshold_m=0.001,
            maximum_kabsch_rms_m=0.001,
            minimum_ransac_inliers=40,
            minimum_inlier_ratio=0.60,
        )
        with self.assertRaises(RigidTrackingError):
            ransac_rigid_kabsch(source, target, config)

    def test_reflection_is_never_returned_as_rotation(self):
        source = synthetic_points()
        target = source.copy()
        target[:, 0] *= -1.0
        estimate = rigid_kabsch(source, target)
        self.assertAlmostEqual(np.linalg.det(estimate.rotation), 1.0, places=12)
        self.assertGreater(estimate.rms_m, 0.01)


class DepthAndOpticalFlowTest(unittest.TestCase):
    def test_rigid_palm_mask_erodes_contour_and_depth_edges(self):
        depth = np.full((HEIGHT, WIDTH), 1000, dtype=np.uint16)
        depth[:, 100:] = 1400
        current = frame(depth_raw=depth)
        config = RGBDRigidTrackerConfig(
            palm_bbox_erosion_fraction=0.20,
            maximum_depth_from_roi_median_m=0.15,
        )
        mask = build_rigid_palm_mask(current, config)
        self.assertEqual(mask[10, 10], 0)
        self.assertEqual(mask[60, 100], 0)
        self.assertGreater(np.count_nonzero(mask[:, :75]), 0)

    def test_d455_adapter_uses_real_color_id_and_timestamp(self):
        source = frame(timestamp_s=1.25, frame_number=42)
        fake = type("D455", (), {
            "rgb": source.rgb,
            "aligned_depth_raw": source.aligned_depth_raw,
            "color_intrinsics": source.color_intrinsics,
            "depth_scale_m_per_unit": 0.001,
            "color_frame_number": 42,
            "depth_frame_number": 42,
            "color_timestamp_ms": 1250.0,
            "depth_timestamp_ms": 1250.2,
            "color_timestamp_domain": "global_time",
            "depth_timestamp_domain": "global_time",
        })()
        adapted = rgbd_tracker_frame_from_d455(fake, ROI)
        self.assertEqual(adapted.frame_number, 42)
        self.assertEqual(adapted.timestamp_s, 1.25)
        self.assertEqual(adapted.timestamp_domain, "global_time")
    def test_inverse_brown_deprojection_matches_recorded_sdk_reference(self):
        d455_intrinsics = {
            "width": 640,
            "height": 480,
            "fx": 387.07330322265625,
            "fy": 386.6429748535156,
            "ppx": 317.37567138671875,
            "ppy": 245.79708862304688,
            "distortion_model": "distortion.inverse_brown_conrady",
            "coeffs": [
                -0.05682506784796715,
                0.06772343069314957,
                -0.0002363910898566246,
                0.0011772684520110488,
                -0.02240004763007164,
            ],
        }
        pixels = np.asarray([[0.0, 0.0], [100.0, 100.0], [320.0, 240.0], [639.0, 479.0]])
        expected_sdk_2_58 = np.asarray(
            [
                [-0.83116728, -0.64314383, 1.0],
                [-0.57083362, -0.38280016, 1.0],
                [0.00677956, -0.01499321, 1.0],
                [0.83722055, 0.60890615, 1.0],
            ]
        )
        actual = deproject_pixels(pixels, np.ones(4), d455_intrinsics)
        np.testing.assert_allclose(actual, expected_sdk_2_58, atol=5e-5)

    def test_real_shi_tomasi_klt_fb_chain(self):
        config = RGBDRigidTrackerConfig()
        tracks = track_shi_tomasi_klt_fb(feature_image(), feature_image(2.0), ROI, ROI, config)
        self.assertGreaterEqual(tracks.shi_tomasi_candidates, 30)
        self.assertGreaterEqual(tracks.fb_tracks, config.minimum_fb_tracks)
        displacement = tracks.current_pixels - tracks.previous_pixels
        np.testing.assert_allclose(np.median(displacement, axis=0), [2.0, 0.0], atol=0.08)

    def test_both_depth_frames_are_required(self):
        config = RGBDRigidTrackerConfig()
        previous = frame()
        current = frame(
            shift_x=2.0,
            timestamp_s=1.0 + 1.0 / 30.0,
            frame_number=11,
            depth_raw=np.zeros((HEIGHT, WIDTH), dtype=np.uint16),
        )
        tracks = track_shi_tomasi_klt_fb(previous.rgb, current.rgb, ROI, ROI, config)
        previous_3d, current_3d = build_3d_correspondences(
            tracks.previous_pixels, tracks.current_pixels, previous, current, config
        )
        self.assertEqual(previous_3d.shape, (0, 3))
        self.assertEqual(current_3d.shape, (0, 3))


class StatefulRGBDTrackerTest(unittest.TestCase):
    def assert_valid_translation(self, dt_s):
        tracker = RGBDRigidTracker()
        initialized = tracker.process(frame(timestamp_s=2.0, frame_number=20))
        self.assertFalse(initialized.valid)
        self.assertEqual(initialized.failure_reason, "INITIALIZING")
        result = tracker.process(
            frame(shift_x=2.0, timestamp_s=2.0 + dt_s, frame_number=21)
        )
        self.assertTrue(result.valid, result.failure_reason)
        self.assertAlmostEqual(result.dt_s, dt_s, places=9)
        np.testing.assert_allclose(result.rotation_increment, np.eye(3), atol=0.003)
        np.testing.assert_allclose(
            result.translation_increment, [2.0 / FX, 0.0, 0.0], atol=0.0015
        )
        self.assertEqual(result.frame_gap, 1)
        self.assertEqual(result.tracker_age, 1)
        self.assertEqual(result.failure_reason, "NONE")
        output = result.as_dict()
        for field in (
            "valid_3d_pairs",
            "ransac_inliers",
            "inlier_ratio",
            "kabsch_rms",
            "rotation_increment",
            "translation_increment",
            "frame_gap",
            "tracker_age",
            "failure_reason",
        ):
            self.assertIn(field, output)

    def test_actual_30hz_dt(self):
        self.assert_valid_translation(1.0 / 30.0)

    def test_actual_15hz_dt(self):
        self.assert_valid_translation(1.0 / 15.0)

    def test_irregular_dt(self):
        self.assert_valid_translation(0.047)

    def test_skipped_frame_reinitializes_without_fake_transform(self):
        tracker = RGBDRigidTracker()
        tracker.process(frame(timestamp_s=1.0, frame_number=10))
        skipped = tracker.process(frame(shift_x=4.0, timestamp_s=1.067, frame_number=12))
        self.assertFalse(skipped.valid)
        self.assertEqual(skipped.failure_reason, "FRAME_GAP_EXCEEDS_MAXIMUM")
        self.assertEqual(skipped.frame_gap, 2)
        self.assertTrue(skipped.reinitialized)
        self.assertIsNone(skipped.rotation_increment)
        self.assertIsNone(skipped.translation_increment)
        recovered = tracker.process(frame(shift_x=6.0, timestamp_s=1.100, frame_number=13))
        self.assertTrue(recovered.valid, recovered.failure_reason)
        self.assertEqual(recovered.tracker_age, 1)

    def test_excessive_dt_reinitializes(self):
        tracker = RGBDRigidTracker()
        tracker.process(frame(timestamp_s=1.0, frame_number=10))
        result = tracker.process(frame(shift_x=2.0, timestamp_s=1.25, frame_number=11))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, "DT_EXCEEDS_MAXIMUM")
        self.assertIsNone(result.rotation_increment)
        self.assertIsNone(result.translation_increment)

    def test_nonincreasing_timestamp_and_domain_switch_reinitialize(self):
        tracker = RGBDRigidTracker()
        tracker.process(frame(timestamp_s=1.0, frame_number=10))
        backwards = tracker.process(frame(timestamp_s=0.9, frame_number=11))
        self.assertEqual(backwards.failure_reason, "TIMESTAMP_NON_INCREASING")
        switched = tracker.process(
            frame(timestamp_s=1.1, frame_number=12, timestamp_domain="hardware_clock")
        )
        self.assertEqual(switched.failure_reason, "TIMESTAMP_DOMAIN_CHANGED")

    def test_invalid_depth_never_becomes_identity_motion(self):
        tracker = RGBDRigidTracker()
        tracker.process(frame(timestamp_s=1.0, frame_number=10))
        invalid = tracker.process(
            frame(
                shift_x=2.0,
                timestamp_s=1.0 + 1.0 / 30.0,
                frame_number=11,
                depth_raw=np.zeros((HEIGHT, WIDTH), dtype=np.uint16),
            )
        )
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.failure_reason, "INSUFFICIENT_VALID_DEPTH_PAIRS")
        self.assertEqual(invalid.valid_3d_pairs, 0)
        self.assertIsNone(invalid.rotation_increment)
        self.assertIsNone(invalid.translation_increment)

    def test_invalid_result_cannot_carry_substitute_identity(self):
        with self.assertRaisesRegex(RigidTrackingError, "substitute transform"):
            RigidTrackResult(
                valid=False,
                valid_3d_pairs=0,
                ransac_inliers=0,
                inlier_ratio=0.0,
                kabsch_rms=None,
                rotation_increment=np.eye(3),
                translation_increment=np.zeros(3),
                frame_gap=1,
                tracker_age=0,
                failure_reason="FAILED",
                dt_s=1.0 / 30.0,
                reinitialized=True,
            )

    def test_large_unreliable_increment_is_rejected(self):
        config = RGBDRigidTrackerConfig(maximum_translation_increment_m=0.005)
        tracker = RGBDRigidTracker(config)
        tracker.process(frame(timestamp_s=1.0, frame_number=10))
        result = tracker.process(frame(shift_x=2.0, timestamp_s=1.033, frame_number=11))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, "TRANSLATION_INCREMENT_EXCEEDS_LIMIT")
        self.assertIsNone(result.rotation_increment)


class RelativeOrientationStateTest(unittest.TestCase):
    def test_startup_gap_remains_initializing_until_first_valid_transform(self):
        tracker = RGBDRelativeOrientationTracker()
        tracker.engage_clutch()
        first = tracker.process(frame(timestamp_s=1.0, frame_number=10))
        gap = tracker.process(frame(timestamp_s=1.3, frame_number=20))
        self.assertEqual(first.state, RelativeTrackingState.INITIALIZING)
        self.assertEqual(gap.state, RelativeTrackingState.INITIALIZING)
        self.assertIsNotNone(gap.accumulated_rotation)
        self.assertFalse(gap.orientation_updated)

    def test_clutch_identity_valid_accumulation_and_external_freeze(self):
        tracker = RGBDRelativeOrientationTracker()
        tracker.engage_clutch()
        first = tracker.process(frame(timestamp_s=1.0, frame_number=10))
        self.assertEqual(first.state, RelativeTrackingState.INITIALIZING)
        np.testing.assert_allclose(first.accumulated_rotation, np.eye(3))
        second = tracker.process(frame(shift_x=2, timestamp_s=1.033, frame_number=11))
        self.assertEqual(second.state, RelativeTrackingState.TRACKING)
        self.assertTrue(second.orientation_updated)
        frozen = tracker.process(
            frame(shift_x=4, timestamp_s=1.066, frame_number=12),
            externally_frozen=True,
            freeze_reason="HAMER_HAND_POSE_CHANGING",
        )
        self.assertEqual(frozen.state, RelativeTrackingState.FROZEN)
        self.assertFalse(frozen.orientation_updated)
        self.assertEqual(frozen.freeze_reason, "HAMER_HAND_POSE_CHANGING")
        np.testing.assert_allclose(
            frozen.accumulated_rotation, second.accumulated_rotation
        )

    def test_reliable_external_freeze_refreshes_loss_timer(self):
        tracker = RGBDRelativeOrientationTracker(lost_after_s=0.05)
        tracker.engage_clutch()
        tracker.process(frame(timestamp_s=1.0, frame_number=10))
        tracker.process(frame(shift_x=1, timestamp_s=1.033, frame_number=11))
        for index in range(2, 8):
            frozen = tracker.process(
                frame(shift_x=index, timestamp_s=1.0 + .033*index,
                      frame_number=10+index),
                externally_frozen=True, freeze_reason="GESTURE",
            )
            self.assertEqual(frozen.state, RelativeTrackingState.FROZEN)
            self.assertFalse(frozen.clutch_required)

    def test_external_freeze_during_startup_cannot_create_lost_state(self):
        tracker = RGBDRelativeOrientationTracker(lost_after_s=0.05)
        tracker.engage_clutch()
        tracker.process(frame(timestamp_s=1.0, frame_number=10))
        result = tracker.process(
            frame(timestamp_s=1.3, frame_number=20),
            externally_frozen=True, freeze_reason="GESTURE",
        )
        self.assertEqual(result.state, RelativeTrackingState.INITIALIZING)
        self.assertFalse(result.clutch_required)

    def test_lost_reacquire_needs_new_clutch(self):
        tracker = RGBDRelativeOrientationTracker(lost_after_s=0.05)
        tracker.engage_clutch()
        tracker.process(frame(timestamp_s=1.0, frame_number=10))
        tracker.process(frame(shift_x=2, timestamp_s=1.033, frame_number=11))
        zero = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        frozen = tracker.process(frame(
            shift_x=3, timestamp_s=1.066, frame_number=12, depth_raw=zero
        ))
        self.assertEqual(frozen.state, RelativeTrackingState.FROZEN)
        lost = tracker.process(frame(
            shift_x=4, timestamp_s=1.133, frame_number=13, depth_raw=zero
        ))
        self.assertEqual(lost.state, RelativeTrackingState.LOST)
        self.assertTrue(lost.clutch_required)
        self.assertIsNone(lost.accumulated_rotation)
        tracker.mark_roi_reacquired()
        still_lost = tracker.process(frame(shift_x=5, timestamp_s=1.166, frame_number=14))
        self.assertEqual(still_lost.state, RelativeTrackingState.LOST)
        self.assertTrue(still_lost.clutch_required)


if __name__ == "__main__":
    unittest.main()
