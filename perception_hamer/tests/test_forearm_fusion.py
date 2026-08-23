#!/usr/bin/env python3

from pathlib import Path
import math
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_hamer.src.forearm_fusion import (  # noqa: E402
    CausalForearmEstimator,
    ForearmFusionConfig,
    ForearmObservation,
    apply_forearm_fusion_to_packet,
    estimate_forearm_from_rgbd,
    fuse_wrist_frame_with_forearm,
)


INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 500.0,
    "fy": 500.0,
    "ppx": 320.0,
    "ppy": 240.0,
}


def _detection():
    return {
        "valid": True,
        "confidence": 0.95,
        "wrist_pixel": [400.0, 240.0],
        "palm_mcp_pixels": [
            [435.0, 224.0],
            [442.0, 234.0],
            [442.0, 246.0],
            [435.0, 256.0],
        ],
    }


def _synthetic_forearm_depth():
    depth = np.zeros((480, 640), dtype=np.uint16)
    # A 0.7 m forearm surface extending left (proximal) from the wrist.
    depth[220:261, 235:405] = 700
    return depth


def _observation(axis, confidence=0.9, valid=True):
    return ForearmObservation(
        valid=valid,
        axis=None if axis is None else np.asarray(axis, dtype=float),
        center_m=np.asarray([0.0, 0.0, 0.7]) if valid else None,
        confidence=confidence,
        reason="ok" if valid else "missing",
        status="tracking" if valid else "mano_only_fallback",
        age_s=0.0,
        span_m=0.14 if valid else 0.0,
        axis_ratio=4.0 if valid else 0.0,
        centerline_rms_m=0.002 if valid else float("nan"),
        point_count=800 if valid else 0,
        cross_section_count=8 if valid else 0,
        wrist_pixel=np.asarray([400.0, 240.0]) if valid else None,
        proximal_pixel=np.asarray([250.0, 240.0]) if valid else None,
        processing_ms=2.0,
    )


class RGBDForearmTest(unittest.TestCase):
    def test_metric_depth_fits_elbow_to_wrist_axis(self):
        result = estimate_forearm_from_rgbd(
            _synthetic_forearm_depth(),
            0.001,
            INTRINSICS,
            [0.112, 0.0, 0.7],
            _detection(),
        )
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(result.confidence, 0.42)
        self.assertIsNotNone(result.axis)
        self.assertGreater(float(result.axis[0]), 0.90)
        self.assertLess(abs(float(result.axis[1])), 0.15)
        self.assertGreater(result.span_m, 0.08)

    def test_same_depth_body_connection_is_clipped_to_local_wrist_region(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        # A long same-depth component emulates forearm pixels connected to the
        # torso.  Only the anatomical wrist-local 19 cm may enter the fit.
        depth[220:261, 20:405] = 700
        result = estimate_forearm_from_rgbd(
            depth,
            0.001,
            INTRINSICS,
            [0.112, 0.0, 0.7],
            _detection(),
        )
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(float(result.axis[0]), 0.90)
        self.assertLess(result.span_m, 0.19)

    def test_missing_depth_never_fabricates_forearm(self):
        result = estimate_forearm_from_rgbd(
            np.zeros((480, 640), dtype=np.uint16),
            0.001,
            INTRINSICS,
            [0.112, 0.0, 0.7],
            _detection(),
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.axis)

    def test_stale_seed_falls_back_without_invalidating_mano(self):
        estimator = CausalForearmEstimator()
        result = estimator.update(
            _synthetic_forearm_depth(),
            0.001,
            INTRINSICS,
            [0.112, 0.0, 0.7],
            _detection(),
            detection_age_s=0.30,
            now_monotonic=10.0,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "mano_only_fallback")


class WristForearmFusionTest(unittest.TestCase):
    def test_invalid_forearm_returns_exact_mano_rotation(self):
        raw = np.eye(3)
        fused, diagnostics = fuse_wrist_frame_with_forearm(
            raw, _observation(None, valid=False)
        )
        np.testing.assert_array_equal(fused, raw)
        self.assertFalse(diagnostics["applied"])

    def test_forearm_regularizes_longitudinal_axis_but_is_bounded(self):
        angle = math.radians(30.0)
        forearm = [math.sin(angle), 0.0, math.cos(angle)]
        settings = ForearmFusionConfig(maximum_fusion_weight=0.20)
        fused, diagnostics = fuse_wrist_frame_with_forearm(
            np.eye(3), _observation(forearm, confidence=1.0), settings
        )
        self.assertTrue(diagnostics["applied"])
        self.assertAlmostEqual(diagnostics["fusion_weight"], 0.20)
        self.assertGreater(float(fused[0, 2]), 0.0)
        self.assertLess(diagnostics["correction_deg"], 8.0)
        np.testing.assert_allclose(fused.T @ fused, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(fused)), 1.0, places=10)

    def test_roll_stays_mano_when_axis_already_agrees(self):
        angle = math.radians(42.0)
        roll = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        fused, diagnostics = fuse_wrist_frame_with_forearm(
            roll, _observation([0.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(fused, roll, atol=1e-10)
        self.assertLess(diagnostics["correction_deg"], 1e-6)

    def test_packet_preserves_raw_rotation_and_marks_fusion(self):
        packet = {
            "palm_rotation_row_major": np.eye(3).reshape(-1).tolist(),
            "orientation_source": "MANO_FILTERED",
            "confidence": [0.8] * 6,
        }
        output = apply_forearm_fusion_to_packet(
            packet, _observation([0.25, 0.0, 0.9682458])
        )
        self.assertTrue(output["forearm_fusion"]["applied"])
        self.assertIn("D455_RGBD_FOREARM", output["orientation_source"])
        np.testing.assert_array_equal(
            output["forearm_fusion"]["raw_mano_rotation_row_major"],
            packet["palm_rotation_row_major"],
        )
        np.testing.assert_array_equal(packet["palm_rotation_row_major"], np.eye(3).reshape(-1))

    def test_held_orientation_is_never_changed_by_forearm_fusion(self):
        angle = math.radians(37.0)
        held = np.asarray([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        packet = {
            "palm_rotation_row_major": held.reshape(-1).tolist(),
            "orientation_source": "HELD_LAST_TRUSTED_MANO_ORIENTATION",
            "orientation_channel_valid": False,
            "orientation_held": True,
            "confidence": [0.8, 0.8, 0.8, 0.0, 0.0, 0.0],
        }
        output = apply_forearm_fusion_to_packet(
            packet, _observation([0.5, 0.0, 0.8660254])
        )
        self.assertFalse(output["forearm_fusion"]["applied"])
        self.assertEqual(
            output["forearm_fusion"]["fallback"],
            "ORIENTATION_HELD_NO_FOREARM_FUSION",
        )
        np.testing.assert_array_equal(
            output["palm_rotation_row_major"],
            packet["palm_rotation_row_major"],
        )


if __name__ == "__main__":
    unittest.main()
