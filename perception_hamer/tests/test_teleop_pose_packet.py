#!/usr/bin/env python3

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_hamer.src.teleop_pose_packet import (  # noqa: E402
    build_invalid_teleop_packet, build_live_teleop_packet, foreground_depth_component,
    metric_wrist_from_arrays, metric_wrist_ring_from_arrays,
)


INTRINSICS = {
    "fx": 100.0, "fy": 100.0, "ppx": 100.0, "ppy": 100.0,
    "coeffs": [0.0]*5, "distortion_model": "none",
}


class TeleopPosePacketTest(unittest.TestCase):
    @staticmethod
    def _ring_vertices():
        vertices = np.zeros((778, 3), dtype=float)
        angles = 2.0*np.pi*np.arange(16)/16.0
        vertices[:16, 0] = 0.05*np.cos(angles)
        vertices[:16, 1] = 0.04*np.sin(angles)
        return vertices

    def test_invalid_heartbeat_contains_identity_but_no_pose_geometry(self):
        packet = build_invalid_teleop_packet(
            "test", 9, 12.5, "no_real_hand", 4, 3, hand_is_right=False
        )
        self.assertFalse(packet["valid"])
        self.assertTrue(packet["hand_identity_present"])
        self.assertFalse(packet["hand_is_right"])
        self.assertEqual(packet["presence_generation"], 4)
        self.assertEqual(packet["active_hand_generation"], 3)
        self.assertNotIn("wrist_position_m", packet)
        self.assertNotIn("palm_rotation_row_major", packet)

    def test_16_point_wrist_ring_center_uses_projected_depth_hull(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[116:141, 113:144] = 1000
        point, confidence, diagnostics, pixels = metric_wrist_ring_from_arrays(
            self._ring_vertices(), np.arange(16), [0.0, 0.0, 1.0],
            [256.0, 256.0], [[1, 0, 0], [0, 1, 0]], depth, 0.001,
            INTRINSICS,
        )
        np.testing.assert_allclose(pixels.mean(axis=0), [128.0, 128.0], atol=1e-6)
        np.testing.assert_allclose(point, [0.28, 0.28, 1.0], atol=1e-12)
        self.assertGreater(confidence, 0.8)
        self.assertEqual(
            diagnostics["reference"],
            "mean_of_16_mano_wrist_opening_vertices",
        )

    def test_foreground_component_rejects_supported_background_surface(self):
        near = np.full(20, 0.65)
        far = np.full(100, 1.05)
        selected, diagnostics = foreground_depth_component(
            np.concatenate((near, far)), mask_count=120
        )
        self.assertAlmostEqual(float(np.median(selected)), 0.65)
        self.assertEqual(diagnostics["component_count"], 2)
        self.assertGreater(diagnostics["separation_m"], 0.3)

    def test_wrist_ring_depth_hole_expands_but_keeps_ring_center_ray(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        # The wrist ring projects around (128,128).  Leave the ring/hole and
        # inner fallbacks empty; valid nearby forearm depth starts 20 px away.
        depth[126:131, 148:154] = 700
        point, confidence, diagnostics, pixels = metric_wrist_ring_from_arrays(
            self._ring_vertices(), np.arange(16), [0.0, 0.0, 1.0],
            [256.0, 256.0], [[1, 0, 0], [0, 1, 0]], depth, 0.001,
            INTRINSICS,
        )
        np.testing.assert_allclose(pixels.mean(axis=0), [128.0, 128.0], atol=1e-6)
        # Depth comes from nearby support, but XYZ remains the ring-centre ray.
        np.testing.assert_allclose(point, [0.196, 0.196, 0.7], atol=1e-12)
        self.assertTrue(diagnostics["center_patch_fallback_used"])
        self.assertEqual(diagnostics["center_patch_fallback_radius_px"], 24)
        self.assertGreater(confidence, 0.0)
        self.assertLess(confidence, 0.5)

    def test_short_depth_hole_holds_only_z_with_low_confidence(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        point, confidence, diagnostics, _pixels = metric_wrist_ring_from_arrays(
            self._ring_vertices(), np.arange(16), [0.0, 0.0, 1.0],
            [256.0, 256.0], [[1, 0, 0], [0, 1, 0]], depth, 0.001,
            INTRINSICS,
            reference_depth_m=0.7,
            reference_depth_age_s=0.08,
        )
        np.testing.assert_allclose(point, [0.196, 0.196, 0.7], atol=1e-12)
        self.assertTrue(diagnostics["depth_reference_hold_used"])
        self.assertLessEqual(confidence, 0.08)

    def test_depth_hold_expires_after_120_ms(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "no aligned depth"):
            metric_wrist_ring_from_arrays(
                self._ring_vertices(), np.arange(16), [0.0, 0.0, 1.0],
                [256.0, 256.0], [[1, 0, 0], [0, 1, 0]], depth, 0.001,
                INTRINSICS,
                reference_depth_m=0.7,
                reference_depth_age_s=0.121,
            )

    def test_hamer_wrist_pixel_plus_depth_is_metric_d455_position(self):
        normalized = np.zeros((21, 2), dtype=float)
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[125:132, 125:132] = 1000
        point, confidence, pixels = metric_wrist_from_arrays(
            normalized, [[1, 0, 0], [0, 1, 0]], depth, 0.001,
            INTRINSICS)
        np.testing.assert_allclose(pixels[0], [128, 128])
        np.testing.assert_allclose(point, [0.28, 0.28, 1.0], atol=1.0e-12)
        self.assertEqual(confidence, 1.0)

    def test_wrist_depth_hole_uses_nearest_larger_neighborhood(self):
        normalized = np.zeros((21, 2), dtype=float)
        depth = np.zeros((480, 640), dtype=np.uint16)
        # No valid depth exists in the legacy 7x7 wrist patch.  The adaptive
        # search finds the nearest ring while the 3-D point remains wrist-ray
        # deprojection at pixel (128, 128).
        depth[128, 133:141] = 800
        point, confidence, _ = metric_wrist_from_arrays(
            normalized, [[1, 0, 0], [0, 1, 0]], depth, 0.001,
            INTRINSICS)
        np.testing.assert_allclose(point, [0.224, 0.224, 0.8], atol=1.0e-12)
        self.assertGreater(confidence, 0.0)
        self.assertLess(confidence, 1.0)

    def test_palm_depth_fallback_keeps_wrist_pixel_as_reference(self):
        normalized = np.zeros((21, 2), dtype=float)
        normalized[[5, 9, 13, 17], 0] = 30.0/256.0
        depth = np.zeros((480, 640), dtype=np.uint16)
        # Projected MCP pixels are (158,128), while the wrist stays (128,128).
        depth[126:131, 156:161] = 600
        point, confidence, _ = metric_wrist_from_arrays(
            normalized, [[1, 0, 0], [0, 1, 0]], depth, 0.001,
            INTRINSICS)
        np.testing.assert_allclose(point, [0.168, 0.168, 0.6], atol=1.0e-12)
        self.assertGreaterEqual(confidence, 0.2)

    def test_all_missing_depth_is_rejected(self):
        normalized = np.zeros((21, 2), dtype=float)
        depth = np.zeros((480, 640), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "wrist/palm"):
            metric_wrist_from_arrays(
                normalized, [[1, 0, 0], [0, 1, 0]], depth, 0.001,
                INTRINSICS)

    def test_live_packet_never_uses_hamer_projection_translation(self):
        normalized = np.zeros((21, 2), dtype=float)
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[125:132, 125:132] = 750
        result = SimpleNamespace(
            is_right=True,
            pred_keypoints_2d_crop_normalized=normalized,
            quality={"affine_original_to_crop": [[1,0,0],[0,1,0]],
                     "bbox_visible_fraction": 0.8},
            timestamp=12.5,
            hamer_crop_projection_translation=np.array([999, 999, 999]),
        )
        frame = SimpleNamespace(
            aligned_depth_raw=depth, depth_scale_m_per_unit=0.001,
            color_intrinsics=INTRINSICS,
        )
        estimates = {"mano_joint_palm_frame": {
            "valid": True, "rotation": np.eye(3).tolist()}}
        packet = build_live_teleop_packet(
            result, estimates, frame, SimpleNamespace(confidence=0.9),
            "test", 3, 2, 1)
        np.testing.assert_allclose(packet["wrist_position_m"], [0.21, 0.21, 0.75])
        self.assertEqual(packet["position_source"],
                         "HAMER_WRIST_RAY_PLUS_D455_ADAPTIVE_ALIGNED_DEPTH")
        self.assertNotEqual(packet["wrist_position_m"], [999, 999, 999])

    def test_crop_and_so3_filter_quality_scale_packet_confidence(self):
        normalized = np.zeros((21, 2), dtype=float)
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[125:132, 125:132] = 1000
        result = SimpleNamespace(
            is_right=False,
            pred_keypoints_2d_crop_normalized=normalized,
            quality={"affine_original_to_crop": [[1,0,0],[0,1,0]],
                     "bbox_visible_fraction": 0.8},
            timestamp=12.5,
        )
        frame = SimpleNamespace(
            aligned_depth_raw=depth, depth_scale_m_per_unit=0.001,
            color_intrinsics=INTRINSICS,
        )
        estimates = {
            "mano_joint_palm_frame": {
                "valid": True,
                "rotation": np.eye(3).tolist(),
                "filter_confidence": 0.30,
                "orientation_source": "FILTERED_TEST",
            },
            "teleop_crop_quality": 0.5,
            "palm_orientation_filter": {"valid": True},
        }
        packet = build_live_teleop_packet(
            result, estimates, frame, SimpleNamespace(confidence=0.9),
            "test", 4, 2, 1,
        )
        np.testing.assert_allclose(packet["confidence"][:3], [0.36] * 3)
        np.testing.assert_allclose(packet["confidence"][3:], [0.30] * 3)
        self.assertEqual(packet["orientation_source"], "FILTERED_TEST")
        self.assertEqual(packet["crop_quality"], 0.5)

    def test_held_orientation_keeps_metric_position_packet_valid(self):
        normalized = np.zeros((21, 2), dtype=float)
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[125:132, 125:132] = 1000
        result = SimpleNamespace(
            is_right=True,
            pred_keypoints_2d_crop_normalized=normalized,
            quality={"affine_original_to_crop": [[1, 0, 0], [0, 1, 0]],
                     "bbox_visible_fraction": 0.8},
            timestamp=12.5,
        )
        frame = SimpleNamespace(
            aligned_depth_raw=depth, depth_scale_m_per_unit=0.001,
            color_intrinsics=INTRINSICS,
        )
        estimates = {
            "mano_joint_palm_frame": {
                "valid": True,
                "rotation": np.eye(3).tolist(),
                "filter_confidence": 0.9,
                "orientation_channel_valid": False,
                "orientation_held": True,
                "failure_reason": "causal_so3_filter:test_jump",
                "orientation_source": "HELD_LAST_TRUSTED_MANO_ORIENTATION",
            },
            "teleop_crop_quality": 0.5,
        }
        packet = build_live_teleop_packet(
            result, estimates, frame, SimpleNamespace(confidence=0.9),
            "test", 5, 4, 3,
        )
        self.assertTrue(packet["valid"])
        self.assertFalse(packet["orientation_channel_valid"])
        self.assertTrue(packet["orientation_held"])
        np.testing.assert_allclose(packet["confidence"][:3], [0.36] * 3)
        np.testing.assert_allclose(packet["confidence"][3:], [0.0] * 3)
        self.assertEqual(
            packet["invalid_reason"], "causal_so3_filter:test_jump"
        )

    def test_live_packet_prefers_16_point_control_reference(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[116:141, 113:144] = 900
        result = SimpleNamespace(
            is_right=True,
            pred_vertices_mano_right_canonical=self._ring_vertices(),
            hamer_crop_projection_translation=np.array([0.0, 0.0, 1.0]),
            hamer_nominal_crop_focal_length=np.array([256.0, 256.0]),
            pred_keypoints_2d_crop_normalized=np.zeros((21, 2)),
            quality={
                "affine_original_to_crop": [[1, 0, 0], [0, 1, 0]],
                "bbox_visible_fraction": 1.0,
            },
            timestamp=13.0,
        )
        frame = SimpleNamespace(
            aligned_depth_raw=depth,
            depth_scale_m_per_unit=0.001,
            color_intrinsics=INTRINSICS,
        )
        estimates = {
            "mano_joint_palm_frame": {
                "valid": True, "rotation": (2.0*np.eye(3)).tolist(),
            },
            "control_wrist_frame": {
                "valid": True,
                "rotation": np.eye(3).tolist(),
                "reference_kind": "MANO_WRIST_RING_16",
                "filter_confidence": 0.8,
                "orientation_source": "RING_FILTER_TEST",
                "quality": {"wrist_loop_vertex_indices": list(range(16))},
            },
        }
        packet = build_live_teleop_packet(
            result, estimates, frame, SimpleNamespace(confidence=1.0),
            "ring-test", 8, 6, 2,
        )
        self.assertEqual(packet["control_reference"], "MANO_WRIST_RING_16")
        self.assertEqual(
            packet["position_source"],
            "MANO_WRIST_RING_16_PROJECTED_HULL_PLUS_D455_ALIGNED_DEPTH",
        )
        self.assertEqual(packet["orientation_source"], "RING_FILTER_TEST")
        np.testing.assert_allclose(packet["palm_rotation_row_major"], np.eye(3).reshape(-1))
        self.assertAlmostEqual(packet["position_diagnostics"]["depth_m"], 0.9)


if __name__ == "__main__":
    unittest.main()
