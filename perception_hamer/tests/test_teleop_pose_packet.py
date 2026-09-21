#!/usr/bin/env python3

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_hamer.src.teleop_pose_packet import (  # noqa: E402
    ProximalWristDepthFilter,
    build_invalid_teleop_packet, build_live_teleop_packet, foreground_depth_component,
    invalidate_stale_teleop_observation, metric_wrist_from_arrays,
    metric_wrist_ring_from_arrays,
    metric_wrist_zero_from_arrays,
    metric_palm_skin_from_arrays,
)


INTRINSICS = {
    "fx": 100.0, "fy": 100.0, "ppx": 100.0, "ppy": 100.0,
    "coeffs": [0.0]*5, "distortion_model": "none",
}


class PalmSkinDepthTest(unittest.TestCase):
    def test_native_origins_share_measured_skin_without_offset(self):
        points = np.array([[100,140],[70,70],[90,60],[110,60],[130,70]])
        depth = np.full((200,200),800,np.uint16)
        outputs = [metric_palm_skin_from_arrays(points,origin,depth,.001,INTRINSICS)
                   for origin in ([100,140],[110,140])]
        np.testing.assert_allclose(outputs[0][0],[0,.32,.8])
        np.testing.assert_allclose(outputs[1][0],[.08,.32,.8])
        for _,_,diagnostics in outputs:
            self.assertEqual(diagnostics['model_depth_offset_m'],0)
            self.assertFalse(diagnostics['depth_reference_hold_used'])
        depth[:] = 0
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT_MEASURED_DEPTH'):
            metric_palm_skin_from_arrays(points,[100,140],depth,.001,INTRINSICS)

    def test_depth_boundary_is_not_accepted_as_uniform_skin(self):
        points = np.array([[100,140],[70,70],[90,60],[110,60],[130,70]])
        depth=np.full((200,200),800,np.uint16)
        depth[:,100:]=900
        with self.assertRaisesRegex(ValueError,'DEPTH_DISCONTINUITY'):
            metric_palm_skin_from_arrays(points,[100,140],depth,.001,INTRINSICS)


class JointZeroLocalDepthTest(unittest.TestCase):
    def geometry(self):
        # Two skin planes around joint 0: front at -20 mm, rear at +20 mm.
        vertices = np.array([[x, y, z] for z in (-.02, .02)
                             for x, y in [(-.04,-.04),(.04,-.04),(.04,.04),(-.04,.04)]])
        return vertices, np.zeros((21, 3)), np.zeros((21, 2)), np.array(
            [[0,1,2],[0,2,3],[4,5,6],[4,6,7]])

    def measure(self, depth, shift=0., affine=None):
        vertices, joints, pixels, faces = self.geometry()
        vertices[:, 2] += shift; joints[:, 2] += shift
        return metric_wrist_zero_from_arrays(vertices, joints, pixels, faces,
            [0,0,1-shift], [256,256], affine or [[1,0,0],[0,1,0]],
            depth, .001, INTRINSICS)

    def test_front_surface_offset_and_real_translation_are_retained(self):
        depth = np.full((256,256), 1000, np.uint16)
        point, confidence, diagnostic = self.measure(depth)
        np.testing.assert_allclose(point, [.2856,.2856,1.02], atol=1e-7)
        self.assertAlmostEqual(diagnostic['model_surface_to_joint0_offset_m'], .02)
        self.assertAlmostEqual(confidence, 1.)
        moved = self.measure(depth-100)[0]
        self.assertAlmostEqual(point[2]-moved[2], .1)
        # A change of model origin/crop-camera translation is not real motion.
        np.testing.assert_allclose(self.measure(depth, shift=.4)[0], point, atol=1e-7)

    def test_left_crop_reflection_maps_joint_zero_to_original_color_ray(self):
        point, _, diagnostic = self.measure(np.full((256,256),1000,np.uint16),
            affine=[[-1,0,240],[0,1,0]])
        np.testing.assert_allclose(diagnostic['center_pixel'], [112,128])
        np.testing.assert_allclose(point, [.1224,.2856,1.02], atol=1e-7)

    def test_internal_offset_changes_with_model_orientation(self):
        vertices, joints, pixels, faces = self.geometry()
        angle = np.pi/6
        rotation = np.array([[np.cos(angle),0,np.sin(angle)],
                             [0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        point, _, diagnostic = metric_wrist_zero_from_arrays(
            vertices @ rotation.T, joints, pixels, faces, [0,0,1], [256,256],
            [[1,0,0],[0,1,0]], np.full((256,256),1000,np.uint16), .001, INTRINSICS)
        self.assertAlmostEqual(point[2], 1.+.02/np.cos(angle), places=7)
        self.assertGreater(diagnostic['model_surface_to_joint0_offset_m'], .02)

    def test_open_wrist_cut_does_not_pair_front_skin_with_rear_wall(self):
        # Analytic forearm cylinder, bone at exactly 1 m. The model ends at
        # its wrist ring while the measured real skin continues past it.
        # Previously the rear wall yielded a plausible but wrong 0.93--0.95 m.
        vertices = np.zeros((778, 3))
        angles = np.arange(16) * 2*np.pi/16
        for start, y in ((0, 0.), (16, .1)):
            vertices[start:start+16] = np.column_stack((
                .025*np.cos(angles), np.full(16, y), .025*np.sin(angles)))
        vertices[32] = [0, .1, 0]
        faces = []
        for i in range(16):
            j = (i+1) % 16
            faces.extend([[i, j, 16+j], [i, 16+j, 16+i], [16+i, 16+j, 32]])
        u, v = np.meshgrid(np.arange(256), np.arange(256))
        rays = np.stack(((u-128)/256, (v-128)/256, np.ones_like(u)), axis=-1)
        intrinsics = dict(INTRINSICS, fx=256., fy=256., ppx=128., ppy=128.)
        for degrees in (15., 30., 45.):
            c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
            rotation = np.array([[1,0,0], [0,c,-s], [0,s,c]])
            axis = rotation[:, 1]
            normal_plane = np.eye(3) - np.outer(axis, axis)
            center = np.array([0., 0., 1.])
            a = np.einsum('...i,ij,...j->...', rays, normal_plane, rays)
            b = -2 * (rays @ (normal_plane @ center))
            discriminant = b*b - 4*a*(center @ normal_plane @ center - .025**2)
            z = (-b - np.sqrt(np.maximum(0., discriminant))) / (2*a)
            depth = np.where(discriminant >= 0., np.rint(z*1000), 0).astype(np.uint16)
            joints = np.zeros((21, 3))
            joints[[5,9,13,17]] = np.array([0., .07, 0.]) @ rotation.T
            point, _, _ = metric_wrist_zero_from_arrays(
                vertices @ rotation.T, joints, np.zeros((21,2)), np.array(faces),
                [0,0,1], [256,256], [[1,0,0],[0,1,0]], depth, .001, intrinsics)
            np.testing.assert_allclose(point, [0., 0., 1.], atol=.001)

    def test_hole_and_mixed_background_are_invalid_without_distant_fallback(self):
        depth = np.full((256,256),1000,np.uint16)
        depth[126:131,126:131] = 0
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT_LOCAL'):
            self.measure(depth)
        depth[126:131,126:131] = 1000
        depth[126:131,129:131] = 1400
        with self.assertRaisesRegex(ValueError, 'DEPTH_DISCONTINUITY'):
            self.measure(depth)

    def test_minority_depth_edge_outliers_do_not_disable_consistent_wrist_measurement(self):
        for outlier in (940, 1400):
            depth = np.full((256,256),1000,np.uint16)
            depth[126:131,126] = outlier
            point, _, diagnostic = self.measure(depth)
            self.assertAlmostEqual(point[2], 1.02)
            self.assertEqual(diagnostic['sample_count'], 20)
            self.assertEqual(diagnostic['depth_outlier_count'], 5)

    def test_packet_uses_no_ring_and_does_not_change_orientation(self):
        vertices, joints, pixels, faces = self.geometry()
        result = SimpleNamespace(is_right=False, timestamp=1.,
            pred_vertices_mano_right_canonical=vertices,
            pred_keypoints_3d_mano_right_canonical=joints,
            pred_keypoints_2d_crop_normalized=pixels,
            hamer_crop_projection_translation=np.array([0.,0.,1.]),
            hamer_nominal_crop_focal_length=np.array([256.,256.]),
            quality={'affine_original_to_crop':[[1,0,0],[0,1,0]],'bbox_visible_fraction':1.})
        control = dict(valid=True, rotation=np.eye(3).tolist(),
            reference_kind='MANO_FIVE_POINT_PALM_USER_XYZ', quality={})
        frame = SimpleNamespace(aligned_depth_raw=np.full((256,256),1000,np.uint16),
            depth_scale_m_per_unit=.001, color_intrinsics=INTRINSICS)
        with patch('perception_hamer.src.teleop_pose_packet.metric_wrist_ring_from_arrays',
                   side_effect=AssertionError('ring must not be used')):
            packet = build_live_teleop_packet(result, {'control_wrist_frame':control},
                frame, SimpleNamespace(confidence=1.), 'joint0', 1, 1, 1,
                wrist_depth_mode='joint0-surface', mano_faces=faces,
                reference_depth_m=8., reference_depth_age_s=0.)
        np.testing.assert_allclose(packet['wrist_position_m'], [.2856,.2856,1.02],atol=1e-7)
        np.testing.assert_allclose(np.reshape(packet['palm_rotation_row_major'],(3,3)),np.eye(3))
        self.assertEqual(packet['position_source'],'HAMER_WRIST_0_LOCAL_RGBD_WITH_MODEL_SURFACE_OFFSET')


class TeleopPosePacketTest(unittest.TestCase):
    def test_new_palm_uses_wrist_zero_without_shifting_cached_surface_depth(self):
        vertices = self._ring_vertices()
        source = vertices.copy();source[:, 0] *= -1
        offset = np.array([.01, -.005, .02])
        joints = np.zeros((21, 3));joints[0] = source[:16].mean(0) + offset
        result = SimpleNamespace(
            is_right=False, timestamp=13.,
            pred_vertices_mano_right_canonical=vertices,
            pred_vertices_source_camera_axes=source,
            pred_keypoints_3d_source_camera_axes=joints,
            hamer_crop_projection_translation=np.array([0., 0., 1.]),
            hamer_nominal_crop_focal_length=np.array([256., 256.]),
            quality={"affine_original_to_crop": [[1,0,0],[0,1,0]], "bbox_visible_fraction": 1.})
        frame = SimpleNamespace(aligned_depth_raw=np.zeros((480,640),np.uint16),
            depth_scale_m_per_unit=.001, color_intrinsics=INTRINSICS)
        control = dict(valid=True, rotation=np.eye(3).tolist(),
            reference_kind="MANO_FIVE_POINT_PALM_USER_XYZ",
            quality={"wrist_loop_vertex_indices": list(range(16))})
        with patch("perception_hamer.src.teleop_pose_packet.metric_wrist_ring_from_arrays",
                   return_value=(np.array([.1,.2,.9]), 1., {"depth_m":.9}, None)) as depth:
            packet = build_live_teleop_packet(result, {"control_wrist_frame":control},
                frame, SimpleNamespace(confidence=1.), "new-palm", 1, 1, 1,
                reference_depth_m=.91, required_reference_kind=control["reference_kind"])
        self.assertEqual(depth.call_args.kwargs["reference_depth_m"], .91)
        np.testing.assert_allclose(packet["wrist_position_m"], [.11,.195,.92])
        self.assertEqual(packet["position_diagnostics"]["depth_m"], .9)
        self.assertEqual(packet["position_diagnostics"]["control_origin"], "MANO_WRIST_JOINT_0")

    @staticmethod
    def _proximal_diagnostic(wrist_z, surface_z, legacy_z):
        return {"depth_m": wrist_z, "proximal_surface_association": {
            "applied": True, "observed_surface_depth_m": surface_z,
            "legacy_depth_m": legacy_z}}

    def test_depth_estimator_switch_does_not_create_a_five_cm_movement(self):
        state = ProximalWristDepthFilter()
        identity = ("session", 1, False)
        first = state.update([.105, .21, 1.05],
                             self._proximal_diagnostic(1.05, 1.02, 1.0),
                             1.0, identity)
        missing = state.update([.1, .2, 1.0], {"depth_m": 1.0}, 1.07, identity)
        recovered = state.update([.105, .21, 1.05],
                                 self._proximal_diagnostic(1.05, 1.02, 1.0),
                                 1.14, identity)
        np.testing.assert_allclose(missing, first, atol=1e-12)
        np.testing.assert_allclose(recovered, first, atol=1e-12)

    def test_real_surface_depth_motion_is_not_low_pass_filtered(self):
        state = ProximalWristDepthFilter()
        state.update([0, 0, 1.05], self._proximal_diagnostic(1.05, 1.02, 1.0),
                     1.0, "hand")
        moved = state.update([0, 0, .95], self._proximal_diagnostic(.95, .92, .9),
                             1.07, "hand")
        self.assertAlmostEqual(moved[2], .95)

    def test_depth_hold_does_not_double_apply_the_estimator_offset(self):
        state = ProximalWristDepthFilter()
        state.update([0, 0, 1.05], self._proximal_diagnostic(1.05, 1.02, 1.0),
                     1.0, "hand")
        diagnostic = {"depth_m": 1.05, "depth_reference_hold_used": True}
        held = state.update([0, 0, 1.05], diagnostic, 1.07, "hand")
        self.assertAlmostEqual(held[2], 1.05)
        self.assertAlmostEqual(diagnostic["depth_offset_filter"]["legacy_depth_m"], 1.0)

    def test_depth_offset_never_crosses_a_hand_identity_change(self):
        state = ProximalWristDepthFilter()
        state.update([0, 0, 1.05], self._proximal_diagnostic(1.05, 1.02, 1.0),
                     1.0, "first")
        changed = state.update([0, 0, .75], {"depth_m": .75}, 1.07, "second")
        self.assertAlmostEqual(changed[2], .75)

    def test_proximal_surface_ignores_nearer_samples_outside_the_palm_region(self):
        vertices = self._ring_vertices()
        grid = np.array([(x, y) for x in np.linspace(-.10, .10, 28)
                         for y in np.linspace(-.10, .10, 27)])
        vertices[16:16+len(grid), :2] = grid
        depth = np.full((480, 640), 1400, dtype=np.uint16)
        depth[128:155, 106:151] = 1000
        # A separate nearer patch satisfies the old first-window support
        # rule, despite not belonging to the proximal hand observation.
        depth[102:112, 102:154] = 960
        args = (vertices, np.arange(16), [0, 0, 1], [256, 256],
                [[1, 0, 0], [0, 1, 0]], depth, .001, INTRINSICS)
        legacy, legacy_confidence, _, _ = metric_wrist_ring_from_arrays(*args)
        corrected, confidence, diagnostic, _ = metric_wrist_ring_from_arrays(
            *args, palm_anchor_pixels=[[128, 149], [106, 106],
                                      [120, 103], [135, 103], [152, 106]])
        self.assertAlmostEqual(legacy[2], .960)
        np.testing.assert_allclose(corrected, [.28, .28, 1.0], atol=1e-9)
        self.assertLessEqual(confidence, legacy_confidence)
        self.assertTrue(diagnostic["proximal_surface_association"]["applied"])
        self.assertEqual(diagnostic["reference"], "mean_of_16_mano_wrist_opening_vertices")

    def test_unusable_palm_region_preserves_legacy_depth_and_validity(self):
        vertices = self._ring_vertices()
        depth = np.full((480, 640), 1000, dtype=np.uint16)
        args = (vertices, np.arange(16), [0, 0, 1], [256, 256],
                [[1, 0, 0], [0, 1, 0]], depth, .001, INTRINSICS)
        old = metric_wrist_ring_from_arrays(*args)
        new = metric_wrist_ring_from_arrays(*args, palm_anchor_pixels=[[20, 20]]*5)
        np.testing.assert_array_equal(old[0], new[0])
        self.assertEqual(old[1], new[1])
        self.assertFalse(new[2]["proximal_surface_association"]["applied"])

    def test_visible_hand_padding_does_not_penalize_finger_crop(self):
        from perception_hamer.src.crop_quality import bbox_crop_quality
        # Hand landmarks were checked independently; only padding crosses
        # the sensor border. Without that evidence retain the old penalty.
        box = [180, 180, 420, 510]
        old = bbox_crop_quality(box, box, 640, 480)
        confirmed = bbox_crop_quality(box, box, 640, 480, content_fully_visible=True)
        self.assertLess(old, .55)
        self.assertGreater(confirmed, .90)
        self.assertEqual(bbox_crop_quality([0, 0, 0, 0], None, 640, 480,
                                          content_fully_visible=True), 0.0)

    def test_visible_hand_depth_rejects_majority_background_at_wrist(self):
        vertices = self._ring_vertices()
        grid = np.array([(x, y) for x in np.linspace(-.10, .10, 28)
                         for y in np.linspace(-.10, .10, 27)])
        vertices[16:16+len(grid), :2] = grid
        depth = np.full((480, 640), 1400, dtype=np.uint16)
        # A small part of the hand has valid foreground depth; the wrist
        # centre and most projected pixels instead contain the background.
        depth[102:111, 102:154] = 1000
        point, _, diagnostics, _ = metric_wrist_ring_from_arrays(
            vertices, np.arange(16), [0,0,1], [256,256],
            [[1,0,0],[0,1,0]], depth, .001, INTRINSICS)
        np.testing.assert_allclose(point, [.28,.28,1.0], atol=1e-9)
        self.assertLess(diagnostics["foreground_sample_fraction"], .5)

    def test_configured_wrist_reference_never_falls_back_after_an_exception(self):
        result = SimpleNamespace(is_right=False)
        estimates = {"mano_joint_palm_frame": {
            "valid": True, "rotation": np.eye(3).tolist()}}
        with self.assertRaisesRegex(ValueError, "required control wrist frame"):
            build_live_teleop_packet(
                result, estimates, None, None, "test", 1, 1, 1,
                required_reference_kind="MANO_WRIST_RING_16")

    def test_configured_wrist_reference_rejects_another_valid_basis(self):
        result = SimpleNamespace(is_right=False)
        estimates = {"control_wrist_frame": {
            "valid": True, "rotation": np.eye(3).tolist(),
            "reference_kind": "MANO_JOINT_0_PALM_FRAME_LEGACY"}}
        with self.assertRaisesRegex(ValueError, "differs from configured"):
            build_live_teleop_packet(
                result, estimates, None, None, "test", 1, 1, 1,
                required_reference_kind="MANO_WRIST_RING_16")

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

    def test_overdue_valid_pose_becomes_identity_preserving_heartbeat(self):
        packet = {
            "valid": True,
            "wrist_position_m": [0.1, 0.2, 0.7],
            "palm_rotation_row_major": np.eye(3).reshape(-1).tolist(),
            "confidence": [0.9] * 6,
            "hand_identity_present": True,
            "hand_is_right": True,
            "presence_generation": 4,
            "active_hand_generation": 2,
            "timing": {"capture_to_publish_s": 0.251},
            "finger_observation": {
                "valid": True,
                "flexion": [0.1] * 5,
                "confidence": 0.8,
                "invalid_reason": "",
            },
        }
        stale = invalidate_stale_teleop_observation(packet, 0.25)
        self.assertFalse(stale["valid"])
        self.assertTrue(stale["hand_identity_present"])
        self.assertEqual(stale["presence_generation"], 4)
        self.assertNotIn("wrist_position_m", stale)
        self.assertNotIn("palm_rotation_row_major", stale)
        self.assertEqual(stale["confidence"], [0.0] * 6)
        self.assertFalse(stale["finger_observation"]["valid"])
        self.assertIn("SOURCE_PIPELINE_LATENCY_EXCEEDED", stale["invalid_reason"])

    def test_fresh_pose_is_not_modified(self):
        packet = {
            "valid": True,
            "timing": {"capture_to_publish_s": 0.249},
        }
        self.assertEqual(
            invalidate_stale_teleop_observation(packet, 0.25), packet
        )

    def test_16_point_wrist_ring_center_uses_visible_hand_depth(self):
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
            INTRINSICS, use_visible_hand_depth=False,
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
        with self.assertRaisesRegex(ValueError, "no supported visible-hand depth"):
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

        # Verified hand visibility only changes the finger channel. It must
        # not increase position/orientation confidence as a side effect.
        from perception_hamer.tests.test_finger_observation import open_hand
        result.pred_keypoints_3d_source_camera_axes = open_hand()
        estimates["teleop_finger_visible_fraction"] = 1.0
        estimates["teleop_finger_crop_quality"] = 0.95
        fixed = build_live_teleop_packet(
            result, estimates, frame, SimpleNamespace(confidence=0.9),
            "test", 5, 2, 1)
        self.assertTrue(fixed["finger_observation"]["valid"])
        self.assertAlmostEqual(fixed["finger_observation"]["confidence"], .855)
        np.testing.assert_allclose(fixed["confidence"], packet["confidence"])
        np.testing.assert_allclose(fixed["wrist_position_m"], packet["wrist_position_m"])

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
            "MANO_WRIST_RING_16_FROM_VISIBLE_HAND_SURFACE_AND_D455_DEPTH",
        )
        self.assertEqual(packet["orientation_source"], "RING_FILTER_TEST")
        np.testing.assert_allclose(packet["palm_rotation_row_major"], np.eye(3).reshape(-1))
        self.assertAlmostEqual(packet["position_diagnostics"]["depth_m"], 0.9)


if __name__ == "__main__":
    unittest.main()
