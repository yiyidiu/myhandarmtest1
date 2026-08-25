#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import numpy as np

from perception_hamer.src.teleoperation_core_mano_renderer import (
    LIGHT_BLUE,
    crop_points_to_original,
    draw_mesh,
    project_vertices_to_crop,
    render_inference_frame,
)


class TeleoperationCoreManoRendererTest(unittest.TestCase):
    def test_archive_crop_camera_projection_uses_all_xyz_components(self):
        vertices = np.asarray([[0.0, 0.0, 0.0], [0.1, -0.2, 0.5]])
        points, depth = project_vertices_to_crop(
            vertices, [0.0, 0.0, 2.0], [100.0, 200.0], 256
        )
        np.testing.assert_allclose(points[0], [128.0, 128.0])
        np.testing.assert_allclose(points[1], [132.0, 112.0])
        np.testing.assert_allclose(depth, [2.0, 2.5])

    def test_combined_left_hand_affine_is_inverted_without_second_flip(self):
        # x_crop = -x_original + 639 models the reflection already included
        # by prepare_hamer_crop for a left hand.
        affine = np.asarray([[-1.0, 0.0, 639.0], [0.0, 1.0, 0.0]])
        original = crop_points_to_original([[539.0, 120.0]], affine)
        np.testing.assert_allclose(original, [[100.0, 120.0]])

    def test_complete_mesh_draws_every_valid_face_without_touching_background(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        points = np.asarray([[20, 20], [70, 20], [70, 60], [20, 60]])
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        rendered = draw_mesh(image, points, faces, alpha=1.0)
        np.testing.assert_array_equal(rendered[30, 45], LIGHT_BLUE)
        np.testing.assert_array_equal(rendered[5, 5], [0, 0, 0])

    def test_full_render_returns_exact_source_and_overlay_pair(self):
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        vertices = np.asarray([
            [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.0, 0.2, 0.0]
        ])
        result = SimpleNamespace(
            pred_vertices_mano_right_canonical=vertices,
            hamer_crop_projection_translation=np.asarray([0.0, 0.0, 2.0]),
            hamer_nominal_crop_focal_length=np.asarray([100.0, 100.0]),
            quality={"affine_original_to_crop": [[1, 0, 48], [0, 1, 68]]},
            requested_bbox_xyxy=np.asarray([50, 70, 110, 110]),
            is_right=True,
            timestamp=12.5,
        )
        rendered = render_inference_frame(
            rgb, result, np.asarray([[0, 1, 2]], dtype=np.int64), image_size=256
        )
        self.assertEqual(rendered.source_bgr.shape, (120, 160, 3))
        self.assertEqual(rendered.overlay_bgr.shape, (120, 160, 3))
        self.assertEqual(rendered.side_by_side_bgr().shape, (120, 320, 3))
        self.assertEqual(rendered.timestamp, 12.5)
        self.assertGreater(np.count_nonzero(rendered.overlay_bgr), 0)


if __name__ == "__main__":
    unittest.main()
