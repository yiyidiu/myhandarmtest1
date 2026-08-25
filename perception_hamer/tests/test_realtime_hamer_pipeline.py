#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from perception_hamer.scripts.evaluate_hamer_palm_stability import (
    _latest_session,
    chordal_so3_reference,
    evaluate_root,
    percentile_summary,
)
from perception_hamer.src.realtime_hamer_pipeline import (
    LatestFrameSlot,
    LiveFramePacket,
    draw_mano_mesh_overlay,
    hand_bbox_alignment,
    normalized_crop_points_to_original,
    project_hamer_vertices_to_original,
    remap_points_between_bboxes,
    so3_geodesic_degrees,
)


class LatestFrameSlotTest(unittest.TestCase):
    def test_old_frames_are_overwritten_not_queued(self):
        slot = LatestFrameSlot()
        for index in range(3):
            slot.publish(LiveFramePacket(index, None, index))
        version, packet = slot.get_after(0)
        self.assertEqual(version, 3)
        self.assertEqual(packet.capture_sequence, 2)
        self.assertEqual(slot.statistics["capacity"], 1)
        self.assertEqual(slot.statistics["overwritten_before_inference"], 2)

    def test_capture_error_is_propagated(self):
        slot = LatestFrameSlot()
        slot.close(RuntimeError("camera disconnected"))
        with self.assertRaisesRegex(RuntimeError, "capture worker failed"):
            slot.get_after(0)


class ProjectionAndSO3Test(unittest.TestCase):
    def test_tracked_crop_must_still_cover_detected_hand(self):
        aligned = hand_bbox_alignment(
            [100, 100, 220, 260], [115, 105, 225, 265]
        )
        drifted = hand_bbox_alignment(
            [300, 80, 380, 220], [40, 220, 180, 390]
        )
        self.assertTrue(aligned["valid"])
        self.assertFalse(drifted["valid"])
        self.assertEqual(
            drifted["reason"], "tracked_bbox_not_on_detected_hand"
        )

    def test_mesh_projection_follows_latest_roi_translation_and_scale(self):
        points = np.array([[10.0, 20.0], [20.0, 30.0]])
        translated = remap_points_between_bboxes(
            points, [0.0, 0.0, 40.0, 40.0], [5.0, -3.0, 45.0, 37.0]
        )
        np.testing.assert_allclose(translated, points+[5.0, -3.0])
        scaled = remap_points_between_bboxes(
            points, [0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 80.0, 80.0]
        )
        np.testing.assert_allclose(scaled, [[20.0, 40.0], [40.0, 60.0]])

    def test_crop_normalized_points_round_trip(self):
        affine = np.array([[2.0, 0.0, -10.0], [0.0, 2.0, -20.0]])
        points = np.array([[0.0, 0.0], [-0.5, -0.5]], dtype=np.float32)
        original = normalized_crop_points_to_original(points, affine, 64)
        np.testing.assert_allclose(original[0], [21.0, 26.0])
        np.testing.assert_allclose(original[1], [5.0, 10.0])

    def test_so3_geodesic_is_not_euler_difference(self):
        theta = np.deg2rad(30.0)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                             [np.sin(theta), np.cos(theta), 0.0],
                             [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(so3_geodesic_degrees(np.eye(3), rotation), 30.0)
        reference = chordal_so3_reference([rotation, rotation])
        np.testing.assert_allclose(reference, rotation, atol=1e-8)

    def test_mano_mesh_projection_uses_crop_camera_only_for_display(self):
        vertices = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                             [0.0, 0.5, 0.0]])
        pixels, depth = project_hamer_vertices_to_original(
            vertices, [0.0, 0.0, 2.0], [256.0, 256.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 256)
        np.testing.assert_allclose(pixels, [[128, 128], [192, 128], [128, 192]])
        np.testing.assert_allclose(depth, [2.0, 2.0, 2.0])

    def test_mano_mesh_overlay_changes_only_projected_triangle(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        rendered = draw_mano_mesh_overlay(
            image, [[10, 10], [50, 10], [10, 50]], [1.0, 1.0, 1.0], [[0, 1, 2]])
        self.assertTrue(np.any(rendered[20, 20] > 0))
        np.testing.assert_array_equal(rendered[60, 60], image[60, 60])


class EvaluationTest(unittest.TestCase):
    def _record(self, index, rotation):
        estimate = {
            "valid": True,
            "rotation": rotation.tolist(),
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
        }
        return {
            "index": index,
            "timestamp": index / 20.0,
            "valid": True,
            "inference_ms": 40.0 + index,
            "betas_user": [0.0] * 10,
            "betas_calibration": {"frozen": True},
            "roi": {
                "bbox": [10 + index, 20, 110 + index, 120],
                "center_jump": 1.0,
                "scale_change": 1.0,
            },
            "palm_frames": {method: estimate for method in (
                "raw_global_orient", "mano_joint_palm_frame",
                "mano_rigid_vertex_palm_frame")},
        }

    def test_three_session_report_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rotation = np.eye(3)
            for experiment in (
                "DEV_HAMER_STATIC", "DEV_HAMER_TRANSLATION", "DEV_HAMER_OPEN_CLOSE"
            ):
                session = root / (experiment + "_20260813T000000")
                session.mkdir()
                with (session / "frames.jsonl").open("w") as handle:
                    for index in range(4):
                        handle.write(json.dumps(self._record(index, rotation)) + "\n")
                (session / "summary.json").write_text(json.dumps({
                    "gpu_system_peak_used_mib": 4800,
                    "roi_seed_hand_presence_validated": True,
                    "experiment_usable": True,
                }))
                (session / "axes_overlay.mp4").write_bytes(b"test")
            report = evaluate_root(root)
            self.assertEqual(len(report["sessions"]), 3)
            self.assertTrue((root / "hamer_palm_stability_metrics.csv").is_file())
            self.assertTrue((root / "development_dataset_index.json").is_file())
            metric = report["sessions"]["DEV_HAMER_STATIC"]["methods"][
                "mano_joint_palm_frame"
            ]["orientation_change_from_chordal_reference_deg"]
            self.assertAlmostEqual(metric["p95"], 0.0)

    def test_background_session_without_presence_preflight_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "DEV_HAMER_STATIC_20260813T000000"
            session.mkdir()
            (session / "summary.json").write_text(json.dumps({
                "roi_seed_hand_presence_validated": False,
                "experiment_usable": False,
            }))
            with self.assertRaisesRegex(FileNotFoundError, "hand-presence-validated"):
                _latest_session(root, "DEV_HAMER_STATIC")


if __name__ == "__main__":
    unittest.main()
