import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from perception_hamer.scripts.run_d455_hamer_crop import (
    LatestDisplayOverlay,
    PendingReinitialization,
    _display_worker,
    _hand_preflight_timeout_diagnostics,
    make_overlay,
)
from perception_hamer.src.hand_pose_overlay import RelativeWristPoseDisplay
from perception_hamer.src.teleop_control_gate import TeleopControlGate
from perception_hamer.src.teleoperation_core_mano_renderer import (
    TeleoperationCoreRenderFrame,
)


class LiveHamerPresenceTest(unittest.TestCase):
    def _confirm_with_delayed_pose(self, ready_at, stop_at):
        pose = RelativeWristPoseDisplay()
        gate = TeleopControlGate(retain_reference_until_operator_action=True)
        overlay = LatestDisplayOverlay()
        overlay.update(SimpleNamespace(is_right=True), {}, 10., "", None,
                       None, None, 0)
        packet = SimpleNamespace(
            frame=SimpleNamespace(rgb=np.zeros((480, 640, 3), dtype=np.uint8)),
            roi=self._roi(),
        )

        class Preview:
            count = 0

            def get_after(self, previous, timeout_s):
                self.count += 1
                if self.count == ready_at:
                    pose.update_from_packet({
                        "sequence": self.count, "stamp": 100.0,
                        "wrist_position_m": [0.0, 0.0, 0.6],
                        "palm_rotation_row_major": np.eye(3).ravel().tolist(),
                        "confidence": [0.9] * 6,
                    }, 0)
                return previous + 1, packet

        preview = Preview()

        class Display:
            stop_requested = False
            confirmed = False

            def pop_confirm_request(self):
                if self.confirmed:
                    return False
                self.confirmed = True
                return True

            def show(self, canvas):
                self.stop_requested = preview.count >= stop_at
                return True

        clock = iter(np.arange(0.0, 100.0, 0.02))
        with patch("perception_hamer.scripts.run_d455_hamer_crop.time.monotonic",
                   side_effect=lambda: next(clock)), patch(
                       "perception_hamer.scripts.run_d455_hamer_crop.make_overlay",
                       return_value=np.zeros((480, 1280, 3), dtype=np.uint8)), patch(
                       "builtins.print"):
            _display_worker(Display(), preview, overlay, pose, gate, None,
                            threading.Event(), "teleoperation-core", 1000.)
        return gate

    def test_c_during_invalid_frame_accepts_next_fresh_pose(self):
        self.assertTrue(self._confirm_with_delayed_pose(3, 5).enabled)

    def test_expired_c_does_not_enable_on_a_later_pose(self):
        self.assertFalse(self._confirm_with_delayed_pose(100, 105).enabled)

    def _roi(self):
        return SimpleNamespace(
            bbox=np.array([300.0, 300.0, 400.0, 400.0]),
            source="tracker_roi",
            lost=False,
            age=10,
            confidence=0.9,
        )

    def test_no_hand_hides_crop_and_never_touches_stale_result(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        hidden = make_overlay(
            rgb,
            self._roi(),
            object(),  # would fail if stale HaMeR fields were accessed
            None,
            0.0,
            "no_real_hand:no_hand_detected",
            hand_presence_valid=False,
            teleoperation_core_render=object(),
        )
        self.assertEqual(hidden.shape, (480, 1280, 3))
        np.testing.assert_array_equal(hidden[350, 300], [0, 0, 0])

    def test_present_hand_draws_current_crop(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        shown = make_overlay(
            rgb,
            self._roi(),
            None,
            None,
            0.0,
            "",
            hand_presence_valid=True,
        )
        self.assertEqual(shown.shape, (480, 1280, 3))
        np.testing.assert_array_equal(shown[350, 300], [70, 255, 70])

    def test_real_hand_elsewhere_does_not_validate_drifted_crop(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        hidden = make_overlay(
            rgb,
            self._roi(),
            object(),
            None,
            0.0,
            "roi_detector_mismatch",
            hand_presence_valid=True,
            roi_matches_detected_hand=False,
        )
        np.testing.assert_array_equal(hidden[350, 300], [0, 0, 0])

    def test_default_renderer_uses_exact_inference_pair_not_latest_rgb(self):
        latest_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        reference = TeleoperationCoreRenderFrame(
            source_bgr=np.full((480, 640, 3), 17, dtype=np.uint8),
            overlay_bgr=np.full((480, 640, 3), 23, dtype=np.uint8),
            crop_points_px=np.zeros((778, 2), dtype=np.float32),
            full_points_px=np.zeros((778, 2), dtype=np.float32),
            vertex_depth=np.ones(778, dtype=np.float32),
            timestamp=1.0,
            sequence_bbox_xyxy=np.asarray([300, 300, 400, 400]),
        )
        result = SimpleNamespace(
            inference_time_s=0.02,
            pred_vertices_mano_right_canonical=np.zeros((778, 3)),
        )
        shown = make_overlay(
            latest_rgb,
            self._roi(),
            result,
            None,
            15.0,
            "",
            mano_faces=np.zeros((1538, 3), dtype=np.int64),
            hand_presence_valid=True,
            roi_matches_detected_hand=True,
            teleoperation_core_render=reference,
        )
        np.testing.assert_array_equal(shown[300, 500], [17, 17, 17])
        np.testing.assert_array_equal(shown[300, 640 + 500], [23, 23, 23])

    def test_pending_reset_overrides_stale_reinitialization(self):
        pending = PendingReinitialization()
        pending.set([10, 20, 100, 120], True)
        pending.request_reset()
        self.assertEqual(pending.pop(), ("reset", None, None))
        self.assertIsNone(pending.pop())

    def test_no_hand_timeout_reports_detector_state_without_seed_roi(self):
        diagnostics = _hand_preflight_timeout_diagnostics(
            "no_hand_detected", 17, 4, 2.0
        )
        self.assertEqual(diagnostics, {
            "valid": False,
            "reason": "no_hand_detected",
            "attempts": 17,
            "detection_results": 4,
            "wait_for_hand_s": 2.0,
        })


if __name__ == "__main__":
    unittest.main()
