#!/usr/bin/env python3

from pathlib import Path
import math
import sys
import threading
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_hamer.src.hand_pose_overlay import (  # noqa: E402
    RelativeWristPoseDisplay,
    draw_hand_pose_panel,
    matrix_to_euler_zyx_deg,
)


def _rotation(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, math.cos(roll), -math.sin(roll)],
         [0.0, math.sin(roll), math.cos(roll)]]
    )
    ry = np.array(
        [[math.cos(pitch), 0.0, math.sin(pitch)],
         [0.0, 1.0, 0.0],
         [-math.sin(pitch), 0.0, math.cos(pitch)]]
    )
    rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0],
         [math.sin(yaw), math.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    )
    return rz @ ry @ rx


def _packet(sequence, position, rotation, confidence=None):
    return {
        "sequence": int(sequence),
        "stamp": 100.0 + sequence / 30.0,
        "wrist_position_m": list(position),
        "palm_rotation_row_major": rotation.reshape(-1).tolist(),
        "confidence": [0.8] * 6 if confidence is None else list(confidence),
    }


class RelativeWristPoseDisplayTest(unittest.TestCase):
    def test_c_zero_then_all_six_components_are_reported_together(self):
        display = RelativeWristPoseDisplay()
        reference = _rotation(7.0, -4.0, 12.0)
        before_zero = display.update_from_packet(
            _packet(1, [0.10, 0.20, 0.50], reference), 4
        )
        self.assertTrue(before_zero.valid)
        self.assertFalse(before_zero.calibrated)
        self.assertTrue(display.calibrate_from_latest(4))

        relative = _rotation(13.0, -8.0, 21.0)
        current = display.update_from_packet(
            _packet(
                2,
                [0.135, 0.181, 0.542],
                relative @ reference,
                [0.9, 0.8, 0.7, 0.95, 0.85, 0.75],
            ),
            4,
        )
        self.assertTrue(current.valid)
        self.assertTrue(current.calibrated)
        np.testing.assert_allclose(current.delta_m, [0.035, -0.019, 0.042])
        self.assertAlmostEqual(current.roll_deg, 13.0, places=6)
        self.assertAlmostEqual(current.pitch_deg, -8.0, places=6)
        self.assertAlmostEqual(current.yaw_deg, 21.0, places=6)
        np.testing.assert_allclose(
            current.confidence, [0.9, 0.8, 0.7, 0.95, 0.85, 0.75]
        )

    def test_relative_rotation_is_so3_not_euler_subtraction(self):
        zero = _rotation(35.0, 70.0, -50.0)
        increment = _rotation(-11.0, 9.0, 17.0)
        yaw, pitch, roll = matrix_to_euler_zyx_deg(
            (increment @ zero) @ zero.T
        )
        np.testing.assert_allclose([roll, pitch, yaw], [-11.0, 9.0, 17.0])

    def test_invalid_measurement_never_reuses_old_pose(self):
        display = RelativeWristPoseDisplay()
        display.update_from_packet(_packet(1, [0.1, 0.2, 0.5], np.eye(3)), 2)
        self.assertTrue(display.calibrate_from_latest(2))
        display.update_from_packet(_packet(2, [0.2, 0.3, 0.6], np.eye(3)), 2)
        invalid = display.invalidate("no_real_hand", 2)
        self.assertFalse(invalid.valid)
        self.assertIsNone(invalid.center_m)
        self.assertIsNone(invalid.delta_m)
        self.assertIsNone(invalid.confidence)

    def test_hand_loss_and_new_presence_generation_preserve_c_zero(self):
        display = RelativeWristPoseDisplay()
        display.update_from_packet(_packet(1, [0.1, 0.2, 0.5], np.eye(3)), 8)
        self.assertTrue(display.calibrate_from_latest(8))
        lost = display.invalidate("no_real_hand", 9)
        self.assertFalse(lost.valid)
        self.assertTrue(lost.calibrated)
        reacquired = display.update_from_packet(
            _packet(2, [0.4, 0.3, 0.8], np.eye(3)), 9
        )
        self.assertTrue(reacquired.valid)
        self.assertTrue(reacquired.calibrated)
        np.testing.assert_allclose(reacquired.delta_m, [0.3, 0.1, 0.3])
        # C still checks that the visible pose belongs to the current interval.
        self.assertFalse(display.calibrate_from_latest(8))

    def test_calibration_and_update_are_thread_safe(self):
        display = RelativeWristPoseDisplay()
        errors = []

        def update_many():
            try:
                for index in range(100):
                    display.update_from_packet(
                        _packet(index, [index * 0.001, 0.2, 0.5], np.eye(3)),
                        1,
                    )
            except Exception as exc:  # pragma: no cover - regression guard
                errors.append(exc)

        worker = threading.Thread(target=update_many)
        worker.start()
        for _index in range(100):
            display.calibrate_from_latest(1)
            display.snapshot()
        worker.join()
        self.assertEqual(errors, [])
        self.assertTrue(display.snapshot().valid)


class HandPosePanelTest(unittest.TestCase):
    def test_panel_is_appended_without_covering_existing_mano_pixels(self):
        source = np.full((480, 1280, 3), 77, dtype=np.uint8)
        display = RelativeWristPoseDisplay()
        delta = display.update_from_packet(
            _packet(1, [0.1, 0.2, 0.5], np.eye(3)), 1
        )
        output = draw_hand_pose_panel(source, delta)
        self.assertEqual(output.shape, (480, 1690, 3))
        np.testing.assert_array_equal(output[:, :1280], source)
        self.assertGreater(np.count_nonzero(output[:, 1280:] != 18), 0)

    def test_invalid_panel_does_not_draw_fabricated_pose_values(self):
        source = np.zeros((480, 640, 3), dtype=np.uint8)
        display = RelativeWristPoseDisplay()
        output = draw_hand_pose_panel(source, display.snapshot())
        self.assertEqual(output.shape, (480, 1050, 3))
        np.testing.assert_array_equal(output[:, :640], source)


if __name__ == "__main__":
    unittest.main()
