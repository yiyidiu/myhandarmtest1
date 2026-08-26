#!/usr/bin/env python3

import unittest

import numpy as np

from handarm_moveit_demo.finger_retargeting import ThreeFingerRetargeter


def retargeter(**overrides):
    config = {
        "minimum_confidence": 0.55,
        "calibration_samples": 4,
        "reference_max_flexion": [0.40] * 5,
        "calibration_max_range": 0.08,
        "human_flexion_deadband": [0.03] * 3,
        "human_close_excursion": [0.60] * 3,
        "maximum_velocity_rad_s": [0.40, 0.75, 0.75, 0.75],
        "command_duration_s": 0.06,
        "maximum_human_flexion_rate_per_s": 6.0,
        "innovation_slack": 0.05,
        "maximum_single_frame_innovation": 0.25,
        "innovation_confirmation_samples": 3,
        "innovation_consistency_tolerance": 0.08,
        "one_euro_minimum_cutoff_hz": 2.0,
        "one_euro_beta": 0.15,
        "one_euro_derivative_cutoff_hz": 1.0,
    }
    config.update(overrides)
    return ThreeFingerRetargeter(
        [0.18, 0.20, 0.20, 0.20],
        [0.18, 0.85, 0.85, 0.90],
        [0.0, 0.0, 0.0, 0.0],
        [3.14, 1.3963, 1.3963, 1.3963],
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],  # index -> f1
            [1.0, 0.0, 0.0, 0.0, 0.0],  # thumb -> opposed f2
            [0.0, 0.0, 0.5, 0.3, 0.2],  # remaining digits -> f3
        ],
        config,
    )


class ThreeFingerRetargeterTest(unittest.TestCase):
    def calibrate(self, controller, token="c:1", start=10.0):
        current = [0.05, 0.04, 0.03, 0.04]
        baseline = [0.12, 0.08, 0.06, 0.10, 0.09]
        results = [
            controller.update(start + index * 0.05, token, baseline, 0.9, current)
            for index in range(4)
        ]
        return results[-1], np.asarray(baseline), np.asarray(current)

    def test_requires_stable_open_c_reference_before_commanding(self):
        controller = retargeter()
        result, _baseline, current = self.calibrate(controller)
        self.assertTrue(result.calibrated)
        self.assertIsNotNone(result.command_target)
        self.assertEqual(result.status, "OPEN_BASELINE")
        # The first command is slew-limited from measured state toward OPEN.
        np.testing.assert_array_less(
            np.abs(result.command_target - current),
            controller.maximum_velocity * controller.command_duration_s + 1.0e-12,
        )

    def test_non_open_or_moving_reference_never_calibrates(self):
        controller = retargeter()
        result = controller.update(
            1.0, "c:1", [0.1, 0.7, 0.1, 0.1, 0.1], 0.9, [0.1] * 4
        )
        self.assertEqual(result.status, "C_REFERENCE_HAND_NOT_OPEN")
        self.assertFalse(result.calibrated)
        self.assertIsNone(result.command_target)

        controller = retargeter(calibration_max_range=0.01)
        for index, value in enumerate([0.10, 0.13, 0.10, 0.13]):
            result = controller.update(
                2.0 + 0.05 * index,
                "c:2",
                [value] * 5,
                0.9,
                [0.1] * 4,
            )
        self.assertEqual(result.status, "C_REFERENCE_HAND_MOVING")
        self.assertFalse(result.calibrated)

    def test_explicit_five_to_three_synergy_mapping(self):
        controller = retargeter(
            maximum_velocity_rad_s=[20.0] * 4,
            one_euro_minimum_cutoff_hz=1000.0,
        )
        _result, baseline, current = self.calibrate(controller)

        for step in range(1, 4):
            index = baseline.copy()
            index[1] += 0.20 * step
            result = controller.update(
                10.15 + 0.05 * step, "c:1", index, 0.9, current
            )
        self.assertGreater(result.desired_target[1], 0.84)
        self.assertAlmostEqual(result.desired_target[2], 0.20, places=5)
        self.assertAlmostEqual(result.desired_target[3], 0.20, places=5)

        # Recalibrate for independent thumb and remaining-finger checks.
        controller = retargeter(
            maximum_velocity_rad_s=[20.0] * 4,
            one_euro_minimum_cutoff_hz=1000.0,
        )
        _result, baseline, current = self.calibrate(controller)
        for step in range(1, 4):
            thumb = baseline.copy()
            thumb[0] += 0.20 * step
            result = controller.update(
                10.15 + 0.05 * step, "c:1", thumb, 0.9, current
            )
        self.assertGreater(result.desired_target[2], 0.84)
        self.assertAlmostEqual(result.desired_target[1], 0.20, places=5)

        controller = retargeter(
            maximum_velocity_rad_s=[20.0] * 4,
            one_euro_minimum_cutoff_hz=1000.0,
        )
        _result, baseline, current = self.calibrate(controller)
        for step in range(1, 4):
            remaining = baseline.copy()
            remaining[2:] += 0.20 * step
            result = controller.update(
                10.15 + 0.05 * step, "c:1", remaining, 0.9, current
            )
        self.assertGreater(result.desired_target[3], 0.89)

    def test_targets_are_bounded_and_each_command_is_slew_limited(self):
        controller = retargeter()
        previous, baseline, current = self.calibrate(controller)
        previous_target = previous.command_target
        closed = np.clip(baseline + 0.20, 0.0, 1.0)
        result = controller.update(10.20, "c:1", closed, 0.9, current)
        maximum_step = controller.maximum_velocity * controller.command_duration_s
        self.assertTrue(np.all(np.abs(result.command_target - previous_target) <= maximum_step + 1e-12))
        self.assertTrue(np.all(result.command_target >= controller.lower_limits))
        self.assertTrue(np.all(result.command_target <= controller.upper_limits))
        self.assertAlmostEqual(result.command_target[0], previous_target[0] + maximum_step[0])

    def test_low_confidence_innovation_and_old_token_hold_fail_closed(self):
        controller = retargeter()
        _result, baseline, current = self.calibrate(controller)
        low = controller.update(10.20, "c:1", baseline, 0.2, current)
        self.assertEqual(low.status, "LOW_FINGER_CONFIDENCE")
        self.assertTrue(low.hold_required)
        self.assertIsNone(low.command_target)

        spike = baseline.copy()
        spike[1] += 0.50
        rejected = controller.update(10.20, "c:1", spike, 0.9, current)
        self.assertEqual(
            rejected.status,
            "FINGER_INNOVATION_REJECTED_PENDING_CONFIRMATION",
        )
        self.assertIsNone(rejected.command_target)

        # A real fast posture change persists.  Confirm it causally instead
        # of rejecting the operator forever because the comparison anchor did
        # not advance after the first sample.
        pending = controller.update(10.25, "c:1", spike, 0.9, current)
        accepted = controller.update(10.30, "c:1", spike, 0.9, current)
        self.assertIsNone(pending.command_target)
        self.assertIsNotNone(accepted.command_target)

        controller.block_active_reference()
        blocked = controller.update(10.25, "c:1", baseline, 0.9, current)
        self.assertEqual(blocked.status, "BLOCKED_REFERENCE_REQUIRES_NEW_C")
        fresh = controller.update(10.30, "c:2", baseline, 0.9, current)
        self.assertEqual(fresh.status, "CALIBRATING_OPEN_HAND")


if __name__ == "__main__":
    unittest.main()
