#!/usr/bin/env python3

import unittest

import numpy as np

from handarm_moveit_demo.finger_retargeting import (
    ThreeFingerRetargeter,
    frozen_finger_hold_target,
)


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
    def test_power_grip_releases_all_contacts_together_from_measured_angles(self):
        controller = retargeter(
            grip_latch_enabled=True,
            one_euro_minimum_cutoff_hz=1000.0,
            maximum_single_frame_innovation=1.0,
            maximum_human_flexion_rate_per_s=100.0,
        )
        controller.open_target[1:] = 0.20
        _, baseline, current = self.calibrate(controller)
        timestamp = 10.15

        def observe(value, measurement=None):
            nonlocal timestamp, current
            timestamp += 0.05
            result = controller.update(
                timestamp, "c:1", value, 0.99,
                current if measurement is None else measurement)
            if result.command_target is not None:
                current = result.command_target.copy()
            return result

        for _ in range(30):
            result = observe(baseline + 0.60)
        self.assertTrue(controller.grip_latched)
        np.testing.assert_allclose(result.command_target, controller.close_target)
        for _ in range(8):
            result = observe(baseline + 0.30)
            np.testing.assert_allclose(result.desired_target, controller.close_target)

        # Unequal contact angles in the 18:25 video must not leave one
        # finger applying preload after the other fingers begin opening.
        contact_angles = np.array([0.18, 0.783, 0.405, 0.664])
        for _ in range(5):
            result = observe(baseline, contact_angles)
            if controller.release_phase is not None:
                break
        self.assertEqual(controller.release_phase, "OPEN_ALL")
        self.assertFalse(controller.grip_latched)
        self.assertTrue(np.all(result.command_target[1:] < contact_angles[1:]))
        progress = ((contact_angles[1:] - result.command_target[1:]) /
                    (contact_angles[1:] - controller.open_target[1:]))
        np.testing.assert_allclose(progress, [progress[0]] * 3)
        np.testing.assert_allclose(result.desired_target[1:], [0.20] * 3)

        for _ in range(25):
            result = observe(baseline)
        self.assertIsNone(controller.release_phase)
        np.testing.assert_allclose(result.command_target, controller.open_target)

    def test_unused_bent_thumb_does_not_block_recorded_open_hand(self):
        controller = retargeter()
        controller.mixing[1] = [0.0, 0.25, 0.25, 0.25, 0.25]
        opened = [0.44694, 0.24340, 0.14263, 0.18371, 0.23119]
        for i in range(4):
            result = controller.update(10 + .05*i, "c:recorded", opened, .97, [.2]*4)
        self.assertTrue(result.calibrated)
        self.assertEqual(result.status, "OPEN_BASELINE")
        # A fist still cannot serve as the open reference.
        fist = [0.47076, 0.81420, 0.83584, 0.82917, 0.75894]
        result = controller.update(11., "c:new", fist, .97, [.2]*4)
        self.assertFalse(result.calibrated)
        self.assertEqual(result.status, "C_REFERENCE_HAND_NOT_OPEN")

    def test_missing_vision_preserves_recorded_contact_preload(self):
        # 2026-09-07 recording, 67.54 s: an invalid image used to replace
        # f2's 0.85 rad command by its blocked 0.40 rad measured position.
        command = np.asarray([0.18, 0.85, 0.85, 0.90])
        measured = np.asarray([0.177, 0.838, 0.400, 0.664])
        hold = frozen_finger_hold_target(None, measured, command)
        np.testing.assert_array_equal(hold, command)
        for drift in (0.0, -0.02, -0.05):
            hold = frozen_finger_hold_target(hold, measured + drift, command)
            np.testing.assert_array_equal(hold, command)

    def test_invalid_heartbeat_hold_target_does_not_chase_joint_drift(self):
        first_measurement = np.asarray([0.05, 0.04, 0.03, 0.04])
        hold = frozen_finger_hold_target(None, first_measurement)
        for drifted_measurement in (
            [0.10, 0.08, 0.06, 0.07],
            [0.30, 0.20, 0.18, 0.19],
            [1.10, 0.50, 0.45, 0.75],
        ):
            hold = frozen_finger_hold_target(hold, drifted_measurement)
            np.testing.assert_array_equal(hold, first_measurement)

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
