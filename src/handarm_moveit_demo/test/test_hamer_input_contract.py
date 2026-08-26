#!/usr/bin/env python3

import unittest

import numpy as np

from handarm_moveit_demo.hamer_input_contract import (
    HamerPacketContract,
    InputWatchdog,
    identity_reference_token,
)


def packet(sequence=1, valid=True):
    value = {
        "schema": "handarm_hamer_pose_v1",
        "session_id": "live-test",
        "sequence": sequence,
        "stamp": 123.5 + sequence,
        "frame_id": "camera_color_optical_frame",
        "valid": bool(valid),
        "invalid_reason": "" if valid else "no_real_hand",
        "hand_identity_present": True,
        "hand_is_right": True,
        "presence_generation": 4,
        "active_hand_generation": 2,
        "control_enabled": True,
        "control_reference_epoch": 3,
        "control_identity_present": True,
        "control_presence_generation": 4,
        "control_active_hand_generation": 2,
        "control_hand_is_right": True,
        "control_reference_token": identity_reference_token(
            "live-test", 3, 4, 2, True
        ),
        "gesture": 0,
        "gesture_confidence": 0.0,
    }
    if valid:
        value.update({
            "wrist_position_m": [0.1, -0.2, 0.7],
            "palm_rotation_row_major": np.eye(3).reshape(-1).tolist(),
            "confidence": [0.9] * 6,
        })
    return value


class HamerPacketContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = HamerPacketContract("camera_color_optical_frame")

    def test_enabled_pose_requires_identity_bound_token(self):
        normalized = self.contract.validate(packet())
        self.assertTrue(normalized["observation_valid"])
        self.assertTrue(normalized["control_enabled"])
        self.assertTrue(normalized["hand_is_right"])
        np.testing.assert_allclose(normalized["position"], [0.1, -0.2, 0.7])

    def test_legacy_epoch_only_token_is_rejected(self):
        value = packet()
        value["control_reference_token"] = "live-test:3"
        with self.assertRaisesRegex(ValueError, "invalid_control_reference_token"):
            self.contract.validate(value)

    def test_observed_identity_must_match_control_identity(self):
        value = packet()
        value["hand_is_right"] = False
        with self.assertRaisesRegex(
            ValueError, "control_identity_does_not_match_observation"
        ):
            self.contract.validate(value)

    def test_invalid_heartbeat_requires_no_pose_geometry(self):
        normalized = self.contract.validate(packet(valid=False))
        self.assertFalse(normalized["observation_valid"])
        self.assertEqual(normalized["invalid_reason"], "no_real_hand")
        np.testing.assert_allclose(normalized["position"], np.zeros(3))
        np.testing.assert_allclose(
            normalized["quaternion"], [0.0, 0.0, 0.0, 1.0]
        )

    def test_rejected_packet_does_not_consume_sequence(self):
        bad = packet(sequence=8)
        bad["control_reference_token"] = "bad"
        with self.assertRaises(ValueError):
            self.contract.validate(bad)
        accepted = self.contract.validate(packet(sequence=8))
        self.assertEqual(accepted["sequence"], 8)

    def test_nonfinite_gesture_confidence_is_rejected(self):
        value = packet()
        value["gesture_confidence"] = float("nan")
        with self.assertRaisesRegex(ValueError, "gesture_confidence must be finite"):
            self.contract.validate(value)

    def test_gesture_must_fit_ros_uint8(self):
        value = packet()
        value["gesture"] = 256
        with self.assertRaisesRegex(ValueError, "gesture must fit uint8"):
            self.contract.validate(value)


class InputWatchdogTest(unittest.TestCase):
    def test_timeout_repeats_until_a_packet_is_accepted(self):
        watchdog = InputWatchdog(0.4, 0.1, start_monotonic=10.0)
        self.assertFalse(watchdog.timeout_due(10.39))
        self.assertTrue(watchdog.timeout_due(10.40))
        self.assertFalse(watchdog.timeout_due(10.45))
        self.assertTrue(watchdog.timeout_due(10.50))
        watchdog.mark_accepted(10.51)
        self.assertFalse(watchdog.timeout_due(10.90))
        self.assertTrue(watchdog.timeout_due(10.91))


if __name__ == "__main__":
    unittest.main()
