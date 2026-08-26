#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_hamer.src.teleop_control_gate import TeleopControlGate  # noqa: E402


class TeleopControlGateTest(unittest.TestCase):
    def test_camera_start_is_locked_until_explicit_confirmation(self):
        gate = TeleopControlGate()
        packet = gate.decorate(
            {
                "session_id": "live-a",
                "sequence": 1,
                "valid": True,
                "hand_identity_present": True,
                "presence_generation": 3,
                "active_hand_generation": 1,
                "hand_is_right": True,
                "finger_observation": {
                    "valid": True,
                    "flexion": [0.1] * 5,
                    "confidence": 0.9,
                    "invalid_reason": "",
                },
            },
            "live-a",
        )
        self.assertFalse(packet["control_enabled"])
        self.assertFalse(packet.get("valid", False))
        self.assertEqual(
            packet["invalid_reason"], "WAITING_FOR_OPERATOR_C_REFERENCE"
        )
        self.assertEqual(packet["control_reference_epoch"], 0)
        self.assertEqual(packet["control_reference_token"], "")
        self.assertFalse(packet["finger_observation"]["valid"])
        self.assertEqual(packet["finger_observation"]["flexion"], [0.0] * 5)
        self.assertEqual(packet["finger_observation"]["confidence"], 0.0)
        self.assertEqual(
            packet["finger_observation"]["invalid_reason"],
            "WAITING_FOR_OPERATOR_C_REFERENCE",
        )

    def test_each_c_confirmation_creates_a_new_reference_token(self):
        gate = TeleopControlGate()
        self.assertEqual(gate.confirm(3, 1, True), 1)
        identity = {
            "session_id": "live-a",
            "hand_identity_present": True,
            "presence_generation": 3,
            "active_hand_generation": 1,
            "hand_is_right": True,
        }
        first = gate.decorate(identity)
        self.assertTrue(first["control_enabled"])
        self.assertEqual(
            first["control_reference_token"], "live-a:1:p3:h1:R"
        )

        gate.disable()
        locked = gate.decorate(identity)
        self.assertFalse(locked["control_enabled"])
        self.assertEqual(locked["control_reference_token"], "")

        self.assertEqual(gate.confirm(3, 1, True), 2)
        second = gate.decorate(identity)
        self.assertEqual(
            second["control_reference_token"], "live-a:2:p3:h1:R"
        )

    def test_presence_generation_change_disarms_until_a_new_c(self):
        gate = TeleopControlGate()
        gate.confirm(3, 1, True)
        changed = gate.decorate({
            "session_id": "live-a",
            "valid": True,
            "hand_identity_present": True,
            "presence_generation": 4,
            "active_hand_generation": 1,
            "hand_is_right": True,
        })
        self.assertFalse(changed["control_enabled"])
        self.assertFalse(changed["valid"])
        self.assertEqual(
            changed["invalid_reason"],
            "HAND_IDENTITY_CHANGED_REQUIRES_NEW_C",
        )
        self.assertEqual(changed["control_reference_token"], "")

    def test_handedness_change_disarms_even_without_presence_change(self):
        gate = TeleopControlGate()
        gate.confirm(7, 2, True)
        self.assertFalse(gate.observe_identity(7, 2, False))
        self.assertFalse(gate.enabled)
        self.assertEqual(
            gate.lock_reason, "HAND_IDENTITY_CHANGED_REQUIRES_NEW_C"
        )

    def test_invalid_measurement_can_report_same_bound_identity(self):
        gate = TeleopControlGate()
        gate.confirm(5, 9, False)
        packet = gate.decorate({
            "session_id": "live-a",
            "valid": False,
            "invalid_reason": "metric_depth_unavailable",
            "hand_identity_present": True,
            "presence_generation": 5,
            "active_hand_generation": 9,
            "hand_is_right": False,
        })
        self.assertTrue(packet["control_enabled"])
        self.assertFalse(packet["valid"])
        self.assertEqual(
            packet["control_reference_token"], "live-a:1:p5:h9:L"
        )

    def test_channel_local_orientation_hold_keeps_c_reference_enabled(self):
        gate = TeleopControlGate()
        gate.confirm(5, 9, False)
        packet = gate.decorate({
            "session_id": "live-a",
            "valid": True,
            "invalid_reason": "causal_so3_filter:orientation_jump",
            "confidence": [0.8, 0.8, 0.8, 0.0, 0.0, 0.0],
            "orientation_channel_valid": False,
            "orientation_held": True,
            "hand_identity_present": True,
            "presence_generation": 5,
            "active_hand_generation": 9,
            "hand_is_right": False,
        })
        self.assertTrue(packet["valid"])
        self.assertTrue(packet["control_enabled"])
        self.assertEqual(
            packet["control_reference_token"], "live-a:1:p5:h9:L"
        )

    def test_operator_latched_c_survives_missing_and_reacquired_detection(self):
        gate = TeleopControlGate(
            retain_reference_until_operator_action=True
        )
        gate.confirm(5, 9, False)
        missing = gate.decorate({
            "session_id": "live-a",
            "valid": False,
            "invalid_reason": "no_real_hand",
            "hand_identity_present": False,
            "presence_generation": 31,
            "active_hand_generation": 4,
        })
        self.assertFalse(missing["valid"])
        self.assertTrue(missing["control_enabled"])
        self.assertTrue(missing["hand_identity_present"])
        self.assertEqual(missing["presence_generation"], 5)
        self.assertEqual(missing["active_hand_generation"], 9)
        self.assertFalse(missing["hand_is_right"])
        self.assertEqual(
            missing["control_reference_token"], "live-a:1:p5:h9:L"
        )

        self.assertTrue(gate.observe_identity(32, 5, True))
        reacquired = gate.decorate({
            "session_id": "live-a",
            "valid": True,
            "hand_identity_present": True,
            "presence_generation": 32,
            "active_hand_generation": 5,
            "hand_is_right": True,
        })
        self.assertTrue(reacquired["valid"])
        self.assertTrue(reacquired["control_enabled"])
        self.assertEqual(reacquired["presence_generation"], 5)
        self.assertEqual(reacquired["active_hand_generation"], 9)
        self.assertFalse(reacquired["hand_is_right"])
        self.assertEqual(
            reacquired["control_reference_token"], "live-a:1:p5:h9:L"
        )

        gate.disable()
        disabled = gate.decorate(reacquired)
        self.assertFalse(disabled["control_enabled"])


if __name__ == "__main__":
    unittest.main()
