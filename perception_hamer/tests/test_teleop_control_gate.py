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
            {"session_id": "live-a", "sequence": 1}, "live-a")
        self.assertFalse(packet["control_enabled"])
        self.assertEqual(packet["control_reference_epoch"], 0)
        self.assertEqual(packet["control_reference_token"], "")

    def test_each_c_confirmation_creates_a_new_reference_token(self):
        gate = TeleopControlGate()
        self.assertEqual(gate.confirm(), 1)
        first = gate.decorate({"session_id": "live-a"})
        self.assertTrue(first["control_enabled"])
        self.assertEqual(first["control_reference_token"], "live-a:1")

        gate.disable()
        locked = gate.decorate({"session_id": "live-a"})
        self.assertFalse(locked["control_enabled"])
        self.assertEqual(locked["control_reference_token"], "")

        self.assertEqual(gate.confirm(), 2)
        second = gate.decorate({"session_id": "live-a"})
        self.assertEqual(second["control_reference_token"], "live-a:2")


if __name__ == "__main__":
    unittest.main()
