"""Keep observed loss/stall states visible without changing robot commands."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_teleop_terminal_monitor import LiveTeleopTerminalMonitor
from handarm_moveit_demo.msg import HamerHandPose


class OperatorFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.monitor = LiveTeleopTerminalMonitor.__new__(LiveTeleopTerminalMonitor)
        m = self.monitor
        m.feedback_file = str(Path(self.folder.name) / "feedback.json")
        m.last_feedback_write = 0.0
        m.last_diagnostic_at = 100.0
        m.latest_diagnostic = dict(valid=True, reference_ready=True,
            current_position=[.3, 0, .4], target_position=[.3, 0, .4],
            rotation_error_rad=[0, 0, 0])
        m.geometry = dict(input_valid=True, approved_velocity=[0]*6)
        m.direction_blocked_since = None
        m.hand_input_valid = True
        m.hand_orientation_valid = True
        m.hand_input_reason = ""
        m.robot_safety_status = "SAFE"
        m.hand_safety_status = "HAND_PATH_ALLOWED"
        m.finger_closure = None
        m.grip_latched = False
        m.release_phase = None
        m.trace = lambda *args: None

    def feedback(self, now=100.0):
        with patch("live_teleop_terminal_monitor.time.monotonic", return_value=now):
            self.monitor.write_feedback()
        return json.loads(Path(self.monitor.feedback_file).read_text())

    def test_pure_rotation_stall_does_not_need_a_position_error(self):
        self.monitor.latest_diagnostic["rotation_error_rad"] = [math.radians(19), 0, 0]
        self.feedback()
        state = self.feedback(100.5)
        self.assertEqual(state["target_gap_cm"], 0.0)
        self.assertAlmostEqual(state["rotation_gap_deg"], 19.0)
        self.assertTrue(state["motion_direction_blocked"])
        self.assertIn("旋转方向受限", state["status_text"])

    def test_translation_motion_does_not_hide_a_rotation_stall(self):
        self.monitor.latest_diagnostic["rotation_error_rad"] = [0, .5, 0]
        self.monitor.geometry["approved_velocity"] = [.02, 0, 0, 0, 0, 0]
        self.feedback()
        self.assertIn("旋转方向受限", self.feedback(100.5)["status_text"])

    def test_held_orientation_is_not_reported_as_following(self):
        message = HamerHandPose()
        message.valid = message.control_enabled = True
        message.confidence = [.95]*3 + [0.0]*3
        message.invalid_reason = "causal_so3_filter:orientation_innovation_exceeds_hard_limit"
        self.monitor.hand_trace_callback(message)
        state = self.feedback()
        self.assertIn("朝向未更新", state["status_text"])
        self.assertFalse(state["orientation_input_valid"])
        self.assertEqual(state["input_reason"], message.invalid_reason)

    def test_hand_loss_has_priority_over_orientation_hold(self):
        self.monitor.hand_input_valid = False
        self.monitor.hand_orientation_valid = False
        self.assertIn("手部输入暂不可用", self.feedback()["status_text"])

    def test_reached_stationary_target_is_not_a_stall(self):
        self.feedback()
        state = self.feedback(100.5)
        self.assertFalse(state["motion_direction_blocked"])
        self.assertEqual(state["level"], "normal")

    def test_moving_toward_rotation_target_is_not_a_stall(self):
        self.monitor.latest_diagnostic["rotation_error_rad"] = [0, .5, 0]
        self.monitor.geometry["approved_velocity"] = [0, 0, 0, 0, .3, 0]
        self.feedback()
        state = self.feedback(100.5)
        self.assertFalse(state["motion_direction_blocked"])
        self.assertIn("旋转目标尚未到达", state["status_text"])


if __name__ == "__main__":
    unittest.main()
