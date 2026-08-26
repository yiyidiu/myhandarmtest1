import unittest

from perception_hamer.src.hand_detection_gate import (
    ConsecutiveHandDetectionGate, ContinuousHandPresenceGate, bbox_iou,
)


class HandDetectionGateTest(unittest.TestCase):
    def test_single_false_positive_never_confirms(self):
        gate = ConsecutiveHandDetectionGate(required_frames=3)
        self.assertIsNone(gate.observe({
            "valid": True, "bbox": [10, 10, 100, 100], "is_right": True,
        }))
        self.assertIsNone(gate.observe({"valid": False, "reason": "no_hand"}))
        self.assertEqual(gate.count, 0)

    def test_stable_same_hand_confirms(self):
        gate = ConsecutiveHandDetectionGate(required_frames=3, minimum_iou=0.5)
        samples = [
            {"valid": True, "bbox": [10+i, 10, 100+i, 100], "is_right": False}
            for i in range(3)
        ]
        self.assertIsNone(gate.observe(samples[0]))
        self.assertIsNone(gate.observe(samples[1]))
        confirmed = gate.observe(samples[2])
        self.assertIsNotNone(confirmed)
        self.assertFalse(confirmed["is_right"])

    def test_hand_switch_or_bbox_jump_restarts_count(self):
        gate = ConsecutiveHandDetectionGate(required_frames=2, minimum_iou=0.5)
        gate.observe({"valid": True, "bbox": [0, 0, 20, 20], "is_right": True})
        self.assertIsNone(gate.observe({
            "valid": True, "bbox": [100, 100, 120, 120], "is_right": True,
        }))
        self.assertEqual(gate.count, 1)
        self.assertIsNone(gate.observe({
            "valid": True, "bbox": [100, 100, 120, 120], "is_right": False,
        }))
        self.assertEqual(gate.count, 1)

    def test_bbox_iou(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1.0/3.0)
        self.assertEqual(bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_continuous_presence_hides_on_first_no_hand_result(self):
        gate = ContinuousHandPresenceGate(required_frames=2, timeout_s=0.25)
        hand = {
            "valid": True, "bbox": [10, 10, 100, 100], "is_right": True,
        }
        self.assertFalse(gate.observe(hand, 1.00)["valid"])
        visible = gate.observe(hand, 1.02)
        self.assertTrue(visible["valid"])
        visible_generation = visible["generation"]

        hidden = gate.observe({"valid": False, "reason": "no_hand_detected"}, 1.04)
        self.assertFalse(hidden["valid"])
        self.assertEqual(hidden["reason"], "no_hand_detected")
        self.assertIsNone(hidden["confirmed_detection"])
        self.assertGreater(hidden["generation"], visible_generation)

    def test_live_one_frame_grace_reduces_flicker_but_second_miss_hides(self):
        gate = ContinuousHandPresenceGate(
            required_frames=2,
            timeout_s=0.25,
            negative_grace_frames=1,
            negative_grace_s=0.08,
        )
        hand = {
            "valid": True,
            "bbox": [10, 10, 100, 100],
            "is_right": True,
        }
        gate.observe(hand, 1.00)
        visible = gate.observe(hand, 1.02)
        first_miss = gate.observe(
            {"valid": False, "reason": "no_hand_detected"}, 1.05
        )
        self.assertTrue(first_miss["valid"])
        self.assertIsNotNone(first_miss["confirmed_detection"])
        self.assertEqual(first_miss["consecutive_negative_results"], 1)
        second_miss = gate.observe(
            {"valid": False, "reason": "no_hand_detected"}, 1.08
        )
        self.assertFalse(second_miss["valid"])
        self.assertIsNone(second_miss["confirmed_detection"])
        self.assertGreater(second_miss["generation"], visible["generation"])

    def test_bounded_miss_run_keeps_current_frame_tracking_identity(self):
        gate = ContinuousHandPresenceGate(
            required_frames=2,
            timeout_s=0.50,
            negative_grace_frames=4,
            negative_grace_s=0.20,
        )
        hand = {
            "valid": True,
            "bbox": [10, 10, 100, 100],
            "is_right": True,
        }
        gate.observe(hand, 1.00)
        visible = gate.observe(hand, 1.02)
        visible_generation = visible["generation"]

        first = gate.observe(
            {"valid": False, "reason": "no_hand_detected"}, 1.05
        )
        second = gate.observe(
            {"valid": False, "reason": "no_hand_detected"}, 1.10
        )
        self.assertTrue(first["valid"])
        self.assertTrue(second["valid"])
        self.assertTrue(second["transient_miss"])
        self.assertEqual(second["generation"], visible_generation)
        self.assertIsNotNone(second["confirmed_detection"])

        recovered = gate.observe(hand, 1.12)
        self.assertTrue(recovered["valid"])
        self.assertFalse(recovered["transient_miss"])
        self.assertEqual(recovered["generation"], visible_generation)

    def test_miss_time_bound_fails_closed_even_with_frame_budget_remaining(self):
        gate = ContinuousHandPresenceGate(
            required_frames=2,
            timeout_s=0.50,
            negative_grace_frames=10,
            negative_grace_s=0.10,
        )
        hand = {
            "valid": True,
            "bbox": [10, 10, 100, 100],
            "is_right": False,
        }
        gate.observe(hand, 2.00)
        visible = gate.observe(hand, 2.02)
        gate.observe({"valid": False, "reason": "no_hand_detected"}, 2.04)
        lost = gate.observe(
            {"valid": False, "reason": "no_hand_detected"}, 2.15
        )
        self.assertFalse(lost["valid"])
        self.assertFalse(lost["transient_miss"])
        self.assertGreater(lost["generation"], visible["generation"])
        self.assertIsNone(lost["confirmed_detection"])

    def test_continuous_presence_needs_two_frames_to_reappear(self):
        gate = ContinuousHandPresenceGate(required_frames=2, timeout_s=0.25)
        hand = {
            "valid": True, "bbox": [10, 10, 100, 100], "is_right": True,
        }
        gate.observe(hand, 2.00)
        gate.observe(hand, 2.02)
        gate.observe({"valid": False, "reason": "no_hand_detected"}, 2.04)
        first = gate.observe(hand, 2.06)
        second = gate.observe(hand, 2.08)
        self.assertFalse(first["valid"])
        self.assertEqual(first["reason"], "hand_reconfirming_1/2")
        self.assertTrue(second["valid"])

    def test_continuous_presence_times_out_fail_closed(self):
        gate = ContinuousHandPresenceGate(required_frames=2, timeout_s=0.25)
        hand = {
            "valid": True, "bbox": [10, 10, 100, 100], "is_right": False,
        }
        gate.observe(hand, 3.00)
        gate.observe(hand, 3.02)
        timed_out = gate.snapshot(3.28)
        self.assertFalse(timed_out["valid"])
        self.assertEqual(timed_out["reason"], "hand_detector_timeout")
        self.assertIsNone(timed_out["confirmed_detection"])


if __name__ == "__main__":
    unittest.main()
