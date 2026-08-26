import unittest

from perception_hamer.src.active_hand_selector import AutomaticActiveHandSelector


def detection(is_right, bbox, confidence=0.9):
    return {
        "valid": True,
        "is_right": bool(is_right),
        "bbox": list(bbox),
        "confidence": float(confidence),
        "bbox_area_fraction": 0.1,
    }


def payload(*detections):
    return {
        "valid": bool(detections),
        "reason": "no_hand_detected" if not detections else "",
        "detections": list(detections),
    }


class AutomaticActiveHandSelectorTest(unittest.TestCase):
    def test_other_visible_hand_cannot_steal_active_crop(self):
        selector = AutomaticActiveHandSelector(switch_frames=3)
        right = detection(True, [100, 100, 220, 260], 0.8)
        left = detection(False, [350, 100, 470, 260], 0.99)
        first = selector.select(payload(right))
        both = selector.select(payload(right, left))
        self.assertTrue(first["active_hand_is_right"])
        self.assertEqual(first["active_hand_generation"], 1)
        self.assertTrue(both["is_right"])
        self.assertEqual(both["bbox"], right["bbox"])
        self.assertEqual(both["ignored_non_active_hand_count"], 1)

    def test_switch_is_automatic_but_requires_stability(self):
        selector = AutomaticActiveHandSelector(switch_frames=3)
        selector.select(payload(detection(True, [100, 100, 220, 260])))
        left = detection(False, [350, 100, 470, 260])
        first = selector.select(payload(left))
        second = selector.select(payload(left))
        switched = selector.select(payload(left))
        self.assertFalse(first["valid"])
        self.assertFalse(second["valid"])
        self.assertTrue(switched["valid"])
        self.assertFalse(switched["active_hand_is_right"])
        self.assertTrue(switched["automatic_hand_switch"])
        self.assertEqual(switched["active_hand_generation"], 2)

    def test_no_hand_does_not_silently_switch_identity(self):
        selector = AutomaticActiveHandSelector(switch_frames=2)
        selector.select(payload(detection(False, [100, 100, 220, 260])))
        missing = selector.select(payload())
        self.assertFalse(missing["valid"])
        self.assertFalse(missing["active_hand_is_right"])
        self.assertEqual(missing["active_hand_generation"], 1)

if __name__ == "__main__":
    unittest.main()
