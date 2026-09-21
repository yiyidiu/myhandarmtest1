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
    def test_fixed_left_reacquires_left_after_gap_with_both_hands_visible(self):
        selector = AutomaticActiveHandSelector(fixed_is_right=False)
        selector.select(payload(detection(False, [352, 105, 399, 174])), 191.5)
        # Sep 19: both boxes no longer overlap the old crop after a pause.
        # The idle right hand has higher confidence, but must never inherit C.
        left = detection(False, [363, 156, 406, 244], .9757)
        right = detection(True, [188, 397, 262, 487], .9921)
        self.assertFalse(selector.select(payload(left, right), 195.7)['valid'])
        chosen = selector.select(payload(left, right), 195.75)
        self.assertEqual(chosen['bbox'], left['bbox'])
        self.assertFalse(chosen['is_right'])
        self.assertTrue(chosen['track_discontinuity'])
        self.assertEqual(chosen['active_hand_generation'], 1)

    def test_fixed_left_never_switches_to_only_remote_right_hand(self):
        selector = AutomaticActiveHandSelector(fixed_is_right=False)
        selector.select(payload(detection(False, [300, 100, 360, 200])), 1.)
        other = payload(detection(True, [100, 350, 180, 460], .99))
        for t in [1.05, 1.1, 1.2, 2., 20.]:
            result = selector.select(other, t)
            self.assertFalse(result['valid'])
            self.assertFalse(result['active_hand_is_right'])
            self.assertEqual(result['active_hand_generation'], 1)

    def test_fixed_left_label_flip_requires_fresh_spatial_evidence(self):
        selector = AutomaticActiveHandSelector(fixed_is_right=False)
        left = detection(False, [300, 100, 360, 200])
        selector.select(payload(left), 1.)
        edge_on = detection(True, [302, 101, 362, 201])
        followed = selector.select(payload(edge_on), 1.05)
        self.assertTrue(followed['valid'])
        self.assertFalse(followed['is_right'])
        self.assertTrue(followed['detector_reported_is_right'])
        self.assertFalse(followed['track_discontinuity'])
        self.assertFalse(selector.select(payload(edge_on), 2.)['valid'])
        self.assertFalse(selector.select(payload(left), 2.05)['valid'])
        restored = selector.select(payload(left), 2.10)
        self.assertTrue(restored['valid'])
        self.assertTrue(restored['track_discontinuity'])

    def test_fixed_left_is_required_before_first_calibration(self):
        selector = AutomaticActiveHandSelector(fixed_is_right=False)
        self.assertFalse(selector.select(payload(detection(True, [0, 0, 100, 100])), 1.)['valid'])
        self.assertIsNone(selector.active_is_right)

    def test_other_hand_one_frame_false_left_label_cannot_poison_track(self):
        selector = AutomaticActiveHandSelector(fixed_is_right=False)
        left = detection(False, [372, 122, 479, 225])
        selector.select(payload(left), 200.206)
        wrong = detection(False, [179, 406, 242, 477])
        self.assertFalse(selector.select(payload(wrong), 200.239)['valid'])
        right = detection(True, wrong['bbox'])
        returned = selector.select(payload(detection(False, [327, 131, 428, 234]), right), 200.273)
        self.assertTrue(returned['valid'])
        self.assertLess(returned['bbox'][1], 200)

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

    def test_same_track_handedness_flip_does_not_change_identity(self):
        selector = AutomaticActiveHandSelector(
            switch_frames=3, handedness_flip_continuity_iou=0.60
        )
        first = selector.select(
            payload(detection(True, [100, 100, 220, 260]))
        )
        flipped = selector.select(
            payload(detection(False, [103, 101, 223, 261]))
        )
        self.assertTrue(first["is_right"])
        self.assertTrue(flipped["valid"])
        self.assertTrue(flipped["is_right"])
        self.assertFalse(flipped["detector_reported_is_right"])
        self.assertTrue(
            flipped["handedness_stabilized_by_spatial_continuity"]
        )
        self.assertEqual(flipped["active_hand_generation"], 1)

    def test_distant_opposite_box_is_still_a_real_switch_candidate(self):
        selector = AutomaticActiveHandSelector(
            switch_frames=2, handedness_flip_continuity_iou=0.60
        )
        selector.select(payload(detection(True, [100, 100, 220, 260])))
        distant = detection(False, [350, 100, 470, 260])
        pending = selector.select(payload(distant))
        switched = selector.select(payload(distant))
        self.assertFalse(pending["valid"])
        self.assertTrue(switched["valid"])
        self.assertFalse(switched["is_right"])
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
