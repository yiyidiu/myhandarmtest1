"""The simulated spring only engages intended, opposing contacts on the object."""
from types import SimpleNamespace
import unittest

import numpy as np

from test_retarget import OPEN, human_hand, load_robot
from object_impedance import ObjectPinch


class ObjectImpedanceTest(unittest.TestCase):
    def setUp(self):
        self.model = load_robot()
        self.q = np.clip(np.asarray(OPEN) + .25, self.model.lower, self.model.upper)
        self.hand = human_hand(bend=(.65, 0, 0, .3), pinch=0)
        self.goal = self.q + .04
        self.spring = ObjectPinch(self.model, [3., 3.2, 1.8, 1.] * 4, 1.5)
        self.target = "target_object::object_link::barrel"
        self.index = "fr3_allegro::algr_lh_if_ds::algr_lh_if_ds_tip_collision_1"
        self.thumb = "fr3_allegro::algr_lh_th_ds::algr_lh_th_ds_tip_collision_1"

    def contacts(self, *pairs, times=(7.94, 7.96, 7.98, 8.)):
        states = [SimpleNamespace(collision1_name=a, collision2_name=b) for a, b in pairs]
        for stamp in times:
            header = SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: stamp))
            self.spring.observe(SimpleNamespace(states=states, header=header))

    def apply(self, intended=((0, 3),), time=8.01):
        return self.spring.apply(self.goal, self.q, self.hand, intended, time)

    def test_requires_intended_bidirectional_pad_contacts(self):
        self.contacts((self.index, self.thumb), (self.thumb, self.target),
                      (self.index, "pinch_stand::support::support"))
        np.testing.assert_array_equal(self.apply(), self.goal)
        self.contacts((self.index, self.target), (self.target, self.thumb), times=(7.8,))
        np.testing.assert_array_equal(self.apply(time=7.81), self.goal)
        self.contacts((self.index, self.target), (self.target, self.thumb))
        np.testing.assert_array_equal(self.apply(intended=((2, 3),)), self.goal)
        measured_pads, _, jacobians, _ = self.model.kinematics(self.q)
        direction = measured_pads[3, 4] - measured_pads[0, 4]
        equilibrium = self.apply()
        delta = (equilibrium - self.q).reshape(4, 4)
        self.assertGreater(direction @ jacobians[0] @ delta[0], 0)
        self.assertLess(direction @ jacobians[3] @ delta[3], 0)
        np.testing.assert_array_equal(equilibrium[4:12], self.goal[4:12])
        self.assertTrue(np.all(equilibrium >= self.model.lower))
        self.assertTrue(np.all(equilibrium <= self.model.upper))

    def test_release_and_expired_contacts_restore_free_motion(self):
        self.contacts((self.index, self.target), (self.target, self.thumb))
        self.apply()
        self.assertEqual(self.spring.partners, (0,))
        np.testing.assert_array_equal(self.apply(intended=()), self.goal)
        self.assertFalse(self.spring.partners)

        self.apply()
        np.testing.assert_array_equal(self.apply(time=8.15), self.goal)
        self.assertFalse(self.spring.partners)

    def test_approach_slows_only_the_commanded_pair_before_contact(self):
        goal = np.clip(self.q + .20, self.model.lower, self.model.upper)
        q = self.spring.approach(goal, self.q, ((0, 3),), .04)
        np.testing.assert_array_equal(q[4:12], goal[4:12])
        for group in (slice(0, 4), slice(12, 16)):
            self.assertLessEqual(np.max(np.abs(q[group]-self.q[group])), .85*.04 + 1e-9)
        self.spring.partners = (0,)
        np.testing.assert_array_equal(self.spring.approach(goal, self.q, ((0, 3),), .04), goal)


if __name__ == "__main__":
    unittest.main()
