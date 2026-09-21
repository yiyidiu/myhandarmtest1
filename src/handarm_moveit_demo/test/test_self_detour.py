#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import numpy as np

from handarm_moveit_demo.self_detour import SelfDetour


class SelfDetourContinuityTest(unittest.TestCase):
    def setUp(self):
        self.clock = patch("handarm_moveit_demo.self_detour.time.monotonic", return_value=1.)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.helper = SelfDetour(True, {})
        self.helper.revision = "same-C"
        self.goal = (np.array([.4, 0., 0.]), np.eye(3))
        self.points = [(np.array([x, 0., 0.]), np.eye(3)) for x in (.1, .2, .3, .4)]
        self.helper.active = dict(goal=self.goal, points=self.points, index=0, start=0.)

    def update(self, current, goal=None, valid=True):
        p, r = self.goal if goal is None else goal
        return self.helper.target(p, r, current, np.eye(3), {}, True, "same-C", valid)

    def test_four_waypoints_finish_only_at_the_operator_target(self):
        for index in range(3):
            next_pose = self.update(self.points[index][0])
            np.testing.assert_allclose(next_pose[0], self.points[index+1][0])
        self.assertIsNone(self.update(self.goal[0]))
        self.assertEqual(self.helper.status, "DETOUR_COMPLETE")
        self.assertEqual(self.helper.revision, "same-C")

    def test_new_operator_target_and_input_loss_cancel_pending_motion(self):
        changed = (self.goal[0]+np.array([.03, 0., 0.]), self.goal[1])
        self.assertIsNone(self.update(np.zeros(3), changed))
        self.assertIsNone(self.helper.active)
        self.helper.active = dict(goal=self.goal, points=self.points, index=0, start=0.)
        self.assertIsNone(self.update(np.zeros(3), valid=False))
        self.assertIsNone(self.helper.active)

    def test_robot_motion_during_planning_does_not_cache_a_false_failure(self):
        self.helper.active = None
        self.helper.result = (self.helper.generation, self.goal, self.points,
                              np.zeros(3), "BOUNDED_IK_POSE_DETOUR_CHECKED")
        self.assertIsNone(self.update(np.array([.03, 0., 0.])))
        self.assertIsNone(self.helper.failed_goal)
        self.assertEqual(self.helper.status, "REPLAN_ROBOT_MOVED")


if __name__ == "__main__":
    unittest.main()
