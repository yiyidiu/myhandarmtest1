"""Regressions for the large-tilt translation and retained-pinch failures."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from test_retarget import HUMAN, OPEN, ROBOT, Retargeter, human_hand, load_robot
from test_wrist_hand import READY, arm_model
from test_fingertip_graph import aim_finger, retained_ring_pose
from test_wrist_motion import POSITION, PALM_TO_CAMERA
from wrist_motion import PalmMotionSupport
from arm_reference import ArmReference, SelfClearance


class EdgeTracking(unittest.TestCase):
    def setUp(self):
        self.shape = human_hand()[[0, 5, 9, 13, 17]]
        palm = self.shape @ PALM_TO_CAMERA.T + POSITION
        self.xyz = np.vstack([palm, 0.55 * palm[0] + 0.45 * palm[1:]])
        self.rays = palm[:, :2] / palm[:, 2:]
        self.motion = PalmMotionSupport(self.xyz, POSITION, PALM_TO_CAMERA, self.rays)
        for i in range(6):
            self.motion.update(self.xyz, POSITION, PALM_TO_CAMERA, 0.04 * i, True, self.rays)

    def test_sparse_translation_and_bounded_recovery(self):
        partial = self.xyz.copy()
        partial[3:] = np.nan
        for i, x in enumerate(np.linspace(0, 0.04, 33)):
            p = POSITION + [0.0, x, 0.0]
            self.motion.update(partial + [0.0, x, 0.0], p, PALM_TO_CAMERA, 0.3 + 0.04 * i, True)
        self.assertEqual(self.motion.mode, "translation")
        self.assertAlmostEqual(self.motion.position[1] - POSITION[1], 0.04, delta=0.0021)
        np.testing.assert_allclose(self.motion.rotation, PALM_TO_CAMERA, atol=1e-9)
        p = POSITION + [0.0, 0.12, 0.0]
        for i in range(7):
            self.motion.update(None, p, PALM_TO_CAMERA, 2.0 + 0.04 * i, True)
        previous = self.motion.position.copy()
        for i in range(20):
            self.motion.update(self.xyz + [0.0, 0.12, 0.0], p, PALM_TO_CAMERA, 2.4 + 0.04 * i, True)
            self.assertLessEqual(np.linalg.norm(self.motion.position - previous), 0.010001)
            previous = self.motion.position.copy()
        np.testing.assert_allclose(previous, p, atol=0.0001)

    def test_image_only_allows_lateral_translation_and_rejects_finger_bias(self):
        for i in range(15):
            self.motion.update(
                None, POSITION + [0.013, 0.014, 0.012], PALM_TO_CAMERA, 0.3 + 0.04 * i, True, self.rays
            )
        np.testing.assert_allclose(self.motion.position, POSITION, atol=1e-9)
        for i, x in enumerate(np.linspace(0, 0.04, 30)):
            p = POSITION + [x, 0.0, 0.0]
            rays = (self.xyz[:5, :2] + [x, 0]) / self.xyz[:5, 2:]
            self.motion.update(None, p, PALM_TO_CAMERA, 1.0 + 0.04 * i, True, rays)
        self.assertAlmostEqual(self.motion.position[0] - POSITION[0], 0.04, delta=0.0021)
        self.assertEqual(self.motion.position[2], POSITION[2])

    def test_archived_contact_objectives_keep_interposed_object(self):
        from compare_paper_objectives import paper_solver
        model = load_robot()
        m = paper_solver.Retargeter(model, OPEN, human_hand())
        q = np.array(OPEN)
        for t in np.r_[np.zeros(24), np.linspace(0, 1, 21), np.ones(15)]:
            h = retained_ring_pose(t, t)
            delta = h[16] - h[4]
            aim_finger(h, HUMAN[2], h[16] + 0.006 * delta / np.linalg.norm(delta))
            q = m.step(h, q, 0.05, measured=OPEN)  # No robot pad contact can be inferred.
            self.assertEqual(m.gesture, "thumb_ring_tip")
        for _ in range(10):
            q = m.step(human_hand(), q, 0.05, measured=OPEN)
        self.assertEqual(m.gesture, "free")


class ArmReferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = arm_model()
        hand = load_robot()
        self.guard = SelfClearance(
            (ROBOT / "urdf/fr3_allegro_left.urdf").read_text(),
            self.model.names + hand.names,
            ROBOT / "worlds/bench.world",
        )
        self.addCleanup(self.guard.close)

    def test_thumb_housing_keeps_other_digits_mobile_and_allows_release(self):
        initial = np.asarray(OPEN)
        goal = initial.copy()
        goal[4:12] += [.05, .7, .6, .4, .03, .8, .5, .4]
        # Actual measured palm/proximal-thumb contact from the 16:51 recording.
        goal[12:] = [.400481, -.113572, 1.52106, .10548]
        q = initial.copy()
        for u in np.linspace(0., 1., 31):
            q = self.guard.limit_hand(q, initial + u*(goal - initial))
        self.assertGreater(np.max(np.abs(q[12:] - goal[12:])), .10)
        np.testing.assert_allclose(q[:12], goal[:12], atol=1e-8)
        for _ in range(30):
            q = self.guard.limit_hand(q, q + np.clip(initial - q, -.1, .1))
        np.testing.assert_allclose(q, initial, atol=1e-8)

    def test_swept_geometry_allows_retreat(self):
        safe = [0.49135454, -0.86916190, 1.40910070, -2.84665824, -1.37499707, 4.01133107, -0.36803427]
        blocked = [0.48527583, -0.86654246, 1.42606559, -2.85099087, -1.37041855, 4.02446019, -0.36137537]
        self.assertFalse(self.guard.allows(np.r_[safe, OPEN], np.r_[blocked, OPEN]))
        self.assertTrue(self.guard.allows(np.r_[blocked, OPEN], np.r_[safe, OPEN]))

    def test_fixed_rate_velocity_acceleration_and_expiry(self):
        accel = np.array([7.0, 7.0, 7.0, 8.0, 10.0, 12.0, 14.0])
        r = ArmReference(self.model, self.guard, accel)
        measured = READY.copy()
        goal = READY + [0.1, 0.04, -0.06, 0.08, 0.0, -0.12, 0.4]
        for i in range(60):
            if i < 30 and i % 4 == 0:
                r.goal_at(goal, measured, i * 0.01)
            old = r.velocity.copy()
            measured = r.tick(measured, OPEN, i * 0.01, 0.01)
            self.assertFalse(r.blocked)
            self.assertTrue(np.all(np.abs(r.velocity - old) <= 0.01 * accel + 1e-9))
            self.assertTrue(np.all(np.abs(r.velocity) <= self.model.speed))
        self.assertLess(np.linalg.norm(r.velocity), 0.001)


class HomeFeedback(unittest.TestCase):
    def test_ros_shutdown_is_not_a_homing_failure(self):
        import motion_demo as home

        with patch.object(home.rospy, "is_shutdown", return_value=True):
            with self.assertRaises(home.rospy.ROSInterruptException):
                home.wait_for_home([], 0.)

    def test_teleop_readiness_requires_continuously_settled_measured_joints(self):
        import motion_demo as home

        goal = Mock()
        goal.get_state.return_value = home.GoalStatus.ABORTED
        goal.get_result.return_value.error_code = home.FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED
        names = ["fr3_joint1", "algr_lh_if_axl"]
        for kind in ("stable", "wrong_finger", "oscillation", "stale"):
            with self.subTest(kind=kind):
                iteration = [0]
                def read(*_args, **_kwargs):
                    iteration[0] += 1
                    stamp = home.rospy.Time.from_sec(20 + .05*iteration[0])
                    error = {"stable": .013, "wrong_finger": .8,
                             "oscillation": .2 * (-1)**iteration[0], "stale": 0.}[kind]
                    return SimpleNamespace(header=SimpleNamespace(stamp=stamp), name=names,
                                           position=[0., np.deg2rad(error)])

                def now():
                    lag = .8 if kind == "stale" else .01
                    return home.rospy.Time.from_sec(20 + .05*iteration[0] + lag)

                with patch.object(home.rospy, "wait_for_message", side_effect=read), \
                     patch.object(home.rospy.Time, "now", side_effect=now), \
                     patch.object(home.time, "monotonic", side_effect=lambda: 2.*iteration[0]):
                    targets = [((goal, names), [0., 0.])]
                    if kind == "stable":
                        home.wait_for_home(targets, 20.)
                        self.assertGreaterEqual(iteration[0], 9)
                    else:
                        with self.assertRaises(RuntimeError):
                            home.wait_for_home(targets, 20.)


if __name__ == "__main__":
    unittest.main()
