"""Regression for an 80 ms simulated path expiring in wall-clock time."""
import importlib.util
import pathlib
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import rospy
from std_msgs.msg import Bool, Float64, Float64MultiArray

PATH = pathlib.Path(__file__).resolve().parents[1] / 'scripts/egm_position_reference_adapter.py'
SPEC = importlib.util.spec_from_file_location('tested_egm_adapter', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PredictionClockTest(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.now = dict(ros=12.0, wall=1000.0, monotonic=20.0)
        params = {'/use_sim_time': True,
                  '/shared_teleop/control/constrained_velocity_projection': True}
        self.context.enter_context(patch.object(rospy, 'get_param',
                                   side_effect=lambda key, default=None: params.get(key, default)))
        for name in ('Publisher', 'Subscriber', 'Timer', 'on_shutdown', 'logwarn', 'loginfo'):
            self.context.enter_context(patch.object(rospy, name, Mock()))
        self.context.enter_context(patch.object(rospy.Time, 'now',
                                   side_effect=lambda: rospy.Time.from_sec(self.now['ros'])))
        self.context.enter_context(patch.object(MODULE.time, 'time',
                                   side_effect=lambda: self.now['wall']))
        self.context.enter_context(patch.object(MODULE.time, 'monotonic',
                                   side_effect=lambda: self.now['monotonic']))
        self.adapter = MODULE.EgmPositionReferenceAdapter()
        self.adapter.latest_actual = self.adapter.model.reference.copy()
        self.adapter.strict_command_safe_callback(Bool(data=True))
        self.adapter.collision_scale_callback(Float64(data=1.0))
        self.adapter.tick(None)

    def command(self, motion_deadline=12.08):
        values = [.3, 0., 0., 0., 0., 0., 1000.1]
        if motion_deadline is not None:
            values.append(motion_deadline)
        else:
            values[-1] = 1000.08  # Backward-compatible seven-value protocol.
        self.adapter.velocity_callback(Float64MultiArray(data=values))

    def tick(self, ros, wall):
        self.now.update(ros=ros, wall=wall, monotonic=20.0 + wall - 1000.0)
        self.adapter.tick(None)

    def test_slow_simulation_retains_still_checked_motion(self):
        self.command()
        self.tick(12.04, 1000.085)
        self.assertFalse(self.adapter.hard_safety_hold_active)
        self.assertGreater(self.adapter.model.feedforward_velocity[0], 0.)

    def test_motion_horizon_still_expires_at_eighty_simulated_ms(self):
        self.command()
        self.tick(12.04, 1000.05)
        self.tick(12.081, 1000.09)
        self.assertTrue(self.adapter.hard_safety_hold_active)
        self.assertEqual(self.adapter.model.feedforward_velocity[0], 0.)

    def test_wall_watchdog_stops_even_when_simulation_advances_slowly(self):
        self.command()
        self.tick(12.04, 1000.101)
        self.assertTrue(self.adapter.hard_safety_hold_active)

    def test_old_seven_value_commands_retain_their_wall_expiry(self):
        self.command(None)
        self.tick(12.04, 1000.085)
        self.assertTrue(self.adapter.hard_safety_hold_active)

    def test_geometry_rejection_overrides_unexpired_motion(self):
        self.command()
        self.adapter.strict_command_safe_callback(Bool(data=False))
        self.tick(12.02, 1000.025)
        self.assertTrue(self.adapter.hard_safety_hold_active)

    def test_clock_reset_does_not_resume_old_motion(self):
        self.command()
        self.tick(12.04, 1000.05)
        reference = self.adapter.model.reference.copy()
        self.tick(1.0, 1000.06)
        self.assertTrue((reference == self.adapter.model.reference).all())
        self.assertEqual(self.adapter.model.feedforward_velocity[0], 0.)


if __name__ == '__main__':
    unittest.main()
