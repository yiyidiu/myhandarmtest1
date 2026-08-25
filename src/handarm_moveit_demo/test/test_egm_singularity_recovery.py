#!/usr/bin/env python3

import math
from pathlib import Path
import unittest

import numpy as np

from handarm_moveit_demo.egm_singularity_recovery import (
    DirectionalSingularityRecovery, UrdfSerialChain, jacobian_condition)


PACKAGE = Path(__file__).resolve().parents[1]
URDF = PACKAGE.parent / "abb120_moveit_config1/config/gazebo_handarm.urdf"
INITIAL = np.asarray([0.0, 0.0, 0.0, 0.0, math.pi / 2.0, 0.0])
LOWER = [-2.87979, -1.91986, -1.91986, -2.79253, -2.094395, -6.981317]
UPPER = [2.87979, 1.91986, 1.22173, 2.79253, 2.094395, 6.981317]
MAXIMUM_VELOCITY = [4.36332, 4.36332, 4.36332, 5.58505, 5.58505, 7.33038]


class EgmSingularityRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = UrdfSerialChain.from_urdf_xml(
            URDF.read_text(encoding="utf-8"))

    def resolver(self, **overrides):
        values = dict(
            chain=self.chain,
            preferred_configuration=INITIAL,
            lower_limits=LOWER,
            upper_limits=UPPER,
            maximum_velocity=MAXIMUM_VELOCITY,
        )
        values.update(overrides)
        return DirectionalSingularityRecovery(**values)

    def test_urdf_chain_and_initial_condition_match_irb120(self):
        self.assertEqual(
            self.chain.joint_names,
            ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"))
        position, _, jacobian = self.chain.forward_and_jacobian(INITIAL)
        condition, _, singular_values, _ = jacobian_condition(jacobian)
        np.testing.assert_allclose(position, [0.302, 0.0, 0.388], atol=1.0e-6)
        self.assertAlmostEqual(condition, 12.4558, places=3)
        self.assertGreater(singular_values[-1], 0.14)

    def test_analytic_jacobian_matches_finite_difference(self):
        q = np.asarray([0.2, -0.35, -0.4, 0.3, 1.1, -0.2])
        position, rotation, jacobian = self.chain.forward_and_jacobian(q)
        epsilon = 1.0e-7
        numerical = np.zeros_like(jacobian)
        for index in range(6):
            shifted = q.copy()
            shifted[index] += epsilon
            next_position, next_rotation, _ = self.chain.forward_and_jacobian(
                shifted)
            numerical[:3, index] = (next_position - position) / epsilon
            delta = next_rotation @ rotation.T
            numerical[3:, index] = np.asarray([
                delta[2, 1] - delta[1, 2],
                delta[0, 2] - delta[2, 0],
                delta[1, 0] - delta[0, 1],
            ]) / (2.0 * epsilon)
        np.testing.assert_allclose(jacobian, numerical, atol=2.0e-6)

    def test_normal_damped_inverse_tracks_cartesian_twist(self):
        resolver = self.resolver()
        requested = np.asarray([0.02, -0.01, 0.015, 0.08, -0.04, 0.05])
        result = resolver.resolve(INITIAL, requested)
        _, _, jacobian = self.chain.forward_and_jacobian(INITIAL)
        self.assertEqual(result.mode, "NORMAL")
        self.assertFalse(result.recovery_active)
        np.testing.assert_allclose(
            jacobian @ result.joint_velocity, requested, atol=2.0e-6)

    def test_hard_singularity_retreats_without_zero_or_repeated_reset(self):
        resolver = self.resolver()
        resolver.resolve(INITIAL, np.zeros(6))
        # This is the exact posture captured from the reported status-2 loop.
        bad = np.asarray([
            0.5053, 0.09236, -1.3292, -0.7011, 1.6276, 0.5150])
        dangerous = np.asarray([0.5, -0.5, -0.5, -6.0, 3.0, 6.0])
        result = resolver.resolve(bad, dangerous)
        self.assertEqual(result.mode, "SINGULARITY_RECOVERY")
        self.assertTrue(result.recovery_active)
        self.assertGreater(result.condition_number, 1000.0)
        self.assertGreater(np.linalg.norm(result.joint_velocity), 0.05)
        self.assertGreater(np.dot(
            INITIAL - bad, result.joint_velocity), 0.0)
        self.assertTrue(np.all(
            np.abs(result.joint_velocity) <=
            0.45 * np.asarray(MAXIMUM_VELOCITY) + 1.0e-9))
        self.assertLess(
            result.predicted_condition_number, result.condition_number)

    def test_position_hold_reset_clears_recovery_memory(self):
        resolver = self.resolver()
        bad = np.asarray([
            0.5053, 0.09236, -1.3292, -0.7011, 1.6276, 0.5150])
        resolver.resolve(bad, [0.5, -0.5, -0.5, -6.0, 3.0, 6.0])
        self.assertTrue(resolver.recovery_active)
        latched = np.asarray([0.2, -0.1, -0.3, 0.1, 1.2, -0.2])
        resolver.reset(latched)
        self.assertFalse(resolver.recovery_active)
        self.assertIsNone(resolver.blocked_direction)
        self.assertEqual(resolver.release_counter, 0)
        np.testing.assert_allclose(resolver.last_safe, latched)

    def test_recovery_holds_deeper_component_until_operator_retreats(self):
        resolver = self.resolver(release_cycles=3)
        dangerous = np.asarray([0.5, -0.5, -0.5, -6.0, 3.0, 6.0])
        bad = np.asarray([
            0.5053, 0.09236, -1.3292, -0.7011, 1.6276, 0.5150])
        resolver.resolve(INITIAL, np.zeros(6))
        entered = resolver.resolve(bad, dangerous)
        self.assertTrue(entered.recovery_active)
        held = resolver.resolve(INITIAL, dangerous)
        self.assertTrue(held.recovery_active)
        self.assertGreater(held.blocked_twist_component, 0.0)
        self.assertAlmostEqual(
            float(np.dot(
                resolver.blocked_direction, held.projected_twist)),
            0.0, places=10)
        for _ in range(3):
            released = resolver.resolve(INITIAL, -dangerous)
        self.assertFalse(released.recovery_active)
        self.assertEqual(released.mode, "RECOVERY_RELEASED")

    def test_exact_singularity_produces_finite_protected_motion(self):
        resolver = self.resolver()
        resolver.resolve(INITIAL, np.zeros(6))
        result = resolver.resolve(np.zeros(6), np.ones(6))
        self.assertTrue(result.recovery_active)
        self.assertTrue(math.isinf(result.condition_number))
        self.assertTrue(np.all(np.isfinite(result.joint_velocity)))
        self.assertGreater(np.linalg.norm(result.joint_velocity), 0.0)

    def test_last_safe_target_is_always_inside_release_band(self):
        resolver = self.resolver(release_cycles=3)
        resolver.resolve(INITIAL, np.zeros(6))
        recorded_safe = resolver.last_safe.copy()
        # This posture was captured from a zero-input runtime stall.  It is
        # below the damping threshold but above the release threshold.
        damped_only = np.asarray([
            -0.32715599, 1.77334973, -1.62919826,
            -2.30375350, 0.99109641, -0.68633146])
        intermediate = resolver.resolve(damped_only, np.zeros(6))
        self.assertGreater(intermediate.condition_number, 45.0)
        self.assertLess(intermediate.condition_number, 60.0)
        np.testing.assert_allclose(resolver.last_safe, recorded_safe)

        bad = np.asarray([
            0.5053, 0.09236, -1.3292, -0.7011, 1.6276, 0.5150])
        resolver.resolve(bad, np.ones(6))
        joints = bad.copy()
        for _ in range(1000):
            result = resolver.resolve(joints, np.zeros(6))
            joints += result.joint_velocity * 0.01
            if not result.recovery_active:
                break
        self.assertFalse(result.recovery_active)
        self.assertLessEqual(result.condition_number, 45.0)


if __name__ == "__main__":
    unittest.main()
