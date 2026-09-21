"""Physical gesture signs and compound-pose preservation for whole wrist mode."""
import sys
import unittest
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/handarm_moveit_demo/src')]
from handarm_moveit_demo.shared_teleop_core import WholeWristPoseMapper, so3_exp, so3_log
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame


def palm_points():
    points = np.zeros((21, 3))
    points[0] = [.05, .04, .7]
    points[[5, 9, 13, 17]] = points[0] + np.array([
        [-.03, -.07, 0], [-.01, -.09, 0], [.01, -.08, 0], [.03, -.08, 0]])
    return points


class WholeWristMappingTest(unittest.TestCase):
    def setUp(self):
        config = yaml.safe_load((ROOT / 'src/handarm_moveit_demo/config/shared_teleop.yaml').read_text())
        self.mapper = WholeWristPoseMapper(
            config['mapping_profiles']['whole_wrist_pose']['translation_matrix'],
            config['mapping_profiles']['whole_wrist_pose']['translation_gain'],
            [2., 2., 2.], np.deg2rad(179), np.deg2rad(88))
        self.points = palm_points()
        self.h0 = build_hamer_five_point_palm_frame(self.points, False).rotation
        self.e0 = np.diag([-1., 1., -1.])

    def relative_from_camera_motion(self, camera_rotation):
        moved = ((self.points - self.points[0]) @ camera_rotation.T + self.points[0])
        hand = build_hamer_five_point_palm_frame(moved, False).rotation
        return self.h0.T @ hand

    def test_fixed_ring_basis_preserves_physical_pose_and_velocity(self):
        config=yaml.safe_load((ROOT/'src/handarm_moveit_demo/config/shared_teleop.yaml').read_text())
        basis=np.array(config['mapping_profiles']['whole_wrist_pose']['wrist_ring_to_palm_basis'])
        ring=WholeWristPoseMapper(self.mapper.translation_matrix,self.mapper.translation_gain,
            [2,2,2],np.deg2rad(179),np.deg2rad(88))
        ring.rotation_matrix=ring.rotation_matrix@basis.T
        relative=self.relative_from_camera_motion(so3_exp([.2,-.3,.15]))
        ring_relative=basis@relative@basis.T
        for sign in (-1,1):
            r=relative if sign>0 else relative.T
            rr=ring_relative if sign>0 else ring_relative.T
            pp,rp=self.mapper.map([.02,-.01,.03],r,[.3,0,.4],self.e0)
            pb,rb=ring.map([.02,-.01,.03],rr,[.3,0,.4],self.e0)
            np.testing.assert_allclose(rp,rb,atol=1e-10)
            np.testing.assert_allclose(pp,pb,atol=1e-10)
            velocity=np.array([.01,.02,.03,.04,-.05,.06])
            vp=self.mapper.map_target_velocity(velocity,self.h0,self.e0,relative_hand_rotation=r)
            vb=ring.map_target_velocity(velocity,self.h0@basis.T,self.e0,relative_hand_rotation=rr)
            np.testing.assert_allclose(vp,vb,atol=1e-7)

    def test_physical_finger_approach_directions_not_euler_labels(self):
        angle = np.deg2rad(20)
        # Camera rotation Y+: little-finger MCP approaches the lens.
        # Camera rotation X+: four-finger MCPs approach the lens.
        for camera_axis, expected_tool_axis in (
                ([0, 1, 0], [-1, 0, 0]),
                ([1, 0, 0], [0, 1, 0]),
                ([0, 0, -1], [0, 0, -1])):
            for sign in (-1, 1):
                relative = self.relative_from_camera_motion(
                    so3_exp(np.array(camera_axis) * sign * angle))
                p, target = self.mapper.map([0, 0, 0], relative, [.3, 0, .4], self.e0)
                np.testing.assert_allclose(so3_log(self.e0.T @ target),
                    np.array(expected_tool_axis) * sign * angle, atol=1e-10)
                np.testing.assert_allclose(p, [.3, 0, .4])
        # User-labelled right turn is -tool-X, as identified in the actual
        # Sep 19 video. Do not infer its name from approach-vector tilt.
        relative = self.relative_from_camera_motion(so3_exp([0, angle, 0]))
        _, target = self.mapper.map([0, 0, 0], relative, [0, 0, 0], self.e0)
        self.assertGreater(target[1, 2], 0)

    def test_combined_pose_preserves_order_and_zero(self):
        camera = so3_exp([0, 0, -.2]) @ so3_exp([0, .3, 0]) @ so3_exp([-.1, 0, 0])
        relative = self.relative_from_camera_motion(camera)
        _, actual = self.mapper.map([0, 0, 0], relative, [0, 0, 0], self.e0)
        expected = self.e0 @ so3_exp([0, 0, -.2]) @ so3_exp([-.3, 0, 0]) @ so3_exp([0, -.1, 0])
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        p, r = self.mapper.map([0, 0, 0], np.eye(3), [.3, 0, .4], self.e0)
        np.testing.assert_allclose(r, self.e0, atol=1e-12)
        np.testing.assert_allclose(p, [.3, 0, .4], atol=1e-12)

    def test_camera_depth_controls_height_and_image_vertical_controls_forward(self):
        # Independent expectations from the user's specified physical actions.
        for hand, expected in (([0, 0, -.02], [0, 0, -.024]),
                               ([-.02, 0, 0], [0, .04, 0]),
                               ([0, -.02, 0], [.032, 0, 0])):
            for sign in (-1, 1):
                p, r = self.mapper.map(np.array(hand)*sign, np.eye(3), [0, 0, 0], self.e0)
                np.testing.assert_allclose(p, np.array(expected)*sign, atol=1e-12)
                np.testing.assert_allclose(r, self.e0, atol=1e-12)
                velocity = self.mapper.map_target_velocity(
                    np.r_[np.array(hand)*sign, np.zeros(3)], self.h0, self.e0,
                    relative_hand_rotation=np.eye(3))
                np.testing.assert_allclose(velocity[:3], np.array(expected)*sign, atol=1e-12)
                np.testing.assert_allclose(velocity[3:], np.zeros(3), atol=1e-12)

    def test_side_bound_shortens_whole_rotation_without_axis_change(self):
        for sign in (-1, 1):
            requested = np.deg2rad([110*sign, 10, -15])
            a = self.mapper.rotation_matrix
            relative = a.T @ so3_exp(requested) @ a
            _, actual = self.mapper.map([0, 0, 0], relative, [0, 0, 0], np.eye(3))
            side = np.arctan2(-actual[1, 2], actual[2, 2])
            self.assertLessEqual(abs(side), np.deg2rad(88) + 1e-8)
            vector = so3_log(actual)
            np.testing.assert_allclose(vector/np.linalg.norm(vector),
                requested/np.linalg.norm(requested), atol=1e-10)

    def test_angular_feedforward_matches_compound_pose_mapping(self):
        relative = so3_exp([.2, -.3, -.1])
        velocity = np.array([.03, -.01, .02, .10, .07, -.09])
        dt = 1e-6
        p0, r0 = self.mapper.map([0, 0, 0], relative, [0, 0, 0], self.e0)
        p1, r1 = self.mapper.map(velocity[:3]*dt,
            so3_exp(self.h0.T @ velocity[3:]*dt) @ relative, [0, 0, 0], self.e0)
        mapped = self.mapper.map_target_velocity(velocity, self.h0, self.e0,
                                                  relative_hand_rotation=relative)
        np.testing.assert_allclose(mapped[:3], (p1-p0)/dt, atol=1e-8)
        np.testing.assert_allclose(mapped[3:], so3_log(r1 @ r0.T)/dt, atol=1e-5)


class WholeWristUniformGainTest(unittest.TestCase):
    def setUp(self):
        config = yaml.safe_load((ROOT / 'src/handarm_moveit_demo/config/shared_teleop.yaml').read_text())
        profile = config['mapping_profiles']['whole_wrist_pose']
        self.assertEqual(profile['orientation_gain'], 1.3)
        self.mapper = WholeWristPoseMapper(profile['translation_matrix'],
            profile['translation_gain'], [2., 2., 2.], np.deg2rad(179),
            np.deg2rad(88), orientation_gain=profile['orientation_gain'])

    def hand_relative(self, tool_vector):
        basis = self.mapper.rotation_matrix
        return basis.T @ so3_exp(tool_vector) @ basis

    def test_gain_preserves_combined_axis_and_zero_without_changing_translation(self):
        for degrees in ([20, 0, 0], [-20, 0, 0], [0, 20, 0],
                        [0, 0, 20], [25, -15, 12]):
            vector = np.deg2rad(degrees)
            p, r = self.mapper.map([.01, -.02, .03], self.hand_relative(vector),
                                   [.3, 0, .4], np.eye(3))
            np.testing.assert_allclose(so3_log(r), 1.3*vector, atol=1e-10)
            np.testing.assert_allclose(p, [.332, -.02, .436], atol=1e-12)
        p, r = self.mapper.map([0, 0, 0], np.eye(3), [.3, 0, .4], np.eye(3))
        np.testing.assert_allclose(p, [.3, 0, .4], atol=1e-12)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-12)

    def test_overrange_and_compound_requests_keep_both_side_limits(self):
        for side in (-170, -120, -90, -70, 70, 90, 120, 170):
            for pitch, roll in ((0, 0), (15, -10), (-30, 20)):
                requested = np.deg2rad([side, pitch, roll])
                _, r = self.mapper.map([0, 0, 0], self.hand_relative(requested),
                                       [0, 0, 0], np.eye(3))
                actual_side = np.arctan2(-r[1, 2], r[2, 2])
                self.assertLessEqual(abs(actual_side), np.deg2rad(88) + 1e-8)
                if pitch == roll == 0:
                    self.assertAlmostEqual(np.rad2deg(actual_side), np.sign(side)*88, places=4)

    def test_velocity_matches_amplified_pose_and_stops_pushing_past_limit(self):
        relative = self.hand_relative(np.deg2rad([25, -15, 12]))
        omega = np.array([.1, .07, -.09])
        dt = 1e-6
        _, r0 = self.mapper.map([0, 0, 0], relative, [0, 0, 0], np.eye(3))
        _, r1 = self.mapper.map([0, 0, 0], so3_exp(omega*dt) @ relative,
                               [0, 0, 0], np.eye(3))
        mapped = self.mapper.map_target_velocity(np.r_[np.zeros(3), omega],
            np.eye(3), np.eye(3), relative_hand_rotation=relative)
        np.testing.assert_allclose(mapped[3:], so3_log(r1 @ r0.T)/dt, atol=1e-5)
        for sign in (-1., 1.):
            relative = self.hand_relative(np.deg2rad([100*sign, 0, 0]))
            hand_omega = self.mapper.rotation_matrix.T @ np.array([sign*.2, 0, 0])
            mapped = self.mapper.map_target_velocity(np.r_[np.zeros(3), hand_omega],
                np.eye(3), np.eye(3), relative_hand_rotation=relative)
            self.assertLess(np.linalg.norm(mapped[3:]), .005)


if __name__ == '__main__':
    unittest.main()
