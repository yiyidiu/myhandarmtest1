"""Operator-direction regression checks for the opt-in five-point palm mode."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src/handarm_moveit_demo/src'))
from handarm_moveit_demo.shared_teleop_core import PalmAxesToolZeroMapper,so3_exp,so3_log
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame


def left_palm():
    p=np.zeros((21,3));p[0]=[.05,.04,.7]
    p[[5,9,13,17]]=p[0]+np.array([[-.03,-.07,0],[-.01,-.09,0],[.01,-.08,0],[.03,-.08,0]])
    return p


class NewPalmDirectionTest(unittest.TestCase):
    def mapper(self, base_translation=False):
        options=({'camera_to_base_translation_matrix':[[0,0,-1],[-1,0,0],[0,-1,0]]}
                 if base_translation else {})
        m=PalmAxesToolZeroMapper([1,1,1],[1,1,1],[1,1,1],np.deg2rad(179),maximum_side_rotation_rad=np.deg2rad(88),**options)
        m.set_hand_zero_rotation(so3_exp([.3,-.2,.4]));return m

    def test_open_left_hand_axes_have_requested_physical_directions(self):
        p=left_palm();r=build_hamer_five_point_palm_frame(p,False)
        self.assertTrue(r.valid)
        np.testing.assert_allclose(r.rotation[:,0],[0,0,-1],atol=1e-12)
        np.testing.assert_allclose(r.rotation[:,1],[1,0,0],atol=1e-12)
        np.testing.assert_allclose(r.rotation[:,2],[0,-1,0],atol=1e-12)
        np.testing.assert_allclose(r.origin,p[0]);self.assertAlmostEqual(np.linalg.det(r.rotation),1.)
        # The ring-finger base contributes to the long axis; it is not the
        # old wrist-to-middle-only construction.
        p[13]+=[0,0,.025];r=build_hamer_five_point_palm_frame(p,False)
        z=p[[5,9,13,17]].mean(0)-p[0];z/=np.linalg.norm(z)
        np.testing.assert_allclose(r.rotation[:,2],z,atol=1e-12)

    def test_translation_in_hand_zero_axes_moves_captured_tool_axes(self):
        m=self.mapper();tool=so3_exp([-.4,.3,.2]);zero=np.array([.3,.1,.4])
        for axis in range(3):
            for sign in [-1,1]:
                local=np.eye(3)[axis]*.01*sign
                p,_=m.map(m.hand_zero_rotation@local,np.eye(3),zero,tool)
                expected=local*np.array([1,-1,1])
                np.testing.assert_allclose(tool.T@(p-zero),expected,atol=1e-12)
        p,_=m.map(m.hand_zero_rotation@np.array([.02,.01,-.01]),so3_exp([.4,.3,.2]),zero,tool)
        q,_=m.map(m.hand_zero_rotation@np.array([.02,.01,-.01]),np.eye(3),zero,tool)
        np.testing.assert_allclose(p,q,atol=1e-12)

    def test_physical_forward_and_up_do_not_flip_with_hand_or_tool_zero(self):
        m=self.mapper(base_translation=True)
        palm=build_hamer_five_point_palm_frame(left_palm(),False).rotation
        zero=np.array([.3,.1,.4])
        # Independent physical expectations, not axes derived from the mapper:
        # camera toward lens -> robot base forward; image up -> base up.
        movements=[([0,0,-.025],[.025,0,0]),
                   ([0,-.025,0],[0,0,.025]),
                   ([.025,0,0],[0,-.025,0])]
        for hand_zero in (palm,so3_exp([.3,-.2,.4])@palm):
            m.set_hand_zero_rotation(hand_zero)
            for tool_zero in (np.diag([-1.,1.,-1.]),so3_exp([-.4,.3,.2])):
                for camera_delta,expected_base_delta in movements:
                    for sign in (-1,1):
                        p,_=m.map(np.array(camera_delta)*sign,np.eye(3),zero,tool_zero)
                        np.testing.assert_allclose(p-zero,np.array(expected_base_delta)*sign,atol=1e-12)
                        twist=m.map_target_velocity(np.r_[np.array(camera_delta)*sign,np.zeros(3)],hand_zero,tool_zero)
                        np.testing.assert_allclose(twist[:3],np.array(expected_base_delta)*sign,atol=1e-12)

    def test_spatial_translation_preserves_signed_rotation_mapping(self):
        old=self.mapper();new=self.mapper(base_translation=True);tool=so3_exp([-.4,.3,.2])
        for rotation in (so3_exp([.12,-.16,.09]),so3_exp([-.12,.16,-.09])):
            _,before=old.map([0,0,0],rotation,[0,0,0],tool)
            _,after=new.map([0,0,0],rotation,[0,0,0],tool)
            np.testing.assert_allclose(after,before,atol=1e-12)
            velocity=np.array([.03,-.01,.02,.10,.07,-.09])
            before=old.map_target_velocity(velocity,old.hand_zero_rotation,tool,relative_hand_rotation=rotation)
            after=new.map_target_velocity(velocity,new.hand_zero_rotation,tool,relative_hand_rotation=rotation)
            np.testing.assert_allclose(after[3:],before[3:],atol=1e-12)

    def test_both_rotation_signs_use_requested_tool_axes_without_reflection(self):
        m=self.mapper();tool=so3_exp([-.4,.3,.2])
        for axis in range(3):
            for sign in [-1,1]:
                v=np.eye(3)[axis]*np.deg2rad(5)*sign
                _,target=m.map(np.zeros(3),so3_exp(v),np.zeros(3),tool)
                np.testing.assert_allclose(so3_log(tool.T@target),v*[1,-1,1],atol=1e-10)
                self.assertAlmostEqual(np.linalg.det(target),1.,places=10)

    def test_feedforward_matches_pose_map_for_combined_motion(self):
        m=self.mapper();tool=so3_exp([-.4,.3,.2]);zero=np.array([.3,0,.4]);h=m.hand_zero_rotation
        p=np.array([.01,.02,-.01]);rotation=so3_exp([.12,-.16,.09]);velocity=np.array([.03,-.01,.02,.10,.07,-.09]);dt=1e-6
        pa,ra=m.map(p,rotation,zero,tool)
        pb,rb=m.map(p+velocity[:3]*dt,so3_exp(h.T@velocity[3:]*dt)@rotation,zero,tool)
        actual=m.map_target_velocity(velocity,h,tool,relative_hand_rotation=rotation)
        np.testing.assert_allclose(actual[:3],(pb-pa)/dt,atol=1e-8)
        np.testing.assert_allclose(actual[3:],so3_log(rb@ra.T)/dt,atol=1e-5)

    def test_missing_zero_and_wrong_hand_are_not_silently_accepted(self):
        m=PalmAxesToolZeroMapper([1]*3,[1]*3,[1]*3,3.)
        with self.assertRaises(ValueError):m.map([0,0,0],np.eye(3),[0,0,0],np.eye(3))
        self.assertFalse(build_hamer_five_point_palm_frame(left_palm(),True).valid)

    def reversed_turn_mappers(self):
        options = dict(
            translation_gain=[1.6, 1.6, 1.2], rotation_gain=[9/7, 1, 3],
            maximum_relative_translation=[.2]*3,
            maximum_relative_rotation_rad=np.deg2rad(179),
            maximum_side_rotation_rad=np.deg2rad(88),
            positive_side_rotation_gain=18/7,
            camera_to_base_translation_matrix=[[0,0,-1],[-1,0,0],[0,-1,0]])
        before = PalmAxesToolZeroMapper(**options)
        after = PalmAxesToolZeroMapper(reverse_palm_turn=True, **options)
        for mapper in (before, after):
            mapper.set_hand_zero_rotation(so3_exp([.3,-.2,.4]))
        return before, after

    def test_reverse_turn_keeps_input_gains_limits_other_angles_and_position(self):
        before, after = self.reversed_turn_mappers()
        tool = so3_exp([-.4,.3,.2])
        # +/-10 degrees originally request +25.714/-12.857 degrees. Reverse
        # those outputs, without exchanging the two input-dependent gains.
        for x,y,z in ((10,0,0),(-10,0,0),(35,0,0),(-70,0,0),
                      (10,7,-4),(-10,-7,4),(0,10,5)):
            hand = (so3_exp(np.deg2rad([x,0,0])) @
                    so3_exp(np.deg2rad([0,y,0])) @
                    so3_exp(np.deg2rad([0,0,z])))
            output_x = np.clip(x*(18/7 if x >= 0 else 9/7), -88, 88)
            expected = (so3_exp(np.deg2rad([-output_x,0,0])) @
                        so3_exp(np.deg2rad([0,-y,0])) @
                        so3_exp(np.deg2rad([0,0,3*z])))
            old_p, _ = before.map([.01,-.02,-.03],hand,[.3,0,.4],tool)
            new_p, new_r = after.map([.01,-.02,-.03],hand,[.3,0,.4],tool)
            np.testing.assert_allclose(new_p, old_p, atol=1e-12)
            np.testing.assert_allclose(new_r, tool @ expected, atol=1e-10)

    def test_reversed_turn_velocity_follows_changed_pose_for_both_directions(self):
        before, after = self.reversed_turn_mappers()
        tool = so3_exp([-.4,.3,.2])
        hand_zero = after.hand_zero_rotation
        velocity = np.array([.03,-.01,.02,.10,.07,-.09])
        dt = 1e-6
        for x in (-.2,.2,1.3):
            hand = so3_exp([x,-.1,.08])
            advanced = so3_exp(hand_zero.T @ velocity[3:] * dt) @ hand
            p0, r0 = after.map([0,0,0], hand, [0,0,0], tool)
            p1, r1 = after.map(velocity[:3]*dt, advanced, [0,0,0], tool)
            new_v = after.map_target_velocity(
                velocity,hand_zero,tool,relative_hand_rotation=hand)
            old_v = before.map_target_velocity(
                velocity,hand_zero,tool,relative_hand_rotation=hand)
            np.testing.assert_allclose(new_v[:3],old_v[:3],atol=1e-12)
            np.testing.assert_allclose(new_v[:3],(p1-p0)/dt,atol=1e-8)
            np.testing.assert_allclose(new_v[3:],so3_log(r1@r0.T)/dt,atol=1e-5)


if __name__=='__main__':unittest.main()
