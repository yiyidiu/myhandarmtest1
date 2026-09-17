"""Protect against wrist-ray coupling and against stopping the arm on bad fingers."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/handarm_moveit_demo/src'))
from perception_mediapipe.fingers import observe_fingers
from perception_mediapipe.run_live import PosePublisher
from handarm_moveit_demo.hamer_input_contract import HamerPacketContract


def hand(curl=False):
    p=np.zeros((21,3));p[:,2]=.8
    for digit,base in enumerate([1,5,9,13,17]):
        p[base]=[(digit-2)*.018,.06,.8]
        angles=np.deg2rad([0,70,140]) if curl else np.zeros(3)
        for j,(length,a)in enumerate(zip([.03,.024,.017],angles),1):
            p[base+j]=p[base+j-1]+[0,length*np.cos(a),length*np.sin(a)]
    return p


def project(p):return p[:,:2]/p[:,2,None]*600+[320,240]


class FingerRegression(unittest.TestCase):
    def test_wrist_point_shift_does_not_create_finger_closure(self):
        p=hand();first=observe_fingers(p,project(p),.9)
        p[0]+=[.04,-.05,.15]
        moved=observe_fingers(p,project(p),.9)
        self.assertTrue(first['valid'] and moved['valid'])
        np.testing.assert_allclose(moved['flexion'],first['flexion'],atol=1e-10)
        self.assertLess(max(moved['flexion']),1e-6)

    def test_visible_fist_is_retained_and_conflicting_shape_is_held(self):
        closed=hand(True)
        good=observe_fingers(closed,project(closed),.9)
        self.assertTrue(good['valid']);self.assertGreater(min(good['flexion'][1:]),.6)
        uncertain=observe_fingers(closed,project(hand()),.9)
        self.assertFalse(uncertain['valid'])
        self.assertIn('HOLD_GRIP',uncertain['invalid_reason'])

    def test_uncertain_fingers_do_not_disable_pose_or_cancel_C(self):
        pub=PosePublisher('world');p=hand();good=observe_fingers(p,project(p),.9)
        pose=dict(valid=True,position=[0,0,.8],rotation=np.eye(3).reshape(-1).tolist(),
                  confidence=.9,is_right=True,fingers=good)
        pub.request_reference(0)
        for i in range(1,5):first=pub.packet(pose,100+i/30,i,.02,.04)
        self.assertTrue(first['control_enabled'])
        token=first['control_reference_token']
        pose['fingers']=observe_fingers(hand(True),project(hand()),.9)
        pose['position']=[.03,0,.8]
        result=pub.packet(pose,100+5/30,5,.02,.04)
        accepted=HamerPacketContract('camera_color_optical_frame',.25,True,True).validate(result)
        self.assertTrue(accepted['observation_valid'])
        self.assertFalse(accepted['finger_tracking_valid'])
        self.assertEqual(result['control_reference_token'],token)
        self.assertEqual(result['wrist_position_m'],pose['position'])


if __name__=='__main__':unittest.main()
