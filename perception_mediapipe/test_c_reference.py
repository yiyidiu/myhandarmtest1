"""Regression for C being blocked by a stale, previously accepted orientation.

Controlled positions/fingers isolate the state-transition fault; this is not
a claim that a new live human calibration has been measured.
"""
import copy
import unittest
import numpy as np
from perception_mediapipe.run_live import PosePublisher
from perception_hamer.src.finger_observation import FINGER_FEATURE_DEFINITION


def pose(degrees):
    a=np.deg2rad(degrees);c,s=np.cos(a),np.sin(a)
    return dict(valid=True,reason='',position=[.02,.08,.8],
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]]).reshape(-1).tolist(),
        confidence=.9,is_right=False,geometry_ms=.5,
        fingers=dict(contract_version=1,feature_definition=FINGER_FEATURE_DEFINITION,
                     valid=True,flexion=[.1]*5,confidence=.9,invalid_reason=''))


class CReferenceRegression(unittest.TestCase):
    def sample(self,publisher,value,seq):
        return publisher.packet(copy.deepcopy(value),100+seq/30,seq,.02,.04)

    def test_new_C_can_calibrate_beyond_old_jump_limit_without_reference_jump(self):
        pub=PosePublisher('world');old,current=pose(0),pose(137)
        self.sample(pub,old,1)
        rejected=self.sample(pub,current,2)
        self.assertFalse(rejected['measurement_valid_before_C_gate'])
        self.assertEqual(rejected['measurement_reason'],'orientation_innovation_exceeds_hard_limit')
        pub.request_reference(2)
        # A stale/equal-sequence packet cannot prime the new filter history.
        self.assertFalse(self.sample(pub,old,2)['control_enabled'])
        packets=[self.sample(pub,current,i) for i in range(3,7)]
        self.assertTrue(all(not p['control_enabled'] for p in packets[:-1]))
        first=packets[-1]
        self.assertTrue(first['valid'] and first['control_enabled'])
        self.assertEqual(first['calibration_status'],'calibrated')
        self.assertEqual(first['control_reference_pose']['sequence'],first['sequence'])
        np.testing.assert_allclose(first['palm_rotation_row_major'],current['rotation'],atol=1e-12)
        self.assertEqual(first['control_reference_pose']['palm_rotation_row_major'],first['palm_rotation_row_major'])
        self.assertEqual(first['control_reference_pose']['wrist_position_m'],first['wrist_position_m'])
        # A later outlier remains rejected; it cannot reset or move C by itself.
        bad=self.sample(pub,old,7)
        self.assertFalse(bad['valid'])
        self.assertTrue(bad['control_enabled'])
        recovered=self.sample(pub,current,8)
        self.assertTrue(recovered['valid'])
        self.assertEqual(recovered['control_reference_token'],first['control_reference_token'])

    def test_invalid_pose_and_closed_hand_cannot_be_armed_by_reset(self):
        for mode in ['world','rgbd']:
            pub=PosePublisher(mode);pub.request_reference(0)
            for i in range(1,6):
                out=self.sample(pub,dict(valid=False,reason='NO_HAND'),i)
                self.assertFalse(out['control_enabled'])
                self.assertEqual(out['calibration_status'],'waiting_for_hand_pose')
            closed=pose(0);closed['fingers']['flexion']=[.8]*5
            for i in range(6,11):
                out=self.sample(pub,closed,i)
                self.assertFalse(out['control_enabled'])
                self.assertEqual(out['calibration_status'],'waiting_for_open_hand')


if __name__=='__main__':unittest.main()
