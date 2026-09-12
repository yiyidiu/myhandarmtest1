import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from perception_hamer.src.teleop_pose_packet import CameraViewBiasCorrection


class CameraViewBiasTest(unittest.TestCase):
    def setUp(self):
        self.reference = dict(sequence=123, stamp=1., wrist_position_m=[.04,.1,.9],
                              palm_rotation_row_major=np.eye(3).reshape(-1).tolist())
        self.packet = dict(valid=True, hand_is_right=False,
                           wrist_position_m=[.04,.1,.9], confidence=[.8]*6,
                           palm_rotation_row_major=np.eye(3).reshape(-1).tolist())

    def test_c_and_five_degree_rotations_at_c_keep_every_axis(self):
        correction = CameraViewBiasCorrection(1.188)
        original_reference = copy.deepcopy(self.reference)
        for axis in np.eye(3):
            rotation = Rotation.from_rotvec(axis*np.deg2rad(5)).as_matrix()
            packet = dict(self.packet, palm_rotation_row_major=rotation.reshape(-1).tolist())
            result = correction.apply(packet,self.reference)
            np.testing.assert_allclose(result['palm_rotation_row_major'],rotation.reshape(-1),atol=1e-12)
            self.assertEqual(result['wrist_position_m'],self.packet['wrist_position_m'])
        self.assertEqual(self.reference,original_reference)

    def test_invalid_orientation_keeps_last_corrected_pose(self):
        correction = CameraViewBiasCorrection(1.188)
        moved = dict(self.packet,wrist_position_m=[.3,.1,.9])
        valid = correction.apply(moved,self.reference)
        held = dict(moved,orientation_held=True,wrist_position_m=[.4,.1,.9],
                    palm_rotation_row_major=Rotation.from_rotvec([.3,0,0]).as_matrix().reshape(-1).tolist())
        result = correction.apply(held,self.reference)
        self.assertEqual(result['palm_rotation_row_major'],valid['palm_rotation_row_major'])
        self.assertEqual(result['wrist_position_m'],held['wrist_position_m'])

    def test_new_c_removes_previous_correction_and_other_hand_is_unchanged(self):
        correction = CameraViewBiasCorrection(1.188)
        moved = dict(self.packet,wrist_position_m=[.3,.1,.9])
        correction.apply(moved,self.reference)
        reference = dict(self.reference,sequence=456,wrist_position_m=moved['wrist_position_m'])
        result = correction.apply(moved,reference)
        np.testing.assert_allclose(result['palm_rotation_row_major'],moved['palm_rotation_row_major'])
        other = dict(moved,hand_is_right=True)
        self.assertEqual(correction.apply(other,self.reference),other)
        self.assertEqual(CameraViewBiasCorrection().apply(moved,self.reference),moved)


if __name__ == '__main__':
    unittest.main()
