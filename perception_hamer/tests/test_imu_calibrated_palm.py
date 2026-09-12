from pathlib import Path
import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from perception_hamer.src.imu_calibrated_palm import ImuCalibratedPalmCorrector, measure_palm_surface
from perception_hamer.src.teleop_control_gate import TeleopControlGate

CONFIG = Path(__file__).resolve().parents[1] / 'config/imu_guided_palm_20260912.json'


def packet(sequence=1, rotation=None, surface=None, position=None):
    rotation = np.eye(3) if rotation is None else rotation
    return dict(sequence=sequence, stamp=sequence*.1, valid=True,
        presence_generation=1, active_hand_generation=1, hand_is_right=False,
        wrist_position_m=[0., 0., .8] if position is None else position,
        palm_rotation_row_major=rotation.reshape(-1).tolist(), confidence=[.8]*6,
        finger_observation={'valid': True, 'flexion': [.1]*5},
        palm_surface_observation={'valid': True, 'rotation_row_major':
            (rotation if surface is None else surface).reshape(-1).tolist()})


class ImuCalibratedPalmTest(unittest.TestCase):
    def test_disabled_does_not_require_calibration_or_touch_packet(self):
        corrector = ImuCalibratedPalmCorrector('/missing/file', enabled=False)
        original = packet()
        self.assertEqual(corrector.apply(original, None), original)

    def test_c_and_repeated_c_use_same_pose_and_geometry_without_hold(self):
        corrector = ImuCalibratedPalmCorrector(CONFIG, enabled=True)
        gate = TeleopControlGate()
        for base in (0., 28.):
            gate.request_confirmation(0, now=0)
            raw = R.from_euler('xyz', [base, 11., -7.], degrees=True).as_matrix()
            surface = R.from_euler('xyz', [8., -base, 22.], degrees=True).as_matrix()
            for seq in range(1, 5):
                current = packet(seq, raw, surface, [.25, .1, .8])
                output = corrector.apply(current, gate.reference_pose)
                gate.confirm_packet(output, lambda: True, now=seq*.1)
            for seq in range(5, 10):
                output = corrector.apply(packet(seq, raw, surface, [.25, .1, .8]), gate.reference_pose)
                np.testing.assert_allclose(output['palm_rotation_row_major'], raw.reshape(-1), atol=1e-12)
            small = R.from_euler('y', 5., degrees=True).as_matrix()
            current = packet(10, small @ raw, small @ surface, [.25, .1, .8])
            output = corrector.apply(current, gate.reference_pose)
            np.testing.assert_allclose(output['palm_rotation_row_major'], current['palm_rotation_row_major'], atol=1e-12)
            self.assertNotIn('orientation_held', output)

    def test_geometry_loss_does_not_switch_to_uncorrected_rotation(self):
        corrector = ImuCalibratedPalmCorrector(CONFIG, enabled=True)
        reference = packet()
        raw = R.from_euler('x', 40., degrees=True).as_matrix()
        for seq in range(2, 50):
            current = packet(seq, raw, np.eye(3), [.4, 0., .8])
            output = corrector.apply(current, reference)
        previous = R.from_matrix(np.array(output['palm_rotation_row_major']).reshape(3, 3))
        current['sequence'] += 1
        current['stamp'] += .1
        current['palm_surface_observation'] = {'valid': False}
        original = copy.deepcopy(current)
        output = corrector.apply(current, reference)
        after = R.from_matrix(np.array(output['palm_rotation_row_major']).reshape(3, 3))
        self.assertLess(np.rad2deg((after * previous.inv()).magnitude()), .001)
        self.assertGreater(np.rad2deg((after * R.from_matrix(raw).inv()).magnitude()), 10.)
        for field in ('wrist_position_m', 'confidence', 'finger_observation'):
            self.assertEqual(output[field], original[field])
        self.assertEqual(current, original)
        current['orientation_held'] = True
        current['palm_rotation_row_major'] = np.eye(3).reshape(-1).tolist()
        held = corrector.apply(current, reference)
        self.assertEqual(held['palm_rotation_row_major'], output['palm_rotation_row_major'])

    def test_depth_plane_recovers_known_surface_normal(self):
        k = dict(width=640, height=480, fx=600., fy=600., ppx=320., ppy=240., coeffs=[0.]*5)
        y, x = np.indices((480, 640))
        normal = np.array([.2, -.3, 1.]); normal /= np.linalg.norm(normal)
        depth = np.round(.8/((x-320.)/600.*normal[0]+(y-240.)/600.*normal[1]+normal[2])/.001).astype(np.uint16)
        surface = measure_palm_surface(depth, .001, k, [[320, 330], [290, 250], [310, 245], [330, 245], [350, 250]])
        self.assertTrue(surface['valid'])
        self.assertLess(np.rad2deg(np.arccos(np.clip(np.dot(surface['normal'], normal), -1., 1.))), .3)
        frame = np.array(surface['rotation_row_major']).reshape(3, 3)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
