"""Independent anatomical hand, camera gate, and URDF kinematics regression checks."""
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parents[1]
ROBOT = PROJECT / "reproduction/fr3/fr3_allegro_sim"
sys.path[:0] = [str(PACKAGE/"scripts"), str(PROJECT), str(ROBOT/"scripts")]

from finger_stream import BASELINE, FRAME, FingerStream
from motion_demo import OPEN
from perception_hamer.src.teleop_control_gate import TeleopControlGate
from perception_hamer.src.teleop_pose_packet import wrist_local_landmarks
from retarget import AllegroModel, HUMAN, Retargeter, matrices, unit


def human_hand(bend=(0, 0, 0, 0), spread=0.0, pinch=None, pinch_site=3):
    """Synthetic human proportions and axes, independent of the robot's forward kinematics."""
    out = np.zeros((21, 3))
    roots = ([-.020, .068], [0, .075], [.020, .068], [.038, .059])
    for f, (lateral, height) in enumerate(roots):
        chain = np.arange(4) + 5 + 4*f
        out[chain[0]] = [0., lateral, height]
        azimuth = (f-1) * (.065 + spread)
        angles = .075 + np.array([.93, 1.99, 2.80]) * (bend[f] if f < 3 else 0)
        for j, (angle, length) in enumerate(zip(angles, [.042, .026, .020])):
            out[chain[j+1]] = out[chain[j]] + length*np.array([
                np.sin(angle), np.cos(angle)*np.sin(azimuth), np.cos(angle)*np.cos(azimuth)])
    out[1] = [0, -.025, .025]
    lengths = [.039, .033, .024]
    for i, length in enumerate(lengths):
        d = unit(np.array([.28+(i+1)*.16*bend[3], -.80+(i+1)*.55*bend[3], .64]))
        out[i+2] = out[i+1] + length*d
    if pinch is not None:
        # Solve a fixed-length three-bone human thumb toward the chosen human pad.
        dest = out[HUMAN[pinch, pinch_site]] + [-.003, -.005, -.004]
        thumb = out[1:5].copy()
        root = thumb[0].copy()
        for _ in range(70):
            thumb[-1] = dest
            for j in (2, 1, 0):
                thumb[j] = thumb[j+1] + lengths[j]*unit(thumb[j]-thumb[j+1])
            thumb[0] = root
            for j in (0, 1, 2):
                thumb[j+1] = thumb[j] + lengths[j]*unit(thumb[j+1]-thumb[j])
        out[1:5] = thumb
    return out


def load_robot():
    description = (ROBOT/"urdf/fr3_allegro_left.urdf").read_text()
    names = yaml.safe_load((ROBOT/"config/controllers.yaml").read_text())["hand_controller"]["joints"]
    return AllegroModel(description, names)


def settle(mapping, hand, count=36):
    q = np.array(OPEN)
    for _ in range(count):
        last = q
        q = mapping.step(hand, last, .08)
        if np.any(np.abs(q-last) > mapping.model.speed*.08 + 1e-9):
            raise AssertionError("joint slew limit exceeded")
        if np.any(q < mapping.model.lower) or np.any(q > mapping.model.upper):
            raise AssertionError("joint position limit exceeded")
    return q


def camera_packet(seq, capture, hand):
    return dict(schema="handarm_hamer_pose_v1", session_id="test", sequence=seq, stamp=1000.+capture,
                frame_id="camera_color_optical_frame", position_source="D455_WRIST_RING_CENTER_FROM_SAME_PIXEL_SURFACE_OFFSET",
                wrist_pose_baseline=BASELINE, valid=True, hand_identity_present=True,
                hand_is_right=False, presence_generation=3, active_hand_generation=4,
                wrist_position_m=[0., 0., .50], palm_rotation_row_major=np.eye(3).ravel().tolist(),
                confidence=[1.]*6, orientation_channel_valid=True, orientation_held=False,
                capture_monotonic_s=capture, timing=dict(contract_version=1, capture_sequence=seq+40, dropped_capture_frames=0,
                                                         inference_call_s=.025, model_inference_s=.021, postprocess_s=.007, capture_to_publish_s=.04,
                                                         inference_executed=True),
                finger_observation=dict(contract_version=1, feature_definition="mano_openpose_chain_total_bend_over_pi_v1",
                                        valid=True, flexion=[0.1]*5, confidence=.95),
                hand_landmarks=dict(frame=FRAME, points=hand.tolist(), confidence=.95))


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_robot()
        cls.reference = human_hand()
        cls.mapping = Retargeter(cls.model, OPEN, cls.reference)

    def test_urdf_jacobians(self):
        q = np.linspace(.1, .8, 16)
        from retarget import ROBOT_MIDDLE
        pos, dirs, jac, djac, middle_jac = self.model.kinematics(q, with_middle=True)
        for j in range(16):
            changed = q.copy()
            changed[j] += 1e-7
            p, d, _, _ = self.model.kinematics(changed)
            f, col = divmod(j, 4)
            np.testing.assert_allclose((p[f, 4]-pos[f, 4])/1e-7, jac[f, :, col], atol=2e-7)
            np.testing.assert_allclose((d[f]-dirs[f])/1e-7, djac[f, :, :, col], atol=2e-7)
            center = ROBOT_MIDDLE[f]
            np.testing.assert_allclose((p[f, center]-pos[f, center])/1e-7, middle_jac[f, :, col], atol=2e-7)

    def test_same_hand_rigid_motion_and_pinky(self):
        hand = human_hand([.8, .4, .3, .7])
        rotation = matrices(Rotation.from_rotvec([1.3, -.7, .4]))
        translation = np.array([-.1, .25, .7])
        local = wrist_local_landmarks(hand @ rotation.T+translation, translation, rotation)
        np.testing.assert_allclose(local, hand, atol=1e-12)
        q = self.mapping.step(hand, OPEN, .08)
        np.testing.assert_allclose(q, self.mapping.step(local, OPEN, .08), atol=1e-8)
        hand[18:21] += [.01, .02, -.02]
        np.testing.assert_allclose(q, self.mapping.step(hand, OPEN, .08), atol=1e-9)
        np.testing.assert_allclose(self.mapping.step(self.reference, OPEN, .08), OPEN, atol=1e-8)

    def test_single_digits_and_power_grasp(self):
        for digit in range(4):
            bend = np.eye(4)[digit]
            q = settle(self.mapping, human_hand(bend))
            excursions = np.abs(q-np.asarray(OPEN)).reshape(4, 4).max(axis=1)
            self.assertGreater(excursions[digit], .22)
            self.assertLess(np.delete(excursions, digit).max(), .035)
        q = settle(self.mapping, human_hand([1, 1, 1, 1]))
        self.assertTrue(np.all(q[:12].reshape(3, 4)[:, 1:].mean(axis=1) > .5))
        self.assertGreater(np.max(np.abs(q[12:] - np.array(OPEN[12:]))), .22)

    def test_spread(self):
        # Allegro's axial joints chiefly change lateral pad position when the fingers are flexed.
        a = settle(self.mapping, human_hand([.3, .3, .3, 0], spread=-.1))
        b = settle(self.mapping, human_hand([.3, .3, .3, 0], spread=.22))
        tips_a, tips_b = [self.model.kinematics(q)[0][:, 4] for q in (a, b)]
        self.assertGreater((tips_b[2, 1] - tips_b[0, 1]) - (tips_a[2, 1] - tips_a[0, 1]), .012)

    def test_three_oppositions(self):
        for digit in range(3):
            bend = np.eye(4)[digit]
            hand = human_hand(bend, pinch=digit)
            q = settle(self.mapping, hand)
            tips = self.model.kinematics(q)[0][:, 4]
            span = np.linalg.norm(tips[digit]-tips[3])
            self.assertGreater(self.mapping.targets(hand)[2][digit], .98)
            self.assertGreater(span, .023)
            self.assertLess(span, .032, (digit, span))

    def test_gate_recalibration_drop_reorder_timeout(self):
        gate = TeleopControlGate(require_hand_landmarks=True)
        stream = FingerStream(self.model, OPEN)

        def ingest(packet, now=None):
            t = packet["capture_monotonic_s"]
            return stream.accept(packet, t+.04 if now is None else now, t+1, OPEN)

        gate.request_confirmation(after_sequence=0, now=0.)
        for i in range(1, 6):
            raw = camera_packet(i, i*.06, self.reference)
            if i == 1:
                raw.pop("hand_landmarks")
            gate.confirm_packet(raw, lambda: True, now=i*.06)
            packet = gate.decorate(raw)
            if i < 5:
                self.assertIsNone(ingest(packet))
        self.assertEqual(5, packet["control_reference_pose"]["sequence"])
        self.assertEqual(45, packet["control_reference_pose"]["capture_sequence"])
        self.assertAlmostEqual(.30, packet["control_reference_pose"]["capture_monotonic_s"])
        # Drop the first enabled datagram. A later moving frame must still use sequence 5.
        moving = human_hand([1, 0, 0, 0])
        packet = gate.decorate(camera_packet(6, .36, moving))
        self.assertIsNotNone(ingest(packet))
        self.assertGreater(np.max(np.abs(stream.q-OPEN)), .01)
        self.assertIsNone(ingest(packet))
        for i, changed in enumerate([dict(orientation_held=True), dict(hand_is_right=True),
                                     dict(control_reference_token="incorrect"), dict(valid=False),
                                     dict(hand_landmarks=None)], start=7):
            p = gate.decorate(camera_packet(i, .06*i, moving))
            p.update(changed)
            last = stream.q.copy()
            self.assertIsNone(ingest(p))
            np.testing.assert_equal(last, stream.q)
        self.assertIsNone(ingest(gate.decorate(camera_packet(12, .72, moving)), now=1.1))
        gate.observe_identity(4, 5, False)
        self.assertIsNone(ingest(gate.decorate(camera_packet(13, .78, moving))))
        self.assertIsNone(ingest(packet))  # Delayed datagram from the disarmed identity.
        gate.request_confirmation(13, now=.78)
        for seq in range(14, 18):
            p = camera_packet(seq, seq*.06, self.reference)
            p["presence_generation"], p["active_hand_generation"] = 4, 5
            gate.confirm_packet(p, lambda: True, now=seq*.06)
        new = gate.decorate(p)
        self.assertEqual(new["control_reference_epoch"], 2)
        self.assertIsNotNone(ingest(new))
        np.testing.assert_allclose(stream.q, OPEN, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
