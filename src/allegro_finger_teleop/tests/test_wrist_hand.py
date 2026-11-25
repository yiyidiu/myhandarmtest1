"""Regressions for full finger articulation, stable opposition and the moving palm TCP."""
import unittest
from xml.etree import ElementTree
import numpy as np
import yaml

from test_retarget import (OPEN, ROBOT, FingerStream, HUMAN, Retargeter, Rotation, TeleopControlGate,
                           camera_packet, human_hand, load_robot, matrices, settle)
from retarget import ThumbFilter
from wrist_arm import ArmModel, WristArm, rotation_vector
from perception_hamer.src.active_hand_selector import AutomaticActiveHandSelector
from perception_hamer.src.hand_detection_gate import ContinuousHandPresenceGate
from perception_hamer.src.teleop_pose_packet import build_invalid_teleop_packet


def long_hand(digit, angles):
    """Three independent human flexion angles on one finger, preserving phalanx lengths."""
    h = human_hand()
    chain = HUMAN[digit]
    azimuth = .065*(digit-1)
    cumulative = .075 + np.cumsum(angles)
    for i, (length, angle) in enumerate(zip([.042,.026,.020], cumulative)):
        h[chain[i+1]] = h[chain[i]] + length*np.array([np.sin(angle),
                                    np.cos(angle)*np.sin(azimuth), np.cos(angle)*np.cos(azimuth)])
    return h


def thumb_sweep(angle):
    h = human_hand()
    rot = matrices(Rotation.from_rotvec([-angle, 0., 0.]))
    h[2:5] = (h[2:5]-h[1]) @ rot.T + h[1]
    return h


def arm_model():
    cfg = yaml.safe_load((ROBOT/"config/controllers.yaml").read_text())
    return ArmModel((ROBOT/"urdf/fr3_allegro_left.urdf").read_text(), cfg["arm_controller"]["joints"])


_ready = ElementTree.parse(ROBOT/"launch/bringup.launch").find("./arg[@name='ready_degrees']")
READY = np.deg2rad(np.fromstring(_ready.get("default"), sep=" "))
HPOS = np.array([.04,-.07,.56])
HROT = matrices(Rotation.from_rotvec([1.1,-.8,.35]))
# Optical +Z is depth. Both reference hands have their middle finger pointing up the image.
PALM_TO_CAMERA = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
BACK_TO_CAMERA = PALM_TO_CAMERA @ np.diag([-1., -1., 1.])


class FingerAccuracy(unittest.TestCase):
    def test_isolated_hinges_and_deep_curl(self):
        model = load_robot()
        for digit in range(3):
            mapping = Retargeter(model, OPEN, human_hand())
            for angles in ([0., 1.25, 0.], [0., 0., 1.05], [1.40, 1.53, 1.36]):
                q = settle(mapping, long_hand(digit, angles))
                actual = (q-OPEN).reshape(4,4)[digit,1:]
                np.testing.assert_allclose(actual, angles, atol=.09)

    def test_thumb_sweep_reversal_and_quiet_hold(self):
        model = load_robot()
        mapping = Retargeter(model, OPEN, human_hand())
        positive = settle(mapping, thumb_sweep(.40))
        negative = settle(mapping, thumb_sweep(-.18))
        self.assertGreater(positive[12]-negative[12], .27)
        self.assertLess(abs(positive[15]-OPEN[15]), .08)
        smoother = ThumbFilter(human_hand())
        rng = np.random.RandomState(11)
        q = np.array(OPEN)
        hist = []
        for _ in range(70):
            h = human_hand()
            h[1:5] += rng.normal(0, .00075, (4,3))
            q = mapping.step(smoother.update(h, .05), q, .05)
            hist.append(q[12:].copy())
        self.assertLess(np.rad2deg(np.std(hist[15:],axis=0)).max(), 2.5)


class WristTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = arm_model()
        cls.basis = Retargeter(load_robot(), OPEN, human_hand()).basis

    def test_tcp_fk_and_geometric_jacobian(self):
        m = self.model
        p, r, j = m.fk(READY)
        np.testing.assert_allclose(p, [.430,0.,.700], atol=.00015)
        np.testing.assert_allclose(r[:,0], [0,0,-1], atol=1e-7)
        np.testing.assert_allclose(r[:,2], [1,0,0], atol=1e-7)
        # The angular Jacobian columns give the J5 shaft and J7 flange axes.
        parallel = np.abs(j[3:,4] @ j[3:,6])
        self.assertAlmostEqual(np.degrees(np.arccos(np.clip(parallel, 0., 1.))), 44., places=4)
        for i in range(7):
            q = READY.copy()
            q[i] += 1e-6
            p1,r1,_ = m.fk(q)
            np.testing.assert_allclose(np.r_[(p1-p)/1e-6, rotation_vector(r1@r.T)/1e-6], j[:,i], atol=2e-6)

    def test_capture_time_arm_reference(self):
        at_c = READY + [.10, -.04, 0., 0., .02, 0., 0.]
        mapping = WristArm(self.model, dict(wrist_position_m=HPOS, palm_rotation_row_major=HROT),
                           self.basis, READY, at_reference=at_c)
        expected_p, expected_r, _ = self.model.fk(at_c)
        p, r = mapping.target(HPOS, HROT)
        np.testing.assert_allclose(p, expected_p, atol=1e-10)
        np.testing.assert_allclose(r, expected_r, atol=1e-10)
        np.testing.assert_allclose(mapping.command, READY)

    def test_optical_thumb_edge_toward_camera_tips_palm_toward_base_minus_y(self):
        # Rotate the entire human in optical space, independently of the wrist mapper.
        # Give the simulated wrist ring a different basis from the anatomical hand.
        ring_from_hand = matrices(Rotation.from_rotvec([-.55, .26, .38]))
        points = human_hand() @ ring_from_hand.T
        hand_basis = Retargeter(load_robot(), OPEN, points).basis
        thumb_minus_little = points[1] - points[17]
        for hand_to_camera in (PALM_TO_CAMERA, BACK_TO_CAMERA):
            for oblique in (0., .39):
                cam = matrices(Rotation.from_rotvec([0., oblique, 0.])) @ hand_to_camera @ ring_from_hand.T
                ref = dict(wrist_position_m=HPOS, palm_rotation_row_major=cam)
                solver = WristArm(self.model, ref, hand_basis, READY)
                for yaw in np.deg2rad([-25, -10, 10, 25]):
                    turned = matrices(Rotation.from_rotvec([0., yaw, 0.])) @ cam
                    dz = ((turned-cam) @ thumb_minus_little)[2]
                    p, rot = solver.target(HPOS, turned)
                    normal = rot[:, 0]
                    self.assertAlmostEqual(normal[1], np.sign(dz)*np.sin(abs(yaw)), places=7)
                    self.assertAlmostEqual(normal[2], -np.cos(yaw), places=7)
                    np.testing.assert_allclose(p, solver.rpos)
                    np.testing.assert_allclose(rot[:, 2], [1., 0., 0.], atol=1e-8)

    def test_tilt_convention_for_other_axes_half_turn_and_new_robot_reference(self):
        ref = dict(wrist_position_m=HPOS, palm_rotation_row_major=PALM_TO_CAMERA)
        solver = WristArm(self.model, ref, self.basis, READY)
        mixed_axis = np.array([.36, -.48, .80])
        turns = [[.4, 0., 0.], [0., -.42, 0.], [.24, -.31, .45]]
        turns.extend(mixed_axis*np.deg2rad(angle) for angle in (179.99, 180., 180.01))
        for axis_angle in turns:
            observed = PALM_TO_CAMERA @ matrices(Rotation.from_rotvec(axis_angle))
            p, r = solver.target(HPOS+[.03, -.01, .02], observed)
            expected = matrices(Rotation.from_rotvec(np.array(axis_angle)*[1., 1., -1.]))
            np.testing.assert_allclose(r, solver.rrot @ expected, atol=1e-10)
            np.testing.assert_allclose(p, solver.rpos+[.01, -.03, .02], atol=1e-10)
        # With the palm only one degree away from edge-on on either side, depth
        # cannot select a reliable sign. Both C poses must keep the same convention.
        twist = matrices(Rotation.from_rotvec([0., 0., .35]))
        for angle in (89., 91.):
            edge = PALM_TO_CAMERA @ matrices(Rotation.from_rotvec([0., 0., np.deg2rad(angle)]))
            edge_solver = WristArm(self.model, dict(ref, palm_rotation_row_major=edge), self.basis, READY)
            _, r = edge_solver.target(HPOS, edge @ twist)
            np.testing.assert_allclose(r, edge_solver.rrot @ twist, atol=1e-10)
        turned_arm = READY.copy()
        turned_arm[-1] += np.pi
        solver = WristArm(self.model, ref, self.basis, READY, at_reference=turned_arm)
        self.assertLess(solver.rrot[1,1], -.99)
        optical_turn = matrices(Rotation.from_rotvec([0., -.35, 0.]))
        _, r = solver.target(HPOS, optical_turn @ PALM_TO_CAMERA)
        self.assertLess(r[1,0]-solver.rrot[1,0], -.30)

    def test_six_directions_and_range(self):
        m = self.model
        mapping = WristArm(m, dict(wrist_position_m=HPOS, palm_rotation_row_major=HROT.ravel()), self.basis, READY)
        for d in np.eye(3):
            for sign in (-1., 1.):
                camera_delta = sign*.035*d
                local_turn = matrices(Rotation.from_rotvec(sign*.17*d))
                observed = HPOS+camera_delta, HROT@self.basis@local_turn@self.basis.T
                expected_p = mapping.rpos + [-camera_delta[1],-camera_delta[0],camera_delta[2]]
                expected_r = mapping.rrot@local_turn
                q = mapping.command
                for _ in range(28):
                    q, goal = mapping.step(observed, q, .06)
                p,r,_ = m.fk(q)
                np.testing.assert_allclose(goal[0], expected_p, atol=1e-10)
                self.assertLess(np.linalg.norm(p-expected_p), .0003)
                self.assertLess(np.linalg.norm(rotation_vector(r@expected_r.T)), .002)
        q = mapping.command
        for _ in range(80):
            q2,_ = mapping.step((HPOS+[1.,.7,.55], HROT), q, .03)
            self.assertTrue(np.all((m.lower <= q2) & (q2 <= m.upper)))
            self.assertTrue(np.all(np.abs(q2-q) <= .03*m.speed+1e-8))
            q = q2

    def test_gating_channels_and_new_zero(self):
        hand = human_hand()
        g = TeleopControlGate(require_hand_landmarks=True)
        s = FingerStream(load_robot(), OPEN, self.model)
        g.request_confirmation(0, now=1.)
        def raw(i):
            p = camera_packet(i, 1.+i*.06, hand)
            p.update(wrist_position_m=HPOS.tolist(), palm_rotation_row_major=HROT.ravel().tolist())
            return p
        for i in range(1,5):
            p=raw(i)
            g.confirm_packet(p, lambda: True, now=p["capture_monotonic_s"])
        def send(p, measured=READY, delay=.02):
            t=p["capture_monotonic_s"]
            return s.accept_frame(g.decorate(p), t+delay, t, OPEN, measured)
        result=send(p)
        np.testing.assert_allclose(result.arm, READY, atol=1e-9)
        p=raw(5)
        p["wrist_position_m"] = (HPOS+[0.,-.04,0.]).tolist()
        p.pop("hand_landmarks")
        result=send(p)
        self.assertIsNone(result.hand)
        self.assertGreater(np.linalg.norm(result.arm-READY), .01)
        p=raw(6)
        p["palm_rotation_row_major"] = [0.]*9
        result=send(p)
        self.assertIsNone(result.arm)
        self.assertIsNotNone(result.hand)
        self.assertIsNone(send(raw(7), delay=.6))
        g.request_confirmation(7, now=1.42)
        self.assertIsNone(send(raw(8)))
        changed = READY + [.07,.04,0,0,0,0,0]
        for i in range(9,13):
            p=raw(i)
            p["wrist_position_m"] = (HPOS+[.02,.04,.01]).tolist()
            g.confirm_packet(p, lambda: True, now=p["capture_monotonic_s"])
        result=send(p, changed)
        np.testing.assert_allclose(result.arm, changed, atol=1e-9)

        p = raw(13)
        p.update(orientation_held=True, orientation_channel_valid=False, confidence=[1.]*3 + [0.]*3,
                 palm_rotation_row_major=np.eye(3).ravel().tolist())
        result = send(p, changed)
        self.assertIsNone(result.hand)
        np.testing.assert_allclose(result.palm_target[1], self.model.fk(changed)[1], atol=1e-9)
        self.assertGreater(np.linalg.norm(result.palm_target[0]-self.model.fk(changed)[0]), .03)
        p = raw(14)
        p.pop("position_source")
        self.assertIsNone(send(p, changed).arm)

    def test_fixed_left_hand_loss_return_and_explicit_disarm(self):
        selector = AutomaticActiveHandSelector(fixed_is_right=False)
        presence = ContinuousHandPresenceGate(required_frames=2, minimum_iou=.25, timeout_s=.30)
        gate = TeleopControlGate(require_hand_landmarks=True, resume_fixed_hand_after_loss=True)
        stream = FingerStream(load_robot(), OPEN, self.model)
        gate.request_confirmation(0, now=1.)
        left = dict(valid=True, is_right=False, bbox=[50, 40, 140, 190], confidence=.9)
        missing = dict(valid=False, reason="no_hand_detected")

        def frame(i, det, shift=0., measured=READY):
            t = 1.+.06*i
            selected = selector.select(det, t)
            state = presence.observe(selected, t)
            generation, active = state["generation"], selector.active_hand_generation
            if state["valid"]:
                gate.observe_identity(generation, active, False)
                p = camera_packet(i, t, human_hand())
                p.update(presence_generation=generation, active_hand_generation=active,
                         wrist_position_m=(HPOS+[0.,-shift,0.]).tolist())
            else:
                p = build_invalid_teleop_packet("test", i, 1000.+t, state["reason"], generation, active)
                p.update(capture_monotonic_s=t, timing=dict(capture_sequence=i+40,
                         inference_executed=False, capture_to_publish_s=.02))
            gate.confirm_packet(p, lambda: True, now=t)
            message = gate.decorate(p)
            return message, stream.accept_frame(message,t+.02,t,OPEN,measured)

        for i in range(1,7):
            original, command = frame(i, left)
        self.assertIsNotNone(command.arm)
        held_token = stream.token
        for i in range(7,21):
            packet, command = frame(i, missing)
            self.assertFalse(packet["hand_identity_present"])
            self.assertFalse(packet["valid"])
            self.assertTrue(packet["control_enabled"])
            self.assertIsNone(command)
            self.assertEqual(stream.token, held_token)
        # A remote right hand does not take over the fixed-left detector.
        remote = dict(left,is_right=True,bbox=[320,40,420,180])
        self.assertIsNone(frame(21, remote)[1])
        measured = READY + [.025, -.012, 0., 0., 0., 0., 0.]
        for i in range(22,28):
            packet, command = frame(i, left, .065, measured)
            if command is not None:
                break
        self.assertIsNotNone(command)
        self.assertEqual(packet["control_reference_pose"], original["control_reference_pose"])
        self.assertEqual(packet["control_reference_token"], held_token)
        self.assertGreater(packet["presence_generation"], original["presence_generation"])
        self.assertEqual(packet["active_hand_generation"], original["active_hand_generation"])
        self.assertIsNotNone(command.hand)
        self.assertLessEqual(command.dt,1/30)
        self.assertTrue(np.all(np.abs(command.arm-measured) <= self.model.speed*command.dt + 1e-9))
        np.testing.assert_allclose(command.palm_target[0], self.model.fk(READY)[0]+[.065,0.,0.], atol=1e-8)
        gate.disable("OPERATOR_STOPPED")
        self.assertIsNone(frame(28, left)[1])
        self.assertFalse(gate.enabled)


if __name__ == "__main__":
    unittest.main()
