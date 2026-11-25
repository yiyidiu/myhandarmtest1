"""Separate finger articulation from wrist motion through the live RGB-D and C protocol."""
import unittest
import numpy as np

from test_retarget import OPEN, FingerStream, TeleopControlGate, camera_packet, human_hand, load_robot, matrices, Rotation
from test_wrist_hand import PALM_TO_CAMERA, READY, arm_model
from test_hand_isolation import image_observation
from test_fingertip_graph import folded_fist, opposition
from perception_hamer.src.teleop_pose_packet import observed_palm_support
from perception_hamer.src.finger_observation import finger_shape_signature
from wrist_arm import rotation_vector
from wrist_motion import PalmMotionSupport, palm_displacement

POSITION = np.array([.01, -.01, .65])
INTRINSICS = dict(fx=606., fy=606., ppx=319., ppy=249., coeffs=[0.]*5, distortion_model='none')


def measured_hand(shape, rot, pos):
    """Project human landmarks and ray-cast a moving palm plane into D455 depth patches."""
    obs = image_observation(shape, rot, pos)
    pixels = obs['image_landmarks_px']
    palm = pixels[[0,5,9,13,17]]
    sites = np.vstack([palm, .55*palm[0]+.45*palm[1:]])
    raw = np.zeros((480,640), np.uint16)
    normal = rot[:,0]
    for u,v in np.rint(sites).astype(int):
        x,y = np.meshgrid(np.arange(u-2,u+3),np.arange(v-2,v+3))
        rays = np.stack([(x-319.)/606., (y-249.)/606., np.ones_like(x)],axis=-1)
        raw[y,x] = np.rint(1000*(normal@pos)/(rays@normal)).astype(np.uint16)
    support = observed_palm_support(pixels,raw,.001,INTRINSICS,pos[2])
    obs['shape_signatures'] = np.stack((finger_shape_signature(shape), finger_shape_signature(shape@rot.T+pos)))
    return {k:v.tolist() for k,v in obs.items()}, support


def observed_packet(seq, now, shape, rot, pos, bias_m, bias_deg, session):
    observation, support = measured_hand(shape,rot,pos)
    model_rot = matrices(Rotation.from_rotvec(np.deg2rad(bias_deg))) @ rot
    packet = camera_packet(seq,now,shape)
    packet.update(session_id=session,wrist_position_m=(pos+bias_m).tolist(),
                  palm_rotation_row_major=model_rot.ravel().tolist(),
                  measured_wrist_rotation_row_major=model_rot.ravel().tolist(),wrist_motion_support=support)
    packet['hand_landmarks'].update(observation)
    return packet


class CrossChannelMotion(unittest.TestCase):
    def setUp(self):
        self.gate = TeleopControlGate(require_hand_landmarks=True,resume_fixed_hand_after_loss=True)
        self.arm = arm_model()
        self.arm.control_point = 'wrist'
        self.stream = FingerStream(load_robot(),OPEN,self.arm)
        self.gate.request_confirmation(0,now=1.)
        self.seq = 0
        self.q, self.a = np.array(OPEN), READY.copy()
        self.neutral = human_hand()
        for _ in range(12):
            self.send(self.neutral)
        self.assertTrue(self.gate.enabled)

    def send(self, hand, position=POSITION, rot=PALM_TO_CAMERA, bias_mm=0., bias_deg=0., lost=False):
        self.seq += 1
        t = 1. + self.seq*.05
        msg = observed_packet(self.seq,t,hand,rot,position,np.array([.8,0.,1.])*bias_mm*.001,
                              [0.,bias_deg,0.],'test-palm')
        if lost:
            msg['wrist_motion_support'] = None
        self.gate.confirm_packet(msg,lambda:True,now=t)
        cmd = self.stream.accept_frame(self.gate.decorate(msg),t,t,self.q,self.a,READY)
        if cmd is not None:
            if cmd.arm is not None:self.a = cmd.arm
            if cmd.hand is not None:self.q = cmd.hand
        return cmd

    def test_flex_pinch_fist_keep_both_wrist_channels_fixed(self):
        largest_hand = 0.
        for hand in (human_hand([.82,0,0,0]),opposition((2,)), folded_fist(), self.neutral):
            for i in range(16):
                self.send(hand,bias_mm=17*np.sin(.2*i+.2),bias_deg=15*np.sin(.3*i+.3))
                largest_hand = max(largest_hand,np.max(np.abs(self.q-OPEN)))
                np.testing.assert_allclose(self.a,READY,atol=2e-10)
        self.assertGreater(largest_hand,.65)
        # In the same test, true slow wrist motion must survive simultaneous closure.
        for s in np.linspace(0.,1.,61):
            rot=matrices(Rotation.from_euler('y',18*s,degrees=True))@PALM_TO_CAMERA
            self.send(human_hand([s,.72*s,0,0]),POSITION+[0.,0.,.040*s],rot)
        support = self.stream.palm_support
        self.assertLess(np.linalg.norm(support.position-POSITION-[0,0,.040]),.004)
        self.assertLess(np.degrees(np.linalg.norm(rotation_vector(support.rotation@rot.T))),2.5)

    def test_depth_loss_changes_only_fingers_then_recovers(self):
        for _ in range(20):
            self.send(self.neutral,POSITION+[.035,0,0],lost=True)
        self.assertFalse(self.stream.palm_support.available)
        self.send(folded_fist(),bias_mm=18,bias_deg=12,lost=True)
        np.testing.assert_allclose(self.a,READY,atol=2e-10)
        for _ in range(20):
            self.send(self.neutral,POSITION+[.035,0,0])
        self.assertTrue(self.stream.palm_support.available)
        np.testing.assert_allclose(self.stream.palm_support.position,POSITION+[.035,0,0],atol=.002)

    def test_live_mode_without_independent_depth_holds_arm_while_fingers_work(self):
        stream = FingerStream(load_robot(),OPEN,self.arm,require_wrist_support=True)
        for i in range(20):
            self.seq += 1
            t = 1.+self.seq*.05
            pkt = observed_packet(self.seq,t,human_hand([0,1,0,0]),PALM_TO_CAMERA,POSITION,
                                  [.014,-.012,.018],[0.,15.,0.],"test-palm")
            pkt = self.gate.decorate(pkt)
            pkt.pop("wrist_motion_support")
            pkt["control_reference_pose"] = dict(pkt["control_reference_pose"])
            pkt["control_reference_pose"].pop("wrist_motion_support",None)
            cmd = stream.accept_frame(pkt,t,t,self.q,self.a,READY)
            self.assertIsNotNone(cmd)
            self.q, self.a = cmd.hand, cmd.arm
            self.assertFalse(stream.palm_support.available)
            np.testing.assert_allclose(self.a,READY,atol=1.e-10)
        self.assertGreater(np.max(np.abs(self.q[4:8]-OPEN[4:8])),.35)

    def test_stationary_palm_with_depth_noise_and_a_bad_knuckle(self):
        cloud = np.array(measured_hand(self.neutral,PALM_TO_CAMERA,POSITION)[1],dtype=float)
        w = PalmMotionSupport(cloud,POSITION,PALM_TO_CAMERA)
        rng = np.random.RandomState(141)
        for i in range(90):
            noisy = cloud + rng.normal(0,.0004,cloud.shape)
            noisy[1+i%4,2] += .026
            bad_rot = matrices(Rotation.from_euler('y',15*np.sin(i/9.),degrees=True))@PALM_TO_CAMERA
            p,r = w.update(noisy, POSITION+[.018*np.sin(i/6.),.012,0.],bad_rot,1.+.04*i,True)
            np.testing.assert_allclose(p,POSITION,atol=1.e-9)
            np.testing.assert_allclose(r,PALM_TO_CAMERA,atol=1.e-9)

    def test_missing_reference_outlier_and_held_orientation(self):
        c=np.array(measured_hand(self.neutral,PALM_TO_CAMERA,POSITION)[1],dtype=float)
        dr=matrices(Rotation.from_euler('y',20.,degrees=True))
        dp=np.array([.025,.015,-.01])
        moved=(c-POSITION)@dr.T+POSITION+dp
        moved[3] += [.05, .02, .06]
        shift,rot=palm_displacement(c,moved,POSITION)
        np.testing.assert_allclose(shift,dp,atol=1.e-10)
        np.testing.assert_allclose(rot,dr,atol=1.e-10)
        w=PalmMotionSupport(None,POSITION,PALM_TO_CAMERA)
        for i in range(4):w.update(c,POSITION+dp,dr@PALM_TO_CAMERA,.05*i,True)
        np.testing.assert_allclose(w.position,POSITION)
        np.testing.assert_allclose(w.rotation,PALM_TO_CAMERA)
        for i in range(4):w.update(moved,POSITION+2*dp,dr@dr@PALM_TO_CAMERA,.3+.05*i,False)
        np.testing.assert_allclose(w.position,POSITION+dp,atol=1.e-10)
        np.testing.assert_allclose(w.rotation,PALM_TO_CAMERA)
        for i in range(4):w.update(moved,POSITION+2*dp,dr@dr@PALM_TO_CAMERA,.7+.05*i,True)
        np.testing.assert_allclose(w.rotation,dr@PALM_TO_CAMERA,atol=1.e-10)
        for value in (None,[1,2],[[None]*3]*9,[[0,0,.60]]*9):
            w.update(value,POSITION,PALM_TO_CAMERA,1.2,True)
            self.assertFalse(w.available)
            np.testing.assert_allclose(w.position,POSITION+dp,atol=1.e-10)
