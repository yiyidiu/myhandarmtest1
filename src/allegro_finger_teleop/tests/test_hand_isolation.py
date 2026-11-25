"""Wrist-only motion, simultaneous digit articulation and thumb contact selection."""
import unittest
import numpy as np

from test_retarget import HUMAN, OPEN, PACKAGE, Retargeter, human_hand, load_robot, matrices, Rotation, settle
from test_wrist_hand import PALM_TO_CAMERA, READY, arm_model
from hand_shape import HandShape
from wrist_arm import WristArm
from perception_hamer.src.finger_observation import visible_calibration_palm


def project(hand, rotation, position):
    camera = hand @ rotation.T + position
    return 606*camera[:,:2]/camera[:,2:3] + [319.,249.]

def image_observation(hand, rotation, position):
    return dict(image_landmarks_px=project(hand, rotation, position),
                surface_depths_m=(hand @ rotation.T + position)[:, 2])


class HandIsolation(unittest.TestCase):
    def setUp(self):
        self.model = load_robot()
        self.open = human_hand()
        self.position = np.array([0., 0., .65])

    def test_edge_on_identity_requires_visible_acquisition_and_release(self):
        from perception_hamer.src.finger_observation import normalized_thumb_tip_gaps
        face = PALM_TO_CAMERA
        side = matrices(Rotation.from_euler("y", 89, degrees=True)) @ face
        shape = HandShape(self.open, image_observation(self.open, face, self.position), face)
        retarget = Retargeter(self.model, OPEN, self.open)
        def observe(mano, image_hand, rotation, count=6):
            for _ in range(count):
                obs = image_observation(image_hand, rotation, self.position)
                obs["independent_tip_gaps"] = normalized_thumb_tip_gaps(image_hand)
                held = shape.update(mano, obs, rotation)
                retarget.targets(held, .04, shape.observed_tip, shape.separated_tip, shape.edge_on)
        index, middle, ring = [human_hand(np.eye(4)[f] * .85, pinch=f) for f in range(3)]
        observe(middle, index, side)
        self.assertTrue(shape.edge_on)
        self.assertFalse(retarget.contacts)
        self.assertTrue(shape.held_digits[3])
        observe(index, index, side)
        self.assertFalse(retarget.contacts)
        observe(ring, ring, face)
        self.assertFalse(shape.edge_on)
        self.assertEqual(retarget.contacts, ((2, 3),))
        observe(index, ring, side)
        self.assertEqual(retarget.contacts, ((2, 3),))
        observe(self.open, self.open, side)
        self.assertFalse(retarget.contacts)
        self.assertFalse(shape.hidden_partners.any())
        observe(middle, middle, face)
        self.assertEqual(retarget.contacts, ((1, 3),))

    def test_fixed_shape_and_simultaneous_real_articulation(self):
        shape = HandShape(self.open, image_observation(self.open, PALM_TO_CAMERA, self.position), PALM_TO_CAMERA)
        rng = np.random.RandomState(362)
        for deg in [5,20,35,55,75,95,120,80,40,15,0]:
            rot = matrices(Rotation.from_euler('y',deg,degrees=True)) @ PALM_TO_CAMERA
            pos = self.position + [.06*np.sin(np.deg2rad(deg)), .015, .07]
            false_curl = human_hand([.20,.15,.10,.18])
            noisy = false_curl*1.08+[.006,-.001,.018]+rng.normal(0,.00025,(21,3))
            # Independent image retains a rigid, open hand despite the model's false curl.
            clean = shape.update(noisy, image_observation(self.open,rot,pos), rot)
            np.testing.assert_allclose(clean,self.open,atol=1e-8)
        rot = matrices(Rotation.from_euler('y',32,degrees=True)) @ PALM_TO_CAMERA
        pinched = human_hand([0,1,0,0],pinch=1)
        for _ in range(2):
            current = shape.update(pinched,dict(image_landmarks_px=project(pinched,rot,self.position)),rot)
        mapped = Retargeter(self.model,OPEN,self.open)
        q = settle(mapped,current)
        self.assertEqual(mapped.gesture,'thumb_middle_tip')
        self.assertGreater(np.max(np.abs(q[4:8]-OPEN[4:8])),.5)
        np.testing.assert_allclose(current[HUMAN[[0,2]]],self.open[HUMAN[[0,2]]],atol=1e-8)
        # On losing the image, hold the already bent pose, including when the model predicts opening.
        np.testing.assert_equal(shape.update(self.open,dict(image_landmarks_px=None),rot),current)
        np.testing.assert_equal(shape.update(self.open,{},rot),current)
        for deg in [45,60,80,30]:
            rot = matrices(Rotation.from_euler('y',deg,degrees=True)) @ PALM_TO_CAMERA
            recovered = shape.update(pinched,dict(image_landmarks_px=project(pinched,rot,self.position)),rot)
            np.testing.assert_allclose(recovered,current,atol=1e-8)
        fist = human_hand([1,1,1,1])
        for _ in range(2):
            fitted = shape.update(fist,dict(image_landmarks_px=project(fist,rot,self.position)),rot)
        q = settle(mapped,fitted)
        self.assertEqual(mapped.gesture,'power_grasp')
        self.assertGreater(np.min(np.mean(q[:12].reshape(3,4)[:,1:],axis=1)),.55)

    def test_gradual_depth_flexion_during_wrist_translation(self):
        rot = PALM_TO_CAMERA
        filt = HandShape(self.open, image_observation(self.open, rot, self.position), rot)
        first, path = None, []
        for fraction in np.linspace(0, 1, 61):
            hand = human_hand([0, fraction, 0, 0])
            pos = self.position + [.04*fraction, -.02*fraction, .09*fraction]
            current = filt.update(hand, image_observation(hand, rot, pos), rot)
            tip_change = np.linalg.norm(current[12]-self.open[12])
            if first is None and tip_change > .008:
                first = fraction
            path.append(tip_change)
            np.testing.assert_allclose(current[HUMAN[[0,2,3]]],self.open[HUMAN[[0,2,3]]],atol=1e-8)
        self.assertIsNotNone(first)
        self.assertLess(first,.24)
        self.assertLess(np.linalg.norm(current[12]-hand[12]),.011)
        self.assertGreater(max(path),.07)

    def test_depth_reference_recovers_after_changing_view(self):
        rot = matrices(Rotation.from_euler('y', 24, degrees=True)) @ PALM_TO_CAMERA
        filt = HandShape(self.open, image_observation(self.open, PALM_TO_CAMERA, self.position), PALM_TO_CAMERA)
        for _ in range(5):
            resting = filt.update(self.open, image_observation(self.open, rot, self.position), rot)
            np.testing.assert_allclose(resting, self.open, atol=1e-8)
        for fraction in np.linspace(0, .18, 18):
            bent = human_hand([0, fraction, 0, 0])
            result = filt.update(bent, image_observation(bent, rot, self.position), rot)
        self.assertGreater(np.linalg.norm(result[12]-self.open[12]), .014)
        np.testing.assert_allclose(result[HUMAN[[0,2]]], self.open[HUMAN[[0,2]]], atol=1e-8)

    def test_fist_transitions_and_individual_knuckles(self):
        m=Retargeter(self.model,OPEN,self.open)
        for f in range(3):
            for site in range(4):
                targeted=human_hand(np.eye(4)[f]*.74,pinch=f,pinch_site=site)
                m.targets(targeted)
                self.assertEqual(m.contact_site,(f,site))
                for t in (.65,.8,1.):
                    m.targets(human_hand([t]*4))
                    self.assertIsNone(m.contact_site)
                    self.assertEqual(m.gesture,'power_grasp')
        # Check each target-point Jacobian against FK, independent of the thumb solver.
        q=np.linspace(.1,.8,16)
        for f in range(3):
            for site in range(4):
                kin=self.model.kinematics(q,(f,site))
                for j in range(4):
                    dq=q.copy()
                    dq[4*f+j]+=1e-7
                    actual=(self.model.kinematics(dq)[0][f,site+1]-kin[0][f,site+1])/1e-7
                    np.testing.assert_allclose(actual,kin[4][:,j],atol=1e-7)

    def test_recorded_wrist_turn_with_open_fingers(self):
        with np.load(PACKAGE/'data/recorded_left_hand.npz') as a:
            ci=int(a['calibration_index'])
            points,xy,raw=(a[k] for k in ('points','image_landmarks_px','measured_wrist_rotation_row_major'))
            self.assertFalse(visible_calibration_palm(dict(points=points[0],image_landmarks_px=xy[0]),raw[0]))
            self.assertTrue(visible_calibration_palm(dict(points=points[ci],image_landmarks_px=xy[ci]),raw[ci]))
            depths = a['surface_depths_m']
            signatures = a['shape_signatures']
            def observation(i):
                return dict(image_landmarks_px=xy[i], surface_depths_m=depths[i], shape_signatures=signatures[i])
            filt = HandShape(points[ci], observation(ci), raw[ci])
            for i in range(ci,len(points)):
                np.testing.assert_allclose(filt.update(points[i], observation(i), raw[i]), points[ci], atol=.002)
            relative=np.einsum('ji,njk->nik',raw[ci].reshape(3,3),raw.reshape(-1,3,3)[ci:])
            angles=np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
            self.assertGreater(max(angles),np.deg2rad(60))

    def test_pivot_moves_about_selected_reference(self):
        m=arm_model()
        hrot=PALM_TO_CAMERA
        ref=dict(wrist_position_m=self.position,palm_rotation_row_major=hrot)
        mapper=WristArm(m,ref,Retargeter(self.model,OPEN,self.open).basis,READY)
        # A change of pivot only transports the rigid offset. Joint and angular conventions coincide.
        h1=matrices(Rotation.from_euler('y',.40))@hrot
        pc,rc=mapper.target(self.position,hrot)
        pp,rp=mapper.target(self.position,h1)
        m.control_point='wrist'
        pw,rw=mapper.target(self.position,h1)
        np.testing.assert_allclose(pp,pc,atol=1e-10)
        np.testing.assert_allclose(rp,rw,atol=1e-10)
        np.testing.assert_allclose(pw-rw@m.wrist_to_palm,pc-rc@m.wrist_to_palm,atol=1e-10)


if __name__ == '__main__':
    unittest.main()
