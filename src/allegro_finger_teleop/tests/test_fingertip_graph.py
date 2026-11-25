"""Independent human finger chains for simultaneous opposition and a thumb folded across a fist."""
import unittest
import numpy as np

from test_retarget import FingerStream, HUMAN, OPEN, Retargeter, human_hand, load_robot, settle, unit
from test_wrist_hand import long_hand


def aim_finger(points, chain, target):
    joints = points[chain].copy()
    lengths = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    root = joints[0].copy()
    for _ in range(45):
        joints[-1] = target
        for i in (2, 1, 0):
            joints[i] = joints[i+1] + lengths[i] * unit(joints[i]-joints[i+1])
        joints[0] = root
        for i in range(3):
            joints[i+1] = joints[i] + lengths[i] * unit(joints[i+1]-joints[i])
    points[chain] = joints


def opposition(digits):
    hand = human_hand()
    center_y = np.mean(hand[HUMAN[list(digits), 0], 1])
    # Separate adjacent human pads even while all of them meet the same thumb pad.
    pad_y = np.linspace(-.008, .008, len(digits)) if len(digits) > 1 else [0.]
    for i, y in zip(digits, pad_y):
        hand[HUMAN[i]] = long_hand(i, [.45, .6, .4])[HUMAN[i]]
        aim_finger(hand, HUMAN[i], [.045, center_y+y, .098])
    aim_finger(hand, HUMAN[3], [.036, center_y, .094])
    return hand


def folded_fist(thumb_closed=True):
    hand = human_hand()
    for i in range(3):
        hand[HUMAN[i]] = long_hand(i, [1.35, 1.5, 1.25])[HUMAN[i]]
    if thumb_closed:
        aim_finger(hand, HUMAN[3], [.043, .014, .065])
    return hand


def retained_ring_pose(index=0., middle=0., join_middle=False):
    """Keep thumb and ring geometry fixed while freely moving the other fingers."""
    hand = opposition((2,))
    aim_finger(hand, HUMAN[2], hand[4] + [.016, 0., .005])
    for i, amount in enumerate((index, middle)):
        hand[HUMAN[i]] = long_hand(i, amount*np.array([1.22, 1.43, 1.20]))[HUMAN[i]]
    if join_middle:
        aim_finger(hand, HUMAN[1], hand[4] + [.002, -.012, .003])
    return hand


class FingertipGraphTest(unittest.TestCase):
    def test_quiet_thumb_filter_is_independent_of_moving_index(self):
        def trajectory(index_moving):
            stream = FingerStream(self.model, OPEN)
            raw = np.asarray(OPEN).copy()
            output = raw.copy()
            rows = []
            for frame in range(25):
                goal = np.asarray(OPEN).copy()
                goal[14] += .012*(-1)**frame
                if index_moving:
                    goal[1] += .045*frame
                output = stream.filter_finger_goals(goal, output, raw, 1./30., frame == 0)
                raw = goal
                rows.append(output.copy())
            return np.asarray(rows)
        still, moving = trajectory(False), trajectory(True)
        np.testing.assert_allclose(still[:,14], moving[:,14], atol=1e-12)
        self.assertLess(np.ptp(moving[5:,14]), .009)
        self.assertGreater(moving[-1,1]-OPEN[1], .045*24 - .10)

    def setUp(self):
        self.model = load_robot()
        self.open = human_hand()
        self.mapping = Retargeter(self.model, OPEN, self.open)

    def test_depth_supported_pinch_requires_independent_digit_identity(self):
        from hand_shape import HandShape
        filt = HandShape(self.open)
        xy = self.open[:, :2].copy()*1000
        xy[[0, 5, 9, 17, 8, 12, 16, 4]] = [[40, 100], [0, 60], [40, 40], [80, 55],
                                                [0, 0], [32, 5], [63, 0], [60, 1]]
        obs = dict(image_landmarks_px=xy, surface_depths_m=np.full(21, .90),
                   independent_tip_gaps=[1.2, 1.1, 1.1])
        for _ in range(4):
            filt.opposition_support(self.open, obs)
        self.assertFalse(filt.observed_tip.any())
        obs["independent_tip_gaps"][-1] = .19
        for _ in range(4):
            filt.opposition_support(self.open, obs)
        np.testing.assert_array_equal(filt.observed_tip, [False, False, True])
        obs["image_landmarks_px"] = None
        for _ in range(8):
            filt.opposition_support(self.open, obs)
        self.assertFalse(filt.observed_tip.any())

    def test_moving_an_opposed_thumb_to_a_different_pad(self):
        from hand_shape import HandShape
        shape = HandShape(self.open)
        image = self.open[:, :2].copy()
        image[[0, 5, 9, 17, 4, 8, 12, 16]] = [
            [40, 100], [0, 60], [40, 40], [80, 55], [0, 0], [2, 0], [40, 0], [60, 0]]
        points = self.open.copy()
        points[[8, 12, 16]] = points[4] + shape.width * np.array([[.30, 0., 0.], [.70, 0., 0.], [1., 0., 0.]])
        obs = dict(image_landmarks_px=image, surface_depths_m=np.full(21, .90),
                   independent_tip_gaps=[.20, .70, .90])
        for _ in range(4):
            shape.opposition_support(points, obs)
        np.testing.assert_array_equal(shape.observed_tip, [True, False, False])
        image[[8, 12]] = [[27, 0], [3, 0]]
        points[[8, 12]] = points[4] + shape.width * np.array([[.55, 0., 0.], [.36, 0., 0.]])
        obs["independent_tip_gaps"] = [.41, .62, .90]
        shape.opposition_support(points, obs)
        self.assertTrue(shape.observed_tip[0])
        for _ in range(3):
            shape.opposition_support(points, obs)
        np.testing.assert_array_equal(shape.observed_tip, [False, True, False])

    def test_oblique_pinch_requires_the_same_pair_and_releases(self):
        from hand_shape import HandShape
        filt = HandShape(self.open)
        width = filt.width
        points = self.open.copy()
        points[4] = points[8] + [0.5*width, 0., 0.]
        image = self.open[:,:2].copy()*1000
        image[[0,5,9,17,8,12,16,4]] = [[40,100],[0,60],[40,40],[80,55],
                                        [0,0],[35,0],[65,0],[3,0]]
        # Oblique projection leaves a gap in MANO. A single sample and an
        # ambiguous image overlap must both leave the hard contact inactive.
        filt.opposition_support(points, {"image_landmarks_px":image})
        self.assertFalse(filt.observed_tip.any())
        bad = image.copy()
        bad[12] = bad[8]
        filt.opposition_support(points, {"image_landmarks_px":bad})
        self.assertFalse(filt.observed_tip.any())
        for _ in range(2):
            filt.opposition_support(points, {"image_landmarks_px":image})
        self.assertEqual(filt.observed_tip.tolist(),[True,False,False])
        released = filt.opposition_support(self.open, {})
        self.assertFalse(filt.observed_tip.any())
        self.assertTrue(released[0] and released[3])
        for _ in range(8):
            released = filt.opposition_support(self.open, {})
        self.assertFalse(released.any())

    def test_surface_patch_confirms_oblique_ring_without_moving_other_fingers(self):
        from hand_shape import HandShape
        filt = HandShape(self.open)
        points = self.open.copy()
        points[4] = points[16] + [0., 0., .9*filt.width]
        points[8] = points[4] + [1.4*filt.width,0.,0.]
        points[12] = points[4] + [0.,1.4*filt.width,0.]
        image = self.open[:,:2].copy()*1000
        image[[0,5,9,17,8,12,16,4]] = [[40,100],[0,60],[40,40],[80,55],
                                          [0,0],[35,0],[66,0],[69,0]]
        depths = np.ones(21)*.70
        depths[4] = .66
        packet = dict(image_landmarks_px=image, surface_depths_m=depths)
        for _ in range(4):
            filt.opposition_support(points,packet)
        self.assertFalse(filt.observed_tip.any())
        depths[16] = .664
        for _ in range(2):
            filt.opposition_support(points,packet)
        self.assertEqual(filt.observed_tip.tolist(), [False,False,True])
        filt.interrupt()
        self.assertFalse(filt.observed_tip.any())

    def test_independent_middle_bend_requires_a_stable_view(self):
        from hand_shape import HandShape
        from test_wrist_hand import long_hand
        filt = HandShape(self.open)
        bent, _ = filt.register(long_hand(1, [.5,.7,.35]))
        for _ in range(2):
            self.assertFalse(filt.free_digit_support(bent,np.eye(3),.9).any())
        self.assertEqual(filt.free_digit_support(bent,np.eye(3),.9).tolist(),[False,True,False])
        tilt = np.diag([1.,-1.,-1.])
        self.assertFalse(filt.free_digit_support(bent,tilt,.9).any())

    def test_single_double_and_triple_contacts(self):
        for group in ((0,), (1,), (2,), (0,1), (1,2), (0,2), (0,1,2)):
            q = settle(self.mapping, opposition(group), count=32)
            tips = self.model.kinematics(q)[0][:,4]
            self.assertEqual(self.mapping.contacts, tuple((i,3) for i in group))
            for i in group:
                self.assertAlmostEqual(np.linalg.norm(tips[i]-tips[3]), .024, delta=.0012)
            # Pad centers cannot occupy the same location. Preserve finger order across the palm.
            for left, right in zip(group, group[1:]):
                self.assertGreater(np.linalg.norm(tips[left]-tips[right]), .0233)
                self.assertGreater(tips[right,1]-tips[left,1], .009)
            for i in set(range(3))-set(group):
                self.assertLess(np.abs((q-OPEN).reshape(4,4)[i]).max(), np.deg2rad(3.0))

    def test_wrist_and_contact_mapping_after_closed_pose(self):
        q = np.array(OPEN)
        for group in ((0,), (0,1), (0,1,2), (1,2), (2,), (1,), (0,)):
            pose = opposition(group)
            for _ in range(30):
                q = self.mapping.step(pose, q, .07)
            self.assertEqual(self.mapping.contacts, tuple((i,3) for i in group))
            tips = self.model.kinematics(q)[0][:,4]
            for i in group:
                self.assertLess(abs(np.linalg.norm(tips[i]-tips[3])-.024), .0015)
        # Fully curled idle fingers can coexist with one opposed fingertip.
        p = folded_fist(False)
        pin = opposition((1,))
        p[HUMAN[[1,3]]] = pin[HUMAN[[1,3]]]
        self.mapping.targets(p)
        self.assertEqual(self.mapping.contacts, ((1,3),))

    def test_thumb_crosses_or_stays_extended_according_to_observation(self):
        folded = settle(self.mapping, folded_fist())
        self.assertEqual(self.mapping.gesture, "power_grasp")
        extended = settle(self.mapping, folded_fist(False))
        self.assertTrue(np.all(np.abs(extended[12:]-OPEN[12:]) < np.deg2rad(3)))
        self.assertGreater(folded[14]-extended[14], np.deg2rad(40))
        for q in (folded, extended):
            self.assertGreater((q[:12]-OPEN[:12]).reshape(3,4)[:,1:].min(), np.deg2rad(55))

    def test_retained_pair_is_independent_of_other_fingers_and_releases(self):
        q = np.array(OPEN)
        still = retained_ring_pose()
        for _ in range(55):
            q = self.mapping.step(still, q, .05, measured=q)
        original = q.copy()
        for t in np.linspace(0., 2*np.pi, 66):
            shape = retained_ring_pose(.5-.5*np.cos(t), .5-.5*np.cos(2*t))
            np.testing.assert_array_equal(shape[HUMAN[[2,3]]], still[HUMAN[[2,3]]])
            q = self.mapping.step(shape, q, .05, measured=q)
            self.assertEqual(self.mapping.contacts, ((2,3),))
            tips = self.model.kinematics(q)[0][:,4]
            self.assertLess(np.linalg.norm(tips[2]-tips[3]), .025)
            self.assertLess(np.max(np.abs(q[8:]-original[8:])), np.deg2rad(4.))
        # One frame of apparent proximity from a bystander cannot add a contact.
        self.mapping.step(retained_ring_pose(join_middle=True), q, .033, measured=q)
        self.assertEqual(self.mapping.contacts, ((2,3),))
        self.mapping.step(self.open, q, .033, measured=q)
        self.assertEqual(self.mapping.contacts, ())

    def test_loose_approach_does_not_lock_the_thumb_during_fist_closure(self):
        mapping = self.mapping
        fist = folded_fist()
        entry = fist.copy()
        entry[HUMAN[0]] = self.open[HUMAN[0]]
        gaps = mapping.width*np.array([[.87,.64,.36,.45],[.87,.64,.34,.38],[.84,.54,.34,.51]])
        mapping.contacts = mapping.choose_contacts(entry@mapping.basis, gaps, .05)
        self.assertEqual(mapping.contacts, ((1,3),))
        mapping.contacts = mapping.choose_contacts(fist@mapping.basis, gaps, .05)
        self.assertEqual(mapping.gesture, "power_grasp")

    def test_close_tips_at_neutral_have_finite_scale(self):
        close = self.open.copy()
        close[8] = close[12]
        mapping = Retargeter(self.model, OPEN, close)
        np.testing.assert_allclose(mapping.step(close, OPEN, .06), OPEN, atol=1e-8)
        self.assertTrue(np.isfinite(mapping.pair_scales).all())

    def test_obscured_thumb_with_visible_closure_and_unseen_hand(self):
        from hand_shape import HandShape
        from test_hand_isolation import image_observation, PALM_TO_CAMERA
        from perception_hamer.src.finger_observation import finger_shape_signature
        pos = np.array([0., 0., .63])
        def sensor(model, independent, rotation):
            obs = image_observation(independent, rotation, pos)
            obs["shape_signatures"] = [finger_shape_signature(model), finger_shape_signature(independent)]
            return obs
        filt = HandShape(self.open, sensor(self.open, self.open, PALM_TO_CAMERA), PALM_TO_CAMERA)
        rot = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]) @ PALM_TO_CAMERA
        closed = folded_fist()
        for _ in range(3):
            measured = filt.update(closed, sensor(closed,closed,rot), rot)
        self.assertGreater(np.linalg.norm(measured[4]-self.open[4]), .05)
        for _ in range(4):
            missing = sensor(self.open,self.open,rot)
            missing["image_landmarks_px"] = None
            np.testing.assert_allclose(filt.update(self.open,missing,rot), measured)
        # A rigid hand turn with a falsely curled MANO thumb supplies no corroboration.
        quiet = HandShape(self.open, sensor(self.open,self.open,PALM_TO_CAMERA), PALM_TO_CAMERA)
        for _ in range(5):
            same = quiet.update(closed, sensor(closed,self.open,rot), rot)
        np.testing.assert_allclose(same[HUMAN[3]], self.open[HUMAN[3]], atol=1e-8)
        # An optional shape observation can reappear after C. Only a stationary
        # finger may bind that observation to its previously accepted reference.
        late = HandShape(self.open, image_observation(self.open, PALM_TO_CAMERA, pos), PALM_TO_CAMERA)
        late.update(self.open, sensor(self.open, self.open, rot), rot)
        for _ in range(3):
            recovered = late.update(closed, sensor(closed, closed, rot), rot)
        self.assertGreater(np.linalg.norm(recovered[4]-self.open[4]), .05)

    def test_controller_reads_official_degree_endpoints(self):
        from allegro_v4_limits import hardware_limits
        self.assertEqual(self.model.names, [name for name,_ in hardware_limits()])
        bounds = np.array([span for _,span in hardware_limits()])
        np.testing.assert_allclose(self.model.lower, bounds[:,0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(self.model.upper, bounds[:,1], atol=1e-10, rtol=0)
