"""Wrist-local task-space retargeting for the simulated left Allegro, in metres and radians."""
from itertools import combinations
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# In the hand controller's order: index, middle, ring, thumb (OpenPose has the thumb first).
DIGITS = ("if", "mf", "rf", "th")
HUMAN = np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [1, 2, 3, 4]])
ROWS = np.arange(4)
PROXIMAL = np.array([1, 1, 1, 2])
# The proximal interphalangeal centers on the long fingers and the thumb's single IP.
HUMAN_MIDDLE = np.array([6, 10, 14, 3])
ROBOT_MIDDLE = np.array([2, 2, 2, 3])
MIDDLE_WEIGHTS = (1.1, 1.1, 0.)  # wrist–thumb IP, wrist–long PIPs, thumb IP–long PIPs
FINGERTIP_PAIRS = np.array(list(combinations(range(4), 2)))

# Residual weights. Unit directions and joint-angle offsets are converted to metres.
SEGMENT_WEIGHT_M = .034
PINCH_DISTANCE_WEIGHT = 18.0
PINCH_VECTOR_WEIGHT = .65
PAD_CLEARANCE_WEIGHT = 8.0
TEMPORAL_WEIGHT_M_PER_RAD = .008
OPEN_WEIGHT_M_PER_RAD = .005
CONTACT_CHANGE_TIME_S = .85
CONTACT_CHANGE_CLEARANCE_M = .021


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-7) or not np.isfinite(n).all():
        raise ValueError("degenerate hand geometry")
    return v / n


def coordinate(node, key):
    return np.array([float(x) for x in node.get(key, "0 0 0").split()])


def matrices(rotation):
    # ROS Noetic ships SciPy 1.3, in which as_matrix was named as_dcm.
    return rotation.as_matrix() if hasattr(rotation, "as_matrix") else rotation.as_dcm()


def direction_alignment(source, target):
    """Shortest proper rotation from each calibrated human segment to its robot segment."""
    result = []
    for a, b in zip(source.reshape(-1, 3), target.reshape(-1, 3)):
        axis = np.cross(a, b)
        sine = np.linalg.norm(axis)
        cosine = np.clip(a @ b, -1.0, 1.0)
        if sine > 1e-8:
            rotvec = axis * (np.arctan2(sine, cosine) / sine)
        elif cosine < 0:
            rotvec = np.pi * unit(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]))
        else:
            rotvec = np.zeros(3)
        result.append(matrices(Rotation.from_rotvec(rotvec)))
    return np.array(result).reshape(source.shape[:-1] + (3, 3))


class AllegroModel:
    def __init__(self, urdf, joint_names):
        root = ET.fromstring(urdf)
        nodes = {j.get("name"): j for j in root.findall("joint")}
        links = {l.get("name"): l for l in root.findall("link")}
        suffixes = [("axl", "mcp", "pip", "dip")] * 3 + [("cmc", "axl", "mcp", "ipl")]
        self.names = ["algr_lh_" + d + "_" + s for d, group in zip(DIGITS, suffixes) for s in group]
        if list(joint_names) != self.names:
            raise ValueError("hand controller joint order does not match the Allegro URDF chains")
        chain = [nodes[name] for name in self.names]
        self.offset = np.array([coordinate(j.find("origin"), "xyz") for j in chain]).reshape(4, 4, 3)
        rpy = [coordinate(j.find("origin"), "rpy") for j in chain]
        self.fixed = matrices(Rotation.from_euler("xyz", rpy)).reshape(4, 4, 3, 3)
        self.axes = unit(np.array([coordinate(j.find("axis"), "xyz") for j in chain]).reshape(4, 4, 3))
        self.lower, self.upper, self.speed = [
            np.array([float(j.find("limit").get(key)) for j in chain]) for key in ("lower", "upper", "velocity")]
        pads = []
        for i, digit in enumerate(DIGITS):
            parent = "algr_lh_palm"
            for j in chain[4*i:4*i+4]:
                if j.find("parent").get("link") != parent:
                    raise ValueError("disconnected finger " + digit)
                parent = j.find("child").get("link")
            sphere, = [c for c in links[parent].findall("collision") if c.find("geometry/sphere") is not None]
            pads.append((coordinate(sphere.find("origin"), "xyz"), float(sphere.find("geometry/sphere").get("radius"))))
        self.pad_offsets = np.array([p for p, _ in pads])
        self.pad_radii = np.array([r for _, r in pads])
        self.knuckle_radii = np.array([
            [min(coordinate(links[chain[4*f+j].find("child").get("link")].find("collision/geometry/box"), "size")[:2])/2
             for j in (1, 2, 3)] for f in range(3)])
        self.tcp = coordinate(next(j for j in nodes.values() if j.find("child").get("link") == "palm_tcp").find("origin"), "xyz")

    def kinematics(self, q, contact_site=None, with_middle=False):
        """Joint origins, pad centres and geometric Jacobians in the Allegro palm basis."""
        angles = np.asarray(q).reshape(4, 4, 1)
        moving = matrices(Rotation.from_rotvec((self.axes * angles).reshape(16, 3))).reshape(4, 4, 3, 3)
        rot = np.broadcast_to(np.eye(3), (4, 3, 3))
        p = np.zeros((4, 3))
        origins, axes = [], []
        for i in range(4):
            p = p + (rot @ self.offset[:, i, :, None])[:, :, 0]
            origins.append(p)
            rot = rot @ self.fixed[:, i]
            axes.append((rot @ self.axes[:, i, :, None])[:, :, 0])
            rot = rot @ moving[:, i]
        pads = p + (rot @ self.pad_offsets[:, :, None])[:, :, 0]
        origins = np.stack(origins + [pads], axis=1)
        axes = np.stack(axes, axis=1)
        # Express tip coordinates as palm_tcp-to-pad displacements; its axes equal the palm axes.
        origins -= self.tcp
        pad_jac = np.cross(axes, origins[:, 4:5] - origins[:, :4]).transpose(0, 2, 1)
        proximal = unit(origins[ROWS, PROXIMAL + 1] - origins[ROWS, PROXIMAL])
        distal = unit(origins[:, 4] - origins[:, 3])
        directions = np.stack([proximal, distal], axis=1)
        dir_jac = np.cross(axes[:, None, :, :], directions[:, :, None, :]).transpose(0, 1, 3, 2)
        dir_jac[:, 0] *= (np.arange(4) <= PROXIMAL[:, None])[:, None, :]
        result = (origins, directions, pad_jac, dir_jac)
        if contact_site is not None:
            f,site = contact_site
            jac = np.cross(axes[f], origins[f,site+1]-origins[f,:4]).T
            jac[:, site+1:] = 0
            result += (jac,)
        if with_middle:
            center = origins[ROWS, ROBOT_MIDDLE]
            jac = np.cross(axes, center[:,None]-origins[:,:4]).transpose(0,2,1)
            jac *= (np.arange(4) < ROBOT_MIDDLE[:,None])[:,None,:]
            result += (jac,)
        return result


def human_features(points):
    fingers = points[HUMAN]
    return fingers[:, 3], np.stack([unit(fingers[ROWS, [1, 1, 1, 2]] - fingers[ROWS, [0, 0, 0, 1]]),
                                     unit(fingers[:, 3] - fingers[:, 2])], axis=1)


def anatomical_angles(points, flexion_axes):
    """Long-finger MCP/PIP/DIP flexion, and thumb spread, elevation, MCP/IP flexion."""
    segments = unit(np.diff(points[HUMAN], axis=1))
    proximal = segments[:3, 0]
    forward = np.cross([1., 0., 0.], flexion_axes)
    mcp = np.arctan2(proximal[:, 0], np.sum(proximal*forward, axis=1))
    before, after = segments[:3, :-1], segments[:3, 1:]
    hinge = np.arctan2(np.sum(np.cross(before, after)*flexion_axes[:, None], axis=2),
                       np.sum(before*after, axis=2))
    thumb = segments[3]
    metacarpal = thumb[0]
    flex = np.arctan2(np.linalg.norm(np.cross(thumb[:-1], thumb[1:]), axis=1),
                      np.sum(thumb[:-1]*thumb[1:], axis=1))
    basal = [np.arctan2(metacarpal[1], metacarpal[2]),
             np.arctan2(metacarpal[0], np.linalg.norm(metacarpal[1:]))]
    return np.r_[np.column_stack((mcp, hinge)).ravel(), basal, flex]


class ThumbFilter:
    """Low pass the four wrist-local thumb points; open the bandwidth during intentional motion."""
    def __init__(self, reference):
        self.thumb = reference[1:5].copy()

    def update(self, points, dt):
        thumb = points[1:5]
        # Both the basal segment and the tip can initiate a motion.
        speed = np.max(np.linalg.norm(thumb-self.thumb, axis=1)) / dt
        motion = np.clip((speed-.025)/.18, 0, 1)
        gain = -np.expm1(-dt/(.15-.125*motion))
        self.thumb += gain*(thumb-self.thumb)
        output = points.copy()
        output[1:5] = self.thumb
        return output


class Retargeter:
    def __init__(self, model, open_q, reference, middle_weights=MIDDLE_WEIGHTS):
        self.model = model
        self.open_q = np.asarray(open_q)
        forward = unit(reference[9] - reference[0])
        side = reference[13] - reference[5]
        side = unit(side - (side @ forward) * forward)
        self.basis = np.column_stack((np.cross(side, forward), side, forward))
        reference = reference @ self.basis
        self.width = np.linalg.norm(reference[17] - reference[5])
        self.human_open, directions = human_features(reference)
        self.recent_tips = self.human_open.copy()
        self.flexion_axes = unit(np.cross(directions[:3, 0], [1., 0., 0.]))
        self.open_angles = anatomical_angles(reference, self.flexion_axes)
        self.angle_indices = np.r_[1:4, 5:8, 9:12, 12:16]
        self.contacts = ()
        self.join_time = np.zeros(3)
        self.established_tip = np.zeros(3, dtype=bool)
        self.approach_remaining = np.zeros(3)
        self.pad_contact_seen = np.zeros(3, dtype=bool)
        self.hold_reference = np.full((3, 2, 4, 3), np.nan)
        self.hold_angles = np.zeros((3, 2, 4))
        self.last_hand_q = self.open_q.copy()
        self.power_grasp = False
        self.observed_sweep = 0.0
        skeleton, robot_dirs, _, _ = model.kinematics(self.open_q)
        self.robot_open = skeleton[:, 4]
        self.wrist_open = self.robot_open + model.tcp
        self.human_thumb_base = reference[1].copy()
        self.robot_thumb_base = skeleton[3, 0].copy()
        self.thumb_axis = model.fixed[3, 0] @ model.axes[3, 0]
        human_lengths = np.linalg.norm(np.diff(reference[HUMAN], axis=1), axis=2).sum(axis=1)
        robot_lengths = np.linalg.norm(np.diff(skeleton[:, 1:], axis=1), axis=2).sum(axis=1)
        # The thumb CMC-to-MCP span contains an extra joint in the robot.
        robot_lengths[3] = (np.linalg.norm(skeleton[3, 2] - skeleton[3, 0])
                            + np.linalg.norm(np.diff(skeleton[3, 2:], axis=0), axis=1).sum())
        self.scale = robot_lengths / human_lengths
        if not (0.025 < self.width < 0.13 and np.all((self.scale > 0.5) & (self.scale < 4.5))):
            raise ValueError("implausible size of the open-hand reference")
        self.align_dirs = direction_alignment(directions, robot_dirs)
        self.open_sites = reference[HUMAN[:3]].copy()
        self.open_span = np.linalg.norm(self.open_sites - self.human_open[3], axis=2)
        self.middle_weights = np.array(middle_weights, dtype=float)
        if self.middle_weights.shape != (3,) or not np.isfinite(self.middle_weights).all() or np.any(self.middle_weights < 0):
            raise ValueError("expected three nonnegative middle-joint weights")
        self.use_middle = bool(self.middle_weights.any())
        human_across = reference[13]-reference[5]
        robot_across = skeleton[2,1]-skeleton[0,1]
        self.human_across = human_across / np.dot(human_across, human_across)
        self.robot_across = unit(robot_across)
        self.robot_palm_width = np.linalg.norm(robot_across)
        self.middle_human_open = reference[HUMAN_MIDDLE]
        self.middle_robot_open = skeleton[ROWS, ROBOT_MIDDLE]
        human_ray = self.middle_human_open - reference[HUMAN[:,0]]
        robot_ray = self.middle_robot_open - skeleton[ROWS, [1,1,1,0]]
        first_bone = reference[HUMAN[:,1]]-reference[HUMAN[:,0]]
        # A folded thumb in an older calibration can put its IP near the root.
        human_ray = np.where((np.linalg.norm(human_ray,axis=1) < .25*np.linalg.norm(first_bone,axis=1))[:,None],
                             first_bone, human_ray)
        self.middle_scale = np.linalg.norm(robot_ray,axis=1)/np.linalg.norm(human_ray,axis=1)
        self.middle_alignment = direction_alignment(unit(human_ray), unit(robot_ray))
        pair_h = self.middle_human_open[:3]-self.middle_human_open[3]
        pair_r = self.middle_robot_open[:3]-self.middle_robot_open[3]
        # Knuckles can overlap in a usable calibration. Use the separation of
        # the finger roots for that pair, as for touching open fingertips below.
        root_h = reference[HUMAN[:3,0]]-reference[HUMAN[3,0]]
        pair_axis = np.where((np.linalg.norm(pair_h,axis=1) < .25*np.linalg.norm(root_h,axis=1))[:,None],
                             root_h, pair_h)
        pair_axis = np.where((np.linalg.norm(pair_axis,axis=1) < 1.e-6)[:,None], first_bone[:3], pair_axis)
        self.middle_pair_alignment = direction_alignment(unit(pair_axis), unit(pair_r))
        self.middle_pair_scale = np.linalg.norm(pair_r,axis=1)/np.linalg.norm(pair_axis,axis=1)
        first, second = FINGERTIP_PAIRS.T
        self.human_pairs_open = self.human_open[first] - self.human_open[second]
        self.robot_pairs_open = self.robot_open[first] - self.robot_open[second]
        root_span = reference[HUMAN[first,0]] - reference[HUMAN[second,0]]
        # Adjacent tips may touch in an otherwise valid open-hand calibration.
        # Use their metacarpal separation to establish that pair's scale and lateral axis.
        close = np.linalg.norm(self.human_pairs_open, axis=1) < .55*np.linalg.norm(root_span, axis=1)
        calibration_span = np.where(close[:,None], root_span, self.human_pairs_open)
        self.pair_scales = np.linalg.norm(self.robot_pairs_open, axis=1) / np.linalg.norm(calibration_span, axis=1)
        self.pair_alignment = direction_alignment(unit(calibration_span), unit(self.robot_pairs_open))

    @property
    def contact_site(self):
        return self.contacts[0] if len(self.contacts) == 1 else None

    def choose_contacts(self, local, distance, dt=None):
        curls = (anatomical_angles(local, self.flexion_axes)[:9] - self.open_angles[:9]).reshape(3,3)
        curls = np.maximum(curls, 0).sum(axis=1)
        # Three long fingers closing together is the power-grasp pattern. A thumb
        # resting across their knuckles must not pull a single finger into opposition.
        self.power_grasp = bool(curls.min() > 2.25 or (curls.min() > .65 and np.ptp(curls) < .45))
        held_thumb = np.linalg.norm(local[HUMAN[:3]] - self.human_open[3], axis=2)
        approach = held_thumb-distance
        entry = self.width*np.array([.32, .32, .32, .58])
        eligible = ((distance < entry) & (self.open_span-distance > .16*self.width)
                    & (approach > .09*self.width))
        # An isolated fingertip can close against a stationary opposed thumb.
        eligible[:,3] |= ((distance[:,3] < .24*self.width) &
                          (self.open_span[:,3]-distance[:,3] > .25*self.width))
        # Several distal pads can meet one thumb. Test each identity and require
        # comparable clearance to distinguish a second contact from a nearby bystander.
        tip_gap = distance[:, 3]
        was_tip = np.array([(i, 3) in self.contacts for i in range(3)])
        own_knuckle_gap = distance[:, :3].min(axis=1)
        clustered = (eligible[:, 3] & (tip_gap < (.32+.035*was_tip)*self.width)
                     & (tip_gap < tip_gap.min()+(.12+.035*was_tip)*self.width)
                     & (tip_gap < own_knuckle_gap+.065*self.width))
        # Once two distal pads are opposed, only the geometry of that same
        # digit and the thumb should release them. Bending other fingers may
        # satisfy the whole-hand fist rule or move another knuckle closer.
        retained = (was_tip & (tip_gap < .40*self.width)
                    & (self.open_span[:,3]-tip_gap > .12*self.width)
                    & (tip_gap < own_knuckle_gap+.11*self.width))
        # A loose approach can be labelled as a single pinch while the last
        # finger is still closing into a fist. Only a previously observed tight
        # pad pair earns the right to override the subsequent full-hand closure.
        self.established_tip = (self.established_tip & was_tip) | clustered
        if self.power_grasp:
            retained &= self.established_tip
        new_tip = clustered & ~was_tip
        if dt is not None:
            self.join_time = np.where(new_tip, self.join_time + min(dt, .06), 0.)
            new_tip &= self.join_time >= .09
        if np.any(retained):
            self.power_grasp = False
            return tuple((i,3) for i in np.flatnonzero(retained | new_tip))
        if clustered.sum() >= 2:
            self.power_grasp = False
            return tuple((i, 3) for i in np.flatnonzero(clustered))
        if self.power_grasp:
            # The other fingers may deliberately be curled during a precision
            # pinch. A single distal pad clearly nearer than every knuckle is
            # still an observed fingertip contact.
            i = int(np.argmin(tip_gap))
            if (clustered[i] and tip_gap[i] < .24*self.width
                    and tip_gap[i]+.045*self.width < distance[:, :3].min()):
                self.power_grasp = False
                return ((i, 3),)
            return ()
        score = distance.copy()
        score[:,:3] += .12*self.width  # A nearby distal joint also accompanies a tip pinch.
        if self.contact_site is not None:
            f,site = self.contact_site
            if distance[f,site] < entry[site]+.09*self.width and self.open_span[f,site]-distance[f,site] > .12*self.width:
                eligible[f,site] = True
                score[f,site] -= .06*self.width
        if not np.any(eligible):
            return ()
        return (np.unravel_index(np.argmin(np.where(eligible, score, np.inf)), score.shape),)

    @property
    def gesture(self):
        if not self.contacts:
            return "power_grasp" if self.power_grasp else "free"
        names = ("index","middle","ring")
        return "thumb_{}_{}".format("_".join(names[f] for f, _ in self.contacts),
                                    ("mcp","pip","dip","tip")[self.contacts[0][1]])

    def targets(self, points, dt=None, observed_tip=None, separated_tip=None, retained_only=False):
        local = points @ self.basis
        tips, directions = human_features(local)
        # Human points are relative to the C-calibrated wrist ring. The robot
        # wrist is the mounting origin; FK uses palm_tcp so translate at its boundary.
        wrist_vectors = self.wrist_open + self.scale[:, None] * (tips - self.human_open)
        target = wrist_vectors - self.model.tcp
        dirs = (self.align_dirs @ directions[..., None])[..., 0]
        sites = local[HUMAN[:3]]
        distances = np.linalg.norm(sites-tips[3], axis=2)
        if observed_tip is not None:
            # A confirmed opposing pad is a contact objective even when MANO
            # leaves a gap at an oblique view. Other pairs keep continuous IK.
            distances[observed_tip,3] = np.minimum(distances[observed_tip,3], .30*self.width)
        if separated_tip is not None:
            # A measured open pair cannot acquire a hard pad-contact objective.
            # Knuckle targets and all continuous shape tasks retain their own geometry.
            distances[separated_tip, 3] = np.maximum(distances[separated_tip, 3], .70*self.width)
        self.contacts = self.choose_contacts(local, distances, dt)
        if retained_only:
            self.contacts = tuple((i, 3) for i in np.flatnonzero(observed_tip))
            if self.contacts:
                self.power_grasp = False
        elif observed_tip is not None and np.any(observed_tip) and not self.power_grasp:
            verified = {(i,3) for i in np.flatnonzero(observed_tip)}
            # Unverified neighbors may join once their own distal pads also
            # converge. A merely flexing finger cannot displace the verified pair.
            additions = {c for c in self.contacts if c[1] == 3 and distances[c] < .24*self.width}
            self.contacts = tuple(sorted(verified | additions))
        relative = tips[:3] - tips[3]
        pinch, span = np.zeros(3), np.zeros(3)
        for f, site in self.contacts:
            tip = site == 3
            start = (.67 if tip else .41)*self.width
            near = (.35 if tip else .23)*self.width
            t = np.clip((start-distances[f,site])/(start-near), 0, 1)
            pinch[f] = t*t*(3-2*t)
            radius = self.model.pad_radii[f] if tip else self.model.knuckle_radii[f,site]
            span[f] = (self.model.pad_radii[3]+radius +
                       max(distances[f,site]-near,0)*(self.scale[f]+self.scale[3])/2)
            relative[f] = sites[f,site]-tips[3]
        distance = np.linalg.norm(relative, axis=1)
        axis = relative / np.maximum(distance[:, None], 1e-6)
        return target, dirs, pinch, span, axis

    def _contact_support(self, local, previous, dt, old_contacts, feedback):
        """Maintain settled opposition and stage an additional pad in the wrist frame."""
        tips = local[HUMAN[:,3]]
        tip_motion = np.linalg.norm(tips-self.recent_tips, axis=1)
        self.recent_tips += -np.expm1(-dt/.40)*(tips-self.recent_tips)
        active = np.array([(f,3) in self.contacts for f in range(3)])
        if feedback is not None:
            gaps = np.linalg.norm(feedback[:3,4]-feedback[3,4], axis=1)-self.model.pad_radii[:3]-self.model.pad_radii[3]
            self.pad_contact_seen |= active & (gaps < .0008)
        self.approach_remaining[~active] = 0.
        self.pad_contact_seen &= active
        # A changed group only repositions pads that are still approaching
        # or whose own human contact has moved in the wrist frame. Bystander
        # motion never requests an opening of a stationary, established pair.
        if (feedback is not None and old_contacts and self.contacts != old_contacts
                and max(len(old_contacts), len(self.contacts)) > 1):
            relocating = np.maximum(tip_motion[:3], tip_motion[3]) > .08*self.width
            joining = np.array([(f,3) not in old_contacts for f in range(3)])
            self.approach_remaining[active & (joining | relocating | ~self.pad_contact_seen)] = CONTACT_CHANGE_TIME_S
        elapsed = CONTACT_CHANGE_TIME_S-self.approach_remaining
        clearance_gain = np.minimum(np.minimum(elapsed/.16, self.approach_remaining/.42), 1.)
        self.approach_remaining = np.maximum(0., self.approach_remaining-dt)
        # Pinch distance alone leaves both hands' local contact point free to
        # drift. Retain a settled pair's robot posture while its own two human
        # chains are stationary, even when neighboring digits flex or extend.
        hold_q = np.zeros((4,4))
        hold_count = np.zeros(4)
        speed = np.max(np.abs(previous-self.last_hand_q).reshape(4,4),axis=1)/dt
        self.last_hand_q = np.array(previous, copy=True)
        for f in range(3):
            pair = [f,3]
            chains = local[HUMAN[pair]]
            if not active[f]:
                self.hold_reference[f].fill(np.nan)
                continue
            reference = self.hold_reference[f]
            if np.isfinite(reference).all() and np.max(np.linalg.norm(chains-reference,axis=2)) > .085*self.width:
                reference.fill(np.nan)
            # Save the loaded posture after both digits have settled.
            if (not np.isfinite(reference).all() and feedback is not None
                    and gaps[f] < .0008 and tip_motion[pair].max() < .055*self.width
                    and speed[pair].max() < .055 and self.approach_remaining[f] == 0.):
                reference[:] = chains
                self.hold_angles[f] = previous.reshape(4,4)[pair]
            if np.isfinite(reference).all():
                hold_q[pair] += self.hold_angles[f]
                hold_count[pair] += 1.
        hold_q /= np.maximum(1., hold_count[:,None])
        # A single contact leaves the pair's posture underdetermined. In a
        # multipinch, the additional pad constrains it and the thumb can roll.
        holding_weight = .36 if np.count_nonzero(active) == 1 else 0.
        hold_weight = np.repeat((hold_count>0)*holding_weight, 4)
        return clearance_gain, hold_q.ravel(), hold_weight

    def step(self, points, previous, dt, measured=None, observed_tip=None, separated_tip=None, retained_only=False):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        old_contacts = self.contacts
        target, dirs, pinch, span, axis = self.targets(points, dt, observed_tip, separated_tip, retained_only)
        model = self.model
        previous = np.clip(previous, model.lower, model.upper)
        local = points @ self.basis
        feedback = model.kinematics(measured)[0] if measured is not None and self.contacts else None
        clearance_gain, hold_q, hold_weight = self._contact_support(
            local, previous, dt, old_contacts, feedback)
        span += CONTACT_CHANGE_CLEARANCE_M*clearance_gain*pinch
        hold_jac = hold_weight[:,None]*np.eye(16)
        strength = np.r_[pinch, np.max(pinch)]
        delta_angle = anatomical_angles(local, self.flexion_axes) - self.open_angles
        delta_angle = np.arctan2(np.sin(delta_angle), np.cos(delta_angle))
        # A human in-plane sweep and the robot CMC rotate about different physical
        # axes. Transport the sweep through the real CMC before fitting residual
        # thumb flexion; otherwise the solver curls the IP to chase an impossible arc.
        sweep = delta_angle[9]
        basal_projection = np.linalg.norm(unit(local[2]-local[1])[1:])
        if basal_projection >= .45:
            self.observed_sweep = sweep
        else:
            # Azimuth becomes undefined when the metacarpal points along the palm normal.
            observed = np.clip((basal_projection-.18)/.27, 0., 1.)
            difference = sweep-self.observed_sweep
            sweep = self.observed_sweep + observed*np.arctan2(np.sin(difference), np.cos(difference))
            delta_angle[9] = sweep
        cmc_turn = np.clip(self.open_q[12]+sweep, model.lower[12], model.upper[12])-self.open_q[12]
        human_turn = matrices(Rotation.from_rotvec([-sweep, 0., 0.]))
        robot_turn = matrices(Rotation.from_rotvec(self.thumb_axis*cmc_turn))
        rest_ray = self.human_open[3]-self.human_thumb_base
        residual_tip = human_turn.T @ (local[4]-self.human_thumb_base) - rest_ray
        free_tip = self.robot_thumb_base + robot_turn @ (
            self.robot_open[3]-self.robot_thumb_base + self.scale[3]*residual_tip)
        human_dirs = unit(np.diff(local[1:5], axis=0))[1:]
        free_dirs = (robot_turn @ self.align_dirs[3] @ human_turn.T @ human_dirs[..., None])[..., 0]
        # A curled/opposed thumb requires its full wrist-to-tip target also in
        # a fist. The basal-axis transport is for a largely pure lateral sweep.
        fold = np.clip(np.max(np.abs(delta_angle[10:]))/.55, 0, 1)
        thumb_task = max(strength[3], fold)
        target[3] = (1-thumb_task)*free_tip + thumb_task*target[3]
        dirs[3] = unit((1-thumb_task)*free_dirs + thumb_task*dirs[3])
        # A nearly straight thumb carries little observable roll about its own long axis.
        # Stabilize the axial joint from basal elevation, keeping free lateral sweep on the CMC.
        across = np.dot(local[4]-self.human_open[3], self.human_across)
        # Across-palm thumb opposition is dimensionless on the human hand.
        # Apply it in the robot palm width, gradually as the thumb folds in.
        across_weight = 2.8*(1-strength[3])**2 * np.clip((across-.55)/.65,0.,1.) * fold
        across_goal = across*self.robot_palm_width
        angle_target = self.open_q[self.angle_indices] + delta_angle
        angle_target = np.clip(angle_target, model.lower[self.angle_indices], model.upper[self.angle_indices])
        angle_weights = np.r_[np.repeat(.095*(1-.88*pinch), 3),
                              np.array([.14, .12, .10, .12])*(1-.93*strength[3])]
        angle_jac = angle_weights[:, None]*np.eye(16)[self.angle_indices]
        step_limit = model.speed * min(dt, 0.12)
        lo, hi = np.maximum(model.lower, previous-step_limit), np.minimum(model.upper, previous+step_limit)
        first, second = FINGERTIP_PAIRS.T
        human_pairs = local[HUMAN[first,3]] - local[HUMAN[second,3]]
        pair_change = human_pairs - self.human_pairs_open
        pair_goal = self.robot_pairs_open + self.pair_scales[:, None] * (self.pair_alignment @ pair_change[:, :, None])[:, :, 0]
        # A stationary digit remains a good anchor when a neighboring digit
        # moves independently. Do not obtain a pinch by moving unrelated fingers.
        hand_motion = np.linalg.norm(local[HUMAN[:,3]]-self.human_open, axis=1) / self.width
        activity = np.clip(hand_motion/.22, 0., 1.)
        anchor = np.repeat(.32*(1-np.maximum(activity,strength))**2, 4)
        anchor_jac = anchor[:,None]*np.eye(16)
        pair_weights = .12 + .26*np.minimum(activity[first], activity[second])
        # The isolated opposition term below controls exact surface contact.
        # The complete graph still constrains the relative orientation of all
        # other fingertips, including adjacent long fingers in a multi-pinch.
        tip_pinch = np.array([pinch[i] if (i,3) in self.contacts else 0. for i in range(3)])
        if len(self.contacts) > 1:
            long_pair = second != 3
            paired = np.minimum(tip_pinch[first[long_pair]], tip_pinch[second[long_pair]])
            pair_weights[long_pair] += .85*paired
        thumb_rows = second == 3
        thumb_vec = pair_goal[thumb_rows]
        thumb_length = np.linalg.norm(thumb_vec, axis=1, keepdims=True)
        # At a near-zero human separation its direction is poorly determined.
        # Let the surface-distance term dominate there while retaining a weak
        # directional preference and a distinct vector for each physical pad.
        thumb_length_goal = (1-tip_pinch)*thumb_length[:,0] + tip_pinch*span
        pair_goal[thumb_rows] = thumb_vec * (thumb_length_goal[:,None]/np.maximum(thumb_length, 1e-6))
        pair_weights[thumb_rows] *= 1-.75*tip_pinch
        pair_weights = pair_weights[:,None]
        distance_weight = PINCH_DISTANCE_WEIGHT * pinch
        vector_weight = (PINCH_VECTOR_WEIGHT * pinch*(1-pinch))[:, None]
        tip_weight = (1-.55*strength)[:, None]
        dir_weight = (SEGMENT_WEIGHT_M*(1-.58*strength))[:, None, None]
        pair_contact = model.pad_radii[first] + model.pad_radii[second]
        if feedback is not None:
            requested = model.kinematics(previous)[0]
            for finger, site in self.contacts:
                deflection = (np.linalg.norm(feedback[finger,site+1]-feedback[3,4])
                              - np.linalg.norm(requested[finger,site+1]-requested[3,4]))
                preload = (1-clearance_gain[finger])*pinch[finger]**4 * (.0009 + np.clip(deflection, 0., .005))
                span[finger] -= preload
                if site == 3:
                    pair_contact[(first == finger) & (second == 3)] -= preload
        temporal_jac = TEMPORAL_WEIGHT_M_PER_RAD * np.eye(16)
        open_jac = OPEN_WEIGHT_M_PER_RAD * np.eye(16)
        if self.use_middle:
            mid_delta = local[HUMAN_MIDDLE]-self.middle_human_open
            mid_goal = self.middle_robot_open + self.middle_scale[:,None]*(self.middle_alignment @ mid_delta[:,:,None])[:,:,0]
            pair_delta = mid_delta[:3]-mid_delta[3]
            mid_pair_goal = (self.middle_robot_open[:3]-self.middle_robot_open[3]
                             + self.middle_pair_scale[:,None]*(self.middle_pair_alignment @ pair_delta[:,:,None])[:,:,0])
            mid_weight = np.r_[np.repeat(self.middle_weights[1],3), self.middle_weights[0]][:,None]
            # During one tight pinch this pair can roll when another digit curls.
            # Increase pad-contact priority continuously as opposition closes.
            mid_pair_weight = self.middle_weights[2]
            if len(self.contacts) == 1:
                posture_weight = 1. - np.max(pinch)**2
                mid_weight *= posture_weight
                mid_pair_weight *= posture_weight
        cached_q = cached_residual = cached_jac = None

        def evaluate(q):
            nonlocal cached_q, cached_residual, cached_jac
            if cached_q is not None and np.array_equal(q, cached_q):
                return cached_residual, cached_jac
            kin = model.kinematics(q, self.contact_site, self.use_middle)
            origins, directions, tip_jac, dir_jac = kin[:4]
            tips = origins[:, 4]
            jac = np.zeros((4, 3, 16))
            dir_full = np.zeros((4, 2, 3, 16))
            for finger in range(4):
                columns = slice(4*finger, 4*finger + 4)
                jac[finger, :, columns] = tip_jac[finger]
                dir_full[finger, :, :, columns] = dir_jac[finger]
            diff = tips[:3] - tips[3]
            rel_jac = jac[:3] - jac[3]
            if self.contact_site is not None:
                f,site = self.contact_site
                diff[f] = origins[f,site+1]-tips[3]
                rel_jac[f,:,4*f:4*f+4] = kin[4]
            dist = np.linalg.norm(diff, axis=1)
            normals = diff / np.maximum(dist[:, None], 1e-8)
            pairs = tips[first] - tips[second]
            pair_dist = np.linalg.norm(pairs, axis=1)
            clearance = pair_dist - pair_contact
            pair_slope = np.einsum("ij,ijk->ik", pairs/np.maximum(pair_dist[:, None], 1e-8), jac[first]-jac[second])
            # All task residuals are in metres; directions and joint regularizers have lever-arm weights.
            residuals = [tip_weight*(tips-target), dir_weight*(directions-dirs),
                         angle_weights*(q[self.angle_indices]-angle_target),
                         distance_weight*(dist-span), vector_weight*(diff-span[:, None]*axis),
                         pair_weights*(pairs-pair_goal), anchor*(q-self.open_q),
                         hold_weight*(q-hold_q),
                         PAD_CLEARANCE_WEIGHT*np.minimum(0, clearance),
                         TEMPORAL_WEIGHT_M_PER_RAD*(q-previous), OPEN_WEIGHT_M_PER_RAD*(q-self.open_q)]
            jacobians = [tip_weight[:, :, None]*jac, dir_weight[:, :, :, None]*dir_full, angle_jac,
                         distance_weight[:, None]*np.einsum("ij,ijk->ik", normals, rel_jac),
                         vector_weight[:, :, None]*rel_jac,
                         pair_weights[:, :, None]*(jac[first]-jac[second]), anchor_jac,
                         hold_jac,
                         PAD_CLEARANCE_WEIGHT*(clearance < 0)[:, None]*pair_slope, temporal_jac, open_jac]
            if observed_tip is not None and across_weight > 0.:
                residuals.append(np.atleast_1d(across_weight*((tips[3]-self.robot_open[3])@self.robot_across-across_goal)))
                jacobians.append(across_weight*(self.robot_across @ jac[3])[None, :])
            if self.use_middle:
                centers = origins[ROWS, ROBOT_MIDDLE]
                mid_jac = np.zeros((4,3,16))
                for f in range(4):
                    mid_jac[f,:,4*f:4*f+4] = kin[-1][f]
                residuals.extend([mid_weight*(centers-mid_goal), mid_pair_weight*(centers[:3]-centers[3]-mid_pair_goal)])
                jacobians.extend([mid_weight[:,:,None]*mid_jac, mid_pair_weight*(mid_jac[:3]-mid_jac[3])])
            cached_q = q.copy()
            cached_residual = np.concatenate([r.ravel() for r in residuals])
            cached_jac = np.concatenate([j.reshape(-1, 16) for j in jacobians])
            return cached_residual, cached_jac

        solved = least_squares(lambda q: evaluate(q)[0], previous, jac=lambda q: evaluate(q)[1],
                               bounds=(lo, hi), max_nfev=12, ftol=1e-5, xtol=1e-5, gtol=1e-7)
        if not np.isfinite(solved.x).all():
            raise ValueError("non-finite Allegro solution")
        return solved.x
