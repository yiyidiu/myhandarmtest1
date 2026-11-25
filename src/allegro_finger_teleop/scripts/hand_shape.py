"""Isolate digit articulation from a moving rigid palm, before robot retargeting."""
import numpy as np

from retarget import HUMAN, unit


PALM = [0, 5, 9, 13, 17]
CHAINS = np.vstack((HUMAN, [17, 18, 19, 20]))


class HandShape:
    def __init__(self, reference, observation=None, wrist_rotation=None):
        self.reference = reference.copy()
        self.held = reference.copy()
        self.palm = reference[PALM] - reference[PALM].mean(axis=0)
        self.lengths = np.linalg.norm(np.diff(reference[CHAINS], axis=1), axis=2)
        self.width = np.linalg.norm(reference[5]-reference[17])
        self.depth_bias = np.full((len(CHAINS), 3), np.nan)
        self.base_depth_bias = np.full(len(CHAINS), np.nan)
        self.depth_axis = np.zeros((len(CHAINS), 3))
        self.steady_axis = np.zeros(3)
        self.steady_frames = 0
        self.image_offset = np.zeros((len(CHAINS),3,3))
        self.last_image = None
        self.open_signatures = self.signatures(observation or {})
        self.signature_reference = self.open_signatures.copy() if self.open_signatures is not None else None
        self.image_required = observation is not None and "image_landmarks_px" in observation
        self.tip_evidence_frames = np.zeros(3, dtype=int)
        self.release_frames = np.zeros(3, dtype=int)
        self.surface_memory = np.zeros(3, dtype=int)
        self.depth_pair_votes = np.zeros(3, dtype=int)
        self.depth_pair_age = np.full(3, 99, dtype=int)
        self.separated_votes = np.zeros(3, dtype=int)
        self.separated_age = np.full(3, 99, dtype=int)
        self.separation_frames = np.zeros(3, dtype=int)
        self.observed_tip = np.zeros(3, dtype=bool)
        self.edge_on = False
        self.clear_views = 0
        self.hidden_partners = np.zeros(3, dtype=bool)
        self.edge_release = np.zeros(3, dtype=int)
        self.pending = np.zeros(len(CHAINS), dtype=bool)
        self.pending_displacement = np.zeros((len(CHAINS),3,3))
        self.free_observations = []
        self.thumb_witnesses = []
        self.thumb_witness_age = 0
        self.thumb_evidence = 0
        self.missing_image_frames = 0
        if self.image_required:
            view = self.image_view(np.eye(3), wrist_rotation, observation["image_landmarks_px"])
            if view is not None:
                projection, measured, _ = view
                bones = reference[CHAINS[:,1:]]-reference[CHAINS[:,:1]]
                # MediaPipe and MANO locate the same joints slightly differently.
                # Carry their measured C-frame offset with the palm as the view changes.
                self.image_offset = (measured-bones @ projection) @ projection.T
                self.last_image = np.repeat(np.asarray(observation["image_landmarks_px"])[None], len(CHAINS), axis=0)
                self.anchor_depth(reference, np.asarray(wrist_rotation).reshape(3, 3)[2],
                                  observation, np.ones(len(CHAINS), dtype=bool))

    def register(self, points):
        """Fit only wrist/MCP anchors. Never fit a rotating palm using moving fingertips."""
        palm = points[PALM] - points[PALM].mean(axis=0)
        u, s, vt = np.linalg.svd(palm.T @ self.palm)
        if s[1] < .012*s[0] or s[0] < 1e-6:
            raise ValueError("掌骨坐标退化，等待可用手形")
        u[:, -1] *= np.linalg.det(u @ vt)
        to_reference = u @ vt
        directions = unit(np.diff(points[CHAINS], axis=1)) @ to_reference
        candidate = self.reference.copy()
        candidate[CHAINS[:, 1:]] = (self.reference[CHAINS[:, :1]] +
                                     np.cumsum(self.lengths[..., None]*directions, axis=1))
        return candidate, to_reference

    def image_view(self, to_reference, wrist_rotation, pixels):
        xy = np.asarray(pixels, dtype=float)
        rotation = np.asarray(wrist_rotation, dtype=float)
        if xy.shape != (21, 2) or rotation.size != 9 or not np.isfinite(xy).all():
            return None
        view = to_reference.T @ rotation.reshape(3, 3).T
        if not np.isfinite(view).all():
            return None
        # Weak perspective over the palm. Fit pixel scale and in-plane roll from
        # the five rigid anchors, never from points whose motion is being tested.
        a = (self.palm @ view)[:, :2]
        b = xy[PALM] - xy[PALM].mean(axis=0)
        u, _, vt = np.linalg.svd(a.T @ b)
        u[:, -1] *= np.linalg.det(u @ vt)
        roll = u @ vt
        scale = np.sum((a @ roll) * b) / max(np.sum(a*a), 1e-9)
        if scale*self.width < 12:
            return None
        measured = (xy[CHAINS[:, 1:]] - xy[CHAINS[:, :1]]) / scale
        projection = view[:, :2] @ roll
        # When the MCPs overlap in the image, depth ordering of the digits is ambiguous.
        palm_axis = xy[9] - xy[0]
        side = xy[17] - xy[5]
        visible_width = abs(palm_axis[0]*side[1]-palm_axis[1]*side[0]) / max(np.linalg.norm(palm_axis), 1.)
        coverage = visible_width/(scale*self.width)
        if coverage < .40:
            return None
        return projection, measured, coverage

    def independent_digit_change(self, pixels):
        """Compare observed images after transporting the previous pose with the observed palm."""
        if self.last_image is None:
            return np.ones(len(CHAINS), dtype=bool)
        xy = np.asarray(pixels, dtype=float)
        now_palm = xy[PALM] - xy[PALM].mean(axis=0)
        now = xy[CHAINS[:,1:]] - xy[CHAINS[:,:1]]
        size = max(np.linalg.norm(xy[17]-xy[5]), .55*np.linalg.norm(xy[9]-xy[0]), 12.)
        def curvature(ray):
            # Signed image distances of the two knuckles from the base-to-tip line.
            tip = ray[-1]
            return (tip[0]*ray[:2,1]-tip[1]*ray[:2,0])/max(np.linalg.norm(tip), 1.)
        evidence = []
        for digit, previous in enumerate(self.last_image):
            old_palm = previous[PALM]-previous[PALM].mean(axis=0)
            affine = np.linalg.lstsq(old_palm, now_palm, rcond=None)[0]
            predicted = (previous[CHAINS[:,1:]]-previous[CHAINS[:,:1]]) @ affine
            delta = now-predicted
            common = np.median(delta[[0,1,2,4]], axis=0)
            error = np.linalg.norm(delta[digit], axis=1).mean()/size
            distinct = np.linalg.norm(delta[digit]-common, axis=1).mean()/size
            bend = np.max(np.abs(curvature(now[digit])-curvature(predicted[digit])))/size
            # A common shortening of straight projected fingers also occurs in wrist pitch.
            # A real closing fist supplies bend or a much larger relative contraction.
            evidence.append(error > .08 and (distinct > .13 or bend > .055 or error > .46))
        return np.array(evidence)

    def anchor_depth(self, shape, axis, observation, digits):
        depths = np.asarray(observation.get("surface_depths_m"), dtype=float)
        if depths.shape == (21,):
            measured = depths[CHAINS[:, 1:]] - depths[CHAINS[:, :1]]
            base_offsets = self.base_surface_offsets(depths, axis)
            digits = digits & np.isfinite(measured).all(axis=1) & np.isfinite(base_offsets)
            bones = shape[CHAINS[digits, 1:]] - shape[CHAINS[digits, :1]]
            self.depth_bias[digits] = measured[digits] - bones @ axis
            self.depth_axis[digits] = axis
            self.base_depth_bias[digits] = base_offsets[digits]

    def base_surface_offsets(self, depths, axis):
        bases = CHAINS[:, 0]
        metacarpals = [5, 9, 13]
        expected = (self.reference[bases]-self.reference[metacarpals].mean(axis=0)) @ axis
        return depths[bases] - depths[metacarpals].mean() - expected

    def depth_support(self, candidate, axis, depths):
        """Confirm depth-dominant, individual flexion against neighboring fingers on the D455."""
        supported = np.zeros(len(CHAINS), dtype=bool)
        depths = np.asarray(depths, dtype=float)
        if depths.shape != (21,):
            return supported
        held = self.held[CHAINS[:, 1:]] - self.held[CHAINS[:, :1]]
        observed = depths[CHAINS[:, 1:]] - depths[CHAINS[:, :1]] - self.depth_bias - held @ axis
        # Skin-depth offsets only transfer between nearby viewing directions.
        # After a large turn, the D455 may see a different side of the same bone.
        comparable = (self.depth_axis @ axis) > np.cos(np.deg2rad(10.))
        # A moving/occluded surface at the root of the chain can make all three
        # joints appear to advance together. Cross-check that root against the rigid palm.
        offsets = self.base_surface_offsets(depths, axis) - self.base_depth_bias
        comparable &= np.isfinite(offsets) & (np.nan_to_num(np.abs(offsets)) < .004)
        observed[~comparable] = np.nan
        delta = (candidate[CHAINS[:, 1:]] - self.held[CHAINS[:, 1:]]) @ axis
        for digit in range(len(CHAINS)):
            neighbors = observed[[i for i in (0, 1, 2, 4) if i != digit]]
            # Subtract common surface changes from wrist pitch or the visible side of a finger.
            common = np.ma.median(np.ma.masked_invalid(neighbors), axis=0).filled(np.nan)
            measured = observed[digit] - common
            usable = (np.isfinite(neighbors).sum(axis=0) >= 2) & np.isfinite(measured)
            if usable[-1] and usable.sum() >= 2:
                residual = measured[usable]
                agree = (np.abs(residual-delta[digit,usable]) < .45*np.abs(residual)) & (np.abs(residual) > .006)
                supported[digit] = agree.sum() >= 2 and agree[-1]
        return supported

    @staticmethod
    def signatures(observation):
        features = np.asarray(observation.get("shape_signatures"), dtype=float)
        if features.shape == (2, 5, 4) and np.isfinite(features).all():
            return features.copy()
        return None

    def intrinsic_support(self, shapes):
        """Recover a curled digit when its 2-D tip lies over the palm or other fingers."""
        if shapes is None or self.signature_reference is None:
            return np.zeros(len(CHAINS), dtype=bool)
        # A newly bound digit can carry an empty last-confirmed signature.
        # Missing witnesses supply zero motion evidence until bound again.
        mano, image = np.nan_to_num(shapes-self.signature_reference)
        # Compare changes since that digit's last confirmed pose. Both models
        # must see shortening/extension in the same direction and of similar magnitude.
        same = (mano*image > 0) & (np.abs(mano-image) < .18+.45*np.maximum(np.abs(mano),np.abs(image)))
        move = np.where(same, np.minimum(np.abs(mano),np.abs(image)), 0.)
        supported = (move[:,0] > .13) & ((move[:,1] > .21) | (move[:,2] > .21))
        # Opposition can occur with an extended IP. Observe both IP and tip
        # relative to the opposite metacarpals; motion of neighboring tips cannot trigger it.
        supported[3] |= move[3,2] > .26 and move[3,3] > .15 and mano[3,2]*mano[3,3] > 0
        return supported

    def image_support(self, candidate, to_reference, wrist_rotation, observation, signatures):
        """Does this finger shape explain the independent 2-D joints better than holding it?"""
        pixels = observation.get("image_landmarks_px")
        xy = np.asarray(pixels, dtype=float)
        intrinsic = (self.intrinsic_support(signatures) if xy.shape == (21,2) and np.isfinite(xy).all()
                     else np.zeros(len(CHAINS), dtype=bool))
        view = self.image_view(to_reference, wrist_rotation, pixels)
        if view is None:
            self.steady_frames = 0
            return intrinsic
        projection, measured, coverage = view
        held = self.held[CHAINS[:,1:]] - self.held[CHAINS[:,:1]] + self.image_offset
        # First give rigid motion a chance to explain the entire image. Fit a
        # small shared view correction about the palm, using bounded influence
        # so that one intentionally moving finger cannot rotate all the others.
        difference = measured-held @ projection
        weight = np.minimum(1., .12*self.width/np.maximum(np.linalg.norm(difference,axis=2), 1e-8))
        regularizer = 4*self.palm.T @ self.palm + (.50*self.width)**2*np.eye(3)
        gram = np.einsum("nji,nj,njk->nik",held,weight,held)
        rhs = np.einsum("nji,nj,njk->nik",held,weight,difference)
        # Leave the evaluated digit out. A bent finger protruding from the palm
        # otherwise dominates the fitted depth slope and can prevent further flexion.
        correction = np.linalg.solve(regularizer+gram.sum(axis=0)-gram, rhs.sum(axis=0)-rhs)
        bound = np.minimum(1., .40/np.maximum(np.linalg.norm(correction,axis=(1,2)),1e-8))
        projection = projection + correction*bound[:,None,None]
        def error(shape):
            bones = shape[CHAINS[:,1:]] - shape[CHAINS[:,:1]] + self.image_offset
            return np.sqrt(np.mean(np.sum((bones @ projection-measured)**2, axis=2), axis=1))
        held_error, new_error = error(self.held), error(candidate)
        # Normalize the required improvement to apparent palm width and to the
        # cross-model residual. Small apparent improvements at grazing angles
        # are common even on recordings of a rigid, open hand.
        margin = .065*self.width / min(coverage, 1.)
        rgb_support = ((held_error-new_error > margin) & (new_error < .55*held_error)
                       & self.independent_digit_change(pixels))
        if coverage < .65:
            self.steady_frames = 0
            return rgb_support | intrinsic
        axis = to_reference.T @ np.asarray(wrist_rotation).reshape(3, 3)[2]
        if self.steady_frames == 0 or self.steady_axis @ axis < np.cos(np.deg2rad(3.)):
            self.steady_axis, self.steady_frames = axis.copy(), 0
        self.steady_frames += 1
        if self.steady_frames >= 4:
            # A visible resting hand supplies a fresh surface reference after a turn.
            # Require agreement of the held shape in both the image and the 3-D fit.
            new_view = (self.depth_axis @ axis) <= np.cos(np.deg2rad(10.))
            tip_deltas = candidate[CHAINS[:,1:]] - self.held[CHAINS[:,1:]]
            steady = (np.linalg.norm(tip_deltas,axis=2).max(axis=1) < .003) & (held_error < .08*self.width)
            self.anchor_depth(self.held, axis, observation, new_view & steady)
        depth_support = self.depth_support(candidate, axis, observation.get("surface_depths_m"))
        return rgb_support | intrinsic | (depth_support & (new_error <= held_error + .10*self.width))

    def update_pinch_visibility(self, candidate, to_reference, wrist_rotation, observation):
        """Estimate whether the camera separates the index-to-little knuckle line."""
        xy = np.asarray(observation.get("image_landmarks_px"), dtype=float)
        measured = xy.shape == (21, 2) and np.isfinite(xy).all()
        if not measured:
            rotation = np.asarray(wrist_rotation, dtype=float)
            if rotation.size != 9 or not np.isfinite(rotation).all():
                return
            xy = (candidate @ to_reference.T @ rotation.reshape(3, 3).T)[:, :2]
        axis, side = xy[9] - xy[0], xy[17] - xy[5]
        length = max(np.linalg.norm(axis), 1e-9)
        width = abs(axis[0]*side[1] - axis[1]*side[0]) / length
        ratio = width / length
        # In a nearly edge-on view the 2-D detector and MANO can BOTH label a
        # visible fingertip as index. Depth sampled at that pixel sees one surface.
        if ratio < .24 or (measured and ratio < .40 and width < 6.):
            self.edge_on = True
            self.clear_views = 0
        elif measured:
            self.clear_views = self.clear_views + 1 if ratio > .36 and width >= 8. else 0
            if self.clear_views >= 2:
                self.edge_on = False
                self.hidden_partners.fill(False)
                self.edge_release.fill(0)

    @property
    def held_digits(self):
        unverified = self.hidden_partners & ~self.observed_tip
        return np.r_[unverified, unverified.any() and not self.observed_tip.any()]

    def edge_opposition(self, gap, xy):
        """Carry a verified pair through occlusion; unlock it on visible separation."""
        close = gap < .67
        if xy.shape == (21, 2) and np.isfinite(xy).all():
            size = max(np.linalg.norm(xy[9]-xy[0]), np.linalg.norm(xy[5]-xy[17]), 1.)
            image_gap = np.linalg.norm(xy[[8, 12, 16]] - xy[4], axis=1) / size
            close |= image_gap < .32
            apart = (image_gap > .46) & (gap > .70)
            self.edge_release = np.where(apart, np.minimum(self.edge_release + 1, 2), 0)
        else:
            self.edge_release[close] = 0
        self.hidden_partners |= close & ~self.observed_tip
        released = self.edge_release >= 2
        just_released = released & (self.observed_tip | self.hidden_partners)
        self.release_frames = np.where(just_released, 5, np.maximum(0, self.release_frames - 1))
        self.hidden_partners[released] = False
        self.observed_tip[released] = False
        # Never carry a speculative join across an unobservable interval.
        self.tip_evidence_frames.fill(0)
        self.surface_memory.fill(0)
        self.depth_pair_votes.fill(0)
        update = self.observed_tip | (self.release_frames > 0)
        return np.r_[update, update.any(), False]

    def opposition_support(self, candidate, observation):
        """Use digit identity in 3-D and, when visible, independent image convergence."""
        xy = np.asarray(observation.get("image_landmarks_px"), dtype=float)
        gap = np.linalg.norm(candidate[[8, 12, 16]] - candidate[4], axis=1) / self.width
        if self.edge_on:
            return self.edge_opposition(gap, xy)
        other = np.array([np.min(np.delete(gap, i)) for i in range(3)])
        # A well-separated, small 3-D gap can be observed even when the image
        # witness is momentarily absent. Never decide from a 2-D overlap alone.
        approach = (gap < .60) & (gap+.28 < other)
        tight = (gap < .32) & (gap+.25 < other)
        surface = np.zeros(3, dtype=bool)
        self.depth_pair_age = np.minimum(self.depth_pair_age + 1, 99)
        self.depth_pair_votes[self.depth_pair_age > 5] = 0
        self.separated_age = np.minimum(self.separated_age + 1, 99)
        self.separated_votes[self.separated_age > 5] = 0
        image_valid = xy.shape == (21, 2) and np.isfinite(xy).all()
        self.missing_image_frames = 0 if image_valid else self.missing_image_frames + 1
        # A briefly missing independent observation is common between detector frames.
        # Confirm release by positive separation of this pair, even when other fingers bend.
        apart = (gap > 1.20) | ((gap > .78) & (self.missing_image_frames >= 6))
        if image_valid:
            size = max(np.linalg.norm(xy[9]-xy[0]), np.linalg.norm(xy[5]-xy[17]), 1.)
            image_gap = np.linalg.norm(xy[[8,12,16]]-xy[4],axis=1)/size
            second = np.array([np.min(np.delete(image_gap, i)) for i in range(3)])
            tight |= (gap < .55) & (gap+.045 < other) & (image_gap < .22) & (image_gap+.075 < second)
            # Two aligned depth patches and separated image identities can
            # establish the pair even when MANO overestimates its depth gap.
            depths = np.asarray(observation.get("surface_depths_m"), dtype=float)
            unique = (image_gap < .22) & (image_gap + .15 < second)
            # A different, visibly separated tip cannot join a retained pair on a
            # contrary single-model depth estimate.
            tight &= (image_gap < .26) | (image_gap <= second + .06)
            if depths.shape == (21,):
                dz = np.nan_to_num(np.abs(depths[[8,12,16]] - depths[4]), nan=np.inf)
                surface = unique & (gap < 1.35) & (dz < .016)
                world = np.asarray(observation.get("independent_tip_gaps"), dtype=float)
                if world.shape == (3,) and np.isfinite(world).all() and np.all(world >= 0.):
                    # Independent 3-D digit identity plus aligned sensor depth can
                    # resolve a pinch that monocular MANO misses at a pitched wrist.
                    close_in_both_models = (gap < .32) & (world < .25) & (image_gap < .22)
                    witness = ((unique & (world < .40)) | close_in_both_models) & (dz < .018)
                    separated = (world > .57) & (image_gap > .26) & (gap > .45)
                    # During a deliberate change of partner, an old, partly open
                    # pair can hover until both models report a very wide gap.
                    # Compare its own separations against a newly visible pad.
                    next_pad = unique & (dz < .012) & ((gap < .52) | (world < .40))
                    if next_pad.any():
                        witness |= next_pad
                        nearest = np.min(image_gap[next_pad])
                        separated |= (~next_pad & (image_gap > nearest + .12) &
                                      (image_gap > .26) & (gap > .43) & (world > .34))
                    self.separated_votes[witness] = 0
                    self.separated_votes[separated] = np.minimum(3, self.separated_votes[separated] + 1)
                    self.separated_age[separated] = 0
                    self.depth_pair_votes[separated] = 0
                    self.depth_pair_votes[witness] = np.minimum(3, self.depth_pair_votes[witness] + 1)
                    self.depth_pair_age[witness] = 0
            apart = (gap > .70) & (image_gap > .37)
        # Deeply curled long fingers with a thumb resting across the knuckles
        # can resemble a loose tip pinch in projection, even on older recordings.
        long_chains = candidate[CHAINS[:3]]
        reach = np.linalg.norm(long_chains[:, -1] - long_chains[:, 0], axis=1) / self.lengths[:3].sum(axis=1)
        if np.all(reach < .65):
            apart |= (gap > .52) & ~surface
        depth_pair = (self.depth_pair_votes >= 2) & (self.depth_pair_age <= 5)
        apart &= ~depth_pair
        if depth_pair.any():
            # A neighboring MANO tip that sweeps past the thumb cannot displace
            # a measured pair. A true second contact still has its own evidence.
            tight &= depth_pair | (surface & (gap < .32))
        tight |= depth_pair
        self.surface_memory = np.where(surface, 7, np.maximum(0, self.surface_memory-1))
        tight |= (self.surface_memory > 0) & (gap < 1.35) & ~apart
        self.tip_evidence_frames = np.where(tight, np.minimum(self.tip_evidence_frames + 1, 3), 0)
        was = self.observed_tip.copy()
        self.separation_frames = np.where(apart, self.separation_frames + 1, 0)
        open_pair = np.linalg.norm(candidate[[8,12,16]] - self.reference[[8,12,16]], axis=1) < .10*self.width
        open_pair &= np.linalg.norm(candidate[4] - self.reference[4]) < .10*self.width
        lost = self.separated_tip | (self.separation_frames >= 3) | (apart & open_pair & (gap > 1.22))
        self.observed_tip = ((was & ~lost) | (self.tip_evidence_frames >= 2)) & ~lost
        # Give the independently confirmed pair enough samples to pass the
        # two-frame articulation gate on release as well as on approach.
        self.release_frames = np.where(was & ~self.observed_tip, 8, np.maximum(0, self.release_frames - 1))
        observed = approach | self.observed_tip | ((self.release_frames > 0) & (gap > .62))
        return np.r_[observed, observed.any(), False]

    @property
    def separated_tip(self):
        return (self.separated_votes >= 2) & (self.separated_age <= 5)

    def free_digit_support(self, candidate, rotation, confidence):
        """Confirm sustained local bending in a stable view, separately for each free digit."""
        rotation = np.asarray(rotation, dtype=float)
        if confidence < .65 or rotation.size != 9:
            self.free_observations.clear()
            return np.zeros(3, dtype=bool)
        rotation = rotation.reshape(3,3)
        def curl(points):
            chains = points[CHAINS[:3]]
            bones = unit(np.diff(np.concatenate((np.broadcast_to(points[0],(3,1,3)),chains),axis=1),axis=1))
            return np.arccos(np.clip((bones[:,1:]*bones[:,:-1]).sum(axis=2),-1.,1.)).sum(axis=1)
        self.free_observations.append((rotation, curl(candidate)))
        self.free_observations = self.free_observations[-3:]
        if len(self.free_observations) < 3:
            return np.zeros(3, dtype=bool)
        # Only corroborate intrinsic bending, never a change in palm view.
        if any(np.trace(rotation.T @ r) < 1+2*np.cos(np.deg2rad(8.)) for r,_ in self.free_observations):
            return np.zeros(3, dtype=bool)
        error = np.array([angles for _,angles in self.free_observations])-curl(self.held)
        consistent = (error > .42).all(axis=0) | (error < -.42).all(axis=0)
        if not self.observed_tip.any():
            # In a free hand, require the moving digit to bend independently of a neighbor.
            magnitude = abs(np.median(error,axis=0))
            independent = np.array([magnitude[i] > np.min(np.delete(magnitude,i))+.28 for i in range(3)])
            consistent &= (magnitude>.52) & independent
        return consistent & ~self.observed_tip

    def thumb_motion_support(self, signatures, rotation):
        """Corroborate isolated thumb motion in a steady oblique view."""
        self.thumb_witness_age += 1
        self.thumb_evidence = max(0, self.thumb_evidence - 1)
        view = np.asarray(rotation, dtype=float)
        if view.size != 9 or self.thumb_witness_age > 7:
            self.thumb_witnesses.clear()
            self.thumb_evidence = 0
        if view.size != 9:
            return False
        view = view.reshape(3, 3)
        recent = [(r, s) for r, s in self.thumb_witnesses
                  if np.trace(r.T @ view) > 1 + 2 * np.cos(np.deg2rad(9.))]
        if not recent:
            self.thumb_evidence = 0
        if signatures is not None:
            current = signatures[:, 3]
            for _, previous in recent:
                delta = current - previous
                common = (delta[0] * delta[1] > 0) & (np.min(abs(delta), axis=0) > [.025, .065, .085, .045])
                if (common[0] and common[1]) or (common[2] and common[3] and delta[0, 2]*delta[0, 3] > 0):
                    self.thumb_evidence = 4
                    break
            recent.append((view.copy(), current.copy()))
            self.thumb_witness_age = 0
        self.thumb_witnesses = recent[-4:]
        return self.thumb_evidence > 0

    def interrupt(self):
        self.free_observations.clear()
        self.thumb_witnesses.clear()
        self.thumb_evidence = self.missing_image_frames = 0
        self.pending.fill(False)
        self.tip_evidence_frames.fill(0)
        self.release_frames.fill(0)
        self.surface_memory.fill(0)
        self.depth_pair_votes.fill(0)
        self.depth_pair_age.fill(99)
        self.separated_votes.fill(0)
        self.separated_age.fill(99)
        self.separation_frames.fill(0)
        self.observed_tip.fill(False)
        self.hidden_partners.fill(False)
        self.edge_release.fill(0)
        self.clear_views = 0
        self.steady_frames = 0

    def update(self, points, observation, wrist_rotation=None):
        candidate, to_reference = self.register(points)
        displacement = candidate[CHAINS[:,1:]]-self.held[CHAINS[:,1:]]
        movement = np.linalg.norm(displacement, axis=2)
        changed = np.max(movement, axis=1) > .035*self.width
        if self.image_required or "image_landmarks_px" in observation:
            self.update_pinch_visibility(candidate, to_reference, wrist_rotation, observation)
            signatures = self.signatures(observation)
            if signatures is not None:
                if self.signature_reference is None:
                    self.signature_reference = np.full_like(signatures, np.nan)
                # Bind a late or recovered independent measurement only to a
                # finger that still agrees with the last accepted shape.
                missing = ~np.isfinite(self.signature_reference).all(axis=(0, 2))
                stationary = np.max(movement, axis=1) < .025*self.width
                bind = missing & stationary
                self.signature_reference[:, bind] = signatures[:, bind]
            supported = self.image_support(candidate, to_reference, wrist_rotation, observation, signatures)
            free_digits = self.free_digit_support(candidate, wrist_rotation, float(observation.get("confidence", 0.)))
            supported[:3] |= free_digits
            xy = np.asarray(observation.get("image_landmarks_px"), dtype=float)
            image_valid = xy.shape == (21,2) and np.isfinite(xy).all()
            # Both estimators can see all long digits folded even with their
            # identity order obscured or no independent signature at calibration.
            whole_fist = bool(image_valid and signatures is not None and
                              np.all(signatures[:, :3, 0] < .70) and np.all(signatures[:, :3, 1] < 1.10))
            thumb_evidence = self.thumb_motion_support(signatures if image_valid else None, wrist_rotation)
            supported[3] |= thumb_evidence and (image_valid or float(observation.get("confidence", 0.)) >= .65)
            if image_valid and signatures is not None and self.open_signatures is not None:
                # A last-confirmed pose may be partly flexed. Absolute extension
                # and closure relative to C remain observable in both estimators.
                straight = (signatures[:,:3,0] >= self.open_signatures[:,:3,0]-.055)
                extended = (signatures[:,:3,1] >= self.open_signatures[:,:3,1]-.14)
                open_digit = (straight & extended).all(axis=0)
                # Compare both current estimates to the articulated shape the
                # robot has accepted. A missing image on a previous frame can
                # otherwise leave a moderately curled free digit stuck at C.
                chains = self.held[CHAINS[:3]]
                lengths = self.lengths[:3].sum(axis=1)
                palm_size = np.linalg.norm(self.reference[PALM[1:]]-self.reference[0], axis=1).mean()
                held_features = np.column_stack((np.linalg.norm(chains[:,3]-chains[:,0],axis=1)/lengths,
                                                 np.linalg.norm(chains[:,3]-self.held[0],axis=1)/palm_size))
                delta = signatures[:, :3, :2] - self.open_signatures[:, :3, :2]
                discrepancy = delta - (held_features-self.open_signatures[0,:3,:2])
                agrees = (discrepancy[0]*discrepancy[1] > 0) & (np.min(np.abs(discrepancy),axis=0) > [.065,.14])
                supported[:3] |= agrees.all(axis=1)
                curled = ((signatures[:,:3,0] < .77) &
                          (signatures[:,:3,1] < self.open_signatures[:,:3,1]-.50))
                # Reopen each flexed finger separately. Preserve demonstrated
                # spreading through the normal directional evidence path.
                neutral = self.reference[CHAINS[:3,1:]] - self.held[CHAINS[:3,1:]]
                forward = unit(self.reference[CHAINS[:3,-1]]-self.reference[CHAINS[:3,0]])
                shortfall = np.sum(neutral[:,-1] * forward, axis=1) / self.width
                reopen = open_digit & (shortfall > .045) & ~free_digits
                supported[:3] |= reopen | curled.all(axis=0)
                whole_fist |= np.all(curled)
                if whole_fist:
                    supported[3] = True
                candidate[CHAINS[:3][reopen]] = self.reference[CHAINS[:3][reopen]]
                displacement = candidate[CHAINS[:,1:]]-self.held[CHAINS[:,1:]]
            # The stream may supply fresh, confidence-gated MANO points while
            # the independent detector misses a frame. With no current hand
            # observation at all, keep the accepted pose through the gap.
            if image_valid or float(observation.get("confidence", 0.)) >= .35:
                supported |= self.opposition_support(candidate, observation)
            else:
                self.tip_evidence_frames.fill(0)
                self.release_frames.fill(0)
            changed &= supported
            if self.edge_on:
                if whole_fist and not self.observed_tip.any():
                    self.hidden_partners.fill(False)
                changed[:4] &= ~self.held_digits
            agreement = np.sum(displacement*self.pending_displacement,axis=(1,2))
            confirmed = changed & self.pending & (agreement > 0)
            self.pending = changed
            self.pending_displacement = displacement
            changed = confirmed
            if changed.any() and self.signature_reference is not None:
                self.signature_reference[:, changed] = (signatures[:, changed] if signatures is not None else np.nan)
            if self.last_image is not None and changed.any() and image_valid:
                self.last_image[changed] = observation["image_landmarks_px"]
                # Re-anchor the cross-detector image offset at the independently
                # confirmed pose. Perspective foreshortening varies as fingers curl.
                view = self.image_view(to_reference, wrist_rotation, observation["image_landmarks_px"])
                if view is not None:
                    projection, measured, _ = view
                    bones = candidate[CHAINS[changed, 1:]] - candidate[CHAINS[changed, :1]]
                    residual = measured[changed] - (bones+self.image_offset[changed]) @ projection
                    self.image_offset[changed] += residual @ projection.T
                    axis = to_reference.T @ np.asarray(wrist_rotation).reshape(3, 3)[2]
                    self.anchor_depth(candidate, axis, observation, changed)
        self.held[CHAINS[changed]] = candidate[CHAINS[changed]]
        return self.held.copy()
