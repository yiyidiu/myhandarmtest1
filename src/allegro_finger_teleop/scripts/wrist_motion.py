"""Confirm HaMeR wrist motion against independent, depth-supported palm landmarks."""
import numpy as np
from scipy.spatial.transform import Rotation

from retarget import matrices
from wrist_arm import rotation_vector


def palm_points(value):
    if not isinstance(value, (list, tuple, np.ndarray)) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return None
    if len(value) != 9:
        return None
    try:
        points = np.asarray([[np.nan]*3 if row is None else row for row in value], dtype=float)
    except (TypeError, ValueError):
        return None
    if points.shape != (9, 3):
        return None
    valid = np.isfinite(points).all(axis=1)
    depth = np.where(valid, points[:,2], 0.)
    valid &= (depth > .12) & (depth < 3.)
    return np.where(valid[:,None], points, np.nan) if valid.sum() >= 2 else None


def palm_cloud(value):
    points = palm_points(value)
    if points is None:
        return None
    good = points[np.isfinite(points[:,0])]
    if len(good) < 4 or np.linalg.svd(good-good.mean(0), compute_uv=False)[1] < .012:
        return None
    return points


def palm_rays(value):
    try:
        rays = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return rays if rays.shape == (5,2) and np.isfinite(rays).all() else None


def palm_displacement(reference, current, wrist):
    """Return the motion at the calibrated wrist, discarding inconsistent depth patches."""
    mask = np.isfinite(reference[:, 0]) & np.isfinite(current[:, 0])
    required = max(4, int(np.ceil(.6 * mask.sum())))
    while mask.sum() >= required:
        a, b = reference[mask], current[mask]
        center_a, center_b = a.mean(0), b.mean(0)
        x, y = a-center_a, b-center_b
        u, singular, vt = np.linalg.svd(x.T @ y)
        if singular[1] < .012**2:
            return None
        signs = [1., 1., np.linalg.det(vt.T @ u.T)]
        rot = (vt.T * signs) @ u.T
        residual = np.linalg.norm(y-x @ rot.T, axis=1)
        if residual.max() <= .008 and np.mean(residual**2) <= .0045**2:
            return center_b + rot @ (wrist-center_a) - wrist, rot
        mask[np.flatnonzero(mask)[np.argmax(residual)]] = False
    return None


class MotionConfirmation:
    """Accumulate slow displacement, require two agreeing samples when leaving rest."""
    def __init__(self, start_threshold, moving_threshold):
        self.start = start_threshold
        self.moving = moving_threshold
        self.last_motion = -np.inf
        self.pending = None

    def confirm(self, change, stamp):
        moving = stamp-self.last_motion < .40
        distance = np.linalg.norm(change)
        if distance < (self.moving if moving else self.start):
            self.pending = None
            return False
        direction = change/distance
        if not moving and (self.pending is None or direction @ self.pending < .75):
            self.pending = direction
            return False
        self.last_motion = stamp
        self.pending = None
        return True

    def interrupt(self):
        self.pending, self.last_motion = None, -np.inf


def bounded_difference(raw, observed, tolerance):
    """Keep the QB0 solution inside a neighbourhood of the independent observation."""
    difference = raw-observed
    return observed + difference * min(1., tolerance/max(np.linalg.norm(difference), 1.e-12))


class PalmMotionSupport:
    def __init__(self, reference, position, rotation, rays=None):
        self.reference = palm_cloud(reference)
        self.reference_wrist = position.copy()
        self.origin_p, self.origin_r = position.copy(), rotation.copy()
        self.raw_p, self.raw_r = position.copy(), rotation.copy()
        self.position, self.rotation = position.copy(), rotation.copy()
        self.seen_p, self.seen_r = np.zeros(3), np.eye(3)
        self.last_raw_p, self.last_raw_r = position.copy(), rotation.copy()
        self.last_cloud = None
        self.cloud_p, self.cloud_r = np.zeros(3), np.eye(3)
        self.translation = MotionConfirmation(.006, .002)
        self.turn = MotionConfirmation(np.deg2rad(5.), np.deg2rad(1.6))
        self.goal_p, self.goal_r = position.copy(), rotation.copy()
        self.last_stamp = None
        self.sparse_anchor = (palm_points(reference), palm_rays(rays), position.copy(), rotation.copy(), position.copy())
        self.sparse_confirm = MotionConfirmation(.006, .002)
        self.mode = "hold"
        self.available = False

    def interrupt(self):
        self.translation.interrupt()
        self.turn.interrupt()
        self.sparse_confirm.interrupt()
        self.sparse_anchor = None

    def sparse_translation(self, points, rays, position, rotation, stamp):
        # At a stationary tilt, multiple independently tracked palm sites should
        # translate together. Two sites suffice for translation but not 3-D rotation.
        anchor = self.sparse_anchor
        if anchor is None or np.linalg.norm(rotation_vector(rotation @ anchor[3].T)) > np.deg2rad(6.):
            self.sparse_anchor = (points, rays, position.copy(), rotation.copy(), self.position.copy())
            self.sparse_confirm.interrupt()
            return None
        before, old_rays, raw0, _, out0 = anchor
        raw = position-raw0
        candidate = None
        if points is not None and before is not None:
            common = np.isfinite(points[:,0]) & np.isfinite(before[:,0])
            changes = points[common]-before[common]
            if len(changes) >= 2:
                center = np.median(changes, axis=0)
                good = np.linalg.norm(changes-center, axis=1) < .0045
                # Do not count two coincident samples as independent observations.
                sites = before[common][good]
                if (good.sum() >= max(2, int(np.ceil(.6*len(changes))))
                        and np.max(np.linalg.norm(sites-sites[0], axis=1)) > .012):
                    witness = np.median(changes[good], axis=0)
                    if np.linalg.norm(raw-witness) <= .005+.24*np.linalg.norm(witness):
                        candidate = out0 + bounded_difference(raw, witness, .003)
        if candidate is None and rays is not None and old_rays is not None:
            # A 2-D palm translation certifies lateral motion only. Compensate
            # perspective with the separately observed metric wrist depth.
            displacements = rays*position[2] - old_rays*raw0[2]
            observed = np.median(displacements, axis=0)
            good = np.linalg.norm(displacements-observed, axis=1) < .004
            if good.sum() >= 3 and np.linalg.norm(raw[:2]-observed) <= .004+.20*np.linalg.norm(observed):
                candidate = self.position.copy()
                candidate[:2] = out0[:2] + bounded_difference(raw[:2], observed, .002)
        if candidate is None:
            if (points is not None and before is None) or (rays is not None and old_rays is None):
                self.sparse_anchor = (points, rays, position.copy(), rotation.copy(), self.position.copy())
            self.sparse_confirm.interrupt()
            return None
        if self.sparse_confirm.confirm(candidate-self.goal_p, stamp):
            self.goal_p = candidate
        return self.goal_p

    def advance(self, dt, turn):
        self.position = self.position + bounded_difference(self.goal_p-self.position, np.zeros(3), .25*dt)
        if turn:
            delta = rotation_vector(self.goal_r @ self.rotation.T)
            self.rotation = matrices(Rotation.from_rotvec(bounded_difference(delta,np.zeros(3),2.4*dt))) @ self.rotation
        return self.position, self.rotation

    def update(self, value, position, rotation, stamp, fresh_rotation, rays=None):
        dt = min(.065, max(.001, stamp-self.last_stamp)) if self.last_stamp is not None else .033
        self.last_stamp = stamp
        points, rays = palm_points(value), palm_rays(rays)
        cloud = palm_cloud(value)
        self.available = False
        if cloud is not None and self.reference is None:
            # C can be valid before the palm depth is readable. Acquire a local
            # witness anchor without releasing the earlier unseen wrist motion.
            self.reference, self.reference_wrist = cloud.copy(), position.copy()
            self.origin_p, self.origin_r = self.position.copy(), self.rotation.copy()
            self.raw_p, self.raw_r = position.copy(), rotation.copy()
            self.last_raw_p, self.last_raw_r = position.copy(), rotation.copy()
        motion = (palm_displacement(self.reference, cloud, self.reference_wrist)
                  if cloud is not None and self.reference is not None else None)
        if motion is None and cloud is not None and self.last_cloud is not None:
            # A large turn changes which surface of the palm the D455 sees.
            # Bridge overlapping local views, closing back onto C whenever available.
            local = palm_displacement(self.last_cloud, cloud, self.reference_wrist+self.cloud_p)
            if local is not None:
                motion = self.cloud_p+local[0], local[1] @ self.cloud_r
        if motion is None:
            self.translation.interrupt()
            self.turn.interrupt()
            if self.mode == "full":
                self.goal_p, self.goal_r = self.position.copy(), self.rotation.copy()
            self.mode = "hold"
            if self.sparse_translation(points, rays, position, rotation, stamp) is None:
                self.goal_p = self.position.copy()
                return self.position, self.rotation
            self.available, self.mode = True, "translation"
            return self.advance(dt, False)
        self.available = True
        self.last_cloud, (self.cloud_p, self.cloud_r) = cloud, motion
        observed, observed_r = motion
        # As the view turns edge-on, the measured surface moves across the palm
        # thickness. Use it to confirm motion while allowing the original
        # wrist-ring/QB0 estimate more latitude in these oblique views.
        view_turn = np.linalg.norm(rotation_vector(observed_r))
        if self.translation.confirm(observed-self.seen_p, stamp):
            if np.linalg.norm(position-self.last_raw_p) > .0008:
                delta = bounded_difference(position-self.raw_p, observed, .006 + .072*(1.-np.cos(view_turn)))
                self.goal_p = self.origin_p + delta
                self.last_raw_p = position.copy()
            self.seen_p = observed
        if fresh_rotation and self.turn.confirm(rotation_vector(observed_r @ self.seen_r.T), stamp):
            if np.linalg.norm(rotation_vector(rotation @ self.last_raw_r.T)) > np.deg2rad(.55):
                residual = rotation_vector((rotation @ self.raw_r.T) @ observed_r.T)
                corrected = bounded_difference(residual, np.zeros(3), np.deg2rad(5.) + .35*view_turn)
                self.goal_r = matrices(Rotation.from_rotvec(corrected)) @ observed_r @ self.origin_r
                self.last_raw_r = rotation.copy()
            self.seen_r = observed_r
        if not fresh_rotation:
            self.turn.interrupt()
            self.goal_r = self.rotation.copy()
        self.mode = "full"
        self.sparse_confirm.interrupt()
        self.advance(dt, fresh_rotation)
        self.sparse_anchor = (points, rays, position.copy(), rotation.copy(), self.position.copy())
        return self.position, self.rotation
