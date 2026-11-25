"""Suppress small, unsupported wrist excursions while the observed fingers articulate."""
from collections import deque

import numpy as np

from .shared_teleop_core import matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix, so3_log


class ArticulationWristFilter:
    def __init__(self):
        self.token = None
        self.holding = False
        self.active_until = 0.
        self.history = deque()
        self.position = self.rotation = self.wrist_patch = None
        self.moving_until = np.zeros(2)
        self.release_direction = np.zeros((2,3))

    @staticmethod
    def observed_patch(packet):
        if packet.get("wrist_support_source") != "mediapipe":
            return None
        cloud = np.asarray(packet.get("wrist_motion_support"), dtype=float)
        if cloud.shape == (9, 3) and np.isfinite(cloud[0]).all():
            return cloud[0]
        return None

    def update(self, packet, pose):
        token = pose["control_reference_token"]
        if not token or not pose["observation_valid"]:
            self.history.clear()
            self.holding = False
            self.active_until = 0.
            self.moving_until.fill(0)
            self.release_direction.fill(0)
            self.wrist_patch = None
            self.token = None
            return
        now = pose["source_stamp"]
        p = pose["position"]
        r = quaternion_xyzw_to_matrix(pose["quaternion"])
        patch = self.observed_patch(packet)
        if token != self.token or (self.history and now-self.history[-1][0] > .22):
            self.history.clear()
            self.holding = False
            self.active_until = 0.
            self.moving_until.fill(0)
            self.release_direction.fill(0)
            self.position, self.rotation, self.wrist_patch = p, r, patch
            self.token = token
        self.history.append((now, pose["finger_flexion"].copy(), p.copy(), r.copy()))
        while len(self.history) > 1 and now - self.history[0][0] > .28:
            self.history.popleft()
        old, flex, last_p, last_r = self.history[0]
        articulating = (pose["finger_tracking_valid"] and now-old > .10 and
                       np.max(np.abs(pose["finger_flexion"]-flex)) > .065)
        if articulating:
            self.active_until = now + .32
        if (now > self.active_until and not self.holding) or now-old <= .10:
            self.position, self.rotation, self.wrist_patch = p, r, patch
            return

        # A fresh wrist depth patch can confirm translations below the single-
        # observation noise envelope. Large, sustained displacements pass even
        # when the physical wrist patch is occluded by a fingertip.
        shift = p-self.position
        corroborated = False
        if self.wrist_patch is not None and patch is not None:
            witness = patch-self.wrist_patch
            corroborated = np.dot(witness,shift) > .005*max(.006, np.linalg.norm(shift))
        neutral_fingers = now > self.active_until
        fresh_rotation = packet.get("orientation_channel_valid") and not packet.get("orientation_held")
        turn = so3_log(self.rotation.T @ r) if fresh_rotation else np.zeros(3)
        limits = (.007 if neutral_fingers or corroborated else .016, np.deg2rad(5. if neutral_fingers else 11.))
        for axis, (change, threshold) in enumerate(zip((shift, turn), limits)):
            magnitude = np.linalg.norm(change)
            if magnitude > threshold:
                direction = change/magnitude
                if self.release_direction[axis] @ direction > .72 or magnitude > 2.4*threshold:
                    self.moving_until[axis] = now+.18
                self.release_direction[axis] = direction
            else:
                self.release_direction[axis].fill(0)
        if now < self.moving_until[0]:
            # Once released, a small but directionally consistent slow motion
            # must keep the channel open rather than relocking every few mm.
            travel = sum(np.linalg.norm(b[2]-a[2]) for a,b in zip(self.history,list(self.history)[1:]))
            displacement = np.linalg.norm(p-last_p)
            if displacement > .012 or (displacement > .003 and displacement > .78*travel):
                self.moving_until[0] = now+.18
            self.position, self.wrist_patch = p, patch
        if fresh_rotation and now < self.moving_until[1]:
            if np.linalg.norm(so3_log(last_r.T @ r)) > np.deg2rad(9.):
                self.moving_until[1] = now+.18
            self.rotation = r
        pose["position"] = self.position
        if fresh_rotation:
            pose["quaternion"] = matrix_to_quaternion_xyzw(self.rotation)
        self.holding = bool(fresh_rotation and now >= max(self.moving_until) and now-old >= .16)
        return self.holding
