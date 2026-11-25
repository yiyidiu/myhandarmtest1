"""Contact-conditioned equilibrium for the simulated Allegro effort servos."""
import numpy as np

from retarget import DIGITS, HUMAN


class ObjectPinch:
    def __init__(self, model, stiffness, preload_n):
        self.model = model
        self.stiffness = np.asarray(stiffness, dtype=float).reshape(4, 4)
        self.preload_n = float(preload_n)
        if not np.isfinite(self.stiffness).all() or np.any(self.stiffness <= 0) or not 0 < self.preload_n <= 2:
            raise ValueError("invalid Allegro grasp stiffness or preload")
        self.contact = (-np.inf, ())
        self.touch_times = {}
        self.partners = ()
        self.equilibrium = None
        self.human_center = None

    def observe(self, message):
        """Only a target-to-distal-pad contact may establish opposition."""
        pads = set()
        for state in message.states:
            a, b = state.collision1_name, state.collision2_name
            if a.startswith("target_object::"):
                other = b
            elif b.startswith("target_object::"):
                other = a
            else:
                continue
            for i, digit in enumerate(DIGITS):
                if other.startswith("fr3_allegro::algr_lh_" + digit + "_ds::") and "_tip_collision_" in other:
                    pads.add(i)
        stamp = message.header.stamp.to_sec()
        self.touch_times = {i: times for i, times in self.touch_times.items() if 0 <= stamp - times[1] <= .035}
        stable = []
        if 3 in pads:
            for i in sorted(pads - {3}):
                first, _ = self.touch_times.get(i, (stamp, stamp))
                self.touch_times[i] = (first, stamp)
                if stamp - first >= .025:
                    stable.append(i)
        if stable:
            self.contact = (stamp, tuple(stable))

    def approach(self, goal, previous, intended, dt):
        """Slow the approaching pads so the first touch does not knock the object away."""
        digits = [i for i, site in intended if site == 3 and i not in self.partners]
        if not digits:
            return goal
        if not self.partners:
            digits.append(3)
        delta = (goal - previous).reshape(4, 4)[digits]
        start = self.model.kinematics(previous)[0][digits, 4]
        finish = self.model.kinematics(goal)[0][digits, 4]
        pad_travel = np.linalg.norm(finish - start, axis=1)
        joint_travel = np.max(np.abs(delta), axis=1)
        duration = np.maximum(pad_travel / .025, joint_travel / .85)
        fraction = np.minimum(1., dt / np.maximum(duration, 1e-9))
        out = goal.reshape(4, 4).copy()
        out[digits] = previous.reshape(4, 4)[digits] + fraction[:, None] * delta
        return out.ravel()

    def apply(self, goal, measured, points, intended, sim_time):
        stamp, touched = self.contact
        # The physics callback may arrive one tick after this camera iteration started.
        fresh = -.03 <= sim_time - stamp <= .12
        partners = tuple(f for f in touched if (f, 3) in intended and fresh)
        if not partners:
            self.partners = ()
            return goal
        digits = list(partners) + [3]
        center = points[HUMAN[digits, -1]].mean(axis=0)
        if partners != self.partners or np.linalg.norm(center-self.human_center) > .014:
            # A blocked pad cannot reach the empty-hand pinch target. Establish
            # its equilibrium at the actual contact with a bounded spring preload.
            origins, _, jacobians, _ = self.model.kinematics(measured)
            pads = origins[:, 4]
            forces = np.zeros((4, 3))
            for finger in partners:
                inward = pads[3]-pads[finger]
                forces[finger] = self.preload_n * inward / max(np.linalg.norm(inward), 1e-8)
            forces[3] = -forces.sum(axis=0)
            torque = np.einsum("fij,fi->fj", jacobians, forces)
            self.equilibrium = np.clip(measured + (torque / self.stiffness).ravel(),
                                       self.model.lower, self.model.upper).reshape(4, 4)
            self.partners = partners
            self.human_center = center.copy()
        out = goal.reshape(4, 4).copy()
        out[digits] = self.equilibrium[digits]
        return out.ravel()
