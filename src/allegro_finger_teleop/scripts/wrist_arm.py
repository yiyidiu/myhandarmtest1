"""C-relative human wrist to Allegro palm_tcp and bounded differential IK for the FR3."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from retarget import coordinate, matrices, unit


# Operator view established by the original whole-wrist teleoperation profile.
# Optical right -> base right (-Y), image up -> base forward (+X), farther -> up (+Z).
CAMERA_TRANSLATION_TO_BASE = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., 1.]])
# In anatomical palm coordinates Z runs from the wrist towards the middle finger.
PALM_XY_REFLECTION = np.diag([1., 1., -1.])


def rotation_vector(matrix):
    constructor = getattr(Rotation, "from_matrix", None) or Rotation.from_dcm
    return constructor(matrix).as_rotvec()


def read_wrist(pose):
    position = np.asarray(pose["wrist_position_m"], dtype=float)
    rot = np.asarray(pose["palm_rotation_row_major"], dtype=float).reshape(3, 3)
    if (position.shape != (3,) or not np.isfinite(position).all() or not .12 < position[2] < 3.
            or not np.isfinite(rot).all() or not np.allclose(rot.T@rot, np.eye(3), atol=2e-4)
            or not np.isclose(np.linalg.det(rot), 1, atol=2e-4)):
        raise ValueError("腕环位姿无效")
    return position, rot


class ArmModel:
    def __init__(self, description, controller_names, control_point="palm"):
        root = ET.fromstring(description)
        by_child = {j.find("child").get("link"):j for j in root.findall("joint")}
        if control_point not in ("palm", "wrist"):
            raise ValueError("control_point must be palm or wrist")
        self.control_point = control_point
        tcp = by_child["palm_tcp"]
        if tcp.find("parent").get("link") != "algr_lh_palm" or tcp.get("type") != "fixed":
            raise ValueError("机械手安装原点到掌心的固定变换缺失")
        origin = tcp.find("origin")
        self.wrist_to_palm = (matrices(Rotation.from_euler("xyz", coordinate(origin, "rpy"))).T
                              @ coordinate(origin, "xyz"))
        chain = []
        link = "palm_tcp"
        while link != "fr3_link0":
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        chain.reverse()
        active = [j for j in chain if j.get("type") != "fixed"]
        self.names = [j.get("name") for j in active]
        if self.names != list(controller_names) or any(j.get("type") != "revolute" for j in active):
            raise ValueError("FR3 轨迹控制器与掌心运动链不匹配")
        self.axes = unit(np.array([coordinate(j.find("axis"), "xyz") for j in active]))
        self.lower, self.upper, self.speed = [np.array([float(j.find("limit").get(k)) for j in active])
                                              for k in ("lower", "upper", "velocity")]
        # Collapse any fixed transforms between active joints, including the hand mount and palm_tcp.
        self.origins, self.bases = [], []
        position, basis = np.zeros(3), np.eye(3)
        for joint in chain:
            origin = joint.find("origin")
            if origin is not None:
                position = position + basis @ coordinate(origin, "xyz")
                basis = basis @ matrices(Rotation.from_euler("xyz", coordinate(origin, "rpy")))
            if joint.get("type") != "fixed":
                self.origins.append(position)
                self.bases.append(basis)
                position, basis = np.zeros(3), np.eye(3)
        self.tip_offset, self.tip_basis = position, basis

    def fk(self, q, with_elbow=False):
        rotations = matrices(Rotation.from_rotvec(self.axes * np.asarray(q)[:, None]))
        pos, rot = np.zeros(3), np.eye(3)
        origins, axes = [], []
        for origin, basis, axis, motion in zip(self.origins, self.bases, self.axes, rotations):
            pos = pos + rot@origin
            rot = rot@basis
            origins.append(pos)
            axes.append(rot@axis)
            rot = rot@motion
        tip = pos + rot@self.tip_offset
        jac = np.vstack((np.cross(axes, tip-origins).T, np.asarray(axes).T))
        if with_elbow:
            elbow = np.zeros((3, len(q)))
            elbow[:,:3] = np.cross(axes[:3], origins[3]-np.asarray(origins[:3])).T
            return tip, rot@self.tip_basis, jac, elbow
        return tip, rot@self.tip_basis, jac


class WristArm:
    def __init__(self, model, reference, hand_to_palm, measured, at_reference=None):
        self.model = model
        self.hpos, self.hrot = read_wrist(reference)
        self.last_rotation = self.hrot
        self.basis = hand_to_palm.copy()
        self.command = np.clip(measured, model.lower, model.upper)
        self.rpos, self.rrot, _ = model.fk(measured if at_reference is None else at_reference)
        # A positive turn about the middle finger moves the human thumb edge toward
        # anatomical +X (palm normal), and the robot palm normal toward its own +Y.
        # Match their optical-depth and base-Y signs: nearer thumb -> more negative Y.
        human_depth_slope = (self.hrot @ self.basis)[2, 0]
        robot_y_slope = self.rrot[1, 1]
        # Within 3 degrees of edge-on, the requested sign is poorly observable.
        # Keep the anatomical direction there and keep this choice fixed from C to C.
        observable = min(abs(human_depth_slope), abs(robot_y_slope)) > np.sin(np.deg2rad(3.))
        self.reverse_long_axis = observable and human_depth_slope * robot_y_slope < 0.

    def target(self, position, rotation):
        p = self.rpos + CAMERA_TRANSLATION_TO_BASE @ (position-self.hpos)
        # Express C-relative wrist rotation about the anatomical palm axes, then apply it
        # in the calibrated robot palm. The translation map is an operator convention.
        delta = self.basis.T @ self.hrot.T @ rotation @ self.basis
        if self.reverse_long_axis:
            # For any axis-angle vector (x, y, z), F R.T F is the proper rotation
            # with vector (x, y, -z). This also works through a half turn.
            delta = PALM_XY_REFLECTION @ delta.T @ PALM_XY_REFLECTION
        target_rotation = self.rrot @ delta
        if self.model.control_point == "wrist":
            # IK and the published target still describe the palm TCP. Transport
            # its fixed lever arm about the hand's mounting origin instead.
            p = p + (target_rotation-self.rrot) @ self.model.wrist_to_palm
        return p, target_rotation

    def step(self, observation, measured, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid IK interval")
        goal_pos, goal_rot = self.target(*observation)
        model, prev = self.model, self.command
        cur, rot, jac, elbow = model.fk(prev, with_elbow=True)
        # Solve at the selected physical pivot. At the wrist the flange spin can
        # reorient the hand without translating the pivot or sweeping the elbow.
        if model.control_point == "wrist":
            offset = rot @ model.wrist_to_palm
            cur = cur - offset
            jac[:3] -= np.cross(jac[3:].T, offset).T
            goal_point = goal_pos - goal_rot @ model.wrist_to_palm
        else:
            goal_point = goal_pos
        error = np.r_[goal_point-cur, rotation_vector(goal_rot@rot.T)]
        # Metres per radian weights orientation against palm displacement. The damping
        # penalizes increments, so a reachable static target has no neutral-posture bias.
        weights = np.array([1., 1., 1., .18, .18, .18])
        slew = model.speed * min(dt, .12)
        lo, hi = np.maximum(model.lower-prev, -slew), np.minimum(model.upper-prev, slew)
        gain = -np.expm1(-dt/.055)
        step = gain*error
        spin_axis = jac[3:,-1]
        spin = spin_axis @ step[3:]
        preferred = np.zeros(len(prev))
        preferred[-1] = np.clip(spin, lo[-1], hi[-1])
        if model.lower[-1] < prev[-1]+spin_axis@error[3:] < model.upper[-1]:
            step[3:] += (preferred[-1]-spin)*spin_axis
        damping = .012*np.array([2.,2.,2.,1.5,1.,1.,1.])
        # Trade some elbow motion for axial travel before the flange reaches a
        # stop. The task still has access to the entire official joint interval.
        edge = 0.
        if self.model.control_point == "wrist" and spin*(prev[-1]-(model.lower[-1]+model.upper[-1])/2) > 0:
            margin = model.upper[-1]-prev[-1] if spin > 0 else prev[-1]-model.lower[-1]
            edge = np.clip(1.-margin/np.deg2rad(48.), 0., 1.)
            damping[-1] += .16*edge
            preferred[-1] *= 1.-edge
        task = np.vstack([weights[:,None]*jac, np.diag(damping), .22*(1.-.85*edge)*elbow])
        rhs = np.r_[weights*step, damping*preferred, np.zeros(3)]
        dq = lsq_linear(task, rhs, bounds=(lo, hi), method="bvls", tol=1e-9).x
        # If physics is obstructed, stop extending the target ahead of feedback.
        following = np.asarray(measured)-prev
        dq = np.clip(dq, np.minimum(0., following-.10), np.maximum(0., following+.10))
        self.command = np.clip(prev+dq, model.lower, model.upper)
        self.last_rotation = observation[1]
        return self.command.copy(), (goal_pos, goal_rot)
