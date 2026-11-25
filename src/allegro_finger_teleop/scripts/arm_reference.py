"""Fixed-rate joint reference with feedback, braking and URDF self-clearance."""

import ctypes
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rospkg
from scipy.spatial.transform import Rotation
from retarget import matrices


class SelfClearance:
    def __init__(self, description, joint_names, workbench_world=None):
        try:
            self.lib = ctypes.CDLL("libfr3_arm_clearance.so")
        except OSError:
            package = Path(rospkg.RosPack().get_path("allegro_finger_teleop"))
            library = package.parents[1] / ".build/devel/lib/libfr3_arm_clearance.so"
            self.lib = ctypes.CDLL(str(library))
        vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(23,), flags="C_CONTIGUOUS")
        self.lib.fr3_guard_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.fr3_guard_new.restype = ctypes.c_void_p
        self.lib.fr3_guard_free.argtypes = [ctypes.c_void_p]
        self.lib.fr3_guard_free.restype = None
        self.lib.fr3_guard_allows.argtypes = [ctypes.c_void_p, vector, vector]
        self.lib.fr3_guard_allows.restype = ctypes.c_int
        self.lib.fr3_thumb_housings_allow.argtypes = [ctypes.c_void_p, vector, vector]
        self.lib.fr3_thumb_housings_allow.restype = ctypes.c_int
        self.handle = self.lib.fr3_guard_new(description.encode(), " ".join(joint_names).encode())
        if not self.handle:
            raise ValueError("无法加载臂手碰撞几何")

        if workbench_world:
            data = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
            self.lib.fr3_guard_table.argtypes = [ctypes.c_void_p, data, data]
            self.lib.fr3_guard_table.restype = ctypes.c_int
            table = ET.parse(workbench_world).find(".//model[@name='workbench']")
            if table is None:
                raise ValueError("仿真场景中缺少工作台模型")

            def transform(node):
                values = np.fromstring(node.findtext("pose", "0 0 0 0 0 0"), sep=" ")
                frame = np.eye(4)
                frame[:3, 3] = values[:3]
                frame[:3, :3] = matrices(Rotation.from_euler("xyz", values[3:]))
                return frame

            for link in table.findall("link"):
                for collision in link.findall("collision"):
                    size = np.fromstring(collision.findtext("geometry/box/size", ""), sep=" ")
                    frame = transform(table) @ transform(link) @ transform(collision)
                    if (
                        size.shape != (3,)
                        or not np.all(size > 0)
                        or not self.lib.fr3_guard_table(self.handle, np.ascontiguousarray(frame[:3]), size)
                    ):
                        raise ValueError("工作台碰撞几何无效")

    def allows(self, previous, future):
        return bool(self.handle and self.lib.fr3_guard_allows(self.handle, previous, future))

    def limit_hand(self, previous, goal):
        """Keep the thumb clear of rigid housings while the other phalanges continue."""
        # All tested links are in the palm frame, so arm angles cancel.
        start = np.r_[np.zeros(7), previous]
        def allowed(hand):
            return bool(self.lib.fr3_thumb_housings_allow(self.handle, start, np.r_[start[:7], hand]))

        if allowed(goal):
            return goal
        free = goal.copy()
        free[12:] = previous[12:]
        if not allowed(free):
            # These three base axes are the only long-finger joints that move a housing.
            free[[0, 4, 8]] = previous[[0, 4, 8]]
        if not allowed(free):
            return previous.copy()
        # Advance to the largest checked prefix from the independent-finger command.
        low, high = 0., 1.
        for _ in range(6):
            mid = (low + high) / 2.
            if allowed(free + mid * (goal - free)):
                low = mid
            else:
                high = mid
        return free + low * (goal - free)

    def close(self):
        if self.handle:
            self.lib.fr3_guard_free(self.handle)
            self.handle = None


class ArmReference:
    def __init__(self, model, clearance, acceleration):
        self.model, self.clearance = model, clearance
        self.acceleration = np.asarray(acceleration, dtype=float)
        if self.acceleration.shape != model.speed.shape or not np.all(
            np.isfinite(self.acceleration) & (self.acceleration > 0)
        ):
            raise ValueError("无效的仿真关节参考加速度")
        self.q = self.goal = None
        self.velocity = np.zeros_like(model.speed)
        self.stamp = -np.inf
        self.blocked = False

    def goal_at(self, goal, measured, stamp, reset=False):
        if self.q is None or reset:
            self.q = np.array(measured, copy=True)
            self.velocity.fill(0.0)
        self.goal, self.stamp = np.array(goal, copy=True), stamp
        self.blocked = False

    def tick(self, measured, hand, now, dt):
        if self.q is None or not 0 < dt <= 0.04:
            return None
        error = self.goal - self.q
        speed = np.minimum(self.model.speed, np.sqrt(1.5 * self.acceleration * np.abs(error)))
        wanted = (
            np.clip(error / 0.042, -speed, speed) if 0 <= now - self.stamp <= 0.22 else np.zeros_like(error)
        )
        # Reduce the reference speed as it gets ahead of measured joints. The
        # future stopping point is for collision checks, not a tracking error.
        lead = np.sign(wanted) * (self.q - measured)
        following_speed = np.sqrt(1.5 * self.acceleration * np.maximum(0.12 - lead, 0.0))
        wanted = np.clip(wanted, -following_speed, following_speed)
        speed_next = self.velocity + np.clip(
            wanted - self.velocity, -self.acceleration * dt, self.acceleration * dt
        )
        next_q = np.clip(self.q + 0.5 * (self.velocity + speed_next) * dt, self.model.lower, self.model.upper)
        # Protect the full committed motion while the velocity reference brakes.
        braking = np.clip(
            next_q + speed_next * np.abs(speed_next) / (2 * self.acceleration),
            self.model.lower,
            self.model.upper,
        )
        if np.max(np.abs(braking - measured)) > 0.00015:
            start, turn, stop = (np.r_[joints, hand] for joints in (measured, next_q, braking))
            if not self.clearance.allows(start, turn) or not self.clearance.allows(turn, stop):
                self.blocked = True
                self.velocity.fill(0.0)
                self.q = np.array(measured, copy=True)
                self.goal = self.q.copy()
                return self.q.copy()
        self.q, self.velocity = next_q, speed_next
        return self.q.copy()
