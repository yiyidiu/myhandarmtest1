"""Continuous Cartesian-to-joint reference with directional singular recovery.

This module is ROS independent so the simulation-only EGM profile can test its
kinematics and recovery state machine without starting Gazebo.  It deliberately
does not alter the established MoveIt Servo velocity profile.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np


_EPS = 1.0e-12


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must contain {} finite values".format(name, size))
    return result


def _xml_vector(text: Optional[str], default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(value) for value in text.split()], dtype=float)


def rpy_matrix(values: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = _vector(values, 3, "rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    unit = _vector(axis, 3, "axis")
    norm = float(np.linalg.norm(unit))
    if norm <= _EPS:
        raise ValueError("joint axis must be non-zero")
    x, y, z = unit / norm
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    one_minus = 1.0 - cosine
    return np.asarray([
        [cosine + x * x * one_minus,
         x * y * one_minus - z * sine,
         x * z * one_minus + y * sine],
        [y * x * one_minus + z * sine,
         cosine + y * y * one_minus,
         y * z * one_minus - x * sine],
        [z * x * one_minus - y * sine,
         z * y * one_minus + x * sine,
         cosine + z * z * one_minus],
    ], dtype=float)


@dataclass(frozen=True)
class ChainJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray


class UrdfSerialChain:
    """Minimal URDF FK/Jacobian implementation for one serial chain."""

    def __init__(self, joints: Sequence[ChainJoint], base_link: str,
                 tip_link: str):
        self.joints = tuple(joints)
        self.base_link = str(base_link)
        self.tip_link = str(tip_link)
        self.movable = tuple(
            joint for joint in self.joints
            if joint.joint_type in ("revolute", "continuous", "prismatic"))
        self.joint_names = tuple(joint.name for joint in self.movable)
        if not self.movable or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("serial chain must contain unique movable joints")

    @classmethod
    def from_urdf_xml(cls, xml_text: str, base_link: str = "base_link",
                      tip_link: str = "tool0") -> "UrdfSerialChain":
        root = ET.fromstring(xml_text)
        by_child = {}
        for element in root.findall("joint"):
            child_element = element.find("child")
            parent_element = element.find("parent")
            if child_element is None or parent_element is None:
                continue
            child = child_element.attrib["link"]
            origin = element.find("origin")
            axis = element.find("axis")
            xyz = _xml_vector(
                None if origin is None else origin.attrib.get("xyz"),
                [0.0, 0.0, 0.0])
            rpy = _xml_vector(
                None if origin is None else origin.attrib.get("rpy"),
                [0.0, 0.0, 0.0])
            axis_value = _xml_vector(
                None if axis is None else axis.attrib.get("xyz"),
                [1.0, 0.0, 0.0])
            axis_norm = float(np.linalg.norm(axis_value))
            joint_type = element.attrib["type"]
            if axis_norm <= _EPS and joint_type != "fixed":
                raise ValueError("joint {} has a zero axis".format(
                    element.attrib["name"]))
            if axis_norm <= _EPS:
                axis_value = np.asarray([1.0, 0.0, 0.0], dtype=float)
                axis_norm = 1.0
            by_child[child] = ChainJoint(
                name=element.attrib["name"],
                joint_type=joint_type,
                parent=parent_element.attrib["link"],
                child=child,
                xyz=xyz,
                rotation=rpy_matrix(rpy),
                axis=axis_value / axis_norm,
            )

        reverse_chain = []
        link = str(tip_link)
        while link != base_link:
            if link not in by_child:
                raise ValueError("no URDF chain from {} to {}".format(
                    base_link, tip_link))
            joint = by_child[link]
            reverse_chain.append(joint)
            link = joint.parent
        reverse_chain.reverse()
        return cls(reverse_chain, base_link, tip_link)

    def forward_and_jacobian(
            self, positions: Sequence[float]) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray]:
        values = _vector(positions, len(self.movable), "joint positions")
        by_name = dict(zip(self.joint_names, values))
        rotation = np.eye(3, dtype=float)
        position = np.zeros(3, dtype=float)
        columns = []
        for joint in self.joints:
            position = position + rotation @ joint.xyz
            rotation = rotation @ joint.rotation
            if joint.joint_type in ("revolute", "continuous"):
                columns.append((joint.joint_type, position.copy(),
                                rotation @ joint.axis))
                rotation = rotation @ axis_angle_matrix(
                    joint.axis, by_name[joint.name])
            elif joint.joint_type == "prismatic":
                columns.append((joint.joint_type, position.copy(),
                                rotation @ joint.axis))
                position = position + (
                    rotation @ joint.axis) * by_name[joint.name]

        end_position = position.copy()
        jacobian = np.zeros((6, len(columns)), dtype=float)
        for index, (joint_type, joint_position, axis) in enumerate(columns):
            if joint_type in ("revolute", "continuous"):
                jacobian[:3, index] = np.cross(
                    axis, end_position - joint_position)
                jacobian[3:, index] = axis
            else:
                jacobian[:3, index] = axis
        return end_position, rotation, jacobian


def jacobian_condition(jacobian: Sequence[Sequence[float]]) \
        -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must be a finite matrix")
    left, singular_values, right_t = np.linalg.svd(
        matrix, full_matrices=False)
    smallest = float(singular_values[-1])
    condition = (
        float("inf") if smallest <= _EPS else
        float(singular_values[0] / smallest))
    return condition, left, singular_values, right_t


@dataclass(frozen=True)
class SingularityResolution:
    joint_velocity: np.ndarray
    condition_number: float
    minimum_singular_value: float
    damping: float
    mode: str
    recovery_active: bool
    blocked_twist_component: float
    projected_twist: np.ndarray
    predicted_condition_number: float
    last_safe_configuration: np.ndarray


class DirectionalSingularityRecovery:
    """Damped inverse plus a bumpless, directional hard-singularity escape.

    The weakest Cartesian component that drove into a hard singularity is
    blocked, while tangential and retreat motion remains available.  If that
    motion cannot improve conditioning, the output follows the most recent
    well-conditioned joint reference.  No repeated reset or command queue is
    involved.
    """

    def __init__(
            self, chain: UrdfSerialChain,
            preferred_configuration: Sequence[float],
            lower_limits: Sequence[float], upper_limits: Sequence[float],
            maximum_velocity: Sequence[float],
            damping_start_condition: float = 60.0,
            hard_condition: float = 180.0,
            release_condition: float = 45.0,
            minimum_damping: float = 1.0e-4,
            maximum_damping: float = 0.12,
            posture_gain_per_s: float = 0.35,
            recovery_gain_per_s: float = 2.0,
            recovery_velocity_utilization: float = 0.45,
            prediction_horizon_s: float = 0.04,
            minimum_prediction_improvement: float = 0.002,
            joint_soft_margin_rad: float = 0.12,
            release_cycles: int = 8):
        self.chain = chain
        size = len(chain.joint_names)
        self.preferred = _vector(
            preferred_configuration, size, "preferred configuration")
        self.lower = _vector(lower_limits, size, "lower limits")
        self.upper = _vector(upper_limits, size, "upper limits")
        self.maximum_velocity = _vector(
            maximum_velocity, size, "maximum velocity")
        if np.any(self.lower >= self.upper) or np.any(self.maximum_velocity <= 0.0):
            raise ValueError("joint limits and maximum velocities are invalid")
        self.damping_start_condition = float(damping_start_condition)
        self.hard_condition = float(hard_condition)
        self.release_condition = float(release_condition)
        self.minimum_damping = float(minimum_damping)
        self.maximum_damping = float(maximum_damping)
        self.posture_gain_per_s = float(posture_gain_per_s)
        self.recovery_gain_per_s = float(recovery_gain_per_s)
        self.recovery_velocity_utilization = float(
            recovery_velocity_utilization)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.minimum_prediction_improvement = float(
            minimum_prediction_improvement)
        self.joint_soft_margin_rad = float(joint_soft_margin_rad)
        self.release_cycles = int(release_cycles)
        if not (1.0 < self.release_condition <
                self.damping_start_condition < self.hard_condition):
            raise ValueError("singularity conditions must have release < start < hard")
        if (not 0.0 < self.minimum_damping <= self.maximum_damping or
                self.posture_gain_per_s < 0.0 or
                self.recovery_gain_per_s <= 0.0 or
                not 0.0 < self.recovery_velocity_utilization <= 1.0 or
                self.prediction_horizon_s <= 0.0 or
                not 0.0 <= self.minimum_prediction_improvement < 1.0 or
                self.joint_soft_margin_rad <= 0.0 or
                self.release_cycles <= 0):
            raise ValueError("singularity recovery parameters are invalid")
        self.last_safe = np.clip(self.preferred, self.lower, self.upper)
        self.recovery_active = False
        self.blocked_direction: Optional[np.ndarray] = None
        self.release_counter = 0

    def reset(self, positions: Sequence[float]) -> None:
        """Clear recovery memory at a deliberately latched joint position."""

        joints = _vector(
            positions, len(self.chain.joint_names), "reset positions")
        self.last_safe = np.clip(joints, self.lower, self.upper)
        self.recovery_active = False
        self.blocked_direction = None
        self.release_counter = 0

    @staticmethod
    def _smoothstep(value: float) -> float:
        bounded = float(np.clip(value, 0.0, 1.0))
        return bounded * bounded * (3.0 - 2.0 * bounded)

    def _damping(self, condition: float) -> float:
        if condition <= self.damping_start_condition:
            return self.minimum_damping
        fraction = ((condition - self.damping_start_condition) /
                    (self.hard_condition - self.damping_start_condition))
        weight = self._smoothstep(fraction)
        return (self.minimum_damping +
                weight * (self.maximum_damping - self.minimum_damping))

    def _bounded_velocity(self, velocity: np.ndarray,
                          positions: np.ndarray) -> np.ndarray:
        result = np.asarray(velocity, dtype=float).copy()
        for index in range(len(result)):
            if result[index] < 0.0:
                distance = positions[index] - self.lower[index]
            elif result[index] > 0.0:
                distance = self.upper[index] - positions[index]
            else:
                continue
            result[index] *= float(np.clip(
                distance / self.joint_soft_margin_rad, 0.0, 1.0))
        utilization = np.max(np.abs(result) / self.maximum_velocity)
        if utilization > 1.0:
            result /= utilization
        return result

    def _condition_at(self, positions: np.ndarray) -> float:
        bounded = np.clip(positions, self.lower, self.upper)
        _, _, jacobian = self.chain.forward_and_jacobian(bounded)
        condition, _, _, _ = jacobian_condition(jacobian)
        return condition

    def resolve(self, positions: Sequence[float],
                cartesian_twist: Sequence[float]) -> SingularityResolution:
        joints = _vector(positions, len(self.chain.joint_names), "positions")
        twist = _vector(cartesian_twist, 6, "cartesian twist")
        _, _, jacobian = self.chain.forward_and_jacobian(joints)
        condition, left, singular_values, _ = jacobian_condition(jacobian)
        hard = not math.isfinite(condition) or condition >= self.hard_condition

        if not self.recovery_active and not hard:
            # A recovery target must itself be inside the release band.
            # Remembering a merely damped posture (for example condition 59
            # with a release threshold of 45) can otherwise leave zero-input
            # recovery permanently parked above the hysteresis threshold.
            if condition <= self.release_condition:
                self.last_safe = joints.copy()
        elif not self.recovery_active:
            self.recovery_active = True
            weakest = left[:, -1].copy()
            if float(np.dot(weakest, twist)) < 0.0:
                weakest *= -1.0
            self.blocked_direction = weakest
            self.release_counter = 0

        projected_twist = twist.copy()
        blocked_component = 0.0
        if self.recovery_active and self.blocked_direction is not None:
            blocked_component = float(np.dot(
                self.blocked_direction, projected_twist))
            if blocked_component > 0.0:
                projected_twist -= (
                    blocked_component * self.blocked_direction)

        damping = self._damping(condition)
        identity_task = np.eye(jacobian.shape[0], dtype=float)
        normal = jacobian @ jacobian.T + damping * damping * identity_task
        try:
            inverse = np.linalg.solve(normal, identity_task)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(normal)
        damped_inverse = jacobian.T @ inverse
        candidate = damped_inverse @ projected_twist
        nullspace = np.eye(jacobian.shape[1]) - damped_inverse @ jacobian
        candidate += nullspace @ (
            self.posture_gain_per_s * (self.preferred - joints))
        candidate = self._bounded_velocity(candidate, joints)
        # The expensive second FK/SVD is only needed while making a recovery
        # decision.  Normal 250 Hz operation uses the measured condition.
        predicted_condition = condition

        mode = "NORMAL"
        output = candidate
        if self.recovery_active:
            mode = "SINGULARITY_RECOVERY"
            predicted_condition = self._condition_at(
                joints + candidate * self.prediction_horizon_s)
            required = condition * (1.0 - self.minimum_prediction_improvement)
            candidate_improves = (
                math.isfinite(predicted_condition) and
                (not math.isfinite(condition) or predicted_condition < required))
            if not candidate_improves:
                output = self._bounded_velocity(
                    self.recovery_gain_per_s * (self.last_safe - joints), joints)
                predicted_condition = self._condition_at(
                    joints + output * self.prediction_horizon_s)

            recovery_limits = (
                self.maximum_velocity * self.recovery_velocity_utilization)
            utilization = np.max(np.abs(output) / recovery_limits)
            if utilization > 1.0:
                output /= utilization
                predicted_condition = self._condition_at(
                    joints + output * self.prediction_horizon_s)

            # Clear only after the robot is safely conditioned and the human
            # command no longer pushes into the component that caused entry.
            if (condition <= self.release_condition and
                    blocked_component <= 0.0):
                self.release_counter += 1
            else:
                self.release_counter = 0
            if self.release_counter >= self.release_cycles:
                self.recovery_active = False
                self.blocked_direction = None
                self.release_counter = 0
                self.last_safe = joints.copy()
                mode = "RECOVERY_RELEASED"
        elif condition > self.damping_start_condition:
            mode = "DAMPED"

        output = self._bounded_velocity(output, joints)
        if not np.all(np.isfinite(output)):
            output = np.zeros_like(joints)
            mode = "NONFINITE_HOLD"
        return SingularityResolution(
            joint_velocity=output.copy(),
            condition_number=float(condition),
            minimum_singular_value=float(singular_values[-1]),
            damping=float(damping),
            mode=mode,
            recovery_active=bool(self.recovery_active),
            blocked_twist_component=float(blocked_component),
            projected_twist=projected_twist.copy(),
            predicted_condition_number=float(predicted_condition),
            last_safe_configuration=self.last_safe.copy(),
        )


__all__ = [
    "ChainJoint", "DirectionalSingularityRecovery", "SingularityResolution",
    "UrdfSerialChain", "axis_angle_matrix", "jacobian_condition", "rpy_matrix",
]
