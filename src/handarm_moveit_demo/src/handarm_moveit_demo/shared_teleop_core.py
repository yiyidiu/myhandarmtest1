#!/usr/bin/env python3
"""ROS-independent math and safety logic for shared hand/arm teleoperation.

All internal orientation operations use SO(3) matrices, quaternions, or rotation
vectors. Euler angles are intentionally absent from the control path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1.0e-10
GESTURE_NONE = 0
GESTURE_OPEN = 1
GESTURE_CLOSE = 2
GESTURE_CONFIGURATION = 3
GESTURE_NAMES = {
    GESTURE_NONE: "NONE",
    GESTURE_OPEN: "OPEN",
    GESTURE_CLOSE: "CLOSE",
    GESTURE_CONFIGURATION: "CONFIGURATION",
}
REAL_ROBOT_AUTHORIZATION_TOKEN = "I_CONFIRM_REAL_ABB_IRB120"


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite {}-vector".format(name, size))
    return result.copy()


def normalize(vector: Sequence[float], name: str = "vector") -> np.ndarray:
    value = _finite_vector(vector, 3, name)
    length = float(np.linalg.norm(value))
    if length < EPS:
        raise ValueError("{} has zero length".format(name))
    return value / length


def project_to_so3(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(value)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    if not np.allclose(result.T @ result, np.eye(3), atol=1.0e-7):
        raise ValueError("rotation cannot be projected to SO(3)")
    return result


def project_to_orthogonal(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Project a linear-axis mapping to O(3), preserving possible reflection.

    Translational teleoperation directions are an operator convention rather
    than a physical frame rotation.  They may therefore intentionally contain
    one reflected axis (for example, reversing only image left/right).  SO(3)
    remains mandatory for every orientation operation.
    """
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("matrix must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(value)
    result = u @ vt
    if not np.allclose(result.T @ result, np.eye(3), atol=1.0e-7):
        raise ValueError("matrix cannot be projected to O(3)")
    if not math.isclose(abs(float(np.linalg.det(result))), 1.0,
                        rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError("orthogonal mapping must have determinant +/-1")
    return result


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = _finite_vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(q))
    if norm < EPS:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = q / norm
    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quaternion_xyzw(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    r = project_to_so3(rotation)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array([(r[2, 1]-r[1, 2])/scale,
                      (r[0, 2]-r[2, 0])/scale,
                      (r[1, 0]-r[0, 1])/scale, 0.25*scale])
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            scale = math.sqrt(1.0+r[0, 0]-r[1, 1]-r[2, 2])*2.0
            q = np.array([0.25*scale, (r[0, 1]+r[1, 0])/scale,
                          (r[0, 2]+r[2, 0])/scale, (r[2, 1]-r[1, 2])/scale])
        elif index == 1:
            scale = math.sqrt(1.0+r[1, 1]-r[0, 0]-r[2, 2])*2.0
            q = np.array([(r[0, 1]+r[1, 0])/scale, 0.25*scale,
                          (r[1, 2]+r[2, 1])/scale, (r[0, 2]-r[2, 0])/scale])
        else:
            scale = math.sqrt(1.0+r[2, 2]-r[0, 0]-r[1, 1])*2.0
            q = np.array([(r[0, 2]+r[2, 0])/scale,
                          (r[1, 2]+r[2, 1])/scale, 0.25*scale,
                          (r[1, 0]-r[0, 1])/scale])
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q *= -1.0
    return q


def skew(vector: Sequence[float]) -> np.ndarray:
    x, y, z = _finite_vector(vector, 3, "vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotation_vector: Sequence[float]) -> np.ndarray:
    value = _finite_vector(rotation_vector, 3, "rotation_vector")
    theta = float(np.linalg.norm(value))
    if theta < 1.0e-8:
        k = skew(value)
        return project_to_so3(np.eye(3) + k + 0.5 * k @ k)
    axis = value / theta
    k = skew(axis)
    return np.eye(3) + math.sin(theta)*k + (1.0-math.cos(theta))*(k @ k)


def so3_log(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    r = project_to_so3(rotation)
    cosine = float(np.clip((np.trace(r)-1.0)*0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1.0e-8:
        return 0.5 * np.array([r[2, 1]-r[1, 2],
                               r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    if math.pi - theta < 1.0e-5:
        # Stable near-pi axis extraction from the symmetric part.
        diagonal = np.maximum((np.diag(r)+1.0)*0.5, 0.0)
        axis = np.sqrt(diagonal)
        index = int(np.argmax(axis))
        if axis[index] < EPS:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            if index == 0:
                axis[1] = math.copysign(axis[1], r[0, 1]+r[1, 0])
                axis[2] = math.copysign(axis[2], r[0, 2]+r[2, 0])
            elif index == 1:
                axis[0] = math.copysign(axis[0], r[0, 1]+r[1, 0])
                axis[2] = math.copysign(axis[2], r[1, 2]+r[2, 1])
            else:
                axis[0] = math.copysign(axis[0], r[0, 2]+r[2, 0])
                axis[1] = math.copysign(axis[1], r[1, 2]+r[2, 1])
            axis /= np.linalg.norm(axis)
        return theta * axis
    return theta/(2.0*math.sin(theta)) * np.array([
        r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])


def rotation_distance(first: Sequence[Sequence[float]],
                      second: Sequence[Sequence[float]]) -> float:
    return float(np.linalg.norm(so3_log(project_to_so3(second) @
                                        project_to_so3(first).T)))


def interpolate_pose_ray(
        start_position: Sequence[float],
        start_rotation: Sequence[Sequence[float]],
        end_position: Sequence[float],
        end_rotation: Sequence[Sequence[float]],
        fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a pose along one fixed C-zero-relative SE(3) ray.

    Translation is linear in the base frame. Rotation follows the shortest
    relative SO(3) logarithm, which preserves the operator's axis and sign
    while a reachability search changes only the radial magnitude.
    """

    value = float(fraction)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("pose interpolation fraction must be in [0, 1]")
    position_a = _finite_vector(start_position, 3, "start_position")
    position_b = _finite_vector(end_position, 3, "end_position")
    rotation_a = project_to_so3(start_rotation)
    rotation_b = project_to_so3(end_rotation)
    relative_vector = so3_log(rotation_a.T @ rotation_b)
    return (position_a + value * (position_b - position_a),
            project_to_so3(rotation_a @ so3_exp(value * relative_vector)))


@dataclass(frozen=True)
class PerspectiveDecoupledIntent:
    """Hand translation expressed as screen-plane motion plus depth.

    A metric D455 point obeys ``X = ray_x * Z`` and ``Y = ray_y * Z``.
    Subtracting two metric XYZ points therefore makes a pure motion along a
    fixed camera ray look like lateral motion whenever the hand is away from
    the optical centre.  The control coordinates below keep image-plane
    displacement at the C-zero depth independent from axial depth motion.
    """

    relative_position: np.ndarray
    velocity: np.ndarray
    zero_ray: np.ndarray
    current_ray: np.ndarray


class PerspectiveIntentDecoupler:
    """Convert D455 XYZ motion into independent image-plane/depth intent."""

    def __init__(self, minimum_depth_m: float = 0.12) -> None:
        self.minimum_depth_m = float(minimum_depth_m)
        if (not math.isfinite(self.minimum_depth_m) or
                self.minimum_depth_m <= 0.0):
            raise ValueError("minimum_depth_m must be finite and positive")

    def transform(
            self, relative_position: Sequence[float],
            raw_velocity: Sequence[float],
            zero_position: Sequence[float]) -> PerspectiveDecoupledIntent:
        relative = _finite_vector(
            relative_position, 3, "relative_position")
        velocity = _finite_vector(raw_velocity, 6, "raw_velocity")
        zero = _finite_vector(zero_position, 3, "zero_position")
        current = zero + relative
        if (zero[2] < self.minimum_depth_m or
                current[2] < self.minimum_depth_m):
            raise ValueError(
                "perspective intent requires valid positive D455 depth")

        zero_ray = zero[:2] / zero[2]
        current_ray = current[:2] / current[2]
        position = np.array([
            zero[2] * (current_ray[0] - zero_ray[0]),
            zero[2] * (current_ray[1] - zero_ray[1]),
            current[2] - zero[2],
        ], dtype=np.float64)

        # d(X/Z)/dt = (Vx*Z - X*Vz)/Z^2.  Angular velocity remains in the
        # original camera spatial frame and is intentionally not changed.
        transformed_velocity = velocity.copy()
        transformed_velocity[0] = zero[2] * (
            velocity[0] * current[2] - current[0] * velocity[2]) / (
                current[2] * current[2])
        transformed_velocity[1] = zero[2] * (
            velocity[1] * current[2] - current[1] * velocity[2]) / (
                current[2] * current[2])
        transformed_velocity[2] = velocity[2]
        return PerspectiveDecoupledIntent(
            position, transformed_velocity, zero_ray.copy(),
            current_ray.copy())


@dataclass(frozen=True)
class IndependentReachabilityProjection:
    position: np.ndarray
    rotation: np.ndarray
    translation_fraction: float
    orientation_fraction: float
    active: bool
    cache_hit: bool
    ik_calls: int


class IndependentPoseReachabilityProjector:
    """Project translation to an IK boundary without consuming wrist range.

    The camera mapper intentionally keeps translation and orientation
    independent.  A position-only Monte-Carlo envelope can nevertheless
    overestimate the attainable position for a particular tool orientation.
    This projector first keeps as much requested orientation as is feasible at
    C-zero, then shortens only the translation ray.  Cached safe boundaries
    prevent noisy hand samples from repeatedly crossing the same IK edge.
    """

    def __init__(self, bisection_iterations: int = 6,
                 safety_factor: float = 0.96,
                 cache_direction_cosine: float = 0.995,
                 cache_orientation_tolerance_rad: float = math.radians(2.0),
                 cache_size: int = 96) -> None:
        self.bisection_iterations = int(bisection_iterations)
        self.safety_factor = float(safety_factor)
        self.cache_direction_cosine = float(cache_direction_cosine)
        self.cache_orientation_tolerance_rad = float(
            cache_orientation_tolerance_rad)
        self.cache_size = int(cache_size)
        if (self.bisection_iterations < 1 or
                not 0.0 < self.safety_factor <= 1.0 or
                not 0.0 < self.cache_direction_cosine <= 1.0 or
                not math.isfinite(self.cache_orientation_tolerance_rad) or
                self.cache_orientation_tolerance_rad < 0.0 or
                self.cache_size < 1):
            raise ValueError("invalid independent reachability settings")
        self.cache: List[Dict[str, object]] = []

    def reset(self) -> None:
        self.cache = []

    def _remember(self, axis: np.ndarray, rotation: np.ndarray,
                  maximum_distance: float) -> None:
        self.cache.append({
            "axis": axis.copy(),
            "rotation": rotation.copy(),
            "maximum_distance": float(maximum_distance),
        })
        if len(self.cache) > self.cache_size:
            self.cache.pop(0)

    def _cached_boundary(self, axis: np.ndarray,
                         rotation: np.ndarray) -> Optional[float]:
        best = None
        best_cosine = -1.0
        for entry in self.cache:
            cosine = float(np.dot(axis, entry["axis"]))
            orientation_error = rotation_distance(
                entry["rotation"], rotation)
            if (cosine >= self.cache_direction_cosine and
                    orientation_error <=
                    self.cache_orientation_tolerance_rad and
                    cosine > best_cosine):
                best = float(entry["maximum_distance"])
                best_cosine = cosine
        return best

    def project(
            self, zero_position: Sequence[float],
            zero_rotation: Sequence[Sequence[float]],
            target_position: Sequence[float],
            target_rotation: Sequence[Sequence[float]],
            is_reachable: Callable[[np.ndarray, np.ndarray], bool]) \
            -> IndependentReachabilityProjection:
        zero_p = _finite_vector(zero_position, 3, "zero_position")
        zero_r = project_to_so3(zero_rotation)
        target_p = _finite_vector(target_position, 3, "target_position")
        target_r = project_to_so3(target_rotation)
        if not callable(is_reachable):
            raise ValueError("is_reachable must be callable")

        calls = 0

        def query(position: np.ndarray, rotation: np.ndarray) -> bool:
            nonlocal calls
            calls += 1
            return bool(is_reachable(position.copy(), rotation.copy()))

        delta = target_p - zero_p
        distance = float(np.linalg.norm(delta))
        axis = None if distance < EPS else delta / distance

        # At a previously verified outer boundary, test the cached safe point
        # before spending one IK timeout on the known-unreachable raw target.
        # The candidate is rechecked with this frame's exact orientation.
        if axis is not None:
            cached_distance = self._cached_boundary(axis, target_r)
            if (cached_distance is not None and
                    cached_distance < distance - EPS):
                candidate_p = zero_p + cached_distance * axis
                if query(candidate_p, target_r):
                    return IndependentReachabilityProjection(
                        candidate_p, target_r,
                        cached_distance / distance, 1.0,
                        True, True, calls)

        # Normal interior motion takes one fast successful IK call and remains
        # exactly equal to the hand-mapped target.
        if query(target_p, target_r):
            return IndependentReachabilityProjection(
                target_p, target_r, 1.0, 1.0, False, False, calls)

        # Preserve the requested wrist orientation whenever it is feasible at
        # the known C-zero position.  Only if that orientation itself is
        # impossible do we shorten its SO(3) ray independently.
        orientation_fraction = 1.0
        projected_r = target_r
        if not query(zero_p, target_r):
            if not query(zero_p, zero_r):
                raise RuntimeError("C-zero pose is not IK reachable")
            low = 0.0
            high = 1.0
            for _ in range(self.bisection_iterations):
                middle = 0.5 * (low + high)
                _, candidate_r = interpolate_pose_ray(
                    zero_p, zero_r, zero_p, target_r, middle)
                if query(zero_p, candidate_r):
                    low = middle
                else:
                    high = middle
            orientation_fraction = low * self.safety_factor
            _, projected_r = interpolate_pose_ray(
                zero_p, zero_r, zero_p, target_r,
                orientation_fraction)

        if distance < EPS:
            return IndependentReachabilityProjection(
                zero_p, projected_r, 1.0, orientation_fraction,
                orientation_fraction < 1.0, False, calls)
        assert axis is not None

        # A nearby previously verified boundary is both faster and provides
        # hysteresis against static hand noise at the edge.  Verify it once
        # with the current orientation before reuse.
        cached_distance = self._cached_boundary(axis, projected_r)
        if cached_distance is not None:
            candidate_distance = min(distance, cached_distance)
            candidate_p = zero_p + candidate_distance * axis
            if query(candidate_p, projected_r):
                fraction = candidate_distance / distance
                return IndependentReachabilityProjection(
                    candidate_p, projected_r, fraction,
                    orientation_fraction,
                    fraction < 1.0 or orientation_fraction < 1.0,
                    True, calls)

        # Orientation projection may itself make the original translation
        # feasible.  Avoid an unnecessary translation shrink in that case.
        if orientation_fraction < 1.0 and query(target_p, projected_r):
            return IndependentReachabilityProjection(
                target_p, projected_r, 1.0, orientation_fraction,
                True, False, calls)

        # zero_p/projected_r is reachable by construction.  Find the last
        # reachable point on the fixed-orientation translation ray.
        low = 0.0
        high = 1.0
        for _ in range(self.bisection_iterations):
            middle = 0.5 * (low + high)
            candidate_p = zero_p + middle * delta
            if query(candidate_p, projected_r):
                low = middle
            else:
                high = middle
        translation_fraction = low * self.safety_factor
        projected_distance = distance * translation_fraction
        projected_p = zero_p + translation_fraction * delta
        self._remember(axis, projected_r, projected_distance)
        return IndependentReachabilityProjection(
            projected_p, projected_r, translation_fraction,
            orientation_fraction, True, False, calls)


@dataclass(frozen=True)
class DirectionPreservingReachabilityProjection:
    position: np.ndarray
    rotation: np.ndarray
    translation_fraction: float
    orientation_fraction: float
    active: bool
    ik_calls: int


class DirectionPreservingIncrementProjector:
    """Clip an IK target along its requested increment without side-sliding.

    The former projector shortened the complete C-zero-to-target ray.  Near a
    lateral workspace edge, changing only depth could consequently change the
    already established lateral coordinate.  This stateful projector instead
    applies only the raw target increment requested since the preceding hand
    frame, starting at the last accepted pose.  It may shorten or stop that
    increment, but it cannot rotate it, consume an older clipped residual, or
    create a Cartesian component that the mapper did not request.
    """

    def __init__(self, bisection_iterations: int = 6,
                 safety_factor: float = 0.98) -> None:
        self.bisection_iterations = int(bisection_iterations)
        self.safety_factor = float(safety_factor)
        if (self.bisection_iterations < 1 or
                not 0.0 < self.safety_factor <= 1.0):
            raise ValueError("invalid direction-preserving IK settings")
        self.reset()

    def reset(self) -> None:
        self.zero_position: Optional[np.ndarray] = None
        self.zero_rotation: Optional[np.ndarray] = None
        self.output_position: Optional[np.ndarray] = None
        self.output_rotation: Optional[np.ndarray] = None
        self.requested_position: Optional[np.ndarray] = None
        self.requested_rotation: Optional[np.ndarray] = None

    def project(
            self, zero_position: Sequence[float],
            zero_rotation: Sequence[Sequence[float]],
            target_position: Sequence[float],
            target_rotation: Sequence[Sequence[float]],
            is_reachable: Callable[[np.ndarray, np.ndarray], bool]) \
            -> DirectionPreservingReachabilityProjection:
        zero_p = _finite_vector(zero_position, 3, "zero_position")
        zero_r = project_to_so3(zero_rotation)
        target_p = _finite_vector(target_position, 3, "target_position")
        target_r = project_to_so3(target_rotation)
        if not callable(is_reachable):
            raise ValueError("is_reachable must be callable")

        if (self.zero_position is None or self.zero_rotation is None or
                np.linalg.norm(zero_p - self.zero_position) > 1.0e-9 or
                rotation_distance(zero_r, self.zero_rotation) > 1.0e-9):
            self.zero_position = zero_p.copy()
            self.zero_rotation = zero_r.copy()
            self.output_position = zero_p.copy()
            self.output_rotation = zero_r.copy()
            self.requested_position = zero_p.copy()
            self.requested_rotation = zero_r.copy()

        previous_p = self.output_position.copy()
        previous_r = self.output_rotation.copy()
        previous_requested_p = self.requested_position.copy()
        requested_translation_increment = target_p - previous_requested_p
        candidate_p = previous_p + requested_translation_increment
        # Never let an earlier clipped increment carry the command past the
        # current absolute C-zero-relative target.  This is the final guard in
        # case a caller other than AxisDecoupledWorkspaceMapper supplies the
        # projector with a stateful/clipped target stream.
        target_relative = target_p - zero_p
        candidate_relative = candidate_p - zero_p
        candidate_relative = np.minimum(
            np.maximum(candidate_relative,
                       np.minimum(np.zeros(3), target_relative)),
            np.maximum(np.zeros(3), target_relative))
        candidate_p = zero_p + candidate_relative
        requested_translation_increment = candidate_p - previous_p

        # Orientation is already an absolute pose relative to C-zero.  Using
        # the absolute target avoids accumulating a blocked SO(3) residual and
        # guarantees that a zero-snapped hand orientation requests exactly the
        # captured robot orientation.
        candidate_r = target_r.copy()
        # Record the raw request even when this increment is clipped.  The
        # next frame may only act on a newly requested component; it must not
        # silently spend an old blocked lateral residual while depth changes.
        self.requested_position = target_p.copy()
        self.requested_rotation = target_r.copy()
        calls = 0

        def query(position: np.ndarray, rotation: np.ndarray) -> bool:
            nonlocal calls
            calls += 1
            return bool(is_reachable(position.copy(), rotation.copy()))

        # C-zero is an explicit absolute contract, not an accumulated
        # increment.  Returning all six hand components to zero therefore
        # restores the captured robot pose exactly whenever it is reachable.
        returning_to_zero = bool(
            np.linalg.norm(target_p - zero_p) <= 1.0e-9 and
            rotation_distance(target_r, zero_r) <= 1.0e-9)
        if returning_to_zero and query(zero_p, zero_r):
            self.output_position = zero_p.copy()
            self.output_rotation = zero_r.copy()
            return DirectionPreservingReachabilityProjection(
                zero_p, zero_r, 1.0, 1.0, False, calls)

        if query(candidate_p, candidate_r):
            self.output_position = candidate_p.copy()
            self.output_rotation = candidate_r.copy()
            return DirectionPreservingReachabilityProjection(
                candidate_p, candidate_r, 1.0, 1.0, False, calls)

        # Preserve the last Cartesian position while finding the reachable
        # fraction of this frame's orientation increment.
        orientation_fraction = 1.0
        projected_r = candidate_r
        if not query(previous_p, candidate_r):
            if not query(previous_p, previous_r):
                raise RuntimeError("last accepted IK pose is no longer reachable")
            low = 0.0
            high = 1.0
            for _ in range(self.bisection_iterations):
                middle = 0.5 * (low + high)
                _, candidate_r = interpolate_pose_ray(
                    previous_p, previous_r,
                    previous_p, projected_r, middle)
                if query(previous_p, candidate_r):
                    low = middle
                else:
                    high = middle
            orientation_fraction = low * self.safety_factor
            _, projected_r = interpolate_pose_ray(
                previous_p, previous_r,
                previous_p, projected_r, orientation_fraction)

        if query(candidate_p, projected_r):
            self.output_position = candidate_p.copy()
            self.output_rotation = projected_r.copy()
            return DirectionPreservingReachabilityProjection(
                candidate_p, projected_r, 1.0, orientation_fraction,
                orientation_fraction < 1.0, calls)

        delta = requested_translation_increment
        distance = float(np.linalg.norm(delta))
        if distance < EPS:
            self.output_rotation = projected_r.copy()
            return DirectionPreservingReachabilityProjection(
                previous_p, projected_r, 1.0, orientation_fraction,
                orientation_fraction < 1.0, calls)

        # Bisection changes only one scalar multiplier of the requested
        # increment.  A pure base-X request therefore always keeps Y/Z fixed.
        low = 0.0
        high = 1.0
        for _ in range(self.bisection_iterations):
            middle = 0.5 * (low + high)
            candidate_p = previous_p + middle * delta
            if query(candidate_p, projected_r):
                low = middle
            else:
                high = middle
        translation_fraction = low * self.safety_factor
        projected_p = previous_p + translation_fraction * delta
        self.output_position = projected_p.copy()
        self.output_rotation = projected_r.copy()
        return DirectionPreservingReachabilityProjection(
            projected_p, projected_r, translation_fraction,
            orientation_fraction, True, calls)


@dataclass(frozen=True)
class ReachabilityProjectionContinuity:
    position: np.ndarray
    rotation: np.ndarray
    active: bool
    reason: str
    raw_step_m: float
    candidate_step_m: float
    candidate_progress_m: float
    blocked_reverse_m: float


class ReachabilityProjectionContinuityGuard:
    """Stop an IK edge projection from reversing a continuous hand command.

    Independent IK queries can return different feasible radii for adjacent
    poses, especially near a singularity or after a service timeout.  A raw
    target that advances continuously can therefore produce a projected
    target that jumps backwards.  The guard compares both targets in the same
    robot-base coordinates and holds the last accepted projected position when
    the candidate moves against the current raw-target increment.

    The guard is deliberately local and reversible: a genuine hand reversal
    reverses ``raw_delta`` as well and is accepted immediately.  Orientation
    is not filtered here; a material orientation change disables the position
    guard for that sample so IK can retreat to a newly feasible position.
    """

    def __init__(
            self, minimum_raw_step_m: float = 0.0005,
            reverse_tolerance_m: float = 0.0002,
            stationary_candidate_tolerance_m: float = 0.0015,
            orientation_tolerance_rad: float = math.radians(2.0)) -> None:
        self.minimum_raw_step_m = float(minimum_raw_step_m)
        self.reverse_tolerance_m = float(reverse_tolerance_m)
        self.stationary_candidate_tolerance_m = float(
            stationary_candidate_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        if (not math.isfinite(self.minimum_raw_step_m) or
                self.minimum_raw_step_m < 0.0 or
                not math.isfinite(self.reverse_tolerance_m) or
                self.reverse_tolerance_m < 0.0 or
                not math.isfinite(self.stationary_candidate_tolerance_m) or
                self.stationary_candidate_tolerance_m < 0.0 or
                not math.isfinite(self.orientation_tolerance_rad) or
                self.orientation_tolerance_rad < 0.0):
            raise ValueError("invalid reachability continuity settings")
        self.reset()

    def reset(self) -> None:
        self.previous_raw_position: Optional[np.ndarray] = None
        self.output_position: Optional[np.ndarray] = None
        self.output_rotation: Optional[np.ndarray] = None

    def apply(
            self, raw_position: Sequence[float],
            candidate_position: Sequence[float],
            candidate_rotation: Sequence[Sequence[float]]) \
            -> ReachabilityProjectionContinuity:
        raw = _finite_vector(raw_position, 3, "raw_position")
        candidate = _finite_vector(
            candidate_position, 3, "candidate_position")
        rotation = project_to_so3(candidate_rotation)

        if (self.previous_raw_position is None or
                self.output_position is None or
                self.output_rotation is None):
            self.previous_raw_position = raw.copy()
            self.output_position = candidate.copy()
            self.output_rotation = rotation.copy()
            return ReachabilityProjectionContinuity(
                candidate.copy(), rotation.copy(), False, "INITIALIZED",
                0.0, 0.0, 0.0, 0.0)

        raw_delta = raw - self.previous_raw_position
        candidate_delta = candidate - self.output_position
        raw_step = float(np.linalg.norm(raw_delta))
        candidate_step = float(np.linalg.norm(candidate_delta))
        orientation_step = rotation_distance(
            self.output_rotation, rotation)
        candidate_progress = 0.0
        reason = "PASS"

        # Translation intent has priority over per-frame MANO orientation
        # noise.  Otherwise a >2 degree wrist fluctuation can bypass the
        # monotonic check and recreate the very position reversal this guard
        # exists to prevent.  When translation is stationary, a material
        # orientation change remains free to select a different IK position.
        if raw_step >= self.minimum_raw_step_m:
            candidate_progress = float(np.dot(
                candidate_delta, raw_delta / raw_step))
            if candidate_progress < -self.reverse_tolerance_m:
                reason = "BLOCK_REVERSE_PROJECTION"
        elif (orientation_step <= self.orientation_tolerance_rad and
              candidate_step > self.stationary_candidate_tolerance_m):
            reason = "BLOCK_STATIONARY_IK_JUMP"

        # Direction is always measured from the immediately preceding raw
        # sample, including after a blocked candidate.  This prevents a later
        # candidate from being forward relative to an old anchor but backward
        # relative to the operator's current frame-to-frame motion.
        self.previous_raw_position = raw.copy()

        if reason.startswith("BLOCK_"):
            return ReachabilityProjectionContinuity(
                self.output_position.copy(), self.output_rotation.copy(),
                True, reason, raw_step, candidate_step, candidate_progress,
                max(0.0, -candidate_progress))

        self.output_position = candidate.copy()
        self.output_rotation = rotation.copy()
        return ReachabilityProjectionContinuity(
            candidate.copy(), rotation.copy(), False, reason, raw_step,
            candidate_step, candidate_progress, 0.0)


def rotation_between_vectors(source: Sequence[float], target: Sequence[float],
                             fallback_axis: Optional[Sequence[float]] = None) -> np.ndarray:
    a = normalize(source, "source")
    b = normalize(target, "target")
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine > 1.0e-9:
        return so3_exp(cross/sine * math.atan2(sine, cosine))
    if cosine > 0.0:
        return np.eye(3)
    if fallback_axis is not None:
        axis = np.asarray(fallback_axis, dtype=np.float64)
        axis -= np.dot(axis, a) * a
        if np.linalg.norm(axis) > 1.0e-8:
            return so3_exp(normalize(axis) * math.pi)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, a))) > 0.8:
        trial = np.array([0.0, 1.0, 0.0])
    return so3_exp(normalize(np.cross(a, trial)) * math.pi)


def closest_rotation_with_axis(current_rotation: Sequence[Sequence[float]],
                               local_axis: Sequence[float],
                               target_direction: Sequence[float]) -> np.ndarray:
    """Closest orientation mapping ``local_axis`` onto ``target_direction``."""
    current = project_to_so3(current_rotation)
    local = normalize(local_axis, "local_axis")
    desired = normalize(target_direction, "target_direction")
    current_direction = current @ local
    secondary = current @ normalize(np.roll(local, 1), "secondary")
    alignment = rotation_between_vectors(current_direction, desired, secondary)
    target = project_to_so3(alignment @ current)
    if not np.allclose(target @ local, desired, atol=1.0e-7):
        raise ValueError("axis alignment failed")
    return target


def compose_pose(position_a: Sequence[float], rotation_a: Sequence[Sequence[float]],
                 position_ab: Sequence[float], rotation_ab: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    p_a = _finite_vector(position_a, 3, "position_a")
    r_a = project_to_so3(rotation_a)
    p_ab = _finite_vector(position_ab, 3, "position_ab")
    r_ab = project_to_so3(rotation_ab)
    return p_a + r_a @ p_ab, project_to_so3(r_a @ r_ab)


def flange_pose_for_fixed_center(center_position: Sequence[float],
                                 center_rotation: Sequence[Sequence[float]],
                                 flange_to_center_position: Sequence[float],
                                 flange_to_center_rotation: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    p_center = _finite_vector(center_position, 3, "center_position")
    r_center = project_to_so3(center_rotation)
    p_fc = _finite_vector(flange_to_center_position, 3, "flange_to_center_position")
    r_fc = project_to_so3(flange_to_center_rotation)
    r_flange = project_to_so3(r_center @ r_fc.T)
    p_flange = p_center - r_flange @ p_fc
    return p_flange, r_flange


def clamp_each(value: Sequence[float], limits: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    maximum = np.asarray(limits, dtype=np.float64)
    if vector.shape != maximum.shape or np.any(maximum < 0.0):
        raise ValueError("value and non-negative limits must have matching shapes")
    return np.minimum(np.maximum(vector, -maximum), maximum)


@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    position: np.ndarray
    rotation: np.ndarray
    confidence: np.ndarray
    valid: bool = True
    gesture: int = GESTURE_NONE
    gesture_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "position", _finite_vector(self.position, 3, "position"))
        object.__setattr__(self, "rotation", project_to_so3(self.rotation))
        confidence = _finite_vector(self.confidence, 6, "confidence")
        object.__setattr__(self, "confidence", np.clip(confidence, 0.0, 1.0))
        if int(self.gesture) not in GESTURE_NAMES:
            raise ValueError("unknown gesture")
        if not math.isfinite(float(self.gesture_confidence)):
            raise ValueError("gesture confidence must be finite")


class AprilTagV3PoseContinuityFilter:
    """Port the V3 sender's step clamp and causal pose low-pass to HaMeR.

    The AprilTag V3 sender does not reject a large recovered observation and
    then jump its target to that observation.  It first limits the position and
    rotation step and then low-pass filters the limited pose.  Applying the
    same operation to the MANO wrist-ring pose keeps the perception source
    unchanged while giving the downstream target-pose receiver the same
    continuous input contract.
    """

    def __init__(self, maximum_position_step_m: float = 0.035,
                 maximum_rotation_step_rad: float = math.radians(15.0),
                 position_alpha: float = 0.30,
                 rotation_alpha: float = 0.28,
                 maximum_rotation_innovation_rad: float = math.pi,
                 maximum_position_innovation_m: float = float("inf"),
                 position_alpha_max: Optional[float] = None,
                 rotation_alpha_max: Optional[float] = None,
                 position_quiet_step_m: float = 0.0,
                 position_responsive_step_m: Optional[float] = None,
                 rotation_quiet_step_rad: float = 0.0,
                 rotation_responsive_step_rad: Optional[float] = None) -> None:
        self.maximum_position_step_m = float(maximum_position_step_m)
        self.maximum_rotation_step_rad = float(maximum_rotation_step_rad)
        self.position_alpha = float(position_alpha)
        self.rotation_alpha = float(rotation_alpha)
        self.maximum_rotation_innovation_rad = float(
            maximum_rotation_innovation_rad)
        self.maximum_position_innovation_m = float(
            maximum_position_innovation_m)
        self.position_alpha_max = float(
            position_alpha if position_alpha_max is None else
            position_alpha_max)
        self.rotation_alpha_max = float(
            rotation_alpha if rotation_alpha_max is None else
            rotation_alpha_max)
        self.position_quiet_step_m = float(position_quiet_step_m)
        self.position_responsive_step_m = float(
            position_quiet_step_m
            if position_responsive_step_m is None else
            position_responsive_step_m)
        self.rotation_quiet_step_rad = float(rotation_quiet_step_rad)
        self.rotation_responsive_step_rad = float(
            rotation_quiet_step_rad
            if rotation_responsive_step_rad is None else
            rotation_responsive_step_rad)
        if (not math.isfinite(self.maximum_position_step_m) or
                self.maximum_position_step_m <= 0.0 or
                not math.isfinite(self.maximum_rotation_step_rad) or
                self.maximum_rotation_step_rad <= 0.0 or
                not math.isfinite(self.maximum_rotation_innovation_rad) or
                self.maximum_rotation_innovation_rad <= 0.0 or
                math.isnan(self.maximum_position_innovation_m) or
                self.maximum_position_innovation_m <= 0.0 or
                not 0.0 < self.position_alpha <= 1.0 or
                not self.position_alpha <= self.position_alpha_max <= 1.0 or
                not 0.0 < self.rotation_alpha <= 1.0 or
                not self.rotation_alpha <= self.rotation_alpha_max <= 1.0 or
                not 0.0 <= self.position_quiet_step_m <=
                self.position_responsive_step_m or
                not 0.0 <= self.rotation_quiet_step_rad <=
                self.rotation_responsive_step_rad):
            raise ValueError("V3 pose-continuity limits and alphas are invalid")
        self.last_position: Optional[np.ndarray] = None
        self.last_rotation: Optional[np.ndarray] = None
        self.last_reason = ""
        self.last_position_alpha = 1.0
        self.last_rotation_alpha = 1.0

    def reset(self) -> None:
        self.last_position = None
        self.last_rotation = None
        self.last_reason = ""
        self.last_position_alpha = 1.0
        self.last_rotation_alpha = 1.0

    @staticmethod
    def _adaptive_alpha(step_size: float, minimum: float, maximum: float,
                        quiet_step: float, responsive_step: float) -> float:
        if maximum <= minimum or responsive_step <= quiet_step:
            return minimum
        value = float(np.clip(
            (step_size - quiet_step) / (responsive_step - quiet_step),
            0.0, 1.0))
        weight = value * value * (3.0 - 2.0 * value)
        return minimum + weight * (maximum - minimum)

    def update(
            self, sample: PoseSample,
            return_reference_position: Optional[Sequence[float]] = None,
            return_reference_rotation: Optional[
                Sequence[Sequence[float]]] = None) -> PoseSample:
        """Filter a pose while guaranteeing progress back to the C reference.

        Large innovations are normally held as perception outliers.  Once a C
        reference exists, however, an observation that is strictly closer to
        that reference is a safety retreat and must not be stranded by the
        same gate.  It still passes through the configured step clamp and
        adaptive low-pass, so this override cannot create a pose jump.
        """

        if self.last_position is None or self.last_rotation is None:
            if sample.valid:
                self.last_position = sample.position.copy()
                self.last_rotation = sample.rotation.copy()
                self.last_reason = ""
            return sample

        # Invalid observations must not replace the last physical MANO pose.
        # This is the HaMeR equivalent of AprilTag V3 HOLD_LAST.
        if not sample.valid:
            self.last_reason = "INPUT_INVALID"
            return PoseSample(
                sample.timestamp, self.last_position, self.last_rotation,
                sample.confidence, False, sample.gesture,
                sample.gesture_confidence)

        rotation_step = so3_log(self.last_rotation.T @ sample.rotation)
        rotation_size = float(np.linalg.norm(rotation_step))
        retreat_override = False
        # AprilTag has a unique coded orientation; monocular MANO does not.
        # A persistent palm/back ambiguity can therefore be wrong for several
        # consecutive frames.  Do not slowly integrate such a 90--180 degree
        # flip into the robot target: hold the last physical wrist pose until
        # the measurement returns to a plausible neighborhood.
        if rotation_size > self.maximum_rotation_innovation_rad:
            rotation_returning = False
            if return_reference_rotation is not None:
                reference_rotation = project_to_so3(
                    return_reference_rotation)
                previous_distance = rotation_distance(
                    reference_rotation, self.last_rotation)
                observed_distance = rotation_distance(
                    reference_rotation, sample.rotation)
                rotation_returning = bool(
                    observed_distance + 1.0e-6 < previous_distance)
            if not rotation_returning:
                self.last_reason = "ORIENTATION_INNOVATION_REJECTED"
                return PoseSample(
                    sample.timestamp, self.last_position, self.last_rotation,
                    sample.confidence, False, sample.gesture,
                    sample.gesture_confidence)
            retreat_override = True

        position_step = sample.position - self.last_position
        position_size = float(np.linalg.norm(position_step))
        if position_size > self.maximum_position_innovation_m:
            position_returning = False
            if return_reference_position is not None:
                reference_position = _finite_vector(
                    return_reference_position, 3,
                    "return_reference_position")
                previous_distance = float(np.linalg.norm(
                    self.last_position - reference_position))
                observed_distance = float(np.linalg.norm(
                    sample.position - reference_position))
                position_returning = bool(
                    observed_distance + 1.0e-6 < previous_distance)
            if not position_returning:
                self.last_reason = "POSITION_INNOVATION_REJECTED"
                return PoseSample(
                    sample.timestamp, self.last_position, self.last_rotation,
                    sample.confidence, False, sample.gesture,
                    sample.gesture_confidence)
            retreat_override = True
        self.last_position_alpha = self._adaptive_alpha(
            position_size, self.position_alpha, self.position_alpha_max,
            self.position_quiet_step_m, self.position_responsive_step_m)
        self.last_rotation_alpha = self._adaptive_alpha(
            rotation_size, self.rotation_alpha, self.rotation_alpha_max,
            self.rotation_quiet_step_rad,
            self.rotation_responsive_step_rad)
        if position_size > self.maximum_position_step_m:
            position_step *= self.maximum_position_step_m / position_size
        limited_position = self.last_position + position_step
        filtered_position = (
            self.last_position +
            self.last_position_alpha *
            (limited_position - self.last_position))

        if rotation_size > self.maximum_rotation_step_rad:
            rotation_step *= self.maximum_rotation_step_rad / rotation_size
        filtered_rotation = project_to_so3(
            self.last_rotation @ so3_exp(
                self.last_rotation_alpha * rotation_step))

        output = PoseSample(
            sample.timestamp, filtered_position, filtered_rotation,
            sample.confidence, True, sample.gesture,
            sample.gesture_confidence)
        self.last_position = output.position.copy()
        self.last_rotation = output.rotation.copy()
        self.last_reason = (
            "C_ZERO_RETREAT_OVERRIDE" if retreat_override else "")
        return output


@dataclass(frozen=True)
class CollisionRetreatResult:
    velocity: np.ndarray
    active: bool
    reason: str
    linear_retreat_allowed: bool
    angular_retreat_allowed: bool


class CollisionRetreatGuard:
    """Stop motion deeper into proximity while preserving a C-zero retreat.

    MoveIt Servo's threshold-distance checker applies one scalar to the whole
    Cartesian command and is not directional.  If teleoperation keeps pushing
    toward a self collision, the scalar can approach zero and make even the
    command back out appear frozen.  This guard stops at a useful scale and
    then only passes velocity components that reduce the measured tool-pose
    error to the robot pose captured at C.  Using measured robot motion instead
    of hand-target progress avoids a false retreat decision while the robot is
    lagging behind its target.  It never bypasses Servo collision checking.
    """

    def __init__(self, enter_scale: float = 0.20,
                 release_scale: float = 0.80,
                 translation_progress_m: float = 0.001,
                 rotation_progress_rad: float = math.radians(1.0)) -> None:
        self.enter_scale = float(enter_scale)
        self.release_scale = float(release_scale)
        self.translation_progress_m = float(translation_progress_m)
        self.rotation_progress_rad = float(rotation_progress_rad)
        if (not 0.0 < self.enter_scale < self.release_scale <= 1.0 or
                self.translation_progress_m < 0.0 or
                self.rotation_progress_rad < 0.0):
            raise ValueError("collision retreat guard parameters are invalid")
        self.reset()

    def reset(self) -> None:
        self.active = False

    def apply(self, collision_scale: float,
              current_position: Sequence[float],
              current_rotation: Sequence[Sequence[float]],
              robot_zero_position: Sequence[float],
              robot_zero_rotation: Sequence[Sequence[float]],
              velocity: Sequence[float],
              active_reason: str =
              "COLLISION_PROXIMITY_RETURN_TOWARD_C_ZERO") -> CollisionRetreatResult:
        scale = float(collision_scale)
        if not math.isfinite(scale):
            raise ValueError("collision scale must be finite")
        scale = float(np.clip(scale, 0.0, 1.0))
        position_error_from_zero = (
            _finite_vector(current_position, 3, "current_position") -
            _finite_vector(robot_zero_position, 3, "robot_zero_position"))
        current_rotation = project_to_so3(current_rotation)
        robot_zero_rotation = project_to_so3(robot_zero_rotation)
        rotation_error_to_zero = so3_log(
            robot_zero_rotation @ current_rotation.T)
        position_norm = float(np.linalg.norm(position_error_from_zero))
        rotation_norm = float(np.linalg.norm(rotation_error_to_zero))
        command = _finite_vector(velocity, 6, "velocity")

        if not self.active and scale <= self.enter_scale:
            self.active = True
        elif self.active and scale >= self.release_scale:
            self.reset()

        if not self.active:
            return CollisionRetreatResult(
                command, False, "NONE", True, True)

        linear_speed = float(np.linalg.norm(command[:3]))
        angular_speed = float(np.linalg.norm(command[3:]))
        linear_allowed = bool(
            linear_speed <= 1e-12 or
            (position_norm > self.translation_progress_m and
             np.dot(position_error_from_zero, command[:3]) < 0.0))
        angular_allowed = bool(
            angular_speed <= 1e-12 or
            (rotation_norm > self.rotation_progress_rad and
             np.dot(rotation_error_to_zero, command[3:]) > 0.0))
        if not linear_allowed:
            command[:3] = 0.0
        if not angular_allowed:
            command[3:] = 0.0
        return CollisionRetreatResult(
            command, True, str(active_reason),
            linear_allowed, angular_allowed)


@dataclass(frozen=True)
class TrendResult:
    timestamp: float
    raw_velocity: np.ndarray
    relative_position: np.ndarray
    relative_rotation: np.ndarray
    confidence: np.ndarray
    valid: bool
    reason: str
    gesture: int
    gesture_confidence: float


class SixDofTrendEstimator:
    """Multi-frame causal 6-DoF velocity estimator with jump rejection."""

    def __init__(self, window_size: int = 4,
                 translation_deadband_m: Sequence[float] = (0.0015,)*3,
                 rotation_deadband_rad: Sequence[float] = (0.015,)*3,
                 jump_translation_m: float = 0.08,
                 jump_rotation_rad: float = math.radians(45.0),
                 minimum_dt_s: float = 0.008,
                 maximum_dt_s: float = 0.20,
                 smoothing_alpha: float = 0.45,
                 reanchor_after_rejections: int = 3) -> None:
        if int(window_size) < 2:
            raise ValueError("window_size must be >= 2")
        self.samples: deque = deque(maxlen=int(window_size))
        self.translation_deadband = _finite_vector(translation_deadband_m, 3, "translation_deadband")
        self.rotation_deadband = _finite_vector(rotation_deadband_rad, 3, "rotation_deadband")
        self.jump_translation_m = float(jump_translation_m)
        self.jump_rotation_rad = float(jump_rotation_rad)
        self.minimum_dt_s = float(minimum_dt_s)
        self.maximum_dt_s = float(maximum_dt_s)
        self.smoothing_alpha = float(smoothing_alpha)
        self.reanchor_after_rejections = int(reanchor_after_rejections)
        self.zero_position: Optional[np.ndarray] = None
        self.zero_rotation: Optional[np.ndarray] = None
        self.filtered_velocity = np.zeros(6)
        self.rejection_count = 0
        self.tracking_gap_pending = False

    def reset_zero(self) -> None:
        self.samples.clear()
        self.zero_position = None
        self.zero_rotation = None
        self.filtered_velocity[:] = 0.0
        self.rejection_count = 0
        self.tracking_gap_pending = False

    @staticmethod
    def _linear_slope(times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centered = times - np.mean(times)
        denominator = float(np.dot(centered, centered))
        if denominator < EPS:
            raise ValueError("sample timestamps have no span")
        slope = centered @ values / denominator
        intercept = np.mean(values, axis=0) - slope * np.mean(times)
        residual = values - (intercept[None, :] + times[:, None]*slope[None, :])
        return slope, np.sqrt(np.mean(residual*residual, axis=0))

    def _invalid(self, sample: PoseSample, reason: str) -> TrendResult:
        relative_position = (np.zeros(3) if self.zero_position is None else
                             sample.position-self.zero_position)
        relative_rotation = (np.eye(3) if self.zero_rotation is None else
                             self.zero_rotation.T @ sample.rotation)
        self.filtered_velocity *= 1.0-self.smoothing_alpha
        return TrendResult(sample.timestamp, self.filtered_velocity.copy(),
                           relative_position, relative_rotation, np.zeros(6),
                           False, reason, sample.gesture,
                           float(sample.gesture_confidence))

    def _begin_tracking_gap(self, sample: PoseSample, reason: str) -> TrendResult:
        """Freeze immediately and discard motion that was not observed.

        This follows the robust gap policy from the supplied V8.3 observer:
        recovery rebuilds the local derivative window, but it never changes
        the operator-confirmed zero pose and never pays back motion that
        happened while tracking was invalid.
        """
        self.samples.clear()
        self.filtered_velocity[:] = 0.0
        self.rejection_count = 0
        self.tracking_gap_pending = True
        relative_position = (np.zeros(3) if self.zero_position is None else
                             sample.position-self.zero_position)
        relative_rotation = (np.eye(3) if self.zero_rotation is None else
                             self.zero_rotation.T @ sample.rotation)
        return TrendResult(sample.timestamp, np.zeros(6), relative_position,
                           relative_rotation, np.zeros(6), False, reason,
                           sample.gesture, float(sample.gesture_confidence))

    def update(self, sample: PoseSample) -> TrendResult:
        if not sample.valid:
            return self._begin_tracking_gap(sample, "INPUT_INVALID_GAP")
        if self.zero_position is None:
            self.zero_position = sample.position.copy()
            self.zero_rotation = sample.rotation.copy()
            self.samples.append(sample)
            self.tracking_gap_pending = False
            return TrendResult(sample.timestamp, np.zeros(6), np.zeros(3),
                               np.eye(3), sample.confidence.copy(), True,
                               "ZERO_INITIALIZED", sample.gesture,
                               float(sample.gesture_confidence))
        if self.tracking_gap_pending:
            self.samples.append(sample)
            self.filtered_velocity[:] = 0.0
            self.tracking_gap_pending = False
            return TrendResult(
                sample.timestamp,
                np.zeros(6),
                sample.position-self.zero_position,
                self.zero_rotation.T @ sample.rotation,
                sample.confidence.copy(),
                True,
                "TRACKING_REACQUIRED_NO_PAYBACK",
                sample.gesture,
                float(sample.gesture_confidence),
            )
        previous = self.samples[-1] if self.samples else None
        if previous is not None:
            dt = sample.timestamp-previous.timestamp
            if dt <= 0.0:
                return self._invalid(sample, "NON_INCREASING_TIMESTAMP")
            if dt > self.maximum_dt_s:
                # A timestamp gap is missing evidence, not a slow command.
                # Start the slope window at the current observation so motion
                # during the gap cannot be emitted later as a catch-up spike.
                self.samples.clear()
                self.samples.append(sample)
                self.filtered_velocity[:] = 0.0
                self.rejection_count = 0
                return TrendResult(
                    sample.timestamp,
                    np.zeros(6),
                    sample.position-self.zero_position,
                    self.zero_rotation.T @ sample.rotation,
                    sample.confidence.copy(),
                    True,
                    "TIMING_GAP_REANCHORED_NO_PAYBACK",
                    sample.gesture,
                    float(sample.gesture_confidence),
                )
            translation_jump = float(np.linalg.norm(sample.position-previous.position))
            rotation_jump = rotation_distance(previous.rotation, sample.rotation)
            if (dt < self.minimum_dt_s or
                    translation_jump > self.jump_translation_m or
                    rotation_jump > self.jump_rotation_rad):
                self.rejection_count += 1
                reason = ("TIMING_REJECTED" if dt < self.minimum_dt_s
                          else "POSE_JUMP_REJECTED")
                if self.rejection_count >= self.reanchor_after_rejections:
                    self.samples.clear()
                    self.samples.append(sample)
                    self.rejection_count = 0
                    reason += "_REANCHORED"
                return self._invalid(sample, reason)
        self.rejection_count = 0
        self.samples.append(sample)
        relative_position = sample.position-self.zero_position
        # Match the proven AprilTag V3 target-pose contract: the hand
        # increment is expressed in the C-zero hand frame (R0^T * Rnow), not
        # as a camera-frame spatial increment (Rnow * R0^T).
        relative_rotation = self.zero_rotation.T @ sample.rotation
        if len(self.samples) < 2:
            return TrendResult(sample.timestamp, np.zeros(6), relative_position,
                               relative_rotation, sample.confidence.copy(), True,
                               "WINDOW_WARMUP", sample.gesture,
                               float(sample.gesture_confidence))
        sequence = list(self.samples)
        times = np.array([entry.timestamp-sequence[0].timestamp for entry in sequence])
        positions = np.array([entry.position for entry in sequence])
        rotation_reference = sequence[0].rotation
        rotation_vectors = np.array([so3_log(entry.rotation @ rotation_reference.T)
                                     for entry in sequence])
        linear, linear_residual = self._linear_slope(times, positions)
        angular, angular_residual = self._linear_slope(times, rotation_vectors)
        total_position = positions[-1]-positions[0]
        total_rotation = rotation_vectors[-1]
        linear[np.abs(total_position) < self.translation_deadband] = 0.0
        angular[np.abs(total_rotation) < self.rotation_deadband] = 0.0
        raw = np.concatenate((linear, angular))
        self.filtered_velocity = (self.smoothing_alpha*raw +
                                  (1.0-self.smoothing_alpha)*self.filtered_velocity)
        input_confidence = np.mean(np.array([entry.confidence for entry in sequence]), axis=0)
        residual = np.concatenate((linear_residual, angular_residual))
        residual_scale = np.concatenate((np.maximum(self.translation_deadband, 1.0e-5),
                                         np.maximum(self.rotation_deadband, 1.0e-4)))
        stability = np.exp(-residual/residual_scale)
        confidence = np.clip(input_confidence*stability, 0.0, 1.0)
        return TrendResult(sample.timestamp, self.filtered_velocity.copy(),
                           relative_position, relative_rotation, confidence,
                           True, "NONE", sample.gesture,
                           float(sample.gesture_confidence))


class CoordinateVelocityMapper:
    """Map camera-frame 6-D velocity into base axes without axis suppression."""

    def __init__(self, translation_matrix: Sequence[Sequence[float]],
                 rotation_matrix: Sequence[Sequence[float]],
                 translation_gain: Sequence[float], rotation_gain: Sequence[float],
                 maximum_linear_velocity: Sequence[float],
                 maximum_angular_velocity: Sequence[float]) -> None:
        # Linear operator directions may intentionally reflect one axis.
        # Rotation-vector mappings must remain a proper SO(3) rotation.
        self.translation_matrix = project_to_orthogonal(translation_matrix)
        self.rotation_matrix = project_to_so3(rotation_matrix)
        self.gain = np.concatenate((_finite_vector(translation_gain, 3, "translation_gain"),
                                    _finite_vector(rotation_gain, 3, "rotation_gain")))
        self.limits = np.concatenate((_finite_vector(maximum_linear_velocity, 3, "maximum_linear_velocity"),
                                      _finite_vector(maximum_angular_velocity, 3, "maximum_angular_velocity")))

    def map(self, raw_velocity: Sequence[float], confidence: Sequence[float]) -> np.ndarray:
        raw = _finite_vector(raw_velocity, 6, "raw_velocity")
        confidence_value = self.map_confidence(confidence)
        mapped = np.concatenate((self.translation_matrix @ raw[:3],
                                 self.rotation_matrix @ raw[3:]))
        return clamp_each(mapped*self.gain*confidence_value, self.limits)

    def map_confidence(self, confidence: Sequence[float]) -> np.ndarray:
        value = np.clip(_finite_vector(confidence, 6, "confidence"), 0.0, 1.0)
        translation_weights = np.abs(self.translation_matrix)
        rotation_weights = np.abs(self.rotation_matrix)
        translated = translation_weights @ value[:3] / np.sum(translation_weights, axis=1)
        rotated = rotation_weights @ value[3:] / np.sum(rotation_weights, axis=1)
        return np.clip(np.concatenate((translated, rotated)), 0.0, 1.0)


@dataclass(frozen=True)
class SideAxisProjectionResult:
    """Result of an odd-symmetric local-axis side-grasp projection."""

    rotation: np.ndarray
    input_rotation_vector: np.ndarray
    projected_rotation_vector: np.ndarray
    weight: float
    active: bool
    side_sign: int


class SymmetricSideGraspProjector:
    """Suppress coupled wrist rotation while keeping local-X side grasp 1:1.

    A monocular MANO wrist pose that visually looks like a roll commonly also
    contains pitch/yaw.  Sending that complete rotation through differential
    IK can drive joint 5 to its limit in one roll direction while the opposite
    direction remains reachable.  When one configured C-zero-local rotation
    axis dominates, this projector smoothly removes only the cross-axis
    components.  The selected-axis angle is never scaled or snapped, and the
    operation is odd symmetric: projecting ``-r`` gives ``-project(r)``.
    """

    AXES = {"x": 0, "y": 1, "z": 2}

    def __init__(self, enabled: bool = True, axis: str = "x",
                 blend_start_rad: float = math.radians(30.0),
                 blend_full_rad: float = math.radians(55.0),
                 dominance_start_ratio: float = 0.90,
                 dominance_full_ratio: float = 1.15) -> None:
        axis_name = str(axis).lower()
        if axis_name not in self.AXES:
            raise ValueError("side-grasp projection axis must be x, y, or z")
        self.enabled = bool(enabled)
        self.axis_name = axis_name
        self.axis = self.AXES[axis_name]
        self.blend_start_rad = float(blend_start_rad)
        self.blend_full_rad = float(blend_full_rad)
        self.dominance_start_ratio = float(dominance_start_ratio)
        self.dominance_full_ratio = float(dominance_full_ratio)
        if (not 0.0 <= self.blend_start_rad < self.blend_full_rad or
                not 0.0 <= self.dominance_start_ratio <
                self.dominance_full_ratio or
                not all(math.isfinite(value) for value in (
                    self.blend_start_rad, self.blend_full_rad,
                    self.dominance_start_ratio,
                    self.dominance_full_ratio))):
            raise ValueError("side-grasp projection thresholds are invalid")

    @staticmethod
    def _smoothstep(value: float, lower: float, upper: float) -> float:
        normalized = float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
        return normalized * normalized * (3.0 - 2.0 * normalized)

    def project(self, relative_rotation: Sequence[Sequence[float]]) \
            -> SideAxisProjectionResult:
        rotation = project_to_so3(relative_rotation)
        input_vector = so3_log(rotation)
        axis_angle = abs(float(input_vector[self.axis]))
        cross_vector = input_vector.copy()
        cross_vector[self.axis] = 0.0
        cross_size = float(np.linalg.norm(cross_vector))
        if axis_angle <= EPS:
            dominance_ratio = 0.0
        elif cross_size <= EPS:
            dominance_ratio = float("inf")
        else:
            dominance_ratio = axis_angle / cross_size

        if not self.enabled:
            weight = 0.0
        else:
            angle_weight = self._smoothstep(
                axis_angle, self.blend_start_rad, self.blend_full_rad)
            dominance_weight = (
                1.0 if math.isinf(dominance_ratio) else self._smoothstep(
                    dominance_ratio, self.dominance_start_ratio,
                    self.dominance_full_ratio))
            weight = angle_weight * dominance_weight

        projected_vector = input_vector.copy()
        for index in range(3):
            if index != self.axis:
                projected_vector[index] *= 1.0 - weight
        side_sign = (1 if input_vector[self.axis] > EPS else
                     -1 if input_vector[self.axis] < -EPS else 0)
        return SideAxisProjectionResult(
            so3_exp(projected_vector), input_vector, projected_vector,
            float(weight), bool(weight > 1.0e-9), side_sign)

    def project_local_angular_velocity(
            self, local_angular_velocity: Sequence[float],
            projection: SideAxisProjectionResult) -> np.ndarray:
        """Apply the same cross-axis suppression to feed-forward velocity."""

        velocity = _finite_vector(
            local_angular_velocity, 3, "local_angular_velocity")
        for index in range(3):
            if index != self.axis:
                velocity[index] *= 1.0 - float(projection.weight)
        return velocity


@dataclass(frozen=True)
class StationaryFeedforwardResult:
    velocity: np.ndarray
    linear_weight: float
    angular_weight: float


class StationaryFeedforwardGate:
    """Fade noisy derivative feed-forward out near a stationary hand.

    The absolute pose target remains active, so slow intentional motion is
    still tracked by pose-error feedback.  Only the derivative term—which is
    especially sensitive to MANO frame jitter—is suppressed in the measured
    stationary-noise band and smoothly restored for purposeful motion.
    """

    def __init__(self, linear_quiet_mps: float = 0.012,
                 linear_full_mps: float = 0.040,
                 angular_quiet_radps: float = 0.18,
                 angular_full_radps: float = 0.60) -> None:
        self.linear_quiet_mps = float(linear_quiet_mps)
        self.linear_full_mps = float(linear_full_mps)
        self.angular_quiet_radps = float(angular_quiet_radps)
        self.angular_full_radps = float(angular_full_radps)
        if (not 0.0 <= self.linear_quiet_mps < self.linear_full_mps or
                not 0.0 <= self.angular_quiet_radps <
                self.angular_full_radps or
                not all(math.isfinite(value) for value in (
                    self.linear_quiet_mps, self.linear_full_mps,
                    self.angular_quiet_radps, self.angular_full_radps))):
            raise ValueError("stationary feed-forward thresholds are invalid")

    @staticmethod
    def _weight(speed: float, quiet: float, full: float) -> float:
        value = float(np.clip((speed - quiet) / (full - quiet), 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def apply(self, velocity: Sequence[float]) -> StationaryFeedforwardResult:
        result = _finite_vector(velocity, 6, "feedforward_velocity")
        linear_weight = self._weight(
            float(np.linalg.norm(result[:3])), self.linear_quiet_mps,
            self.linear_full_mps)
        angular_weight = self._weight(
            float(np.linalg.norm(result[3:])), self.angular_quiet_radps,
            self.angular_full_radps)
        result[:3] *= linear_weight
        result[3:] *= angular_weight
        return StationaryFeedforwardResult(
            result, linear_weight, angular_weight)


class RelativePoseMapper:
    """Map a C-zero-relative hand pose into a robot pose target.

    The pose composition follows ``ros_udp_target_pose_receiver_apriltag_v3``:
    translation is added in ``base_link`` while the relative orientation is
    right-multiplied onto the captured tool pose.  Thus ``R_target =
    R_robot_zero * R_delta`` and rotation axes remain local to the captured
    MANO/tool frames.  Gains are displacement/angle ratios, not servo gains.
    """

    def __init__(self, translation_matrix: Sequence[Sequence[float]],
                 rotation_matrix: Sequence[Sequence[float]],
                 translation_gain: Sequence[float],
                 rotation_gain: Sequence[float],
                 maximum_relative_translation: Sequence[float],
                 maximum_relative_rotation_rad: float) -> None:
        self.translation_matrix = project_to_orthogonal(translation_matrix)
        self.rotation_matrix = project_to_so3(rotation_matrix)
        self.translation_gain = _finite_vector(
            translation_gain, 3, "translation_gain")
        self.rotation_gain = _finite_vector(
            rotation_gain, 3, "rotation_gain")
        self.maximum_relative_translation = _finite_vector(
            maximum_relative_translation, 3,
            "maximum_relative_translation")
        self.maximum_relative_rotation_rad = float(
            maximum_relative_rotation_rad)
        if (np.any(self.translation_gain < 0.0) or
                np.any(self.rotation_gain < 0.0) or
                np.any(self.maximum_relative_translation <= 0.0) or
                not math.isfinite(self.maximum_relative_rotation_rad) or
                self.maximum_relative_rotation_rad <= 0.0):
            raise ValueError("relative-pose gains and limits must be positive")

    def map(self, relative_hand_position: Sequence[float],
            relative_hand_rotation: Sequence[Sequence[float]],
            robot_zero_position: Sequence[float],
            robot_zero_rotation: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
        hand_position = _finite_vector(
            relative_hand_position, 3, "relative_hand_position")
        zero_position = _finite_vector(
            robot_zero_position, 3, "robot_zero_position")
        zero_rotation = project_to_so3(robot_zero_rotation)
        mapped_translation = (
            self.translation_matrix @ hand_position) * self.translation_gain
        mapped_translation = clamp_each(
            mapped_translation, self.maximum_relative_translation)

        hand_rotation_vector = so3_log(relative_hand_rotation)
        mapped_rotation_vector = (
            self.rotation_matrix @ hand_rotation_vector) * self.rotation_gain
        rotation_size = float(np.linalg.norm(mapped_rotation_vector))
        if rotation_size > self.maximum_relative_rotation_rad:
            mapped_rotation_vector *= (
                self.maximum_relative_rotation_rad / rotation_size)
        target_position = zero_position + mapped_translation
        target_rotation = zero_rotation @ so3_exp(mapped_rotation_vector)
        return target_position, project_to_so3(target_rotation)

    def map_target_velocity(
            self, raw_hand_velocity: Sequence[float],
            hand_zero_rotation: Sequence[Sequence[float]],
            robot_zero_rotation: Sequence[Sequence[float]],
            relative_hand_position: Optional[Sequence[float]] = None,
            robot_zero_position: Optional[Sequence[float]] = None) -> np.ndarray:
        """Map measured hand motion to the target-pose feed-forward twist.

        Linear motion is expressed in fixed camera axes.  Angular motion from
        the trend estimator is a camera-frame spatial angular velocity; it is
        first expressed in the captured hand-zero frame, mapped to tool-local
        axes, then expressed in ``base_link`` for MoveIt Servo.
        """

        raw = _finite_vector(raw_hand_velocity, 6, "raw_hand_velocity")
        hand_zero = project_to_so3(hand_zero_rotation)
        robot_zero = project_to_so3(robot_zero_rotation)
        linear = (
            self.translation_matrix @ raw[:3]) * self.translation_gain
        angular_hand_zero = hand_zero.T @ raw[3:]
        angular_tool_zero = (
            self.rotation_matrix @ angular_hand_zero) * self.rotation_gain
        angular_base = robot_zero @ angular_tool_zero
        return np.concatenate((linear, angular_base))

    def mapping_diagnostics(self) -> Dict[str, object]:
        return {
            "mode": "LINEAR_RELATIVE_POSE",
            "human_translation_fraction": None,
            "human_boundary_distance_m": None,
            "robot_boundary_distance_m": None,
            "translation_saturated": False,
        }

    def map_confidence(self, confidence: Sequence[float]) -> np.ndarray:
        value = np.clip(_finite_vector(confidence, 6, "confidence"), 0.0, 1.0)
        translation_weights = np.abs(self.translation_matrix)
        rotation_weights = np.abs(self.rotation_matrix)
        translated = translation_weights @ value[:3] / np.sum(
            translation_weights, axis=1)
        rotated = rotation_weights @ value[3:] / np.sum(
            rotation_weights, axis=1)
        return np.clip(np.concatenate((translated, rotated)), 0.0, 1.0)


class GroundSectorWorkspace:
    """Convex front/ground-clipped ellipsoid in the robot base frame.

    The raw IRB120 position cloud is close to an ellipsoid but the operational
    teleoperation volume deliberately keeps only the front side and everything
    above a configured tool-height plane.  ``utilization`` and
    ``boundary_margin_m`` shrink the raw Monte-Carlo envelope before it is
    exposed to the operator.  This class is a positional envelope; robot IK,
    self collision, and orientation feasibility remain separate checks.
    """

    def __init__(self, center_base_m: Sequence[float], radii_m: Sequence[float],
                 minimum_forward_x_m: float, minimum_tool_z_m: float,
                 utilization: float = 1.0,
                 boundary_margin_m: float = 0.0) -> None:
        self.center = _finite_vector(center_base_m, 3, "workspace_center")
        raw_radii = _finite_vector(radii_m, 3, "workspace_radii")
        self.minimum_forward_x_m = float(minimum_forward_x_m)
        self.minimum_tool_z_m = float(minimum_tool_z_m)
        self.utilization = float(utilization)
        self.boundary_margin_m = float(boundary_margin_m)
        if (np.any(raw_radii <= 0.0) or
                not 0.0 < self.utilization <= 1.0 or
                not math.isfinite(self.boundary_margin_m) or
                self.boundary_margin_m < 0.0 or
                not math.isfinite(self.minimum_forward_x_m) or
                not math.isfinite(self.minimum_tool_z_m)):
            raise ValueError("invalid ground-sector workspace parameters")
        self.radii = raw_radii * self.utilization - self.boundary_margin_m
        if np.any(self.radii <= 0.0):
            raise ValueError("workspace margin consumes an ellipsoid radius")
        self.minimum_forward_x_m += self.boundary_margin_m
        self.minimum_tool_z_m += self.boundary_margin_m
        if (self.minimum_forward_x_m >= self.center[0] + self.radii[0] or
                self.minimum_tool_z_m >= self.center[2] + self.radii[2]):
            raise ValueError("workspace clipping planes remove the ellipsoid")

    def ellipsoid_value(self, position: Sequence[float]) -> float:
        point = _finite_vector(position, 3, "workspace_position")
        return float(np.sum(((point - self.center) / self.radii) ** 2))

    def contains(self, position: Sequence[float], tolerance: float = 1.0e-9) -> bool:
        point = _finite_vector(position, 3, "workspace_position")
        return bool(
            point[0] >= self.minimum_forward_x_m - tolerance and
            point[2] >= self.minimum_tool_z_m - tolerance and
            self.ellipsoid_value(point) <= 1.0 + tolerance)

    def ray_distance(self, origin: Sequence[float],
                     direction: Sequence[float]) -> float:
        """Return the first positive distance from an interior point to a boundary."""

        point = _finite_vector(origin, 3, "workspace_ray_origin")
        axis = normalize(direction, "workspace_ray_direction")
        if not self.contains(point, tolerance=1.0e-7):
            raise ValueError("workspace ray origin is outside the ground sector")
        offset = point - self.center
        inverse_square = 1.0 / (self.radii * self.radii)
        coefficient_a = float(np.sum(axis * axis * inverse_square))
        coefficient_b = float(2.0 * np.sum(offset * axis * inverse_square))
        coefficient_c = float(np.sum(offset * offset * inverse_square) - 1.0)
        discriminant = coefficient_b * coefficient_b - (
            4.0 * coefficient_a * coefficient_c)
        if discriminant < -1.0e-9:
            raise ValueError("workspace ray does not intersect ellipsoid")
        discriminant = max(0.0, discriminant)
        roots = [
            (-coefficient_b - math.sqrt(discriminant)) /
            (2.0 * coefficient_a),
            (-coefficient_b + math.sqrt(discriminant)) /
            (2.0 * coefficient_a),
        ]
        candidates = [value for value in roots if value >= -1.0e-9]
        if not candidates:
            raise ValueError("workspace ellipsoid has no forward ray root")
        boundary = max(candidates)
        if axis[0] < -EPS:
            plane = (self.minimum_forward_x_m - point[0]) / axis[0]
            if plane >= -1.0e-9:
                boundary = min(boundary, max(0.0, plane))
        if axis[2] < -EPS:
            plane = (self.minimum_tool_z_m - point[2]) / axis[2]
            if plane >= -1.0e-9:
                boundary = min(boundary, max(0.0, plane))
        return max(0.0, float(boundary))

    def limit_velocity(self, position: Sequence[float],
                       velocity: Sequence[float],
                       soft_margin_m: float,
                       direction_preserving: bool = False) \
            -> Tuple[np.ndarray, List[str]]:
        """Limit motion at the clipped ellipsoid boundary.

        The legacy policy removes the outward normal and deliberately leaves a
        tangential velocity.  ``direction_preserving`` is the teleoperation
        policy: it only scales or stops the requested vector, so an input with
        zero lateral component can never acquire one at the boundary.
        """

        point = _finite_vector(position, 3, "workspace_position")
        command = _finite_vector(velocity, 3, "workspace_velocity")
        margin = float(soft_margin_m)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("workspace soft margin must be non-negative")
        reasons: List[str] = []

        # If numerical tracking has already crossed a plane, preserve only the
        # component that returns to the valid side instead of permanently
        # freezing all Cartesian motion.
        if point[0] <= self.minimum_forward_x_m and command[0] < 0.0:
            command[0] = 0.0
            reasons.append("WORKSPACE_HARD_FRONT")
        elif (margin > 0.0 and command[0] < 0.0 and
              point[0] - self.minimum_forward_x_m < margin):
            command[0] *= max(
                0.0, (point[0] - self.minimum_forward_x_m) / margin)
            reasons.append("WORKSPACE_SOFT_FRONT")
        if point[2] <= self.minimum_tool_z_m and command[2] < 0.0:
            command[2] = 0.0
            reasons.append("WORKSPACE_HARD_GROUND")
        elif (margin > 0.0 and command[2] < 0.0 and
              point[2] - self.minimum_tool_z_m < margin):
            command[2] *= max(
                0.0, (point[2] - self.minimum_tool_z_m) / margin)
            reasons.append("WORKSPACE_SOFT_GROUND")

        ellipsoid = self.ellipsoid_value(point)
        gradient = 2.0 * (point - self.center) / (self.radii * self.radii)
        outward = float(np.dot(gradient, command)) > 0.0
        if ellipsoid >= 1.0 and outward:
            if direction_preserving:
                command[:] = 0.0
                reasons.append(
                    "WORKSPACE_HARD_ELLIPSOID_DIRECTION_HOLD")
            else:
                normal_square = float(np.dot(gradient, gradient))
                if normal_square > EPS:
                    outward_component = (
                        float(np.dot(command, gradient)) /
                        normal_square) * gradient
                    command -= outward_component
                reasons.append("WORKSPACE_HARD_ELLIPSOID")
        elif margin > 0.0 and outward and np.linalg.norm(command) > EPS:
            try:
                distance = self.ray_distance(point, command)
            except ValueError:
                distance = 0.0
            if distance < margin:
                command *= max(0.0, distance / margin)
                reasons.append("WORKSPACE_SOFT_ELLIPSOID")
        return command, reasons

    def as_dict(self) -> Dict[str, object]:
        return {
            "center_base_m": self.center.tolist(),
            "effective_radii_m": self.radii.tolist(),
            "minimum_forward_x_m": self.minimum_forward_x_m,
            "minimum_tool_z_m": self.minimum_tool_z_m,
            "utilization": self.utilization,
            "boundary_margin_m": self.boundary_margin_m,
        }


class CameraRangeWorkspaceMapper(RelativePoseMapper):
    """Map the calibrated visible hand envelope to a ground-sector boundary.

    Positive and negative camera ranges are calibrated independently.  Their
    asymmetric ellipsoid supplies a direction-dependent hand boundary; the
    same normalized fraction is sent along the mapped robot ray.  Rotation
    intentionally keeps the proven relative SO(3) path from
    :class:`RelativePoseMapper`.
    """

    def __init__(self, translation_matrix: Sequence[Sequence[float]],
                 rotation_matrix: Sequence[Sequence[float]],
                 rotation_gain: Sequence[float],
                 maximum_relative_rotation_rad: float,
                 human_negative_extent_m: Sequence[float],
                 human_positive_extent_m: Sequence[float],
                 robot_workspace: GroundSectorWorkspace,
                 response_exponent: float = 1.0,
                 human_orientation_negative_extent_rad: Optional[
                     Sequence[float]] = None,
                 human_orientation_positive_extent_rad: Optional[
                     Sequence[float]] = None,
                 robot_orientation_negative_extent_rad: Optional[
                     Sequence[float]] = None,
                 robot_orientation_positive_extent_rad: Optional[
                     Sequence[float]] = None,
                 combine_translation_rotation: bool = False,
                 orientation_zero_snap_rad: float = 0.0) -> None:
        super().__init__(
            translation_matrix, rotation_matrix,
            [1.0, 1.0, 1.0], rotation_gain,
            [1.0e6, 1.0e6, 1.0e6], maximum_relative_rotation_rad)
        self.human_negative_extent = _finite_vector(
            human_negative_extent_m, 3, "human_negative_extent_m")
        self.human_positive_extent = _finite_vector(
            human_positive_extent_m, 3, "human_positive_extent_m")
        self.robot_workspace = robot_workspace
        self.response_exponent = float(response_exponent)
        self.combine_translation_rotation = bool(
            combine_translation_rotation)
        self.orientation_zero_snap_rad = float(
            orientation_zero_snap_rad)
        orientation_arguments = (
            human_orientation_negative_extent_rad,
            human_orientation_positive_extent_rad,
            robot_orientation_negative_extent_rad,
            robot_orientation_positive_extent_rad,
        )
        if self.combine_translation_rotation and any(
                value is None for value in orientation_arguments):
            raise ValueError(
                "combined pose normalization requires human and robot "
                "orientation extents")
        if any(value is not None for value in orientation_arguments) and not all(
                value is not None for value in orientation_arguments):
            raise ValueError(
                "orientation extents must be supplied as a complete set")
        if all(value is not None for value in orientation_arguments):
            self.human_orientation_negative_extent = _finite_vector(
                human_orientation_negative_extent_rad, 3,
                "human_orientation_negative_extent_rad")
            self.human_orientation_positive_extent = _finite_vector(
                human_orientation_positive_extent_rad, 3,
                "human_orientation_positive_extent_rad")
            self.robot_orientation_negative_extent = _finite_vector(
                robot_orientation_negative_extent_rad, 3,
                "robot_orientation_negative_extent_rad")
            self.robot_orientation_positive_extent = _finite_vector(
                robot_orientation_positive_extent_rad, 3,
                "robot_orientation_positive_extent_rad")
            if any(np.any(value <= 0.0) for value in (
                    self.human_orientation_negative_extent,
                    self.human_orientation_positive_extent,
                    self.robot_orientation_negative_extent,
                    self.robot_orientation_positive_extent)):
                raise ValueError("orientation workspace extents must be positive")
        else:
            self.human_orientation_negative_extent = None
            self.human_orientation_positive_extent = None
            self.robot_orientation_negative_extent = None
            self.robot_orientation_positive_extent = None
        self._last_boundary_position = None
        self._last_boundary_rotation = None
        self._last_pose_direction = np.zeros(6)
        self._last_pose_fraction = 0.0
        if (np.any(self.human_negative_extent <= 0.0) or
                np.any(self.human_positive_extent <= 0.0) or
                not math.isfinite(self.response_exponent) or
                self.response_exponent <= 0.0 or
                not math.isfinite(self.orientation_zero_snap_rad) or
                self.orientation_zero_snap_rad < 0.0):
            raise ValueError("camera workspace extents must be positive")
        self._last_mapping = {
            "mode": "CAMERA_RANGE_TO_GROUND_SECTOR",
            "human_translation_fraction": 0.0,
            "human_boundary_distance_m": 0.0,
            "robot_boundary_distance_m": 0.0,
            "translation_saturated": False,
        }

    @staticmethod
    def _normalized_components(value: np.ndarray, negative_extent: np.ndarray,
                               positive_extent: np.ndarray) -> np.ndarray:
        extent = np.where(value >= 0.0, positive_extent, negative_extent)
        return value / extent

    @staticmethod
    def _directional_boundary(axis: np.ndarray, negative_extent: np.ndarray,
                              positive_extent: np.ndarray) -> float:
        extent = np.where(axis >= 0.0, positive_extent, negative_extent)
        inverse_boundary_square = float(np.sum((axis / extent) ** 2))
        if inverse_boundary_square <= EPS:
            return 0.0
        return 1.0 / math.sqrt(inverse_boundary_square)

    def _combined_pose_target(
            self, relative_hand_position: Sequence[float],
            relative_hand_rotation: Sequence[Sequence[float]],
            robot_zero_position: Sequence[float],
            robot_zero_rotation: Sequence[Sequence[float]]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the calibrated camera translation and rotation jointly.

        The six normalized human components form a radial task-space command.
        Pure translation therefore reaches the selected positional shell, pure
        rotation reaches the corresponding asymmetric wrist limit, and mixed
        commands share the available 6-D radius instead of asking a 6-DOF arm
        to realize two independent extrema simultaneously.
        """

        hand_position = _finite_vector(
            relative_hand_position, 3, "relative_hand_position")
        hand_rotation_vector = so3_log(relative_hand_rotation)
        zero_position = _finite_vector(
            robot_zero_position, 3, "robot_zero_position")
        zero_rotation = project_to_so3(robot_zero_rotation)
        normalized_translation = self._normalized_components(
            hand_position, self.human_negative_extent,
            self.human_positive_extent)
        normalized_rotation = self._normalized_components(
            hand_rotation_vector,
            self.human_orientation_negative_extent,
            self.human_orientation_positive_extent)
        normalized_pose = np.concatenate((
            normalized_translation, normalized_rotation))
        raw_fraction = float(np.linalg.norm(normalized_pose))
        if raw_fraction < EPS:
            self._last_boundary_position = zero_position.copy()
            self._last_boundary_rotation = zero_rotation.copy()
            self._last_pose_direction = np.zeros(6)
            self._last_pose_fraction = 0.0
            self._last_mapping = {
                "mode": "CAMERA_POSE_RANGE_TO_REACHABLE_GROUND_SECTOR",
                "human_pose_fraction": 0.0,
                "human_translation_fraction": 0.0,
                "human_rotation_fraction": 0.0,
                "human_boundary_distance_m": 0.0,
                "robot_boundary_distance_m": 0.0,
                "translation_saturated": False,
                "pose_saturated": False,
            }
            return zero_position, zero_rotation

        pose_direction = normalized_pose / raw_fraction
        translation_direction = pose_direction[:3]
        rotation_direction = pose_direction[3:]
        translation_share = float(np.linalg.norm(translation_direction))
        rotation_share = float(np.linalg.norm(rotation_direction))

        outer_translation = np.zeros(3)
        robot_boundary = 0.0
        human_boundary = 0.0
        if translation_share > EPS:
            camera_axis = translation_direction / translation_share
            robot_axis = normalize(
                self.translation_matrix @ camera_axis,
                "mapped_robot_direction")
            robot_boundary = self.robot_workspace.ray_distance(
                zero_position, robot_axis)
            outer_translation = (
                translation_share * robot_boundary * robot_axis)
            if float(np.linalg.norm(hand_position)) > EPS:
                physical_hand_axis = normalize(
                    hand_position, "relative_hand_translation_direction")
                human_boundary = self._directional_boundary(
                    physical_hand_axis, self.human_negative_extent,
                    self.human_positive_extent)

        outer_rotation_vector = np.zeros(3)
        robot_rotation_boundary = 0.0
        if rotation_share > EPS:
            normalized_rotation_axis = rotation_direction / rotation_share
            robot_rotation_axis = normalize(
                self.rotation_matrix @ normalized_rotation_axis,
                "mapped_robot_rotation_direction")
            robot_rotation_boundary = self._directional_boundary(
                robot_rotation_axis,
                self.robot_orientation_negative_extent,
                self.robot_orientation_positive_extent)
            outer_rotation_vector = (
                rotation_share * robot_rotation_boundary *
                robot_rotation_axis)

        fraction = float(np.clip(raw_fraction, 0.0, 1.0))
        shaped_fraction = fraction ** self.response_exponent
        boundary_position = zero_position + outer_translation
        boundary_rotation = project_to_so3(
            zero_rotation @ so3_exp(outer_rotation_vector))
        target_position = zero_position + shaped_fraction * outer_translation
        target_rotation = project_to_so3(
            zero_rotation @ so3_exp(
                shaped_fraction * outer_rotation_vector))
        self._last_boundary_position = boundary_position
        self._last_boundary_rotation = boundary_rotation
        self._last_pose_direction = pose_direction.copy()
        self._last_pose_fraction = shaped_fraction
        self._last_mapping = {
            "mode": "CAMERA_POSE_RANGE_TO_REACHABLE_GROUND_SECTOR",
            "human_pose_fraction": float(raw_fraction),
            "human_translation_fraction": float(
                np.linalg.norm(normalized_translation)),
            "human_rotation_fraction": float(
                np.linalg.norm(normalized_rotation)),
            "human_boundary_distance_m": float(human_boundary),
            "robot_boundary_distance_m": float(robot_boundary),
            "robot_rotation_boundary_deg": float(math.degrees(
                robot_rotation_boundary)),
            "outer_translation_m": outer_translation.tolist(),
            "outer_rotation_vector_deg": np.degrees(
                outer_rotation_vector).tolist(),
            "pose_direction": pose_direction.tolist(),
            "translation_saturated": bool(
                np.linalg.norm(normalized_translation) >= 1.0),
            "pose_saturated": bool(raw_fraction >= 1.0),
        }
        return target_position, target_rotation

    def reachability_boundary(self) \
            -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return the last C-zero-relative outer pose and human fraction."""

        if (self._last_boundary_position is None or
                self._last_boundary_rotation is None):
            raise RuntimeError("map must be called before reachability_boundary")
        return (self._last_boundary_position.copy(),
                self._last_boundary_rotation.copy(),
                float(self._last_pose_fraction),
                self._last_pose_direction.copy())

    def _translation_target(self, relative_hand_position: Sequence[float],
                            robot_zero_position: Sequence[float],
                            record: bool = True) -> np.ndarray:
        hand = _finite_vector(
            relative_hand_position, 3, "relative_hand_position")
        zero = _finite_vector(robot_zero_position, 3, "robot_zero_position")
        hand_distance = float(np.linalg.norm(hand))
        if hand_distance < EPS:
            if record:
                self._last_mapping = {
                    "mode": "CAMERA_RANGE_TO_GROUND_SECTOR",
                    "human_translation_fraction": 0.0,
                    "human_boundary_distance_m": 0.0,
                    "robot_boundary_distance_m": 0.0,
                    "translation_saturated": False,
                }
            return zero
        hand_axis = hand / hand_distance
        extent = np.where(
            hand_axis >= 0.0,
            self.human_positive_extent,
            self.human_negative_extent)
        inverse_boundary_square = float(np.sum((hand_axis / extent) ** 2))
        if inverse_boundary_square <= EPS:
            raise ValueError("invalid camera workspace directional boundary")
        human_boundary = 1.0 / math.sqrt(inverse_boundary_square)
        raw_fraction = hand_distance / human_boundary
        fraction = float(np.clip(raw_fraction, 0.0, 1.0))
        shaped_fraction = fraction ** self.response_exponent
        robot_axis = normalize(
            self.translation_matrix @ hand_axis, "mapped_robot_direction")
        robot_boundary = self.robot_workspace.ray_distance(zero, robot_axis)
        target = zero + shaped_fraction * robot_boundary * robot_axis
        if record:
            self._last_mapping = {
                "mode": "CAMERA_RANGE_TO_GROUND_SECTOR",
                "human_translation_fraction": float(raw_fraction),
                "human_boundary_distance_m": float(human_boundary),
                "robot_boundary_distance_m": float(robot_boundary),
                "translation_saturated": bool(raw_fraction >= 1.0),
            }
        return target

    def map(self, relative_hand_position: Sequence[float],
            relative_hand_rotation: Sequence[Sequence[float]],
            robot_zero_position: Sequence[float],
            robot_zero_rotation: Sequence[Sequence[float]]) \
            -> Tuple[np.ndarray, np.ndarray]:
        if self.combine_translation_rotation:
            return self._combined_pose_target(
                relative_hand_position, relative_hand_rotation,
                robot_zero_position, robot_zero_rotation)
        target_position = self._translation_target(
            relative_hand_position, robot_zero_position, record=True)
        if self.human_orientation_negative_extent is None:
            _, target_rotation = super().map(
                [0.0, 0.0, 0.0], relative_hand_rotation,
                robot_zero_position, robot_zero_rotation)
        else:
            target_rotation = self._independent_rotation_target(
                relative_hand_rotation, robot_zero_rotation)
        return target_position, target_rotation

    def _independent_rotation_target(
            self, relative_hand_rotation: Sequence[Sequence[float]],
            robot_zero_rotation: Sequence[Sequence[float]]) -> np.ndarray:
        """Keep wrist rotation independent from the positional workspace ray.

        The live hand angle remains 1:1 until the configured directional robot
        orientation boundary is reached.  Unlike the former combined 6-D
        radius, translation can no longer consume wrist range and a difficult
        wrist orientation can no longer shorten an otherwise reachable
        position target.
        """

        hand_vector = so3_log(relative_hand_rotation)
        zero_rotation = project_to_so3(robot_zero_rotation)
        normalized_hand = self._normalized_components(
            hand_vector, self.human_orientation_negative_extent,
            self.human_orientation_positive_extent)
        human_fraction = float(np.linalg.norm(normalized_hand))
        mapped_vector = (
            self.rotation_matrix @ hand_vector) * self.rotation_gain
        mapped_size = float(np.linalg.norm(mapped_vector))
        if mapped_size <= max(EPS, self.orientation_zero_snap_rad):
            self._last_mapping.update({
                "orientation_mapping": "INDEPENDENT_1_TO_1_DIRECTIONAL_CAP",
                "human_rotation_fraction": 0.0,
                "robot_rotation_boundary_deg": 0.0,
                "target_rotation_deg": 0.0,
                "orientation_zero_snapped": bool(mapped_size >= EPS),
                "rotation_saturated": False,
            })
            return zero_rotation

        mapped_axis = mapped_vector / mapped_size
        robot_boundary = self._directional_boundary(
            mapped_axis, self.robot_orientation_negative_extent,
            self.robot_orientation_positive_extent)
        target_size = min(mapped_size, robot_boundary)
        target_vector = target_size * mapped_axis
        self._last_mapping.update({
            "orientation_mapping": "INDEPENDENT_1_TO_1_DIRECTIONAL_CAP",
            "human_rotation_fraction": human_fraction,
            "robot_rotation_boundary_deg": float(math.degrees(
                robot_boundary)),
            "target_rotation_deg": float(math.degrees(target_size)),
            "orientation_zero_snapped": False,
            "rotation_saturated": bool(mapped_size >= robot_boundary),
        })
        return project_to_so3(zero_rotation @ so3_exp(target_vector))

    def map_target_velocity(
            self, raw_hand_velocity: Sequence[float],
            hand_zero_rotation: Sequence[Sequence[float]],
            robot_zero_rotation: Sequence[Sequence[float]],
            relative_hand_position: Optional[Sequence[float]] = None,
            robot_zero_position: Optional[Sequence[float]] = None) -> np.ndarray:
        raw = _finite_vector(raw_hand_velocity, 6, "raw_hand_velocity")
        if relative_hand_position is None or robot_zero_position is None:
            raise ValueError(
                "camera-range feed-forward requires hand and robot positions")
        hand = _finite_vector(
            relative_hand_position, 3, "relative_hand_position")
        zero = _finite_vector(robot_zero_position, 3, "robot_zero_position")
        # A short directional finite difference follows the nonlinear radial
        # mapping, including asymmetric camera ranges and boundary saturation.
        derivative_dt = 1.0e-3
        current = self._translation_target(hand, zero, record=False)
        advanced = self._translation_target(
            hand + raw[:3] * derivative_dt, zero, record=False)
        linear = (advanced - current) / derivative_dt
        legacy = super().map_target_velocity(
            raw, hand_zero_rotation, robot_zero_rotation)
        legacy[:3] = linear
        # At an orientation boundary the absolute pose target remains active.
        # Removing only angular feed-forward prevents a continuing hand motion
        # from pushing Servo farther into the same joint limit; moving back
        # immediately produces pose-error feedback toward C-zero.
        if self._last_mapping.get("rotation_saturated", False):
            legacy[3:] = 0.0
        return legacy

    def mapping_diagnostics(self) -> Dict[str, object]:
        result = dict(self._last_mapping)
        result["robot_workspace"] = self.robot_workspace.as_dict()
        return result


class AxisDecoupledWorkspaceMapper(CameraRangeWorkspaceMapper):
    """Map three calibrated hand axes without radial cross-axis coupling.

    Each signed camera intent component is normalized independently and maps
    to the corresponding signed base-axis boundary.  Their unconstrained sum
    may lie outside the front/ground ellipsoid; in that case only the new raw
    target increment since the preceding hand frame is applied from the last
    feasible target and shortened if needed.  The increment is never rotated
    and an older clipped residual is never spent implicitly, so changing depth
    cannot alter an established lateral target merely to slide around the shell.
    """

    def __init__(self, translation_matrix: Sequence[Sequence[float]],
                 rotation_matrix: Sequence[Sequence[float]],
                 rotation_gain: Sequence[float],
                 maximum_relative_rotation_rad: float,
                 human_negative_extent_m: Sequence[float],
                 human_positive_extent_m: Sequence[float],
                 robot_workspace: GroundSectorWorkspace,
                 response_exponent: float = 1.0,
                 human_orientation_negative_extent_rad: Optional[
                     Sequence[float]] = None,
                 human_orientation_positive_extent_rad: Optional[
                     Sequence[float]] = None,
                 robot_orientation_negative_extent_rad: Optional[
                     Sequence[float]] = None,
                 robot_orientation_positive_extent_rad: Optional[
                     Sequence[float]] = None,
                 workspace_projection_safety_factor: float = 0.995,
                 translation_zero_snap_fraction: float = 0.02,
                 orientation_zero_snap_rad: float = math.radians(2.0)) -> None:
        super().__init__(
            translation_matrix, rotation_matrix, rotation_gain,
            maximum_relative_rotation_rad,
            human_negative_extent_m, human_positive_extent_m,
            robot_workspace, response_exponent,
            human_orientation_negative_extent_rad,
            human_orientation_positive_extent_rad,
            robot_orientation_negative_extent_rad,
            robot_orientation_positive_extent_rad,
            combine_translation_rotation=False,
            orientation_zero_snap_rad=orientation_zero_snap_rad)
        self.workspace_projection_safety_factor = float(
            workspace_projection_safety_factor)
        self.translation_zero_snap_fraction = float(
            translation_zero_snap_fraction)
        if not 0.0 < self.workspace_projection_safety_factor <= 1.0:
            raise ValueError(
                "workspace_projection_safety_factor must be in (0, 1]")
        if (not math.isfinite(self.translation_zero_snap_fraction) or
                not 0.0 <= self.translation_zero_snap_fraction < 1.0):
            raise ValueError(
                "translation_zero_snap_fraction must be in [0, 1)")
        gram = self.translation_matrix.T @ self.translation_matrix
        if not np.allclose(gram, np.eye(3), atol=1.0e-9):
            raise ValueError(
                "axis-decoupled translation_matrix must be orthonormal")
        self.reset()

    def reset(self) -> None:
        self._axis_zero_position: Optional[np.ndarray] = None
        self._axis_target_position: Optional[np.ndarray] = None
        self._axis_requested_position: Optional[np.ndarray] = None
        self._positive_robot_boundaries = np.zeros(3)
        self._negative_robot_boundaries = np.zeros(3)

    def _prepare_axis_boundaries(self, zero_position: np.ndarray) -> None:
        if (self._axis_zero_position is not None and
                np.linalg.norm(
                    zero_position - self._axis_zero_position) <= 1.0e-9):
            return
        positive = np.zeros(3)
        negative = np.zeros(3)
        for index in range(3):
            camera_positive_axis = np.zeros(3)
            camera_positive_axis[index] = 1.0
            robot_positive_axis = normalize(
                self.translation_matrix @ camera_positive_axis,
                "mapped_positive_axis")
            positive[index] = self.robot_workspace.ray_distance(
                zero_position, robot_positive_axis)
            negative[index] = self.robot_workspace.ray_distance(
                zero_position, -robot_positive_axis)
        self._axis_zero_position = zero_position.copy()
        self._axis_target_position = zero_position.copy()
        self._axis_requested_position = zero_position.copy()
        self._positive_robot_boundaries = positive
        self._negative_robot_boundaries = negative

    def _raw_axis_target(
            self, relative_hand_position: Sequence[float],
            robot_zero_position: Sequence[float]) \
            -> Tuple[np.ndarray, np.ndarray]:
        hand = _finite_vector(
            relative_hand_position, 3, "relative_hand_position")
        zero = _finite_vector(
            robot_zero_position, 3, "robot_zero_position")
        self._prepare_axis_boundaries(zero)
        extent = np.where(
            hand >= 0.0,
            self.human_positive_extent,
            self.human_negative_extent)
        normalized = np.clip(hand / extent, -1.0, 1.0)
        shaped = np.sign(normalized) * (
            np.abs(normalized) ** self.response_exponent)
        target = zero.copy()
        for index, value in enumerate(shaped):
            if abs(float(value)) < EPS:
                continue
            camera_axis = np.zeros(3)
            camera_axis[index] = math.copysign(1.0, float(value))
            robot_axis = normalize(
                self.translation_matrix @ camera_axis,
                "mapped_signed_axis")
            boundary = (
                self._positive_robot_boundaries[index]
                if value >= 0.0 else
                self._negative_robot_boundaries[index])
            target += abs(float(value)) * boundary * robot_axis
        return target, normalized

    def _translation_target(self, relative_hand_position: Sequence[float],
                            robot_zero_position: Sequence[float],
                            record: bool = True) -> np.ndarray:
        hand = _finite_vector(
            relative_hand_position, 3, "relative_hand_position")
        zero = _finite_vector(
            robot_zero_position, 3, "robot_zero_position")
        raw_target, normalized = self._raw_axis_target(hand, zero)

        zero_snapped = bool(
            np.max(np.abs(normalized)) <= self.translation_zero_snap_fraction)
        if zero_snapped:
            # C-zero is an absolute target.  Clear the raw request as well as
            # the output so a previously clipped workspace residual can never
            # be released on the following frame.
            raw_target = zero.copy()
            target = zero.copy()
            candidate = zero.copy()
            raw_requested_step = zero - (
                zero if self._axis_requested_position is None else
                self._axis_requested_position)
            requested_step = zero - (
                zero if self._axis_target_position is None else
                self._axis_target_position)
            fraction = 1.0
            limited = False
        else:
            previous = (
                zero.copy() if self._axis_target_position is None else
                self._axis_target_position.copy())
            previous_requested = (
                zero.copy() if self._axis_requested_position is None else
                self._axis_requested_position.copy())
            if not self.robot_workspace.contains(previous, tolerance=1.0e-7):
                previous = zero.copy()
            # Apply only the newly requested raw-target increment.  If a
            # lateral component was clipped on an earlier frame, a later pure
            # depth change cannot spend that old residual and move laterally.
            raw_requested_step = raw_target - previous_requested
            candidate = previous + raw_requested_step

            # A clipped raw target can be farther from C-zero than the last
            # feasible output.  Applying its full reverse delta later used to
            # cross zero and throw the arm to the opposite workspace edge.
            # Express the candidate in the three mapped camera axes and keep
            # every component between C-zero and the current absolute request.
            target_coordinates = self.translation_matrix.T @ (
                raw_target - zero)
            candidate_coordinates = self.translation_matrix.T @ (
                candidate - zero)
            candidate_coordinates = np.minimum(
                np.maximum(candidate_coordinates,
                           np.minimum(np.zeros(3), target_coordinates)),
                np.maximum(np.zeros(3), target_coordinates))
            candidate = zero + self.translation_matrix @ candidate_coordinates
            requested_step = candidate - previous
            step_length = float(np.linalg.norm(requested_step))
            if step_length < EPS:
                target = previous
                fraction = 0.0
            elif self.robot_workspace.contains(candidate):
                target = candidate.copy()
                fraction = 1.0
            else:
                available = self.robot_workspace.ray_distance(
                    previous, requested_step / step_length)
                if available < step_length:
                    available *= self.workspace_projection_safety_factor
                allowed = float(np.clip(available, 0.0, step_length))
                fraction = allowed / step_length
                target = previous + fraction * requested_step
            limited = bool(
                fraction < 1.0 - 1.0e-12 or
                not self.robot_workspace.contains(raw_target))

        self._axis_requested_position = raw_target.copy()
        self._axis_target_position = target.copy()
        if record:
            self._last_mapping = {
                "mode": "PERSPECTIVE_DECOUPLED_AXIS_PRESERVING_GROUND_SECTOR",
                "human_axis_fraction": normalized.tolist(),
                "human_translation_fraction": float(
                    np.linalg.norm(normalized)),
                "raw_axis_target_m": raw_target.tolist(),
                "raw_requested_increment_m": raw_requested_step.tolist(),
                "requested_increment_m": requested_step.tolist(),
                "increment_candidate_m": candidate.tolist(),
                "unspent_raw_residual_m": (
                    raw_target - target).tolist(),
                "workspace_projection_fraction": float(fraction),
                "workspace_limited": bool(limited),
                "translation_zero_snapped": zero_snapped,
                "translation_saturated": bool(
                    np.any(np.abs(normalized) >= 1.0 - 1.0e-12)),
                "positive_robot_axis_boundaries_m": (
                    self._positive_robot_boundaries.tolist()),
                "negative_robot_axis_boundaries_m": (
                    self._negative_robot_boundaries.tolist()),
            }
        return target

    def map_target_velocity(
            self, raw_hand_velocity: Sequence[float],
            hand_zero_rotation: Sequence[Sequence[float]],
            robot_zero_rotation: Sequence[Sequence[float]],
            relative_hand_position: Optional[Sequence[float]] = None,
            robot_zero_position: Optional[Sequence[float]] = None) -> np.ndarray:
        raw = _finite_vector(raw_hand_velocity, 6, "raw_hand_velocity")
        if relative_hand_position is None or robot_zero_position is None:
            raise ValueError(
                "axis-decoupled feed-forward requires hand and robot positions")
        hand = _finite_vector(
            relative_hand_position, 3, "relative_hand_position")
        derivative_dt = 1.0e-3
        current, _ = self._raw_axis_target(hand, robot_zero_position)
        advanced, _ = self._raw_axis_target(
            hand + raw[:3] * derivative_dt, robot_zero_position)
        angular = RelativePoseMapper.map_target_velocity(
            self, raw, hand_zero_rotation, robot_zero_rotation)
        angular[:3] = (advanced - current) / derivative_dt
        if self._last_mapping.get("rotation_saturated", False):
            angular[3:] = 0.0
        return angular


class RelativePoseServoController:
    """Convert target error plus target feed-forward into a base-frame twist."""

    def __init__(self, translation_error_gain: Sequence[float],
                 rotation_error_gain: Sequence[float],
                 maximum_linear_velocity: Sequence[float],
                 maximum_angular_velocity: Sequence[float],
                 maximum_linear_speed_norm: Optional[float] = None,
                 maximum_angular_speed_norm: Optional[float] = None,
                 translation_feedforward_gain: Optional[Sequence[float]] = None,
                 rotation_feedforward_gain: Optional[Sequence[float]] = None) -> None:
        self.translation_error_gain = _finite_vector(
            translation_error_gain, 3, "translation_error_gain")
        self.rotation_error_gain = _finite_vector(
            rotation_error_gain, 3, "rotation_error_gain")
        self.maximum_linear_velocity = _finite_vector(
            maximum_linear_velocity, 3, "maximum_linear_velocity")
        self.maximum_angular_velocity = _finite_vector(
            maximum_angular_velocity, 3, "maximum_angular_velocity")
        self.translation_feedforward_gain = _finite_vector(
            ([0.0] * 3 if translation_feedforward_gain is None else
             translation_feedforward_gain),
            3, "translation_feedforward_gain")
        self.rotation_feedforward_gain = _finite_vector(
            ([0.0] * 3 if rotation_feedforward_gain is None else
             rotation_feedforward_gain),
            3, "rotation_feedforward_gain")
        self.maximum_linear_speed_norm = float(
            np.linalg.norm(self.maximum_linear_velocity)
            if maximum_linear_speed_norm is None else maximum_linear_speed_norm)
        self.maximum_angular_speed_norm = float(
            np.linalg.norm(self.maximum_angular_velocity)
            if maximum_angular_speed_norm is None else maximum_angular_speed_norm)
        if (np.any(self.translation_error_gain <= 0.0) or
                np.any(self.rotation_error_gain <= 0.0) or
                np.any(self.maximum_linear_velocity <= 0.0) or
                np.any(self.maximum_angular_velocity <= 0.0) or
                np.any(self.translation_feedforward_gain < 0.0) or
                np.any(self.rotation_feedforward_gain < 0.0) or
                not math.isfinite(self.maximum_linear_speed_norm) or
                not math.isfinite(self.maximum_angular_speed_norm) or
                self.maximum_linear_speed_norm <= 0.0 or
                self.maximum_angular_speed_norm <= 0.0):
            raise ValueError("pose-servo gains and limits must be positive")

    @staticmethod
    def clamp_norm(value: np.ndarray, maximum_norm: float) -> np.ndarray:
        size = float(np.linalg.norm(value))
        if size > maximum_norm:
            return value * (maximum_norm / size)
        return value

    def command(self, current_position: Sequence[float],
                current_rotation: Sequence[Sequence[float]],
                target_position: Sequence[float],
                target_rotation: Sequence[Sequence[float]],
                target_velocity: Optional[Sequence[float]] = None) -> np.ndarray:
        position_error = (
            _finite_vector(target_position, 3, "target_position") -
            _finite_vector(current_position, 3, "current_position"))
        rotation_error = so3_log(
            project_to_so3(target_rotation) @
            project_to_so3(current_rotation).T)
        feedforward = (
            np.zeros(6) if target_velocity is None else
            _finite_vector(target_velocity, 6, "target_velocity"))
        linear = clamp_each(
            position_error * self.translation_error_gain +
            feedforward[:3] * self.translation_feedforward_gain,
            self.maximum_linear_velocity)
        angular = clamp_each(
            rotation_error * self.rotation_error_gain +
            feedforward[3:] * self.rotation_feedforward_gain,
            self.maximum_angular_velocity)
        linear = self.clamp_norm(linear, self.maximum_linear_speed_norm)
        angular = self.clamp_norm(angular, self.maximum_angular_speed_norm)
        return np.concatenate((linear, angular))


def confirmed_position_reference_hold(
        holding_last_target: bool, target_age_s: float,
        confirmation_delay_s: float) -> bool:
    """Request a stiff joint-reference hold after a confirmed target loss.

    Short HaMeR frame gaps must keep using the last Cartesian target so live
    tracking does not repeatedly brake.  Once the configured delay elapses,
    however, continuing to servo toward a stale and possibly unreachable 6-D
    pose can excite singularity recovery.  A positive delay enables the
    position-reference hold; zero keeps the established velocity profile
    behavior unchanged.
    """

    delay = float(confirmation_delay_s)
    age = float(target_age_s)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError(
            "position-reference hold delay must be finite and non-negative")
    if not holding_last_target or delay == 0.0:
        return False
    return math.isfinite(age) and age >= delay


@dataclass(frozen=True)
class ShapedCommand:
    velocity: np.ndarray
    valid: bool
    input_age_s: float
    reason: str


class LatestCommandShaper:
    """50 Hz latest-sample bridge with acceleration and timeout stop limits."""

    def __init__(self, maximum_velocity: Sequence[float],
                 maximum_acceleration: Sequence[float],
                 input_timeout_s: float = 0.09,
                 timeout_zero_deadline_s: float = 0.15) -> None:
        self.maximum_velocity = _finite_vector(maximum_velocity, 6, "maximum_velocity")
        self.maximum_acceleration = _finite_vector(maximum_acceleration, 6, "maximum_acceleration")
        if np.any(self.maximum_velocity < 0.0) or np.any(self.maximum_acceleration <= 0.0):
            raise ValueError("velocity limits must be non-negative and acceleration limits positive")
        self.input_timeout_s = float(input_timeout_s)
        self.timeout_zero_deadline_s = float(timeout_zero_deadline_s)
        if not 0.0 < self.input_timeout_s < self.timeout_zero_deadline_s:
            raise ValueError("timeout must be positive and precede zero deadline")
        required = self.maximum_velocity/(self.timeout_zero_deadline_s-self.input_timeout_s)
        if np.any(self.maximum_acceleration+1.0e-12 < required):
            raise ValueError("acceleration limits cannot meet configured timeout zero deadline")
        self.target = np.zeros(6)
        self.source_timestamp: Optional[float] = None
        self.source_valid = False
        self.output = np.zeros(6)
        self.last_tick: Optional[float] = None

    def update(self, velocity: Sequence[float], source_timestamp: float, valid: bool) -> None:
        if not math.isfinite(float(source_timestamp)):
            raise ValueError("source timestamp must be finite")
        if self.source_timestamp is not None and source_timestamp < self.source_timestamp-1.0e-9:
            return
        self.target = clamp_each(_finite_vector(velocity, 6, "velocity"), self.maximum_velocity)
        self.source_timestamp = float(source_timestamp)
        self.source_valid = bool(valid)

    def tick(self, now: float) -> ShapedCommand:
        if not math.isfinite(float(now)):
            raise ValueError("now must be finite")
        dt = 0.0 if self.last_tick is None else max(0.0, float(now)-self.last_tick)
        self.last_tick = float(now)
        raw_age = math.inf if self.source_timestamp is None else float(now)-self.source_timestamp
        age = max(0.0, raw_age)
        if self.source_timestamp is None:
            desired = np.zeros(6); reason = "NO_INPUT"; valid = False
        elif raw_age < -0.05:
            desired = np.zeros(6); reason = "INPUT_CLOCK_MISMATCH"; valid = False
        elif not self.source_valid:
            desired = np.zeros(6); reason = "INPUT_INVALID"; valid = False
        elif age > self.input_timeout_s:
            desired = np.zeros(6); reason = "INPUT_TIMEOUT"; valid = False
        else:
            desired = self.target; reason = "NONE"; valid = True
        if dt > 0.0:
            delta = clamp_each(desired-self.output, self.maximum_acceleration*dt)
            self.output = clamp_each(self.output+delta, self.maximum_velocity)
        if age >= self.timeout_zero_deadline_s or not np.all(np.isfinite(self.output)):
            self.output[:] = 0.0
            if age >= self.timeout_zero_deadline_s:
                reason = "INPUT_TIMEOUT_ZERO"
            valid = False
        return ShapedCommand(self.output.copy(), valid, age, reason)


@dataclass(frozen=True)
class GestureResult:
    hold_arm: bool
    action: Optional[int]
    active_gesture: int
    reason: str


class GestureIsolationGate:
    """Debounce discrete hand actions and emit exactly one event per gesture."""

    def __init__(self, stable_duration_s: float = 0.30,
                 release_duration_s: float = 0.30,
                 confidence_threshold: float = 0.75) -> None:
        self.stable_duration_s = float(stable_duration_s)
        self.release_duration_s = float(release_duration_s)
        self.confidence_threshold = float(confidence_threshold)
        self.candidate = GESTURE_NONE
        self.candidate_since: Optional[float] = None
        self.active = GESTURE_NONE
        self.neutral_since: Optional[float] = None

    def update(self, timestamp: float, gesture: int, confidence: float) -> GestureResult:
        timestamp = float(timestamp); gesture = int(gesture); confidence = float(confidence)
        recognized = (gesture if gesture in GESTURE_NAMES and gesture != GESTURE_NONE and
                      confidence >= self.confidence_threshold else GESTURE_NONE)
        if self.active != GESTURE_NONE:
            if recognized == GESTURE_NONE:
                if self.neutral_since is None:
                    self.neutral_since = timestamp
                if timestamp-self.neutral_since >= self.release_duration_s:
                    self.active = GESTURE_NONE
                    self.candidate = GESTURE_NONE
                    self.candidate_since = None
                    return GestureResult(False, None, GESTURE_NONE, "GESTURE_RELEASED")
            else:
                self.neutral_since = None
            return GestureResult(True, None, self.active, "GESTURE_HOLD")
        if recognized == GESTURE_NONE:
            self.candidate = GESTURE_NONE
            self.candidate_since = None
            return GestureResult(False, None, GESTURE_NONE, "NONE")
        if recognized != self.candidate:
            self.candidate = recognized
            self.candidate_since = timestamp
            return GestureResult(False, None, GESTURE_NONE, "GESTURE_DEBOUNCE")
        if self.candidate_since is not None and timestamp-self.candidate_since >= self.stable_duration_s:
            self.active = recognized
            self.neutral_since = None
            return GestureResult(True, recognized, recognized, "GESTURE_TRIGGERED")
        return GestureResult(False, None, GESTURE_NONE, "GESTURE_DEBOUNCE")


@dataclass(frozen=True)
class OrientationCandidate:
    label: str
    rotation: np.ndarray
    distance_rad: float
    feasible: bool


def top_grasp_candidate(current_center_rotation: Sequence[Sequence[float]],
                        approach_axis_local: Sequence[float],
                        table_normal_base: Sequence[float]) -> OrientationCandidate:
    current = project_to_so3(current_center_rotation)
    target = closest_rotation_with_axis(current, approach_axis_local,
                                        -normalize(table_normal_base, "table_normal"))
    return OrientationCandidate("top", target, rotation_distance(current, target), True)


def side_grasp_candidates(current_center_rotation: Sequence[Sequence[float]],
                          approach_axis_local: Sequence[float],
                          side_directions: Dict[str, Sequence[float]],
                          feasibility: Optional[Callable[[str, np.ndarray], bool]] = None
                          ) -> List[OrientationCandidate]:
    current = project_to_so3(current_center_rotation)
    result = []
    for label, direction in side_directions.items():
        target = closest_rotation_with_axis(current, approach_axis_local, direction)
        possible = True if feasibility is None else bool(feasibility(label, target))
        result.append(OrientationCandidate(label, target,
                                           rotation_distance(current, target), possible))
    return result


def select_nearest_candidate(candidates: Iterable[OrientationCandidate]) -> OrientationCandidate:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        raise ValueError("no kinematically feasible orientation candidate")
    return min(feasible, key=lambda candidate: (candidate.distance_rad, candidate.label))


@dataclass(frozen=True)
class AssistResult:
    velocity: np.ndarray
    strength: float
    assist_velocity: np.ndarray
    selected_label: str
    target_center_rotation: Optional[np.ndarray]
    target_flange_position: Optional[np.ndarray]
    target_flange_rotation: Optional[np.ndarray]
    opposing: bool


class MinimumInterventionOrientationAssist:
    """Continuous orientation assistance around a configured grasp center."""

    def __init__(self, flange_to_center_position: Sequence[float],
                 flange_to_center_rotation: Sequence[Sequence[float]],
                 angular_gain: float = 1.5, position_gain: float = 2.0,
                 maximum_assist_angular_speed: float = 0.45,
                 maximum_assist_linear_speed: float = 0.08,
                 rise_rate_per_s: float = 1.5, fall_rate_per_s: float = 3.0,
                 opposition_dot_threshold: float = -0.002,
                 opposition_duration_s: float = 0.20) -> None:
        self.p_fc = _finite_vector(flange_to_center_position, 3, "flange_to_center_position")
        self.r_fc = project_to_so3(flange_to_center_rotation)
        self.angular_gain = float(angular_gain)
        self.position_gain = float(position_gain)
        self.maximum_assist_angular_speed = float(maximum_assist_angular_speed)
        self.maximum_assist_linear_speed = float(maximum_assist_linear_speed)
        self.rise_rate_per_s = float(rise_rate_per_s)
        self.fall_rate_per_s = float(fall_rate_per_s)
        self.opposition_dot_threshold = float(opposition_dot_threshold)
        self.opposition_duration_s = float(opposition_duration_s)
        self.requested_strength = 0.0
        self.strength = 0.0
        self.selected_label = "none"
        self.target_center_rotation: Optional[np.ndarray] = None
        self.opposition_time = 0.0

    def cancel(self) -> None:
        self.requested_strength = 0.0
        self.selected_label = "none"
        self.target_center_rotation = None

    def activate(self, selected: OrientationCandidate, requested_strength: float = 1.0) -> None:
        self.selected_label = selected.label
        self.target_center_rotation = project_to_so3(selected.rotation)
        self.requested_strength = float(np.clip(requested_strength, 0.0, 1.0))
        self.opposition_time = 0.0

    @staticmethod
    def _limit_norm(value: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        return value if norm <= maximum or norm < EPS else value*(maximum/norm)

    def compute(self, dt: float, flange_position: Sequence[float],
                flange_rotation: Sequence[Sequence[float]],
                operator_velocity: Sequence[float]) -> AssistResult:
        dt = max(0.0, float(dt))
        p_flange = _finite_vector(flange_position, 3, "flange_position")
        r_flange = project_to_so3(flange_rotation)
        operator = _finite_vector(operator_velocity, 6, "operator_velocity")
        p_center, _ = compose_pose(p_flange, r_flange, self.p_fc, self.r_fc)
        assist = np.zeros(6)
        p_target = None; r_target_flange = None
        opposing = False
        if self.target_center_rotation is not None and self.requested_strength > 0.0:
            p_target, r_target_flange = flange_pose_for_fixed_center(
                p_center, self.target_center_rotation, self.p_fc, self.r_fc)
            rotation_error = so3_log(r_target_flange @ r_flange.T)
            assist[3:] = self._limit_norm(self.angular_gain*rotation_error,
                                          self.maximum_assist_angular_speed)
            assist[:3] = self._limit_norm(self.position_gain*(p_target-p_flange),
                                          self.maximum_assist_linear_speed)
            dot = float(np.dot(operator[3:], assist[3:]))
            opposing = (np.linalg.norm(operator[3:]) > 1.0e-4 and
                        np.linalg.norm(assist[3:]) > 1.0e-4 and
                        dot < self.opposition_dot_threshold)
            self.opposition_time = self.opposition_time+dt if opposing else max(0.0, self.opposition_time-dt)
        else:
            self.opposition_time = 0.0
        desired_strength = self.requested_strength
        if self.opposition_time >= self.opposition_duration_s:
            desired_strength = 0.0
        rate = self.rise_rate_per_s if desired_strength > self.strength else self.fall_rate_per_s
        step = rate*dt
        self.strength += float(np.clip(desired_strength-self.strength, -step, step))
        self.strength = float(np.clip(self.strength, 0.0, 1.0))
        return AssistResult(operator+self.strength*assist, self.strength,
                            assist, self.selected_label,
                            None if self.target_center_rotation is None else self.target_center_rotation.copy(),
                            None if p_target is None else p_target.copy(),
                            None if r_target_flange is None else r_target_flange.copy(),
                            opposing)


def apply_workspace_boundary(position: Sequence[float], velocity: Sequence[float],
                             lower: Sequence[float], upper: Sequence[float],
                             soft_margin_m: float) -> Tuple[np.ndarray, List[str]]:
    """Limit outward Cartesian velocity at a preset axis-aligned workspace.

    This is deliberately only a configured workspace envelope, not unknown
    obstacle avoidance.
    """
    p = _finite_vector(position, 3, "position")
    command = _finite_vector(velocity, 3, "velocity")
    low = _finite_vector(lower, 3, "lower")
    high = _finite_vector(upper, 3, "upper")
    if np.any(low >= high) or soft_margin_m < 0.0:
        raise ValueError("invalid workspace boundary")
    reasons: List[str] = []
    for axis in range(3):
        if command[axis] < 0.0:
            distance = p[axis]-low[axis]
        elif command[axis] > 0.0:
            distance = high[axis]-p[axis]
        else:
            continue
        if distance <= 0.0:
            command[axis] = 0.0; reasons.append("WORKSPACE_HARD_AXIS_{}".format(axis))
        elif soft_margin_m > 0.0 and distance < soft_margin_m:
            command[axis] *= distance/soft_margin_m
            reasons.append("WORKSPACE_SOFT_AXIS_{}".format(axis))
    return command, reasons


def apply_ground_sector_workspace_boundary(
        position: Sequence[float], velocity: Sequence[float],
        workspace: GroundSectorWorkspace,
        soft_margin_m: float,
        direction_preserving: bool = False) \
        -> Tuple[np.ndarray, List[str]]:
    """Apply the same ground-sector envelope used by camera-range targets."""

    if not isinstance(workspace, GroundSectorWorkspace):
        raise TypeError("workspace must be a GroundSectorWorkspace")
    return workspace.limit_velocity(
        position, velocity, soft_margin_m, direction_preserving)


def robot_output_allowed(simulation: bool, enable_robot: bool,
                         calibration_confirmed: bool, authorization: str) -> bool:
    """Central fail-closed gate for any physical ABB output adapter."""
    if bool(simulation):
        return True
    return bool(enable_robot) and bool(calibration_confirmed) and str(authorization) == REAL_ROBOT_AUTHORIZATION_TOKEN


__all__ = [
    "AprilTagV3PoseContinuityFilter", "AssistResult",
    "CollisionRetreatGuard", "CollisionRetreatResult",
    "CoordinateVelocityMapper", "GESTURE_CLOSE",
    "GESTURE_CONFIGURATION", "GESTURE_NAMES", "GESTURE_NONE", "GESTURE_OPEN",
    "REAL_ROBOT_AUTHORIZATION_TOKEN",
    "GestureIsolationGate", "LatestCommandShaper",
    "MinimumInterventionOrientationAssist", "OrientationCandidate", "PoseSample",
    "AxisDecoupledWorkspaceMapper", "CameraRangeWorkspaceMapper",
    "DirectionPreservingIncrementProjector",
    "DirectionPreservingReachabilityProjection", "GroundSectorWorkspace",
    "PerspectiveDecoupledIntent", "PerspectiveIntentDecoupler",
    "RelativePoseMapper", "RelativePoseServoController",
    "SideAxisProjectionResult", "SymmetricSideGraspProjector",
    "StationaryFeedforwardGate", "StationaryFeedforwardResult",
    "SixDofTrendEstimator", "TrendResult", "apply_workspace_boundary",
    "apply_ground_sector_workspace_boundary",
    "closest_rotation_with_axis", "compose_pose", "flange_pose_for_fixed_center",
    "interpolate_pose_ray",
    "matrix_to_quaternion_xyzw", "project_to_so3", "quaternion_xyzw_to_matrix",
    "rotation_distance", "select_nearest_candidate", "side_grasp_candidates",
    "so3_exp", "so3_log", "top_grasp_candidate", "robot_output_allowed",
]
