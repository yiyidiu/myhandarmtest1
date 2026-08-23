#!/usr/bin/env python3
"""Pure geometry for object-relative three-finger enclosure planning.

This module has no ROS, Gazebo or MoveIt side effects.  It derives the hand
kinematics and collision proxies from the installed URDF, simulates the fixed
palm OPEN->CLOSE sweep, and fails closed unless all f1/f2/f3 families have a
distinct predicted object contact without an early palm/table/self collision.

The result is an enclosure-geometry check, not a force-closure claim.
"""

from dataclasses import dataclass
import math

import numpy as np
import yaml
from urdf_parser_py.urdf import Box, Cylinder, Sphere, URDF


EPS = 1.0e-12


def _finite_array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("{} must be finite with shape {}".format(name, shape))
    return result


def _rotation_axis_angle(axis, angle):
    axis = _finite_array(axis, (3,), "joint axis")
    norm = np.linalg.norm(axis)
    if norm < EPS:
        raise ValueError("joint axis is degenerate")
    x, y, z = axis / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def rotation_from_rpy(rpy):
    roll, pitch, yaw = _finite_array(rpy, (3,), "rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def transform(rotation=None, translation=None):
    result = np.eye(4, dtype=float)
    if rotation is not None:
        result[:3, :3] = _finite_array(rotation, (3, 3), "rotation")
    if translation is not None:
        result[:3, 3] = _finite_array(translation, (3,), "translation")
    return result


def origin_transform(origin):
    if origin is None:
        return np.eye(4, dtype=float)
    return transform(rotation_from_rpy(origin.rpy), origin.xyz)


def validate_rotation(rotation, name="rotation", tolerance=1.0e-7):
    rotation = _finite_array(rotation, (3, 3), name)
    error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    if error > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError(
            "{} is not SO(3): orthogonality_error={}, det={}".format(
                name, error, determinant
            )
        )
    return rotation


@dataclass(frozen=True)
class OBB:
    center: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "center", _finite_array(self.center, (3,), "OBB center"))
        object.__setattr__(
            self, "rotation", validate_rotation(self.rotation, "OBB rotation")
        )
        half = _finite_array(self.half_extents, (3,), "OBB half extents")
        if np.any(half <= 0.0):
            raise ValueError("OBB half extents must be positive")
        object.__setattr__(self, "half_extents", half)


@dataclass(frozen=True)
class Capsule:
    first: np.ndarray
    second: np.ndarray
    radius: float
    link: str
    family: str


@dataclass(frozen=True)
class BoxProxy:
    center: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray
    link: str


@dataclass(frozen=True)
class ContactPrediction:
    family: str
    link: str
    closure_fraction: float
    signed_distance_m: float
    point_hand_m: np.ndarray
    normal_hand: np.ndarray
    face_axis: int
    face_sign: int

    def as_dict(self):
        return {
            "family": self.family,
            "link": self.link,
            "closure_fraction": self.closure_fraction,
            "signed_distance_m": self.signed_distance_m,
            "point_hand_m": self.point_hand_m.tolist(),
            "normal_hand": self.normal_hand.tolist(),
            "face_axis": self.face_axis,
            "face_sign": self.face_sign,
        }


@dataclass(frozen=True)
class EnclosureResult:
    valid: bool
    failure_reasons: tuple
    contacts: dict
    object_inside_three_finger_envelope: bool
    palm_clearance_m: float
    table_clearance_m: float
    cross_finger_clearance_m: float
    projected_contact_area_m2: float

    def as_dict(self):
        return {
            "valid": self.valid,
            "failure_reasons": list(self.failure_reasons),
            "predicted_contact_families": sorted(self.contacts),
            "predicted_contacts": {
                name: value.as_dict() for name, value in sorted(self.contacts.items())
            },
            "object_inside_three_finger_envelope": self.object_inside_three_finger_envelope,
            "three_finger_enclosure_valid": self.valid,
            "palm_clearance_m": self.palm_clearance_m,
            "table_clearance_m": self.table_clearance_m,
            "cross_finger_clearance_m": self.cross_finger_clearance_m,
            "projected_contact_area_m2": self.projected_contact_area_m2,
        }


@dataclass(frozen=True)
class GraspCandidate:
    family: str
    direction: str
    roll_deg: float
    tilt_deg: float
    object_center_axial_offset_m: float
    side_height_m: float
    center_offset_hand_m: np.ndarray
    T_world_hand: np.ndarray
    T_world_grasp_center: np.ndarray
    T_world_tool0: np.ndarray
    object_obb_hand: OBB
    enclosure: EnclosureResult

    def as_dict(self):
        return {
            "family": self.family,
            "direction": self.direction,
            "roll_deg": self.roll_deg,
            "tilt_deg": self.tilt_deg,
            "object_center_axial_offset_m": self.object_center_axial_offset_m,
            "side_height_m": self.side_height_m,
            "center_offset_hand_m": self.center_offset_hand_m.tolist(),
            "T_world_hand": self.T_world_hand.tolist(),
            "T_world_grasp_center": self.T_world_grasp_center.tolist(),
            "T_world_tool0": self.T_world_tool0.tolist(),
            "enclosure": self.enclosure.as_dict(),
        }


def _segment_segment_distance(a0, a1, b0, b1):
    """Exact Euclidean distance between two finite 3-D segments."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    denom = aa * cc - bb * bb
    s_num, s_den = denom, denom
    t_num, t_den = denom, denom
    if denom < EPS:
        s_num, s_den = 0.0, 1.0
        t_num, t_den = ee, cc
    else:
        s_num = bb * ee - cc * dd
        t_num = aa * ee - bb * dd
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, ee, cc
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, ee + bb, cc
    if t_num < 0.0:
        t_num = 0.0
        if -dd < 0.0:
            s_num = 0.0
        elif -dd > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd, aa
    elif t_num > t_den:
        t_num = t_den
        if -dd + bb < 0.0:
            s_num = 0.0
        elif -dd + bb > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd + bb, aa
    sc = 0.0 if abs(s_num) < EPS else s_num / s_den
    tc = 0.0 if abs(t_num) < EPS else t_num / t_den
    return float(np.linalg.norm(w + sc * u - tc * v))


def capsule_to_obb(capsule, obb, axis_samples):
    """Return signed distance and an OBB surface contact estimate.

    Cylinder axes are sampled densely, while their radius remains explicit;
    this is materially different from treating an entire finger as one wide
    point-distance threshold.  All calculations are in the OBB local frame.
    """
    axis_samples = int(axis_samples)
    if axis_samples < 3:
        raise ValueError("capsule_axis_samples must be at least 3")
    alpha = np.linspace(0.0, 1.0, axis_samples)[:, None]
    points_hand = capsule.first + alpha * (capsule.second - capsule.first)
    points_local = (obb.rotation.T @ (points_hand - obb.center).T).T
    clamped = np.minimum(np.maximum(points_local, -obb.half_extents), obb.half_extents)
    outside_vectors = points_local - clamped
    outside_distances = np.linalg.norm(outside_vectors, axis=1)
    inside = np.all(np.abs(points_local) <= obb.half_extents + EPS, axis=1)
    signed = outside_distances - float(capsule.radius)
    if np.any(inside):
        inside_indices = np.flatnonzero(inside)
        margins = obb.half_extents - np.abs(points_local[inside_indices])
        nearest_margins = np.min(margins, axis=1)
        signed[inside_indices] = -(nearest_margins + float(capsule.radius))
    index = int(np.argmin(signed))
    point_local = points_local[index]
    surface_local = clamped[index].copy()
    if inside[index]:
        margin = obb.half_extents - np.abs(point_local)
        face_axis = int(np.argmin(margin))
        face_sign = 1 if point_local[face_axis] >= 0.0 else -1
        surface_local[face_axis] = face_sign * obb.half_extents[face_axis]
        normal_local = np.zeros(3, dtype=float)
        normal_local[face_axis] = float(face_sign)
    else:
        delta = point_local - surface_local
        norm = float(np.linalg.norm(delta))
        if norm < EPS:
            margin = obb.half_extents - np.abs(point_local)
            face_axis = int(np.argmin(margin))
            face_sign = 1 if point_local[face_axis] >= 0.0 else -1
            normal_local = np.zeros(3, dtype=float)
            normal_local[face_axis] = float(face_sign)
        else:
            normal_local = delta / norm
            face_axis = int(np.argmax(np.abs(normal_local)))
            face_sign = 1 if normal_local[face_axis] >= 0.0 else -1
    return (
        float(signed[index]),
        obb.center + obb.rotation @ surface_local,
        obb.rotation @ normal_local,
        face_axis,
        face_sign,
    )


def obb_overlap(first, second, tolerance=1.0e-9):
    """Separating-axis OBB overlap test (15 axes)."""
    a = first.rotation
    b = second.rotation
    rotation = a.T @ b
    absolute = np.abs(rotation) + tolerance
    translation = a.T @ (second.center - first.center)
    for i in range(3):
        ra = first.half_extents[i]
        rb = float(second.half_extents @ absolute[i, :])
        if abs(translation[i]) > ra + rb:
            return False
    for j in range(3):
        ra = float(first.half_extents @ absolute[:, j])
        rb = second.half_extents[j]
        if abs(float(translation @ rotation[:, j])) > ra + rb:
            return False
    for i in range(3):
        for j in range(3):
            ra = (
                first.half_extents[(i + 1) % 3] * absolute[(i + 2) % 3, j]
                + first.half_extents[(i + 2) % 3] * absolute[(i + 1) % 3, j]
            )
            rb = (
                second.half_extents[(j + 1) % 3] * absolute[i, (j + 2) % 3]
                + second.half_extents[(j + 2) % 3] * absolute[i, (j + 1) % 3]
            )
            value = abs(
                translation[(i + 2) % 3] * rotation[(i + 1) % 3, j]
                - translation[(i + 1) % 3] * rotation[(i + 2) % 3, j]
            )
            if value > ra + rb:
                return False
    return True


def obb_principal_axis_clearance(first, second):
    """Conservative separation on the six face-normal axes.

    A positive result proves separation.  Zero means overlapping or that only
    a cross-product separating axis was found, which is intentionally treated
    fail-closed by the palm gate.
    """
    axes = [first.rotation[:, index] for index in range(3)] + [
        second.rotation[:, index] for index in range(3)
    ]
    delta = second.center - first.center
    gaps = []
    for axis in axes:
        first_radius = float(np.abs(first.rotation.T @ axis) @ first.half_extents)
        second_radius = float(np.abs(second.rotation.T @ axis) @ second.half_extents)
        gaps.append(abs(float(delta @ axis)) - first_radius - second_radius)
    return max(0.0, max(gaps))


def _triangle_encloses_origin(points, center, tolerance):
    relative = np.asarray(points, dtype=float)[:, :2] - np.asarray(center, dtype=float)[:2]
    first, second, third = relative
    twice_area = float(np.cross(second - first, third - first))
    area = abs(twice_area) * 0.5
    if area < EPS:
        return False, area
    matrix = np.column_stack((first - third, second - third))
    try:
        weights = np.linalg.solve(matrix, -third)
    except np.linalg.LinAlgError:
        return False, area
    barycentric = np.array([weights[0], weights[1], 1.0 - weights.sum()])
    characteristic = max(float(np.linalg.norm(relative[i] - relative[j])) for i in range(3) for j in range(i))
    barycentric_tolerance = tolerance / max(characteristic, EPS)
    return bool(np.min(barycentric) >= -barycentric_tolerance), area


class HandGeometry:
    """URDF-derived fixed-palm kinematics and enclosure evaluator."""

    def __init__(self, urdf_path, config_path):
        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        if self.config.get("schema_version") != 1:
            raise ValueError("unsupported three-finger geometry schema")
        with open(urdf_path, "r", encoding="utf-8") as stream:
            self.robot = URDF.from_xml_string(stream.read())
        self.parents = {
            joint.child: (joint.parent, joint) for joint in self.robot.joints
        }
        self.hand_frame = self.config["frames"]["hand"]
        self.tool_frame = self.config["frames"]["tool"]
        fixed = self.config["fixed_palm"]
        self.open_joints = {fixed["joint"]: float(fixed["value_rad"])}
        self.open_joints.update(
            {name: float(value) for name, value in fixed["open_flexion_rad"].items()}
        )
        self.close_joints = {fixed["joint"]: float(fixed["value_rad"])}
        self.close_joints.update(
            {name: float(value) for name, value in fixed["close_flexion_rad"].items()}
        )
        self.closure_samples = int(fixed["closure_samples"])
        if self.closure_samples < 20:
            raise ValueError("closure_samples must be at least 20")
        finger = self.config["finger_geometry"]
        self.families = {
            name: tuple(links) for name, links in finger["families"].items()
        }
        if set(self.families) != {"f1", "f2", "f3"}:
            raise ValueError("finger families must be exactly f1/f2/f3")
        self.distal_pad_links = dict(finger["distal_pad_links"])
        self.non_contact_links = tuple(finger["non_contact_links"])
        self.axis_samples = int(finger["capsule_axis_samples"])
        self._validate_links()
        self.T_tool_hand = self.relative_transform(
            self.tool_frame, self.hand_frame, self.open_joints
        )
        self.T_hand_grasp_center = self._derive_grasp_center()
        self.T_tool_grasp_center = self.T_tool_hand @ self.T_hand_grasp_center
        self._validate_configured_grasp_center()
        self._closure_sweep = self._build_closure_sweep()

    def _validate_links(self):
        names = set(self.robot.link_map)
        required = {self.hand_frame, self.tool_frame}
        required.update(self.non_contact_links)
        for links in self.families.values():
            required.update(links)
        required.update(self.distal_pad_links.values())
        missing = sorted(required - names)
        if missing:
            raise ValueError("URDF is missing required links: {}".format(missing))

    def _joint_values_with_mimics(self, values):
        result = {
            joint.name: 0.0
            for joint in self.robot.joints
            if joint.type in ("revolute", "continuous", "prismatic")
        }
        result.update({name: float(value) for name, value in values.items()})
        for joint in self.robot.joints:
            if joint.mimic is not None:
                result[joint.name] = (
                    result[joint.mimic.joint] * float(joint.mimic.multiplier)
                    + float(joint.mimic.offset)
                )
        return result

    def root_transforms(self, values):
        joint_values = self._joint_values_with_mimics(values)
        memo = {}

        def calculate(link):
            if link in memo:
                return memo[link]
            if link not in self.parents:
                memo[link] = np.eye(4, dtype=float)
                return memo[link]
            parent, joint = self.parents[link]
            current = calculate(parent) @ origin_transform(joint.origin)
            if joint.type in ("revolute", "continuous"):
                motion = transform(
                    _rotation_axis_angle(joint.axis, joint_values[joint.name]),
                    None,
                )
                current = current @ motion
            elif joint.type == "prismatic":
                current = current @ transform(
                    None,
                    np.asarray(joint.axis, dtype=float) * joint_values[joint.name],
                )
            memo[link] = current
            return current

        for link in self.robot.link_map:
            calculate(link)
        return memo

    def relative_transform(self, reference, target, values):
        root = self.root_transforms(values)
        return np.linalg.inv(root[reference]) @ root[target]

    def joint_values_at(self, closure_fraction):
        fraction = float(closure_fraction)
        if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError("closure fraction must be in [0, 1]")
        return {
            name: self.open_joints[name]
            + fraction * (self.close_joints[name] - self.open_joints[name])
            for name in self.open_joints
        }

    def _collision_proxies(self, links, values, family=""):
        roots = self.root_transforms(values)
        T_root_hand = roots[self.hand_frame]
        T_hand_root = np.linalg.inv(T_root_hand)
        capsules, boxes = [], []
        for link_name in links:
            T_hand_link = T_hand_root @ roots[link_name]
            for collision in self.robot.link_map[link_name].collisions:
                geometry = collision.geometry
                T_hand_geometry = T_hand_link @ origin_transform(collision.origin)
                if isinstance(geometry, Sphere):
                    center = T_hand_geometry[:3, 3]
                    capsules.append(
                        Capsule(center, center, float(geometry.radius), link_name, family)
                    )
                elif isinstance(geometry, Cylinder) and family:
                    axis = T_hand_geometry[:3, :3] @ np.array([0.0, 0.0, 1.0])
                    half = 0.5 * float(geometry.length)
                    center = T_hand_geometry[:3, 3]
                    capsules.append(
                        Capsule(
                            center - half * axis,
                            center + half * axis,
                            float(geometry.radius),
                            link_name,
                            family,
                        )
                    )
                elif isinstance(geometry, Cylinder):
                    # A bare URDF cylinder has flat ends.  Treating the large
                    # handbase cylinder as a capsule would invent 69 mm
                    # hemispherical caps and falsely report palm collision.
                    # Its circumscribed OBB preserves the finite axial extent
                    # and is conservative only in the radial cross-section.
                    boxes.append(
                        BoxProxy(
                            T_hand_geometry[:3, 3],
                            validate_rotation(T_hand_geometry[:3, :3]),
                            np.array(
                                [
                                    float(geometry.radius),
                                    float(geometry.radius),
                                    0.5 * float(geometry.length),
                                ]
                            ),
                            link_name,
                        )
                    )
                elif isinstance(geometry, Box):
                    boxes.append(
                        BoxProxy(
                            T_hand_geometry[:3, 3],
                            validate_rotation(T_hand_geometry[:3, :3]),
                            0.5 * np.asarray(geometry.size, dtype=float),
                            link_name,
                        )
                    )
                else:
                    raise ValueError(
                        "unsupported collision geometry {} on {}".format(
                            type(geometry).__name__, link_name
                        )
                    )
        return capsules, boxes

    def _derive_grasp_center(self):
        reference = float(
            self.config["grasp_center"]["reference_closure_fraction"]
        )
        values = self.joint_values_at(reference)
        centers = []
        for family in ("f1", "f2", "f3"):
            link = self.distal_pad_links[family]
            capsules, _ = self._collision_proxies((link,), values, family)
            cylinders = [item for item in capsules if np.linalg.norm(item.second - item.first) > EPS]
            if len(cylinders) != 1:
                raise ValueError(
                    "{} must have exactly one distal cylinder proxy".format(link)
                )
            centers.append(0.5 * (cylinders[0].first + cylinders[0].second))
        result = np.eye(4, dtype=float)
        result[:3, 3] = np.mean(np.asarray(centers), axis=0)
        return result

    def _validate_configured_grasp_center(self):
        section = self.config["grasp_center"]
        tolerance = float(section["validation_tolerance_m"])
        expected_hand = _finite_array(
            section["expected_handbase_xyz_m"], (3,), "expected hand grasp center"
        )
        expected_tool = _finite_array(
            section["expected_tool0_xyz_m"], (3,), "expected tool grasp center"
        )
        hand_error = float(np.linalg.norm(self.T_hand_grasp_center[:3, 3] - expected_hand))
        tool_error = float(np.linalg.norm(self.T_tool_grasp_center[:3, 3] - expected_tool))
        if hand_error > tolerance or tool_error > tolerance:
            raise ValueError(
                "URDF-derived grasp center drifted: hand_error={:.6f}m, "
                "tool_error={:.6f}m".format(hand_error, tool_error)
            )

    def transform_summary(self):
        return {
            "definition": "centroid_of_three_distal_collision_cylinders_at_half_closure",
            "T_handbase_grasp_center": self.T_hand_grasp_center.tolist(),
            "T_tool0_grasp_center": self.T_tool_grasp_center.tolist(),
            "fixed_palm_joint_rad": self.open_joints[
                self.config["fixed_palm"]["joint"]
            ],
        }

    def _build_closure_sweep(self):
        """Cache URDF FK/proxies so candidate searches remain bounded."""
        sweep = []
        for fraction in np.linspace(0.0, 1.0, self.closure_samples):
            values = self.joint_values_at(float(fraction))
            family_capsules = {}
            for family, links in self.families.items():
                capsules, _ = self._collision_proxies(links, values, family)
                family_capsules[family] = tuple(capsules)
            palm_capsules, palm_boxes = self._collision_proxies(
                self.non_contact_links, values
            )
            sweep.append(
                (
                    float(fraction),
                    family_capsules,
                    tuple(palm_capsules),
                    tuple(palm_boxes),
                )
            )
        return tuple(sweep)

    def evaluate_enclosure(self, object_obb_hand, T_world_hand=None, table_z=None):
        if not isinstance(object_obb_hand, OBB):
            raise TypeError("object_obb_hand must be OBB")
        if (T_world_hand is None) != (table_z is None):
            raise ValueError("T_world_hand and table_z must be supplied together")
        if T_world_hand is not None:
            T_world_hand = _finite_array(T_world_hand, (4, 4), "T_world_hand")
            validate_rotation(T_world_hand[:3, :3], "T_world_hand rotation")
            table_z = float(table_z)
            if not math.isfinite(table_z):
                raise ValueError("table_z must be finite")

        limits = self.config["contact_geometry"]
        contact_tolerance = float(limits["contact_tolerance_m"])
        required_open_clearance = float(limits["required_open_clearance_m"])
        contacts = {}
        reasons = []
        minimum_palm_clearance = math.inf
        minimum_cross_clearance = math.inf
        minimum_table_clearance = math.inf
        for sample_index, sweep_sample in enumerate(self._closure_sweep):
            fraction, family_capsules, palm_capsules, palm_boxes = sweep_sample
            for family in self.families:
                capsules = family_capsules[family]
                best = None
                for capsule in capsules:
                    estimate = capsule_to_obb(capsule, object_obb_hand, self.axis_samples)
                    if best is None or estimate[0] < best[0]:
                        best = estimate + (capsule,)
                if sample_index == 0 and best[0] < required_open_clearance:
                    reasons.append("{}_OPEN_COLLISION".format(family.upper()))
                if family not in contacts and best[0] <= contact_tolerance:
                    contacts[family] = ContactPrediction(
                        family=family,
                        link=best[5].link,
                        closure_fraction=float(fraction),
                        signed_distance_m=float(best[0]),
                        point_hand_m=np.asarray(best[1], dtype=float),
                        normal_hand=np.asarray(best[2], dtype=float),
                        face_axis=int(best[3]),
                        face_sign=int(best[4]),
                    )

            for capsule in palm_capsules:
                estimate = capsule_to_obb(capsule, object_obb_hand, self.axis_samples)
                minimum_palm_clearance = min(minimum_palm_clearance, estimate[0])
            for box in palm_boxes:
                palm_obb = OBB(box.center, box.rotation, box.half_extents)
                if obb_overlap(palm_obb, object_obb_hand):
                    minimum_palm_clearance = min(minimum_palm_clearance, -0.0)
                else:
                    minimum_palm_clearance = min(
                        minimum_palm_clearance,
                        obb_principal_axis_clearance(palm_obb, object_obb_hand),
                    )

            families = ("f1", "f2", "f3")
            for first_index, first_family in enumerate(families):
                for second_family in families[first_index + 1 :]:
                    for first in family_capsules[first_family]:
                        for second in family_capsules[second_family]:
                            clearance = _segment_segment_distance(
                                first.first, first.second, second.first, second.second
                            ) - first.radius - second.radius
                            minimum_cross_clearance = min(
                                minimum_cross_clearance, clearance
                            )

            if T_world_hand is not None:
                all_capsules = list(palm_capsules) + [
                    capsule
                    for family in families
                    for capsule in family_capsules[family]
                ]
                for capsule in all_capsules:
                    first_world = T_world_hand[:3, :3] @ capsule.first + T_world_hand[:3, 3]
                    second_world = T_world_hand[:3, :3] @ capsule.second + T_world_hand[:3, 3]
                    clearance = min(first_world[2], second_world[2]) - capsule.radius - table_z
                    minimum_table_clearance = min(minimum_table_clearance, clearance)
                for box in palm_boxes:
                    half_z_extent = float(
                        np.abs((T_world_hand[:3, :3] @ box.rotation)[2, :])
                        @ box.half_extents
                    )
                    center_world = T_world_hand[:3, :3] @ box.center + T_world_hand[:3, 3]
                    minimum_table_clearance = min(
                        minimum_table_clearance,
                        float(center_world[2] - half_z_extent - table_z),
                    )

        required = tuple(limits["required_finger_families"])
        for family in required:
            if family not in contacts:
                reasons.append("{}_NO_CONTACT".format(family.upper()))

        if minimum_palm_clearance < float(limits["minimum_palm_object_clearance_m"]):
            reasons.append("PALM_OBJECT_COLLISION")
        if minimum_cross_clearance < float(limits["minimum_cross_finger_clearance_m"]):
            reasons.append("SELF_COLLISION")
        if T_world_hand is not None and minimum_table_clearance < float(
            limits["minimum_table_clearance_m"]
        ):
            reasons.append("TABLE_COLLISION")

        inside, area = False, 0.0
        if set(contacts) == set(required):
            fractions = [contacts[name].closure_fraction for name in required]
            if max(fractions) - min(fractions) > float(
                limits["maximum_contact_fraction_spread"]
            ):
                reasons.append("CONTACT_FRACTION_SPREAD")
            faces = {
                (contacts[name].face_axis, contacts[name].face_sign)
                for name in required
            }
            if len(faces) != 3:
                reasons.append("CONTACT_SURFACES_NOT_DISTINCT")
            inside, area = _triangle_encloses_origin(
                [contacts[name].point_hand_m for name in required],
                object_obb_hand.center,
                float(limits["envelope_tolerance_m"]),
            )
            if area < float(limits["minimum_projected_triangle_area_m2"]):
                reasons.append("CONTACT_TRIANGLE_DEGENERATE")
                inside = False
            if not inside:
                reasons.append("OBJECT_OUTSIDE_ENVELOPE")

        reasons = tuple(sorted(set(reasons)))
        return EnclosureResult(
            valid=not reasons,
            failure_reasons=reasons,
            contacts=contacts,
            object_inside_three_finger_envelope=inside,
            palm_clearance_m=float(minimum_palm_clearance),
            table_clearance_m=float(minimum_table_clearance),
            cross_finger_clearance_m=float(minimum_cross_clearance),
            projected_contact_area_m2=float(area),
        )

    def minimum_table_clearance(self, T_world_hand, table_z):
        """Exact support-height gate for cached capsules and box proxies."""
        T_world_hand = _finite_array(T_world_hand, (4, 4), "T_world_hand")
        validate_rotation(T_world_hand[:3, :3], "T_world_hand rotation")
        table_z = float(table_z)
        if not math.isfinite(table_z):
            raise ValueError("table_z must be finite")
        minimum = math.inf
        for _, families, palm_capsules, palm_boxes in self._closure_sweep:
            capsules = list(palm_capsules) + [
                capsule for items in families.values() for capsule in items
            ]
            for capsule in capsules:
                first_world = (
                    T_world_hand[:3, :3] @ capsule.first + T_world_hand[:3, 3]
                )
                second_world = (
                    T_world_hand[:3, :3] @ capsule.second + T_world_hand[:3, 3]
                )
                minimum = min(
                    minimum,
                    min(first_world[2], second_world[2])
                    - capsule.radius
                    - table_z,
                )
            for box in palm_boxes:
                world_rotation = T_world_hand[:3, :3] @ box.rotation
                half_z = float(np.abs(world_rotation[2, :]) @ box.half_extents)
                center_world = (
                    T_world_hand[:3, :3] @ box.center + T_world_hand[:3, 3]
                )
                minimum = min(minimum, float(center_world[2] - half_z - table_z))
        return float(minimum)

    def make_candidate(
        self,
        T_world_object,
        object_size_m,
        table_z,
        family,
        direction,
        roll_deg,
        planar_offset_hand_m=(0.0, 0.0),
        side_height_m=0.0,
        tilt_deg=0.0,
        object_center_axial_offset_m=None,
    ):
        """Construct one object-relative hand/tool candidate and evaluate it."""
        T_world_object = _finite_array(
            T_world_object, (4, 4), "T_world_object"
        )
        validate_rotation(T_world_object[:3, :3], "object rotation")
        size = _finite_array(object_size_m, (3,), "object size")
        if np.any(size <= 0.0):
            raise ValueError("object size must be positive")
        roll = math.radians(float(roll_deg))
        c_roll, s_roll = math.cos(roll), math.sin(roll)
        R_roll = np.array(
            [[c_roll, -s_roll, 0.0], [s_roll, c_roll, 0.0], [0.0, 0.0, 1.0]]
        )
        if family in ("top_down", "top_oblique"):
            if direction != "object_pos_z":
                raise ValueError("top approach direction must be object_pos_z")
            # Hand +z points from above toward the object (-object z).
            R_object_hand_base = np.diag([1.0, -1.0, -1.0])
        elif family == "side":
            bases = {
                # Columns are hand x/y/z expressed in the object frame.
                "object_pos_x": np.array(
                    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
                ),
                "object_neg_x": np.array(
                    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
                ),
                "object_pos_y": np.array(
                    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
                ),
                "object_neg_y": np.array(
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
                ),
            }
            if direction not in bases:
                raise ValueError("unknown side direction {}".format(direction))
            R_object_hand_base = bases[direction]
        else:
            raise ValueError("family must be top_down, top_oblique or side")
        if family == "top_down" and abs(float(tilt_deg)) > 1.0e-12:
            raise ValueError("strict top_down cannot have a tilt")
        if family == "top_oblique":
            tilt = math.radians(float(tilt_deg))
            R_tilt = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(tilt), -math.sin(tilt)],
                    [0.0, math.sin(tilt), math.cos(tilt)],
                ]
            )
        else:
            R_tilt = np.eye(3)
        R_object_hand = validate_rotation(
            R_object_hand_base @ R_roll @ R_tilt,
            "object-to-hand candidate rotation",
        )
        R_world_hand = T_world_object[:3, :3] @ R_object_hand

        planar = _finite_array(planar_offset_hand_m, (2,), "planar offset")
        object_center_hand = self.T_hand_grasp_center[:3, 3].copy()
        object_center_hand[:2] += planar
        if family in ("top_down", "top_oblique"):
            if object_center_axial_offset_m is None:
                object_center_axial_offset_m = float(
                    self.config["search"][
                        "top_object_center_in_grasp_frame_z_m"
                    ]
                )
            object_center_hand[2] += float(object_center_axial_offset_m)
        else:
            # A positive side-height places the grasp center above the object.
            world_up_hand = R_world_hand.T @ np.array([0.0, 0.0, 1.0])
            object_center_hand -= float(side_height_m) * world_up_hand
        hand_translation = (
            T_world_object[:3, 3] - R_world_hand @ object_center_hand
        )
        T_world_hand = transform(R_world_hand, hand_translation)
        R_hand_object = R_world_hand.T @ T_world_object[:3, :3]
        object_obb = OBB(object_center_hand, R_hand_object, 0.5 * size)
        table_clearance = self.minimum_table_clearance(T_world_hand, table_z)
        minimum_table = float(
            self.config["contact_geometry"]["minimum_table_clearance_m"]
        )
        if table_clearance < minimum_table:
            enclosure = EnclosureResult(
                valid=False,
                failure_reasons=("TABLE_COLLISION",),
                contacts={},
                object_inside_three_finger_envelope=False,
                palm_clearance_m=math.inf,
                table_clearance_m=table_clearance,
                cross_finger_clearance_m=math.inf,
                projected_contact_area_m2=0.0,
            )
        else:
            enclosure = self.evaluate_enclosure(object_obb, T_world_hand, table_z)
        T_world_grasp_center = T_world_hand @ self.T_hand_grasp_center
        T_world_tool0 = T_world_grasp_center @ np.linalg.inv(
            self.T_tool_grasp_center
        )
        return GraspCandidate(
            family=family,
            direction=direction,
            roll_deg=float(roll_deg),
            tilt_deg=float(tilt_deg),
            object_center_axial_offset_m=float(
                object_center_axial_offset_m or 0.0
            ),
            side_height_m=float(side_height_m),
            center_offset_hand_m=object_center_hand
            - self.T_hand_grasp_center[:3, 3],
            T_world_hand=T_world_hand,
            T_world_grasp_center=T_world_grasp_center,
            T_world_tool0=T_world_tool0,
            object_obb_hand=object_obb,
            enclosure=enclosure,
        )

    def coarse_geometry_candidates(
        self, T_world_object, object_size_m, table_z, family="auto"
    ):
        """Bounded 15-degree object-relative geometry search.

        This stage deliberately performs no IK.  MoveIt validation is a later
        hard gate and must not be approximated by changing joint_6 directly.
        """
        configured = set(self.config["search"]["grasp_families"])
        known = {"top_down", "top_oblique", "side", "auto"}
        if family not in known:
            raise ValueError("unsupported grasp family {}".format(family))
        explicitly_enabled = configured & {"top_down", "top_oblique", "side"}
        requested = explicitly_enabled if family == "auto" else {family}
        if family != "auto" and family not in explicitly_enabled:
            raise ValueError(
                "grasp family {} is not enabled by search configuration".format(
                    family
                )
            )
        if family == "auto" and not explicitly_enabled:
            raise ValueError("auto grasp search has no enabled concrete families")
        directions = []
        if "top_down" in requested:
            directions.append(("top_down", "object_pos_z"))
        if "top_oblique" in requested:
            directions.append(("top_oblique", "object_pos_z"))
        if "side" in requested:
            directions.extend(
                ("side", item)
                for item in (
                    "object_pos_x",
                    "object_neg_x",
                    "object_pos_y",
                    "object_neg_y",
                )
            )
        offsets = [float(value) for value in self.config["search"]["center_offsets_m"]]
        step = int(self.config["search"]["coarse_roll_step_deg"])
        heights = [float(value) for value in self.config["search"]["side_height_offsets_m"]]
        results = []
        for candidate_family, direction in directions:
            family_heights = (
                (0.0,)
                if candidate_family in ("top_down", "top_oblique")
                else heights
            )
            if candidate_family == "top_oblique":
                tilt_axial_pairs = [
                    (
                        float(item["tilt_deg"]),
                        float(item["object_center_axial_offset_m"]),
                    )
                    for item in self.config["search"][
                        "top_oblique_tilt_axial_pairs"
                    ]
                ]
                # These are audited x/y pairs, not independent axis values.
                # Keeping them paired avoids an arbitrary Cartesian product
                # while allowing the table-clearance compensation to reach IK.
                planar_offsets = [
                    tuple(float(value) for value in pair)
                    for pair in self.config["search"]
                    ["top_oblique_planar_offsets_m"]
                ]
            else:
                tilt_axial_pairs = [(0.0, None)]
                planar_offsets = [
                    (first, second) for first in offsets for second in offsets
                ]
            for roll_deg in range(0, 360, step):
                for planar_offset in planar_offsets:
                    for height in family_heights:
                        for tilt_deg, axial_offset in tilt_axial_pairs:
                            results.append(
                                self.make_candidate(
                                    T_world_object,
                                    object_size_m,
                                    table_z,
                                    candidate_family,
                                    direction,
                                    roll_deg,
                                    planar_offset,
                                    height,
                                    tilt_deg,
                                    axial_offset,
                                )
                            )
        return results


def load_hand_geometry(urdf_path, config_path):
    return HandGeometry(urdf_path, config_path)
