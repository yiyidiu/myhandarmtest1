#!/usr/bin/env python3
"""Robust MANO wrist-opening reference adapted from teleoperation core V9.

The control origin is the centre of MANO's open wrist boundary (16 vertices
for the licensed HaMeR MANO topology).  Orientation is the robust similarity
fit of that ring against a neutral, side-specific template.  Finger joints
are deliberately not used to define the control axes.

This module contains geometry only: it does not hold a pose through missing
observations and it does not send robot commands.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .palm_frame import (
    PalmFrameError,
    align_quaternion_sign,
    project_to_so3,
    require_so3,
    rotation_matrix_to_quaternion_xyzw,
)


EXPECTED_MANO_VERTICES = 778
EXPECTED_WRIST_RING_VERTICES = 16
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
LITTLE_MCP = 17


def _normalized(vector: Any, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1.0e-10:
        raise ValueError("cannot normalize {}".format(name))
    return value / norm


def find_boundary_edges(faces: Any) -> np.ndarray:
    """Return triangle edges that occur exactly once."""

    triangles = np.asarray(faces, dtype=np.int64)
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(triangles) == 0
        or np.min(triangles) < 0
    ):
        raise ValueError("faces must be a non-empty Nx3 index array")
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return unique_edges[counts == 1]


def ordered_largest_boundary_loop(boundary_edges: Any) -> np.ndarray:
    """Order the largest connected open-mesh boundary without guessing IDs."""

    edges = np.asarray(boundary_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("boundary_edges must be Nx2")
    adjacency: Dict[int, list] = defaultdict(list)
    for first, second in edges:
        first_value, second_value = int(first), int(second)
        adjacency[first_value].append(second_value)
        adjacency[second_value].append(first_value)
    if not adjacency:
        raise ValueError("MANO mesh has no open boundary")

    unseen = set(adjacency)
    components = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)
    component = max(components, key=len)
    component_set = set(component)
    if any(
        len([item for item in adjacency[index] if item in component_set]) != 2
        for index in component
    ):
        raise ValueError("largest MANO boundary is not a simple closed loop")
    ordered = [min(component)]
    previous = None
    current = ordered[0]
    while len(ordered) < len(component):
        candidates = [
            item
            for item in adjacency[current]
            if item in component_set and item != previous
        ]
        next_item = next((item for item in candidates if item not in ordered), None)
        if next_item is None:
            raise ValueError("could not order MANO wrist boundary loop")
        ordered.append(next_item)
        previous, current = current, next_item
    return np.asarray(ordered, dtype=np.int64)


def ring_geometry(points: Any) -> Dict[str, Any]:
    ring = np.asarray(points, dtype=np.float64)
    if (
        ring.ndim != 2
        or ring.shape[1] != 3
        or len(ring) < 3
        or not np.all(np.isfinite(ring))
    ):
        raise ValueError("ring must be a finite Nx3 point array")
    center = ring.mean(axis=0)
    centered = ring - center
    _, singular_values, vh_matrix = np.linalg.svd(centered, full_matrices=False)
    normal = _normalized(vh_matrix[-1], "ring normal")
    distances = centered @ normal
    pairwise = ring[:, None, :] - ring[None, :, :]
    diameter = float(np.max(np.linalg.norm(pairwise, axis=2)))
    if not math.isfinite(diameter) or diameter < 1.0e-9:
        raise ValueError("wrist ring diameter is degenerate")
    plane_rms = float(np.sqrt(np.mean(np.square(distances))))
    plane_max = float(np.max(np.abs(distances)))
    return {
        "center": center,
        "normal": normal,
        "singular_values": singular_values,
        "principal_directions": vh_matrix,
        "diameter": diameter,
        "plane_rms": plane_rms,
        "plane_max": plane_max,
        "plane_rms_ratio": plane_rms / diameter,
        "plane_max_ratio": plane_max / diameter,
    }


def weighted_similarity_kabsch(
    reference_points: Any,
    current_points: Any,
    weights: Optional[Any] = None,
) -> Dict[str, Any]:
    reference = np.asarray(reference_points, dtype=np.float64)
    current = np.asarray(current_points, dtype=np.float64)
    if (
        reference.shape != current.shape
        or reference.ndim != 2
        or reference.shape[1] != 3
        or len(reference) < 3
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
    ):
        raise ValueError("Kabsch inputs must be matching finite Nx3 arrays")
    if weights is None:
        weight = np.ones(len(reference), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(len(reference))
    weight = np.where(np.isfinite(weight), np.maximum(weight, 0.0), 0.0)
    if float(weight.sum()) <= 1.0e-9:
        raise ValueError("Kabsch weights contain no positive mass")
    weight /= float(weight.sum())
    reference_center = np.sum(reference * weight[:, None], axis=0)
    current_center = np.sum(current * weight[:, None], axis=0)
    reference_zero = reference - reference_center
    current_zero = current - current_center
    covariance = reference_zero.T @ (current_zero * weight[:, None])
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(vt_matrix.T @ u_matrix.T) < 0.0:
        correction[-1, -1] = -1.0
    rotation = project_to_so3(vt_matrix.T @ correction @ u_matrix.T)
    denominator = float(np.sum(weight * np.sum(reference_zero**2, axis=1)))
    scale = float(
        np.sum(singular_values * np.diag(correction))
        / max(denominator, 1.0e-12)
    )
    translation = current_center - scale * rotation @ reference_center
    predicted = (scale * (rotation @ reference.T)).T + translation
    residuals = np.linalg.norm(current - predicted, axis=1)
    return {
        "rotation": rotation,
        "scale": scale,
        "translation": translation,
        "residuals": residuals,
    }


def robust_similarity_kabsch(
    reference_points: Any,
    current_points: Any,
    *,
    huber_scale: float = 1.5,
    iterations: int = 5,
) -> Dict[str, Any]:
    """IRLS/Huber fit so one deformed wrist vertex cannot turn the frame."""

    reference = np.asarray(reference_points, dtype=np.float64)
    current = np.asarray(current_points, dtype=np.float64)
    if reference.shape != current.shape or reference.ndim != 2:
        raise ValueError("Kabsch point shapes do not match")
    base = np.ones(len(reference), dtype=np.float64) / float(len(reference))
    weights = base.copy()
    fit: Dict[str, Any] = {}
    for _ in range(max(1, int(iterations))):
        fit = weighted_similarity_kabsch(reference, current, weights)
        residuals = np.asarray(fit["residuals"], dtype=np.float64)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_sigma = max(1.4826 * mad, 1.0e-6)
        normalized_residual = residuals / (
            max(float(huber_scale), 0.1) * robust_sigma
        )
        huber = np.ones_like(normalized_residual)
        large = normalized_residual > 1.0
        huber[large] = 1.0 / normalized_residual[large]
        weights = base * huber
        weights /= max(float(weights.sum()), 1.0e-12)
    residuals = np.asarray(fit["residuals"], dtype=np.float64)
    fit["weights"] = weights
    fit["weighted_rms"] = float(
        np.sqrt(np.sum(weights * np.square(residuals)))
    )
    return fit


@dataclass(frozen=True)
class ManoWristDefinition:
    is_right: bool
    vertices: np.ndarray
    joints: np.ndarray
    wrist_loop: np.ndarray
    ring: np.ndarray
    frame: np.ndarray
    center: np.ndarray
    diameter: float


@dataclass(frozen=True)
class ManoWristFrameResult:
    valid: bool
    rotation: Optional[np.ndarray]
    quaternion_xyzw: Optional[np.ndarray]
    origin: Optional[np.ndarray]
    is_right: bool
    failure_reason: str
    quality: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "rotation": None if self.rotation is None else self.rotation.tolist(),
            "quaternion_xyzw": (
                None
                if self.quaternion_xyzw is None
                else self.quaternion_xyzw.tolist()
            ),
            "origin": None if self.origin is None else self.origin.tolist(),
            "is_right": bool(self.is_right),
            "failure_reason": str(self.failure_reason),
            "quality": dict(self.quality),
            "reference_kind": "MANO_WRIST_RING_16",
            "definition": (
                "origin=mean(16 MANO wrist-opening vertices); "
                "orientation=robust similarity Kabsch(neutral ring,current ring)"
            ),
            "global_orient_used": False,
            "finger_joints_used_for_live_axes": False,
        }


def build_mano_wrist_definition(
    neutral_vertices_source: Any,
    neutral_joints_source: Any,
    faces: Any,
    is_right: bool,
) -> ManoWristDefinition:
    """Construct the neutral wrist-ring reference for one physical hand side."""

    vertices = np.asarray(neutral_vertices_source, dtype=np.float64)
    joints = np.asarray(neutral_joints_source, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.shape != (EXPECTED_MANO_VERTICES, 3) or not np.all(np.isfinite(vertices)):
        raise ValueError("neutral MANO vertices must be finite 778x3")
    if joints.ndim != 2 or joints.shape[0] < 18 or joints.shape[1] != 3:
        raise ValueError("neutral MANO joints must contain wrist and MCP joints")
    loop = ordered_largest_boundary_loop(find_boundary_edges(triangles))
    if len(loop) != EXPECTED_WRIST_RING_VERTICES:
        raise ValueError(
            "expected 16 MANO wrist boundary vertices, got {}".format(len(loop))
        )
    if int(np.max(loop)) >= len(vertices):
        raise ValueError("wrist boundary contains an invalid MANO vertex index")
    ring = vertices[loop]
    geometry = ring_geometry(ring)
    center = np.asarray(geometry["center"], dtype=np.float64)
    principal = np.asarray(geometry["principal_directions"], dtype=np.float64)

    z_axis = _normalized(principal[-1], "neutral wrist +Z")
    longitudinal_hint = joints[MIDDLE_MCP] - joints[WRIST]
    if float(np.dot(z_axis, longitudinal_hint)) < 0.0:
        z_axis = -z_axis
    y_axis = principal[0] - float(np.dot(principal[0], z_axis)) * z_axis
    y_axis = _normalized(y_axis, "neutral wrist +Y")
    width_hint = joints[INDEX_MCP] - joints[LITTLE_MCP]
    width_hint -= float(np.dot(width_hint, z_axis)) * z_axis
    width_hint = _normalized(width_hint, "neutral anatomical width")
    if float(np.dot(y_axis, width_hint)) < 0.0:
        y_axis = -y_axis
    x_axis = _normalized(np.cross(y_axis, z_axis), "neutral wrist +X")
    y_axis = _normalized(np.cross(z_axis, x_axis), "neutral wrist +Y")
    frame = require_so3(np.column_stack((x_axis, y_axis, z_axis)), atol=1.0e-6)
    return ManoWristDefinition(
        is_right=bool(is_right),
        vertices=vertices.copy(),
        joints=joints.copy(),
        wrist_loop=loop.copy(),
        ring=ring.copy(),
        frame=frame,
        center=center,
        diameter=float(geometry["diameter"]),
    )


def estimate_mano_wrist_frame(
    current_vertices_source: Any,
    definition: ManoWristDefinition,
    previous_quaternion_xyzw: Optional[Sequence[float]] = None,
) -> ManoWristFrameResult:
    """Estimate one live control frame from the 16-point MANO wrist ring."""

    vertices = np.asarray(current_vertices_source, dtype=np.float64)
    if vertices.shape != (EXPECTED_MANO_VERTICES, 3) or not np.all(np.isfinite(vertices)):
        return ManoWristFrameResult(
            False, None, None, None, definition.is_right,
            "vertices_must_be_finite_778x3", {},
        )
    try:
        current_ring = vertices[definition.wrist_loop]
        geometry = ring_geometry(current_ring)
        fit = robust_similarity_kabsch(definition.ring, current_ring)
        diameter = max(float(geometry["diameter"]), 1.0e-9)
        residual_ratio = float(fit["weighted_rms"]) / diameter
        plane_ratio = float(geometry["plane_rms_ratio"])
        rotation = require_so3(
            np.asarray(fit["rotation"], dtype=np.float64) @ definition.frame,
            atol=1.0e-6,
        )
        quaternion = align_quaternion_sign(
            rotation_matrix_to_quaternion_xyzw(rotation),
            previous_quaternion_xyzw,
        )
        plane_score = math.exp(-((max(plane_ratio, 0.0) / 0.018) ** 2))
        template_score = math.exp(-((max(residual_ratio, 0.0) / 0.024) ** 2))
        geometric_confidence = float(
            np.clip(math.sqrt(plane_score * template_score), 0.0, 1.0)
        )
        quality = {
            "wrist_loop_vertex_indices": definition.wrist_loop.tolist(),
            "wrist_loop_vertex_count": int(len(definition.wrist_loop)),
            "ring_diameter_model_units": diameter,
            "plane_rms_ratio": plane_ratio,
            "plane_max_ratio": float(geometry["plane_max_ratio"]),
            "template_residual_ratio": residual_ratio,
            "template_scale": float(fit["scale"]),
            "vertex_weights": np.asarray(fit["weights"]).tolist(),
            "vertex_residuals_model_units": np.asarray(fit["residuals"]).tolist(),
            "geometric_confidence": geometric_confidence,
            "origin_units": "mano_model_coordinates_not_d455_metric",
            "origin_definition": "mean_of_16_mano_wrist_opening_vertices",
            "orientation_definition": "irls_huber_similarity_kabsch_from_neutral_ring",
        }
        return ManoWristFrameResult(
            True,
            rotation,
            quaternion,
            np.asarray(geometry["center"], dtype=np.float64),
            definition.is_right,
            "",
            quality,
        )
    except (ValueError, PalmFrameError, np.linalg.LinAlgError) as exc:
        return ManoWristFrameResult(
            False, None, None, None, definition.is_right,
            "wrist_ring_fit_failed:{}".format(exc), {},
        )


__all__ = [
    "EXPECTED_WRIST_RING_VERTICES",
    "ManoWristDefinition",
    "ManoWristFrameResult",
    "build_mano_wrist_definition",
    "estimate_mano_wrist_frame",
    "find_boundary_edges",
    "ordered_largest_boundary_loop",
    "ring_geometry",
    "robust_similarity_kabsch",
    "weighted_similarity_kabsch",
]
