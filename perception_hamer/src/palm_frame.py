#!/usr/bin/env python3
"""HaMeR/MANO palm-frame construction with strict SO(3) and session gates.

Three P4 methods are exposed:

``raw_global_orient``
    Diagnostic baseline only.  It is explicitly forbidden for robot control.
``mano_joint_palm_frame``
    Uses the wrist and four MCP joints, never fingertips or distal joints.
``mano_rigid_vertex_palm_frame``
    Uses frozen, root-dominated wrist/palm/palm-back MANO vertices.

Left-hand points have already been restored to source-camera axes by the crop
API.  A two-sided reflection convention is used for rotations/local axes so
every emitted matrix remains in SO(3).  Metric position still belongs to the
aligned D455 depth path; the origins here are MANO-relative diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


EVIDENCE_SCOPE = "USB2_DEVELOPMENT_ONLY"
MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
RAW_GLOBAL_ORIENT = "raw_global_orient"
MANO_JOINT_PALM_FRAME = "mano_joint_palm_frame"
MANO_RIGID_VERTEX_PALM_FRAME = "mano_rigid_vertex_palm_frame"
PALM_FRAME_METHODS = {
    RAW_GLOBAL_ORIENT,
    MANO_JOINT_PALM_FRAME,
    MANO_RIGID_VERTEX_PALM_FRAME,
}

# HaMeR's MANO wrapper reorders its 21 joints to OpenPose hand order:
# wrist, thumb[1:tip], index[1:tip], middle[1:tip], ring[1:tip], pinky[1:tip].
WRIST_INDEX = 0
INDEX_MCP_INDEX = 5
MIDDLE_MCP_INDEX = 9
RING_MCP_INDEX = 13
PINKY_MCP_INDEX = 17
JOINT_PALM_INDICES = (
    WRIST_INDEX,
    INDEX_MCP_INDEX,
    MIDDLE_MCP_INDEX,
    RING_MCP_INDEX,
    PINKY_MCP_INDEX,
)


class PalmFrameError(RuntimeError):
    """Raised when geometry cannot produce a trustworthy palm frame."""


@dataclass(frozen=True)
class RigidPalmVertexConfig:
    mano_vertex_count: int
    wrist: Tuple[int, ...]
    distal_palm: Tuple[int, ...]
    radial_palm: Tuple[int, ...]
    ulnar_palm: Tuple[int, ...]
    rigid_palm: Tuple[int, ...]
    evidence_scope: str
    source_path: str
    mano_right_sha256: str
    minimum_root_skinning_weight: float

    def __post_init__(self) -> None:
        if self.mano_vertex_count < 1:
            raise ValueError("mano_vertex_count must be positive")
        for name in (
            "wrist",
            "distal_palm",
            "radial_palm",
            "ulnar_palm",
            "rigid_palm",
        ):
            indices = tuple(int(index) for index in getattr(self, name))
            if len(indices) < 3 or len(indices) != len(set(indices)):
                raise ValueError(f"{name} must contain at least three unique vertices")
            if min(indices) < 0 or max(indices) >= self.mano_vertex_count:
                raise ValueError(f"{name} contains an out-of-range vertex")
            object.__setattr__(self, name, indices)
        if not set(self.wrist).issubset(set(self.rigid_palm)):
            raise ValueError("wrist must be a subset of rigid_palm")
        if not set(self.distal_palm).issubset(set(self.rigid_palm)):
            raise ValueError("distal_palm must be a subset of rigid_palm")
        if not math.isfinite(self.minimum_root_skinning_weight):
            raise ValueError("minimum_root_skinning_weight must be finite")


@dataclass(frozen=True)
class PalmFrameEstimate:
    method: str
    valid: bool
    rotation: Optional[np.ndarray]
    quaternion_xyzw: Optional[np.ndarray]
    origin: Optional[np.ndarray]
    is_right: Optional[bool]
    reacquired: bool
    control_allowed: bool
    reason: str
    quality: Mapping[str, Any]
    betas_frozen: bool = False
    evidence_scope: str = EVIDENCE_SCOPE

    def __post_init__(self) -> None:
        if self.method not in PALM_FRAME_METHODS:
            raise ValueError("unsupported palm-frame method")
        if self.valid:
            rotation = require_so3(self.rotation, "rotation")
            quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
            origin = np.asarray(self.origin, dtype=np.float64)
            if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
                raise ValueError("valid estimate needs a finite quaternion")
            norm = float(np.linalg.norm(quaternion))
            if abs(norm - 1.0) > 1e-6:
                raise ValueError("quaternion must be unit length")
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                raise ValueError("valid estimate needs a finite origin")
            object.__setattr__(self, "rotation", rotation)
            object.__setattr__(self, "quaternion_xyzw", quaternion.copy())
            object.__setattr__(self, "origin", origin.copy())
        elif any(value is not None for value in (self.rotation, self.quaternion_xyzw, self.origin)):
            raise ValueError("invalid estimate must not expose stale geometry")
        if self.method == RAW_GLOBAL_ORIENT and self.control_allowed:
            raise ValueError("raw global orientation is baseline-only")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "valid": bool(self.valid),
            "rotation": None if self.rotation is None else self.rotation.tolist(),
            "quaternion_xyzw": (
                None if self.quaternion_xyzw is None else self.quaternion_xyzw.tolist()
            ),
            "origin": None if self.origin is None else self.origin.tolist(),
            "origin_metric_position_valid": False,
            "is_right": None if self.is_right is None else bool(self.is_right),
            "reacquired": bool(self.reacquired),
            "control_allowed": bool(self.control_allowed),
            "reason": self.reason,
            "quality": dict(self.quality),
            "betas_frozen": bool(self.betas_frozen),
            "evidence_scope": self.evidence_scope,
        }


def require_so3(
    rotation: Any, name: str = "rotation", atol: float = 1e-6
) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise PalmFrameError(f"{name} must be a finite 3x3 matrix")
    orthogonality_error = float(np.linalg.norm(value.T @ value - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(value))
    if orthogonality_error > atol or abs(determinant - 1.0) > atol:
        raise PalmFrameError(
            f"{name} is not SO(3): orthogonality_error={orthogonality_error}, "
            f"det={determinant}"
        )
    return value.copy()


def project_to_so3(rotation: Any, maximum_correction: float = 0.05) -> np.ndarray:
    """SVD-orthogonalize a near rotation and reject large/reflection repairs."""

    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise PalmFrameError("rotation must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(value)
    candidate = u @ vt
    if np.linalg.det(candidate) < 0.0:
        u[:, -1] *= -1.0
        candidate = u @ vt
    correction = float(np.linalg.norm(candidate - value, ord="fro"))
    if correction > float(maximum_correction):
        raise PalmFrameError(
            f"rotation correction {correction} exceeds {maximum_correction}"
        )
    return require_so3(candidate)


def rotation_matrix_to_quaternion_xyzw(rotation: Any) -> np.ndarray:
    """Convert SO(3) to a deterministic unit quaternion in xyzw order."""

    matrix = require_so3(rotation)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    # Deterministic first sample.  Temporal continuity is applied separately.
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def align_quaternion_sign(
    quaternion_xyzw: Sequence[float], previous_xyzw: Optional[Sequence[float]]
) -> np.ndarray:
    current = np.asarray(quaternion_xyzw, dtype=np.float64)
    if current.shape != (4,) or not np.all(np.isfinite(current)):
        raise PalmFrameError("quaternion must be finite xyzw")
    current_norm = float(np.linalg.norm(current))
    if current_norm < 1e-12:
        raise PalmFrameError("quaternion has zero norm")
    current = current / current_norm
    if previous_xyzw is not None:
        previous = np.asarray(previous_xyzw, dtype=np.float64)
        if previous.shape != (4,) or not np.all(np.isfinite(previous)):
            raise PalmFrameError("previous quaternion must be finite xyzw")
        previous_norm = float(np.linalg.norm(previous))
        if previous_norm < 1e-12:
            raise PalmFrameError("previous quaternion has zero norm")
        if float(np.dot(current, previous / previous_norm)) < 0.0:
            current *= -1.0
    return current


def mirror_canonical_rotation_to_source(rotation: Any, is_right: bool) -> np.ndarray:
    """Map MANO_RIGHT canonical rotation to source axes without det=-1."""

    if not isinstance(is_right, (bool, np.bool_)):
        raise TypeError("is_right must be bool")
    canonical = project_to_so3(rotation)
    if bool(is_right):
        return canonical
    return require_so3(MIRROR_X @ canonical @ MIRROR_X)


def _finite_points(points: Any, minimum_count: int, name: str) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] < minimum_count:
        raise PalmFrameError(f"{name} must have shape (N,3), N>={minimum_count}")
    if not np.all(np.isfinite(value)):
        raise PalmFrameError(f"{name} contains NaN/Inf")
    return value


def _construct_frame(
    origin: np.ndarray,
    transverse: np.ndarray,
    longitudinal: np.ndarray,
    is_right: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Build mirror-canonical local +X, distal +Y, right-handed +Z."""

    if not isinstance(is_right, (bool, np.bool_)):
        raise TypeError("is_right must be bool")
    transverse = np.asarray(transverse, dtype=np.float64)
    longitudinal = np.asarray(longitudinal, dtype=np.float64)
    transverse_norm = float(np.linalg.norm(transverse))
    longitudinal_norm = float(np.linalg.norm(longitudinal))
    if transverse_norm < 1e-8 or longitudinal_norm < 1e-8:
        raise PalmFrameError("palm axes have insufficient baseline")
    raw_cosine = float(np.dot(transverse, longitudinal) / (transverse_norm * longitudinal_norm))
    raw_sine = math.sqrt(max(0.0, 1.0 - min(1.0, raw_cosine * raw_cosine)))
    if raw_sine < 0.10:
        raise PalmFrameError("palm axes are nearly collinear")
    x_axis = transverse / transverse_norm
    if not bool(is_right):
        x_axis *= -1.0
    y_projected = longitudinal - x_axis * float(np.dot(x_axis, longitudinal))
    y_norm = float(np.linalg.norm(y_projected))
    if y_norm < 1e-8:
        raise PalmFrameError("longitudinal palm axis collapsed during projection")
    y_axis = y_projected / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = require_so3(np.column_stack([x_axis, y_axis, z_axis]))
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise PalmFrameError("origin must be finite xyz")
    quality = {
        "determinant": float(np.linalg.det(rotation)),
        "orthogonality_error": float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
        ),
        "transverse_baseline": transverse_norm,
        "longitudinal_baseline": longitudinal_norm,
        "axis_sine": raw_sine,
    }
    return origin.copy(), rotation, quality


def _valid_estimate(
    method: str,
    origin: np.ndarray,
    rotation: np.ndarray,
    is_right: bool,
    quality: Mapping[str, Any],
    control_allowed: bool,
) -> PalmFrameEstimate:
    return PalmFrameEstimate(
        method=method,
        valid=True,
        rotation=rotation,
        quaternion_xyzw=rotation_matrix_to_quaternion_xyzw(rotation),
        origin=origin,
        is_right=bool(is_right),
        reacquired=False,
        control_allowed=control_allowed,
        reason="",
        quality=dict(quality),
    )


def raw_global_orient_baseline(
    global_orient: Any, is_right: bool
) -> PalmFrameEstimate:
    rotation = mirror_canonical_rotation_to_source(global_orient, is_right)
    quality = {
        "determinant": float(np.linalg.det(rotation)),
        "orthogonality_error": float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
        ),
        "baseline_only": True,
        "rotation_source": "hamer_global_orient_mano_right_canonical",
    }
    return _valid_estimate(
        RAW_GLOBAL_ORIENT,
        np.zeros(3, dtype=np.float64),
        rotation,
        is_right,
        quality,
        control_allowed=False,
    )


def mano_joint_palm_frame(joints: Any, is_right: bool) -> PalmFrameEstimate:
    points = _finite_points(joints, max(JOINT_PALM_INDICES) + 1, "MANO joints")
    wrist = points[WRIST_INDEX]
    index_mcp = points[INDEX_MCP_INDEX]
    middle_mcp = points[MIDDLE_MCP_INDEX]
    ring_mcp = points[RING_MCP_INDEX]
    pinky_mcp = points[PINKY_MCP_INDEX]
    mcp_center = np.mean(
        [index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0
    )
    origin = 0.5 * (wrist + mcp_center)
    transverse = index_mcp - pinky_mcp
    longitudinal = mcp_center - wrist
    origin, rotation, quality = _construct_frame(
        origin, transverse, longitudinal, is_right
    )
    quality.update(
        {
            "joint_indices": list(JOINT_PALM_INDICES),
            "uses_fingertips": False,
            "uses_distal_finger_joints": False,
            "origin_semantics": "mano_relative_midpoint_wrist_to_mcp_centroid",
        }
    )
    return _valid_estimate(
        MANO_JOINT_PALM_FRAME,
        origin,
        rotation,
        is_right,
        quality,
        control_allowed=True,
    )


def default_rigid_vertex_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "mano_rigid_palm_vertices.yaml"


def load_rigid_palm_vertex_config(
    path: Optional[str] = None,
) -> RigidPalmVertexConfig:
    source = default_rigid_vertex_config_path() if path is None else Path(path).resolve()
    try:
        # JSON is a strict subset of YAML 1.2.  Keeping this configuration in
        # that subset avoids making the lightweight MediaPipe environment
        # depend on PyYAML merely to consume frozen integer index lists.
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PalmFrameError(f"failed to load rigid palm vertex config: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise PalmFrameError("unsupported rigid palm vertex config schema")
    groups = payload.get("groups")
    provenance = payload.get("selection_provenance")
    if not isinstance(groups, dict) or not isinstance(provenance, dict):
        raise PalmFrameError("rigid palm vertex config is incomplete")
    try:
        return RigidPalmVertexConfig(
            mano_vertex_count=int(payload["mano_vertex_count"]),
            wrist=tuple(groups["wrist"]),
            distal_palm=tuple(groups["distal_palm"]),
            radial_palm=tuple(groups["radial_palm"]),
            ulnar_palm=tuple(groups["ulnar_palm"]),
            rigid_palm=tuple(groups["rigid_palm"]),
            evidence_scope=str(payload["evidence_scope"]),
            source_path=str(source),
            mano_right_sha256=str(provenance["mano_right_sha256"]),
            minimum_root_skinning_weight=float(
                provenance["minimum_root_skinning_weight"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PalmFrameError(f"invalid rigid palm vertex config: {exc}") from exc


def mano_rigid_vertex_palm_frame(
    vertices: Any,
    is_right: bool,
    config: Optional[RigidPalmVertexConfig] = None,
) -> PalmFrameEstimate:
    config = load_rigid_palm_vertex_config() if config is None else config
    points = _finite_points(vertices, config.mano_vertex_count, "MANO vertices")
    if points.shape[0] != config.mano_vertex_count:
        raise PalmFrameError(
            f"MANO vertices must contain exactly {config.mano_vertex_count} points"
        )
    wrist_center = np.mean(points[list(config.wrist)], axis=0)
    distal_center = np.mean(points[list(config.distal_palm)], axis=0)
    radial_center = np.mean(points[list(config.radial_palm)], axis=0)
    ulnar_center = np.mean(points[list(config.ulnar_palm)], axis=0)
    rigid_points = points[list(config.rigid_palm)]
    origin = np.mean(rigid_points, axis=0)
    transverse = radial_center - ulnar_center
    longitudinal = distal_center - wrist_center
    origin, rotation, quality = _construct_frame(
        origin, transverse, longitudinal, is_right
    )
    centered = rigid_points - origin
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.shape != (3,) or singular_values[1] < 1e-8:
        raise PalmFrameError("rigid palm vertices are geometrically degenerate")
    quality.update(
        {
            "rigid_vertex_count": len(config.rigid_palm),
            "uses_fingertips": False,
            "uses_articulated_finger_vertices": False,
            "rigid_cloud_singular_values": singular_values.astype(float).tolist(),
            "rigid_cloud_planarity_ratio": float(
                singular_values[2] / singular_values[1]
            ),
            "vertex_config": config.source_path,
            "mano_right_sha256": config.mano_right_sha256,
            "minimum_root_skinning_weight": config.minimum_root_skinning_weight,
            "evidence_scope": config.evidence_scope,
            "origin_semantics": "mano_relative_rigid_palm_vertex_centroid",
        }
    )
    return _valid_estimate(
        MANO_RIGID_VERTEX_PALM_FRAME,
        origin,
        rotation,
        is_right,
        quality,
        control_allowed=True,
    )


def compare_palm_frame_methods(
    global_orient: Any,
    joints: Any,
    vertices: Any,
    is_right: bool,
    vertex_config: Optional[RigidPalmVertexConfig] = None,
) -> Dict[str, PalmFrameEstimate]:
    """Compute all P4 candidates; raw orientation remains baseline-only."""

    estimates = {
        RAW_GLOBAL_ORIENT: raw_global_orient_baseline(global_orient, is_right),
        MANO_JOINT_PALM_FRAME: mano_joint_palm_frame(joints, is_right),
        MANO_RIGID_VERTEX_PALM_FRAME: mano_rigid_vertex_palm_frame(
            vertices, is_right, vertex_config
        ),
    }
    for estimate in estimates.values():
        require_so3(estimate.rotation)
    return estimates


def rotation_distance_rad(first: Any, second: Any) -> float:
    relative = require_so3(first).T @ require_so3(second)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _invalid_estimate(
    method: str,
    is_right: Optional[bool],
    reason: str,
    quality: Optional[Mapping[str, Any]] = None,
) -> PalmFrameEstimate:
    return PalmFrameEstimate(
        method=method,
        valid=False,
        rotation=None,
        quaternion_xyzw=None,
        origin=None,
        is_right=is_right,
        reacquired=False,
        control_allowed=False,
        reason=str(reason),
        quality={} if quality is None else dict(quality),
        betas_frozen=False,
    )


class PalmFrameSession:
    """Stateful beta freeze, invalidation, reacquisition, and quaternion gate."""

    def __init__(
        self,
        method: str = MANO_RIGID_VERTEX_PALM_FRAME,
        vertex_config: Optional[RigidPalmVertexConfig] = None,
        beta_tolerance: float = 1e-6,
    ) -> None:
        if method not in PALM_FRAME_METHODS:
            raise ValueError("unsupported palm-frame method")
        self.method = method
        self.vertex_config = vertex_config
        self.beta_tolerance = float(beta_tolerance)
        if not math.isfinite(self.beta_tolerance) or self.beta_tolerance < 0.0:
            raise ValueError("beta_tolerance must be finite and non-negative")
        self.start_session()

    def start_session(self, is_right: Optional[bool] = None) -> None:
        if is_right is not None and not isinstance(is_right, (bool, np.bool_)):
            raise TypeError("is_right must be bool or None")
        self._expected_is_right = None if is_right is None else bool(is_right)
        self._frozen_betas: Optional[np.ndarray] = None
        self._previous_quaternion: Optional[np.ndarray] = None
        self._pending_reacquire = True

    @property
    def frozen_betas(self) -> Optional[np.ndarray]:
        return None if self._frozen_betas is None else self._frozen_betas.copy()

    def invalidate(self, reason: str) -> PalmFrameEstimate:
        self._pending_reacquire = True
        return _invalid_estimate(
            self.method,
            self._expected_is_right,
            reason,
            {"reacquire_required": True},
        )

    def _build(
        self,
        global_orient: Any,
        joints: Any,
        vertices: Any,
        is_right: bool,
    ) -> PalmFrameEstimate:
        if self.method == RAW_GLOBAL_ORIENT:
            return raw_global_orient_baseline(global_orient, is_right)
        if self.method == MANO_JOINT_PALM_FRAME:
            return mano_joint_palm_frame(joints, is_right)
        return mano_rigid_vertex_palm_frame(vertices, is_right, self.vertex_config)

    def update(
        self,
        *,
        global_orient: Any,
        joints: Any,
        vertices: Any,
        betas: Any,
        is_right: bool,
    ) -> PalmFrameEstimate:
        if not isinstance(is_right, (bool, np.bool_)):
            return self.invalidate("is_right_must_be_boolean")
        handedness = bool(is_right)
        if self._expected_is_right is None:
            self._expected_is_right = handedness
        elif self._expected_is_right != handedness:
            return self.invalidate("handedness_changed_requires_new_session")
        beta_value = np.asarray(betas, dtype=np.float64)
        if beta_value.shape != (10,) or not np.all(np.isfinite(beta_value)):
            return self.invalidate("invalid_betas")
        if self._frozen_betas is None:
            beta_drift = 0.0
        else:
            beta_drift = float(np.max(np.abs(beta_value - self._frozen_betas)))
            if beta_drift > self.beta_tolerance:
                return self.invalidate("betas_changed_within_session")
        try:
            estimate = self._build(
                global_orient, joints, vertices, handedness
            )
            quaternion = align_quaternion_sign(
                estimate.quaternion_xyzw, self._previous_quaternion
            )
        except (PalmFrameError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return self.invalidate(f"invalid_palm_geometry:{exc}")
        if self._frozen_betas is None:
            self._frozen_betas = beta_value.copy()
        quality = dict(estimate.quality)
        quality.update(
            {
                "betas_max_abs_drift": beta_drift,
                "reacquire_required": False,
                "quaternion_sign_continuous": self._previous_quaternion is not None,
            }
        )
        estimate = replace(
            estimate,
            quaternion_xyzw=quaternion,
            reacquired=self._pending_reacquire,
            quality=quality,
            betas_frozen=True,
        )
        self._previous_quaternion = quaternion.copy()
        self._pending_reacquire = False
        return estimate

    def update_from_hamer(self, result: Any) -> PalmFrameEstimate:
        """Consume the documented :class:`HamerInferenceResult` contract."""

        required = (
            "global_orient",
            "pred_keypoints_3d_source_camera_axes",
            "pred_vertices_source_camera_axes",
            "betas",
            "is_right",
        )
        missing = [name for name in required if not hasattr(result, name)]
        if missing:
            return self.invalidate("hamer_result_missing:" + ",".join(missing))
        return self.update(
            global_orient=result.global_orient,
            joints=result.pred_keypoints_3d_source_camera_axes,
            vertices=result.pred_vertices_source_camera_axes,
            betas=result.betas,
            is_right=result.is_right,
        )


__all__ = [
    "EVIDENCE_SCOPE",
    "INDEX_MCP_INDEX",
    "JOINT_PALM_INDICES",
    "MANO_JOINT_PALM_FRAME",
    "MANO_RIGID_VERTEX_PALM_FRAME",
    "MIDDLE_MCP_INDEX",
    "PALM_FRAME_METHODS",
    "PINKY_MCP_INDEX",
    "PalmFrameError",
    "PalmFrameEstimate",
    "PalmFrameSession",
    "RAW_GLOBAL_ORIENT",
    "RING_MCP_INDEX",
    "RigidPalmVertexConfig",
    "WRIST_INDEX",
    "align_quaternion_sign",
    "compare_palm_frame_methods",
    "load_rigid_palm_vertex_config",
    "mano_joint_palm_frame",
    "mano_rigid_vertex_palm_frame",
    "mirror_canonical_rotation_to_source",
    "project_to_so3",
    "raw_global_orient_baseline",
    "require_so3",
    "rotation_distance_rad",
    "rotation_matrix_to_quaternion_xyzw",
]
