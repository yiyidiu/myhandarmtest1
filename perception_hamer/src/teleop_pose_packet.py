#!/usr/bin/env python3
"""Convert existing HaMeR output plus aligned D455 depth to teleop pose packets."""

from functools import lru_cache
from typing import Any, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .finger_observation import (
    finger_shape_signature,
    finger_surface_depths,
    sample_surface_depths,
    invalid_finger_observation,
    observe_mano_fingers,
)
from .realtime_hamer_pipeline import (
    normalized_crop_points_to_original,
    project_hamer_vertices_to_original,
)
from .rgbd_rigid_tracker import RigidTrackingError, deproject_pixels
from .mano_wrist_reference import (
    HAMER_ROOT_WRIST, WRIST_BASELINE_ID, find_boundary_edges, ordered_largest_boundary_loop,
)


def observed_palm_support(pixels, raw_depth, depth_scale, intrinsics, wrist_z):
    """Depth-supported optical XYZ for the wrist, four MCPs and four interior palm patches."""
    xy = np.asarray(pixels, dtype=float)
    if xy.shape != (21,2) or not np.isfinite(xy[[0,5,9,13,17]]).all():
        return None
    palm = xy[[0,5,9,13,17]]
    samples = np.vstack((palm, .55*palm[0]+.45*palm[1:]))
    depths = np.asarray(sample_surface_depths(samples, raw_depth, depth_scale, wrist_z), dtype=float)
    # A curled finger crossing a palm patch puts its own surface in front of the
    # D455. Ignore those depths even when the projected MCP is still reported.
    segments = xy[1:].reshape(5,4,2)
    starts = segments[:,1:3].reshape(-1,2)
    edges = (segments[:,2:4]-segments[:,1:3]).reshape(-1,2)
    offset = samples[:,None,:]-starts
    along = np.clip(np.sum(offset*edges, axis=2)/np.maximum(np.sum(edges**2, axis=1), 1.e-6), 0., 1.)
    distance = np.linalg.norm(offset-along[:,:,None]*edges, axis=2)
    occluded = np.any(distance < max(2.5, .085*np.linalg.norm(palm[1]-palm[4])), axis=1)
    valid = np.isfinite(depths) & ~occluded
    if valid.sum() < 2:
        return None
    cloud = [[None,None,None] for _ in samples]
    try:
        xyz = deproject_pixels(samples[valid], depths[valid], intrinsics)
    except (KeyError, ValueError, RigidTrackingError):
        return None
    for i, point in zip(np.flatnonzero(valid), xyz):
        cloud[i] = point.tolist()
    return cloud


class CameraViewBiasCorrection:
    """Optional calibrated correction for the recorded physical-left hand.

    Only the positive horizontal camera ray is calibrated. C is the zero of
    the correction, so rotations at the C position keep the existing axes
    and gains. This is an empirical view-bias model, not independent truth.
    """

    def __init__(self, positive_gain: float = 0.0):
        self.positive_gain = float(positive_gain)
        if not np.isfinite(self.positive_gain) or not 0.0 <= self.positive_gain <= 3.0:
            raise ValueError("positive view-bias gain must be in [0, 3]")
        self.reference_key = None
        self.last_rotation = None

    def apply(self, packet: Mapping[str, Any], reference: Any) -> dict:
        output = dict(packet)
        if reference is None:
            self.reference_key = None
            self.last_rotation = None
            return output
        if (self.positive_gain == 0.0 or packet.get("valid") is not True
                or packet.get("hand_is_right") is not False):
            return output
        key = (reference["sequence"], reference["stamp"])
        if key != self.reference_key:
            self.reference_key = key
            self.last_rotation = np.asarray(reference["palm_rotation_row_major"], dtype=float).reshape(3, 3)
        held = packet.get("orientation_held", False) or not packet.get("orientation_channel_valid", True)
        if held:
            output["palm_rotation_row_major"] = self.last_rotation.reshape(-1).tolist()
            output["view_bias_correction"] = {"held": True, "positive_gain": self.positive_gain}
            return output
        position = np.asarray(packet["wrist_position_m"], dtype=float)
        zero = np.asarray(reference["wrist_position_m"], dtype=float)
        if (position.shape != (3,) or zero.shape != (3,) or
                not np.all(np.isfinite(position)) or not np.all(np.isfinite(zero)) or
                min(position[2], zero[2]) <= 0):
            raise ValueError("view correction requires valid camera-frame positions")
        ray = max(0.0, float(np.arctan2(position[0], position[2])))
        zero_ray = max(0.0, float(np.arctan2(zero[0], zero[2])))
        angle = self.positive_gain * (ray-zero_ray)
        c, s = float(np.cos(angle)), float(np.sin(angle))
        correction = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        rotation = correction @ np.asarray(packet["palm_rotation_row_major"], dtype=float).reshape(3, 3)
        self.last_rotation = rotation.copy()
        output["palm_rotation_row_major"] = rotation.reshape(-1).tolist()
        output["view_bias_correction"] = {
            "held": False, "positive_gain": self.positive_gain,
            "reference_ray_rad": zero_ray, "current_ray_rad": ray,
            "correction_deg": float(np.rad2deg(angle)),
            "scope": "EMPIRICAL_POSITIVE_HORIZONTAL_VIEW_PHYSICAL_LEFT_HAND",
        }
        return output


class ProximalWristDepthFilter:
    """Keep observed surface motion current; smooth its model depth offset.

    This is measurement state, not a new teleoperation zero. During a brief
    loss of proximal support, preserve the offset between estimators while
    applying the legacy depth increment. Switching algorithms must not emit
    their entire depth disagreement as a one-frame hand movement.
    """

    def __init__(self, time_constant_s: float = .08, maximum_gap_s: float = .30):
        if not np.isfinite(time_constant_s) or time_constant_s <= 0:
            raise ValueError("depth-offset time constant must be positive")
        if not np.isfinite(maximum_gap_s) or maximum_gap_s <= 0:
            raise ValueError("depth-offset maximum gap must be positive")
        self.time_constant_s = float(time_constant_s)
        self.maximum_gap_s = float(maximum_gap_s)
        self._last = None

    def update(self, position: Any, diagnostics: dict,
               timestamp: float, identity: Any) -> np.ndarray:
        point = np.asarray(position, dtype=np.float64).copy()
        association = diagnostics.get("proximal_surface_association") or {}
        legacy = float(association.get("legacy_depth_m", diagnostics["depth_m"]))
        available = bool(association.get("applied", False))
        surface = float(association["observed_surface_depth_m"]) if available else None
        previous = self._last
        dt = None if previous is None else float(timestamp) - previous["t"]
        if (previous is None or previous["identity"] != identity or
                dt < 0 or dt > self.maximum_gap_s):
            depth, status = float(point[2]), "initialized"
        elif diagnostics.get("depth_reference_hold_used", False):
            # The upstream hold already contains our previous filtered Z.
            # Do not add the estimator offset a second time.
            depth, status = float(point[2]), "existing_depth_hold"
            legacy = previous["legacy"]
            surface = None
        elif dt == 0:
            depth, status = previous["z"], "repeated_source_frame"
        elif available:
            increment = (surface - previous["surface"]
                         if previous["surface"] is not None
                         else legacy - previous["legacy"])
            predicted = previous["z"] + increment
            gain = 1.0 - float(np.exp(-dt / self.time_constant_s))
            depth = predicted + gain * (float(point[2]) - predicted)
            status = "surface_motion_with_filtered_model_offset"
        else:
            depth = previous["z"] + legacy - previous["legacy"]
            status = "continuous_legacy_increment"
        if not np.isfinite(depth) or not .12 < depth < 3.0:
            self._last = None
            raise ValueError("invalid temporally associated wrist depth")
        self._last = dict(t=float(timestamp), identity=identity, z=depth,
                          legacy=legacy, surface=surface)
        diagnostics["depth_offset_filter"] = {
            "status": status, "raw_depth_m": float(point[2]),
            "filtered_depth_m": depth, "time_constant_s": self.time_constant_s,
            "legacy_depth_m": legacy,
        }
        diagnostics["depth_m"] = depth
        # Preserve the same wrist-ring camera ray for all three coordinates.
        return point * (depth / point[2])


def foreground_depth_component(
    samples: Any,
    mask_count: int,
    reference_depth_m: float = None,
) -> Tuple[np.ndarray, dict]:
    """Choose a supported wrist surface instead of a background depth mode."""

    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 8:
        return values, {
            "component_count": int(values.size > 0),
            "selected_fraction": 1.0 if values.size else 0.0,
            "separation_m": 0.0,
        }
    ordered = np.sort(values)
    median = float(np.median(ordered))
    split_gap_m = float(np.clip(0.025 + 0.015 * median, 0.028, 0.050))
    components = np.split(ordered, np.flatnonzero(np.diff(ordered) > split_gap_m) + 1)
    minimum_support = max(8, int(np.ceil(0.04 * max(int(mask_count), 1))))
    supported = [item for item in components if item.size >= minimum_support]
    reference_valid = bool(
        reference_depth_m is not None
        and np.isfinite(float(reference_depth_m))
        and 0.12 < float(reference_depth_m) < 3.0
    )
    if reference_valid:
        association_support = max(5, int(np.ceil(0.01 * max(int(mask_count), 1))))
        candidates = [item for item in components if item.size >= association_support]
        if candidates:
            selected = min(
                candidates,
                key=lambda item: (
                    abs(float(np.median(item)) - float(reference_depth_m)),
                    -int(item.size),
                ),
            )
        elif supported:
            selected = min(supported, key=lambda item: float(np.median(item)))
        else:
            selected = ordered
    elif supported:
        selected = min(supported, key=lambda item: float(np.median(item)))
    else:
        selected = ordered
    selected_median = float(np.median(selected))
    later = [
        item for item in supported if float(np.median(item)) > selected_median
    ]
    separation = 0.0
    if later:
        next_component = min(later, key=lambda item: float(np.median(item)))
        separation = max(
            0.0,
            float(np.quantile(next_component, 0.10))
            - float(np.quantile(selected, 0.90)),
        )
    return selected, {
        "component_count": int(len(components)),
        "selected_fraction": float(selected.size / max(ordered.size, 1)),
        "separation_m": float(separation),
    }


def _proximal_palm_depth_offsets(
    palm_pixels: Any, depth_raw: np.ndarray, scale: float,
    vertices: np.ndarray, wrist_center: np.ndarray,
    surface_pixels: np.ndarray, projected_vertex_ids: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Associate depth near the wrist, away from articulated fingertips.

    The 2-D points only seed the sampling region. XYZ still uses the MANO
    wrist-ring centre ray and each sampled surface-to-ring depth offset.
    No detector prediction is accepted as a metric 3-D position.
    """
    points = np.asarray(palm_pixels, dtype=np.float64)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        return np.empty(0), {"reason": "invalid_palm_pixels"}
    wrist = points[0]
    along = points[1:].mean(axis=0) - wrist
    across = points[1] - points[-1]
    polygon = cv2.convexHull(np.asarray([
        wrist + .05 * along + .25 * across,
        wrist + .05 * along - .25 * across,
        wrist + .40 * along - .30 * across,
        wrist + .40 * along + .30 * across,
    ], dtype=np.float32)).astype(np.int32)
    mask = np.zeros(depth_raw.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    observed = depth_raw[(mask > 0) & (depth_raw > 0)].astype(float) * scale
    observed = observed[(observed > .12) & (observed < 3.0)]
    if observed.size < 12:
        return np.empty(0), {"reason": "insufficient_proximal_depth"}
    surface_depth = float(np.median(observed))
    surface_mad = float(np.median(np.abs(observed - surface_depth)))
    uv = np.rint(surface_pixels).astype(np.int32)
    ids = projected_vertex_ids
    ids = ids[mask[uv[ids, 1], uv[ids, 0]] > 0]
    measured = depth_raw[uv[ids, 1], uv[ids, 0]].astype(float) * scale
    good = ((measured > .12) & (measured < 3.0) &
            (np.abs(measured - surface_depth) <= max(.010, 3.0 * surface_mad)))
    offsets = measured[good] - (vertices[ids[good], 2] - wrist_center[2])
    offsets = offsets[np.isfinite(offsets) & (offsets > .12) & (offsets < 3.0)]
    details = {
        "reason": "proximal_surface_associated" if offsets.size >= 12
                  else "insufficient_model_surface_overlap",
        "polygon_pixels": polygon.reshape(-1, 2).tolist(),
        "observed_surface_depth_m": surface_depth,
        "observed_surface_mad_m": surface_mad,
        "observed_depth_count": int(observed.size),
        "model_surface_count": int(ids.size),
        "associated_sample_count": int(offsets.size),
    }
    return offsets, details


def metric_wrist_ring_from_arrays(
    canonical_vertices: Any,
    wrist_loop: Any,
    camera_translation: Any,
    focal_length: Any,
    affine_original_to_crop: Any,
    aligned_depth_raw: Any,
    depth_scale_m_per_unit: float,
    color_intrinsics: Mapping[str, Any],
    reference_depth_m: float = None,
    reference_depth_age_s: float = None,
    maximum_reference_hold_s: float = 0.12,
    crop_image_size: int = 256,
    use_visible_hand_depth: bool = True,
    palm_anchor_pixels: Any = None,
) -> Tuple[np.ndarray, float, dict, np.ndarray]:
    """Return the projected 16-point wrist-ring centre in metric D455 axes."""

    vertices = np.asarray(canonical_vertices, dtype=np.float64)
    loop = np.asarray(wrist_loop, dtype=np.int64).reshape(-1)
    if (
        vertices.shape != (778, 3)
        or len(loop) != 16
        or np.min(loop) < 0
        or np.max(loop) >= len(vertices)
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError("metric wrist ring requires 778 vertices and 16 valid indices")
    ring = vertices[loop]
    points = np.vstack((ring, ring.mean(axis=0)))
    pixels, _ = project_hamer_vertices_to_original(
        points,
        camera_translation,
        focal_length,
        affine_original_to_crop,
        crop_image_size,
    )
    ring_pixels = np.asarray(pixels[:-1], dtype=np.float64)
    center_pixel = np.asarray(pixels[-1], dtype=np.float64)
    depth_raw = np.asarray(aligned_depth_raw)
    scale = float(depth_scale_m_per_unit)
    if depth_raw.ndim != 2 or not np.issubdtype(depth_raw.dtype, np.integer):
        raise ValueError("aligned depth must be a raw integer image")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid D455 depth scale")
    height, width = depth_raw.shape
    if (
        not np.all(np.isfinite(ring_pixels))
        or not np.all(np.isfinite(center_pixel))
        or center_pixel[0] < 0.0
        or center_pixel[0] >= width
        or center_pixel[1] < 0.0
        or center_pixel[1] >= height
    ):
        raise ValueError("projected MANO wrist ring is outside aligned depth")
    if use_visible_hand_depth:
        # The wrist-opening projection is often on a stereo silhouette hole.
        # In the Sep 10 recording its few remaining pixels belonged to the
        # torso, changing 1.02 m into 1.45 m during an in-plane hand turn.
        # Estimate the translation of the visible HAND surface using aligned
        # depth, subtracting each MANO point's depth relative to the wrist.
        # The control origin and its camera ray remain the same wrist ring.
        surface_pixels, _ = project_hamer_vertices_to_original(
            vertices, camera_translation, focal_length,
            affine_original_to_crop, crop_image_size,
        )
        uv = np.rint(surface_pixels).astype(np.int32)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < height))
        ids = np.flatnonzero(inside)
        # Do not count occluded rear vertices or repeated pixels as separate
        # observations. Camera +Z points away from the sensor.
        order = ids[np.argsort(vertices[ids, 2])]
        _, first = np.unique(uv[order, 1] * width + uv[order, 0], return_index=True)
        ids = order[first]
        measured = depth_raw[uv[ids, 1], uv[ids, 0]].astype(np.float64) * scale
        good = np.isfinite(measured) & (measured > 0.12) & (measured < 3.0)
        offsets = measured[good] - (vertices[ids[good], 2] - ring.mean(axis=0)[2])
        offsets = np.sort(offsets[np.isfinite(offsets) & (offsets > 0.12) & (offsets < 3.0)])
        selected = np.empty(0, dtype=np.float64)
        if offsets.size >= 12:
            # A supported foreground band, not the global median: most valid
            # pixels can be background when a side-facing hand has depth holes.
            ends = np.searchsorted(offsets, offsets + 0.025)
            counts = ends - np.arange(offsets.size)
            support = max(12, int(np.ceil(0.04 * offsets.size)))
            candidates = np.flatnonzero(counts >= support)
            if candidates.size:
                start = int(candidates[0])
                centre = float(np.median(offsets[start:ends[start]]))
                selected = offsets[np.abs(offsets - centre) <= 0.020]
        held = False
        if selected.size < 12:
            if (reference_depth_m is not None and np.isfinite(reference_depth_m) and
                    reference_depth_age_s is not None and
                    0.0 <= reference_depth_age_s <= maximum_reference_hold_s):
                selected = np.full(12, reference_depth_m, dtype=np.float64)
                held = True
            else:
                raise ValueError("no supported visible-hand depth surface")
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        fraction = float(min(1.0, selected.size / max(offsets.size, 1)))
        confidence = float(np.sqrt(fraction) * np.exp(-0.5 * (mad / 0.010) ** 2))
        if held:
            confidence = 0.08
        association = None
        if palm_anchor_pixels is not None and not held:
            proximal, association = _proximal_palm_depth_offsets(
                palm_anchor_pixels, depth_raw, scale, vertices,
                ring.mean(axis=0), surface_pixels, ids,
            )
            association["applied"] = bool(proximal.size >= 12)
            if association["applied"]:
                association["legacy_depth_m"] = median
                association["legacy_confidence"] = confidence
                median = float(np.median(proximal))
                mad = float(np.median(np.abs(proximal - median)))
                fraction = float(min(1.0, proximal.size / max(
                    association["model_surface_count"], 1)))
                # Do not increase control authority merely because a smaller
                # ROI has higher coverage. Keep the previous confidence cap.
                confidence = min(confidence, float(
                    np.sqrt(fraction) * np.exp(-0.5 * (mad / .010) ** 2)))
                selected = proximal
        center_m = deproject_pixels(center_pixel.reshape(1, 2),
                                   np.asarray([median]), dict(color_intrinsics))[0]
        diagnostics = {
            "reference": "mean_of_16_mano_wrist_opening_vertices",
            "method": "visible_hand_surface_depth_offset",
            "center_pixel": center_pixel.tolist(), "depth_m": median,
            "depth_mad_m": mad, "sample_count": int(selected.size),
            "total_sample_count": int(offsets.size),
            "foreground_sample_fraction": fraction,
            "depth_reference_hold_used": held,
            "depth_reference_age_s": reference_depth_age_s,
            "maximum_depth_reference_hold_s": float(maximum_reference_hold_s),
        }
        if association is not None:
            diagnostics["proximal_surface_association"] = association
            if association["applied"]:
                diagnostics["method"] = "proximal_hand_surface_depth_offset"
                diagnostics["all_surface_sample_count"] = diagnostics["total_sample_count"]
                diagnostics["total_sample_count"] = association["model_surface_count"]
        return center_m, confidence, diagnostics, ring_pixels
    polygon = cv2.convexHull(np.rint(ring_pixels).astype(np.int32)).reshape(-1, 2)
    if len(polygon) < 3 or abs(float(cv2.contourArea(polygon))) < 2.0:
        raise ValueError("projected MANO wrist ring hull is degenerate")
    mask = np.zeros(depth_raw.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    if int(mask.sum()) >= 30:
        eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
        if int(eroded.sum()) >= 12:
            mask = eroded
    depth_m = depth_raw.astype(np.float64) * scale
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.12) & (depth_m < 3.0)
    samples = depth_m[(mask > 0) & valid_depth]
    mask_count = int(np.count_nonzero(mask))
    fallback_used = False
    fallback_radius_px = 0
    reference_hold_used = False
    if samples.size < 8:
        x, y = np.rint(center_pixel).astype(int)
        patch = np.empty(0, dtype=np.float64)
        # Wrist silhouettes commonly contain RealSense depth holes.  Expand
        # causally around the same projected 16-point wrist centre; sampled
        # depth may come from nearby hand/forearm pixels, but the public XYZ
        # is still deprojected on the wrist-centre ray.
        for radius in (6, 10, 16, 24, 32):
            x1, x2 = max(0, x - radius), min(width, x + radius + 1)
            y1, y2 = max(0, y - radius), min(height, y + radius + 1)
            candidate = depth_m[y1:y2, x1:x2]
            candidate = candidate[
                np.isfinite(candidate)
                & (candidate > 0.12)
                & (candidate < 3.0)
            ]
            if candidate.size >= 6:
                patch = candidate
                fallback_radius_px = int(radius)
                break
        if patch.size < 6:
            reference_hold_valid = bool(
                reference_depth_m is not None
                and np.isfinite(float(reference_depth_m))
                and 0.12 < float(reference_depth_m) < 3.0
                and reference_depth_age_s is not None
                and np.isfinite(float(reference_depth_age_s))
                and 0.0 <= float(reference_depth_age_s)
                <= float(maximum_reference_hold_s)
            )
            if not reference_hold_valid:
                raise ValueError(
                    "no aligned depth at projected MANO wrist-ring centre"
                )
            # A short RealSense silhouette hole must not flicker the entire 6-D
            # stream.  Hold only Z, keep the current wrist-centre camera ray,
            # and advertise very low confidence.  The caller must not refresh
            # the age while this fallback is in use.
            patch = np.full(8, float(reference_depth_m), dtype=np.float64)
            reference_hold_used = True
        samples = patch
        mask_count = int(max(mask_count, patch.size))
        fallback_used = True
    total_samples = int(samples.size)
    selected, component = foreground_depth_component(
        samples, mask_count, reference_depth_m
    )
    if selected.size == 0:
        raise ValueError("no supported aligned-depth wrist surface")
    median = float(np.median(selected))
    mad = float(np.median(np.abs(selected - median)))
    valid_fraction = float(np.clip(total_samples / max(mask_count, 1), 0.0, 1.0))
    coverage_score = float(np.clip(valid_fraction / 0.55, 0.0, 1.0))
    noise_score = float(np.exp(-((max(mad, 0.0) / 0.010) ** 2)))
    confidence = float(np.sqrt(coverage_score * noise_score))
    if fallback_used:
        radius_confidence = float(np.clip(3.0 / max(fallback_radius_px, 1), 0.08, 0.35))
        confidence = min(confidence, radius_confidence)
    if reference_hold_used:
        confidence = min(confidence, 0.08)
    center_m = deproject_pixels(
        center_pixel.reshape(1, 2),
        np.asarray([median], dtype=np.float64),
        dict(color_intrinsics),
    )[0]
    diagnostics = {
        "reference": "mean_of_16_mano_wrist_opening_vertices",
        "center_pixel": center_pixel.tolist(),
        "polygon": polygon.tolist(),
        "depth_m": median,
        "depth_mad_m": mad,
        "valid_fraction": valid_fraction,
        "sample_count": int(selected.size),
        "total_sample_count": total_samples,
        "depth_component_count": int(component["component_count"]),
        "foreground_sample_fraction": float(component["selected_fraction"]),
        "foreground_background_separation_m": float(component["separation_m"]),
        "center_patch_fallback_used": fallback_used,
        "center_patch_fallback_radius_px": int(fallback_radius_px),
        "depth_reference_hold_used": bool(reference_hold_used),
        "depth_reference_age_s": (
            None
            if reference_depth_age_s is None
            else float(reference_depth_age_s)
        ),
        "maximum_depth_reference_hold_s": float(maximum_reference_hold_s),
    }
    return center_m, confidence, diagnostics, ring_pixels


@lru_cache(maxsize=2)
def _open_wrist_loop(face_bytes: bytes) -> np.ndarray:
    faces = np.frombuffer(face_bytes, dtype=np.int64).reshape(-1, 3)
    boundary = find_boundary_edges(faces)
    if not len(boundary):
        return np.empty(0, dtype=np.int64)
    loop = ordered_largest_boundary_loop(boundary)
    if len(loop) != 16:
        raise ValueError("WRIST0_UNEXPECTED_OPEN_MODEL_BOUNDARY")
    return loop


def _wrist_depth_surface(vertices, joints, faces):
    """Continue the open MANO skin locally; never move the joint-0 origin.

    The real forearm continues past MANO's cut wrist. Without that surface,
    rays on one side of the 5x5 patch hit the back wall THROUGH the opening.
    Adding its offset to real front-skin depth produced two joint estimates
    6--9 cm apart even on a flat measured patch. A short open cuff supplies
    front skin correspondence at that cut. It is only a local depth-offset
    approximation, not an observed forearm or an orientation measurement.
    """
    if len(vertices) != 778 or len(joints) < 18:
        return vertices, faces, False
    loop = _open_wrist_loop(np.asarray(faces, dtype=np.int64).tobytes())
    if not len(loop):
        return vertices, faces, False
    ring = vertices[loop]
    axis = joints[[5, 9, 13, 17]].mean(axis=0) - ring.mean(axis=0)
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        raise ValueError("WRIST0_DEGENERATE_WRIST_SURFACE")
    extended = ring - 0.03 * axis / length
    n = len(vertices)
    i = np.arange(len(loop))
    j = (i + 1) % len(loop)
    extra_faces = np.vstack((np.column_stack((loop[i], loop[j], n+j)),
                             np.column_stack((loop[i], n+j, n+i))))
    return np.vstack((vertices, extended)), np.vstack((faces, extra_faces)), True


def _palm_sample_pixels(palm_pixels, origin_pixel, depth_shape):
    """Sample the central 7x7 palm patch inside its eroded hull."""
    points = np.asarray(palm_pixels, dtype=float)
    origin = np.asarray(origin_pixel, dtype=float)
    if (points.shape != (5, 2) or origin.shape != (2,) or
            not np.all(np.isfinite(points)) or not np.all(np.isfinite(origin))):
        raise ValueError("PALM_SKIN_INVALID_INPUT")
    h, w = depth_shape
    center = 0.5 * (points[0] + points[1:].mean(axis=0))
    if not (0 <= origin[0] < w and 0 <= origin[1] < h and
            0 <= center[0] < w and 0 <= center[1] < h):
        raise ValueError("PALM_SKIN_PIXEL_OUTSIDE_IMAGE")
    x, y = np.rint(center).astype(int)
    x0, x1 = max(0, x-3), min(w, x+4)
    y0, y1 = max(0, y-3), min(h, y+4)
    # Rasterize locally; no full-resolution mesh or segmentation pass.
    mask = np.zeros((y1-y0+2, x1-x0+2), np.uint8)
    hull = cv2.convexHull(np.rint(points-[x0-1, y0-1]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    mask = cv2.erode(mask, np.ones((3,3),np.uint8))[1:-1,1:-1].astype(bool)
    row, col = np.nonzero(mask)
    return np.column_stack((col + x0, row + y0)), center


def metric_palm_skin_from_arrays(palm_pixels, origin_pixel, aligned_depth_raw,
                                depth_scale_m_per_unit, color_intrinsics, allow_palm_hull=False):
    """Same-frame visible palm skin Z, without any model depth/offset/hold.

    Start with the central 7x7 patch in the eroded palm hull. The optional
    hull fallback uses only spatially distributed, same-frame depth returns.
    Their native origin rays remain distinct. The returned control point is
    a skin-depth proxy on that ray, not a measured internal anatomical joint.
    """
    origin = np.asarray(origin_pixel, dtype=float)
    depth = np.asarray(aligned_depth_raw)
    scale = float(depth_scale_m_per_unit)
    if (depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer)
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("PALM_SKIN_INVALID_INPUT")
    query, center = _palm_sample_pixels(palm_pixels, origin, depth.shape)
    values = depth[query[:, 1], query[:, 0]].astype(float) * scale
    good = values[(values > .12) & (values < 3.0)]
    total_samples = len(query)
    fallback = len(good) < 6 and allow_palm_hull
    central = good
    if fallback:
        # At an edge-on view the 7x7 square may contain only one palm pixel.
        # Use measured pixels along the same projected palm, excluding its rim.
        points = np.asarray(palm_pixels)
        start = np.maximum(0, np.floor(points.min(axis=0)).astype(int)-2)
        end = np.minimum(depth.shape[::-1], np.ceil(points.max(axis=0)).astype(int)+3)
        mask = np.zeros(tuple((end-start)[::-1]), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(points-start).astype(np.int32)), 1)
        mask = cv2.erode(mask, np.ones((3,3),np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        y, x = np.nonzero(mask)
        query = np.column_stack((x, y)) + start
        total_samples = len(query)
        values = depth[query[:, 1], query[:, 0]] * scale
        valid = (values > .12) & (values < 3.)
        query, good = query[valid], values[valid]
    if len(good) < 6:
        raise ValueError("PALM_SKIN_INSUFFICIENT_MEASURED_DEPTH")
    median = float(np.median(good))
    mad = float(np.median(np.abs(good-median)))
    inliers = good[np.abs(good-median) <= max(.006, 3*mad)]
    if (len(inliers) < max(6, int(np.ceil(2*len(good)/3))) or
            np.percentile(inliers,90)-np.percentile(inliers,10) > .025):
        raise ValueError("PALM_SKIN_DEPTH_DISCONTINUITY")
    z = float(np.median(inliers))
    if fallback:
        accepted = query[np.abs(good-median) <= max(.006, 3*mad)]
        # Distinct parts of the palm must support one surface. A central return,
        # when present, must agree. Never fill an empty frame with older depth.
        if (np.ptp(accepted,axis=0).max() < 5 or
                (len(central) and abs(np.median(central)-z) > .020)):
            raise ValueError("PALM_SKIN_DEPTH_DISCONTINUITY")
    point = deproject_pixels(origin.reshape(1,2), np.asarray([z]), dict(color_intrinsics))[0]
    confidence = float(min(1., np.sqrt(len(inliers)/49))*np.exp(-.5*(mad/.010)**2))
    return point, confidence, dict(method="eroded_palm_depth_consensus" if fallback else "direct_palm_skin_depth", depth_m=z,
        measured_surface_depth_m=z, center_pixel=center.tolist(), origin_pixel=origin.tolist(),
        patch_radius_px=None if fallback else 3, sample_count=len(inliers), total_sample_count=total_samples,
        depth_mad_m=mad, depth_consensus_fraction=len(inliers)/len(good),
        model_depth_offset_m=0.0, depth_reference_hold_used=False,
        position_semantics="NATIVE_ORIGIN_RAY_AT_VISIBLE_PALM_SKIN_DEPTH_NOT_ANATOMICAL_BONE")


def _front_surface_values(query, projected, faces, projection_z, vertex_values):
    """Interpolate vertex values on the frontmost triangle in the projection camera."""
    missing = np.full(len(query), np.nan)
    if not len(query):
        return missing
    triangles = projected[faces].astype(np.float64)
    overlaps = ((triangles.max(axis=1) >= query.min(axis=0)).all(axis=1) &
                (triangles.min(axis=1) <= query.max(axis=0)).all(axis=1))
    triangles, faces = triangles[overlaps], faces[overlaps]
    if not len(faces):
        return missing
    edge1, edge2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    determinant = edge1[:, 0]*edge2[:, 1] - edge1[:, 1]*edge2[:, 0]
    nondegenerate = np.abs(determinant) > 1e-8
    determinant = np.where(nondegenerate, determinant, 1.0)
    delta = query[None, :, :] - triangles[:, None, 0, :]
    b = (delta[:, :, 0]*edge2[:, None, 1] - delta[:, :, 1]*edge2[:, None, 0]) / determinant[:, None]
    c = (edge1[:, None, 0]*delta[:, :, 1] - edge1[:, None, 1]*delta[:, :, 0]) / determinant[:, None]
    inside = nondegenerate[:, None] & (b >= -1e-7) & (c >= -1e-7) & (b+c <= 1+1e-7)
    weights = np.stack((1-b-c, b, c), axis=2)
    weights_over_z = weights / projection_z[faces][:, None, :]
    inverse_z = weights_over_z.sum(axis=2)
    # For positive depths, the frontmost surface maximizes 1/Z.
    inverse_z = np.where(inside & (inverse_z > 0), inverse_z, 0.0)
    nearest = inverse_z.argmax(axis=0)
    columns = np.arange(len(query))
    denominator = inverse_z[nearest, columns]
    numerator = (weights_over_z[nearest, columns] * vertex_values[faces[nearest]]).sum(axis=1)
    return np.divide(numerator, denominator, out=missing, where=denominator > 0)


def metric_wrist_center_from_rgbd(result, wrist_loop, frame, mano_faces):
    """Recover the metric 16-vertex centre on its own D455 camera ray."""
    # Preserve the native 16-vertex mean used by the original projection.
    vertices = np.asarray(result.pred_vertices_mano_right_canonical)
    source = np.asarray(result.pred_vertices_source_camera_axes, dtype=np.float64)
    loop = np.asarray(wrist_loop)
    faces = np.asarray(mano_faces)
    if (vertices.shape != (778, 3) or source.shape != vertices.shape
            or not np.isfinite(source).all() or loop.shape != (16,)
            or not np.issubdtype(loop.dtype, np.integer) or np.unique(loop).size != 16
            or loop.min() < 0 or loop.max() >= len(vertices) or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("WRIST_CENTER_INVALID_MESH")
    depth = np.asarray(frame.aligned_depth_raw)
    scale = float(frame.depth_scale_m_per_unit)
    if (depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer)
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("WRIST_CENTER_INVALID_DEPTH_IMAGE")
    warp = np.asarray(result.quality["affine_original_to_crop"], dtype=np.float64)
    translation = result.hamer_crop_projection_translation
    focal = result.hamer_nominal_crop_focal_length
    origin, _ = project_hamer_vertices_to_original(
        vertices[loop].mean(axis=0, keepdims=True), translation, focal, warp)
    palm = normalized_crop_points_to_original(result.pred_keypoints_2d_crop_normalized, warp, 256)
    query, patch_center = _palm_sample_pixels(palm[[0, 5, 9, 13, 17]], origin[0], depth.shape)
    # Intersect in HaMeR's projection plane so perspective interpolation also
    # works with an off-axis crop homography. Interpolate offsets in physical
    # D455 axes, where the depth samples were measured.
    if warp.shape == (2, 3):
        warp = np.vstack((warp, [0., 0., 1.]))
    query_h = np.column_stack((query, np.ones(len(query)))) @ warp.T
    if (np.abs(query_h[:, 2]) < 1e-9).any():
        raise ValueError("WRIST_CENTER_INVALID_PIXEL_MAP")
    query_crop = query_h[:, :2] / query_h[:, 2, None]
    crop_pixels, _ = project_hamer_vertices_to_original(vertices, translation, focal, np.eye(2, 3))
    projection_z = vertices[:, 2].astype(np.float64) + translation[2]
    relative_z = source[:, 2] - source[loop, 2].mean()
    surface_minus_center = _front_surface_values(query_crop, crop_pixels, faces, projection_z, relative_z)
    observed = depth[query[:, 1], query[:, 0]].astype(np.float64) * scale
    valid = np.isfinite(surface_minus_center) & (observed > 0.12) & (observed < 3.0)
    corrected = observed[valid] - surface_minus_center[valid]
    if len(corrected) < 6:
        raise ValueError("WRIST_CENTER_INSUFFICIENT_RGBD_SUPPORT")
    median = np.median(corrected)
    mad = float(np.median(np.abs(corrected - median)))
    inliers = np.abs(corrected - median) <= max(0.006, 3.0 * mad)
    if (inliers.sum() < max(6, 2 * len(corrected) / 3)
            or np.ptp(np.percentile(corrected[inliers], [10, 90])) > 0.025):
        raise ValueError("WRIST_CENTER_DEPTH_DISCONTINUITY")
    z = float(np.median(corrected[inliers]))
    if not 0.12 < z < 3.0:
        raise ValueError("WRIST_CENTER_INVALID_ESTIMATED_DEPTH")
    point = deproject_pixels(origin, np.array([z]), dict(frame.color_intrinsics))[0]
    confidence = float(np.sqrt(inliers.sum() / 49) * np.exp(-0.5 * (mad / 0.010)**2))
    return point, confidence, {
        "reference": "mean_of_16_mano_wrist_opening_vertices",
        "control_origin": "MANO_WRIST_RING_16", "method": "wrist_center_surface_offset",
        "center_pixel": origin[0].tolist(), "surface_patch_pixel": patch_center.tolist(),
        "depth_m": z, "measured_surface_depth_m": float(np.median(observed[valid][inliers])),
        "model_surface_minus_center_z_m": float(np.median(surface_minus_center[valid][inliers])),
        "sample_count": int(inliers.sum()), "total_sample_count": len(query),
        "depth_mad_m": mad, "depth_reference_hold_used": False,
    }


def metric_wrist_zero_from_arrays(
    canonical_vertices: Any,
    canonical_joints: Any,
    normalized_keypoints: Any,
    mano_faces: Any,
    camera_translation: Any,
    focal_length: Any,
    affine_original_to_crop: Any,
    aligned_depth_raw: Any,
    depth_scale_m_per_unit: float,
    color_intrinsics: Mapping[str, Any],
    crop_image_size: int = 256,
    patch_radius: int = 2,
) -> Tuple[np.ndarray, float, dict]:
    """Estimate joint 0 from local measured depth plus a model surface offset.

    The 5x5 color-grid neighborhood is centered on joint 0, never on a ring
    or MCP. At each pixel use the frontmost projected MANO triangle, with
    perspective-correct interpolation, to obtain Z_joint0 - Z_surface.
    Add that offset to the SAME pixel's aligned real depth, robustly combine
    the joint-depth estimates, then deproject on the joint-0 camera ray.
    The crop camera only establishes correspondence; its absolute Z is not
    a metric position source. Missing/ambiguous support fails without a
    previous-depth, other-joint, wrist-ring, or monocular-depth fallback.
    """
    vertices = np.asarray(canonical_vertices, dtype=np.float64)
    joints = np.asarray(canonical_joints, dtype=np.float64)
    faces = np.asarray(mano_faces)
    depth = np.asarray(aligned_depth_raw)
    scale = float(depth_scale_m_per_unit)
    radius = int(patch_radius)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or
            joints.ndim != 2 or joints.shape[1] != 3 or len(joints) < 1 or
            not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(joints)) or
            faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or
            not np.issubdtype(faces.dtype, np.integer) or
            np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("WRIST0_INVALID_MODEL_GEOMETRY")
    if (depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer) or
            not np.isfinite(scale) or scale <= 0.0 or radius < 1):
        raise ValueError("WRIST0_INVALID_DEPTH_IMAGE")
    pixels = normalized_crop_points_to_original(
        normalized_keypoints, affine_original_to_crop, crop_image_size)
    if len(pixels) < 1:
        raise ValueError("WRIST0_MISSING_PIXEL")
    center = pixels[0].astype(np.float64)
    height, width = depth.shape
    if not (0 <= center[0] < width and 0 <= center[1] < height):
        raise ValueError("WRIST0_PIXEL_OUTSIDE_DEPTH")
    x, y = np.rint(center).astype(int)
    yy, xx = np.mgrid[max(0, y-radius):min(height, y+radius+1),
                       max(0, x-radius):min(width, x+radius+1)]
    query = np.column_stack((xx.ravel(), yy.ravel()))
    measured = depth[yy, xx].ravel().astype(float) * scale
    vertices, faces, wrist_surface_extended = _wrist_depth_surface(vertices, joints, faces)
    projected, camera_z = project_hamer_vertices_to_original(
        vertices, camera_translation, focal_length,
        affine_original_to_crop, crop_image_size)
    camera_z = vertices[:, 2] + float(np.asarray(camera_translation)[2])
    surface_camera_z = _front_surface_values(query, projected, faces, camera_z, camera_z)
    if not np.any(np.isfinite(surface_camera_z)):
        raise ValueError("WRIST0_NO_LOCAL_MODEL_SURFACE")
    model_offsets = (joints[0, 2] + float(np.asarray(camera_translation)[2])
                     - surface_camera_z)
    supported = np.isfinite(model_offsets) & (measured > 0.12) & (measured < 3.0)
    if int(supported.sum()) < 6:
        raise ValueError("WRIST0_INSUFFICIENT_LOCAL_RGBD_SUPPORT")
    estimates = measured[supported] + model_offsets[supported]
    median = float(np.median(estimates))
    mad = float(np.median(np.abs(estimates-median)))
    inliers = np.abs(estimates-median) <= max(0.006, 3.0*mad)
    if int(inliers.sum()) < 6:
        raise ValueError("WRIST0_INSUFFICIENT_LOCAL_RGBD_INLIERS")
    # Edge-on wrists often contain a few depth-edge outliers: the recorded
    # 5x5 patches had 18--22 consistent points and 3--7 points ~6 cm away.
    # Rejecting the whole patch BEFORE robust selection disabled valid RGB-D
    # poses. Require a two-thirds majority, then apply the same spread bound to
    # that consensus. Balanced/ambiguous surfaces still fail; no old depth,
    # monocular estimate or nearest-surface guess substitutes for measurement.
    if (int(inliers.sum()) < int(np.ceil(2.0 * len(estimates) / 3.0)) or
            np.percentile(estimates[inliers], 90) -
            np.percentile(estimates[inliers], 10) > 0.025):
        raise ValueError("WRIST0_LOCAL_DEPTH_DISCONTINUITY")
    measured_inliers = measured[supported][inliers]
    offsets_inliers = model_offsets[supported][inliers]
    joint_z = float(np.median(estimates[inliers]))
    if not np.isfinite(joint_z) or not 0.12 < joint_z < 3.0:
        raise ValueError("WRIST0_INVALID_ESTIMATED_DEPTH")
    point = deproject_pixels(center.reshape(1, 2), np.asarray([joint_z]), dict(color_intrinsics))[0]
    confidence = float(np.sqrt(inliers.sum()/len(query)) * np.exp(-0.5*(mad/0.010)**2))
    diagnostics = {
        "reference": "MANO_WRIST_JOINT_0",
        "control_origin": "MANO_WRIST_JOINT_0",
        "method": "joint0_local_rgbd_model_surface_offset",
        "aggregation": "median_of_per_pixel_corrected_joint_depths",
        "center_pixel": center.tolist(), "patch_radius_px": radius,
        "depth_m": joint_z,
        "measured_surface_depth_m": float(np.median(measured_inliers)),
        "model_surface_to_joint0_offset_m": float(np.median(offsets_inliers)),
        "model_offset_range_m": [float(offsets_inliers.min()), float(offsets_inliers.max())],
        "depth_mad_m": mad, "sample_count": int(inliers.sum()),
        "depth_consensus_fraction": float(inliers.sum()/len(estimates)),
        "depth_outlier_count": int(len(estimates)-inliers.sum()),
        "total_sample_count": int(len(query)),
        "model_surface_pixel_count": int(np.isfinite(surface_camera_z).sum()),
        "open_wrist_surface_extended": wrist_surface_extended,
        "depth_reference_hold_used": False,
        "position_semantics": "MODEL_JOINT0_ESTIMATE_FROM_MEASURED_SURFACE_NOT_MEASURED_BONE",
    }
    return point, confidence, diagnostics


def metric_wrist_from_arrays(
    normalized_wrist_points: Any,
    affine_original_to_crop: Any,
    aligned_depth_raw: Any,
    depth_scale_m_per_unit: float,
    color_intrinsics: Mapping[str, Any],
    crop_image_size: int = 256,
    patch_radius: int = 3,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return D455-frame metric wrist, depth confidence, and original pixels.

    The MANO wrist remains the position reference.  D455 aligned depth often
    contains small holes exactly on the wrist silhouette, so depth is searched
    in progressively larger wrist neighborhoods.  As a final low-confidence
    fallback, depth is sampled around the four MANO MCP joints and applied to
    the wrist camera ray; the MCP centroid itself never becomes the reference.
    """

    points = normalized_crop_points_to_original(
        normalized_wrist_points, affine_original_to_crop, crop_image_size)
    if points.shape[0] < 1:
        raise ValueError("HaMeR output does not contain a wrist point")
    u, v = points[0]
    x = int(round(float(u))); y = int(round(float(v)))
    depth = np.asarray(aligned_depth_raw)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
        raise ValueError("aligned depth must be a raw integer image")
    scale = float(depth_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid D455 depth scale")
    if not np.all(np.isfinite([u, v])) or not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        raise ValueError("HaMeR wrist pixel is outside aligned depth image")

    def nearest_samples(pixel_x: int, pixel_y: int, radius: int) -> Tuple[np.ndarray, int]:
        x0, x1 = max(0, pixel_x-radius), min(depth.shape[1], pixel_x+radius+1)
        y0, y1 = max(0, pixel_y-radius), min(depth.shape[0], pixel_y+radius+1)
        patch = depth[y0:y1, x0:x1]
        rows, columns = np.nonzero(patch > 0)
        if rows.size == 0:
            return np.empty(0, dtype=np.float64), int(patch.size)
        distances = (columns + x0 - pixel_x)**2 + (rows + y0 - pixel_y)**2
        order = np.argsort(distances, kind="stable")[:64]
        return patch[rows[order], columns[order]].astype(np.float64), int(patch.size)

    base_radius = max(1, int(patch_radius))
    radii = []
    for candidate in (base_radius, max(5, base_radius+2),
                      max(8, base_radius*2), max(12, base_radius*3)):
        if candidate not in radii:
            radii.append(candidate)
    raw_depth = None
    confidence = 0.0
    valid_counts = []
    for radius in radii:
        samples, patch_size = nearest_samples(x, y, radius)
        valid_counts.append(int(samples.size))
        if samples.size < 8:
            continue
        raw_depth = float(np.median(samples))
        radius_penalty = min(1.0, float(2*base_radius+1)/float(2*radius+1))
        coverage = min(1.0, float(samples.size)/float(max(8, patch_size)))
        confidence = max(0.15, radius_penalty*coverage)
        break

    if raw_depth is None:
        palm_depths = []
        # MANO indices 5/9/13/17 are the four non-thumb MCP roots.  They are
        # only a source of nearby surface depth, never a replacement position.
        for index in (5, 9, 13, 17):
            if index >= points.shape[0] or not np.all(np.isfinite(points[index])):
                continue
            palm_x, palm_y = np.rint(points[index]).astype(int)
            if not (0 <= palm_x < depth.shape[1] and 0 <= palm_y < depth.shape[0]):
                continue
            samples, _ = nearest_samples(int(palm_x), int(palm_y), 5)
            if samples.size >= 4:
                palm_depths.append(float(np.median(samples)))
        if palm_depths:
            raw_depth = float(np.median(palm_depths))
            confidence = 0.15 + 0.05*min(4, len(palm_depths))
        else:
            raise ValueError(
                "insufficient aligned depth near HaMeR wrist/palm pixels "
                "(wrist valid counts={})".format(valid_counts)
            )

    depth_m = raw_depth * scale
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("invalid metric wrist depth")
    point = deproject_pixels(
        np.asarray([[u, v]], dtype=np.float64), np.asarray([depth_m]),
        dict(color_intrinsics),
    )[0]
    return point, confidence, points


def wrist_local_landmarks(joints, ring_center, wrist_rotation):
    """Express metric MANO model points in the same frame as the raw Q_t B0 wrist."""
    points = np.asarray(joints, dtype=float)
    center = np.asarray(ring_center, dtype=float)
    rotation = np.asarray(wrist_rotation, dtype=float)
    if (points.shape != (21, 3) or center.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(points).all() or not np.isfinite(center).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
        raise ValueError("invalid same-frame wrist or hand geometry")
    return (points - center) @ rotation


def build_live_teleop_packet(
    result: Any,
    estimates: dict,
    frame: Any,
    roi: Any,
    session_id: str,
    sequence: int,
    presence_generation: int,
    active_hand_generation: int,
    reference_depth_m: float = None,
    reference_depth_age_s: float = None,
    required_reference_kind: str = None,
    palm_anchor_pixels: Any = None,
    depth_filter: Any = None,
    wrist_depth_mode: str = None,
    mano_faces: Any = None,
    include_hand_landmarks: bool = False,
    allow_palm_hull_depth: bool = False,
) -> dict:
    """Build one fail-closed pose packet from the configured control frame."""

    if not isinstance(getattr(result, "is_right", None), (bool, np.bool_)):
        raise ValueError("HaMeR result must carry boolean is_right identity")
    presence_generation = int(presence_generation)
    active_hand_generation = int(active_hand_generation)
    if presence_generation < 0 or active_hand_generation < 0:
        raise ValueError("hand identity generations must be non-negative")

    if required_reference_kind and "control_wrist_frame" not in estimates:
        # A failed wrist estimate must never silently change the basis used
        # by C calibration. The legacy joint-palm basis differs by >100 deg.
        raise ValueError("required control wrist frame is unavailable")
    control = estimates.get(
        "control_wrist_frame", estimates.get("mano_joint_palm_frame")
    )
    if not control or not control.get("valid"):
        raise ValueError("HaMeR control wrist frame is invalid")
    reference_kind = str(
        control.get("reference_kind", "MANO_JOINT_0_PALM_FRAME_LEGACY")
    )
    if required_reference_kind and reference_kind != required_reference_kind:
        raise ValueError("control reference kind differs from configured reference")
    position_diagnostics = None
    if wrist_depth_mode is None:
        wrist_depth_mode = "wrist-center-offset" if control.get("global_orient_used") else "visible-surface"
    if wrist_depth_mode == "wrist-center-offset":
        if reference_kind != "MANO_WRIST_RING_16" or not control.get("global_orient_used"):
            raise ValueError("wrist-center-offset requires the HaMeR root / neutral wrist frame")
        wrist, depth_confidence, position_diagnostics = metric_wrist_center_from_rgbd(
            result, control["quality"]["wrist_loop_vertex_indices"], frame, mano_faces)
        position_source = "D455_WRIST_RING_CENTER_FROM_SAME_PIXEL_SURFACE_OFFSET"
    elif wrist_depth_mode == "palm-skin":
        pixels = normalized_crop_points_to_original(result.pred_keypoints_2d_crop_normalized,
            result.quality["affine_original_to_crop"], 256)
        origin = pixels[0]
        if reference_kind == "MANO_WRIST_RING_16":
            loop = control["quality"]["wrist_loop_vertex_indices"]
            origin = project_hamer_vertices_to_original(
                np.asarray(result.pred_vertices_mano_right_canonical)[loop].mean(axis=0)[None],
                result.hamer_crop_projection_translation, result.hamer_nominal_crop_focal_length,
                result.quality["affine_original_to_crop"], 256)[0][0]
        wrist, depth_confidence, position_diagnostics = metric_palm_skin_from_arrays(
            pixels[[0,5,9,13,17]], origin, frame.aligned_depth_raw,
            frame.depth_scale_m_per_unit, frame.color_intrinsics, allow_palm_hull_depth)
        position_diagnostics["control_origin"] = reference_kind
        position_source = "DIRECT_CAMERA_PALM_SKIN_DEPTH_NO_MODEL_OFFSET"
    elif wrist_depth_mode == "joint0-surface":
        if reference_kind != "MANO_FIVE_POINT_PALM_USER_XYZ":
            raise ValueError("joint0-surface requires the five-point palm reference")
        wrist, depth_confidence, position_diagnostics = metric_wrist_zero_from_arrays(
            result.pred_vertices_mano_right_canonical,
            result.pred_keypoints_3d_mano_right_canonical,
            result.pred_keypoints_2d_crop_normalized, mano_faces,
            result.hamer_crop_projection_translation,
            result.hamer_nominal_crop_focal_length,
            result.quality["affine_original_to_crop"], frame.aligned_depth_raw,
            frame.depth_scale_m_per_unit, frame.color_intrinsics,
        )
        position_source = "HAMER_WRIST_0_LOCAL_RGBD_WITH_MODEL_SURFACE_OFFSET"
    elif reference_kind in ("MANO_WRIST_RING_16", "MANO_FIVE_POINT_PALM_USER_XYZ"):
        loop = control.get("quality", {}).get("wrist_loop_vertex_indices")
        wrist_offset = np.zeros(3)
        if reference_kind == "MANO_FIVE_POINT_PALM_USER_XYZ":
            vertices = np.asarray(result.pred_vertices_source_camera_axes)
            joints = np.asarray(result.pred_keypoints_3d_source_camera_axes)
            wrist_offset = joints[0] - vertices[np.asarray(loop, dtype=int)].mean(axis=0)
        wrist, depth_confidence, position_diagnostics, _ = (
            metric_wrist_ring_from_arrays(
                result.pred_vertices_mano_right_canonical,
                loop,
                result.hamer_crop_projection_translation,
                result.hamer_nominal_crop_focal_length,
                result.quality["affine_original_to_crop"],
                frame.aligned_depth_raw,
                frame.depth_scale_m_per_unit,
                frame.color_intrinsics,
                # The caller caches diagnostics["depth_m"], which belongs to
                # this surface-association anchor, not the shifted wrist-0.
                reference_depth_m=reference_depth_m,
                reference_depth_age_s=reference_depth_age_s,
                palm_anchor_pixels=palm_anchor_pixels,
            )
        )
        position_source = (
            "MANO_WRIST_RING_16_FROM_VISIBLE_HAND_SURFACE_AND_D455_DEPTH"
        )
        if depth_filter is not None:
            wrist = depth_filter.update(
                wrist, position_diagnostics, result.timestamp,
                (session_id, active_hand_generation, bool(result.is_right)),
            )
            legacy_depth = position_diagnostics["depth_offset_filter"]["legacy_depth_m"]
            # The optional position experiment must not also change the
            # wrist anchor used by the existing forearm-orientation channel.
            position_diagnostics["legacy_wrist_position_m"] = (
                wrist * (legacy_depth / wrist[2])).tolist()
        if reference_kind == "MANO_FIVE_POINT_PALM_USER_XYZ":
            # Match experiment C's metric wrist-0 origin. The visible-surface
            # depth association is shared; no ring orientation enters control.
            wrist = wrist + wrist_offset
            position_source = "HAMER_WRIST_0_FROM_VISIBLE_HAND_SURFACE_AND_REAL_DEPTH"
            position_diagnostics["control_origin"] = "MANO_WRIST_JOINT_0"
    else:
        wrist, depth_confidence, _ = metric_wrist_from_arrays(
            result.pred_keypoints_2d_crop_normalized,
            result.quality["affine_original_to_crop"],
            frame.aligned_depth_raw, frame.depth_scale_m_per_unit,
            frame.color_intrinsics,
        )
        position_source = "HAMER_WRIST_RAY_PLUS_D455_ADAPTIVE_ALIGNED_DEPTH"
    roi_confidence = float(np.clip(getattr(roi, "confidence", 0.0), 0.0, 1.0))
    visible = float(np.clip(result.quality.get("bbox_visible_fraction", 0.0), 0.0, 1.0))
    crop_quality = float(np.clip(
        estimates.get("teleop_crop_quality", 1.0), 0.0, 1.0
    ))
    position_confidence = (
        roi_confidence * visible * crop_quality * depth_confidence
    )
    rotation_confidence = float(np.clip(
        control.get(
            "filter_confidence", roi_confidence * visible * crop_quality
        ),
        0.0,
        1.0,
    ))
    orientation_channel_valid = bool(
        control.get("orientation_channel_valid", True)
    )
    orientation_held = bool(control.get("orientation_held", False))
    if not orientation_channel_valid:
        # Position and orientation are independent channels.  A held
        # quaternion keeps the packet mathematically complete, while zero
        # rotational confidence prevents it from being reported as a fresh
        # posture observation.
        rotation_confidence = 0.0
    filter_diagnostics = estimates.get("palm_orientation_filter")
    finger_observation = observe_mano_fingers(
        getattr(result, "pred_keypoints_3d_source_camera_axes", None),
        roi_confidence,
        estimates.get("teleop_finger_visible_fraction", visible),
        estimates.get("teleop_finger_crop_quality", crop_quality),
    )
    provenance = {}
    if wrist_depth_mode == "wrist-center-offset":
        provenance["wrist_pose_baseline"] = WRIST_BASELINE_ID
    if include_hand_landmarks:
        palm_pixels = estimates.get("wrist_image_landmarks_px", estimates.get("finger_image_landmarks_px"))
        xy = np.asarray(palm_pixels, dtype=float)
        if xy.shape == (21,2) and np.isfinite(xy[[0,5,9,13,17]]).all():
            try:
                rays = deproject_pixels(xy[[0,5,9,13,17]], np.ones(5), frame.color_intrinsics)
                provenance["wrist_palm_rays"] = rays[:,:2].tolist()
                provenance["wrist_support_source"] = "mediapipe"
            except (KeyError, ValueError, RigidTrackingError):
                pass
            provenance["wrist_motion_support"] = observed_palm_support(
                xy, frame.aligned_depth_raw,
                frame.depth_scale_m_per_unit, frame.color_intrinsics, wrist[2])
    if include_hand_landmarks or wrist_depth_mode == "wrist-center-offset":
        measured = estimates.get("finger_wrist_frame_raw", estimates.get("control_wrist_frame_raw", control))
        if measured.get("valid") and measured.get("orientation_source") == HAMER_ROOT_WRIST:
            provenance["measured_wrist_rotation_row_major"] = np.asarray(measured["rotation"]).reshape(-1).tolist()
            if (include_hand_landmarks and finger_observation.valid and finger_observation.confidence >= .35
                    and estimates.get("teleop_fingers_visible") is not False):
                local = wrist_local_landmarks(result.pred_keypoints_3d_source_camera_axes,
                                             measured["origin"], measured["rotation"])
                provenance["hand_landmarks"] = {
                    "frame": "wrist_qb0_openpose21_m", "points": local.tolist(),
                    "wrist_frame_measured": True,
                    "confidence": float(finger_observation.confidence),
                }
                if "finger_image_landmarks_px" in estimates:
                    xy = estimates["finger_image_landmarks_px"]
                    provenance["hand_landmarks"]["image_landmarks_px"] = xy
                    provenance["hand_landmarks"]["surface_depths_m"] = finger_surface_depths(
                        xy, frame.aligned_depth_raw, frame.depth_scale_m_per_unit, wrist[2])
                    if xy is not None and estimates.get("independent_tip_gaps") is not None:
                        provenance["hand_landmarks"]["independent_tip_gaps"] = estimates["independent_tip_gaps"]
                    if estimates.get("finger_shape_signature") is not None:
                        signature = finger_shape_signature(local)
                        if signature is not None:
                            provenance["hand_landmarks"]["shape_signatures"] = [
                                signature.tolist(), estimates["finger_shape_signature"]]
    return {
        **provenance, "schema": "handarm_hamer_pose_v1",
        "session_id": str(session_id), "sequence": int(sequence),
        "stamp": float(result.timestamp), "frame_id": "camera_color_optical_frame",
        "wrist_position_m": wrist.tolist(),
        "palm_rotation_row_major": np.asarray(control["rotation"]).reshape(-1).tolist(),
        "confidence": [position_confidence]*3+[rotation_confidence]*3,
        "valid": True, "gesture": 0, "gesture_confidence": 0.0,
        "hand_identity_present": True,
        "hand_is_right": bool(result.is_right),
        "presence_generation": presence_generation,
        "active_hand_generation": active_hand_generation,
        "invalid_reason": (
            str(control.get("failure_reason", ""))
            if not orientation_channel_valid else ""
        ),
        "position_source": position_source,
        "orientation_source": control.get(
            "orientation_source", HAMER_ROOT_WRIST if control.get("global_orient_used")
            else "HAMER_MANO_JOINT_PALM_FRAME_COARSE_ONLY"
        ),
        "control_reference": reference_kind,
        "orientation_channel_valid": orientation_channel_valid,
        "orientation_held": orientation_held,
        "position_diagnostics": position_diagnostics,
        "crop_quality": crop_quality,
        "orientation_filter": filter_diagnostics,
        "finger_observation": (
            invalid_finger_observation("FINGER_LANDMARKS_OUTSIDE_IMAGE")
            if estimates.get("teleop_fingers_visible") is False
            else finger_observation.as_packet()),
    }


def build_invalid_teleop_packet(
    session_id: str,
    sequence: int,
    stamp: float,
    invalid_reason: str,
    presence_generation: int,
    active_hand_generation: int,
    hand_is_right: Any = None,
    frame_id: str = "camera_color_optical_frame",
) -> dict:
    """Build an explicit no-pose heartbeat without fabricating geometry."""

    presence_generation = int(presence_generation)
    active_hand_generation = int(active_hand_generation)
    if presence_generation < 0 or active_hand_generation < 0:
        raise ValueError("hand identity generations must be non-negative")
    identity_present = isinstance(hand_is_right, (bool, np.bool_))
    reason = str(invalid_reason or "HAMER_POSE_INVALID")
    return {
        "schema": "handarm_hamer_pose_v1",
        "session_id": str(session_id),
        "sequence": int(sequence),
        "stamp": float(stamp),
        "frame_id": str(frame_id),
        "valid": False,
        "invalid_reason": reason,
        "confidence": [0.0] * 6,
        "gesture": 0,
        "gesture_confidence": 0.0,
        "finger_observation": invalid_finger_observation(reason),
        "hand_identity_present": bool(identity_present),
        "hand_is_right": bool(hand_is_right) if identity_present else False,
        "presence_generation": presence_generation,
        "active_hand_generation": active_hand_generation,
    }


def invalidate_stale_teleop_observation(
    packet: Mapping[str, Any], maximum_pipeline_latency_s: float
) -> dict:
    """Convert an overdue pose into a geometry-free, identity-preserving heartbeat.

    The ROS receiver still independently enforces the same latency ceiling.
    Performing the check at the producer prevents a single ordinary scheduling
    spike from being misclassified as a malformed enabled packet: the active C
    token remains visible, downstream control holds its last bounded target,
    and the next fresh observation can resume without silently re-zeroing.
    """

    maximum = float(maximum_pipeline_latency_s)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_pipeline_latency_s must be finite and positive")
    if not isinstance(packet, Mapping):
        raise TypeError("packet must be a mapping")
    output = dict(packet)
    if output.get("valid") is not True:
        return output
    timing = output.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("valid teleop observation requires a timing mapping")
    latency = float(timing.get("capture_to_publish_s"))
    if not np.isfinite(latency) or latency < 0.0:
        raise ValueError("capture_to_publish_s must be finite and non-negative")
    if latency <= maximum:
        return output

    reason = "SOURCE_PIPELINE_LATENCY_EXCEEDED:{:.6f}>{:.6f}".format(
        latency, maximum
    )
    output["valid"] = False
    output["invalid_reason"] = reason
    output["confidence"] = [0.0] * 6
    for geometry_key in (
        "wrist_position_m",
        "palm_rotation_row_major",
        "measured_wrist_rotation_row_major",
        "wrist_motion_support", "wrist_palm_rays",
        "palm_quaternion_xyzw",
        "hand_landmarks",
    ):
        output.pop(geometry_key, None)
    fingers = output.get("finger_observation")
    if isinstance(fingers, Mapping):
        fingers = dict(fingers)
        fingers["valid"] = False
        fingers["flexion"] = [0.0] * 5
        fingers["confidence"] = 0.0
        fingers["invalid_reason"] = reason
        output["finger_observation"] = fingers
    return output


__all__ = [
    "wrist_local_landmarks",
    "ProximalWristDepthFilter",
    "build_live_teleop_packet",
    "build_invalid_teleop_packet",
    "invalidate_stale_teleop_observation",
    "foreground_depth_component",
    "metric_wrist_from_arrays",
    "metric_wrist_ring_from_arrays",
    "metric_wrist_center_from_rgbd",
    "metric_wrist_zero_from_arrays",
]
