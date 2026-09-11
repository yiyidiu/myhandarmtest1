#!/usr/bin/env python3
"""Convert existing HaMeR output plus aligned D455 depth to teleop pose packets."""

from typing import Any, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .finger_observation import (
    invalid_finger_observation,
    observe_mano_fingers,
)
from .realtime_hamer_pipeline import (
    normalized_crop_points_to_original,
    project_hamer_vertices_to_original,
)
from .rgbd_rigid_tracker import deproject_pixels


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
    if reference_kind == "MANO_WRIST_RING_16":
        loop = control.get("quality", {}).get("wrist_loop_vertex_indices")
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
    return {
        "schema": "handarm_hamer_pose_v1",
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
            "orientation_source", "HAMER_MANO_JOINT_PALM_FRAME_COARSE_ONLY"
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
        "palm_quaternion_xyzw",
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
    "ProximalWristDepthFilter",
    "build_live_teleop_packet",
    "build_invalid_teleop_packet",
    "invalidate_stale_teleop_observation",
    "foreground_depth_component",
    "metric_wrist_from_arrays",
    "metric_wrist_ring_from_arrays",
]
