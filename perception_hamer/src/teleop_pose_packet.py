#!/usr/bin/env python3
"""Convert existing HaMeR output plus aligned D455 depth to teleop pose packets."""

from typing import Any, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .realtime_hamer_pipeline import (
    normalized_crop_points_to_original,
    project_hamer_vertices_to_original,
)
from .rgbd_rigid_tracker import deproject_pixels


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
) -> dict:
    """Build one fail-closed pose packet from the configured control frame."""

    if not isinstance(getattr(result, "is_right", None), (bool, np.bool_)):
        raise ValueError("HaMeR result must carry boolean is_right identity")
    presence_generation = int(presence_generation)
    active_hand_generation = int(active_hand_generation)
    if presence_generation < 0 or active_hand_generation < 0:
        raise ValueError("hand identity generations must be non-negative")

    control = estimates.get(
        "control_wrist_frame", estimates.get("mano_joint_palm_frame")
    )
    if not control or not control.get("valid"):
        raise ValueError("HaMeR control wrist frame is invalid")
    reference_kind = str(
        control.get("reference_kind", "MANO_JOINT_0_PALM_FRAME_LEGACY")
    )
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
            )
        )
        position_source = (
            "MANO_WRIST_RING_16_PROJECTED_HULL_PLUS_D455_ALIGNED_DEPTH"
        )
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
        "hand_identity_present": bool(identity_present),
        "hand_is_right": bool(hand_is_right) if identity_present else False,
        "presence_generation": presence_generation,
        "active_hand_generation": active_hand_generation,
    }


__all__ = [
    "build_live_teleop_packet",
    "build_invalid_teleop_packet",
    "foreground_depth_component",
    "metric_wrist_from_arrays",
    "metric_wrist_ring_from_arrays",
]
