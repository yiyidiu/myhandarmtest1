#!/usr/bin/env python3
"""Exact-frame MANO renderer ported from ``teleoperation_ubuntu_core``.

The supplied archive's live viewer does not use MediaPipe landmarks to draw a
hand.  It projects all 778 HaMeR ``pred_vertices`` with HaMeR's crop camera and
draws all MANO triangle faces on the *same RGB frame used for inference*.

This Linux adaptation keeps that exact-frame contract.  In particular it does
not move an old mesh onto a newer camera frame using an optical-flow bbox.
That avoids the visually floating/stretched hand produced by cross-frame
display remapping.  Presence gating remains outside this renderer so a caller
can discard the complete pair immediately when the real hand disappears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


# Exact BGR colors used by live_hamer_realsense_mesh_windows.py in the archive.
LIGHT_BLUE = (235, 202, 164)
EDGE_BLUE = (255, 235, 205)


@dataclass(frozen=True)
class TeleoperationCoreRenderFrame:
    """A source/mesh pair tied to one HaMeR inference timestamp."""

    source_bgr: np.ndarray
    overlay_bgr: np.ndarray
    crop_points_px: np.ndarray
    full_points_px: np.ndarray
    vertex_depth: np.ndarray
    timestamp: float
    sequence_bbox_xyxy: np.ndarray

    def side_by_side_bgr(self) -> np.ndarray:
        return np.hstack((self.source_bgr, self.overlay_bgr))


def project_vertices_to_crop(
    vertices: Any,
    camera_translation: Any,
    focal_length: Any,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the archive HaMeR crop-camera equation for every MANO vertex."""

    points = np.asarray(vertices, dtype=np.float64)
    translation = np.asarray(camera_translation, dtype=np.float64)
    focal = np.asarray(focal_length, dtype=np.float64)
    size = int(image_size)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
        or not np.all(np.isfinite(points))
        or translation.shape != (3,)
        or not np.all(np.isfinite(translation))
        or size <= 0
    ):
        raise ValueError("invalid HaMeR vertices, camera translation, or image size")
    if focal.ndim == 0:
        focal_xy = np.repeat(float(focal), 2)
    elif focal.shape == (1,):
        focal_xy = np.repeat(float(focal[0]), 2)
    elif focal.shape == (2,):
        focal_xy = focal
    else:
        raise ValueError("focal_length must be scalar or [fx, fy]")
    if not np.all(np.isfinite(focal_xy)) or np.any(focal_xy <= 0.0):
        raise ValueError("focal_length must be positive and finite")

    camera_vertices = points + translation.reshape(1, 3)
    depth = camera_vertices[:, 2].copy()
    if np.any(depth <= 1.0e-6):
        raise ValueError("MANO vertices must be in front of the HaMeR crop camera")
    pixels = (
        focal_xy.reshape(1, 2)
        * camera_vertices[:, :2]
        / depth[:, None]
        + float(size) / 2.0
    )
    return pixels.astype(np.float32), depth.astype(np.float32)


def crop_points_to_original(
    crop_points: Any,
    affine_original_to_crop: Any,
) -> np.ndarray:
    """Invert the current crop affine, including its left-hand reflection."""

    points = np.asarray(crop_points, dtype=np.float64)
    affine = np.asarray(affine_original_to_crop, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.all(np.isfinite(points))
        or affine.shape != (2, 3)
        or not np.all(np.isfinite(affine))
    ):
        raise ValueError("invalid crop points or original-to-crop affine")
    inverse = cv2.invertAffineTransform(affine)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    original = homogeneous @ inverse.T
    if not np.all(np.isfinite(original)):
        raise ValueError("inverse crop projection produced NaN/Inf")
    return original.astype(np.float32)


def draw_mesh(
    image_bgr: Any,
    points_2d: Any,
    faces: Any,
    alpha: float = 0.48,
) -> np.ndarray:
    """Draw the archive's complete filled MANO topology and sparse edges."""

    image = np.asarray(image_bgr)
    points = np.asarray(points_2d, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    opacity = float(alpha)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or points.ndim != 2
        or points.shape[1] != 2
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(triangles) == 0
        or np.min(triangles) < 0
        or np.max(triangles) >= len(points)
        or not 0.0 <= opacity <= 1.0
    ):
        raise ValueError("invalid image, MANO points, faces, or alpha")

    output = image.copy()
    finite = np.isfinite(points).all(axis=1)
    safe_points = np.nan_to_num(
        points, nan=0.0, posinf=10000.0, neginf=-10000.0
    )
    safe_points = np.clip(safe_points, -10000.0, 10000.0)
    face_valid = finite[triangles].all(axis=1)
    valid_faces = triangles[face_valid]
    if valid_faces.size == 0:
        return output

    polygons = np.rint(safe_points[valid_faces]).astype(np.int32)
    layer = output.copy()
    cv2.fillPoly(layer, list(polygons), LIGHT_BLUE, lineType=cv2.LINE_AA)
    cv2.addWeighted(layer, opacity, output, 1.0 - opacity, 0.0, output)
    cv2.polylines(
        output,
        list(polygons[::4]),
        True,
        EDGE_BLUE,
        1,
        lineType=cv2.LINE_AA,
    )
    return output


def draw_bbox(
    image_bgr: np.ndarray,
    bbox: Any,
    is_right: bool,
) -> None:
    value = np.asarray(bbox, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("bbox must be finite [x1,y1,x2,y2]")
    x1, y1, x2, y2 = np.rint(value).astype(int)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (70, 255, 70), 2)
    cv2.putText(
        image_bgr,
        "HaMeR/MANO {} exact inference frame".format(
            "RIGHT" if bool(is_right) else "LEFT"
        ),
        (max(5, x1), max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (70, 255, 70),
        1,
        cv2.LINE_AA,
    )


def render_inference_frame(
    source_rgb: Any,
    result: Any,
    faces: Any,
    image_size: int = 256,
) -> TeleoperationCoreRenderFrame:
    """Render one complete archive-style source/mesh pair from a HaMeR result."""

    rgb = np.asarray(source_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("source_rgb must be uint8 HxWx3")
    crop_points, depth = project_vertices_to_crop(
        result.pred_vertices_mano_right_canonical,
        result.hamer_crop_projection_translation,
        result.hamer_nominal_crop_focal_length,
        image_size,
    )
    full_points = crop_points_to_original(
        crop_points,
        result.quality["affine_original_to_crop"],
    )
    source_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = draw_mesh(source_bgr, full_points, faces, alpha=0.48)
    bbox = np.asarray(result.requested_bbox_xyxy, dtype=np.float32).copy()
    draw_bbox(source_bgr, bbox, bool(result.is_right))
    draw_bbox(overlay_bgr, bbox, bool(result.is_right))
    return TeleoperationCoreRenderFrame(
        source_bgr=source_bgr,
        overlay_bgr=overlay_bgr,
        crop_points_px=crop_points,
        full_points_px=full_points,
        vertex_depth=depth,
        timestamp=float(result.timestamp),
        sequence_bbox_xyxy=bbox,
    )


__all__ = [
    "EDGE_BLUE",
    "LIGHT_BLUE",
    "TeleoperationCoreRenderFrame",
    "crop_points_to_original",
    "draw_bbox",
    "draw_mesh",
    "project_vertices_to_crop",
    "render_inference_frame",
]
