#!/usr/bin/env python3
"""Small, testable primitives for the live D455 -> ROI -> HaMeR path."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .hamer_palm_frame import build_hamer_joint_palm_frame
from .palm_frame import (
    mano_rigid_vertex_palm_frame,
    raw_global_orient_baseline,
)


@dataclass(frozen=True)
class LiveFramePacket:
    """A synchronized frame and the ROI evaluated on that exact RGB image."""

    frame: Any
    roi: Any
    capture_sequence: int


class LatestFrameSlot:
    """Thread-safe single-slot mailbox which overwrites stale camera frames."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._packet: Optional[LiveFramePacket] = None
        self._version = 0
        self._closed = False
        self._error: Optional[BaseException] = None
        self._published = 0
        self._consumed = 0
        self._overwritten = 0

    def publish(self, packet: LiveFramePacket) -> int:
        with self._condition:
            if self._closed:
                return self._version
            if self._published > self._consumed:
                self._overwritten += 1
            self._packet = packet
            self._version += 1
            self._published += 1
            self._condition.notify_all()
            return self._version

    def get_after(
        self, previous_version: int, timeout_s: float = 3.0
    ) -> Tuple[int, Optional[LiveFramePacket]]:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while self._version <= previous_version and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("latest-frame slot timed out")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("capture worker failed") from self._error
            if self._version <= previous_version:
                return self._version, None
            self._consumed = self._published
            return self._version, self._packet

    def close(self, error: Optional[BaseException] = None) -> None:
        with self._condition:
            self._closed = True
            self._error = error
            self._condition.notify_all()

    @property
    def statistics(self) -> dict:
        with self._condition:
            return {
                "published": self._published,
                "consumed": self._consumed,
                "overwritten_before_inference": self._overwritten,
                "capacity": 1,
                "policy": "overwrite_old_keep_latest",
            }


def normalized_crop_points_to_original(
    points: Any, affine_original_to_crop: Any, image_size: int
) -> np.ndarray:
    """Map HaMeR's normalized crop projection back into original RGB pixels."""

    value = np.asarray(points, dtype=np.float64)
    affine = np.asarray(affine_original_to_crop, dtype=np.float64)
    image_size = int(image_size)
    if (
        value.ndim != 2
        or value.shape[1] != 2
        or not np.all(np.isfinite(value))
        or affine.shape != (2, 3)
        or not np.all(np.isfinite(affine))
        or image_size <= 0
    ):
        raise ValueError("invalid normalized points, affine, or image size")
    inverse = cv2.invertAffineTransform(affine)
    crop_pixels = (value + 0.5) * float(image_size)
    homogeneous = np.column_stack((crop_pixels, np.ones(len(crop_pixels))))
    original = (inverse @ homogeneous.T).T
    if not np.all(np.isfinite(original)):
        raise ValueError("point projection produced NaN/Inf")
    return original.astype(np.float32)


def remap_points_between_bboxes(
    points: Any, source_bbox: Any, target_bbox: Any
) -> np.ndarray:
    """Move a projected mesh with the latest KLT ROI between HaMeR frames.

    A uniform center/scale transform preserves mesh aspect ratio.  It updates
    only display projection; the next HaMeR inference remains the source of
    articulation and 3-D palm orientation.
    """

    value = np.asarray(points, dtype=np.float64)
    source = np.asarray(source_bbox, dtype=np.float64)
    target = np.asarray(target_bbox, dtype=np.float64)
    if (value.ndim != 2 or value.shape[1] != 2 or not np.all(np.isfinite(value))
            or source.shape != (4,) or target.shape != (4,)
            or not np.all(np.isfinite(source)) or not np.all(np.isfinite(target))):
        raise ValueError("points and bboxes must be finite Nx2/[x1,y1,x2,y2]")
    source_size = source[2:]-source[:2]
    target_size = target[2:]-target[:2]
    if np.any(source_size <= 1.0e-6) or np.any(target_size <= 1.0e-6):
        raise ValueError("source and target bboxes must have positive area")
    source_center = 0.5*(source[:2]+source[2:])
    target_center = 0.5*(target[:2]+target[2:])
    scale = float(np.sqrt(np.prod(target_size)/np.prod(source_size)))
    return ((value-source_center)*scale+target_center).astype(np.float32)


def hand_bbox_alignment(
    tracked_bbox: Any,
    detected_bbox: Any,
    minimum_iou: float = 0.05,
    maximum_normalized_center_distance: float = 0.75,
) -> dict:
    """Check that KLT and MediaPipe still refer to the same image region.

    Presence alone is insufficient: MediaPipe may see a real hand elsewhere
    while optical flow has drifted onto a face, sleeve, or background edge.
    IoU handles ordinary overlap; normalized center distance also accepts a
    fast translated hand whose two asynchronously sampled boxes only touch.
    """

    tracked = np.asarray(tracked_bbox, dtype=np.float64)
    detected = np.asarray(detected_bbox, dtype=np.float64)
    minimum_iou = float(minimum_iou)
    maximum_distance = float(maximum_normalized_center_distance)
    if (
        tracked.shape != (4,)
        or detected.shape != (4,)
        or not np.all(np.isfinite(tracked))
        or not np.all(np.isfinite(detected))
        or not 0.0 <= minimum_iou <= 1.0
        or not math.isfinite(maximum_distance)
        or maximum_distance < 0.0
    ):
        return {
            "valid": False,
            "iou": 0.0,
            "normalized_center_distance": float("inf"),
            "reason": "invalid_bbox_alignment_input",
        }
    tracked_size = tracked[2:] - tracked[:2]
    detected_size = detected[2:] - detected[:2]
    if np.any(tracked_size <= 0.0) or np.any(detected_size <= 0.0):
        return {
            "valid": False,
            "iou": 0.0,
            "normalized_center_distance": float("inf"),
            "reason": "nonpositive_bbox_extent",
        }
    lower = np.maximum(tracked[:2], detected[:2])
    upper = np.minimum(tracked[2:], detected[2:])
    intersection = float(np.prod(np.maximum(0.0, upper-lower)))
    tracked_area = float(np.prod(tracked_size))
    detected_area = float(np.prod(detected_size))
    union = tracked_area + detected_area - intersection
    iou = 0.0 if union <= 0.0 else intersection/union
    tracked_center = 0.5*(tracked[:2]+tracked[2:])
    detected_center = 0.5*(detected[:2]+detected[2:])
    normalizer = max(
        math.sqrt(tracked_area), math.sqrt(detected_area), 1.0e-9
    )
    center_distance = float(
        np.linalg.norm(tracked_center-detected_center)/normalizer
    )
    valid = bool(iou >= minimum_iou or center_distance <= maximum_distance)
    return {
        "valid": valid,
        "iou": float(iou),
        "normalized_center_distance": center_distance,
        "reason": "aligned" if valid else "tracked_bbox_not_on_detected_hand",
    }


def project_hamer_vertices_to_original(
    vertices: Any,
    camera_translation: Any,
    focal_length: Any,
    affine_original_to_crop: Any,
    image_size: int = 256,
) -> tuple:
    """Project MANO vertices into the original RGB image for visualization.

    HaMeR's crop camera is used only for drawing the recovered mesh.  This
    projection is deliberately separate from the D455 metric wrist position
    used by teleoperation.
    """

    points = np.asarray(vertices, dtype=np.float64)
    translation = np.asarray(camera_translation, dtype=np.float64)
    focal = np.asarray(focal_length, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points))
            or translation.shape != (3,) or not np.all(np.isfinite(translation))
            or focal.shape != (2,) or not np.all(np.isfinite(focal))
            or np.any(focal <= 0.0)):
        raise ValueError("invalid MANO vertices or HaMeR crop camera")
    camera_points = points + translation[None, :]
    depth = camera_points[:, 2]
    if np.any(depth <= 1.0e-6):
        raise ValueError("MANO mesh contains vertices behind the crop camera")
    normalized = camera_points[:, :2] / depth[:, None]
    normalized *= focal[None, :] / float(image_size)
    pixels = normalized_crop_points_to_original(
        normalized, affine_original_to_crop, image_size)
    return pixels, depth.astype(np.float32)


def draw_mano_mesh_overlay(
    bgr: np.ndarray,
    vertices_px: Any,
    vertex_depth: Any,
    faces: Any,
    alpha: float = 0.52,
) -> np.ndarray:
    """Draw a lightweight filled MANO mesh without constructing pyrender."""

    image = np.asarray(bgr)
    points = np.asarray(vertices_px, dtype=np.float64)
    depth = np.asarray(vertex_depth, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or points.ndim != 2 or points.shape[1] != 2
            or depth.shape != (len(points),)
            or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(depth))
            or len(triangles) == 0 or np.min(triangles) < 0
            or np.max(triangles) >= len(points) or not 0.0 <= alpha <= 1.0):
        raise ValueError("invalid image, MANO projection, faces, or alpha")
    layer = np.zeros_like(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_depth = np.mean(depth[triangles], axis=1)
    order = np.argsort(face_depth)[::-1]
    integer_points = np.rint(points).astype(np.int32)
    polygons = integer_points[triangles]
    low, high = float(np.min(face_depth)), float(np.max(face_depth))
    span = max(high-low, 1.0e-9)
    # A Python/OpenCV call for every one of MANO's ~1538 faces consumed more
    # than a CPU core at 30 Hz and competed with HaMeR preprocessing.  Eight
    # painter-order depth bands preserve a readable filled surface with only
    # a handful of batched C++ calls.
    depth_band_count = 8
    normalized_depth = np.clip((face_depth-low)/span, 0.0, 1.0)
    depth_bands = np.minimum(
        depth_band_count-1,
        np.floor(normalized_depth*depth_band_count).astype(np.int32),
    )
    for band in range(depth_band_count-1, -1, -1):
        indices = order[depth_bands[order] == band]
        if len(indices) == 0:
            continue
        representative = (float(band)+0.5)/float(depth_band_count)
        shade = int(np.clip(205.0-75.0*representative, 95.0, 220.0))
        cv2.fillPoly(
            layer, list(polygons[indices]), (55, shade, 255),
            lineType=cv2.LINE_AA,
        )
    cv2.fillPoly(mask, list(polygons), 255, lineType=cv2.LINE_AA)
    # Sparse triangle edges make articulation visible without an opaque web.
    cv2.polylines(
        layer, list(polygons[order[::4]]), True,
        (30, 75, 170), 1, cv2.LINE_AA,
    )
    blended = cv2.addWeighted(image, 1.0-alpha, layer, alpha, 0.0)
    result = image.copy()
    result[mask > 0] = blended[mask > 0]
    return result


def build_live_palm_estimates(result: Any, previous_joint_quaternion: Any = None) -> dict:
    """Compute A/B/C orientation candidates from one valid HaMeR result."""

    raw = raw_global_orient_baseline(result.global_orient, result.is_right)
    joint = build_hamer_joint_palm_frame(
        result.pred_keypoints_3d_source_camera_axes,
        result.is_right,
        previous_quaternion_xyzw=previous_joint_quaternion,
    )
    try:
        rigid = mano_rigid_vertex_palm_frame(
            result.pred_vertices_source_camera_axes, result.is_right
        )
        rigid_payload = rigid.as_dict()
    except Exception as exc:
        rigid_payload = {
            "method": "mano_rigid_vertex_palm_frame",
            "valid": False,
            "rotation": None,
            "quaternion_xyzw": None,
            "origin": None,
            "reason": f"{type(exc).__name__}:{exc}",
        }
    return {
        "raw_global_orient": raw.as_dict(),
        "mano_joint_palm_frame": joint.as_dict(),
        "mano_rigid_vertex_palm_frame": rigid_payload,
    }


def so3_geodesic_degrees(first: Any, second: Any) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != (3, 3) or second_array.shape != (3, 3):
        raise ValueError("SO(3) inputs must be 3x3")
    relative = first_array.T @ second_array
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    result = math.degrees(math.acos(cosine))
    if not math.isfinite(result):
        raise ValueError("SO(3) distance is not finite")
    return result


__all__ = [
    "LatestFrameSlot",
    "LiveFramePacket",
    "build_live_palm_estimates",
    "normalized_crop_points_to_original",
    "so3_geodesic_degrees",
]
