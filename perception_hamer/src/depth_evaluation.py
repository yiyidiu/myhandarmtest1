#!/usr/bin/env python3
"""Preliminary depth bias and RGB/depth edge-alignment metrics."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np


class DepthEvaluationError(ValueError):
    pass


def validate_roi(roi_xyxy: Sequence[int], width: int, height: int) -> Tuple[int, ...]:
    if len(roi_xyxy) != 4:
        raise DepthEvaluationError("ROI must be x1,y1,x2,y2")
    x1, y1, x2, y2 = (int(value) for value in roi_xyxy)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise DepthEvaluationError("ROI is outside the half-open image bounds")
    return x1, y1, x2, y2


def depth_bias_metrics(
    depth_raw: np.ndarray,
    depth_scale_m_per_unit: float,
    reference_distance_m: float,
    roi_xyxy: Sequence[int],
) -> Dict[str, Any]:
    if depth_raw.dtype != np.uint16 or depth_raw.ndim != 2:
        raise DepthEvaluationError("depth must be uint16 HxW")
    if not np.isfinite(depth_scale_m_per_unit) or depth_scale_m_per_unit <= 0.0:
        raise DepthEvaluationError("depth scale must be finite and positive")
    if not np.isfinite(reference_distance_m) or reference_distance_m <= 0.0:
        raise DepthEvaluationError("reference distance must be finite and positive")
    x1, y1, x2, y2 = validate_roi(roi_xyxy, depth_raw.shape[1], depth_raw.shape[0])
    values_m = depth_raw[y1:y2, x1:x2].astype(np.float64) * depth_scale_m_per_unit
    valid = values_m > 0.0
    if np.count_nonzero(valid) < 10:
        raise DepthEvaluationError("ROI has fewer than 10 valid depth samples")
    errors_mm = (values_m[valid] - reference_distance_m) * 1000.0
    median_error = float(np.median(errors_mm))
    return {
        "reference_distance_m": float(reference_distance_m),
        "roi_xyxy_half_open": [x1, y1, x2, y2],
        "total_samples": int(values_m.size),
        "valid_samples": int(errors_mm.size),
        "valid_fraction": float(errors_mm.size / values_m.size),
        "bias_mean_mm": float(np.mean(errors_mm)),
        "bias_median_mm": median_error,
        "mad_about_median_mm": float(np.median(np.abs(errors_mm - median_error))),
        "absolute_error_p95_mm": float(np.percentile(np.abs(errors_mm), 95)),
    }


def rgb_depth_edge_alignment_metrics(
    rgb: np.ndarray,
    aligned_depth_raw: np.ndarray,
    roi_xyxy: Sequence[int],
    depth_edge_threshold_raw: float = 30.0,
) -> Dict[str, Any]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DepthEvaluationError("RGB must be uint8 HxWx3")
    if aligned_depth_raw.dtype != np.uint16 or aligned_depth_raw.ndim != 2:
        raise DepthEvaluationError("aligned depth must be uint16 HxW")
    if rgb.shape[:2] != aligned_depth_raw.shape:
        raise DepthEvaluationError("RGB and aligned depth shapes differ")
    x1, y1, x2, y2 = validate_roi(roi_xyxy, rgb.shape[1], rgb.shape[0])
    rgb_roi = rgb[y1:y2, x1:x2]
    depth_roi = aligned_depth_raw[y1:y2, x1:x2]
    gray = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 50, 150) > 0
    depth_float = depth_roi.astype(np.float32)
    valid = depth_roi > 0
    gx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
    depth_edges = (np.hypot(gx, gy) >= float(depth_edge_threshold_raw)) & valid
    if np.count_nonzero(rgb_edges) < 5 or np.count_nonzero(depth_edges) < 5:
        raise DepthEvaluationError("insufficient RGB or depth edges in ROI")
    distance_to_rgb = cv2.distanceTransform(
        (~rgb_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    distances = distance_to_rgb[depth_edges]
    return {
        "roi_xyxy_half_open": [x1, y1, x2, y2],
        "rgb_edge_pixels": int(np.count_nonzero(rgb_edges)),
        "depth_edge_pixels": int(np.count_nonzero(depth_edges)),
        "depth_edge_threshold_raw_units": float(depth_edge_threshold_raw),
        "depth_to_nearest_rgb_edge_p50_px": float(np.percentile(distances, 50)),
        "depth_to_nearest_rgb_edge_p95_px": float(np.percentile(distances, 95)),
        "depth_to_nearest_rgb_edge_mean_px": float(np.mean(distances)),
    }
