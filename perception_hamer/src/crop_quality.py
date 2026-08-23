#!/usr/bin/env python3
"""Causal image-space quality for a hand crop.

Adapted from the MIT-licensed ``v9/crop_quality.py`` in the user-supplied
``teleoperation_ubuntu_core`` archive.  This module uses only the current and
previous accepted boxes; it never uses future frames or MediaPipe landmarks as
pose data.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def bbox_crop_quality(
    bbox: Any,
    previous_bbox: Optional[Any],
    image_width: int,
    image_height: int,
) -> float:
    """Score crop support, border truncation and frame-to-frame box jitter."""

    try:
        current = np.asarray(bbox, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return 0.0
    if not np.all(np.isfinite(current)):
        return 0.0
    size = current[2:] - current[:2]
    if np.any(size <= 0.0) or int(image_width) <= 1 or int(image_height) <= 1:
        return 0.0

    diagonal = float(np.linalg.norm(size))
    size_quality = float(np.clip(min(size) / 32.0, 0.0, 1.0))
    aspect_quality = math.sqrt(float(min(size) / max(size)))
    border_distance = min(
        float(current[0]),
        float(current[1]),
        float(image_width - 1 - current[2]),
        float(image_height - 1 - current[3]),
    )
    border_quality = float(
        np.clip(border_distance / max(0.15 * max(size), 1.0), 0.0, 1.0)
    )

    temporal_quality = 1.0
    if previous_bbox is not None:
        try:
            previous = np.asarray(previous_bbox, dtype=np.float64).reshape(4)
        except (TypeError, ValueError):
            previous = np.full(4, np.nan)
        previous_size = previous[2:] - previous[:2]
        if np.all(np.isfinite(previous)) and np.all(previous_size > 0.0):
            center_shift = float(
                np.linalg.norm(
                    0.5 * (current[:2] + current[2:])
                    - 0.5 * (previous[:2] + previous[2:])
                )
                / max(float(np.linalg.norm(previous_size)), diagonal, 1.0)
            )
            scale_change = abs(
                math.log(
                    math.sqrt(float(np.prod(size)))
                    / max(math.sqrt(float(np.prod(previous_size))), 1.0e-6)
                )
            )
            temporal_quality = math.exp(
                -0.35 * (center_shift / 0.50) ** 2
                -0.35 * (scale_change / 0.35) ** 2
            )

    quality = (
        size_quality
        * (0.65 + 0.35 * aspect_quality)
        * (0.50 + 0.50 * border_quality)
        * temporal_quality
    )
    return float(np.clip(quality, 0.0, 1.0))


__all__ = ["bbox_crop_quality"]
