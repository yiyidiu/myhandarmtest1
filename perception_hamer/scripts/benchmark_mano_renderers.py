#!/usr/bin/env python3
"""Deterministic CPU benchmark for the two OpenCV MANO display paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from perception_hamer.src.realtime_hamer_pipeline import (  # noqa: E402
    draw_mano_mesh_overlay,
)
from perception_hamer.src.teleoperation_core_mano_renderer import (  # noqa: E402
    draw_mesh,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = 20, 39
    x_grid, y_grid = np.meshgrid(
        np.linspace(250.0, 390.0, columns),
        np.linspace(150.0, 330.0, rows),
    )
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))[:778]
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            first = row * columns + column
            candidates = (
                (first, first + 1, first + columns),
                (first + 1, first + columns + 1, first + columns),
            )
            for face in candidates:
                if max(face) < len(points):
                    faces.append(face)
    # MANO_RIGHT has 1538 faces.  Repeat local valid faces solely so both
    # renderers receive the same real topology count without loading pickle.
    base_faces = list(faces)
    while len(faces) < 1538:
        faces.append(base_faces[len(faces) % len(base_faces)])
    triangles = np.asarray(faces[:1538], dtype=np.int64)
    depth = np.linspace(0.45, 0.70, len(points))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return image, points, depth, triangles


def _measure(function, iterations: int) -> dict:
    for _ in range(3):
        function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        samples.append(1000.0 * (time.perf_counter() - started))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "iterations": int(iterations),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    if args.iterations < 5:
        raise SystemExit("--iterations must be at least 5")
    image, points, depth, faces = _fixture()
    result = {
        "schema": "mano_renderer_cpu_benchmark_v1",
        "fixture": {
            "image": "640x480",
            "vertices": int(len(points)),
            "faces": int(len(faces)),
            "note": "deterministic local synthetic topology; timing only",
        },
        "teleoperation_core_exact_frame": _measure(
            lambda: draw_mesh(image, points, faces), args.iterations
        ),
        "legacy_depth_bands": _measure(
            lambda: draw_mano_mesh_overlay(image, points, depth, faces),
            args.iterations,
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
