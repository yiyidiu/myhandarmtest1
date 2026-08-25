#!/usr/bin/env python3
"""Build a deterministic, reduced STL for MoveIt hand-base collision checks.

The CAD visual mesh has roughly 466k triangles.  Using it directly in the
60 Hz Servo collision monitor is unnecessarily expensive, while the former
single cylinder fills the finger-root cut-outs and produces permanent false
self-collisions.  Vertex clustering retains those cut-outs at a configurable
resolution and emits a compact triangle soup accepted by FCL/Gazebo.
"""

import argparse
from pathlib import Path
import struct

import numpy as np


STL_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])


def read_binary_stl(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(80)
        count_bytes = stream.read(4)
    if len(count_bytes) != 4:
        raise ValueError("truncated STL: {}".format(path))
    triangle_count = struct.unpack("<I", count_bytes)[0]
    expected_size = 84 + triangle_count * STL_DTYPE.itemsize
    if expected_size != size:
        raise ValueError(
            "only binary STL is supported: size={} expected={}".format(
                size, expected_size))
    records = np.fromfile(
        str(path), dtype=STL_DTYPE, offset=84, count=triangle_count)
    return header, records["vertices"].astype(np.float64)


def clustered_triangles(vertices, grid_m):
    flat = vertices.reshape(-1, 3)
    quantized = np.rint(flat / grid_m).astype(np.int32)
    unique_cells, inverse = np.unique(
        quantized, axis=0, return_inverse=True)
    triangle_indices = inverse.reshape(-1, 3)
    nondegenerate = (
        (triangle_indices[:, 0] != triangle_indices[:, 1]) &
        (triangle_indices[:, 1] != triangle_indices[:, 2]) &
        (triangle_indices[:, 0] != triangle_indices[:, 2]))
    triangle_indices = triangle_indices[nondegenerate]

    # Duplicate CAD facets become the same clustered triangle.  Compare a
    # sorted copy but retain the first facet's original winding.
    canonical = np.sort(triangle_indices, axis=1)
    _, first_indices = np.unique(canonical, axis=0, return_index=True)
    triangle_indices = triangle_indices[np.sort(first_indices)]
    clustered_vertices = unique_cells.astype(np.float64) * grid_m
    return clustered_vertices[triangle_indices]


def write_binary_stl(path, triangles, source_name, grid_m):
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    triangles = triangles[valid]
    normals = normals[valid] / lengths[valid, None]

    records = np.zeros(len(triangles), dtype=STL_DTYPE)
    records["normal"] = normals.astype(np.float32)
    records["vertices"] = triangles.astype(np.float32)
    header_text = (
        "handarm collision mesh; source={}; grid_m={:.6f}".format(
            source_name, grid_m))
    header = header_text.encode("ascii")[:80].ljust(80, b"\0")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(records)))
        records.tofile(stream)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid-mm", type=float, default=8.0)
    args = parser.parse_args()
    if not np.isfinite(args.grid_mm) or args.grid_mm <= 0.0:
        parser.error("--grid-mm must be finite and positive")
    grid_m = args.grid_mm / 1000.0
    _, source_triangles = read_binary_stl(args.input)
    triangles = clustered_triangles(source_triangles, grid_m)
    output_count = write_binary_stl(
        args.output, triangles, args.input.name, grid_m)
    print("source_triangles={}".format(len(source_triangles)))
    print("output_triangles={}".format(output_count))
    print("output={}".format(args.output))


if __name__ == "__main__":
    main()
