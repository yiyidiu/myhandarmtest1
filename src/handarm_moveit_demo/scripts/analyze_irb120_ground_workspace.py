#!/usr/bin/env python3
"""Reproduce the IRB120 tool0 position envelope from the project URDF.

This is an outer position analysis, not a proof that a requested 6-D pose is
collision-free or continuously reachable from the teleoperation zero.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def vector(text, default):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def axis_rotation_batch(axis, angles):
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    values = np.asarray(angles, dtype=np.float64)
    cosine, sine = np.cos(values), np.sin(values)
    one_minus = 1.0 - cosine
    result = np.empty((len(values), 3, 3), dtype=np.float64)
    result[:, 0, 0] = cosine + x * x * one_minus
    result[:, 0, 1] = x * y * one_minus - z * sine
    result[:, 0, 2] = x * z * one_minus + y * sine
    result[:, 1, 0] = y * x * one_minus + z * sine
    result[:, 1, 1] = cosine + y * y * one_minus
    result[:, 1, 2] = y * z * one_minus - x * sine
    result[:, 2, 0] = z * x * one_minus - y * sine
    result[:, 2, 1] = z * y * one_minus + x * sine
    result[:, 2, 2] = cosine + z * z * one_minus
    return result


def read_chain(urdf_path, base_link="base_link", tip_link="tool0"):
    root = ET.parse(str(urdf_path)).getroot()
    by_child = {}
    for element in root.findall("joint"):
        child = element.find("child").attrib["link"]
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        by_child[child] = {
            "name": element.attrib["name"],
            "type": element.attrib["type"],
            "parent": element.find("parent").attrib["link"],
            "child": child,
            "xyz": vector(None if origin is None else origin.attrib.get("xyz"), [0, 0, 0]),
            "rpy": vector(None if origin is None else origin.attrib.get("rpy"), [0, 0, 0]),
            "axis": vector(None if axis is None else axis.attrib.get("xyz"), [1, 0, 0]),
            "lower": None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"]),
            "upper": None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"]),
        }
    chain = []
    link = tip_link
    while link != base_link:
        if link not in by_child:
            raise ValueError("no URDF chain from {} back to {}".format(tip_link, base_link))
        joint = by_child[link]
        chain.append(joint)
        link = joint["parent"]
    chain.reverse()
    movable = [joint for joint in chain if joint["type"] in ("revolute", "continuous")]
    for joint in movable:
        if joint["lower"] is None or joint["upper"] is None:
            raise ValueError("joint {} lacks finite sampling limits".format(joint["name"]))
    return chain, movable


def forward_batch(chain, joint_values_by_name, count):
    rotation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    position = np.zeros((count, 3), dtype=np.float64)
    for joint in chain:
        position += np.einsum("nij,j->ni", rotation, joint["xyz"])
        rotation = rotation @ np.broadcast_to(
            rpy_matrix(joint["rpy"]), (count, 3, 3))
        if joint["type"] in ("revolute", "continuous"):
            rotation = rotation @ axis_rotation_batch(
                joint["axis"], joint_values_by_name[joint["name"]])
    return position, rotation


def default_urdf():
    source_root = Path(__file__).resolve().parents[2]
    candidate = source_root / "abb120_moveit_config1/config/gazebo_handarm_velocity.urdf"
    return str(candidate)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=default_urdf())
    parser.add_argument("--samples", type=int, default=1000000)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=120)
    parser.add_argument("--front-min-x-m", type=float, default=0.0)
    parser.add_argument("--ground-z-m", type=float, default=0.0)
    parser.add_argument(
        "--initial-joints-deg", nargs=6, type=float,
        default=[0.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples <= 0 or args.chunk_size <= 0:
        raise SystemExit("samples and chunk-size must be positive")
    urdf_path = Path(args.urdf).expanduser().resolve()
    chain, movable = read_chain(urdf_path)
    if len(movable) != 6:
        raise SystemExit("expected six movable arm joints, found {}".format(len(movable)))
    rng = np.random.default_rng(args.seed)
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    maximum_origin_radius = 0.0
    sector_count = 0

    for offset in range(0, args.samples, args.chunk_size):
        count = min(args.chunk_size, args.samples - offset)
        values = {
            joint["name"]: rng.uniform(joint["lower"], joint["upper"], count)
            for joint in movable
        }
        positions, _ = forward_batch(chain, values, count)
        minimum = np.minimum(minimum, np.min(positions, axis=0))
        maximum = np.maximum(maximum, np.max(positions, axis=0))
        maximum_origin_radius = max(
            maximum_origin_radius,
            float(np.max(np.linalg.norm(positions, axis=1))))
        sector_count += int(np.count_nonzero(
            (positions[:, 0] >= args.front_min_x_m) &
            (positions[:, 2] >= args.ground_z_m)))

    initial_values = {
        joint["name"]: np.asarray([math.radians(value)])
        for joint, value in zip(movable, args.initial_joints_deg)
    }
    initial_position, initial_rotation = forward_batch(chain, initial_values, 1)
    center = 0.5 * (minimum + maximum)
    radii = 0.5 * (maximum - minimum)
    result = {
        "schema_version": 1,
        "analysis": "POSITION_ONLY_UNIFORM_JOINT_MONTE_CARLO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "urdf": str(urdf_path),
        "base_link": "base_link",
        "tip_link": "tool0",
        "samples": args.samples,
        "seed": args.seed,
        "position_min_m": minimum.tolist(),
        "position_max_m": maximum.tolist(),
        "maximum_base_origin_radius_m": maximum_origin_radius,
        "front_ground_sector": {
            "minimum_x_m": args.front_min_x_m,
            "minimum_z_m": args.ground_z_m,
            "sample_count": sector_count,
            "sample_fraction": float(sector_count) / float(args.samples),
        },
        "suggested_outer_ellipsoid": {
            "center_base_m": center.tolist(),
            "radii_m": radii.tolist(),
        },
        "initial_joint_positions_deg": args.initial_joints_deg,
        "initial_tool0_position_m": initial_position[0].tolist(),
        "initial_tool0_rotation_row_major": initial_rotation[0].reshape(-1).tolist(),
        "limitations": [
            "no collision checking",
            "no orientation-conditioned reachability",
            "no connected-IK-branch guarantee",
            "no joint/singularity safety margin",
        ],
    }
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(result, stream, sort_keys=False)
        print(str(destination))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
