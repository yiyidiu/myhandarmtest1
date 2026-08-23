#!/usr/bin/env python3
"""ROS-free, single-variable size diagnostic for three-finger geometry."""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml


PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from grasp_candidate_quality import evaluate_candidate_quality, validate_quality_gate_config
from grasp_geometry import HandGeometry, transform


TABLE_Z = 0.37
DEFAULT_URDF = os.path.abspath(
    os.path.join(
        PACKAGE_PATH,
        "..",
        "abb120_moveit_config1",
        "config",
        "gazebo_handarm.urdf",
    )
)
DEFAULT_GEOMETRY = os.path.join(
    PACKAGE_PATH, "config", "three_finger_grasp_geometry.yaml"
)


def box_inertia_kg_m2(mass_kg, size_m):
    size = np.asarray(size_m, dtype=float)
    if size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError("size must be three positive values")
    mass = float(mass_kg)
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    x2, y2, z2 = size[0] ** 2, size[1] ** 2, size[2] ** 2
    return {
        "ixx_kg_m2": mass * (y2 + z2) / 12.0,
        "iyy_kg_m2": mass * (x2 + z2) / 12.0,
        "izz_kg_m2": mass * (x2 + y2) / 12.0,
    }


def best_top_down_candidate(geometry, size_m, xy_m, geometry_config, roll_step_deg=15):
    center_z = TABLE_Z + 0.5 * size_m[2]
    T_world_object = transform(np.eye(3), [xy_m[0], xy_m[1], center_z])
    best = None
    for roll_deg in range(0, 360, int(roll_step_deg)):
        candidate = geometry.make_candidate(
            T_world_object,
            size_m,
            TABLE_Z,
            "top_down",
            "object_pos_z",
            float(roll_deg),
        )
        if not candidate.enclosure.valid:
            continue
        if not evaluate_candidate_quality(candidate, geometry_config).passed:
            continue
        if best is None or candidate.enclosure.projected_contact_area_m2 > (
            best.enclosure.projected_contact_area_m2
        ):
            best = candidate
    return best


def evaluate_historical_oblique_candidate(geometry, size_m, xy_m):
    center_z = TABLE_Z + 0.5 * size_m[2]
    T_world_object = transform(np.eye(3), [xy_m[0], xy_m[1], center_z])
    return geometry.make_candidate(
        T_world_object,
        size_m,
        TABLE_Z,
        "top_oblique",
        "object_pos_z",
        268.0,
        (0.006, -0.009),
        0.0,
        -30.0,
        0.052,
    )


def summarize_candidate(candidate, geometry_config):
    enclosure = candidate.enclosure
    fractions = {
        name: contact.closure_fraction for name, contact in enclosure.contacts.items()
    }
    quality = evaluate_candidate_quality(candidate, geometry_config)
    lift_vector = -np.asarray(candidate.T_world_hand, dtype=float)[:3, 2]
    lift_norm = float(np.linalg.norm(lift_vector))
    return {
        "family": candidate.family,
        "roll_deg": candidate.roll_deg,
        "tilt_deg": candidate.tilt_deg,
        "object_center_axial_offset_m": candidate.object_center_axial_offset_m,
        "center_offset_hand_m": candidate.center_offset_hand_m.tolist(),
        "enclosure_valid": enclosure.valid,
        "enclosure_failure_reasons": list(enclosure.failure_reasons),
        "contact_closure_fractions": {
            name: float(fractions.get(name, math.nan)) for name in ("f1", "f2", "f3")
        },
        "contact_closure_spread": (
            max(fractions.values()) - min(fractions.values()) if fractions else math.nan
        ),
        "contact_triangle_area_m2": enclosure.projected_contact_area_m2,
        "palm_clearance_m": enclosure.palm_clearance_m,
        "table_clearance_m": enclosure.table_clearance_m,
        "planned_lift_vector_world": lift_vector.tolist(),
        "planned_lift_world_z_fraction": (
            float(lift_vector[2] / lift_norm) if lift_norm > 0.0 else math.nan
        ),
        "quality_passed": quality.passed,
        "quality_failures": list(quality.failures),
        "quality_metrics": quality.metrics,
    }


def run_size_pose_diagnostic(urdf_path=DEFAULT_URDF, geometry_path=DEFAULT_GEOMETRY):
    with open(geometry_path, "r", encoding="utf-8") as stream:
        geometry_config = yaml.safe_load(stream)
    validate_quality_gate_config(geometry_config)
    geometry = HandGeometry(urdf_path, geometry_path)
    sizes = {
        "baseline_50x60x100": [0.05, 0.06, 0.10],
        "taller_only_50x60x140": [0.05, 0.06, 0.14],
        "larger_70x80x140": [0.07, 0.08, 0.14],
        "moderate_counterexample_60x60x120": [0.06, 0.06, 0.12],
    }
    xy = [0.34, 0.18]
    rows = {}
    for name, size_m in sizes.items():
        top_down = best_top_down_candidate(geometry, size_m, xy, geometry_config)
        oblique = evaluate_historical_oblique_candidate(geometry, size_m, xy)
        rows[name] = {
            "size_m": list(size_m),
            "mass_kg": 0.10,
            "xy_m": list(xy),
            "object_center_z_m": TABLE_Z + 0.5 * size_m[2],
            "inertia": box_inertia_kg_m2(0.10, size_m),
            "centered_top_down_best": (
                summarize_candidate(top_down, geometry_config)
                if top_down is not None
                else {"available": False, "quality_passed": False}
            ),
            "historical_oblique_268_m30_52mm": summarize_candidate(
                oblique, geometry_config
            ),
        }
    return {
        "table_z_m": TABLE_Z,
        "geometry_config": geometry_path,
        "urdf": urdf_path,
        "quality_gates": validate_quality_gate_config(geometry_config),
        "interpretation": {
            "single_variable_order": (
                "First test 70x80x140 at unchanged xy; do not change size and xy together."
            ),
            "height_effect": (
                "Increasing height from 100 to 140 mm raises analytical hand/table "
                "clearance by about 20 mm, but reduces palm/object clearance by about 20 mm."
            ),
            "historical_oblique_52mm": (
                "Rejected as an axial edge grip; its earlier 21 mm slide/lift is not success."
            ),
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--geometry", default=DEFAULT_GEOMETRY)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    record = run_size_pose_diagnostic(args.urdf, args.geometry)
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.json_output:
        with open(args.json_output, "x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
