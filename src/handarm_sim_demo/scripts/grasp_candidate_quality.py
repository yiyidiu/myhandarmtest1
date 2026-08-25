#!/usr/bin/env python3
"""Fail-closed quality gates for stable three-finger grasp candidates."""

from dataclasses import dataclass
import math

import numpy as np


QUALITY_GATE_KEYS = (
    "maximum_centered_object_center_axial_offset_m",
    "maximum_contact_fraction_spread",
    "minimum_projected_contact_area_m2",
    "minimum_normalized_contact_area_fraction",
    "maximum_contact_height_ratio",
    "minimum_lift_world_z_fraction",
)

QUALITY_REFERENCE_MODES = ("centered_object", "top_precision_band")

TOP_PRECISION_BAND_KEYS = (
    "minimum_contact_depth_below_top_m",
    "maximum_contact_depth_below_top_m",
)


def validate_quality_gate_config(config):
    """Return normalized quality gates and reference-mode settings.

    ``quality_reference_mode`` is intentionally mandatory.  ``centered_object``
    preserves the legacy object-centre axial gate unchanged;
    ``top_precision_band`` replaces that one legacy gate with explicit
    per-contact depth bounds below the object's top surface.
    """
    if not isinstance(config, dict):
        raise ValueError("geometry config must be a mapping")
    mode = config.get("quality_reference_mode")
    if mode not in QUALITY_REFERENCE_MODES:
        raise ValueError(
            "quality_reference_mode missing or unknown; expected one of {}".format(
                list(QUALITY_REFERENCE_MODES)
            )
        )

    gates = config.get("quality_gates")
    if not isinstance(gates, dict):
        raise ValueError("geometry config must define quality_gates")
    missing = [key for key in QUALITY_GATE_KEYS if key not in gates]
    if missing:
        raise ValueError("quality_gates missing: {}".format(sorted(missing)))
    result = {"quality_reference_mode": mode}
    for key in QUALITY_GATE_KEYS:
        value = float(gates[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "quality_gates.{} must be finite and non-negative".format(key)
            )
        result[key] = value
    if result["maximum_centered_object_center_axial_offset_m"] > 0.030:
        raise ValueError("maximum centered axial offset must not exceed 0.030 m")
    if result["maximum_contact_height_ratio"] > 0.90:
        raise ValueError("contact height ratio gate must not admit edge-only grips")
    if result["minimum_lift_world_z_fraction"] < 0.70:
        raise ValueError("lift world-z fraction must be at least 0.70")

    if mode == "top_precision_band":
        band = config.get("top_precision_band")
        if not isinstance(band, dict):
            raise ValueError(
                "top_precision_band mode requires top_precision_band settings"
            )
        missing = [key for key in TOP_PRECISION_BAND_KEYS if key not in band]
        if missing:
            raise ValueError("top_precision_band missing: {}".format(sorted(missing)))
        minimum = float(band["minimum_contact_depth_below_top_m"])
        maximum = float(band["maximum_contact_depth_below_top_m"])
        if not all(math.isfinite(value) for value in (minimum, maximum)):
            raise ValueError("top_precision_band bounds must be finite")
        if minimum < 0.0 or maximum < 0.0 or maximum <= minimum:
            raise ValueError(
                "top_precision_band requires 0 <= minimum < maximum depth"
            )
        result["top_precision_band"] = {
            "minimum_contact_depth_below_top_m": minimum,
            "maximum_contact_depth_below_top_m": maximum,
        }
    return result


@dataclass(frozen=True)
class CandidateQualityResult:
    passed: bool
    failures: tuple
    metrics: dict

    def as_dict(self):
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": self.metrics,
        }


def _object_cross_section_metrics(candidate):
    obb = candidate.object_obb_hand
    hand_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    projected_extents = np.abs(obb.rotation.T @ hand_axis) * obb.half_extents
    approach_aligned_axis = int(np.argmax(projected_extents))
    cross_axes = [axis for axis in range(3) if axis != approach_aligned_axis]
    footprint_area_m2 = float(
        4.0 * obb.half_extents[cross_axes[0]] * obb.half_extents[cross_axes[1]]
    )
    return approach_aligned_axis, cross_axes, footprint_area_m2


def _object_vertical_reference(candidate):
    """Return the object OBB vertical axis and its world-up sign."""
    obb = candidate.object_obb_hand
    # ``obb.rotation`` expresses object axes in the hand frame.  Compose it
    # with hand-to-world exactly once before projecting world +z into object
    # coordinates.  The former world-to-hand multiplication applied the hand
    # rotation twice and silently selected the wrong axis for oblique grasps.
    object_to_world_rotation = (
        candidate.T_world_hand[:3, :3] @ obb.rotation
    )
    world_z_in_object = object_to_world_rotation.T @ np.array([0.0, 0.0, 1.0])
    axis = int(np.argmax(np.abs(world_z_in_object)))
    sign = 1 if world_z_in_object[axis] >= 0.0 else -1
    return axis, sign


def _object_vertical_axis(candidate):
    """Return the object OBB axis most aligned with world +z."""
    return _object_vertical_reference(candidate)[0]


def _contact_depth_below_top(candidate, vertical_axis, sign):
    obb = candidate.object_obb_hand
    half_height = float(obb.half_extents[vertical_axis])
    if half_height <= 0.0:
        raise ValueError("object height half extent is not positive")
    depths = {}
    for name, contact in candidate.enclosure.contacts.items():
        point_local = obb.rotation.T @ (contact.point_hand_m - obb.center)
        # sign * local_axis is the world-up coordinate along the selected
        # object axis, so depth below top is half_height - that coordinate.
        depth = half_height - float(sign) * float(point_local[vertical_axis])
        depths[name] = float(depth)
    return depths


def _contact_height_ratios(candidate, vertical_axis):
    obb = candidate.object_obb_hand
    half_height = float(obb.half_extents[vertical_axis])
    if half_height <= 0.0:
        raise ValueError("object height half extent is not positive")
    return {
        name: float(
            (obb.rotation.T @ (contact.point_hand_m - obb.center))[vertical_axis]
            / half_height
        )
        for name, contact in candidate.enclosure.contacts.items()
    }


def evaluate_candidate_quality(candidate, config):
    """Reject edge grips and non-vertical lift candidates before IK/execution."""
    gates = validate_quality_gate_config(config)
    mode = gates["quality_reference_mode"]
    if mode == "centered_object":
        applied = ["maximum_centered_object_center_axial_offset_m"]
    else:
        applied = list(TOP_PRECISION_BAND_KEYS)
    applied.extend(
        key
        for key in (
            "maximum_contact_fraction_spread",
            "minimum_projected_contact_area_m2",
            "minimum_normalized_contact_area_fraction",
            "maximum_contact_height_ratio",
            "minimum_lift_world_z_fraction",
        )
        if key in gates
    )
    applied_quality_gate_keys = sorted(set(applied))
    result = candidate.enclosure
    if not result.valid:
        metrics = {
            "enclosure_valid": False,
            "enclosure_failure_reasons": list(result.failure_reasons),
            "quality_reference_mode": mode,
            "applied_quality_gate_keys": applied_quality_gate_keys,
            "centered_object_axial_gate_applied": mode == "centered_object",
        }
        if mode == "top_precision_band":
            metrics["top_precision_band_m"] = {
                key: float(value)
                for key, value in gates["top_precision_band"].items()
            }
        return CandidateQualityResult(
            passed=False,
            failures=("ENCLOSURE_INVALID",),
            metrics=metrics,
        )

    failures = []
    metrics = {
        "enclosure_valid": True,
        "quality_reference_mode": mode,
        "object_center_axial_offset_m": float(
            candidate.object_center_axial_offset_m
        ),
        "centered_object_axial_gate_applied": mode == "centered_object",
    }
    if mode == "centered_object" and (
        abs(candidate.object_center_axial_offset_m)
        > gates["maximum_centered_object_center_axial_offset_m"]
    ):
        failures.append("AXIAL_OFFSET_TOO_LARGE")

    if set(result.contacts) != {"f1", "f2", "f3"}:
        failures.append("THREE_FINGER_CONTACTS_MISSING")

    fractions = {
        name: contact.closure_fraction for name, contact in result.contacts.items()
    }
    spread = max(fractions.values()) - min(fractions.values()) if fractions else math.inf
    metrics["contact_closure_fractions"] = {
        name: float(value) for name, value in sorted(fractions.items())
    }
    metrics["contact_closure_spread"] = float(spread)
    if spread > gates["maximum_contact_fraction_spread"]:
        failures.append("CONTACT_CLOSURE_SPREAD_TOO_LARGE")

    area_m2 = float(result.projected_contact_area_m2)
    metrics["projected_contact_area_m2"] = area_m2
    if area_m2 < gates["minimum_projected_contact_area_m2"]:
        failures.append("CONTACT_TRIANGLE_AREA_TOO_SMALL")

    _, _, footprint_area_m2 = _object_cross_section_metrics(candidate)
    normalized_area = area_m2 / footprint_area_m2 if footprint_area_m2 > 0.0 else 0.0
    metrics["footprint_area_m2"] = footprint_area_m2
    metrics["normalized_contact_area"] = float(normalized_area)
    if normalized_area < gates["minimum_normalized_contact_area_fraction"]:
        failures.append("NORMALIZED_CONTACT_AREA_TOO_SMALL")

    vertical_axis, vertical_sign = _object_vertical_reference(candidate)
    metrics["object_vertical_axis"] = vertical_axis
    metrics["object_vertical_sign"] = vertical_sign
    height_ratios = _contact_height_ratios(candidate, vertical_axis)
    metrics["contact_height_ratios"] = {
        name: float(value) for name, value in sorted(height_ratios.items())
    }
    maximum_abs_ratio = max((abs(value) for value in height_ratios.values()), default=0.0)
    metrics["maximum_contact_height_ratio"] = float(maximum_abs_ratio)
    if maximum_abs_ratio > gates["maximum_contact_height_ratio"]:
        failures.append("CONTACT_TOO_CLOSE_TO_OBJECT_EDGE")

    if mode == "top_precision_band":
        band = gates["top_precision_band"]
        contact_depths = _contact_depth_below_top(
            candidate, vertical_axis, vertical_sign
        )
        metrics["contact_depth_below_top_m"] = {
            name: float(value) for name, value in sorted(contact_depths.items())
        }
        metrics["top_precision_band_m"] = {
            key: float(value) for key, value in band.items()
        }
        if any(
            value < band["minimum_contact_depth_below_top_m"]
            or value > band["maximum_contact_depth_below_top_m"]
            for value in contact_depths.values()
        ):
            failures.append("CONTACT_DEPTH_OUTSIDE_TOP_PRECISION_BAND")

    metrics["applied_quality_gate_keys"] = applied_quality_gate_keys

    lift_vector = -np.asarray(candidate.T_world_hand, dtype=float)[:3, 2]
    lift_norm = float(np.linalg.norm(lift_vector))
    z_fraction = float(lift_vector[2] / lift_norm) if lift_norm > 0.0 else 0.0
    metrics["lift_vector_world"] = lift_vector.tolist()
    metrics["lift_world_z_fraction"] = z_fraction
    if z_fraction < gates["minimum_lift_world_z_fraction"]:
        failures.append("LIFT_DIRECTION_NOT_WORLD_Z_DOMINANT")

    failures = tuple(sorted(set(failures)))
    return CandidateQualityResult(not failures, failures, metrics)


def validate_planned_lift_vector(vector, config):
    """Validate a planned three-dimensional lift vector before execution."""
    gates = validate_quality_gate_config(config)
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("lift vector must be finite and 3-dimensional")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("lift vector must be non-zero")
    z_fraction = float(vector[2] / norm)
    if z_fraction < gates["minimum_lift_world_z_fraction"]:
        raise ValueError(
            "planned lift direction is not world-z dominant: {:.6f}".format(
                z_fraction
            )
        )
    return {"norm_m": norm, "world_z_fraction": z_fraction}


def evaluate_actual_lift_evidence(vector, config):
    """Pure runtime evidence check for physical object displacement."""
    runtime = config.get("runtime_acceptance", {})
    required = (
        "minimum_object_lift_m",
        "minimum_actual_object_lift_z_fraction",
    )
    missing = [key for key in required if key not in runtime]
    if missing:
        raise ValueError("runtime_acceptance missing: {}".format(sorted(missing)))
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("actual object displacement must be finite and 3-dimensional")
    norm = float(np.linalg.norm(vector))
    z_displacement = float(vector[2])
    z_fraction = z_displacement / norm if norm > 1.0e-12 else 0.0
    failures = []
    if z_displacement < float(runtime["minimum_object_lift_m"]):
        failures.append("OBJECT_WORLD_Z_LIFT_TOO_SMALL")
    if z_fraction < float(runtime["minimum_actual_object_lift_z_fraction"]):
        failures.append("OBJECT_LIFT_NOT_WORLD_Z_DOMINANT")
    return {
        "passed": not failures,
        "failures": failures,
        "displacement_m": vector.tolist(),
        "norm_m": norm,
        "world_z_lift_m": z_displacement,
        "world_z_fraction": z_fraction,
    }
