#!/usr/bin/env python3
"""Measure whether existing development motions separate P/O task subspaces.

The input stage clusters were produced before this academic track and are all
method-development data.  This script reconstructs each normalized 6-D
increment from the stored primary step and direction, then compares only the
translation-energy versus rotation-energy ratio.  It does not train or
validate a decoder and must never relabel these records as held-out evidence.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


EXPECTED_ROLE = "METHOD_DEVELOPMENT_ONLY_NOT_INDEPENDENT_VALIDATION"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def label_summary(items, label):
    values = [score for item_label, score, _ in items if item_label == label]
    return {
        "count": len(values),
        "median_log_translation_rotation_energy_ratio": statistics.median(values),
        "q05": quantile(values, 0.05),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "q95": quantile(values, 0.95),
        "zero_threshold_correct_fraction": sum(
            (score > 0.0) == (label == "POSITION") for score in values
        ) / len(values),
    }


def discrimination_summary(items):
    position = [score for label, score, _ in items if label == "POSITION"]
    rotation = [score for label, score, _ in items if label == "ROTATION"]
    if not position or not rotation:
        raise ValueError("both POSITION and ROTATION clusters are required")
    auc = sum(
        1.0 if p_value > r_value else 0.5 if p_value == r_value else 0.0
        for p_value in position
        for r_value in rotation
    ) / (len(position) * len(rotation))
    thresholds = sorted({score for _, score, _ in items})
    exploratory_best = None
    for threshold in thresholds:
        sensitivity = sum(value > threshold for value in position) / len(position)
        specificity = sum(value <= threshold for value in rotation) / len(rotation)
        balanced_accuracy = 0.5 * (sensitivity + specificity)
        candidate = (balanced_accuracy, threshold, sensitivity, specificity)
        if exploratory_best is None or candidate > exploratory_best:
            exploratory_best = candidate
    zero_correct = sum(
        (score > 0.0) == (label == "POSITION")
        for label, score, _ in items
    )
    return {
        "count": len(items),
        "position": label_summary(items, "POSITION"),
        "rotation": label_summary(items, "ROTATION"),
        "auc_position_score_higher": auc,
        "zero_threshold_accuracy": zero_correct / len(items),
        "exploratory_best_threshold_not_frozen": {
            "balanced_accuracy": exploratory_best[0],
            "log_energy_ratio_threshold": exploratory_best[1],
            "position_sensitivity": exploratory_best[2],
            "rotation_specificity": exploratory_best[3],
        },
        "zero_threshold_misclassified_ids": [
            identifier
            for label, score, identifier in items
            if (score > 0.0) != (label == "POSITION")
        ],
    }


def distribution(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "median": statistics.median(values),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "minimum": min(values),
        "maximum": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-clusters", type=Path, required=True)
    parser.add_argument("--coupling-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.coupling_candidate.read_text(encoding="utf-8"))
    scales = [float(value) for value in candidate["channel_scales"]]
    if len(scales) != 6 or any(value <= 0.0 for value in scales):
        raise ValueError("candidate must define six positive channel_scales")

    records = [
        json.loads(line)
        for line in args.stage_clusters.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    admitted = [record for record in records if record.get("analysis_admissible")]
    unexpected_roles = sorted({
        str(record.get("dataset_role"))
        for record in admitted
        if record.get("dataset_role") != EXPECTED_ROLE
    })
    if unexpected_roles:
        raise ValueError("unexpected dataset roles: {}".format(unexpected_roles))

    step_items = []
    cluster_items = []
    rotation_off_task_endpoint_mm = []
    position_off_task_endpoint_deg = []
    for record in admitted:
        task_label = (
            "POSITION"
            if str(record["primary_channel_name"]).startswith("P")
            else "ROTATION"
        )
        cluster_scores = []
        endpoint = [0.0] * 6
        for step_index, step in enumerate(record["tangent_steps"]):
            primary = float(step["primary_step_normalized"])
            direction = [float(value) for value in step["normalized_direction"]]
            if len(direction) != 6:
                raise ValueError("normalized_direction must contain six values")
            increment = [primary * value for value in direction]
            translation_energy = math.sqrt(sum(value * value for value in increment[:3]))
            rotation_energy = math.sqrt(sum(value * value for value in increment[3:]))
            score = math.log(
                (translation_energy + 1.0e-12) /
                (rotation_energy + 1.0e-12)
            )
            identifier = "{}:step_{:02d}".format(record["cluster_id"], step_index)
            step_items.append((task_label, score, identifier))
            cluster_scores.append(score)
            for channel in range(6):
                endpoint[channel] += increment[channel] * scales[channel]
        if not cluster_scores:
            continue
        cluster_items.append((
            task_label,
            statistics.median(cluster_scores),
            record["cluster_id"],
        ))
        if task_label == "ROTATION":
            rotation_off_task_endpoint_mm.append(
                1000.0 * math.sqrt(sum(value * value for value in endpoint[:3]))
            )
        else:
            position_off_task_endpoint_deg.append(
                math.degrees(math.sqrt(sum(value * value for value in endpoint[3:])))
            )

    result = {
        "schema": "handarm_m2_development_subspace_separability_v1",
        "inputs": {
            "stage_clusters": {
                "path": str(args.stage_clusters),
                "sha256": file_sha256(args.stage_clusters),
            },
            "coupling_candidate_used_only_for_channel_scales": {
                "path": str(args.coupling_candidate),
                "sha256": file_sha256(args.coupling_candidate),
            },
        },
        "channel_scales": scales,
        "score_definition": (
            "log(norm(normalized translation increment) / "
            "norm(normalized rotation increment)); positive favors POSITION"
        ),
        "step_equal": discrimination_summary(step_items),
        "stage_cluster_equal": discrimination_summary(cluster_items),
        "pilot_endpoint_off_task_distributions": {
            "rotation_task_translation_mm": distribution(
                rotation_off_task_endpoint_mm
            ),
            "position_task_rotation_deg": distribution(
                position_off_task_endpoint_deg
            ),
            "interpretation": (
                "sums only admitted local tangent increments; these are input-side "
                "development proxies, not robot end-effector outcomes"
            ),
        },
        "dataset_role": "DEVELOPMENT_ONLY",
        "independent_validation_claimed": False,
        "neural_intent_claimed": False,
        "decision": (
            "The development signal is separable enough to justify one frozen "
            "causal candidate falsification, but overlap forbids treating energy "
            "dominance as observed task intent or deploying it without validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
