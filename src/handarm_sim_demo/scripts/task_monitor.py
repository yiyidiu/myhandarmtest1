#!/usr/bin/env python3
"""Consolidate immutable per-run simulation evidence into final summary files."""

import argparse
import csv
import glob
import json
import os
import statistics


SCENARIOS = (
    "no_obstacle", "single_obstacle", "double_obstacle", "unreachable", "fully_blocked"
)
REQUIRED_POSITIVE_TRIALS = {"no_obstacle": 3, "single_obstacle": 10, "double_obstacle": 10}


def load(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def latest_best(paths, eligibility, score):
    candidates = []
    for path in paths:
        data = load(path)
        if eligibility(data):
            candidates.append((score(data), os.path.getmtime(path), path, data))
    if not candidates:
        raise RuntimeError("no eligible evidence among {}".format(paths))
    return max(candidates, key=lambda item: (item[0], item[1]))[2:]


def median(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.median(values) if values else None


def summarize(results_dir):
    avoidance = {}
    avoidance_rows = []
    for scenario in SCENARIOS:
        paths = glob.glob(os.path.join(results_dir, "avoidance_{}_*.json".format(scenario)))
        eligible = [(os.path.getmtime(path), path, load(path)) for path in paths]
        eligible = [item for item in eligible if item[2].get("all_outcomes_pass")]
        if not eligible:
            raise RuntimeError("no passing avoidance evidence for {}".format(scenario))
        eligible.sort(reverse=True)
        if scenario in REQUIRED_POSITIVE_TRIALS:
            required = REQUIRED_POSITIVE_TRIALS[scenario]
            rows, evidence = [], []
            for _, path, data in eligible:
                remaining = required - len(rows)
                if remaining <= 0:
                    break
                selected = data["results"][:remaining]
                if selected:
                    rows.extend(selected)
                    evidence.append(os.path.basename(path))
            if len(rows) != required:
                raise RuntimeError("{} has only {}/{} passing trials".format(scenario, len(rows), required))
        else:
            _, path, data = eligible[0]
            rows, evidence = data["results"], [os.path.basename(path)]
        avoidance_rows.extend(rows)
        avoidance[scenario] = {
            "evidence": evidence,
            "trials": len(rows),
            "plan_successes": sum(bool(row.get("plan_success")) for row in rows),
            "execution_successes": sum(bool(row.get("execution_success")) for row in rows),
            "outcome_passes": sum(bool(row.get("outcome_pass")) for row in rows),
            "planning_time_median_s": median([row.get("planning_time_s") for row in rows]),
            "final_position_error_max_m": max(
                [row["final_position_error_m"] for row in rows if row.get("final_position_error_m") is not None],
                default=None,
            ),
            "final_orientation_error_max_deg": max(
                [row["final_orientation_error_deg"] for row in rows if row.get("final_orientation_error_deg") is not None],
                default=None,
            ),
        }

    hand_path, hand = latest_best(
        glob.glob(os.path.join(results_dir, "hand_cycles_*.json")),
        lambda item: bool(item.get("all_success")) and item.get("cycles_per_sequence") == 3,
        lambda item: len(item.get("rows", [])),
    )
    hand_rows = hand["rows"]
    command_metrics = {}
    for command in sorted({row["command"] for row in hand_rows}):
        rows = [row for row in hand_rows if row["command"] == command]
        command_metrics[command] = {
            "samples": len(rows),
            "execution_time_median_s": median([row["execution_time_s"] for row in rows]),
            "active_joint_error_max_rad": max(
                max(abs(value) for value in row["active_joint_errors_rad"].values()) for row in rows
            ),
            "mimic_error_max_rad": max(
                max(abs(value) for value in row["mimic_joint_errors_rad"].values()) for row in rows
            ),
            "mean_tip_pairwise_distance_median_m": median(
                [statistics.mean(row["tip_pairwise_distances_m"]) for row in rows]
            ),
        }

    approach_path, approach = latest_best(
        glob.glob(os.path.join(results_dir, "pick_approach_only_*.json")),
        lambda item: bool(item.get("all_success")),
        lambda item: len(item.get("rows", [])),
    )
    pick_path, pick = latest_best(
        glob.glob(os.path.join(results_dir, "pick_deterministic_lift_*.json")),
        lambda item: bool(item.get("all_success"))
        and all(row.get("attachment_used") for row in item.get("rows", []))
        and all(row.get("lift_execution_success") for row in item.get("rows", [])),
        lambda item: len(item.get("rows", [])),
    )
    pick_rows = pick["rows"]
    summary = {
        "schema_version": 1,
        "scope": "simulation_only_known_scene_autonomous_baseline",
        "safety": {"simulation": True, "use_real_robot": False, "use_real_hand": False},
        "avoidance": avoidance,
        "avoidance_success_rates": {
            scenario: {
                "planning": value["plan_successes"] / value["trials"],
                "execution": value["execution_successes"] / value["trials"],
            }
            for scenario, value in avoidance.items()
            if scenario in ("no_obstacle", "single_obstacle", "double_obstacle")
        },
        "hand": {
            "evidence": os.path.basename(hand_path), "rows": len(hand_rows),
            "all_success": hand["all_success"], "command_metrics": command_metrics,
        },
        "scripted_pick": {
            "evidence": os.path.basename(pick_path), "trials": len(pick_rows),
            "all_success": pick["all_success"],
            "cartesian_fraction_min": min(row["approach_fraction"] for row in pick_rows),
            "pregrasp_position_error_max_m": max(row["pregrasp_position_error_m"] for row in pick_rows),
            "pregrasp_orientation_error_max_deg": max(row["pregrasp_orientation_error_deg"] for row in pick_rows),
            "approach_position_error_max_m": max(row["approach_position_error_m"] for row in pick_rows),
            "approach_orientation_error_max_deg": max(row["approach_orientation_error_deg"] for row in pick_rows),
            "attachment_used": all(row["attachment_used"] for row in pick_rows),
            "attachment_type": pick_rows[0]["attachment_type"],
            "lift_fraction_min": min(row["lift_fraction"] for row in pick_rows),
            "object_lift_min_m": min(row["object_lift_m"] for row in pick_rows),
            "physical_grasp_claimed": False,
        },
        "approach_only_baseline": {
            "evidence": os.path.basename(approach_path),
            "trials": len(approach["rows"]),
            "all_success": approach["all_success"],
        },
        "deterministic_lift": {
            "status": "PASS_SIMULATION_FIXED_JOINT_NON_PHYSICAL",
            "evidence": os.path.basename(pick_path),
        },
        "physical_contact_grasp": "NOT_RUN",
        "screenshots": "NOT_RUN_HEADLESS",
        "videos": "NOT_RUN_HEADLESS",
    }
    return summary, avoidance_rows


def write_outputs(results_dir, summary, avoidance_rows):
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    csv_path = os.path.join(results_dir, "avoidance_results.csv")
    fields = sorted({key for row in avoidance_rows for key in row})
    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(avoidance_rows)
    return summary_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    default = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "sim_baseline"))
    parser.add_argument("--results-dir", default=default)
    args = parser.parse_args()
    paths = write_outputs(args.results_dir, *summarize(args.results_dir))
    print("\n".join(paths))


if __name__ == "__main__":
    main()
