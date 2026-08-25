#!/usr/bin/env python3
"""Fail-closed no-load stability test for the simulated three-finger hand."""

import csv
import datetime
import json
import math
import os
import sys
import threading
import time

import rospkg
import rospy
import yaml
from gazebo_msgs.srv import GetJointProperties
from std_msgs.msg import Bool, String


PACKAGE_PATH = rospkg.RosPack().get_path("handarm_sim_demo")
sys.path.insert(0, os.path.join(PACKAGE_PATH, "scripts"))

from hand_commander import HandCommander


SEQUENCE = ("RELEASE", "GRASP", "RELEASE")


class MimicDiagnosticMonitor:
    """Collect event-only 1 kHz plugin diagnostics without controlling joints."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events = []
        self._subscriber = rospy.Subscriber(
            "/handarm_sim_demo/mimic_diagnostics",
            String,
            self._callback,
            queue_size=100,
        )

    def _callback(self, message):
        try:
            event = json.loads(message.data)
        except (TypeError, ValueError):
            event = {"invalid_json": message.data}
        event["received_monotonic_s"] = time.monotonic()
        with self._lock:
            self._events.append(event)

    def messages_between(self, started_s, ended_s):
        with self._lock:
            return [
                dict(event)
                for event in self._events
                if started_s <= event["received_monotonic_s"] <= ended_s
            ]

    def velocity_events_between(self, started_s, ended_s):
        return [
            event for event in self.messages_between(started_s, ended_s)
            if event.get("type") != "heartbeat"
        ]

    def heartbeats_between(self, started_s, ended_s):
        return [
            event for event in self.messages_between(started_s, ended_s)
            if event.get("type") == "heartbeat"
        ]


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    index = int(math.ceil((percent / 100.0) * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def all_joint_limits(config):
    limits = dict(config["joint_limits"])
    for mimic, relation in config["mimic_joints"].items():
        source_lower, source_upper = limits[relation["source"]]
        multiplier = float(relation["multiplier"])
        offset = float(relation["offset"])
        endpoints = (
            multiplier * source_lower + offset,
            multiplier * source_upper + offset,
        )
        limits[mimic] = [min(endpoints), max(endpoints)]
    return limits


def evaluate_heartbeat_coverage(heartbeats, expected_mimics, threshold):
    """Fail closed unless every mimic plugin proves continuous post-step coverage."""
    failures = []
    metrics = {}
    for name in expected_mimics:
        rows = sorted(
            (row for row in heartbeats if row.get("mimic_joint") == name),
            key=lambda row: row.get("sim_time_s", -math.inf),
        )
        if len(rows) < 20:
            failures.append("{} diagnostic heartbeat count".format(name))
            continue
        sim_times = [float(row["sim_time_s"]) for row in rows]
        gaps = [b - a for a, b in zip(sim_times, sim_times[1:])]
        update_counts = [int(row.get("window_update_count", 0)) for row in rows]
        max_source = max(
            float(row.get("max_abs_source_velocity_rad_s", math.inf))
            for row in rows
        )
        max_mimic = max(
            float(row.get("max_abs_mimic_velocity_rad_s", math.inf))
            for row in rows
        )
        metrics[name] = {
            "heartbeat_count": len(rows),
            "sim_time_span_s": sim_times[-1] - sim_times[0],
            "max_sim_time_gap_s": max(gaps) if gaps else math.inf,
            "minimum_window_update_count": min(update_counts),
            "maximum_post_step_source_velocity_rad_s": max_source,
            "maximum_post_step_mimic_velocity_rad_s": max_mimic,
        }
        if metrics[name]["sim_time_span_s"] < 2.0:
            failures.append("{} diagnostic heartbeat span".format(name))
        if metrics[name]["max_sim_time_gap_s"] > 0.20:
            failures.append("{} diagnostic heartbeat gap".format(name))
        if metrics[name]["minimum_window_update_count"] < 50:
            failures.append("{} diagnostic update coverage".format(name))
        if max(max_source, max_mimic) > threshold:
            failures.append("{} post-step settled velocity".format(name))
    return {
        "success": not failures,
        "failure_reasons": failures,
        "joint_metrics": metrics,
    }


def classify_service_rate_outliers(
    joint_metrics, heartbeats, threshold, max_supported_per_joint=1
):
    """Cross-check asynchronous service spikes against post-step windows."""
    unresolved = []
    for joint, metrics in joint_metrics.items():
        supported_count = 0
        for outlier in metrics.get("service_rate_outliers", []):
            before = outlier.get("sim_time_before_s")
            after = outlier.get("sim_time_after_s")
            outlier["classification"] = "UNRESOLVED"
            if before is None or after is None:
                unresolved.append(joint)
                continue
            for heartbeat in heartbeats:
                if joint not in (
                    heartbeat.get("joint"), heartbeat.get("mimic_joint")
                ):
                    continue
                if (
                    float(heartbeat.get("window_start_sim_time_s", math.inf))
                    <= before
                    and float(heartbeat.get("sim_time_s", -math.inf)) >= after
                ):
                    key = (
                        "max_abs_source_velocity_rad_s"
                        if joint == heartbeat.get("joint")
                        else "max_abs_mimic_velocity_rad_s"
                    )
                    post_step_max = float(heartbeat.get(key, math.inf))
                    outlier["post_step_window_max_velocity_rad_s"] = post_step_max
                    if post_step_max <= threshold:
                        outlier["classification"] = (
                            "ASYNC_SERVICE_MID_UPDATE_ARTIFACT_SUPPORTED"
                        )
                        supported_count += 1
                    else:
                        outlier["classification"] = "POST_STEP_CONFIRMED"
                        unresolved.append(joint)
                    break
            if outlier["classification"] == "UNRESOLVED":
                unresolved.append(joint)
        metrics["supported_service_rate_outlier_count"] = supported_count
        if supported_count > max_supported_per_joint:
            unresolved.append(joint)
    return sorted(set(unresolved))


def evaluate_stability(samples, config, command):
    """Evaluate a post-command stability window without hiding invalid data."""
    settings = config["stability_test"]
    if not samples:
        return {"success": False, "failure_reasons": ["no samples"]}
    required = list(config["joint_names"]) + list(config["mimic_joints"])
    limits = all_joint_limits(config)
    tail_start = samples[-1]["elapsed_s"] - float(settings["tail_s"])
    tail = [sample for sample in samples if sample["elapsed_s"] >= tail_start]
    failures = []
    joint_metrics = {}
    margin = float(settings["position_limit_margin_rad"])
    p95_velocity_allowed = float(settings["max_settled_velocity_p95_rad_s"])
    for name in required:
        positions = []
        velocities = []
        for sample in samples:
            state = sample["joints"].get(name)
            if state is None:
                failures.append("{} missing".format(name))
                continue
            position = state["position"]
            velocity = state["velocity"]
            if not math.isfinite(position) or not math.isfinite(velocity):
                failures.append("{} non-finite".format(name))
                continue
            positions.append(position)
            velocities.append(velocity)
            lower, upper = limits[name]
            if position < lower - margin or position > upper + margin:
                failures.append("{} outside limit".format(name))
        tail_positions = [
            sample["joints"][name]["position"]
            for sample in tail
            if name in sample["joints"]
            and math.isfinite(sample["joints"][name]["position"])
        ]
        finite_tail_states = [
            (
                sample["elapsed_s"],
                sample["joints"][name]["position"],
                sample["joints"][name]["velocity"],
            )
            for sample in tail
            if name in sample["joints"]
            and math.isfinite(sample["joints"][name]["position"])
            and math.isfinite(sample["joints"][name]["velocity"])
        ]
        tail_velocities = [abs(state[2]) for state in finite_tail_states]
        service_rate_outliers = []
        for sample in tail:
            state = sample["joints"].get(name)
            if state is None or not math.isfinite(state.get("velocity", math.nan)):
                continue
            if abs(state["velocity"]) > p95_velocity_allowed:
                service_rate_outliers.append(
                    {
                        "elapsed_s": sample["elapsed_s"],
                        "position_rad": state["position"],
                        "velocity_rad_s": state["velocity"],
                        "sim_time_before_s": state.get("sim_time_before_s"),
                        "sim_time_after_s": state.get("sim_time_after_s"),
                    }
                )
        if not positions or not tail_positions or not tail_velocities:
            failures.append("{} insufficient finite samples".format(name))
            continue
        worst_velocity_state = max(
            finite_tail_states, key=lambda state: abs(state[2])
        )
        joint_metrics[name] = {
            "position_min_rad": min(positions),
            "position_max_rad": max(positions),
            "tail_position_range_rad": max(tail_positions) - min(tail_positions),
            "tail_velocity_abs_p95_rad_s": percentile(tail_velocities, 95.0),
            "tail_velocity_abs_max_rad_s": max(tail_velocities),
            "tail_velocity_threshold_exceedance_count": sum(
                value > p95_velocity_allowed for value in tail_velocities
            ),
            "service_rate_outliers": service_rate_outliers,
            "tail_velocity_abs_max_elapsed_s": worst_velocity_state[0],
            "tail_velocity_at_abs_max_rad_s": worst_velocity_state[2],
            "tail_position_at_abs_max_velocity_rad": worst_velocity_state[1],
        }
        if percentile(tail_velocities, 95.0) > p95_velocity_allowed:
            failures.append("{} settled velocity p95".format(name))

    target_key = "CLOSE" if command == "GRASP" else "OPEN"
    target = dict(
        zip(config["joint_names"], config["commands"][target_key]["positions"])
    )
    active_tolerance = float(settings["active_target_tolerance_rad"])
    configuration_tolerance = float(
        settings["configuration_hold_tolerance_rad"]
    )
    configuration = set(config["execution"]["configuration_joint_names"])
    for name in config["joint_names"]:
        allowed = configuration_tolerance if name in configuration else active_tolerance
        maximum_error = max(
            abs(sample["joints"][name]["position"] - target[name])
            for sample in tail
        )
        joint_metrics[name]["tail_target_error_max_rad"] = maximum_error
        if maximum_error > allowed:
            failures.append("{} target error".format(name))

    relation_tolerance = float(settings["mimic_relation_tolerance_rad"])
    range_tolerance = float(settings["passive_tail_range_rad"])
    mimic_metrics = {}
    for mimic, relation in config["mimic_joints"].items():
        errors = []
        for sample in tail:
            expected = (
                sample["joints"][relation["source"]]["position"]
                * float(relation["multiplier"])
                + float(relation["offset"])
            )
            errors.append(abs(sample["joints"][mimic]["position"] - expected))
        mimic_metrics[mimic] = {
            "tail_relation_error_p95_rad": percentile(errors, 95.0),
            "tail_relation_error_max_rad": max(errors),
            "tail_position_range_rad": joint_metrics[mimic][
                "tail_position_range_rad"
            ],
        }
        if max(errors) > relation_tolerance:
            failures.append("{} relation error".format(mimic))
        if joint_metrics[mimic]["tail_position_range_rad"] > range_tolerance:
            failures.append("{} position range".format(mimic))

    return {
        "success": not failures,
        "failure_reasons": sorted(set(failures)),
        "sample_count": len(samples),
        "observed_duration_s": samples[-1]["elapsed_s"],
        "joint_metrics": joint_metrics,
        "mimic_metrics": mimic_metrics,
    }


def sample_joint(get_joint, name):
    sim_time_before_s = rospy.Time.now().to_sec()
    response = get_joint("robot::{}".format(name))
    sim_time_after_s = rospy.Time.now().to_sec()
    if not response.success or not response.position or not response.rate:
        raise RuntimeError(
            "Gazebo joint query failed for {}: {}".format(
                name, response.status_message
            )
        )
    return {
        "position": response.position[0],
        "velocity": response.rate[0],
        "sim_time_before_s": sim_time_before_s,
        "sim_time_after_s": sim_time_after_s,
    }


def observe(get_joint, names, duration_s, sample_period_s):
    samples = []
    started = time.monotonic()
    next_sample = started
    while not rospy.is_shutdown():
        now = time.monotonic()
        if samples and now - started >= duration_s:
            break
        if now < next_sample:
            time.sleep(next_sample - now)
        snapshot = {name: sample_joint(get_joint, name) for name in names}
        samples.append(
            {"elapsed_s": time.monotonic() - started, "joints": snapshot}
        )
        next_sample += sample_period_s
    return samples, started, time.monotonic()


def main():
    rospy.init_node("run_hand_stability_tests")
    ready = rospy.wait_for_message(
        "/handarm_sim_demo/startup_ready", Bool, timeout=90.0
    )
    if not ready.data:
        raise SystemExit(8)
    config_path = rospy.get_param("~hand_config")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    cycles = int(rospy.get_param("~cycles", 3))
    settings = config["stability_test"]
    mimic_diagnostics = MimicDiagnosticMonitor()
    commander = HandCommander(config_path)
    rospy.wait_for_service("/gazebo/get_joint_properties", timeout=15.0)
    get_joint = rospy.ServiceProxy(
        "/gazebo/get_joint_properties", GetJointProperties
    )
    names = list(config["joint_names"]) + list(config["mimic_joints"])
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    rows = []
    for cycle in range(1, cycles + 1):
        for step, command in enumerate(SEQUENCE, 1):
            rospy.loginfo(
                "[hand-stability] cycle %d/%d step %d/%d: %s",
                cycle,
                cycles,
                step,
                len(SEQUENCE),
                command,
            )
            step_started = time.monotonic()
            command_result = commander.command(command)
            if command_result["success"]:
                samples, observation_started, observation_ended = observe(
                    get_joint,
                    names,
                    float(settings["observation_s"]),
                    float(settings["sample_period_s"]),
                )
                stability = evaluate_stability(samples, config, command)
                tail_boundary = observation_started + max(
                    0.0,
                    samples[-1]["elapsed_s"] - float(settings["tail_s"]),
                )
                # Allow the final 0.1 s heartbeat window to close and reach the
                # ROS subscriber. The hand remains under the same hold target.
                time.sleep(0.20)
                diagnostic_collection_ended = time.monotonic()
                diagnostic_events = mimic_diagnostics.velocity_events_between(
                    tail_boundary, diagnostic_collection_ended
                )
                stability["diagnostic_events_step"] = (
                    mimic_diagnostics.velocity_events_between(
                        step_started, diagnostic_collection_ended
                    )
                )
                stability["diagnostic_events"] = diagnostic_events
                heartbeats = mimic_diagnostics.heartbeats_between(
                    tail_boundary + 0.15, diagnostic_collection_ended
                )
                stability["diagnostic_heartbeats"] = heartbeats
                heartbeat_result = evaluate_heartbeat_coverage(
                    heartbeats,
                    config["mimic_joints"],
                    float(settings["post_step_velocity_threshold_rad_s"]),
                )
                stability["diagnostic_heartbeat_result"] = heartbeat_result
                unresolved_outliers = classify_service_rate_outliers(
                    stability["joint_metrics"],
                    heartbeats,
                    float(settings["post_step_velocity_threshold_rad_s"]),
                    int(settings["max_supported_service_outliers_per_joint"]),
                )
                stability["unresolved_service_rate_outlier_joints"] = (
                    unresolved_outliers
                )
                if diagnostic_events:
                    stability["success"] = False
                    stability["failure_reasons"] = sorted(
                        set(stability["failure_reasons"])
                        | {"mimic plugin velocity diagnostic"}
                    )
                if not heartbeat_result["success"]:
                    stability["success"] = False
                    stability["failure_reasons"] = sorted(
                        set(stability["failure_reasons"])
                        | set(heartbeat_result["failure_reasons"])
                    )
                if unresolved_outliers:
                    stability["success"] = False
                    stability["failure_reasons"] = sorted(
                        set(stability["failure_reasons"])
                        | {"unresolved asynchronous service rate outlier"}
                    )
            else:
                stability = {
                    "success": False,
                    "failure_reasons": ["hand command failed"],
                }
            row = {
                "run_id": run_id,
                "cycle": cycle,
                "step": step,
                "command": command,
                "command_result": command_result,
                "stability": stability,
                "success": bool(
                    command_result["success"] and stability["success"]
                ),
            }
            rows.append(row)
            rospy.loginfo(
                "[hand-stability] %s result=%s reasons=%s",
                command,
                "PASS" if row["success"] else "FAIL",
                stability.get("failure_reasons", []),
            )
            if not row["success"]:
                break
        if not rows[-1]["success"]:
            break

    results_dir = os.path.abspath(
        os.path.join(PACKAGE_PATH, "..", "..", "results", "sim_baseline")
    )
    os.makedirs(results_dir, exist_ok=True)
    payload = {
        "run_id": run_id,
        "command_scope": "GRASP_RELEASE_ONLY",
        "test_scope": "NO_LOAD_HAND_STABILITY_ONLY",
        "cycles_requested": cycles,
        "settings": settings,
        "rows": rows,
        "all_success": len(rows) == cycles * len(SEQUENCE)
        and all(row["success"] for row in rows),
    }
    json_path = os.path.join(
        results_dir, "hand_stability_{}.json".format(run_id)
    )
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    csv_path = os.path.join(results_dir, "hand_stability.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "run_id",
                "cycle",
                "step",
                "command",
                "success",
                "failure_reasons",
                "sample_count",
                "observed_duration_s",
                "joint_metrics",
                "mimic_metrics",
            ),
        )
        writer.writeheader()
        for row in rows:
            stability = row["stability"]
            writer.writerow(
                {
                    "run_id": run_id,
                    "cycle": row["cycle"],
                    "step": row["step"],
                    "command": row["command"],
                    "success": row["success"],
                    "failure_reasons": json.dumps(
                        stability.get("failure_reasons", [])
                    ),
                    "sample_count": stability.get("sample_count"),
                    "observed_duration_s": stability.get("observed_duration_s"),
                    "joint_metrics": json.dumps(
                        stability.get("joint_metrics", {}), sort_keys=True
                    ),
                    "mimic_metrics": json.dumps(
                        stability.get("mimic_metrics", {}), sort_keys=True
                    ),
                }
            )
    rospy.loginfo(
        "[hand-stability] DONE all_success=%s JSON=%s CSV=%s",
        payload["all_success"],
        json_path,
        csv_path,
    )
    raise SystemExit(0 if payload["all_success"] else 8)


if __name__ == "__main__":
    main()
