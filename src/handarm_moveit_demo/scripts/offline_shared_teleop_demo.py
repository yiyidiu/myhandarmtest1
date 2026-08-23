#!/usr/bin/env python3
"""Deterministic hardware-free 30 Hz input / 50 Hz control-loop demonstration."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from handarm_moveit_demo.shared_teleop_core import (  # noqa: E402
    CoordinateVelocityMapper, LatestCommandShaper, PoseSample,
    SixDofTrendEstimator, matrix_to_quaternion_xyzw, so3_exp,
)


FIELDS = [
    "timestamp_ros", "raw_hand_stamp", "raw_hand_source_stamp", "raw_hand_frame", "raw_hand_valid",
    "raw_hand_x", "raw_hand_y", "raw_hand_z", "raw_hand_qx", "raw_hand_qy",
    "raw_hand_qz", "raw_hand_qw", "relative_hand_x", "relative_hand_y",
    "relative_hand_z", "raw_vx", "raw_vy", "raw_vz", "raw_wx", "raw_wy",
    "raw_wz", "processed_vx", "processed_vy", "processed_vz", "processed_wx",
    "processed_wy", "processed_wz", "confidence_x", "confidence_y",
    "confidence_z", "confidence_roll", "confidence_pitch", "confidence_yaw",
    "assist_strength", "assist_candidates_json", "selected_correction",
    "selected_correction_quaternion_xyzw", "actual_ee_pose", "input_output_latency_s",
    "control_loop_hz", "processing_ms", "gesture", "gesture_confidence",
    "timeout_reason", "limit_reasons", "jump_reason", "invalid_reason",
    "safety_reasons",
]


def run(duration_s, output):
    estimator = SixDofTrendEstimator(
        window_size=4, translation_deadband_m=[0.0005]*3,
        rotation_deadband_rad=[0.005]*3, smoothing_alpha=0.45)
    mapper = CoordinateVelocityMapper(
        [[1,0,0],[0,0,1],[0,-1,0]], [[1,0,0],[0,0,1],[0,-1,0]],
        [1.2, 1.2, 1.0], [0.9, 0.9, 0.9], [0.10]*3, [0.60]*3)
    shaper = LatestCommandShaper([0.10]*3+[0.60]*3,
                                 [2.0]*3+[12.0]*3, 0.09, 0.15)
    input_period = 1.0/30.0; output_period = 1.0/50.0
    next_input = 0.0; last_pose = None; last_trend = None; mapped = np.zeros(6)
    processing = []; ages = []; rows = []; six_axes_seen = False
    control_position = np.array([0.45, 0.0, 0.45]); control_rotation = np.eye(3)
    for tick in range(int(math.ceil(duration_s/output_period))+1):
        now = tick*output_period
        began = time.perf_counter()
        if now+1.0e-12 >= next_input and next_input <= duration_s-0.30:
            t = next_input
            position = np.array([0.04*math.sin(1.1*t), 0.03*math.sin(0.8*t+0.3),
                                 0.025*math.sin(0.6*t+0.7)])
            rotation_vector = np.array([0.25*math.sin(0.9*t),
                                        0.20*math.sin(0.7*t+0.4),
                                        0.30*math.sin(0.5*t+0.8)])
            last_pose = PoseSample(t, position, so3_exp(rotation_vector),
                                   np.array([0.95, 0.85, 0.90, 0.80, 0.92, 0.88]))
            last_trend = estimator.update(last_pose)
            mapped = mapper.map(last_trend.raw_velocity, last_trend.confidence)
            shaper.update(mapped, t, last_trend.valid)
            next_input += input_period
        shaped = shaper.tick(now)
        dt = output_period if tick else 0.0
        control_position += shaped.velocity[:3]*dt
        control_rotation = so3_exp(shaped.velocity[3:]*dt) @ control_rotation
        elapsed_ms = (time.perf_counter()-began)*1000.0
        processing.append(elapsed_ms); ages.append(shaped.input_age_s)
        if last_trend is not None:
            six_axes_seen = six_axes_seen or bool(np.all(np.abs(last_trend.raw_velocity) > 1.0e-5))
            q = matrix_to_quaternion_xyzw(last_pose.rotation)
            actual_q = matrix_to_quaternion_xyzw(control_rotation)
            row = dict.fromkeys(FIELDS, "")
            row.update({
                "timestamp_ros": now, "raw_hand_stamp": last_pose.timestamp,
                "raw_hand_source_stamp": last_pose.timestamp,
                "raw_hand_frame": "camera_color_optical_frame", "raw_hand_valid": last_pose.valid,
                "raw_hand_x": last_pose.position[0], "raw_hand_y": last_pose.position[1],
                "raw_hand_z": last_pose.position[2], "raw_hand_qx": q[0], "raw_hand_qy": q[1],
                "raw_hand_qz": q[2], "raw_hand_qw": q[3],
                "relative_hand_x": last_trend.relative_position[0],
                "relative_hand_y": last_trend.relative_position[1],
                "relative_hand_z": last_trend.relative_position[2],
                **{name: value for name, value in zip(
                    ["raw_vx","raw_vy","raw_vz","raw_wx","raw_wy","raw_wz"],
                    last_trend.raw_velocity)},
                **{name: value for name, value in zip(
                    ["processed_vx","processed_vy","processed_vz","processed_wx","processed_wy","processed_wz"],
                    shaped.velocity)},
                **{name: value for name, value in zip(
                    ["confidence_x","confidence_y","confidence_z","confidence_roll","confidence_pitch","confidence_yaw"],
                    last_trend.confidence)},
                "assist_strength": 0.0, "assist_candidates_json": "[]",
                "selected_correction": "none", "selected_correction_quaternion_xyzw": "",
                "actual_ee_pose": json.dumps((control_position.tolist()+actual_q.tolist()), separators=(",", ":")),
                "input_output_latency_s": shaped.input_age_s, "control_loop_hz": 50.0,
                "processing_ms": elapsed_ms, "gesture": 0, "gesture_confidence": 0.0,
                "timeout_reason": shaped.reason, "limit_reasons": "[]",
                "jump_reason": last_trend.reason if "REJECTED" in last_trend.reason else "",
                "invalid_reason": "" if last_trend.valid else last_trend.reason,
                "safety_reasons": json.dumps([shaped.reason], separators=(",", ":")),
            })
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    finite_ages = [value for value in ages if math.isfinite(value)]
    summary = {
        "status": "OFFLINE_SIMULATION_COMPLETE_NO_ROBOT_OUTPUT",
        "output_csv": str(output), "duration_s": duration_s,
        "camera_input_nominal_hz": 30.0, "control_loop_actual_hz": 50.0,
        "control_ticks": len(processing), "six_axes_simultaneous_seen": six_axes_seen,
        "processing_ms": {"mean": float(np.mean(processing)),
                          "p95": float(np.percentile(processing, 95)),
                          "maximum": float(np.max(processing))},
        "input_age_s": {"mean": float(np.mean(finite_ages)),
                        "maximum": float(np.max(finite_ages))},
        "timeout_zero_verified": bool(np.allclose(shaped.velocity, 0.0)),
        "real_robot_commands_sent": 0,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/shared_teleop_offline_demo.csv"))
    args = parser.parse_args()
    print(json.dumps(run(args.duration_s, args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
