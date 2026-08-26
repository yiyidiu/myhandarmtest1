#!/usr/bin/env python3
"""Replay recorded open-hand HaMeR joints through the finger controller."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from perception_hamer.src.finger_observation import observe_mano_fingers  # noqa: E402
from handarm_moveit_demo.finger_retargeting import ThreeFingerRetargeter  # noqa: E402


def build_controller(shared, hand):
    names = hand["joint_names"]
    limits = hand["joint_limits"]
    return ThreeFingerRetargeter(
        hand["commands"]["OPEN"]["positions"],
        hand["commands"]["CLOSE"]["positions"],
        [limits[name][0] for name in names],
        [limits[name][1] for name in names],
        shared["source_mixing_matrix"],
        shared,
    )


def replay(session, shared, hand):
    frames_file = session / "frames.jsonl"
    records = [
        json.loads(line)
        for line in frames_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid = [record for record in records if record.get("valid")]
    if not valid:
        raise ValueError("{} contains no valid HaMeR frames".format(session))
    controller = build_controller(shared, hand)
    current = np.asarray(hand["commands"]["OPEN"]["positions"], dtype=float)
    commands = []
    features = []
    statuses = {}
    segment = 0
    segment_lengths = [0]
    previous_timestamp = None
    for record in valid:
        timestamp = float(record["timestamp"])
        if (
            previous_timestamp is not None
            and timestamp - previous_timestamp > float(shared["input_timeout_s"])
        ):
            controller.block_active_reference()
            segment += 1
            segment_lengths.append(0)
        previous_timestamp = timestamp
        segment_lengths[-1] += 1
        observation = observe_mano_fingers(
            record["mano_joints"],
            record.get("roi", {}).get("confidence", 0.0),
            record.get("hamer_quality", {}).get("bbox_visible_fraction", 0.0),
            record.get("crop_quality", 0.0),
        )
        features.append(observation.flexion)
        result = controller.update(
            timestamp,
            "{}:{}".format(session.name, segment),
            observation.flexion,
            observation.confidence,
            current,
        )
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if result.command_target is not None:
            current = result.command_target
            commands.append(current.copy())
    command_array = np.asarray(commands)
    feature_array = np.asarray(features)
    target_range = (
        np.ptp(command_array, axis=0)
        if len(command_array)
        else np.full(4, np.inf)
    )
    return {
        "session": str(session),
        "valid_frame_count": len(valid),
        "segment_lengths": segment_lengths,
        "command_count": len(commands),
        "feature_standard_deviation": feature_array.std(axis=0).tolist(),
        "robot_target_peak_to_peak_rad": target_range.tolist(),
        "statuses": statuses,
        "passed": bool(np.max(target_range) <= 0.01),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="+", type=Path)
    parser.add_argument(
        "--shared-config",
        type=Path,
        default=WORKSPACE / "src/handarm_moveit_demo/config/shared_teleop.yaml",
    )
    parser.add_argument(
        "--hand-config",
        type=Path,
        default=WORKSPACE / "src/handarm_sim_demo/config/hand_commands.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shared = yaml.safe_load(args.shared_config.read_text(encoding="utf-8"))[
        "finger_retargeting"
    ]
    hand = yaml.safe_load(args.hand_config.read_text(encoding="utf-8"))
    results = [replay(path.expanduser().resolve(), shared, hand) for path in args.session]
    output = {"passed": all(item["passed"] for item in results), "sessions": results}
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    return 0 if output["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
