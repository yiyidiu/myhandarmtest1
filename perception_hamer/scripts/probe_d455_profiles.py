#!/usr/bin/env python3
"""Persist D455 profile enumeration and deterministic candidate ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.realsense_capability import (  # noqa: E402
    CapabilityError,
    enumerate_device_profiles,
    rank_rgbd_candidates,
)


def command_evidence(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return {"command": command, "status": "NOT_AVAILABLE", "output": ""}
    except subprocess.TimeoutExpired:
        return {"command": command, "status": "TIMEOUT", "output": ""}
    return {
        "command": command,
        "status": "PASS" if result.returncode == 0 else "FAILED",
        "returncode": result.returncode,
        "output": result.stdout,
        "stderr": result.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        capability = enumerate_device_profiles(args.serial)
    except CapabilityError as exc:
        print(f"profile probe failed: {exc}", file=sys.stderr)
        return 2
    live = rank_rgbd_candidates(capability, "live_algorithm")
    recording = rank_rgbd_candidates(capability, "recording")
    result = {
        "schema_version": 1,
        "captured_host_wall_time_ns": time.time_ns(),
        "evidence_scope": "USB2_DEVELOPMENT_ONLY",
        "formal_result": False,
        "capability": capability,
        "commands": {
            "rs_enumerate_devices": command_evidence(
                ["rs-enumerate-devices", "-o", "-c"]
            ),
            "lsusb": command_evidence(["lsusb"]),
            "lsusb_tree": command_evidence(["lsusb", "-t"]),
        },
        "candidate_count": {"live_algorithm": len(live), "recording": len(recording)},
        "live_algorithm_candidates_ranked": live,
        "recording_candidates_ranked": recording,
        "usb2_live_algorithm_profile": live[0] if live else None,
        "usb2_recording_profile": recording[0] if recording else None,
        "profile_start_policy": "exact selected profile or fail; no silent fallback",
        "candidate_short_tests": "NOT_RUN_BY_ENUMERATION_ONLY",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
