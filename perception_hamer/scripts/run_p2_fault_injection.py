#!/usr/bin/env python3
"""Run only hardware-free P2 fault-injection tests and optionally save JSON."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-report",
        help="atomically write a machine-readable test result to this path",
    )
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(project_dir / "tests"),
        "-p",
        "test_p2_fault_injection.py",
        "-v",
    ]
    started_ns = time.time_ns()
    completed = subprocess.run(
        command,
        cwd=str(project_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    match = re.search(r"Ran (\d+) tests", completed.stdout + completed.stderr)
    if args.json_report:
        report_path = Path(args.json_report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "suite": "P2_SAFE_FAULT_INJECTION",
            "scope": "MOCK_ONLY_NO_CAMERA_NO_GAZEBO",
            "started_host_wall_time_ns": started_ns,
            "completed_host_wall_time_ns": time.time_ns(),
            "python": sys.version.split()[0],
            "test_count": int(match.group(1)) if match else None,
            "return_code": completed.returncode,
            "result": "PASS" if completed.returncode == 0 else "FAIL",
            "safety_constraints": {
                "physical_camera_unplug": "NOT_RUN",
                "real_filesystem_fill": "NOT_RUN",
                "real_enospc": "NOT_RUN",
                "camera_opened": False,
                "gazebo_started": False,
            },
        }
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(str(temporary), str(report_path))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
