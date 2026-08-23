#!/usr/bin/env bash
set -euo pipefail

workspace="/home/diu/myhandarmtest1"
cd "$workspace"
source /opt/ros/noetic/setup.bash
if [[ ! -f devel/setup.bash ]]; then
  echo "Missing devel/setup.bash; run ./scripts/run_stage1_tests.sh first." >&2
  exit 2
fi
source devel/setup.bash

# shared_teleop_safe_demo.launch hard-wires both physical-output gates off.
exec roslaunch handarm_moveit_demo shared_teleop_safe_demo.launch "$@"
