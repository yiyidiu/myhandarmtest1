#!/usr/bin/env bash
set -eo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_directory/.." && pwd)"
output_directory="$(mktemp -d /tmp/handarm_stage1_demo.XXXXXX)"
cd "$workspace"
python3 src/handarm_moveit_demo/scripts/offline_shared_teleop_demo.py \
  --duration-s 6.0 \
  --output "$output_directory/offline_shared_teleop.csv"
