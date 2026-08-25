#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/workspace.sh"
output_directory="$(mktemp -d /tmp/handarm_stage1_demo.XXXXXX)"
cd "$PROJECT_ROOT"
python3 src/handarm_moveit_demo/scripts/offline_shared_teleop_demo.py \
  --duration-s 6.0 \
  --output "$output_directory/offline_shared_teleop.csv"
