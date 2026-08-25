#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/workspace.sh"
cd "$PROJECT_ROOT"
source_ros
source_workspace

# shared_teleop_safe_demo.launch hard-wires both physical-output gates off.
exec roslaunch handarm_moveit_demo shared_teleop_safe_demo.launch "$@"
