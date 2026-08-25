#!/usr/bin/env bash

# Shared path/bootstrap helpers for scripts in this repository. This file is
# sourced, not executed directly.
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source_ros() {
  local ros_setup_file="${ROS_SETUP_FILE:-/opt/ros/noetic/setup.bash}"
  if [[ ! -f "$ros_setup_file" ]]; then
    echo "ROS setup file not found: $ros_setup_file" >&2
    echo "Set ROS_SETUP_FILE to your ROS Noetic setup.bash." >&2
    return 2
  fi
  # shellcheck disable=SC1090
  source "$ros_setup_file"
}

source_workspace() {
  local workspace_setup="$PROJECT_ROOT/devel/setup.bash"
  if [[ ! -f "$workspace_setup" ]]; then
    echo "Workspace is not built; run ./scripts/run_stage1_tests.sh first." >&2
    return 2
  fi
  # shellcheck disable=SC1090
  source "$workspace_setup"
}
