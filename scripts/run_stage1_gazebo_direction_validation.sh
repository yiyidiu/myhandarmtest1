#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_directory/.." && pwd)"
cd "$workspace"
source /opt/ros/noetic/setup.bash
source devel/setup.bash

run_directory="$(mktemp -d /tmp/handarm_gazebo_direction.XXXXXX)"
launch_log="$run_directory/gazebo.log"
result_json="$run_directory/direction_validation.json"

roslaunch handarm_moveit_demo shared_teleop_safe_demo.launch \
  gazebo_gui:=false input_source:=direction_test enable_logger:=false \
  enable_gesture_demo:=false response_first:=true >"$launch_log" 2>&1 &
launch_pid=$!

cleanup() {
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT "$launch_pid" 2>/dev/null || true
    wait "$launch_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=false
for _ in $(seq 1 120); do
  if rosservice info /shared_teleop/start_direction_test >/dev/null 2>&1 && \
     rosservice info /check_state_validity >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    echo "Gazebo launch exited before services became ready; see $launch_log" >&2
    exit 3
  fi
  sleep 0.25
done

if [[ "$ready" != true ]]; then
  echo "Gazebo services were not ready within 30 seconds; see $launch_log" >&2
  exit 4
fi

rosrun handarm_moveit_demo gazebo_direction_validator.py --output "$result_json"
echo "gazebo_log=$launch_log"
echo "result_json=$result_json"
