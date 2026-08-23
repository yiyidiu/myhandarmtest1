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

result_directory="$(mktemp -d /tmp/handarm_live_human_acceptance.XXXXXX)"
result_json="$result_directory/live_human_gazebo_follow.json"

echo "Press C once in the camera window before running this validator."
echo "The existing camera C reference will be observed and will not be changed."
echo "Hold one complete hand in the D455 view and keep a neutral pose."
echo "After the countdown, translate at least 3 cm and rotate at least 15 degrees."
echo "The validator observes the existing live UDP/ROS/Gazebo chain; it publishes no pose, reference reset, or Servo command."

set +e
rosrun handarm_moveit_demo live_gazebo_follow_validator.py \
  --output "$result_json" "$@"
validator_status=$?
set -e

echo "acceptance_json=$result_json"
exit "$validator_status"
