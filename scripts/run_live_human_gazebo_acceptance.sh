#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/workspace.sh"
cd "$PROJECT_ROOT"
source_ros
source_workspace

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
