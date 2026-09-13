#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /opt/ros/noetic/setup.bash
source devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11421
export GAZEBO_MASTER_URI=http://127.0.0.1:11465
export ROS_LOG_DIR="$PWD/.runtime/known_intent_replay_20260913/ros_logs"
experiment_directory="$PWD/.runtime/known_intent_replay_20260913"
mkdir -p "$experiment_directory"
launch_pid=""
cleanup() {
  if [[ -n "$launch_pid" ]]; then
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    wait "$launch_pid" 2>/dev/null || true
    launch_pid=""
  fi
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM
for variant in baseline oracle_continuous; do
  setsid taskset -c 0-7 roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch \
    gazebo_gui:=false with_ground_object:=false empty_motion_scene:=true \
    grasp_alignment_views:=false enable_logger:=false input_source:=external \
    constrained_velocity_projection:=true position_accommodation_m:=0.0 \
    mapping_profile:=current_linear hamer_input_timeout_s:=0.40 \
    hamer_maximum_pipeline_latency_s:=0.25 \
    >"$experiment_directory/${variant}_launch.log" 2>&1 &
  launch_pid=$!
  taskset -c 0-7 /usr/bin/python3 scripts/replay_known_intent.py "$variant" \
    >"$experiment_directory/${variant}_replay.log" 2>&1
  cleanup
done
