#!/usr/bin/env bash
# Three recorded operator trials, fresh simulator per solver, no environment truth.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set +u
source /opt/ros/noetic/setup.bash
source devel/setup.bash
set -u
experiment="$PWD/.runtime/servo_qp_comparison_20260914/capture_clock_exact"
mkdir -p "$experiment"
export ROS_MASTER_URI=http://127.0.0.1:11414
export GAZEBO_MASTER_URI=http://127.0.0.1:11415
export TELEOP_CONTROL_LAUNCH_PREFIX='taskset -c 0-1'
pid=""
cleanup() {
  if [[ -n "$pid" ]]; then
    kill -INT -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    pid=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
cases=(teleop_feedback_20260908T182545_bZ7k hand_motion_observation_20260912T145617_iJYZ teleop_feedback_20260913T205659_IkL0)
ends=(330 321 233)
indices=(0 1 2)
if (( $# > 0 )); then
  case "$1" in 0|1|2) indices=("$1") ;; *) echo "case must be 0, 1 or 2" >&2; exit 2 ;; esac
fi
if (( $# > 2 )); then echo "usage: $0 [case] [qp|servo]" >&2; exit 2; fi
for index in "${indices[@]}"; do
  order=(servo qp)
  if (( index == 1 )); then order=(qp servo); fi
  if (( $# == 2 )); then
    case "$2" in qp|servo) order=("$2") ;; *) echo "solver must be qp or servo" >&2; exit 2 ;; esac
  fi
  for solver in "${order[@]}"; do
    label="${cases[$index]}_${solver}"
    if [[ -e "$experiment/$label.json.gz" ]]; then
      echo "Refusing to overwrite existing comparison: $label" >&2
      exit 2
    fi
    echo "RUN $label"
    setsid taskset -c 2-7 roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch \
      gazebo_gui:=false with_ground_object:=false empty_motion_scene:=true \
      use_simulated_environment_for_control:=false grasp_alignment_views:=false \
      input_source:=external enable_logger:=false constrained_velocity_projection:=true \
      joint_velocity_solver:="$solver" position_accommodation_m:=0.0 \
      mapping_profile:=current_linear hamer_input_timeout_s:=0.40 \
      hamer_maximum_pipeline_latency_s:=0.25 \
      > "$experiment/${label}_launch.log" 2>&1 &
    pid=$!
    taskset -c 4-7 /usr/bin/python3 scripts/compare_servo_qp_replay.py \
      ".runtime/${cases[$index]}" "$experiment/$label.json.gz" --end "${ends[$index]}" \
      2>&1 | tee "$experiment/${label}_replay.log"
    cleanup
  done
done
