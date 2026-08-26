#!/usr/bin/env bash
set -Eeuo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_directory/.." && pwd)"
cd "$workspace"

source /opt/ros/noetic/setup.bash
source devel/setup.bash

conda_executable="/home/dongtian/anaconda3/bin/conda"
asset_root="/home/dongtian/myhandarmtest1-v1.0.1-ubuntu2004/perception_hamer/_DATA"
checkpoint="$asset_root/hamer_ckpts/checkpoints/hamer.ckpt"
runtime_directory="$workspace/.runtime/live_human_validation_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$runtime_directory"

for required in "$conda_executable" "$checkpoint" "$asset_root/data/mano/MANO_RIGHT.pkl"; do
  if [[ ! -e "$required" ]]; then
    echo "缺少运行文件：$required" >&2
    exit 2
  fi
done
if pgrep -f 'run_d455_hamer_crop.py|live_human_ground_gazebo_egm_teleop.launch' \
    >/dev/null 2>&1; then
  echo "已有相机或 Gazebo 遥操作进程。请先在旧窗口按 Q，并结束旧 roslaunch。" >&2
  exit 3
fi

ros_pid=""
link_monitor_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for process_id in "$link_monitor_pid"; do
    if [[ -n "$process_id" ]]; then
      kill "$process_id" 2>/dev/null || true
      wait "$process_id" 2>/dev/null || true
    fi
  done
  if [[ -n "$ros_pid" ]] && kill -0 "$ros_pid" 2>/dev/null; then
    kill -INT -- "-$ros_pid" 2>/dev/null || true
    for _ in $(seq 1 200); do
      kill -0 "$ros_pid" 2>/dev/null || break
      sleep 0.10
    done
    kill -TERM -- "-$ros_pid" 2>/dev/null || true
    wait "$ros_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "正在启动 Gazebo 和 ROS 控制链……"
setsid roslaunch handarm_moveit_demo \
  live_human_ground_gazebo_egm_teleop.launch \
  gazebo_gui:=true with_ground_object:=true enable_logger:=false \
  input_source:=udp \
  mapping_profile:=current_linear \
  hamer_input_timeout_s:=2.00 \
  hamer_maximum_pipeline_latency_s:=0.25 \
  >"$runtime_directory/roslaunch.log" 2>&1 &
ros_pid=$!

ros_ready=false
ros_ready_deadline=$((SECONDS + 150))
while (( SECONDS < ros_ready_deadline )); do
  # An XML-RPC request issued before roscore was responsive used to block for
  # roughly two minutes.  Bound every probe so camera startup follows actual
  # ROS readiness instead of a socket-library timeout.
  if timeout 1 rosnode info /gazebo >/dev/null 2>&1 \
      && timeout 1 rosnode info /hamer_input_adapter >/dev/null 2>&1 \
      && timeout 1 rosnode info /three_finger_retargeting >/dev/null 2>&1 \
      && timeout 1 rosnode info /moveit_servo_output_adapter >/dev/null 2>&1; then
    ros_ready=true
    break
  fi
  if ! kill -0 "$ros_pid" 2>/dev/null; then
    echo "Gazebo/ROS 启动失败：$runtime_directory/roslaunch.log" >&2
    exit 4
  fi
  sleep 0.25
done
if [[ "$ros_ready" != true ]]; then
  echo "Gazebo/ROS 在 150 秒内未就绪：$runtime_directory/roslaunch.log" >&2
  exit 4
fi

scene_ready=false
for _ in $(seq 1 240); do
  # Under Gazebo load the Python rostopic client can need more than one second
  # merely to complete its ROS handshake.  Capture the one-shot latched value
  # first, then inspect it locally; a short grep pipeline plus pipefail could
  # otherwise hide a real True value behind timeout/SIGPIPE status.
  scene_ready_message="$(
    timeout 5 rostopic echo -n 1 /handarm_sim_demo/scene_ready 2>/dev/null
  )" || scene_ready_message=""
  if grep -q 'data: True' <<<"$scene_ready_message"; then
    scene_ready=true
    break
  fi
  sleep 0.25
done
if [[ "$scene_ready" != true ]]; then
  echo "MoveIt/Gazebo 场景安全状态未就绪：$runtime_directory/roslaunch.log" >&2
  exit 5
fi

echo "Gazebo、控制器、场景闸门均已就绪。正在打开相机窗口……"
echo "本次运行日志：$runtime_directory"

# This observer subscribes before C and never gates the control path.  It
# distinguishes camera acceptance, a non-zero Servo command, and real Gazebo
# tool0 displacement so a broken link cannot masquerade as a working one.
rosrun handarm_moveit_demo live_teleop_terminal_monitor.py &
link_monitor_pid=$!

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
camera_affinity_arguments=()
if command -v taskset >/dev/null 2>&1 && (( $(nproc) >= 12 )); then
  # i5-12400F: HaMeR gets the final two physical cores while the GUI and
  # MediaPipe use the preceding two.  ROS/Gazebo remain scheduler-managed.
  camera_affinity_arguments+=(
    --inference-cpu-affinity 8-11
    --sidecar-cpu-affinity 4-7
  )
fi
"$conda_executable" run --no-capture-output -n hamer_rtx2060 \
  python perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe \
  --mediapipe-min-detection-confidence 0.50 \
  --mesh-renderer teleoperation-core \
  --no-mesh-overlay \
  --display-rate-hz 15 \
  --control-reference mano-wrist-ring \
  --roi-smoothing-alpha 1.0 \
  --orientation-filter-large-angle-mode reject \
  --orientation-filter-max-gain 1.0 \
  --forearm-rate-hz 8.0 \
  --forearm-maximum-source-age-s 0.20 \
  --hand-presence-timeout-s 0.30 \
  --hand-miss-grace-frames 2 \
  --hand-miss-grace-s 0.15 \
  --teleop-maximum-pipeline-latency-s 0.25 \
  --teleop-udp-host 127.0.0.1 \
  --teleop-udp-port 5010 \
  --duration-s 3600 \
  --checkpoint "$checkpoint" \
  --data-root "$asset_root" \
  "${camera_affinity_arguments[@]}" \
  2>&1 | tee "$runtime_directory/camera.log"
