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
timestamp="$(date +%Y%m%dT%H%M%S)"
session_directory="$workspace/.runtime/v03_human_teleop_postfix_${timestamp}"
camera_numeric_root="$session_directory/hamer_numeric"
csv_directory="$session_directory/csv"

for required in \
  "$conda_executable" \
  "$checkpoint" \
  "$asset_root/data/mano/MANO_RIGHT.pkl"; do
  if [[ ! -e "$required" ]]; then
    echo "Required live-teleoperation asset is missing: $required" >&2
    exit 2
  fi
done
for command_name in roslaunch rostopic rosbag gst-launch-1.0 vmstat nvidia-smi; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required evidence command is unavailable: $command_name" >&2
    exit 2
  fi
done
if pgrep -f 'run_d455_hamer_crop.py|live_human_ground_gazebo_egm_teleop.launch' \
    >/dev/null 2>&1; then
  echo "Refusing to start over an existing live camera or Gazebo session." >&2
  exit 3
fi

mkdir -p "$camera_numeric_root" "$csv_directory"
printf '%s\n' "$session_directory" > \
  "$workspace/.runtime/latest_live_human_evidence_session.txt"

ros_pid=""
camera_pid=""
screen_pid=""
rosbag_pid=""
csv_pid=""
vmstat_pid=""
gpu_pid=""
recording_started=false
normal_completion=false

process_group_alive() {
  local process_id="$1"
  [[ -n "$process_id" ]] && kill -0 "$process_id" 2>/dev/null
}

stop_process_group() {
  local process_id="$1"
  local first_signal="$2"
  local attempts="${3:-100}"
  if ! process_group_alive "$process_id"; then
    return 0
  fi
  kill "-$first_signal" -- "-$process_id" 2>/dev/null || true
  for _ in $(seq 1 "$attempts"); do
    if ! process_group_alive "$process_id"; then
      wait "$process_id" 2>/dev/null || true
      return 0
    fi
    sleep 0.10
  done
  kill -TERM -- "-$process_id" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! process_group_alive "$process_id"; then
      wait "$process_id" 2>/dev/null || true
      return 0
    fi
    sleep 0.10
  done
  kill -KILL -- "-$process_id" 2>/dev/null || true
  wait "$process_id" 2>/dev/null || true
}

stop_recorders() {
  if [[ "$recording_started" != true ]]; then
    return 0
  fi
  recording_started=false
  stop_process_group "$screen_pid" INT 150
  stop_process_group "$rosbag_pid" INT 200
  stop_process_group "$csv_pid" INT 100
  stop_process_group "$vmstat_pid" TERM 30
  stop_process_group "$gpu_pid" TERM 30
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_recorders
  stop_process_group "$camera_pid" TERM 100
  stop_process_group "$ros_pid" INT 200
  if [[ "$normal_completion" != true ]]; then
    printf 'aborted_wall_time_ns=%s\nexit_status=%s\n' \
      "$(date +%s%N)" "$status" > "$session_directory/ABORTED.txt"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

printf 'session_directory=%s\ngit_commit=%s\nstart_wall_time_ns=%s\n' \
  "$session_directory" "$(git rev-parse HEAD)" "$(date +%s%N)" > \
  "$session_directory/session_metadata.txt"
xrandr --current > "$session_directory/xrandr.txt" 2>&1 || true

setsid roslaunch handarm_moveit_demo \
  live_human_ground_gazebo_egm_teleop.launch \
  gazebo_gui:=true with_ground_object:=true enable_logger:=false \
  input_source:=udp \
  mapping_profile:=current_linear \
  log_directory:="$csv_directory" \
  >"$session_directory/roslaunch.log" 2>&1 &
ros_pid=$!
printf 'roslaunch_pid=%s\n' "$ros_pid" >> "$session_directory/session_metadata.txt"

ros_ready=false
for _ in $(seq 1 600); do
  if rosnode info /gazebo >/dev/null 2>&1 && \
     rosnode info /gazebo_gui >/dev/null 2>&1 && \
     rosnode info /hamer_input_adapter >/dev/null 2>&1 && \
     rosnode info /three_finger_retargeting >/dev/null 2>&1; then
    ros_ready=true
    break
  fi
  if ! process_group_alive "$ros_pid"; then
    echo "Gazebo/ROS launch exited before readiness; see roslaunch.log" >&2
    exit 4
  fi
  sleep 0.25
done
if [[ "$ros_ready" != true ]]; then
  echo "Gazebo/ROS did not become ready within 150 seconds." >&2
  exit 4
fi
echo "GAZEBO_READY session=$session_directory"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid "$conda_executable" run --no-capture-output -n hamer_rtx2060 \
  python perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe \
  --mesh-renderer teleoperation-core \
  --no-mesh-overlay \
  --control-reference mano-wrist-ring \
  --roi-smoothing-alpha 1.0 \
  --orientation-filter-large-angle-mode reject \
  --orientation-filter-max-gain 1.0 \
  --forearm-rate-hz 8.0 \
  --forearm-maximum-source-age-s 0.20 \
  --hand-presence-timeout-s 0.50 \
  --hand-miss-grace-frames 8 \
  --hand-miss-grace-s 0.35 \
  --teleop-udp-host 127.0.0.1 \
  --teleop-udp-port 5010 \
  --experiment DEV_HAMER_OPEN_CLOSE \
  --experiment-jsonl-only \
  --record-after-control-enabled \
  --duration-s 3600 \
  --output-root "$camera_numeric_root" \
  --checkpoint "$checkpoint" \
  --data-root "$asset_root" \
  >"$session_directory/camera.log" 2>&1 &
camera_pid=$!
printf 'camera_pid=%s\n' "$camera_pid" >> "$session_directory/session_metadata.txt"

camera_window_started=false
for _ in $(seq 1 1200); do
  if grep -q 'using the MediaPipe display sidecar' \
      "$session_directory/camera.log" 2>/dev/null; then
    camera_window_started=true
    break
  fi
  if ! process_group_alive "$camera_pid"; then
    echo "Camera process exited before opening its window; see camera.log" >&2
    exit 5
  fi
  sleep 0.10
done
if [[ "$camera_window_started" != true ]]; then
  echo "Camera window did not appear within 120 seconds." >&2
  exit 5
fi
echo "WINDOWS_STARTED_WAITING_FOR_VALID_HAND"

hamer_session_directory=""
measurement_ready_file=""
for _ in $(seq 1 1200); do
  hamer_session_directory="$(find "$camera_numeric_root" \
    -mindepth 1 -maxdepth 1 -type d \
    -name 'DEV_HAMER_OPEN_CLOSE_*' -print -quit)"
  if [[ -n "$hamer_session_directory" ]]; then
    break
  fi
  if ! process_group_alive "$camera_pid"; then
    echo "Camera exited before creating its evidence directory." >&2
    exit 6
  fi
  sleep 0.10
done
if [[ -z "$hamer_session_directory" ]]; then
  echo "Camera evidence directory was not created within 120 seconds." >&2
  exit 6
fi
printf 'hamer_session_directory=%s\n' "$hamer_session_directory" >> \
  "$session_directory/session_metadata.txt"

# Record a lightweight full-screen pre-roll before the operator can press C.
# The persistent camera marker below defines the formal C boundary.  This
# deliberately avoids polling a transient ROS message and makes a missed C
# trigger impossible; analysis trims the pre-roll rather than inventing data.
preroll_wall_ns="$(date +%s%N)"
printf 'preroll_recording_started_wall_ns=%s\n' "$preroll_wall_ns" >> \
  "$session_directory/session_metadata.txt"

setsid gst-launch-1.0 -e \
  ximagesrc use-damage=false show-pointer=true \
  ! video/x-raw,framerate=30/1 \
  ! queue max-size-buffers=3 leaky=downstream \
  ! videoscale method=nearest-neighbour \
  ! video/x-raw,width=1920,height=1080 \
  ! videoconvert \
  ! x264enc threads=2 speed-preset=ultrafast tune=zerolatency bitrate=8000 key-int-max=30 \
  ! mp4mux faststart=true \
  ! filesink location="$session_directory/live_human_teleop.mp4" \
  >"$session_directory/gstreamer.log" 2>&1 &
screen_pid=$!

setsid rosbag record -a -O "$session_directory/live_human_teleop.bag" \
  >"$session_directory/rosbag.log" 2>&1 &
rosbag_pid=$!

setsid rosrun handarm_moveit_demo teleop_csv_logger.py \
  __name:=live_human_evidence_csv_logger \
  _output_file:="$csv_directory/shared_teleop.csv" \
  >"$session_directory/csv_logger.log" 2>&1 &
csv_pid=$!

setsid vmstat -t 1 >"$session_directory/system_vmstat.log" 2>&1 &
vmstat_pid=$!
setsid nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 1 >"$session_directory/gpu_metrics.csv" 2>&1 &
gpu_pid=$!
recording_started=true

printf 'screen_pid=%s\nrosbag_pid=%s\ncsv_pid=%s\nvmstat_pid=%s\ngpu_pid=%s\n' \
  "$screen_pid" "$rosbag_pid" "$csv_pid" "$vmstat_pid" "$gpu_pid" >> \
  "$session_directory/session_metadata.txt"

sleep 0.75
for recorder in "$screen_pid" "$rosbag_pid" "$csv_pid" "$vmstat_pid" "$gpu_pid"; do
  if ! process_group_alive "$recorder"; then
    echo "An evidence recorder failed immediately after C; aborting." >&2
    exit 7
  fi
done
echo "PREROLL_RECORDING_STARTED wall_ns=$preroll_wall_ns"

measurement_ready_file="$hamer_session_directory/measurement_ready.json"
for _ in $(seq 1 36000); do
  if [[ -s "$measurement_ready_file" ]]; then
    break
  fi
  if ! process_group_alive "$camera_pid"; then
    echo "Camera exited while waiting for a stable open-hand measurement." >&2
    exit 6
  fi
  sleep 0.10
done
if [[ ! -s "$measurement_ready_file" ]]; then
  echo "Stable open-hand measurement was not ready within one hour." >&2
  exit 6
fi
cp "$measurement_ready_file" "$session_directory/pre_c_measurement_ready.json"
gnome-screenshot -f "$session_directory/windows_ready.png" >/dev/null 2>&1 || true
echo "READY_FOR_C_RECORDERS_ALREADY_RUNNING session=$session_directory"

# C is persisted by the camera process.  Never ask a human to wait while a
# rostopic subscriber tries to catch a one-frame enabled state.
c_marker="$hamer_session_directory/recording_started.json"
while process_group_alive "$camera_pid"; do
  if [[ -s "$c_marker" ]]; then
    break
  fi
  sleep 0.05
done
if [[ ! -s "$c_marker" ]]; then
  echo "Camera exited before accepting a C reference." >&2
  exit 7
fi
cp "$c_marker" "$session_directory/formal_c_reference.json"
printf 'formal_c_reference_persisted_wall_ns=%s\n' "$(date +%s%N)" >> \
  "$session_directory/session_metadata.txt"
echo "FORMAL_C_REFERENCE_PERSISTED marker=$c_marker"

q_marker="$hamer_session_directory/recording_stopped.json"
while process_group_alive "$camera_pid"; do
  if [[ -s "$q_marker" ]]; then
    break
  fi
  sleep 0.05
done
if [[ ! -s "$q_marker" ]]; then
  echo "Camera exited without persisting the Q/ESC stop marker; evidence is incomplete." >&2
  exit 8
fi
recording_stopped_wall_ns="$(date +%s%N)"
printf 'recording_stopped_wall_ns=%s\n' "$recording_stopped_wall_ns" >> \
  "$session_directory/session_metadata.txt"
echo "CAMERA_Q_DETECTED wall_ns=$recording_stopped_wall_ns"
stop_recorders

for _ in $(seq 1 300); do
  if ! process_group_alive "$camera_pid"; then
    break
  fi
  sleep 0.10
done
if process_group_alive "$camera_pid"; then
  echo "Camera cleanup exceeded 30 seconds; terminating its owned process group." >&2
  stop_process_group "$camera_pid" TERM 50
else
  wait "$camera_pid" 2>/dev/null || true
fi
camera_pid=""

stop_process_group "$ros_pid" INT 200
ros_pid=""

rosbag info "$session_directory/live_human_teleop.bag" \
  > "$session_directory/rosbag_info.txt" 2>&1
gst-discoverer-1.0 "$session_directory/live_human_teleop.mp4" \
  > "$session_directory/video_info.txt" 2>&1
wc -l "$csv_directory/shared_teleop.csv" \
  "$hamer_session_directory/frames.jsonl" > "$session_directory/row_counts.txt"
sha256sum \
  "$session_directory/live_human_teleop.mp4" \
  "$session_directory/live_human_teleop.bag" \
  "$csv_directory/shared_teleop.csv" \
  "$hamer_session_directory/frames.jsonl" \
  > "$session_directory/SHA256SUMS"

normal_completion=true
printf 'complete_wall_time_ns=%s\n' "$(date +%s%N)" >> \
  "$session_directory/session_metadata.txt"
echo "SESSION_COMPLETE=$session_directory"
