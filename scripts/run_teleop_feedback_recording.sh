#!/usr/bin/env bash
# Record the current repair candidate without changing its controller settings.
set -Eeuo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_directory/.." && pwd)"
helpers="$script_directory/feedback_capture"
imu_python="${TELEOP_IMU_PYTHON:-/home/dongtian/anaconda3/envs/mediapipe_env/bin/python}"
reconstruction="$helpers/reconstruct_wt901_sample_time.py"
cd "$workspace"
source /opt/ros/noetic/setup.bash
source "$workspace/devel/setup.bash"
recording_affinity_arguments=()
if command -v taskset >/dev/null 2>&1 && (( $(nproc) >= 12 )); then
  recording_affinity_arguments=(taskset -c 2-7)
fi

if (( $# > 1 )); then
  echo "用法：$0 [新的录制文件夹]" >&2
  exit 2
fi
if pgrep -f 'run_d455_hamer_crop.py|live_human_ground_gazebo_egm_teleop.launch' >/dev/null 2>&1; then
  echo "已有遥操作进程，请先在旧相机窗口按 Q，等待旧命令退出后再录制。" >&2
  exit 3
fi
mkdir -p "$workspace/.runtime"
if (( $# == 1 )); then
  session_directory="$(realpath -m -- "$1")"
  if ! mkdir -- "$session_directory"; then
    echo "请使用新的文件夹；已有录像不会覆盖。省略文件夹参数可自动新建。" >&2
    exit 2
  fi
else
  recording_prefix="${TELEOP_RECORDING_PREFIX:-teleop_feedback}"
  [[ "$recording_prefix" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "Invalid recording prefix" >&2; exit 2; }
  session_directory="$(mktemp -d "$workspace/.runtime/${recording_prefix}_$(date +%Y%m%dT%H%M%S)_XXXX")"
fi
printf '%s\n' "$session_directory" > "$workspace/.runtime/latest_teleop_feedback.txt"
if [[ "${TELEOP_WRIST_DIAGNOSTICS:-false}" == true ]]; then
  printf '%s\n' "$session_directory" > "$workspace/.runtime/latest_hand_motion_observation.txt"
fi
export TELEOP_RGBD_RECORDING_FILE="$session_directory/rgbd.db3"
export TELEOP_WRIST_DIAGNOSTICS_DIRECTORY="$session_directory/perception"
git rev-parse HEAD > "$session_directory/commit.txt"
git status --porcelain -uno > "$session_directory/tracked_changes.txt"
git diff HEAD -- > "$session_directory/code_changes.patch"
if [[ -d "$workspace/src/moveit_servo" ]]; then
  mkdir -p "$session_directory/controller_source"
  cp -a "$workspace/src/moveit_servo" "$session_directory/controller_source/"
fi
if [[ -d "$workspace/src/handarm_moveit_demo/third_party/qpOASES" ]]; then
  mkdir -p "$session_directory/controller_source"
  cp -a "$workspace/src/handarm_moveit_demo/third_party/qpOASES" "$session_directory/controller_source/"
fi
if [[ -d "$workspace/src/handarm_sim_demo/meshes" ]]; then
  mkdir -p "$session_directory/controller_source/handarm_sim_demo"
  cp -a "$workspace/src/handarm_sim_demo/meshes" "$session_directory/controller_source/handarm_sim_demo/"
fi
cp "$script_directory/run_teleop_feedback_recording.sh" "$session_directory/recording_launcher.sh"
printf '%s\n' "${ROS_MASTER_URI:-http://localhost:11311}" > "$session_directory/ros_master_uri.txt"
/usr/bin/python3 - "$session_directory" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1])
value={"status":"PREPARING_RECORDERS","created_host_wall_ns":time.time_ns(),"created_host_monotonic_ns":time.monotonic_ns(),
 "display":os.environ.get("DISPLAY"),"screen_video":"screen.mp4","video_frame_clock":"screen_frame_timeline.csv",
 "imu_raw":"imu_raw.csv","imu_estimated_sample_times":"imu_time_estimated.csv","camera_pose":"camera_hand_pose.csv",
 "host_ros_clock":"host_ros_clock.csv","control_bag":"control_chain.bag","raw_rgbd":"rgbd.db3",
 "imu_mounting_instruction":"Keep the IMU taped to the back of the hand; no removal or repositioning required.",
 "camera_pose_role":"The pose actually received by the pinned controller, including its existing filtering/fusion.",
 "time_alignment":"MP4 frame PTS maps to host monotonic time; camera source and host receipt times are retained; ROS simulation time has a host clock bridge. IMU packet receipt is not device sampling time. Physical latency alignment remains a post-recording analysis.",
 "controller_workspace":str(Path.cwd()),"recording_changes_controller_parameters":False,
 "motion_controller":os.environ.get("TELEOP_MOTION_CONTROLLER","constrained"),
 "alignment_views_requested":os.environ.get("TELEOP_ALIGNMENT_VIEWS","true"),
 "empty_motion_scene":os.environ.get("TELEOP_EMPTY_SCENE","false")=="true",
 "wrist_axes_diagnostics":os.environ.get("TELEOP_WRIST_DIAGNOSTICS","false")=="true",
 "live_rgb_preview":os.environ.get("TELEOP_LIVE_RGB_PREVIEW","false").lower()=="true",
 "wrist_depth_mode":os.environ.get("TELEOP_WRIST_DEPTH_MODE","visible-surface"),
 "forearm_fusion_requested":os.environ.get("TELEOP_FOREARM_FUSION","false").lower()=="true",
 "forearm_control_fusion_enabled":False,
 "positive_view_bias_gain":float(os.environ.get("TELEOP_POSITIVE_VIEW_BIAS_GAIN","0")),
 "imu_calibrated_pose_enabled":os.environ.get("TELEOP_IMU_CALIBRATED_POSE","false").lower()=="true",
 "precise_orientation_enabled":os.environ.get("TELEOP_PRECISE_ORIENTATION","false").lower()=="true",
 "requested_camera_resolution":os.environ.get("TELEOP_CAMERA_RESOLUTION","640x480"),
 "position_accommodation_m":float(os.environ.get("TELEOP_POSITION_ACCOMMODATION_M","0")),
 "imu_used_by_controller":False,"audio_recorded":False}
(p/'session.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
PY

bag_pid=""
run_pid=""
screen_pid=""
imu_pid=""
clock_pid=""
stop_owned_group() {
  local process_id="$1"
  local signal="$2"
  [[ -n "$process_id" ]] || return 0
  kill -0 "$process_id" 2>/dev/null || return 0
  kill "-$signal" -- "-$process_id" 2>/dev/null || true
  local attempts="${3:-100}"
  for _ in $(seq 1 "$attempts"); do
    kill -0 "$process_id" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$process_id" 2>/dev/null; then
    kill -TERM -- "-$process_id" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$process_id" 2>/dev/null || break
      sleep 0.1
    done
  fi
  if kill -0 "$process_id" 2>/dev/null; then
    kill -KILL -- "-$process_id" 2>/dev/null || true
  fi
  wait "$process_id" 2>/dev/null || true
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_owned_group "$run_pid" TERM
  stop_owned_group "$screen_pid" INT 200
  stop_owned_group "$imu_pid" INT
  stop_owned_group "$clock_pid" INT
  stop_owned_group "$bag_pid" INT
  if [[ -s "$session_directory/imu_ready.json" ]]; then
    "$imu_python" "$reconstruction" "$session_directory/imu_raw.csv" \
      --output "$session_directory/imu_time_estimated.csv" --nominal-hz 50 \
      > "$session_directory/imu_time_reconstruction.log" 2>&1 || true
  fi
  /usr/bin/python3 - "$session_directory" "$status" <<'PY'
import csv,json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'session.json').read_text())
def row_count(name):
 path=p/name
 if not path.exists():return 0
 with path.open() as f:return max(0,sum(1 for _ in csv.reader(f))-1)
s.update(control_exit_code=int(sys.argv[2]),stopped_host_wall_ns=time.time_ns(),stopped_host_monotonic_ns=time.monotonic_ns(),
 imu_samples=row_count('imu_raw.csv'),camera_pose_rows=row_count('camera_hand_pose.csv'),host_ros_clock_rows=row_count('host_ros_clock.csv'),
 bag_finalized=(p/'control_chain.bag').exists(),imu_estimated_time_file=(p/'imu_time_estimated.csv').exists())
screen=p/'screen_summary.json';s['screen_summary']=json.loads(screen.read_text()) if screen.exists() else None
video_ok=bool(s['screen_summary'] and s['screen_summary'].get('eos') and s['screen_summary'].get('container_timeline',{}).get('usable_frame_mapping'))
s['imu_available']=s['imu_samples']>0
s['raw_rgbd_file_present']=(p/'rgbd.db3').is_file() and (p/'rgbd.db3').stat().st_size>0
s['perception_diagnostics']=[str(f.relative_to(p)) for f in sorted(p.glob('perception/*/frames.jsonl'))]
s['status']='RECORDING_FILES_FINALIZED' if video_ok and s['camera_pose_rows']>0 and s['bag_finalized'] else 'RECORDING_INCOMPLETE'
s['status_scope']='Recorded stream presence/finalization only; not evidence of accurate sensor alignment or successful teleoperation.'
(p/'session.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
print('录制结果：视频时间表%s，IMU %d条，相机位姿 %d条。'%('已保存' if video_ok else '不完整',s['imu_samples'],s['camera_pose_rows']))
if s['status']=='RECORDING_INCOMPLETE':print('注意：至少一项记录不完整；已保留现有文件和错误日志。')
if not s['imu_available']:print('本次没有收到IMU数据，不能据此对照IMU姿态；录像和控制数据单独保留。')
if not s['raw_rgbd_file_present']:print('本次原始RGB-D记录不可用；已有录像和位姿数据仍保留。')
PY
  printf '\n本次操作日志：%s\n' "$session_directory"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "启动当前修复候选版，自动录制屏幕与控制数据；只控制 Gazebo。"
echo "同时尝试记录手背IMU；未收到IMU数据时会明确提示，不阻止录屏。"
echo "请等待相机窗口，保持自然张手后按一次 C；按 Q 退出后等待录像保存完成。"
echo "本次操作日志：$session_directory"
echo "回放视频：$session_directory/screen.mp4"
echo "原始RGB-D：$session_directory/rgbd.db3"
if [[ "${TELEOP_WRIST_DIAGNOSTICS:-false}" == true ]]; then
  if [[ "${TELEOP_LIVE_RGB_PREVIEW:-false}" == true ]]; then
    echo "左侧为实时相机画面，底部保留原始腕口轴；右侧为姿态计算所用画面。轴颜色：红X、绿Y、蓝Z。"
  else
    echo "腕口轴：红X、绿Y、蓝Z；左侧原始估计，右侧处理后估计。"
  fi
  echo "原始逐帧数值：$session_directory/perception/"
fi
if [[ "${TELEOP_EMPTY_SCENE:-false}" == true ]]; then
  echo "本次场景：机械臂、三指手与地面；桌子和物体关闭。"
fi
echo "反馈时直接使用播放器从 00:00 开始的时间，不需要扣除启动等待。"

setsid "${recording_affinity_arguments[@]}" /usr/bin/python3 "$helpers/record_screen_clock.py" "$session_directory" \
  > "$session_directory/screen_recorder.log" 2>&1 &
screen_pid=$!
for _ in $(seq 1 100); do
  [[ -s "$session_directory/screen_ready.json" ]] && break
  if ! kill -0 "$screen_pid" 2>/dev/null; then
    echo "屏幕录制启动失败：$session_directory/screen_recorder.log" >&2
    exit 4
  fi
  sleep .1
done
if [[ ! -s "$session_directory/screen_ready.json" ]]; then
  echo "尚未录到屏幕帧，未启动遥操作。" >&2
  exit 4
fi

printf -v imu_command '%q ' "${recording_affinity_arguments[@]}" "$imu_python" "$helpers/record_imu_raw.py" \
  --duration 0 --display-hz 1 --csv "$session_directory/imu_raw.csv" \
  --ready-file "$session_directory/imu_ready.json"
setsid sg dialout -c "$imu_command" > "$session_directory/imu_recorder.log" 2>&1 &
imu_pid=$!
echo "屏幕已开始记录，正在启动仿真和相机。IMU是否收到数据会单独报告。"

setsid "${recording_affinity_arguments[@]}" rosbag record --buffsize=64 -O "$session_directory/control_chain.bag" \
  /shared_teleop/hamer_pose /shared_teleop/raw_hand_command \
  /shared_teleop/safe_twist /shared_teleop/trend_diagnostics \
  /shared_teleop/robot_zero_pose \
  /shared_teleop/pose_target \
  /shared_teleop/output_diagnostics /shared_teleop/assist_diagnostics \
  /servo_server/status /servo_server/internal/collision_velocity_scale \
  /tf /tf_static /joint_states /clock \
  /shared_teleop/geometric_filter_diagnostics \
  /shared_teleop/shaped_command \
  /shared_teleop/geometrically_limited_command \
  /full_robot_self_collision_guard/command_diagnostic \
  /egm_position_reference/diagnostics \
  /egm_position_reference/command_state \
  /egm_position_reference/collision_checked_joint_velocity \
  /full_robot_self_collision_guard/command_safe \
  /shared_teleop/finger_diagnostics /controller_gazebo_hand/command \
  /gazebo/model_states /handarm_sim_demo/target_contacts \
  > "$session_directory/recorder.log" 2>&1 &
bag_pid=$!

setsid bash "$workspace/scripts/run_live_human_teleop_validation.sh" \
  > >(tee "$session_directory/console.log") 2>&1 &
run_pid=$!
setsid "${recording_affinity_arguments[@]}" /usr/bin/python3 "$helpers/record_ros_clock_and_hand.py" "$session_directory" \
  > "$session_directory/clock_recorder.log" 2>&1 &
clock_pid=$!

declare -A warned=()
imu_wait_started=$SECONDS
while kill -0 "$run_pid" 2>/dev/null; do
  for stream in screen imu clock bag; do
    variable="${stream}_pid"
    process_id="${!variable}"
    if ! kill -0 "$process_id" 2>/dev/null && [[ -z "${warned[$stream]:-}" ]]; then
      echo "[录制提醒] $stream 记录进程已停止。遥操作不会因此被更改，其余记录继续；退出后检查日志。" | tee -a "$session_directory/recording_warnings.log"
      warned[$stream]=1
    fi
  done
  if [[ -s "$session_directory/imu_ready.json" && -z "${warned[imu_ready]:-}" ]]; then
    echo "[录制] 已收到新的IMU数据，正在同步保存。" | tee -a "$session_directory/recording_warnings.log"
    warned[imu_ready]=1
  elif (( SECONDS - imu_wait_started > 30 )) && [[ ! -s "$session_directory/imu_ready.json" && -z "${warned[imu_missing]:-}" ]]; then
    echo "[录制提醒] 尚未收到IMU数据；录像与控制数据继续。" | tee -a "$session_directory/recording_warnings.log"
    warned[imu_missing]=1
  fi
  if [[ -s "$session_directory/imu_raw.csv" ]]; then
    # The CSV is buffered. The reader flushes its sample-derived display
    # once a second, so its log reflects received packets without reporting
    # a false disconnect while the CSV buffer is still filling.
    age=$(( $(date +%s) - $(stat -c %Y "$session_directory/imu_recorder.log") ))
    if (( age > 5 )) && [[ -z "${warned[imu_stale]:-}" ]]; then
      echo "[录制提醒] IMU超过5秒没有新数据，请检查电源；录像与控制日志继续。" | tee -a "$session_directory/recording_warnings.log"
      warned[imu_stale]=1
    elif (( age <= 5 )); then
      unset 'warned[imu_stale]'
    fi
  fi
  sleep 1
done
wait "$run_pid"
