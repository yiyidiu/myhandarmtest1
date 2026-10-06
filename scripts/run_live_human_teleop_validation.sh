#!/usr/bin/env bash
set -Eeuo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_directory/.." && pwd)"
cd "$workspace"
if [[ -f "$workspace/teleop_machine.env" ]]; then
  source "$workspace/teleop_machine.env"
fi
source "$script_directory/teleop_cpu_profile.sh"
export PYTHONPATH="$workspace/third_party/hamer_runtime${PYTHONPATH:+:$PYTHONPATH}"

# ROS Noetic setup hooks read optional variables before defining them.
# A fresh shell must work as well as a terminal with ROS already sourced.
set +u
source /opt/ros/noetic/setup.bash
source devel/setup.bash
set -u

arm_control_mode="${TELEOP_ARM_CONTROL_MODE:-position}"
case "$arm_control_mode" in
  position) execution_node=/egm_position_reference_adapter ;;
  velocity) execution_node=/direct_joint_velocity_adapter ;;
  *) echo "TELEOP_ARM_CONTROL_MODE must be position or velocity" >&2; exit 2 ;;
esac

motion_controller="${TELEOP_MOTION_CONTROLLER:-constrained}"
case "$motion_controller" in
  constrained) constrained_velocity_projection=true; joint_velocity_solver=qp ;;
  servo) constrained_velocity_projection=true; joint_velocity_solver=servo ;;
  legacy) constrained_velocity_projection=false; joint_velocity_solver=qp ;;
  *) echo "TELEOP_MOTION_CONTROLLER 只能是 constrained、servo 或 legacy。" >&2; exit 2 ;;
esac

hand_backend="${TELEOP_HAND_BACKEND:-mediapipe3d}"
control_reference="${TELEOP_CONTROL_REFERENCE:-mano-wrist-ring}"
mapping_profile="${TELEOP_MAPPING_PROFILE:-current_linear}"
case "$control_reference" in
  mano-wrist-ring|mano-joint-palm|mano-five-point-palm) ;;
  *) echo "Unsupported TELEOP_CONTROL_REFERENCE" >&2; exit 2 ;;
esac
case "$mapping_profile" in
  current_linear|palm_axes_tool_zero|palm_axes_base_translation|whole_wrist_pose) ;;
  *) echo "Unsupported TELEOP_MAPPING_PROFILE" >&2; exit 2 ;;
esac
if [[ "$mapping_profile" == whole_wrist_pose ]]; then
  if [[ "$hand_backend" != hamer || ( "$control_reference" != mano-five-point-palm && "$control_reference" != mano-wrist-ring ) ]]; then
    echo '整体腕姿映射需要 HaMeR 掌骨或腕口输入。' >&2; exit 2
  fi
elif [[ "$mapping_profile" == palm_axes_* || "$control_reference" == mano-five-point-palm ]]; then
  if [[ "$hand_backend" != hamer || "$control_reference" != mano-five-point-palm || "$mapping_profile" == current_linear ]]; then
    echo "新版五点坐标必须与对应的平移/旋转映射配套，请运行 run_five_point_axes_teleop.sh。" >&2
    exit 2
  fi
fi
case "$hand_backend" in
  mediapipe) mediapipe_mode=rgbd ;;
  mediapipe3d) mediapipe_mode=world ;;
  hamer) ;; # retained only for an explicit baseline comparison
  *) echo "TELEOP_HAND_BACKEND: mediapipe, mediapipe3d, hamer" >&2; exit 2 ;;
esac
mediapipe_python="${TELEOP_MEDIAPIPE_PYTHON:-$workspace/.venvs/mediapipe_env/bin/python}"
export MEDIAPIPE_PYTHON="$mediapipe_python"
export MEDIAPIPE_ENV_PREFIX="$(dirname "$(dirname "$mediapipe_python")")"
if [[ "$hand_backend" != hamer ]]; then
  export TELEOP_IMU_CALIBRATED_POSE=false
  export TELEOP_PRECISE_ORIENTATION=false
fi

conda_executable="${TELEOP_CONDA_EXECUTABLE:-$HOME/anaconda3/bin/conda}"
if [[ -n "${TELEOP_HAMER_PYTHON:-}" ]]; then
  hamer_command=("$TELEOP_HAMER_PYTHON")
elif [[ -x "$workspace/.venvs/hamer_rtx2060/bin/python" ]]; then
  hamer_command=("$workspace/.venvs/hamer_rtx2060/bin/python")
else
  hamer_command=("$conda_executable" run --no-capture-output -n hamer_rtx2060 python)
fi
asset_root="${TELEOP_HAMER_ASSET_ROOT:-$workspace/perception_hamer/_DATA}"
checkpoint="$asset_root/hamer_ckpts/checkpoints/hamer.ckpt"
empty_motion_scene="${TELEOP_EMPTY_SCENE:-false}"
case "$empty_motion_scene" in true|false) ;; *) echo "TELEOP_EMPTY_SCENE must be true or false" >&2; exit 2 ;; esac
ground_object_world="${TELEOP_GRASP_WORLD:-$workspace/src/handarm_sim_demo/worlds/handarm_ground_grasp.world}"
geometry_arguments=()
case "${TELEOP_PURE_VIEW_CORRECTION:-false}" in
  true) geometry_arguments=(--pure-view-correction) ;;
  false) ;;
  *) echo 'TELEOP_PURE_VIEW_CORRECTION must be true or false.' >&2; exit 2 ;;
esac
diagnostic_arguments=()
if [[ "${TELEOP_WRIST_DIAGNOSTICS:-false}" == true ]]; then
  diagnostic_arguments=(--show-wrist-axes --experiment DEV_HAMER_6DOF_OBSERVATION
    --experiment-jsonl-only --output-root "${TELEOP_WRIST_DIAGNOSTICS_DIRECTORY:?recording directory required}")
elif [[ "${TELEOP_SINGLE_CAMERA_PREVIEW:-false}" == true ]]; then
  diagnostic_arguments=(--show-wrist-axes)
fi
runtime_directory="$workspace/.runtime/live_human_validation_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$runtime_directory"

required_files=("$mediapipe_python")
if [[ "$hand_backend" == hamer ]]; then
  required_files=("${hamer_command[0]}" "$mediapipe_python" "$checkpoint"
    "$asset_root/hamer_ckpts/model_config.yaml" "$asset_root/data/mano_mean_params.npz"
    "$asset_root/data/mano/MANO_RIGHT.pkl")
fi
if [[ "$empty_motion_scene" == false ]]; then
  required_files+=("$ground_object_world")
fi
for required in "${required_files[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "缺少运行文件：$required" >&2
    echo "首次下载请按 README 执行 scripts/setup_current_release.sh。" >&2
    exit 2
  fi
done
if pgrep -f 'run_d455_hamer_crop.py|perception_mediapipe/run_live.py|live_human_ground_gazebo_egm_teleop.launch' \
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
if [[ "$arm_control_mode" == velocity ]]; then
  echo "机械臂底层：250 Hz 关节速度指令；速度闭环执行，不运行外部关节位置积分器。"
fi
if [[ "$joint_velocity_solver" == servo ]]; then
  echo "本次使用 MoveIt Servo 求解，保留当前反馈、手指保护和 250 Hz 参考链。"
elif [[ "$constrained_velocity_projection" == true ]]; then
  echo "本次使用约束速度控制；旧控制可通过 TELEOP_MOTION_CONTROLLER=legacy 运行。"
else
  echo "本次使用本轮修改前的运动控制；物体使用当前布局。"
fi
simulation_affinity_arguments=()
if [[ -n "$teleop_ros_cpus" ]]; then
  simulation_affinity_arguments=(taskset -c "$teleop_ros_cpus")
  export TELEOP_CONTROL_LAUNCH_PREFIX="${TELEOP_CONTROL_LAUNCH_PREFIX:-taskset -c $teleop_control_cpus}"
fi
setsid "${simulation_affinity_arguments[@]}" roslaunch handarm_moveit_demo \
  live_human_ground_gazebo_egm_teleop.launch \
  gazebo_gui:=true with_ground_object:=true enable_logger:=false \
  arm_control_mode:="$arm_control_mode" \
  empty_motion_scene:="$empty_motion_scene" \
  ground_object_world:="$ground_object_world" \
  use_simulated_environment_for_control:=false \
  grasp_alignment_views:="${TELEOP_ALIGNMENT_VIEWS:-true}" \
  input_source:=udp \
  constrained_velocity_projection:="$constrained_velocity_projection" \
  joint_velocity_solver:="$joint_velocity_solver" \
  position_accommodation_m:="${TELEOP_POSITION_ACCOMMODATION_M:-0.0}" \
  mapping_profile:="$mapping_profile" \
  hamer_input_timeout_s:=0.40 \
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
      && timeout 1 rosnode info /moveit_servo_output_adapter >/dev/null 2>&1 \
      && timeout 1 rosnode info "$execution_node" >/dev/null 2>&1; then
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

# The stock Gazebo launch cannot take a server launch-prefix. Reserve its
# physical core after identifying its own ROS PID, before starting vision.
teleop_pin_gazebo_physics

echo "Gazebo 和控制器已就绪。控制器不读取仿真桌子、物体或地面坐标，自碰撞保护保留。"
echo "正在打开相机窗口……"
echo "本次运行日志：$runtime_directory"

# This observer subscribes before C and never gates the control path.  It
# distinguishes camera acceptance, a non-zero Servo command, and real Gazebo
# tool0 displacement so a broken link cannot masquerade as a working one.
export TELEOP_ROBOT_FEEDBACK_FILE="$runtime_directory/robot_feedback.json"
"${simulation_affinity_arguments[@]}" rosrun handarm_moveit_demo live_teleop_terminal_monitor.py \
  _feedback_file:="$TELEOP_ROBOT_FEEDBACK_FILE" &
link_monitor_pid=$!

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
camera_affinity_arguments=()
if [[ -n "$teleop_inference_cpus" ]]; then
  camera_affinity_arguments+=(
    --inference-cpu-affinity "$teleop_inference_cpus"
    --sidecar-cpu-affinity "$teleop_sidecar_cpus"
    --capture-cpu-affinity "$teleop_capture_cpus"
  )
fi
if [[ "$hand_backend" != hamer ]]; then
  "$mediapipe_python" perception_mediapipe/run_live.py --mode "$mediapipe_mode" \
    --output "${TELEOP_WRIST_DIAGNOSTICS_DIRECTORY:-$runtime_directory/perception}" \
    2>&1 | tee "$runtime_directory/camera.log"
else
"${hamer_command[@]}" perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe \
  --fixed-left-hand \
  "${geometry_arguments[@]}" \
  --mediapipe-min-detection-confidence 0.50 \
  --mesh-renderer teleoperation-core \
  --no-mesh-overlay \
  --display-rate-hz 15 \
  --control-reference "$control_reference" \
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
  "${diagnostic_arguments[@]}" \
  "${camera_affinity_arguments[@]}" \
  2>&1 | tee "$runtime_directory/camera.log"

fi
