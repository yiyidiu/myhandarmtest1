#!/usr/bin/env bash
# FR3 Gazebo entry using the September whole-wrist control path.
set -e
project="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
sim="$project/reproduction/fr3"
prepare=false
gui=true
mode=live
follow=true
record=auto
scene=motion_only
export FR3_WRIST_OBSERVATION="${FR3_WRIST_OBSERVATION:-baseline}"
for option in "$@"; do
  case "$option" in
    --replay) mode=replay ;;
    --prepare) prepare=true ;;
    --headless) gui=false ;;
    --fingers-only) follow=false ;;
    --record) record=true ;;
    --no-recording) record=false ;;
    --qb0) export FR3_WRIST_OBSERVATION=qb0 ;;
    --grasp-scene) scene=ground_grasp ;;
    --stacking-scene) scene=front_stacking ;;
    --pouring-scene) scene=pouring ;;
    --empty-scene) scene=motion_only ;;
    --impedance-grasp) scene=allegro_impedance ;;
    -h|--help)
      echo './run.sh [--prepare] [--record|--no-recording] [--qb0] [--impedance-grasp|--grasp-scene|--pouring-scene|--stacking-scene|--empty-scene] [--replay] [--headless] [--fingers-only]'
      echo '默认场景仅有 FR3 与 Allegro 左手。实时启动默认录制。张开左手，在相机提示就绪后按 C。--qb0 选择腕环中心深度补偿。'
      echo '--prepare 准备本机运行环境和模型，完成后退出。--prepare --replay 只准备仿真和回放。'
      echo '--impedance-grasp 使用单线轴和手指关节阻抗。与 --replay 搭配可观看合成的抓起、放回演示，无需 D455。'
      exit 0 ;;
    *) echo "无法识别参数 $option" >&2; exit 2 ;;
  esac
done
if [[ "$record" == auto ]]; then
  if [[ "$mode" == live ]]; then record=true; else record=false; fi
fi
if [[ "${FR3_SKIP_SYSTEM_INSTALL:-false}" != true ]]; then
  bash "$project/scripts/fr3_system.sh" "$mode" "$record"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ "$mode" == live ]]; then
  /usr/bin/python3 "$project/scripts/fr3_runtime.py"
  export TELEOP_HAMER_PYTHON="$project/.runtime/fr3/envs/hamer_rtx2060/bin/python"
  export TELEOP_MEDIAPIPE_PYTHON="$project/.runtime/fr3/envs/mediapipe_env/bin/python"
  export TELEOP_HAMER_ASSET_ROOT="$project/perception_hamer/_DATA"
fi
exec 9>"${XDG_RUNTIME_DIR:-/tmp}/fr3-allegro-$UID.lock"
flock -n 9 || { echo '已有 FR3 仿真在运行，请在原终端按 Ctrl+C 结束。' >&2; exit 1; }
source /opt/ros/noetic/setup.bash --local
export ROS_MASTER_URI=http://127.0.0.1:11431 GAZEBO_MASTER_URI=http://127.0.0.1:11445
export ROS_IP=127.0.0.1 GAZEBO_IP=127.0.0.1 ALLEGRO_TELEOP_PROJECT="$project"
unset ROS_HOSTNAME
export FR3_WORLD="$project/src/handarm_sim_demo/worlds/handarm_${scene}.world"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="$sim:$project/src:/usr/share/gazebo-11/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
build_target() {
  local src="$1" dir="$2"
  shift 2
  if [[ ! -f "$dir/CMakeCache.txt" ]]; then
    cmake -S "$src" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=OFF -DPYTHON_EXECUTABLE=/usr/bin/python3 >/dev/null
  fi
  cmake --build "$dir" --target "$@" --parallel "${FR3_BUILD_JOBS:-2}" >/dev/null
}
build_target "$sim/fr3_allegro_sim" "$project/.build/robot" fr3_allegro_hw_sim
build_target "$project/src/allegro_finger_teleop" "$project/.build" fr3_arm_clearance
build_target "$project/src/handarm_moveit_demo" "$project/.build/core" full_robot_self_collision_guard
build_target "$project/src/handarm_sim_demo" "$project/.build/scenes" handarm_gui_render_rate_limiter_plugin handarm_pouring_visualization_plugin
source "$project/.build/robot/devel/setup.bash" --extend
source "$project/.build/core/devel/setup.bash" --extend
export GAZEBO_PLUGIN_PATH="$project/.build/robot/devel/lib:$project/.build/scenes/devel/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"
export ROS_PACKAGE_PATH="$project/src:$sim:$ROS_PACKAGE_PATH"
if [[ "$prepare" == true ]]; then
  echo "FR3 + Allegro 环境已就绪 ($mode)。"
  exit 0
fi
export TELEOP_CONTROL_REFERENCE=mano-wrist-ring
source "$project/scripts/teleop_cpu_profile.sh"
launch=(roslaunch -p 11431 allegro_finger_teleop fingers.launch gui:="$gui" source:="$mode" follow_wrist:="$follow")
if [[ "$mode" == replay && -n "${FR3_REPLAY_FILE:-}" ]]; then
  [[ -r "$FR3_REPLAY_FILE" ]] || { echo "无法读取指定的相机回放文件 $FR3_REPLAY_FILE" >&2; exit 2; }
  launch+=(recording:="$(realpath "$FR3_REPLAY_FILE")")
fi
if [[ -n "$teleop_ros_cpus" ]]; then
  export TELEOP_CONTROL_LAUNCH_PREFIX="taskset -c $teleop_control_cpus"
  export FR3_VISION_CPUS="$teleop_inference_cpus" FR3_SIDECAR_CPUS="$teleop_sidecar_cpus" FR3_CAPTURE_CPUS="$teleop_capture_cpus"
  export FR3_PHYSICS_CPUS="$teleop_physics_cpus"
  export FR3_RECORD_PREFIX="taskset -c $teleop_inference_cpus"
  launch=(taskset -c "$teleop_ros_cpus" "${launch[@]}")
fi
if [[ "$record" == true ]]; then
  exec /usr/bin/python3 "$project/scripts/record_fr3_session.py" "$mode" "${launch[@]}"
fi
mkdir -p "$project/.runtime"
export TELEOP_ROBOT_FEEDBACK_FILE="$(mktemp -d "$project/.runtime/fr3-XXXXXXXX")/robot_feedback.json"
unset TELEOP_RGBD_RECORDING_FILE TELEOP_WRIST_DIAGNOSTICS_DIRECTORY FR3_RECORDING_DIRECTORY
exec "${launch[@]}"
