#!/usr/bin/env bash
# Opt-in whole wrist orientation; preserved palm-axis entry stays unchanged.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_HAND_BACKEND=hamer
export TELEOP_CONTROL_REFERENCE="${TELEOP_CONTROL_REFERENCE:-mano-five-point-palm}"
if [[ "$TELEOP_CONTROL_REFERENCE" == mano-wrist-ring ]]; then
  export TELEOP_WRIST_DEPTH_MODE="${TELEOP_WRIST_DEPTH_MODE:-palm-skin}"
else
  export TELEOP_WRIST_DEPTH_MODE="${TELEOP_WRIST_DEPTH_MODE:-joint0-surface}"
fi
export TELEOP_MAPPING_PROFILE=whole_wrist_pose
export TELEOP_IMU_CALIBRATED_POSE=false
export TELEOP_PRECISE_ORIENTATION=false
export TELEOP_POSITIVE_VIEW_BIAS_GAIN=0
export TELEOP_FOREARM_FUSION=false
export TELEOP_EMPTY_SCENE="${TELEOP_EMPTY_SCENE:-false}"
export TELEOP_ALIGNMENT_VIEWS="${TELEOP_ALIGNMENT_VIEWS:-true}"
export TELEOP_SINGLE_CAMERA_PREVIEW="${TELEOP_SINGLE_CAMERA_PREVIEW:-true}"
export TELEOP_SELF_DETOUR="${TELEOP_SELF_DETOUR:-true}"
case "$TELEOP_SELF_DETOUR" in
  true|false) ;;
  *) echo 'TELEOP_SELF_DETOUR 只能是 true 或 false。' >&2; exit 2 ;;
esac
printf '%s\n' \
  '整体腕姿跟随（左手）：小指侧靠近相机→右偏航；拇指侧靠近→左偏航。' \
  '四指靠近相机→下俯仰；四指远离→上俯仰；转掌→末端横滚。' \
  '完整旋转角度统一放大1.3倍，保留合成旋转轴；前后/左右/高度平移增益为1.6/2.0/1.2。' \
  '平移：靠近相机→向下，远离相机→向上；画面上方→向前，画面下方→向后；左右方向不变。' \
  '按C后等待2秒并完成标定，再开始跟随。偏航目标限于±88°，为实际±90°上限保留2°余量；保留关节、自碰撞保护。'
if [[ "$TELEOP_EMPTY_SCENE" == false ]]; then
  if [[ -n "${TELEOP_GRASP_WORLD:-}" ]]; then
    echo '抓取场景：加载当前入口指定的物体布置。'
  else
    echo '抓取场景：三个台子、绿色圆柱、橙色纸盒和水瓶。'
  fi
fi
if [[ "$TELEOP_SELF_DETOUR" == true ]]; then
  echo '自身阻挡时可尝试约4cm小幅绕行，再到原目标；明显的新指令或输入丢失会取消旧绕行。'
fi
if [[ "${1:-}" == "--no-recording" ]]; then
  shift
  if (( $# != 0 )); then
    echo "用法：$0 [新的录制文件夹] 或 $0 --no-recording" >&2
    exit 2
  fi
  export TELEOP_WRIST_DIAGNOSTICS=false
  unset TELEOP_RGBD_RECORDING_FILE TELEOP_WRIST_DIAGNOSTICS_DIRECTORY
  echo '本次不录屏、不采集IMU或RGB-D回放数据。'
  exec bash "$script_directory/run_live_human_teleop_validation.sh"
fi
export TELEOP_WRIST_DIAGNOSTICS=true
export TELEOP_RECORDING_PREFIX="${TELEOP_RECORDING_PREFIX:-teleop_feedback}"
exec bash "$script_directory/run_teleop_feedback_recording.sh" "$@"
