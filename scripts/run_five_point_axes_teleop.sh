#!/usr/bin/env bash
# Opt-in user palm axes; the wrist-ring baseline entry remains available.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_HAND_BACKEND=hamer
export TELEOP_CONTROL_REFERENCE=mano-five-point-palm
case "${TELEOP_PALM_TRANSLATION_FRAME:-base}" in
  base) export TELEOP_MAPPING_PROFILE=palm_axes_base_translation ;;
  tool_zero) export TELEOP_MAPPING_PROFILE=palm_axes_tool_zero ;;
  *) echo "TELEOP_PALM_TRANSLATION_FRAME 只能是 base 或 tool_zero。" >&2; exit 2 ;;
esac
export TELEOP_REVERSE_PALM_TURN="${TELEOP_REVERSE_PALM_TURN:-true}"
case "$TELEOP_REVERSE_PALM_TURN" in
  true|false) ;;
  *) echo "TELEOP_REVERSE_PALM_TURN 只能是 true 或 false。" >&2; exit 2 ;;
esac
export TELEOP_IMU_CALIBRATED_POSE=false
export TELEOP_PRECISE_ORIENTATION=false
export TELEOP_POSITIVE_VIEW_BIAS_GAIN=0
export TELEOP_FOREARM_FUSION=false
export TELEOP_EMPTY_SCENE="${TELEOP_EMPTY_SCENE:-true}"
export TELEOP_ALIGNMENT_VIEWS="${TELEOP_ALIGNMENT_VIEWS:-true}"
printf '%s\n' '新版左手坐标：红X朝手心外，绿Y食指→小指，蓝Z手腕→指根中心。'
if [[ "$TELEOP_REVERSE_PALM_TURN" == true ]]; then
  printf '%s\n' '手掌顺/逆时针对应的末端转向已交换：绕手+X→末端−X；其余旋转仍为+Y→−Y、+Z→+Z。'
else
  printf '%s\n' '旋转方向回退：绕手+X→末端+X、+Y→−Y、+Z→+Z。'
fi
printf '%s\n' '按C记录新的手零位和末端零位；保留原正反向转动幅度。'
if [[ "$TELEOP_MAPPING_PROFILE" == palm_axes_base_translation ]]; then
  printf '%s\n' '平移：手向前（靠近相机）→机械臂向前；手向上→机械臂向上。' \
    '平移方向固定在空间中，不随手或末端的零位朝向翻转；原增益和机械臂保护保留。'
else
  printf '%s\n' '历史回退模式：平移按手零位→末端零位 +X→+X、+Y→−Y、+Z→+Z。' \
    '注意默认末端+X朝后、+Z朝下，此模式会复现修正前的前后/上下反向。'
fi
exec bash "$script_directory/run_live_human_teleop_validation.sh" "$@"
