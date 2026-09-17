#!/usr/bin/env bash
# Roll back by selecting the original source and mapping; no git reset needed.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_HAND_BACKEND=hamer
export TELEOP_CONTROL_REFERENCE=mano-wrist-ring
export TELEOP_MAPPING_PROFILE=current_linear
export TELEOP_IMU_CALIBRATED_POSE=false
export TELEOP_PRECISE_ORIENTATION=false
export TELEOP_POSITIVE_VIEW_BIAS_GAIN=0
export TELEOP_FOREARM_FUSION=false
export TELEOP_EMPTY_SCENE="${TELEOP_EMPTY_SCENE:-true}"
export TELEOP_ALIGNMENT_VIEWS=false
printf '%s\n' '原腕环坐标与原相机固定方向映射。按C重新标定，Q退出。'
exec bash "$script_directory/run_live_human_teleop_validation.sh" "$@"
