#!/usr/bin/env bash
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_PRECISE_ORIENTATION=true
export TELEOP_IMU_CALIBRATED_POSE=false
export TELEOP_POSITIVE_VIEW_BIAS_GAIN=0
exec bash "$script_directory/run_hand_motion_observation_recording.sh" "$@"
