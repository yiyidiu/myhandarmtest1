#!/usr/bin/env bash
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_IMU_CALIBRATED_POSE=true
export TELEOP_POSITIVE_VIEW_BIAS_GAIN=0
exec bash "$script_directory/run_hand_motion_observation_recording.sh" "$@"
