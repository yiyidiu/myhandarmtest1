#!/usr/bin/env bash
# Same whole-wrist control as pouring; only the scene changes.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_RECORD_IMU=false
exec bash "$script_directory/run_hand_imu_observation.sh" --stacking-scene "$@"
