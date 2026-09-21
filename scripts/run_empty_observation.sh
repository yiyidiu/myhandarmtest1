#!/usr/bin/env bash
# Pose/movement observation without tables or props; same control preset.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_RECORD_IMU=false
exec bash "$script_directory/run_hand_imu_observation.sh" --empty-scene "$@"
