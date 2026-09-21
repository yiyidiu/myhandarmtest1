#!/usr/bin/env bash
# Three-object grasp scene with the frozen HaMeR whole-wrist control.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_RECORD_IMU=false
exec bash "$script_directory/run_hand_imu_observation.sh" --grasp-scene "$@"
