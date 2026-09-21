#!/usr/bin/env bash
# Dedicated pouring + recording entry. Keep posture-only experiments separate.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_RECORD_IMU=false
exec bash "$script_directory/run_hand_imu_observation.sh" --pouring-scene "$@"
