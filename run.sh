#!/usr/bin/env bash
# 2026-10-06 baseline: pure perspective correction, ring16, direct joint velocity.
set -Eeuo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scene=--pouring-scene
recording=false
while (($#)); do
  case "$1" in
    --pouring-scene|--stacking-scene|--grasp-scene|--empty-scene) scene="$1" ;;
    --record) recording=true ;;
    --no-recording) recording=false ;;
    -h|--help)
      echo '用法：bash run.sh [--pouring-scene|--stacking-scene|--grasp-scene|--empty-scene] [--record]'
      echo '默认倒水场景、不录制。左手按C标定后控制Gazebo，Q退出。'
      exit 0 ;;
    *) echo "未知选项：$1" >&2; exit 2 ;;
  esac
  shift
done
export PYTHONDONTWRITEBYTECODE=1
export TELEOP_RECORD_IMU=false
export TELEOP_REQUIRE_IMU=false
if [[ "$recording" == true ]]; then
  exec bash "$release_root/scripts/run_hand_imu_observation.sh" "$scene"
fi
exec bash "$release_root/scripts/run_hand_imu_observation.sh" "$scene" --no-recording
