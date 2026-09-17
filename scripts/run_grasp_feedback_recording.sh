#!/usr/bin/env bash
# Match the 2026-09-12 observation settings, with the grasp scene restored.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TELEOP_MOTION_CONTROLLER="${TELEOP_MOTION_CONTROLLER:-constrained}"
export TELEOP_EMPTY_SCENE=false
export TELEOP_HAND_BACKEND="${TELEOP_HAND_BACKEND:-mediapipe3d}"
export TELEOP_IMU_CALIBRATED_POSE=false
export TELEOP_POSITIVE_VIEW_BIAS_GAIN=0
export TELEOP_PRECISE_ORIENTATION=false
export TELEOP_CAMERA_RESOLUTION=640x480
export TELEOP_ALIGNMENT_VIEWS="${TELEOP_ALIGNMENT_VIEWS:-true}"
export TELEOP_WRIST_DIAGNOSTICS=true
export TELEOP_LIVE_RGB_PREVIEW=true
export TELEOP_RECORDING_PREFIX=teleop_feedback

if [[ "${1:-}" == "--no-recording" ]]; then
  shift
  if (( $# != 0 )); then
    echo "用法：$0 --no-recording" >&2
    exit 2
  fi
  # Reuse the same live perception/control settings without launching any
  # screen, IMU, rosbag or RGB-D recorder. Ordinary runtime logs remain.
  export TELEOP_WRIST_DIAGNOSTICS=false
  unset TELEOP_RGBD_RECORDING_FILE TELEOP_WRIST_DIAGNOSTICS_DIRECTORY
  echo "本次只运行相机和 Gazebo，不录屏、不采集 IMU 或 RGB-D 回放数据。"
  exec bash "$script_directory/run_live_human_teleop_validation.sh"
fi

exec bash "$script_directory/run_teleop_feedback_recording.sh" "$@"
