#!/usr/bin/env bash
# Match the 2026-09-12 observation settings, with the grasp scene restored.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TELEOP_MOTION_CONTROLLER=constrained
export TELEOP_EMPTY_SCENE=false
export TELEOP_IMU_CALIBRATED_POSE=true
export TELEOP_POSITIVE_VIEW_BIAS_GAIN=0
export TELEOP_PRECISE_ORIENTATION=false
export TELEOP_CAMERA_RESOLUTION=640x480
export TELEOP_ALIGNMENT_VIEWS=false
export TELEOP_WRIST_DIAGNOSTICS=true
export TELEOP_LIVE_RGB_PREVIEW=true
export TELEOP_RECORDING_PREFIX=teleop_feedback

exec bash "$script_directory/run_teleop_feedback_recording.sh" "$@"
