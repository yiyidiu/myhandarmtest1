#!/usr/bin/env bash
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_EMPTY_SCENE=true
export TELEOP_ALIGNMENT_VIEWS=false
export TELEOP_WRIST_DIAGNOSTICS=true
export TELEOP_RECORDING_PREFIX=hand_motion_observation
exec bash "$script_directory/run_teleop_feedback_recording.sh" "$@"
