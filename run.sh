#!/usr/bin/env bash
# Frozen 2026-09-17 operating preset. Paths are configured by setup_current_release.sh.
set -Eeuo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_MOTION_CONTROLLER=constrained
export TELEOP_PALM_TRANSLATION_FRAME=base
export TELEOP_REVERSE_PALM_TURN=true
export TELEOP_EMPTY_SCENE=false
export TELEOP_ALIGNMENT_VIEWS=true
exec bash "$release_root/scripts/run_five_point_axes_teleop.sh" "$@"
