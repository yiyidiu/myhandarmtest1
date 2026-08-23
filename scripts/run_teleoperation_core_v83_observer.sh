#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="$($PROJECT_ROOT/scripts/prepare_teleoperation_core_v83.sh --print-root)"
ENV_NAME="${TELEOP_V83_ENV:-teleop_v83}"
CHECKPOINT="$PROJECT_ROOT/perception_hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
MODEL="$RUNTIME_ROOT/data/crossuser_models/V83_MULTISCALE_NET_TWO_PERSON_20260728/multiscale_intent_v83.joblib"
OBSERVER="$RUNTIME_ROOT/hamer-win/live_v83_pose_observer_windows.py"
COMPAT_OBSERVER="$PROJECT_ROOT/perception_hamer/scripts/run_teleoperation_core_v83_compat.py"
LOCK="/tmp/handarm_d455_hamer_live_uid$(id -u).lock"

for required in "$CHECKPOINT" "$MODEL" "$OBSERVER" "$COMPAT_OBSERVER"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] required file is missing: $required" >&2
    exit 2
  fi
done

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[ERROR] conda environment '$ENV_NAME' is missing." >&2
  echo "Run: $PROJECT_ROOT/scripts/setup_teleoperation_core_v83_env.sh" >&2
  exit 3
fi

# The current crop runner and this exact archive observer each load one full
# HaMeR checkpoint.  Refuse a second copy instead of reproducing the RTX 2060
# out-of-memory failure.
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "[ERROR] another D455/HaMeR process is still running." >&2
  echo "Stop its terminal with Ctrl-C, then run this command again." >&2
  exit 4
fi

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.runtime/matplotlib_v83}"
export TELEOP_CORE_RUNTIME_ROOT="$RUNTIME_ROOT"
mkdir -p "$MPLCONFIGDIR"

echo "[SAFE] Exact archived V8.3 observer + wrist XYZ/YPR panel"
echo "[SAFE] No UDP, ROS, Gazebo or robot output exists in this entry point."
echo "[INPUT] Use the physical LEFT hand; keep the wrist and >=15 cm forearm visible."
echo "[KEYS]  C=set current wrist pose as zero, R=clear zero, Q/Esc=exit."

cd "$RUNTIME_ROOT/hamer-win"
exec conda run --no-capture-output -n "$ENV_NAME" \
  python -u "$COMPAT_OBSERVER" \
  --intent-checkpoint "$MODEL" \
  --hamer-checkpoint "$CHECKPOINT" \
  --forearm-estimator v5 \
  --sample-hz 10.0 \
  --width 640 \
  --height 480 \
  --camera-fps 30 \
  --bbox-hold-s 0.0 \
  --model-history-grace-s 0.0 \
  "$@"
