#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="$($PROJECT_ROOT/scripts/prepare_teleoperation_core_v83.sh --print-root)"
ENV_NAME="${TELEOP_V83_ENV:-teleop_v83}"
MODEL="$RUNTIME_ROOT/data/crossuser_models/V83_MULTISCALE_NET_TWO_PERSON_20260728/multiscale_intent_v83.joblib"

cd "$RUNTIME_ROOT/hamer-win"
conda run --no-capture-output -n "$ENV_NAME" python -m pytest -q \
  test_v83_pose_overlay.py \
  test_multiscale_intent_v83.py \
  test_forearm_tracker_v5_windows.py \
  test_v83_robust_route.py

conda run --no-capture-output -n "$ENV_NAME" python -c \
  "from multiscale_intent_v83 import load_multiscale_checkpoint; model=load_multiscale_checkpoint(r'$MODEL'); print('V8.3 MODEL OK:', model.metadata['model_name'])"
