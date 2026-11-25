#!/usr/bin/env bash
set -e
project="${ALLEGRO_TELEOP_PROJECT:-$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"
cd "$project"
export PYTHONPATH="$project/third_party/hamer_runtime${PYTHONPATH:+:$PYTHONPATH}"
python="${TELEOP_HAMER_PYTHON:-$project/.runtime/fr3/envs/hamer_rtx2060/bin/python}"
export MEDIAPIPE_PYTHON="${TELEOP_MEDIAPIPE_PYTHON:-$project/.runtime/fr3/envs/mediapipe_env/bin/python}"
export MEDIAPIPE_ENV_PREFIX="$(dirname "$(dirname "$MEDIAPIPE_PYTHON")")"
assets="${TELEOP_HAMER_ASSET_ROOT:-$project/perception_hamer/_DATA}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TELEOP_FOREARM_FUSION=false TELEOP_IMU_CALIBRATED_POSE=false TELEOP_PRECISE_ORIENTATION=false TELEOP_POSITIVE_VIEW_BIAS_GAIN=0
kind=mano-wrist-ring
case "${FR3_WRIST_OBSERVATION:-baseline}" in
  baseline) mode=palm-skin; extra=(--palm-hull-depth) ;;
  qb0) mode=wrist-center-offset; extra=(--pure-view-correction) ;;
  *) echo "未知手腕观测模式" >&2; exit 2 ;;
esac
if [[ -n "${TELEOP_WRIST_DIAGNOSTICS_DIRECTORY:-}" ]]; then
  extra+=(--experiment DEV_HAMER_6DOF_OBSERVATION --experiment-jsonl-only --output-root "$TELEOP_WRIST_DIAGNOSTICS_DIRECTORY")
fi
if [[ -n "${FR3_VISION_CPUS:-}" ]]; then
  extra+=(--inference-cpu-affinity "$FR3_VISION_CPUS" --sidecar-cpu-affinity "$FR3_SIDECAR_CPUS" --capture-cpu-affinity "$FR3_CAPTURE_CPUS")
fi
exec "$python" perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe --fixed-left-hand --mediapipe-min-detection-confidence .50 \
  --mesh-renderer teleoperation-core --no-mesh-overlay --display-rate-hz 15 --finger-landmarks \
  --control-reference "$kind" --wrist-depth-mode "$mode" --roi-smoothing-alpha 1.0 \
  --orientation-filter-large-angle-mode confirm --orientation-filter-max-gain 1.0 \
  --forearm-rate-hz 8.0 --forearm-maximum-source-age-s .20 \
  --hand-presence-timeout-s .30 --hand-miss-grace-frames 2 --hand-miss-grace-s .15 \
  --teleop-maximum-pipeline-latency-s .25 --teleop-udp-host 127.0.0.1 --teleop-udp-port "$1" \
  --show-wrist-axes --single-camera-preview --duration-s 3600 --checkpoint "$assets/hamer_ckpts/checkpoints/hamer.ckpt" \
  --data-root "$assets" "${extra[@]}"
