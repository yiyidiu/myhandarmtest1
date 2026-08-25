#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-${workspace_dir}/datasets/formal_usb3}"

if [[ "${RUN_USB3_CERTIFICATION:-}" != "YES" ]]; then
  echo "NOT RUN: set RUN_USB3_CERTIFICATION=YES only after D455 enumerates as USB 3.x." >&2
  exit 2
fi

mkdir -p "${output_dir}"
conda run --no-capture-output -n mediapipe_env \
  python "${workspace_dir}/perception_hamer/scripts/probe_d455_profiles.py" \
  --output "${output_dir}/D455_PROFILE_PROBE.json"

python3 - "${output_dir}/D455_PROFILE_PROBE.json" <<'PY'
import json, sys
probe=json.load(open(sys.argv[1]))
usb=probe["capability"]["device"]["usb_type_descriptor"]
if not usb.startswith("3"):
    raise SystemExit(f"formal certification refused: USB descriptor is {usb}, not 3.x")
PY

cat <<'EOF'
USB3 descriptor gate passed. Remaining certification is intentionally not automated
into a false PASS. Execute and review, in order:
  1. exact selected-profile 300-frame tests
  2. 10-30 minute recording and offline verification
  3. formal depth-bias and RGB-depth alignment evaluation
  4. formal G00-G09 recordings
  5. end-to-end latency/frequency and fault regression
Only the reviewed acceptance tool may update P2_FORMAL_USB3_GATE.
EOF
