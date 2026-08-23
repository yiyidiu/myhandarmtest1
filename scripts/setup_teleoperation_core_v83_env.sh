#!/usr/bin/env bash
set -euo pipefail

# Keep the proven low-memory HaMeR environment untouched.  The reference
# observer needs GUI OpenCV, MediaPipe, RealSense and the exact sklearn version
# used to serialize the frozen V8.3 model, so it gets a dedicated clone.

SOURCE_ENV="${HAMER_SOURCE_ENV:-hamer_rtx2060}"
TARGET_ENV="${TELEOP_V83_ENV:-teleop_v83}"

if ! conda env list | awk '{print $1}' | grep -Fxq "$SOURCE_ENV"; then
  echo "[ERROR] source environment does not exist: $SOURCE_ENV" >&2
  exit 2
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$TARGET_ENV"; then
  conda create -y -n "$TARGET_ENV" --clone "$SOURCE_ENV"
fi

if conda run --no-capture-output -n "$TARGET_ENV" python -c \
  "import cv2,joblib,mediapipe,pandas,pyrealsense2,pytest,sklearn; assert cv2.__version__=='4.11.0'; assert joblib.__version__=='1.5.3'; assert mediapipe.__version__=='0.10.21'; assert pandas.__version__=='2.2.3'; assert pytest.__version__=='8.3.5'; assert sklearn.__version__=='1.5.2'" \
  >/dev/null 2>&1; then
  echo "[OK] $TARGET_ENV already has the audited V8.3 dependency set."
  exit 0
fi

conda run --no-capture-output -n "$TARGET_ENV" \
  python -m pip uninstall -y opencv-python-headless
conda run --no-capture-output -n "$TARGET_ENV" \
  python -m pip install \
  mediapipe==0.10.21 \
  pyrealsense2==2.58.3.10794 \
  scikit-learn==1.5.2 \
  joblib==1.5.3 \
  numpy==1.26.4 \
  pandas==2.2.3 \
  scipy==1.15.3 \
  opencv-contrib-python==4.11.0.86 \
  pytest==8.3.5

echo "[OK] prepared $TARGET_ENV without modifying $SOURCE_ENV."
