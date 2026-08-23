#!/usr/bin/env bash
set -euo pipefail

# Materialize the user-supplied, frozen Teleoperation Core V8.3 package in a
# content-addressed runtime directory.  The archive is treated as the source
# of truth: none of its Python/model files are rewritten by this project.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${TELEOP_CORE_ARCHIVE:-/home/diu/teleoperation_ubuntu_core.tar.gz}"
EXPECTED_SHA256="87fa1fd27adb67a07e7aaf97509837e49a831260434fa0d2bfe62d61bb783bc9"
RUNTIME_BASE="${TELEOP_CORE_RUNTIME_BASE:-$PROJECT_ROOT/.runtime/teleoperation_core_v83}"
RUNTIME_PARENT="$RUNTIME_BASE/$EXPECTED_SHA256"
RUNTIME_ROOT="$RUNTIME_PARENT/teleoperation_ubuntu_core"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "[ERROR] V8.3 archive not found: $ARCHIVE" >&2
  exit 2
fi

ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "[ERROR] archive SHA-256 differs from the audited reference." >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $ACTUAL_SHA256" >&2
  echo "Set TELEOP_CORE_ARCHIVE only to the audited archive." >&2
  exit 3
fi

MODEL_REL="data/crossuser_models/V83_MULTISCALE_NET_TWO_PERSON_20260728/multiscale_intent_v83.joblib"
POSE_REL="hamer-win/live_v83_pose_observer_windows.py"
if [[ ! -f "$RUNTIME_ROOT/$MODEL_REL" || ! -f "$RUNTIME_ROOT/$POSE_REL" ]]; then
  if [[ -e "$RUNTIME_ROOT" ]]; then
    echo "[ERROR] incomplete runtime already exists: $RUNTIME_ROOT" >&2
    echo "Move that generated directory aside, then run this command again." >&2
    exit 4
  fi
  mkdir -p "$RUNTIME_PARENT"
  STAGING="$(mktemp -d "$RUNTIME_BASE/.extract.XXXXXX")"
  echo "[INFO] extracting audited V8.3 runtime from $ARCHIVE" >&2
  tar -xzf "$ARCHIVE" -C "$STAGING"
  if [[ ! -d "$STAGING/teleoperation_ubuntu_core" ]]; then
    echo "[ERROR] archive top-level directory is invalid: $STAGING" >&2
    exit 5
  fi
  mv "$STAGING/teleoperation_ubuntu_core" "$RUNTIME_ROOT"
  rmdir "$STAGING"
fi

if [[ "${1:-}" != "--print-root" ]]; then
  echo "[OK] audited V8.3 runtime: $RUNTIME_ROOT" >&2
fi
printf '%s\n' "$RUNTIME_ROOT"
