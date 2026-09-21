#!/usr/bin/env bash
# Frozen 2026-09-21 whole-wrist preset. Default scene: pouring, with recording.
set -Eeuo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$release_root/scripts/run_pouring_observation.sh" "$@"
