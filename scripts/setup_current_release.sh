#!/usr/bin/env bash
# Backward-compatible setup entry; run.sh uses the same setup before launching FR3.
set -euo pipefail
project="$(cd "$(dirname "$0")/.." && pwd)"
options=(--prepare)
while (($#)); do
  case "$1" in
    --replay|--no-recording|--record) options+=("$1"); shift ;;
    --skip-apt) export FR3_SKIP_SYSTEM_INSTALL=true; shift ;;
    --jobs) [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || { echo '需要正整数 --jobs N' >&2; exit 2; }
            export FR3_BUILD_JOBS="$2"; shift 2 ;;
    --assets-from) export FR3_ASSETS_FROM="${2:?需要模型目录}"; shift 2 ;;
    --mano) export FR3_MANO_FILE="${2:?需要 MANO 文件}"; shift 2 ;;
    --cache) export FR3_CACHE_DIR="${2:?需要下载缓存目录}"; shift 2 ;;
    --existing-env-root) export FR3_EXISTING_ENV_ROOT="${2:?需要包含两个视觉环境的目录}"; shift 2 ;;
    -h|--help) echo 'setup_current_release.sh [--replay] [--mano FILE | --assets-from DIR] [--cache DIR] [--jobs N] [--skip-apt] [--existing-env-root DIR]'; exit 0 ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done
exec "$project/run.sh" "${options[@]}"
