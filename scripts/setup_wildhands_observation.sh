#!/usr/bin/env bash
# Isolated offline experiment; reuses but never modifies the existing Torch env.
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
experiment_dir="$project_root/.runtime/wildhands_evaluation_20260920"
base_python="${TELEOP_HAMER_PYTHON:-/home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python}"
mano_archive="${MANO_SOURCE_ZIP:-/home/dongtian/下载/mano_v1_2.zip}"
vendor="$project_root/third_party/wildhands_demo"
mkdir -p "$experiment_dir/assets/mano"
if [[ ! -d "$vendor/.git" ]]; then
  git clone --depth 1 --branch demo --single-branch https://github.com/ap229997/hands.git "$vendor"
fi
if [[ "$(git -C "$vendor" rev-parse HEAD)" != f99dfea0d1fce970aed2d31d1018eda280e05f47 ]]; then
  echo '作者代码版本不同，请在独立实验目录核对后使用，避免覆盖已有修改。' >&2
  exit 2
fi
if [[ ! -x "$experiment_dir/venv/bin/python" ]]; then
  "$base_python" -m venv --system-site-packages "$experiment_dir/venv"
fi
experiment_python="$experiment_dir/venv/bin/python"
"$experiment_python" -m pip install 'loguru==0.7.3' 'easydict==1.13' 'fvcore==0.1.5.post20221221' 'iopath==0.1.10'
if ! "$experiment_python" -c 'import pytorch3d' >/dev/null 2>&1; then
  MAX_JOBS=2 "$experiment_python" -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7'
fi
"$experiment_python" - "$mano_archive" "$experiment_dir/assets/mano" <<'PY'
from pathlib import Path
import sys,zipfile
source=Path(sys.argv[1]);out=Path(sys.argv[2])
if not source.is_file():
    raise SystemExit('需要已有授权的MANO压缩包，请用MANO_SOURCE_ZIP指定路径。')
with zipfile.ZipFile(source) as archive:
    for name in ['MANO_LEFT.pkl','MANO_RIGHT.pkl']:
        target=out/name
        if target.exists():continue
        matches=[n for n in archive.namelist() if n.endswith('/'+name) and not n.startswith('__MACOSX/')]
        if len(matches)!=1:raise SystemExit('压缩包缺少唯一的模型文件：'+name)
        target.write_bytes(archive.read(matches[0]))
PY
if [[ ! -f "$experiment_dir/assets/wildhands.ckpt" ]]; then
  "$experiment_python" -m gdown 1FJWBrMmTKjKAo6j5DQS1KYpqqFbAbJ9Q -O "$experiment_dir/assets/wildhands.ckpt"
fi
echo '离线实验依赖和作者权重已就绪；此脚本不启动相机或机械臂。'
