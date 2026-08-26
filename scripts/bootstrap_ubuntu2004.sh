#!/usr/bin/env bash
set -Eeuo pipefail

# One-command bootstrap for the reproducible, hardware-safe project baseline.
# It intentionally does not install NVIDIA drivers or redistribute HaMeR/MANO
# model assets; see docs/UBUNTU2004_FAST_REPRODUCTION_ZH.md for that optional
# hardware layer.

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly ROS_SETUP_FILE="${ROS_SETUP_FILE:-/opt/ros/noetic/setup.bash}"

SKIP_APT=false
SKIP_TESTS=false
JOBS="${REPRO_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

usage() {
  cat <<'EOF'
用法：
  ./scripts/bootstrap_ubuntu2004.sh [选项]

不加选项（推荐）：安装依赖、编译六个 Catkin 包、运行 439 项当前测试。

选项：
  --skip-apt       不安装系统依赖；仅用于依赖已经装好的机器
  --skip-tests     安装并编译，但跳过测试
  --jobs N         编译并行数，默认使用全部 CPU 核心
  -h, --help       显示本帮助
EOF
}

while (($#)); do
  case "$1" in
    --skip-apt)
      SKIP_APT=true
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=true
      shift
      ;;
    --jobs)
      if (($# < 2)); then
        echo "[错误] --jobs 后面需要一个正整数。" >&2
        exit 2
      fi
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[错误] 未知选项：$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[错误] 并行数必须是正整数，当前值：$JOBS" >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/.runtime"
readonly LOG_FILE="$PROJECT_ROOT/.runtime/bootstrap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local status=$?
  local line=${BASH_LINENO[0]:-unknown}
  echo
  echo "[失败] 第 $line 行执行失败（退出码 $status）。"
  echo "[失败] 完整日志：$LOG_FILE"
  exit "$status"
}
trap on_error ERR

echo "[1/6] 检查操作系统和工作空间"
if [[ ! -r /etc/os-release ]]; then
  echo "[错误] 无法读取 /etc/os-release。" >&2
  exit 2
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "20.04" ]]; then
  echo "[错误] 本版本只验收 Ubuntu 20.04；当前为 ${PRETTY_NAME:-未知系统}。" >&2
  exit 2
fi
if [[ "$(dpkg --print-architecture)" != "amd64" ]]; then
  echo "[错误] 本复现包只验收 amd64/x86_64。" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/src/handarm_moveit_demo/package.xml" ]]; then
  echo "[错误] 压缩包目录不完整：找不到 handarm_moveit_demo/package.xml。" >&2
  exit 2
fi
if [[ ! -f "$ROS_SETUP_FILE" ]]; then
  echo "[错误] 找不到 ROS Noetic：$ROS_SETUP_FILE" >&2
  echo "请先确认新电脑安装的是 ROS Noetic desktop-full。" >&2
  exit 2
fi

available_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')"
if [[ "$available_kib" =~ ^[0-9]+$ ]] && ((available_kib < 6 * 1024 * 1024)); then
  echo "[警告] 当前可用空间少于 6 GiB，安装依赖或编译可能失败。"
fi

if [[ "$SKIP_APT" == false ]]; then
  echo "[2/6] 安装 Ubuntu/ROS 依赖（sudo 可能要求输入新电脑的登录密码）"
  if ((EUID == 0)); then
    SUDO=()
  else
    if ! command -v sudo >/dev/null 2>&1; then
      echo "[错误] 找不到 sudo；请用具有 sudo 权限的普通用户运行。" >&2
      exit 2
    fi
    sudo -v
    SUDO=(sudo)
  fi

  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    git \
    python3-catkin-pkg \
    python3-nose \
    python3-numpy \
    python3-opencv \
    python3-pip \
    python3-rosdep \
    python3-rospkg \
    python3-setuptools \
    python3-yaml \
    ros-noetic-catkin \
    ros-noetic-control-toolbox \
    ros-noetic-gazebo-ros-control \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-moveit \
    ros-noetic-moveit-servo \
    ros-noetic-ros-controllers \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-urdfdom-py \
    ros-noetic-xacro

  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    "${SUDO[@]}" rosdep init
  fi
  rosdep update --rosdistro=noetic
  # Package manifests remain the source of truth; the explicit list above
  # covers the common Noetic desktop variants and speeds up first-time setup.
  rosdep install --from-paths "$PROJECT_ROOT/src" \
    --ignore-src --rosdistro=noetic -r -y
else
  echo "[2/6] 已按参数跳过系统依赖安装"
fi

echo "[3/6] 验证 ROS Noetic 与依赖"
# shellcheck disable=SC1090
source "$ROS_SETUP_FILE"
if [[ "${ROS_DISTRO:-}" != "noetic" ]]; then
  echo "[错误] 当前 ROS_DISTRO 不是 noetic：${ROS_DISTRO:-未设置}" >&2
  exit 2
fi
rosdep check --from-paths "$PROJECT_ROOT/src" \
  --ignore-src --rosdistro=noetic

cd "$PROJECT_ROOT"
if [[ ! -e src/CMakeLists.txt ]]; then
  catkin_init_workspace src
fi
export ROS_PARALLEL_JOBS="-j${JOBS} -l${JOBS}"

if [[ "$SKIP_TESTS" == false ]]; then
  echo "[4/6] 编译并运行 439 项当前测试"
  "$PROJECT_ROOT/scripts/run_stage1_tests.sh"
else
  echo "[4/6] 编译六个 Catkin 包（已按参数跳过测试）"
  catkin_make -DCMAKE_BUILD_TYPE=Release
fi

echo "[5/6] 验证六个 Catkin 包均可发现"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/devel/setup.bash"
packages=(
  abb120_moveit_config1
  abb_resources
  handarm_moveit_demo
  handarm_sim_demo
  handarmtest1
  roboticsgroup_upatras_gazebo_plugins
)
for package in "${packages[@]}"; do
  rospack find "$package" >/dev/null
  echo "  [OK] $package"
done

echo "[6/6] 复现完成"
echo "[完成] Ubuntu 20.04 + ROS Noetic 安全基线已经可用。"
echo "新终端先执行：source \"$PROJECT_ROOT/devel/setup.bash\""
echo "安全仿真入口：$PROJECT_ROOT/scripts/run_stage1_safe_demo.sh"
echo "安装日志：$LOG_FILE"
echo
echo "注意：脚本不会启用实体机器人输出，也不包含 HaMeR/MANO 授权模型。"
echo "相机/HaMeR 第二阶段见：docs/UBUNTU2004_FAST_REPRODUCTION_ZH.md"
