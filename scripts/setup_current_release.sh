#!/usr/bin/env bash
set -Eeuo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
skip_apt=false
jobs=3
runtime_args=()
while (($#)); do
  case "$1" in
    --skip-apt) skip_apt=true; shift ;;
    --jobs) jobs="${2:?--jobs requires an integer}"; shift 2 ;;
    --assets-from|--mano|--existing-env-root|--cache)
      runtime_args+=("$1" "${2:?missing argument}"); shift 2 ;;
    -h|--help)
      echo 'Usage: bash scripts/setup_current_release.sh --assets-from /path/to/_DATA [--skip-apt]'
      echo 'Or:    bash scripts/setup_current_release.sh --mano /path/to/MANO_RIGHT.pkl [--skip-apt]'
      echo 'Optional: --existing-env-root /path/to/envs --cache /download/cache --jobs 3'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo '--jobs must be positive' >&2; exit 2; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 20.04 && "$(uname -m)" == x86_64 ]] || {
  echo 'This release targets Ubuntu 20.04 x86_64.' >&2; exit 2;
}
[[ -f /opt/ros/noetic/setup.bash ]] || {
  echo 'ROS Noetic desktop-full is required. See README for system prerequisites.' >&2; exit 2;
}
if [[ "$skip_apt" != true ]]; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git curl ca-certificates \
    python3-catkin-pkg python3-numpy python3-scipy python3-opencv python3-yaml \
    python3-rosdep python3-rospkg python3-setuptools python3-nose \
    ros-noetic-catkin ros-noetic-control-toolbox ros-noetic-gazebo-ros-control \
    ros-noetic-gazebo-ros-pkgs ros-noetic-joint-state-publisher-gui \
    ros-noetic-moveit ros-noetic-moveit-servo ros-noetic-trac-ik-kinematics-plugin \
    ros-noetic-ros-controllers ros-noetic-tf2-geometry-msgs ros-noetic-urdfdom-py ros-noetic-xacro
fi
/usr/bin/python3 "$release_root/scripts/restore_release_runtime.py" "${runtime_args[@]}"
# A previously sourced checkout must not supply this checkout's binaries.
env -u CMAKE_PREFIX_PATH -u ROS_PACKAGE_PATH -u PYTHONPATH -u LD_LIBRARY_PATH \
  bash --noprofile --norc -c '
    set -Ee
    source /opt/ros/noetic/setup.bash
    cd "$1"
    if [[ ! -e src/CMakeLists.txt ]]; then catkin_init_workspace src; fi
    catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -j"$2" -l"$2"
  ' _ "$release_root" "$jobs"
echo 'Setup complete. Run: bash run.sh'
