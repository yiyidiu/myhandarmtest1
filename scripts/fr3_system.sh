#!/usr/bin/env bash
# Install the native Ubuntu dependencies only if the requested mode is missing them.
set -euo pipefail
project="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:?live or replay}"
record="${2:?true or false}"
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 20.04 || "$(uname -m)" != x86_64 ]]; then
  echo "原生版本运行于 Ubuntu 20.04 x86_64。系统检测到 ${PRETTY_NAME:-$ID} / $(uname -m)。" >&2
  exit 2
fi
pkgs=(build-essential cmake pkg-config curl ca-certificates libfcl-dev liburdfdom-dev libgazebo11-dev
      qtbase5-dev python3-opencv python3-numpy python3-scipy python3-yaml python3-rospkg python3-catkin-pkg
      ros-noetic-catkin ros-noetic-gazebo-ros-control ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-plugins
      ros-noetic-moveit ros-noetic-moveit-servo ros-noetic-trac-ik-kinematics-plugin
      ros-noetic-ros-controllers ros-noetic-controller-manager ros-noetic-robot-state-publisher
      ros-noetic-dynamic-reconfigure ros-noetic-xacro ros-noetic-urdfdom-py ros-noetic-rosbag
      ros-noetic-cv-bridge ros-noetic-tf2-geometry-msgs)
if [[ "$record" == true || "$mode" == live ]]; then
  pkgs+=(python3-gi gir1.2-gtk-3.0 gir1.2-wnck-3.0 fonts-noto-cjk x11-utils x11-xserver-utils)
fi
if [[ "$record" == true ]]; then
  pkgs+=(gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gstreamer1.0-tools
         gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-x)
  command -v gst-inspect-1.0 >/dev/null && gst-inspect-1.0 x264enc >/dev/null 2>&1 || pkgs+=(gstreamer1.0-plugins-ugly)
  command -v gst-inspect-1.0 >/dev/null && gst-inspect-1.0 avdec_h264 >/dev/null 2>&1 || pkgs+=(gstreamer1.0-libav)
fi
if [[ "$mode" == live ]]; then pkgs+=(libusb-1.0-0 libgl1 libegl1); fi
missing=()
declare -A installed=()
while read -r package state; do
  if [[ "$state" == installed ]]; then installed["$package"]=1; fi
done < <(dpkg-query -W -f='${Package} ${db:Status-Status}\n' "${pkgs[@]}" 2>/dev/null || true)
for package in "${pkgs[@]}"; do
  [[ -v "installed[$package]" ]] || missing+=("$package")
done
root_cmd() {
  if (( EUID == 0 )); then "$@"; else sudo "$@"; fi
}
if (( ${#missing[@]} )); then
  echo "首次准备将通过 Ubuntu 软件包管理器安装 ${#missing[@]} 个所需软件包。"
  # Install the release key for the signed-by ROS source created below.
  root_cmd install -m 0644 "$project/reproduction/fr3/ros-archive-keyring.gpg" /usr/share/keyrings/fr3-ros-archive.gpg
  if ! grep -Eqhs '^[^#].*(packages\.ros\.org/ros/ubuntu|/ros/ubuntu|/noetic/final/ubuntu).*focal' /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; then
    printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/fr3-ros-archive.gpg] http://packages.ros.org/ros/ubuntu focal main' |
      root_cmd tee /etc/apt/sources.list.d/fr3-noetic.list >/dev/null
  fi
  root_cmd apt-get -o Acquire::Retries=3 update
  if ! apt-cache show libgazebo11-dev >/dev/null 2>&1; then
    printf '%s\n' 'deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://archive.ubuntu.com/ubuntu focal universe multiverse' |
      root_cmd tee /etc/apt/sources.list.d/fr3-focal-universe.list >/dev/null
    root_cmd apt-get -o Acquire::Retries=3 update
  fi
  root_cmd env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install -y --no-install-recommends "${missing[@]}"
fi
if [[ "$mode" == live && -S /run/udev/control && ! -f /etc/udev/rules.d/70-fr3-d455.rules ]]; then
  accessible=false
  for path in /sys/bus/usb/devices/*; do
    [[ -r "$path/idProduct" && -r "$path/idVendor" ]] || continue
    [[ "$(<"$path/idVendor")" == 8086 && "$(<"$path/idProduct")" == 0b5c ]] || continue
    device=$(printf '/dev/bus/usb/%03d/%03d' "$(<"$path/busnum")" "$(<"$path/devnum")")
    if [[ -r "$device" && -w "$device" ]]; then accessible=true; break; fi
  done
  if [[ "$accessible" == false ]]; then
    root_cmd install -m 0644 "$project/reproduction/fr3/70-fr3-d455.rules" /etc/udev/rules.d/70-fr3-d455.rules
    root_cmd udevadm control --reload-rules
    root_cmd udevadm trigger --subsystem-match=usb --attr-match=idVendor=8086 --attr-match=idProduct=0b5c
  fi
fi
