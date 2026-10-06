#!/usr/bin/env bash
# Default to the pouring scene. Empty-scene experiments must request it
# explicitly, so a terminal's old TELEOP_EMPTY_SCENE cannot hide the props.
set -Eeuo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
observation_scene=pouring
case "${1:-}" in
  --pouring-scene) shift ;;
  --empty-scene) observation_scene=empty; shift ;;
  --stacking-scene) observation_scene=stacking; shift ;;
  --grasp-scene) observation_scene=grasp; shift ;;
esac
if (( $# > 1 )) || { [[ "${1:-}" == --* ]] && [[ "${1:-}" != --no-recording ]]; }; then
  echo "用法：$0 [--pouring-scene|--stacking-scene|--grasp-scene|--empty-scene] [新的录制文件夹|--no-recording]（默认倒水场景并录屏）" >&2
  exit 2
fi
export TELEOP_RECORD_IMU="${TELEOP_RECORD_IMU:-false}"
case "$TELEOP_RECORD_IMU" in true|false) ;; *) echo 'TELEOP_RECORD_IMU 只能是 true 或 false。' >&2; exit 2 ;; esac
export TELEOP_REQUIRE_IMU="$TELEOP_RECORD_IMU"
export TELEOP_MOTION_CONTROLLER=constrained
export TELEOP_ARM_CONTROL_MODE=velocity
export TELEOP_PURE_VIEW_CORRECTION=true
export TELEOP_EMPTY_SCENE=false
export TELEOP_ALIGNMENT_VIEWS=true
case "$observation_scene" in
  pouring) observation_world=handarm_pouring.world ;;
  stacking) observation_world=handarm_front_stacking.world ;;
  grasp) observation_world=handarm_ground_grasp.world ;;
  empty)
  observation_world=handarm_motion_only.world
  export TELEOP_EMPTY_SCENE=true
  export TELEOP_ALIGNMENT_VIEWS=false
  ;;
esac
export TELEOP_GRASP_WORLD="$script_directory/../src/handarm_sim_demo/worlds/$observation_world"
if [[ "${TELEOP_GRASP_WORLD##*/}" == handarm_pouring.world &&
      ! -f "$script_directory/../devel/lib/libhandarm_pouring_visualization_plugin.so" ]]; then
  echo '倒水显示插件尚未编译，请先在工程目录运行：' >&2
  echo 'source /opt/ros/noetic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release' >&2
  exit 2
fi
export TELEOP_RECORDING_PREFIX=hand_imu_observation
export TELEOP_IMU_MOUNTING=left_hand_x_distal_y_ulnar_z_dorsal
export TELEOP_CONTROL_REFERENCE=mano-wrist-ring
export TELEOP_WRIST_DEPTH_MODE=palm-skin
case "$TELEOP_CONTROL_REFERENCE" in
  mano-wrist-ring) echo '姿态：原始HaMeR＋纯视角几何修正＋16点腕口配准；五点仅用于掌面深度。' ;;
  mano-five-point-palm) echo '姿态：腕点与四指根五点掌骨法，作为对照。' ;;
  *) echo '此观察入口只支持 mano-wrist-ring 或 mano-five-point-palm。' >&2; exit 2 ;;
esac
echo '两法深度：同一掌部皮肤区域的相机实测Z，不加骨内或网格偏移，不补旧深度。'
if [[ "$TELEOP_EMPTY_SCENE" == true ]]; then
  echo '空场景姿态验证：机械臂、三指手与地面；桌子和物体关闭。'
else
  case "${TELEOP_GRASP_WORLD##*/}" in
  handarm_pouring.world)
    printf '%s\n' \
      '倒水场景：右侧台子放置开盖水瓶及空杯，中心相距22cm；杯外径8cm、杯口内径7.2cm、高12cm。' \
      '水瓶初始水量94ml，倾倒后显示水流；进入杯口才增加杯内水量，倒偏计入洒出量。' \
      '这是简化水流与水量显示，不模拟液体晃动或液体力；物体载荷仍沿用原刚体设置。' ;;
  handarm_front_stacking.world)
    echo '堆叠场景：前方两个长18.67cm、宽7.6cm、高6.9cm的长方体，中心相距36cm。' ;;
  handarm_ground_grasp.world)
    echo '抓取场景：三个台子，左侧绿色圆柱、前方橙色纸盒、右侧矿泉水瓶。' ;;
  *) echo "加载场景：$TELEOP_GRASP_WORLD" ;;
  esac
  if [[ "$TELEOP_ALIGNMENT_VIEWS" == true ]]; then
    echo '双观察窗为5 Hz。先松开并移开机械手，点击任一观察小窗后按T，恢复物体及倒水初始水量；录像继续。'
  fi
fi
if [[ "${1:-}" != --no-recording ]]; then
  echo '同时保存屏幕录像、原始RGB-D、手部逐帧数据与机械臂控制数据；保存路径将在启动时打印。'
fi
if [[ "$TELEOP_RECORD_IMU" == true ]]; then
  printf '%s\n' \
    '同时采集IMU。佩戴：左手背，+X朝中指指尖，+Y朝小指侧，+Z朝手背外；全程保持固定。' \
    '相机就绪后，先不要按C：静止10秒，再做几次小幅转动—停顿—反向，约15秒。' \
    '随后按一次C，在2秒倒计时内把身体和张开的手放稳；提示标定完成后再运动。' \
    '结束前再做一次短转动并静止5秒，按Q结束，等待保存完成。' \
    '原始源帧时间、接收批次和时钟关系都会保留；录后估计时间偏移，不代表硬件同步。'
else
  printf '%s\n' \
    '本次不采集IMU，不需要开启或佩戴IMU，也不等待IMU接收。' \
    '相机就绪后按一次C，在2秒倒计时内把身体和张开的手放稳；提示标定完成后开始控制。' \
    '按Q结束；启用录制时等待数据保存完成。'
fi
exec bash "$script_directory/run_whole_wrist_teleop.sh" "$@"
