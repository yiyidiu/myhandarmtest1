# 整体腕姿映射与四场景固定版 · 2026-09-21

固定标签：**`teleop-whole-wrist-scenes-20260921`**。本版保存当前实际使用的 HaMeR 腕口整体映射、机械臂控制、场景、观察窗口和录制代码。下载时请选择这个标签或同名 Release；根目录 `run.sh` 已更新为本版入口。

## 当前控制配置

- HaMeR 左手模型，以16点腕口配准得到手部姿态；掌部皮肤深度用于位置估计。
- 以 C 零位为参考做整体旋转映射，整体转角统一放大1.3倍；偏航目标限于±88°，实际保护上限±90°。
- 人手小指侧靠近相机对应右偏航，大拇指侧靠近对应左偏航；四指靠近对应向下俯仰，远离对应向上俯仰；转掌对应末端横滚。
- 靠近相机对应向下平移，远离对应向上；画面上方对应向前，下方对应向后。基座 X/Y/Z 平移增益为1.6/2.0/1.2。
- 控制点使用已调整的掌面中心偏移16.58厘米，保留当前约束速度控制、关节和自碰撞保护。
- 控制器不读取 Gazebo 桌子或物体坐标；倒水显示对瓶口和杯口的计算只用于仿真画面及水量记录。
- 四个场景使用同一套感知与运动控制。切换场景不会切换手模型、坐标系或映射比例。

## 下载和恢复

```bash
git clone --branch teleop-whole-wrist-scenes-20260921 --single-branch   https://github.com/yiyidiu/myhandarmtest1.git myhandarmtest1-whole-wrist-scenes-20260921
cd myhandarmtest1-whole-wrist-scenes-20260921
```

GitHub 的 Source code ZIP 也可使用。源码包含程序、场景、机器人网格、倒水显示插件源码和固定依赖恢复入口；编译产物、Python环境、实验录像及授权模型不放进源码包。

目标平台为 **Ubuntu 20.04 x86_64、ROS Noetic、Gazebo 11、X11、支持 CUDA 11.8 的 NVIDIA 显卡，以及 D435/D455 USB3 相机**。新电脑先准备 ROS、显卡驱动与相机权限，再恢复程序依赖：

```bash
bash scripts/setup_current_release.sh --mano /实际路径/MANO_RIGHT.pkl
```

若已有完整的授权 HaMeR `_DATA` 目录，可用 `--assets-from /实际路径/_DATA` 代替 `--mano`，避免重复下载权重。已有系统依赖与两个Python环境时：

```bash
bash scripts/setup_current_release.sh --skip-apt   --existing-env-root "$HOME/anaconda3/envs"   --assets-from /实际路径/_DATA
```

安装入口会生成**当前目录**的 `teleop_machine.env` 并进行 Release 编译，包括新增倒水插件；不要复制别的工程的 `build/`、`devel/` 或本机路径配置。Python环境分卷复用已发布的固定依赖，控制代码使用本标签。依赖及模型来源见 [依赖清单](reproduction/20260921/runtime_reference.json)。MANO文件须由使用者自行取得授权；不在公开仓库再分发。

## 四个场景的命令

在本工程目录内执行任意一条。默认均录制屏幕、RGB-D、手部逐帧数据及机械臂控制数据，**不采集IMU**。

| 场景 | 命令 | 布置 |
|---|---|---|
| 倒水 | `bash scripts/run_pouring_observation.sh` | 右侧台子上的开盖水瓶与空杯；杯外径8厘米、高12厘米 |
| 方块堆叠 | `bash scripts/run_stacking_observation.sh` | 前方台子上的两个18.67×7.6×6.9厘米长方体，中心间距36厘米 |
| 三物体抓取 | `bash scripts/run_grasp_observation.sh` | 左侧绿色圆柱、前方橙色纸盒、右侧水瓶与三个台子 |
| 空场景动作观察 | `bash scripts/run_empty_observation.sh` | 机械臂、三指手、地面；无桌子和物体 |

`bash run.sh` 等同于倒水入口。为避免场景混用，日常使用上表四个独立命令。旧的 `run_grasp_feedback_recording.sh` 等历史入口保留供追溯，不代表本版整体腕姿预设。

需要明确开启IMU时，可以使用共同入口，例如：

```bash
TELEOP_RECORD_IMU=true bash scripts/run_hand_imu_observation.sh --empty-scene
```

## 操作与保存

- 相机窗口按 **C**，用2秒倒计时把手和身体放稳；程序随后取有效稳定帧标定，提示完成后开始控制。
- 有物体的场景打开正面、侧面两个观察窗，显示频率5 Hz。滚轮可放大，Esc关闭观察窗。
- 松开物体并移开机械手后，在观察窗按 **T**，恢复物体原位置。倒水场景同时恢复瓶内94毫升虚拟水量和空杯；不重置机械臂零位。
- 相机窗口按 **Q** 结束，等待数据保存完毕。

每次录制保存到本工程 `.runtime/hand_imu_observation_日期时间_随机码/`，启动时打印完整路径。目录包含 `screen.mp4`、`rgbd.db3`、`control_chain.bag`、实际场景快照 `scene.world`、手部数据与时间对应信息。倒水时同时记录瓶内、杯内、洒出水量。未启用IMU时不会要求接收器在线。

## 倒水显示范围

可看到瓶口水流及杯内水面；进入杯口才增加杯内水量，倒偏计入洒出量，摆正停止出水。水流和水量是简化的可视化计算，不是完整流体动力学：不模拟晃动、液体力或随水量变化的刚体载荷。它用于练习抓瓶、对准和倾倒，不用于证明真实液体控制精度。

## 固定版本边界

本版将运行入口、实际控制源码、模型接口、场景及依赖恢复信息一起保存；旧9月17日等标签继续保留。新开发应另建目录或分支，不移动本标签。相同代码不代表不同硬件上的帧率和延迟必然相同。保存与验证记录见 [版本验证](reproduction/20260921/VALIDATION.md)。
