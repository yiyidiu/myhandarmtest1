# RGB-D 姿态修正试验基线（2026-09-12）

## 固定版本

- 仓库：https://github.com/yiyidiu/myhandarmtest1
- 开发分支：`codex/imu-guided-pose-measurement-20260912`
- 固定基线标签：`teleop-rgbd-pose-baseline-20260912`（包含本说明和更新后的 README）
- 截图对应的功能代码提交：`2a672ae1a7803c3b455092e052cb15b825defef3`
- 对应原始代码标签：`teleop-imu-guided-pose-trial-20260912`
- 修正前回退标签：`teleop-c-source-continuity-20260912`，提交 `587ae6c`

本次保存只补充版本文档，不修改上述功能代码，不包含尚未实施的下一轮改进。固定标签用于恢复；后续修改在新分支提交，不移动或覆盖此标签。

## 版本实际状态

本版通过同帧 RGB-D 掌面几何辅助修正 HaMeR 姿态，参数来自既有 IMU 录像的离线校准。IMU 不参与运行时控制。专用入口启用修正并关闭叠加的视角偏置；通用入口仍默认关闭修正。

已有离线对照显示部分左移误转减少、主动横滚幅度改善，但左移剩余误差仍大，静止抖动未解决，部分动作略有退化。仅比较过原控制用手和 640×480 配置，未验证跨人员、跨配置泛化。本次归档未运行相机、Gazebo 或真人验收；代码保存和上传不能视为运动性能验证。

具体数据与证据边界见 [已有对照记录](../imu_guided_pose_measurement_20260912.md)。该记录中的 `.runtime` 绝对路径只在原机器可用。

## 保存范围与运行依赖

Git 保存源码、配置（含 `perception_hamer/config/imu_guided_palm_20260912.json`）、启动脚本、已有测试与文档，保留祖先提交历史。

按现有 `.gitignore`，录像、RGB-D/ROS 原始记录、`.runtime` 本地备份、模型资产、Python 环境和编译产物不在此次 Git 上传中，原机文件保留。GitHub 上的源码快照不是包含模型与录像的整机备份。

运行需要 Ubuntu 20.04、ROS Noetic、Gazebo、已配置的相机与 HaMeR 模型/Python 环境。入口默认使用原机器 `/home/dongtian/anaconda3/envs/mediapipe_env/bin/python` 记录 IMU，可用 `TELEOP_IMU_PYTHON` 指定已配置的其他解释器。环境配置参考仓库既有安装说明。

首次检出或切换版本后，在仓库根目录重新编译：

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release -j3 -l3
source devel/setup.bash
```

## 截图对应的启动与录制

在原机器的已配置目录执行：

```bash
cd /home/dongtian/myhandarmtest1-teleop-repair
TELEOP_MOTION_CONTROLLER=constrained ./scripts/run_imu_calibrated_hand_observation.sh
```

入口打开空场景、手部窗口和腕口坐标轴，并启动录屏、RGB-D、控制数据和 IMU 参考记录。先打开 IMU，核对终端实际接收状态；手张开放稳后按 `C` 标定，再进行动作，结束时在相机窗口按 `Q` 并等待保存结束。

结果位于 `.runtime/hand_motion_observation_日期时间_随机码/`，录像为其中的 `screen.mp4`。查询本次目录：

```bash
cat .runtime/latest_hand_motion_observation.txt
```

## 恢复与后续开发

先退出已有录制、相机与 Gazebo。确保工作区没有未保存改动，再获取并检出固定标签：

```bash
git fetch origin --tags
git switch --detach teleop-rgbd-pose-baseline-20260912
```

如需恢复修正前的源码：

```bash
git switch --detach teleop-c-source-continuity-20260912
```

切换后按前述步骤重新编译。Git 如因本地修改拒绝切换，先在独立分支提交需要保留的修改，不使用强制覆盖。

从固定基线开始下一轮开发（分支名已存在时换一个新名字）：

```bash
git switch -c codex/rgbd-pose-next teleop-rgbd-pose-baseline-20260912
```

只想关闭修正进行对照，无需切换源码；在当前基线运行：

```bash
TELEOP_MOTION_CONTROLLER=constrained TELEOP_IMU_CALIBRATED_POSE=false TELEOP_POSITIVE_VIEW_BIAS_GAIN=0 ./scripts/run_hand_motion_observation_recording.sh
```
