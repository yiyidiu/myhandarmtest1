# HandArm：D455 + HaMeR + ABB IRB120 三指手遥操作

这是一个 ROS Noetic / MoveIt 1 / Gazebo 11 工作空间，用于把 D455 观测到的人手相对六维运动映射到 ABB IRB120 与自研三指手。当前仓库以仿真、安全门禁和可重复验收为主；实体机器人输出必须经过单独授权、标定和低速安全检查。

## 当前状态

- 6 个 Catkin 包可以完整构建。
- D455 对齐 RGB-D、HaMeR/MANO 低频观测、RGB-D 刚体增量、因果姿态滤波、UDP/ROS 适配、MoveIt Servo/EGM 仿真链已经形成。
- 三指手包含仿真控制、抓放基线、轨迹安全代理和接触相关场景。
- 全机器人自碰撞链按失败关闭原则设计，包含预测速度门和手轨迹采样检查。
- 2026-08-25 的清理后回归基线为 269 项测试（感知 31、仿真 120、遥操作/Catkin 118），0 失败。

详细时间线见 [近 10 天项目节点与里程碑](docs/PROJECT_MILESTONES_20260816_20260825.md)。

## 目录

```text
src/handarmtest1/             机器人 URDF/Xacro、网格和基础控制配置
src/abb120_moveit_config1/    MoveIt、Servo 和控制器配置
src/handarm_moveit_demo/      遥操作、映射、安全门和验收节点
src/handarm_sim_demo/         Gazebo 场景、三指手、抓放和仿真安全逻辑
src/abb_resources/            ABB 公共描述资源（第三方）
src/roboticsgroup_.../        Gazebo mimic/disable-link 插件（第三方，含本地修订）
perception_hamer/             D455、HaMeR/MANO、RGB-D 跟踪与离线测试
scripts/                      构建、演示和验收入口
docs/                         审计、报告、运行说明和里程碑
```

## 环境

- Ubuntu 20.04
- ROS Noetic
- Gazebo 11
- MoveIt 1 / MoveIt Servo
- Python 3.8
- Intel RealSense D455
- NVIDIA RTX 2060（HaMeR 路线按低显存约束设计）

模型、数据集、录制视频、Rosbag、构建目录和备份不会进入 Git。它们仍保留在本机，并由 `.gitignore` 排除。HaMeR checkpoint 默认放在：

```text
perception_hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
```

## 构建与自动测试

在任意当前目录都可以调用入口脚本：

```bash
cd /path/to/myhandarmtest1
./scripts/run_stage1_tests.sh
```

脚本会构建 Catkin 工作空间，运行 `handarm_moveit_demo` 和 `perception_hamer` 的离线测试，并汇总 Catkin 测试结果。若 ROS Noetic 不在默认位置，可先设置：

```bash
export ROS_SETUP_FILE=/your/ros/noetic/setup.bash
```

## 安全仿真入口

```bash
./scripts/run_stage1_safe_demo.sh
```

这个入口默认关闭实体输出门。真人 D455 + Gazebo 的完整步骤见 [现场遥操作验收](docs/17_LIVE_HUMAN_GAZEBO_TELEOP_ACCEPTANCE.md)，地面工作空间映射见 [Ground Workspace](src/handarm_moveit_demo/README_GROUND_WORKSPACE.md)。

## 本机环境覆盖

感知脚本默认寻找当前用户的 `anaconda3/envs/mediapipe_env`。其他安装位置可以显式设置：

```bash
export MEDIAPIPE_ENV_PREFIX=/path/to/mediapipe_env
export MEDIAPIPE_PYTHON=/path/to/mediapipe_env/bin/python
export REALSENSE_SITE_PACKAGES=/path/to/site-packages
```

## 已知边界

- 当前目录最初没有 Git 历史，因此 2026-08-16 至 2026-08-25 的时间线来自验收报告、结果 JSON、备份说明和文件时间，不能等同于历史提交记录。
- D455 USB3 正式认证、真实三指手应变片标定、实体 ABB 授权与低速验收仍未完成，不得写成 `PASS`。
- Gazebo Classic 11 已进入上游弃用阶段；本项目仍按 Ubuntu 20.04 / ROS Noetic 的既定环境维护。

## GitHub

GitHub 新手的日常命令、安全规则和首次上传步骤见 [GitHub 新手手册](docs/GITHUB_BEGINNER_GUIDE_ZH.md)。

## 许可证

本项目自有代码按各 ROS 包声明的许可证使用。`abb_resources` 和 `roboticsgroup_upatras_gazebo_plugins` 是第三方组件，以各自目录中的许可证和版权声明为准；HaMeR 相关第三方说明见 `perception_hamer/THIRD_PARTY_NOTICES.md`。
