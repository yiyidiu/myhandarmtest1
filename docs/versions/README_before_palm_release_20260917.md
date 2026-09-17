# 单 RGB-D 手部遥操作：当前开发版与固定保存版

2026-09-14：当前控制实现为 `dac3518`（9 月 13 日延迟、运动时钟和手臂自碰撞修复之后）。本机的固定运行副本保存在 `/home/dongtian/teleop_versions/20260914_dac3518/`，包含独立源码、编译程序、模型和两个 Python 环境：

```bash
bash /home/dongtian/teleop_versions/20260914_dac3518/run.sh
```

这是固定版本保存，**不表示全部遥操作问题均已验收**。本机固定副本不随开发目录改动。其完整恢复说明和系统依赖边界见该目录的 `使用与恢复.md`。该副本及模型资产不在 GitHub 源码 ZIP 内。

继续运行开发目录（后续修改会生效）：

```bash
bash /home/dongtian/myhandarmtest1-teleop-repair/scripts/run_grasp_feedback_recording.sh
```

只查看真人遥操作效果、不录屏或采集回放数据时，在上述命令末尾加 `--no-recording`。该选项保留掌面辅助、相机设置和两个抓取视角；只保留普通运行日志，不启动录屏、IMU、rosbag、RGB-D 或逐帧诊断记录。

2026-09-14 本轮开发分支 `codex/teleop-without-scene-truth-20260914`：从固定版对应源码 `da8c63d` 出发，停止将 Gazebo 的桌子、物体和地面坐标导入控制器。正式入口强制 `use_simulated_environment_for_control:=false`；Gazebo 外形、物理接触、自碰撞检查、关节限制及既有运动参数保持原样。没有环境感知或标定输入，因此此分支不提供桌面/物体的提前避碰。该参数的 `true` 仅用于复现旧的离线对照，不是当前真人遥操作条件。

后台检查确认控制器环境为空而仿真物体仍在；机器人阶段的小幅上移、侧移可执行，原有手指与前臂的自碰撞姿态仍被拒绝。这不是完整真人跟随或抓取验收。上述本机固定副本与 GitHub 已发布版本未修改。

当前开发分支 `codex/servo-qp-comparison-20260914` 同时提供 QP 和 MoveIt Servo；默认保留 QP。三份录制输入的最终原时钟回放中，QP 的位置、姿态误差和累计停滞均更小；Servo 采用固定 15 mm 预停配置。两者在最终回放中均未触发自碰撞锁停，但精细控制仍未全部达标。详细条件、局限与数据见 [Servo / QP 对照报告](docs/SERVO_QP_COMPARISON_20260914.md)。

以下是历史说明，旧标签和旧入口不代表上述当前开发版。

## 2026-09-12 姿态修正试验基线（历史）

当前固定基线：`teleop-rgbd-pose-baseline-20260912`，开发分支：`codex/imu-guided-pose-measurement-20260912`。

保存的是 RGB-D 姿态修正试验版。左移误转和静止抖动仍未解决，不能视为精细遥操作已验收。IMU 用于离线校准和运行时参考记录，不参与运行时控制；修正功能由专用入口显式开启。

在已配置依赖并编译的本机运行截图对应版本：

```bash
cd /home/dongtian/myhandarmtest1-teleop-repair
TELEOP_MOTION_CONTROLLER=constrained ./scripts/run_imu_calibrated_hand_observation.sh
```

完整的保存范围、依赖边界、编译、录制及回退步骤见 [基线说明](docs/versions/teleop-rgbd-pose-baseline-20260912.md)。离线对照结果见 [姿态修正记录](docs/imu_guided_pose_measurement_20260912.md)。

---

## 2026-09-10 历史快照说明

以下内容描述此前快照；当前试验基线的入口及限制以上文和基线说明为准。

此分支保存 2026-09-10 最新的相机 → HaMeR → 位姿映射 → 六轴机械臂与三指手控制链路，以及配套录屏、RGB-D、控制数据和 IMU 记录功能。

- 分支：`codex/teleop-snapshot-20260910`
- 固定标签：`teleop-snapshot-20260910`
- 平台：Ubuntu 20.04、ROS Noetic、Gazebo 仿真、单 RGB-D 相机、HaMeR。
- 默认机械臂控制：`constrained`。IMU 只记录，不参与控制。

这是可追溯的当前修复快照，**不是所有问题均已解决的版本**。超出工作范围或接近自碰撞边界时仍可能停住；最新完整真人抓取尚未验收。

## 运行当前快照

在已配置相机、模型资产和 Python 环境的本机：

```bash
cd /home/dongtian/myhandarmtest1-teleop-repair
TELEOP_MOTION_CONTROLLER=constrained ./scripts/run_teleop_feedback_recording.sh
```

相机中张手放稳后按 `C` 建立零位；按 `Q` 结束。终端打印录像目录和 IMU 实际接收状态。默认结果目录为 `.runtime/teleop_feedback_日期T时间_随机后缀/`，包括 `screen.mp4`、`control_chain.bag`、`rgbd.db3`、手部位姿与 IMU CSV。

首次检出或切换到此版本后，先重新编译：

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release -j3 -l3
source devel/setup.bash
```

## 版本内容

- 保留机械臂初始关节位姿 `[0, 0, 0, 0, π/2, 0]`；三指根启动与张手目标统一为 **5°**。
- 左偏航手约 **35° → 末端名义90°**，右侧保持 **70° → 名义90°**；目标保留原有 2° 超调余量。横滚保持 3 倍。
- 前方纸盒约 **6.9 × 7.6 × 11.67 cm**；左侧圆柱直径 **6 cm**、Y=**−0.41 m**。
- 修正完整张手被越界背景裁剪框错误扣低置信度、导致一直保留闭合目标的问题。
- 纳入实际使用的 MoveIt Servo 局部修改、qpOASES 源码、碰撞外形和录制脚本，避免依赖未提交的源码。

完整配置、验证结果、已知限制及从固定标签恢复的方法见 [版本说明](docs/versions/teleop-snapshot-20260910.md)。本次录像的问题定位见 [诊断记录](docs/versions/teleop_diagnosis_20260910.md)。较早的说明保存在 [v0.3 历史说明](docs/versions/README_v0.3_history.md)，其中旧参数不代表当前默认值。
