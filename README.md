# 单 RGB-D 手部遥操作 Gazebo：2026-09-10 代码快照

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
