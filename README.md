# v0.3.0-egm-stable-hold

重建自 2026-08-24 18:31:22（Asia/Shanghai）的稳定 EGM 位置参考检查点。

## 冻结时证据

- 87 项 Python 测试通过；
- ROS launch 解析和 profile 参数注入通过；
- 无头 Gazebo 外环 50 Hz、位置参考 250 Hz；
- 丢失输入后六轴命令参考跨度为 0 rad；
- 重新获取输入时从 `POSITION_HOLD` 回到 `TRACKING`，无参考跳变。

## 原始冻结包

`egm_stable_stiff_hold_passed_20260824_183122.tar.gz`

SHA-256：`7b9ea411fddf5beaf5a427ac81576f499396dce17d6c343a21eba154871cc860`

原包只有 15 个最终变更文件。本 Git 版本以最近的完整 EGM 源码快照为基础叠加这些文件，并补齐共同依赖。

## 2026-08-26 遥操作修整分支

- 分支：`fix/v0.3-teleop-p0`
- 诊断基线标签：`teleop-diagnostic-baseline-20260826`
- 第一批修复已经让相机丢帧、手身份变化和异常姿态跳变按 fail-closed
  方式锁定控制。
- 第二批修复将 Gazebo 手指默认执行机构改为单一所有者的有限柔顺弹簧阻尼模型，
  避免主动关节 PID 与 mimic 插件在机械臂运动和接触约束下互相对抗。

人手遥操作仍使用原来的启动入口；`physical_grasp` 已是默认手指模型：

```bash
source devel/setup.bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch
```

如需复现原始 v0.3 手指抖动作为 A/B 对照：

```bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch \
  hand_stability_profile:=original
```

固定手指目标、机械臂往复运动的无头验收测试：

```bash
roslaunch handarm_sim_demo hand_transport_stability_ab.launch \
  hand_stability_profile:=physical_grasp \
  output_file:=/tmp/physical_grasp.json
```

该验收只证明“机械臂搬运期间固定手指目标”的稳定性，不等同于完成了人手指映射、
物体接触抓取或端到端遥操作验收。详细结果见
`docs/teleop_evidence/v0.3_20260826/P0_REMEDIATION_02_HAND_PLANT.md`。
