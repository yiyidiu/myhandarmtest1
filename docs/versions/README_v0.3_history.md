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
- 第三批修复将默认机械臂路径切回 MoveIt Servo，并在其后增加完整臂手状态的
  连续扫掠 FCL 保护器；场景、关节状态、碰撞监视器或速度命令失联时均闭锁。
- 第四批修复把相机、HaMeR、前臂估计和显示解耦为最新帧路径，D435i 固定 ROI
  实测达到 20 Hz，并将采集到发布延迟与真实推理来源写入 UDP/ROS/CSV。
- 第五批修复加入连续手指链路：五个人手 MANO 弯曲特征经 C 后张手标定、因果滤波、
  五到三指协同映射、限幅/限速和失效令牌锁定后，驱动 Gazebo 四个主动关节。

人手遥操作请使用单一入口；它依次等待 Gazebo、控制器和 PlanningScene 就绪后再
打开相机，且 `physical_grasp` 已是默认手指模型：

```bash
cd /home/dongtian/myhandarmtest1-v0.3.0
./scripts/run_live_human_teleop_validation.sh
```

相机窗口中保持中性张手并按一次 `C`，随后移动手腕。终端必须依次出现
`[链路 1/3]`（ROS 接受 C）、`[链路 2/3]`（Servo 非零指令）和
`[链路 3/3]`（Gazebo 末端实际移动）；缺少第三层时脚本会打印 Servo 状态与碰撞
限速比例。按 `Q` 退出相机，脚本会同步关闭它启动的全部 ROS/Gazebo 进程。

当前 EGM 真人入口默认使用未标定也可安全验证的 `current_linear` 相对映射：人手
平移按 0.6 倍映射。`camera_ground_workspace` 只能在完成当前操作者的工作空间标定后
显式启用，不能再把仓库中的示例范围当成真人默认值。

需要正式 C-to-Q 录屏、rosbag、CSV、HaMeR JSONL 和 CPU/GPU 指标时运行：

```bash
./scripts/run_live_human_evidence_session.sh
```

该证据脚本会先启动低开销预录并等待有效张手；相机侧持久化的 C/Q 标记定义正式
测量边界，因此不会依赖后台轮询去“猜”按键时刻。实现与自动验收结果见
`docs/teleop_evidence/v0.3_20260826/P0_REMEDIATION_06_LIVE_CHAIN.md`。

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

该验收只证明“机械臂搬运期间固定手指目标”的稳定性。连续人手指映射现在另由下述
端到端测试覆盖；物体接触抓取和真人 C-to-Q 遥操作仍需单独验收。详细结果见
`docs/teleop_evidence/v0.3_20260826/P0_REMEDIATION_02_HAND_PLANT.md`。

连续手指 UDP→ROS→Gazebo 验收：

```bash
./scripts/run_finger_retargeting_gazebo_validation.sh
```

它验证食指→f1、拇指→对置 f2、中/无名/小指协同→f3 的独立响应，张手回零、机械臂
运动时固定人手指输入的稳定性、INVALID 连续心跳超时和旧 C 令牌拒绝。实现与边界见
`docs/teleop_evidence/v0.3_20260826/P0_REMEDIATION_05_FINGER_RETARGETING.md`。

机械臂自碰撞、关节限位和恢复的自动验收：

```bash
roslaunch handarm_moveit_demo egm_servo_safety_acceptance.launch
```

最终仿真结果为自碰撞接近/恢复、关节上限接近/恢复全部 PASS；状态失联闭锁和公开
入口 PlanningScene 同步也已单独验证。该结论仍不覆盖动态外部障碍物、奇异位形、
真机或端到端真人遥操作质量。实现与测量见
`docs/teleop_evidence/v0.3_20260826/P0_REMEDIATION_03_ARM_SAFETY.md`。
