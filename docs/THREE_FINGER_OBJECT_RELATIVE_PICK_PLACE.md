# 三指手物体相对抓取、抬升与放回验收报告

日期：2026-08-15

## 结论

当前 50 × 60 × 100 mm 盒体在 Gazebo 中已完成两次独立冷启动的物理
`抓取 -> 离桌抬升 -> 悬空保持 -> 放回桌面 -> 松手 -> 撤离`。两次均为 PASS。
任务没有使用固定连接、Gazebo attach、`set_model_state` 搬运物体或 MoveIt
attached object；物体运动来自三根手指与目标物体的接触力。

当前结论仅适用于本仿真模型、当前盒体尺寸和固定掌型。它不是力闭合证明，也不是实体
机械手验收。

## 旧动作为什么失败

旧流程把 YAML 中的初始物体位姿和一个固定世界坐标四元数当作预抓取目标，没有用
Gazebo 的实时物体姿态计算完整的末端位姿；同时 `tool0` 被近似当作抓取中心。结果是
盒体没有进入三指闭合包络中心，f1/f2 先接触，而 f3 仍悬空。继续闭合还会让已接触
手指滚过盒体或把盒体推出抓取区域。

新流程在每次规划前等待物体稳定，然后读取 `/gazebo/get_model_state`，构造
`T_world_object` 并同步 MoveIt PlanningScene。正式目标为：

```text
T_world_grasp_center = T_world_object * T_object_grasp_center
T_world_tool0 = T_world_grasp_center * inverse(T_tool0_grasp_center)
```

因此物体平移和 yaw 会直接进入末端目标。0°、30°、60°、90° yaw 的纯几何回归均验证
候选随物体转动，而不是保持固定世界姿态。

## 抓取中心与候选选择

固定掌型保持 `f1j1 = 0.18 rad`。`grasp_center` 定义为 OPEN 到 CLOSE 中点处三根
远端碰撞圆柱中心的质心，并在启动时从当前 URDF 重新计算。当前变换为：

```text
T_tool0_grasp_center =
[ 0.999999683, 0,          -0.000796327, 0.008505517 ]
[ 0,           1,           0,           0.010089919 ]
[ 0.000796327, 0,           0.999999683, 0.080675925 ]
[ 0,           0,           0,           1           ]
```

规划器生成 `top_down`、`top_oblique`、四个 `side` 方向、中心偏移和绕接近轴 roll
候选，并对三指包络、掌部净空、桌面净空、完整六轴 IK、关节余量、连续接近路径做硬
门禁。最近一次实际规划生成 2833 个候选，91 个通过纯几何门禁；严格竖直上抓不可达，
侧抓因桌面/可达性门禁被拒绝，最终选择物体局部 `+Z` 方向的斜上抓：

```text
family: top_oblique
tilt: -30 deg
roll about approach axis: 268 deg
object center offset in hand: [0.006, -0.009, 0.052] m
joint_6: 2.26419 rad (来自完整 MoveIt IK)
predicted contacts: f1, f2, f3
predicted table clearance: 28.90 mm
predicted palm clearance: 73.92 mm
```

roll 不是在 IK 之后手工修改 `joint_6`。每个 roll 先生成完整工具姿态，再求完整六轴
IK。当前 268° 是实际三指接触校准值；264° 的真实测试只有 f1/f2，已被拒绝。

## 三指实际接触与抬升策略

接近后使用接触限制闭合：检测到 f1/f2/f3 对目标物体连续稳定接触后立即停止继续深闭合。
抬升前只增加有界屈曲预载：

```text
f1j2: +0.08 rad
f2j1: +0.08 rad
f3j2: +0.01 rad
f1j1: 保持不变
```

较小的 f3 预载防止其远端指节卷过盒体。抬升沿所选抓取接近轴的反方向执行，不使用
固定世界 Z，从而保持物体相对抓取关系。运行中持续要求精确接触族等于
`{f1, f2, f3}`；接触丢失超过 0.15 s 即失败。

“抓起”需要同时满足：物体中心实测上升至少 20 mm、目标物体与桌面接触连续消失至少
0.30 s、三指接触仍存在、工具与物体位移差不超过 10 mm。只看机械臂轨迹或物体高度
均不能单独判定成功。

## 放回与桌面干涉保护

放置轨迹是抬升轨迹的实际末端位姿逆过程。松手前必须满足：

- 三指仍接触目标物体；
- 目标物体与桌面接触连续稳定至少 0.30 s；
- 物体回到起始桌面位置误差不超过 15 mm；
- 手部实际桌面净空至少 8 mm。

仅在上述条件满足后执行 `RELEASE`。松手后还要求所有手指与目标物体接触连续清除
0.50 s，同时目标仍由桌面支撑，最后沿接近轴反向撤离 40 mm。失效保护不会在物体仍
悬空时自动张手。

## 两次真实冷启动结果

| 指标 | 第一次 | 第二次 | 门槛 |
|---|---:|---:|---:|
| 三指接触最大丢失 | 0.000 s | 0.000 s | <= 0.15 s |
| 物体中心抬升 | 21.665 mm | 21.173 mm | >= 20 mm |
| 无桌面支撑稳定时间 | 0.301 s | 0.300 s | >= 0.30 s |
| 悬空保持 | 2.002 s | 2 s 以上 | >= 2 s |
| 工具/物体位移差 | 7.559 mm | 7.948 mm | <= 10 mm |
| 放回位置误差 | 7.877 mm | 6.738 mm | <= 15 mm |
| 接触时手—桌净空 | 17.570 mm | 17.561 mm | >= 8 mm |
| 放置时手—桌净空 | 16.352 mm | 17.214 mm | >= 8 mm |
| 撤离后手—桌净空 | 28.825 mm | 28.813 mm | >= 8 mm |
| 松手导致的物体位移 | 0.018 mm | 0.043 mm | 仅报告 |
| lift/place/retreat Cartesian fraction | 1/1/1 | 1/1/1 | >= 0.995 |
| attachment used | false | false | 必须 false |
| 最终结果 | PASS | PASS | 全部门禁通过 |

原始机器证据：

- `results/sim_baseline/three_finger_pick_place_20260815T114222069571Z.json`
- `results/sim_baseline/three_finger_pick_place_20260815T114627688746Z.json`
- `results/sim_baseline/three_finger_contact_only_20260815T114138598339Z.json`
- `results/sim_baseline/three_finger_contact_only_20260815T114544254453Z.json`

## 可视化运行

每次只运行一个顶层 launch。任务最终 PASS 后界面会保留，按 Ctrl-C 才退出。

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch handarm_sim_demo three_finger_pick_place_demo.launch \
  gazebo_gui:=true \
  rviz:=true \
  grasp_family:=auto
```

预期终端最终出现：

```text
[three-finger-contact] PASS actual f1/f2/f3 contact
[three-finger-pick-place] PASS physical lift/place/release
```

如果终端出现任何 `FAILED`，即使画面看起来像抓起也不算通过。不要同时启动
`simulation_baseline.launch`、`move_group`、第二个 Gazebo 或第二个本任务 launch；否则
会再次出现节点、模型和控制器重名。

## 冻结项

本轮没有为通过测试而修改物体质量、重力、摩擦、PID、mimic effort、Gazebo 接触求解
参数、物体尺寸或固定连接。改动仅限物体相对几何规划、三指接触门禁、运动状态机和验收
记录。
