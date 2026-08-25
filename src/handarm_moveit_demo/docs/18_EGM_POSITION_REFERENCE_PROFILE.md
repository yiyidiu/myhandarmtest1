# Gazebo EGM 式位置参考 profile

## 目的与边界

EGM 位置参考层本身只改变 Gazebo 的机械臂执行层，不改变 HaMeR/MANO、C 零位或
MoveIt Servo 的六维接口。正式入口现默认选择可单独回退的
`camera_ground_axis_decoupled` 映射：相机图像平面与深度解耦，边界/IK 保持请求
方向。它模拟 ABB EGM 中外部参考
与机器人内部位置伺服组合工作的行为，但不是 ABB 控制柜的 EGM UDP 驱动，不能
直接用于真机。

两个 profile 完全独立：

- `velocity_rollback`：原命令
  `live_human_ground_gazebo_teleop.launch`，保留原速度执行方式。
- `egm_position`：新命令
  `live_human_ground_gazebo_egm_teleop.launch`，机械臂开启重力并使用有限力矩位置保持；
  机械手默认使用可接触、有限抓力的 `physical_grasp`。

旧速度入口与修改前备份内的 SHA256 都是
`abf36f08fde6a0f8a00784453b8231b75e309866e435ee102b1814213e69f5cb`，说明新
profile 没有覆盖旧入口。

## 控制链

```text
HaMeR/MANO 相对 6D 目标（不变）
  -> 50 Hz 位姿误差闭环（不变）
  -> MoveIt Servo 关节速度前馈（不变）
  -> 最新值队列 queue_size=1
  -> 250 Hz EGM 式位置参考积分器
       - 速度/加速度/关节边界限制
       - 输入超过 0.10 s 后速度降到 0
       - 停止输入后保持最后位置参考，不把实际位置重设为参考
  -> JointGroupPositionController
  -> Gazebo PositionJointInterface + 有限力矩 PI-D + 重力
```

这里没有 `FollowJointTrajectory`，所以不会因不断追加短轨迹而形成越来越长的命令
队列。外力把机械臂推离目标时，位置参考保持不变，关节控制器会主动恢复到原目标。
新 profile 的 PI-D 参数在
`handarm_sim_demo/config/gazebo_arm_egm_position_pid.yaml`，与旧 profile 参数隔离；
积分项有力矩限幅，只补偿重力静差。

机械手与机械臂执行层分开处理。当前人手遥操作链没有持续发布手指命令，机械臂移动
期间手指保持最后预形状。`physical_grasp` 从运行时 URDF 移除 4 个手部
gazebo_ros_control transmission 和旧 mimic 插件，8 个手指关节由一个 Gazebo 插件
统一控制。ODE 在同一次约束求解中处理弹簧、阻尼和物体接触；屈曲主动关节力矩上限
为 `0.60 N·m`，屈曲随动关节为 `0.40 N·m`，所以目标被物体挡住时形成有限抓力，
不会积累无限接触冲击。标准
`/controller_gazebo_hand/follow_joint_trajectory` Action 由兼容层保留，现有
`GRASP/RELEASE` 不需要换接口。

`rigid_transport` 仍完整保留，适合只看遥操作轨迹、不做接触的快速回退；它通过
Gazebo 原生位置约束保持最硬外观，但不能作为物理抓取结论。`original` 保留旧 PID
手部植物，主要用于历史对照。

## 正式运行

先编译一次：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

终端 1，推荐的新位置参考 profile：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch \
  with_ground_object:=true
```

终端 2 仍使用原 HaMeR/D455 命令。Gazebo 完全启动后，把手放在中立位并在相机
窗口按一次 `C`；按 C 前机械臂不会跟随。启动关节角仍为
`[0, 0, 0, 0, 90, 0] deg`。

需要立即回到旧速度方案时，终端 1 改回：

```bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_teleop.launch \
  with_ground_object:=true
```

只回退到上一版无接触刚性手指、保留 EGM 机械臂位置参考层时使用：

```bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch \
  with_ground_object:=true \
  hand_stability_profile:=rigid_transport
```

旧 PID 手部历史对照可将参数改为 `hand_stability_profile:=original`。

只回退透视解耦和方向保持映射、保留 EGM 与当前手爪时：

```bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch \
  with_ground_object:=true \
  mapping_profile:=camera_ground_workspace
```

## 可重复刚度验收

先启动新 Gazebo profile，并确保相机/人手此时不在控制。另开终端运行：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun handarm_moveit_demo egm_position_profile_validator.py \
  --output /tmp/handarm_egm_position_validation.json
```

验收器会主动让关节 4 小幅运动、停手锁位、施加 `35 N·m / 0.15 s` 外部扰动、
检查恢复，再回到初始参考。2026-08-24 本机结果：

- 位置参考输出：Python 验收订阅实测约 `234 Hz`，`rostopic hz` 稳态为 `250 Hz`。
- 初始六轴跟随误差范数：`0.00014 rad`（旧 P-only 临时参数约
  `0.0305 rad`）。
- 关节 4 参考运动：`0.18900 rad`；停手后参考漂移：`0 rad`。
- `35 N·m` 扰动产生 `0.10154 rad` 峰值偏差，随后恢复到 `0.00295 rad`。
- 最后关节 4 参考/实际相对零位：`0.00145/0.00065 rad`。

2026-08-24 机械手移动基座专项对照使用同一条 0.5 s 最小冲击六轴往返轨迹：

- 原有限力矩手部 profile：4 个主动手指关节摆幅
  `0.01375/0.02365/0.01622/0.02004 rad`，速度峰值
  `8.29/2.76/1.22/1.15 rad/s`；4 个随动对速度峰值为
  `9.82/5.82/3.82/5.43 rad/s`。
- `rigid_transport`：主动关节摆幅降到
  `0.000675/0.000012/0.000021/0.000047 rad`，速度峰值降到
  `0.442/0.0069/0.0115/0.0265 rad/s`；随动对峰值降到
  `0.688/0.011/0.012/0.029 rad/s`。
- 手指开合回归：四主动关节对 `[0.18,0.20,0.20,0.20] rad` 的最大误差
  `1.69e-7 rad`，回启动预形状最大误差 `1.73e-7 rad`；机械手没有被锁死。

2026-08-24 新 `physical_grasp` 隔离实例验收：

- 标准 Action 的 `OPEN -> CLOSE -> OPEN` 三段均返回 `SUCCESSFUL`。
- 上层 `GRASP` 空载主动关节误差为 `0.0033/0.0091/0.0100/0.0111 rad`，
  随动关系最大误差 `0.0042 rad`；`RELEASE` 同样通过。
- 机械臂六轴往返运动结束后的手指 1.5 s 位置波动为
  `2.4e-6` 至 `1.7e-5 rad`。
- 蓝块接触并收敛后连续观察 10 s：8 个关节范围为
  `0.00049` 至 `0.00540 rad`，蓝块三轴范围低于 `8e-8 m`。
- 蓝块接触可由 `/handarm_sim_demo/left_object_contacts` 直接验收；测试中识别到
  f1、f2、f3 三个手指族的真实碰撞对。

确定性六轴末端验收也通过：25 mm 人手测试输入得到工具 X/Y/Z
`14.84/14.99/14.70 mm`；0.22 rad 姿态输入得到工具 X/Y/Z
`0.21960/0.21882/0.21959 rad`，比例为 `0.998/0.995/0.998`，无 Servo 危险
状态，整套动作回零误差为 `0.180 mm / 0.023 deg`。

2026-08-24 新轴解耦 profile 的隔离 Gazebo 验收同样通过：三个平移轴和三个局部
姿态轴均通过，Servo 危险状态为空，回零误差 `0.204 mm / 0.328 deg`。离线固定
图像射线深度扫描中，旧径向映射造成 `0.437654 m` 侧向目标变化，新映射为 `0 m`。
报告见 `results/axis_decoupled_mapping/`。本次没有真人实时采集，因此相机画面下的
最终量程与主观手感仍保留为待验收项。

## 修改前回退快照

```text
/home/diu/myhandarmtest1/backups/teleop_before_egm_position_reference_20260824_154746.tar.gz
SHA256 47258b6966f683754e3243ee60ba3b429f83eb05e1b7631facf40617ce5a04ea
```

验证：

```bash
cd /home/diu/myhandarmtest1/backups
sha256sum -c teleop_before_egm_position_reference_20260824_154746.sha256
```

不要直接把压缩包覆盖到工程。旧速度 profile 本身仍可直接启动；只有需要还原整轮
代码时，才把快照解压到临时目录比较后选择性恢复。

加入接触型机械手之前的最新快照为：

```text
/home/diu/myhandarmtest1/backups/before_physical_grasp_hand_complete_20260824_2032.tar.gz
SHA256 1171a2e556242acd6242a459f89539e096896855c79f87dde6498afb34a90a87
```

加入透视解耦与方向保持映射之前的最新完整源码快照为：

```text
/home/diu/myhandarmtest1/backups/before_axis_decoupled_mapping_20260824_211445.tar.gz
SHA256 9c19c1596e8559baa43c3b4e3fbf864560dbf30ee2a63d3c018c94052195f2d4
```
