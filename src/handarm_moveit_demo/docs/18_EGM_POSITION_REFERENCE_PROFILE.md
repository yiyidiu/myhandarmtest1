# Gazebo EGM 式位置参考 profile

## 目的与边界

本 profile 只改变 Gazebo 的机械臂执行层，不改变 HaMeR/MANO、C 零位、相机轴
映射、地面工作空间映射或 MoveIt Servo 的六维目标。它模拟 ABB EGM 中外部参考
与机器人内部位置伺服组合工作的行为，但不是 ABB 控制柜的 EGM UDP 驱动，不能
直接用于真机。

两个 profile 完全独立：

- `velocity_rollback`：原命令
  `live_human_ground_gazebo_teleop.launch`，保留原速度执行方式。
- `egm_position`：新命令
  `live_human_ground_gazebo_egm_teleop.launch`，开启重力并使用有限力矩位置保持。

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

确定性六轴末端验收也通过：25 mm 人手测试输入得到工具 X/Y/Z
`14.84/14.99/14.70 mm`；0.22 rad 姿态输入得到工具 X/Y/Z
`0.21960/0.21882/0.21959 rad`，比例为 `0.998/0.995/0.998`，无 Servo 危险
状态，整套动作回零误差为 `0.180 mm / 0.023 deg`。

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
