# P0 基线运行手册

记录时间：2026-08-13（Asia/Shanghai）  
工作空间：`/home/diu/myhandarmtest1`

## 1. 环境前提

本基线只连接 Gazebo，不连接真实 ABB 控制器，也不启动 D455/HaMeR。当前已确认：

```text
Ubuntu 20.04.6 LTS
ROS Noetic 1.17.4
Gazebo 11.15.1
NVIDIA GeForce RTX 2060, 6144 MiB, driver 570.133.20
```

## 2. 构建

```bash
cd /home/diu/myhandarmtest1
catkin_make
source devel/setup.bash
```

预期结果：`catkin_make` 返回 0，并构建 mimic/disable-link Gazebo 插件。

## 3. 启动现有速度 Servo 基线

```bash
cd /home/diu/myhandarmtest1
source devel/setup.bash
roslaunch abb120_moveit_config1 \
  demo_gazebo_servo_velocity.launch \
  gazebo_gui:=false
```

安全说明：该 launch 只用于现有基线。当前 `servo_abbarm_velocity.yaml` 的
`check_collisions` 是 `false`，不能作为正式安全配置，也不能连接真实机械臂。

已知行为：虽然传入 `gazebo_gui:=false`，当前包含链仍会启动 `/gazebo_gui` 和 RViz。
因此它尚不满足低显存 headless 运行要求。

## 4. 基线观测命令

另开终端：

```bash
cd /home/diu/myhandarmtest1
source devel/setup.bash

rosnode list
rostopic list
rosservice call /controller_manager/list_controllers

rostopic type /joint_states
rostopic echo -n 1 /joint_states
rostopic type /servo_server/status
rostopic type /servo_server/delta_twist_cmds
rostopic type /abbarm_velocity_controller/command
rostopic type /controller_gazebo_hand/follow_joint_trajectory/status

rosparam get /servo_server/check_collisions
rosparam get /servo_server/incoming_command_timeout
rosparam get /servo_server/command_out_topic
```

预期控制器：

```text
joint_state_controller        running
abbarm_velocity_controller    running
controller_gazebo_hand        running
```

预期链路：

```text
/servo_server/delta_twist_cmds       geometry_msgs/TwistStamped
  -> /servo_server
  -> /abbarm_velocity_controller/command  std_msgs/Float64MultiArray
  -> /gazebo
```

三指手 action 基线：

```text
/controller_gazebo_hand/follow_joint_trajectory/*
```

## 5. 零速度链路探测

只允许在上述 Gazebo 基线中执行：

```bash
rostopic pub -r 20 /servo_server/delta_twist_cmds \
  geometry_msgs/TwistStamped \
  "{header: {frame_id: 'base_link'}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}"
```

另一个终端观察：

```bash
rostopic echo -n 1 /servo_server/status
rostopic echo /abbarm_velocity_controller/command
```

2026-08-13 的实测状态为 `data: 0`；零 Twist 期间控制器命令话题没有产生样本，
所以这一探测不能替代后续 P10 的六轴非零合成输入测试。

## 6. 停止

在 roslaunch 终端按一次 `Ctrl-C`，随后确认无残留：

```bash
pgrep -a -f 'roscore|rosmaster|gzserver|gzclient|rviz|servo_server|move_group'
```

若 Gazebo 控制器卸载停滞，等待 roslaunch 正常升级终止信号；不要同时启动第二套
Gazebo/ROS master。

## 7. 静态基线检查

```bash
cd /home/diu/myhandarmtest1

catkin_make run_tests
catkin_test_results --all build/test_results
rosdep check --from-paths src --ignore-src
check_urdf src/abb120_moveit_config1/config/gazebo_handarm_velocity.urdf
```

注意：当前 `run_tests` 显示 0 tests。这只说明没有测试失败，不代表功能通过。
