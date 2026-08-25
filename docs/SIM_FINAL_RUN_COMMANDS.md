# 仿真避障与三指手 GRASP/RELEASE 运行命令

日期：2026-08-15

## 运行规则

每次只运行下面一个顶层 `roslaunch`。不要另开终端再次启动
`simulation_baseline.launch`、`move_group.launch`、Gazebo 或 RViz；这些都已经由顶层
launch 启动。冷启动通常需要 20--30 秒，在终端出现 `READY` 前不要重复执行命令。

任务打印 `DONE` 后 Gazebo/RViz 会保留供检查。开始下一项前，在当前 launch 终端按
Ctrl-C，并等待最后一行 `done`。这样不会出现 `/move_group`、
`/robot_state_publisher`、`/gazebo_gui`、模型 `robot` 或 controller 重名。

## 1. 每个新终端只做一次环境准备

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## 2. 双障碍避障

```bash
roslaunch handarm_sim_demo avoidance_demo.launch \
  gazebo_gui:=true \
  rviz:=true \
  scenario:=double_obstacle \
  repetitions:=1
```

预期终端顺序为 `moving to home -> planning collision-free path -> executing ->`
`goal reached; returning home -> Trial 1 DONE`。最终 GUI 实跑已通过；轨迹逐点无碰撞，
末端位置误差 10.66 mm、姿态误差 2.55°，且成功回 home。

单障碍只需把场景改为：

```bash
roslaunch handarm_sim_demo avoidance_demo.launch \
  gazebo_gui:=true \
  rviz:=true \
  scenario:=single_obstacle \
  repetitions:=1
```

## 3. 仅运行三指手 GRASP/RELEASE

确认上一套 launch 已 Ctrl-C 完整退出。终端 1 只启动这一套仿真，不启动
MoveIt、RViz、避障或抓取脚本：

```bash
roslaunch handarm_sim_demo hand_commands_only.launch gazebo_gui:=true
```

等待终端出现 `Simulated hand commander ready` 和 `Simulation startup ready`。
终端 2 只加载环境，不要再运行 `roslaunch`：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

发送抓握：

```bash
rostopic pub -1 /handarm_sim_demo/hand_command \
  std_msgs/String "data: 'GRASP'"
```

发送松开：

```bash
rostopic pub -1 /handarm_sim_demo/hand_command \
  std_msgs/String "data: 'RELEASE'"
```

查看最近一次命令结果：

```bash
rostopic echo -n1 /handarm_sim_demo/hand_status
```

公开接口仅接受 `GRASP` 和 `RELEASE`。例如构型命令会返回：

```text
success: false
failure_reason: public interface accepts GRASP or RELEASE only
```

`GRASP/RELEASE` 不发布机械臂轨迹。两条命令的掌型关节 `f1j1` 目标均为
`0.18 rad`，只有 `f1j2/f2j1/f3j2` 屈伸；掌型变换不会夹带在 GRASP 内。

本轮 GUI 已实测 `RELEASE -> GRASP -> RELEASE` 命令成功；另完成 5 个循环、15 个
稳态窗口，15/15 PASS，详见 `docs/HAND_GRASP_RELEASE_STABILITY.md`。该结果只证明
手指张合命令链与空载稳定性；
单独的 `GRASP success` 仍不能当作物体已被夹持；完整物理任务使用独立的三指接触、
离桌和桌面支撑门禁。仿真控制不读取
应变片或接触传感器；实体应变片反馈保持在已有实体系统中。

## 4. 物体相对三指抓取、抬升和放回

确认上一套 launch 已 Ctrl-C 完整退出，然后运行唯一一个顶层命令：

```bash
roslaunch handarm_sim_demo three_finger_pick_place_demo.launch \
  gazebo_gui:=true \
  rviz:=true \
  grasp_family:=auto
```

该任务根据 Gazebo 实时物体位姿选择上方、斜上方或侧向候选；当前盒体和 IRB120 可达性
实测选择 `top_oblique/-30 deg/roll 268 deg`。它要求 f1/f2/f3 全部实际接触，物体真实
离桌、保持、放回桌面后才松手，最后撤离。没有固定连接或直接移动物体。两次独立冷启动
均已 PASS，完整指标见 `docs/THREE_FINGER_OBJECT_RELATIVE_PICK_PLACE.md`。

只查看规划与几何标记而不运动：

```bash
roslaunch handarm_sim_demo three_finger_grasp_pose_demo.launch \
  gazebo_gui:=true rviz:=true grasp_family:=auto
```

只执行接近和三指接触，不抬升：

```bash
roslaunch handarm_sim_demo three_finger_grasp_contact_demo.launch \
  gazebo_gui:=true rviz:=true grasp_family:=auto
```

## 5. 重名冲突检查

若启动前怀疑上一套没有退出，先只读检查：

```bash
pgrep -ax rosmaster
pgrep -ax roslaunch
pgrep -ax gzserver
pgrep -ax gzclient
pgrep -ax move_group
```

有输出时不要启动新 launch；回到原 launch 终端 Ctrl-C。只有在确认这些进程都属于
本工程且原终端已经丢失时，才手工结束旧进程。不要在两套任务之间复用仍运行的
Gazebo/MoveIt 栈。
