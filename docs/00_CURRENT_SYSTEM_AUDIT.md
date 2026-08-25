# P0 当前系统审计

审计日期：2026-08-13（Asia/Shanghai）  
工作空间根：`/home/diu/myhandarmtest1`  
范围：只审计现有工程、硬件环境和 P0 基线；本轮未修改核心代码。

> 历史快照说明：本文件冻结 P0 审计时状态。D455 随后已连接，当前结果见
> `docs/04_D455_CAPTURE_AND_RECORDING.md`；下文“未枚举/未连接”不得解释为当前状态。

## 0. 证据口径

- `PASS`：本轮或同一 P0 基线中实际执行并留下结果。
- `STATIC PASS`：只完成语法、依赖解析或配置检查，不代表运行功能通过。
- `PARTIAL`：进程或接口存在，但验收链路不完整。
- `FAIL`：实际检查得到失败或配置明确不满足目标。
- `NOT RUN`：未实际运行，绝不据源码推断为通过。
- 文件结论均给出 `路径:行号`；硬件/进程/目录枚举没有源码行号时，给出实际命令或文件系统证据。

## 1. 结论摘要

P0 已证明该工作空间能够增量构建，并能共同启动 Gazebo、MoveIt 1、MoveIt Servo、ABB 六轴速度控制器和三指手四主动关节轨迹控制器；实际证据见 `docs/02_BASELINE_TEST_REPORT.md:24-47`。这只是“现有基线可启动”，不是正式遥操作通过。

当前正式目标仍被以下问题阻塞：

1. 两套 Servo 配置均关闭碰撞检查，且使用了为调试放宽的奇异点阈值，不能连接真实机械臂：`src/abb120_moveit_config1/config/servo_abbarm.yaml:45-59`、`src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:41-52`。
2. P0 快照时，正式要求的 HaMeR、MANO、RGB-D KLT、RANSAC-Kabsch、SO(3)/SE(3) 融合、clutch、手势状态机、接触闭环与抓取判定在 `src/` 中均无实现；现有相机脚本仍是 MediaPipe 关键点 + D455 深度 + 每帧掌面 RANSAC：`src/handarm_moveit_demo/scripts/d455_conda_udp_sender_servo_v3.py:5-6`、`:306-362`、`:670-741`。
3. 旧姿态映射中的相机到控制坐标矩阵被代数抵消：`R_delta=(A R0)^T(A Rn)=R0^T Rn`，对应 `src/handarm_moveit_demo/scripts/d455_conda_udp_sender_servo_v3.py:514-528`；它没有实现所要求的 `R_EH @ delta_R_hand @ R_EH.T` 工具共轭映射。
4. `gazebo_gui:=false` 的主速度 launch 仍解析出 `/gazebo_gui` 和 RViz；运行基线也已复现，见 `docs/01_BASELINE_RUNBOOK.md:27-41`、`docs/02_BASELINE_TEST_REPORT.md:36-37`。顶层 launch 未暴露 `use_rviz` 参数：`src/abb120_moveit_config1/launch/demo_gazebo_servo_velocity.launch:4-18`，而下层 `demo.launch` 默认 `use_rviz=true`：`src/abb120_moveit_config1/launch/demo.launch:22-24`。
5. Gazebo 运行报告十个主动关节缺少 `/gazebo_ros_control/pid_gains/*` 的 P 增益：`docs/02_BASELINE_TEST_REPORT.md:42-46`。控制器 YAML 内的手部轨迹增益位于控制器命名空间，并不等价于 Gazebo hardware simulation 的 PID 参数：`src/abb120_moveit_config1/config/ros_controllers_velocity.yaml:14-25`。
6. 当前 D455 未被 `lsusb` 枚举到；因此真实 RGB-D 采集仍是 `NOT RUN`。GPU 实测为 `NVIDIA GeForce RTX 2060, 6144 MiB, driver 570.133.20`，必须按 6 GB 低显存方案推进；同一结果记录于 `docs/02_BASELINE_TEST_REPORT.md:28-29`。
7. 工作空间根不是 Git 仓库，且包含 `build/`、`devel/` 与嵌套 `src/roboticsgroup_upatras_gazebo_plugins/.git/`；最终源码包必须排除这些目录。工作空间标记见 `.catkin_workspace:1`，基线报告也记录了交付风险：`docs/02_BASELINE_TEST_REPORT.md:22`。

## 2. 工作空间与 ROS 包

工作空间由 `.catkin_workspace` 标记，源码位于 `/home/diu/myhandarmtest1/src`；当前存在预生成 `build/` 和 `devel/`。`src/CMakeLists.txt` 是指向 `/opt/ros/noetic/share/catkin/cmake/toplevel.cmake` 的符号链接。

| ROS 包 | 类型与职责 | 证据 |
|---|---|---|
| `abb120_moveit_config1` | MoveIt/SRDF、规划、Gazebo、ros_control 与 Servo 配置包 | `src/abb120_moveit_config1/package.xml:3-6`、`:19-40` |
| `abb_resources` | ABB 通用材质/颜色 xacro 资源，无节点 | `src/abb_resources/package.xml:4-12`、`src/abb_resources/CMakeLists.txt:4-9` |
| `handarm_moveit_demo` | Python MoveIt、Servo、UDP、D455 基线脚本与测试 launch | `src/handarm_moveit_demo/package.xml:3-16`、`src/handarm_moveit_demo/CMakeLists.txt:12-34` |
| `handarmtest1` | ABB IRB120 + 三指手 URDF/xacro、旧 Gazebo launch/config | `src/handarmtest1/package.xml:3-5`、`:51-75` |
| `roboticsgroup_upatras_gazebo_plugins` | Gazebo mimic/disable-link 两个共享插件 | `src/roboticsgroup_upatras_gazebo_plugins/package.xml:21-44`、`src/roboticsgroup_upatras_gazebo_plugins/CMakeLists.txt:13-39` |

总计 5 个 ROS 包。包枚举依据全部 `src/*/package.xml`，未发现其他包。

## 3. 当前数据流与缺口

### 3.1 已存在的旧视觉 Servo 链

```text
D455 640x480@30 RGB + depth align
  -> MediaPipe Hands 21 landmarks
  -> landmark depth deprojection + palm mask
  -> per-frame palm plane RANSAC/PCA frame
  -> xyz + Euler delta JSON/UDP :5005
  -> /target_ee_pose
  -> pose tracking node
  -> /servo_server/delta_twist_cmds
  -> MoveIt Servo
  -> /abbarm_velocity_controller/command
  -> Gazebo
```

证据：D455 流和对齐在 `src/handarm_moveit_demo/scripts/d455_conda_udp_sender_servo_v3.py:579-603`、`:631-656`；MediaPipe 深度点和掌面拟合在 `:670-741`；UDP 包在 `:815-830`；ROS 接收器输出 `/target_ee_pose` 在 `src/handarm_moveit_demo/scripts/ros_udp_target_pose_receiver_servo_local.py:76-100`、`:161-185`；跟踪节点输入/输出在 `src/handarm_moveit_demo/scripts/servo_pose_tracking_node_v2.py:184-264`、`:560-572`；Servo 直通速度控制器在 `src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:14-22`。

接收器默认拒绝坏质量包且不重复发布，依靠下游 0.5 s target timeout 归零：`src/handarm_moveit_demo/scripts/ros_udp_target_pose_receiver_servo_local.py:86-91`、`:161-188`，`src/handarm_moveit_demo/scripts/servo_pose_tracking_node_v2.py:231-236`、`:411-420`、`:474-483`。这提供了基础超时停止，但 UDP 没有序列号、会话号、协议版本、CRC 或显式 clutch 状态；本链路不能视为正式安全协议。

### 3.2 正式架构缺失

对 `src/` 的全文审计未发现 HaMeR、MANO、KLT/optical-flow、Kabsch、SE(3) 融合、clutch、gesture decoder、接触传感器、力/力矩反馈或 Planning Scene 障碍物发布实现。`sensors_3d.yaml` 为空：`src/abb120_moveit_config1/config/sensors_3d.yaml:1-2`；传感器 manager launch 为空：`src/abb120_moveit_config1/launch/handarm_moveit_sensor_manager.launch.xml:1-3`。因此点云/Octomap 和动态障碍物并未接入。

## 4. 节点、话题与服务

### 4.1 主速度 Servo launch 的核心节点

`roslaunch --nodes abb120_moveit_config1 demo_gazebo_servo_velocity.launch gazebo_gui:=false` 已静态解析出：`/gazebo`、`/gazebo_gui`、`/spawn_gazebo_model`、`/controller_spawner`、`/gazebo_controller_spawner`、`/robot_state_publisher`、`/virtual_joint_broadcaster_0`、`/move_group`、匿名 `/rviz_*`、`/servo_server`。节点来源分别见：

- Gazebo、模型生成、joint-state controller、robot-state publisher：`src/abb120_moveit_config1/launch/gazebo_velocity.launch:10-33`。
- ABB 速度控制器和手部轨迹控制器 spawner：`src/abb120_moveit_config1/launch/ros_controllers_velocity.launch:3-8`。
- 静态 world→base TF、Move Group、RViz：`src/abb120_moveit_config1/launch/demo.launch:22-27`、`:45-60`。
- Servo server：`src/abb120_moveit_config1/launch/abbarm_servo_velocity.launch:3-9`。

核心接口如下：

| 话题/接口 | 类型/方向 | 证据 | P0 运行状态 |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState`，Servo 与应用订阅 | `src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:35`、`src/handarm_moveit_demo/scripts/joint_target_servo_controller.py:28-30` | PASS，收到 10 个主动关节；`docs/02_BASELINE_TEST_REPORT.md:37` |
| `/servo_server/delta_twist_cmds` | `geometry_msgs/TwistStamped`，Servo 笛卡尔输入 | `src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:14-16` | PASS，类型与 subscriber 已确认；`docs/02_BASELINE_TEST_REPORT.md:38-40` |
| `/servo_server/delta_joint_cmds` | `control_msgs/JointJog`，Servo 关节输入 | `src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:16`、`src/handarm_moveit_demo/scripts/joint_target_servo_controller.py:24-30`、`:111-119` | NOT RUN |
| `/servo_server/status` | `std_msgs/Int8`，Servo 状态 | `src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:35-37` | PASS，零 Twist 时 `data: 0`；`docs/02_BASELINE_TEST_REPORT.md:38` |
| `/abbarm_velocity_controller/command` | `std_msgs/Float64MultiArray`，Servo→六轴速度控制器 | `src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:18-22` | 连接 PASS；零输入无样本，数值链 INCONCLUSIVE；`docs/02_BASELINE_TEST_REPORT.md:40-43` |
| `/controller_gazebo_hand/follow_joint_trajectory/*` | `control_msgs/FollowJointTrajectory` action topics | `src/abb120_moveit_config1/config/ros_controllers_velocity.yaml:14-25` | 控制器 running；动作执行 NOT RUN；`docs/02_BASELINE_TEST_REPORT.md:43` |
| `/target_ee_pose` | `geometry_msgs/PoseStamped`，视觉/测试目标→跟踪器 | `src/handarm_moveit_demo/scripts/ros_udp_target_pose_receiver_servo_local.py:79-94`、`src/handarm_moveit_demo/scripts/servo_pose_tracking_node_v2.py:188-192` | NOT RUN（真实视觉链） |
| `/handarm_trajectory/markers` | `visualization_msgs/MarkerArray`，轨迹显示 | `src/handarm_moveit_demo/scripts/handarm_trajectory_rviz.py:79-106` | NOT RUN |
| `/handarm_trajectory_visualizer/reset` | `std_srvs/Empty` 私有服务 | `src/handarm_moveit_demo/scripts/handarm_trajectory_rviz.py:116-135` | NOT RUN |

### 4.2 应用脚本节点清单

以下是 `handarm_moveit_demo/CMakeLists.txt:12-34` 安装的全部 Python 可执行脚本；话题均为源码默认值。它们都通过了静态 Python 解析，但本轮未逐个启动，包括主提示词点名的 D455 sender、UDP receiver、pose tracker v2、position bridge 和 dynamic target publisher，故这些应用节点的功能状态统一为 `NOT RUN`：

| 节点/进程 | 直接输入 | 直接输出 | 证据 |
|---|---|---|---|
| `print_moveit_info` | MoveIt API | 控制台 | `src/handarm_moveit_demo/scripts/01_print_moveit_info.py:14-35` |
| `move_arm_named`、`move_arm_joint`、`move_arm_relative_pose` | 无直接话题，使用 `MoveGroupCommander("abbarm")` | MoveIt action/service（隐式） | `src/handarm_moveit_demo/scripts/02_move_arm_named.py:14-16`、`src/handarm_moveit_demo/scripts/03_move_arm_joint.py:15-17`、`src/handarm_moveit_demo/scripts/04_move_arm_relative_pose.py:16-18` |
| `move_hand_named` | 无直接话题，使用 `MoveGroupCommander("hand")` | 命名状态 start1→grasp2→start1 | `src/handarm_moveit_demo/scripts/05_move_hand_named.py:22-34`；源码明确说明不是自适应抓取：`:3-7` |
| `dynamic_arm_controller` | `/abbarm/joint_target_deg`、`/abbarm/joint_delta_deg`、`/abbarm/ee_target_xyzrpy_deg`、`/abbarm/ee_delta_xyzrpy_deg` | MoveIt action/service（隐式） | `src/handarm_moveit_demo/scripts/dynamic_arm_controller.py:55-76`、`:105-133` |
| `keyboard_ee_delta_control` | 键盘 | `/abbarm/ee_delta_xyzrpy_deg` `Float64MultiArray` | `src/handarm_moveit_demo/scripts/keyboard_ee_delta_control.py:12-31` |
| `joint_target_servo_controller` | `/joint_states`、`/abbarm/joint_target_deg` | `/servo_server/delta_joint_cmds` `JointJog` | `src/handarm_moveit_demo/scripts/joint_target_servo_controller.py:20-56` |
| `servo_twist_pulse_test` | 参数 | `/servo_server/delta_twist_cmds` | `src/handarm_moveit_demo/scripts/servo_twist_pulse_test.py:11-21` |
| `servo_pose_step_test` | TF base_link→tool0 | `/servo_server/delta_twist_cmds` | `src/handarm_moveit_demo/scripts/servo_pose_step_test.py:25-43` |
| `servo_dynamic_target_test` | TF base_link→tool0 | `/servo_server/delta_twist_cmds` | `src/handarm_moveit_demo/scripts/servo_dynamic_target_test.py:43-70` |
| `servo_pose_tracking_node` | `/target_ee_pose` | `/servo_server/delta_twist_cmds` | `src/handarm_moveit_demo/scripts/servo_pose_tracking_node.py:122-149` |
| `servo_pose_tracking_node_v2` | `/target_ee_pose` | `/servo_server/delta_twist_cmds` | `src/handarm_moveit_demo/scripts/servo_pose_tracking_node_v2.py:184-264` |
| `publish_target_pose_test` | TF base_link→tool0 | `/target_ee_pose` | `src/handarm_moveit_demo/scripts/publish_target_pose_test.py:18-40` |
| `publish_dynamic_target_pose_test` | TF base_link→tool0 | `/target_ee_pose` | `src/handarm_moveit_demo/scripts/publish_dynamic_target_pose_test.py:17-46`、`:98-111` |
| `servo_velocity_to_position_bridge` | `/joint_states`、`/servo_server/raw_joint_cmds` | `/controller_gazebo/command` `JointTrajectory` | `src/handarm_moveit_demo/scripts/servo_velocity_to_position_bridge.py:25-41`、`:111-121` |
| `ros_udp_target_pose_receiver_servo_local` | UDP `127.0.0.1:5005` | `/target_ee_pose` | `src/handarm_moveit_demo/scripts/ros_udp_target_pose_receiver_servo_local.py:76-100` |
| `ros_udp_target_pose_receiver_apriltag`、`..._v3` | UDP `127.0.0.1:5005` | `/target_ee_pose` | `src/handarm_moveit_demo/scripts/ros_udp_target_pose_receiver_apriltag.py:93-113`、`src/handarm_moveit_demo/scripts/ros_udp_target_pose_receiver_apriltag_v3.py:58-78` |
| `handarm_trajectory_visualizer` | `/target_ee_pose` + TF | `/handarm_trajectory/markers` + `~reset` | `src/handarm_moveit_demo/scripts/handarm_trajectory_rviz.py:75-128` |
| `d455_conda_udp_sender_servo_v3.py` | D455、键盘 c/r/ESC | UDP JSON `host:port`，不是 ROS 节点 | `src/handarm_moveit_demo/scripts/d455_conda_udp_sender_servo_v3.py:550-583`、`:815-830`、`:849-887` |

没有 launch 将 D455 sender、UDP receiver、pose tracker 与主 Servo 仿真一键串联。只有 tracker 自己有 leaf launch：`src/handarm_moveit_demo/launch/servo_pose_tracking_v2.launch:42-86`；receiver 与 sender 仅在 CMake 安装列表出现：`src/handarm_moveit_demo/CMakeLists.txt:27-32`。

## 5. Launch 包含关系

### 5.1 当前首选速度链

```text
demo_gazebo_servo_velocity.launch
├─ demo_gazebo_velocity.launch
│  ├─ gazebo_velocity.launch
│  │  ├─ gazebo_ros/empty_world.launch
│  │  └─ ros_controllers_velocity.launch
│  └─ demo.launch
│     ├─ move_group.launch
│     │  ├─ planning_context.launch
│     │  ├─ planning_pipeline.launch.xml (OMPL/CHOMP/Pilz)
│     │  ├─ trajectory_execution.launch.xml
│     │  │  └─ ros_control_moveit_controller_manager.launch.xml
│     │  └─ sensor_manager.launch.xml
│     │     └─ handarm_moveit_sensor_manager.launch.xml (空)
│     ├─ moveit_rviz.launch (默认启用)
│     └─ default_warehouse_db.launch (db=true 时)
└─ abbarm_servo_velocity.launch
```

直接证据：顶层两项 include 在 `src/abb120_moveit_config1/launch/demo_gazebo_servo_velocity.launch:10-18`；Gazebo/MoveIt 分支在 `demo_gazebo_velocity.launch:12-20`；Gazebo 子链在 `gazebo_velocity.launch:10-33`；MoveIt 子链在 `demo.launch:45-65`、`move_group.launch:42-84`；轨迹执行动态 include 在 `trajectory_execution.launch.xml:19-21`；传感器动态 include 在 `sensor_manager.launch.xml:8-15`。

### 5.2 位置桥接链与其他 launch

- `demo_gazebo_servo.launch` → `demo_gazebo.launch` + `abbarm_servo.launch` + `servo_velocity_to_position_bridge`：`src/abb120_moveit_config1/launch/demo_gazebo_servo.launch:10-30`。
- `demo_gazebo.launch` → `gazebo.launch` + `demo.launch`：`src/abb120_moveit_config1/launch/demo_gazebo.launch:13-19`。
- `gazebo.launch` → `gazebo_ros/empty_world.launch` + `ros_controllers.launch`，并加载位置版 URDF：`src/abb120_moveit_config1/launch/gazebo.launch:10-33`。
- `default_warehouse_db.launch` → `warehouse.launch` → `warehouse_settings.launch.xml`：`src/abb120_moveit_config1/launch/default_warehouse_db.launch:3-13`、`warehouse.launch:4-12`。
- `planning_pipeline.launch.xml` 按参数包含 `$(pipeline)_planning_pipeline.launch.xml`：`src/abb120_moveit_config1/launch/planning_pipeline.launch.xml:6-8`；`ompl-chomp` 额外包含 OMPL：`ompl-chomp_planning_pipeline.launch.xml:3-19`。
- `handarmtest1/irb120_gazebo.launch` → `gazebo_ros/empty_world.launch` + `loadarm.launch`；其 ros_control include 被注释，故该旧 launch 本身不启动关节控制器：`src/handarmtest1/launch/irb120_gazebo.launch:11-29`。
- `handarmtest1/testarm.launch` → `loadarm.launch` + joint/robot state publisher + RViz：`src/handarmtest1/launch/testarm.launch:3-7`。
- `handarm_moveit_demo` 的 5 个 launch 均为叶节点，不再 include 其他 launch；节点定义见 `src/handarm_moveit_demo/launch/servo_pose_tracking_v2.launch:42-86`、`servo_pose_tracking.launch:20-41`、`servo_pose_step_test.launch:10-18`、`servo_dynamic_target_test.launch:20-35`、`handarm_trajectory_rviz.launch:20-42`。

## 6. MoveIt、Servo 与控制器

### 6.1 MoveIt 规划组

| 组/状态 | 定义 | 审计结果 |
|---|---|---|
| `abbarm` | chain `base_link` → `tool0` | `src/abb120_moveit_config1/config/handarm.srdf:12-14` |
| `hand` | `f1j1,f1j2,f2j1,f3j2` | `src/abb120_moveit_config1/config/handarm.srdf:15-20` |
| `abbarm/up` | 六轴命名姿态 | `src/abb120_moveit_config1/config/handarm.srdf:22-29` |
| `hand/start1`、`grasp1`、`grasp2` | 四主动关节命名姿态 | `src/abb120_moveit_config1/config/handarm.srdf:30-47` |

`README.md` 把 `abbarm` 错写成 `base_link -> handbase_link`：`src/handarm_moveit_demo/README.md:52-57`；真实 SRDF 端点是 `tool0`。SRDF 没有 `<end_effector>`、`<passive_joint>` 定义，手组只是独立 planning group。

`abbarm` 使用 TRAC-IK：`src/abb120_moveit_config1/config/kinematics.yaml:9-16`；`hand` 没有专用 kinematics 条目。OMPL 同时为 `abbarm` 与 `hand` 配置：`src/abb120_moveit_config1/config/ompl_planning.yaml:167-228`。

### 6.2 Servo 输入输出

| 配置 | 输入 | 输出 | 关键参数/问题 |
|---|---|---|---|
| `servo_abbarm_velocity.yaml` | `/servo_server/delta_twist_cmds`、`/servo_server/delta_joint_cmds` | `/abbarm_velocity_controller/command`，`Float64MultiArray`，joint velocity | 20 ms 发布、0.5 s 输入超时；碰撞关闭；`src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:9-27`、`:34-54` |
| `servo_abbarm.yaml` | 同上 | `/servo_server/raw_joint_cmds`，`JointTrajectory` velocity 字段 | 再由 bridge 积分到 `/controller_gazebo/command`；碰撞关闭；`src/abb120_moveit_config1/config/servo_abbarm.yaml:11-31`、`:38-61` |

位置 bridge 固定使用 `integration_dt=0.12`，而不是按消息时间戳/实际周期积分：`src/handarm_moveit_demo/scripts/servo_velocity_to_position_bridge.py:25-34`、`:101-108`；对应 launch 也固定注入 0.12 s：`src/abb120_moveit_config1/launch/demo_gazebo_servo.launch:20-29`。它只能保留为调试基线，不应成为正式默认链路。

### 6.3 ros_control 控制器

| 控制器 | 类型 | 关节/接口 | 证据 |
|---|---|---|---|
| `abbarm_velocity_controller` | `velocity_controllers/JointGroupVelocityController` | `joint_1..joint_6`；VelocityJointInterface 版 URDF | `src/abb120_moveit_config1/config/ros_controllers_velocity.yaml:1-12`、`config/gazebo_handarm_velocity.urdf:606-665` |
| `controller_gazebo` | `position_controllers/JointTrajectoryController` | `joint_1..joint_6` | `src/abb120_moveit_config1/config/ros_controllers.yaml:10-37` |
| `controller_gazebo_hand` | `position_controllers/JointTrajectoryController` | `f1j1,f1j2,f2j1,f3j2` | `src/abb120_moveit_config1/config/ros_controllers_velocity.yaml:14-28`、`config/ros_controllers.yaml:39-53` |
| `joint_state_controller` | `joint_state_controller/JointStateController` | 50 Hz `/joint_states` | `src/abb120_moveit_config1/config/gazebo_controllers.yaml:1-4` |
| 旧 `handarmtest1/arm_controller` | 位置 JointTrajectoryController | 六轴 + 四个手主动关节混在一个 controller | `src/handarmtest1/config/irb120_3_58_arm_controller.yaml:1-29` |

速度 launch 为 MoveIt 传入 `moveit_controller_manager=ros_control`：`src/abb120_moveit_config1/launch/demo_gazebo_velocity.launch:15-20`。该模式下六轴只有速度 controller，不是 FollowJointTrajectory controller；因此 MoveIt Servo 可直通速度，但普通 `MoveGroupCommander.go()` 的 ABB 轨迹执行能力未验证，标记 `NOT RUN`。手部 FollowJointTrajectory controller 存在。

`simple_moveit_controllers.yaml` 同时列出 `abbarm_controller`、`hand_controller`、`controller_gazebo`、`controller_gazebo_hand`，且四者都 `default: True`：`src/abb120_moveit_config1/config/simple_moveit_controllers.yaml:1-41`。前两个名字与当前 Gazebo spawner 不一致；虽然主 Gazebo 链使用 ros_control manager 而不会加载此文件，但切到 `simple` manager 时存在选择歧义。

## 7. 三指手主动关节、mimic 与抓取能力

### 7.1 主动和 mimic 关系

| 角色 | 关节 | 关系/范围 | 证据 |
|---|---|---|---|
| 主动展合 | `f1j1` | revolute，0..3.14 rad | `src/handarmtest1/xacro/hand.xacro:95-112` |
| 主动弯曲 | `f1j2` | revolute，0..1.3963 rad | `src/handarmtest1/xacro/hand.xacro:153-170` |
| 主动弯曲 | `f2j1` | revolute，0..1.3963 rad | `src/handarmtest1/xacro/hand.xacro:455-472` |
| 主动弯曲 | `f3j2` | revolute，0..1.3963 rad | `src/handarmtest1/xacro/hand.xacro:335-352` |
| mimic | `f1j3 = f1j2` | multiplier 1、offset 0 | `src/handarmtest1/xacro/hand.xacro:211-232` |
| mimic | `f3j1 = f1j1` | multiplier 1、offset 0 | `src/handarmtest1/xacro/hand.xacro:273-294` |
| mimic | `f3j3 = f3j2` | multiplier 1、offset 0 | `src/handarmtest1/xacro/hand.xacro:393-414` |
| mimic | `f2j2 = f2j1` | multiplier 1、offset 0 | `src/handarmtest1/xacro/hand.xacro:513-534` |

Gazebo 实际加载的两份静态 URDF 还显式加载四个 mimic 插件，位置版见 `src/abb120_moveit_config1/config/gazebo_handarm.urdf:672-711`，速度版同样见 `config/gazebo_handarm_velocity.urdf:671-711`。

### 7.2 抓取/接触缺口

- 当前唯一手部高层动作示例只是 MoveIt 命名姿态，不是状态机、自适应闭合或 CAN 逻辑：`src/handarm_moveit_demo/scripts/05_move_hand_named.py:3-7`、`:31-34`。
- 手/臂 xacro 没有 contact sensor、bumper、force/torque 或触觉插件；`hand_g.xacro` 仅添加 4 个位置 transmission 与材质/关闭重力：`src/handarmtest1/xacro/hand_g.xacro:5-43`、`:45-81`。
- 手部碰撞直接复用 STL 网格：`src/handarmtest1/xacro/hand.xacro:37-45`、`:85-93` 等；没有专用简化碰撞体或明确接触摩擦/刚度参数。
- 因而“达到闭合角”不能作为抓取成功；接触数量、持续时间、物体相对位姿、滑移、lift test 与失败超时均为 `NOT RUN`/未实现。

## 8. URDF/SRDF 审计

### 8.1 已通过的静态解析

- `check_urdf` 实测通过：`src/handarmtest1/urdf/arm.urdf`、`src/abb120_moveit_config1/config/gazebo_handarm.urdf`、`config/gazebo_handarm_velocity.urdf`；三者根 link 均为 `world`。
- `xacro` 实测通过：`src/handarmtest1/xacro/arm.xacro`、`arm_g.xacro`。组合入口及 include 位于 `arm.xacro:3-8`、`arm_g.xacro:3-7`。
- `arm.xacro` 当前展开结果与 `src/handarmtest1/urdf/arm.urdf` 在忽略空白后匹配。

这些是 `STATIC PASS`，不证明动力学、碰撞或控制正确。

### 8.2 结构问题

1. **双重 world/base 表达。** URDF 已包含 `world` link 与固定 `world-base_link-fixed`：`src/handarmtest1/urdf/arm.urdf:507-511`；SRDF 又声明 fixed virtual joint `world -> base_link`：`src/abb120_moveit_config1/config/handarm.srdf:48-49`；`demo.launch` 还启动同一静态 TF：`src/abb120_moveit_config1/launch/demo.launch:26-27`。这会造成模型根/virtual-joint 语义不一致和重复 TF 发布风险。
2. **Gazebo xacro 与实际加载 URDF 漂移。** `arm_g.xacro` 包含 `arm_macro_g.xacro`/`hand_g.xacro`：`src/handarmtest1/xacro/arm_g.xacro:3-7`；其中机械臂 transmission 全是 PositionJointInterface：`arm_macro_g.xacro:10-64`，且 `hand_g.xacro` 没有 mimic Gazebo plugin。主速度 launch 却加载手工静态 `gazebo_handarm_velocity.urdf`：`src/abb120_moveit_config1/launch/gazebo_velocity.launch:15-16`，该文件把六轴改为 VelocityJointInterface 并追加 mimic plugins：`config/gazebo_handarm_velocity.urdf:606-711`。当前没有单一可复现的模型源。
3. **位置版静态 URDF 也不是 `arm_g.xacro` 的直接展开。** 文件头标注由 `arm.xacro` 生成，而后又手工含 transmission/plugin；主位置 launch直接 textfile 加载它：`src/abb120_moveit_config1/launch/gazebo.launch:15-16`。xacro 修改可能不会同步到仿真模型。
4. **mimic joint 类型/limit 语义不清。** 四个 mimic 被声明为 `continuous`，同时写 `lower="0" upper="0"`：`src/handarmtest1/xacro/hand.xacro:211-231`、`:273-293`、`:393-413`、`:513-533`。continuous joint 的 position 上下限不生效；真实约束完全依赖 source joint 和 Gazebo mimic plugin。
5. **SRDF 禁碰范围过宽。** 大量手指-手指、手指-机械臂对被标为 `Never`：`src/abb120_moveit_config1/config/handarm.srdf:53-116`。在碰撞检查恢复前必须重新生成/验证允许碰撞矩阵，不能沿用为正式安全结论。
6. **无末端执行器语义。** SRDF 只有两个 group、group states、virtual joint 与 disable-collisions，没有 `<end_effector>`；见完整结构 `src/abb120_moveit_config1/config/handarm.srdf:12-50`。手与臂的语义关联未声明。
7. **手掌与 tool0 是 flange 的两个兄弟分支。** `arm_hand` 把 `handbase_link` 固定到 `flange`：`src/handarmtest1/xacro/hand.xacro:48-53`；`tool0` 也固定到 `flange`：`src/handarmtest1/xacro/arm_macro.xacro:223-235`。Servo 以 `tool0` 为末端：`src/abb120_moveit_config1/config/servo_abbarm_velocity.yaml:9-12`；必须实测 tool0 是否等于期望掌心/工具控制点。
8. **碰撞几何与接触参数不足。** 机械臂有独立 collision STL，但三指手直接用外观 STL；Gazebo 手部逐 link `turnGravityOff=true`：`src/handarmtest1/xacro/hand_g.xacro:45-81`。没有抓取所需接触传感、摩擦或软接触标定证据。

## 9. 依赖审计

### 9.1 当前机器实际可解析

本轮在 source `/opt/ros/noetic/setup.bash` 和 `devel/setup.bash` 后实际检查：`roscpp`、`rospy`、`std_msgs`、`sensor_msgs`、`trajectory_msgs`、`control_msgs`、`geometry_msgs`、`tf`、`tf2_ros`、`moveit_msgs`、`moveit_commander`、`moveit_servo`、`controller_manager`、`gazebo_ros`、`gazebo_ros_control`、`velocity_controllers`、`position_controllers`、`joint_trajectory_controller`、`joint_state_controller`、`trac_ik_kinematics_plugin` 均可由 `rospack find` 解析。

`rosdep check --from-paths src --ignore-src` 实测返回 `All system dependencies have been satisfied`，但它只检查已经声明的依赖，不能发现 manifest 漏声明；同一限制记录于 `docs/02_BASELINE_TEST_REPORT.md:30-33`。

### 9.2 manifest 漏声明

`handarm_moveit_demo/package.xml` 只声明 `rospy`、`moveit_commander`、`moveit_msgs`、`geometry_msgs`、`std_msgs`、`tf`：`src/handarm_moveit_demo/package.xml:9-17`。实际源码还直接 import：

- `sensor_msgs`、`control_msgs`：`src/handarm_moveit_demo/scripts/joint_target_servo_controller.py:6-9`；
- `sensor_msgs`、`trajectory_msgs`：`scripts/servo_velocity_to_position_bridge.py:19-22`；
- `std_srvs`、`visualization_msgs`：`scripts/handarm_trajectory_rviz.py:29-34`。

因此至少漏声明 `sensor_msgs`、`trajectory_msgs`、`control_msgs`、`std_srvs`、`visualization_msgs`。`handarm_moveit_demo/CMakeLists.txt` 的 catkin components 也只有 `rospy/moveit_msgs/geometry_msgs`：`src/handarm_moveit_demo/CMakeLists.txt:4-8`。

`abb120_moveit_config1/package.xml` 声明了 MoveIt、joint-state、RViz、TF2、xacro、handarmtest1 与 moveit_servo：`src/abb120_moveit_config1/package.xml:17-40`，但其 launch 还直接使用 `gazebo_ros`、`gazebo_ros_control`、`controller_manager`、velocity/position/joint trajectory/joint state controllers，以及配置使用 TRAC-IK；其中 Gazebo 两项甚至仅保留为注释：`:32-35`。这些应在该包自己的 manifest 明确声明，而不是依赖其他包传递带入。

包元数据还有两个清洁度问题：`handarmtest1` 的 license 仍为 `TODO`：`src/handarmtest1/package.xml:13-16`；`roboticsgroup_upatras_gazebo_plugins` 的根标签后多出一个文本字符 `>`：`src/roboticsgroup_upatras_gazebo_plugins/package.xml:21`。后者虽被当前 XML/catkin 解析接受，仍应在清洁交付前修正并做 schema 检查。

### 9.3 Python、D455 与 HaMeR

- 系统 Python 3.8 可 import `rospy`、`moveit_commander`、`cv2`、`numpy`，不能 import `pyrealsense2`、`mediapipe`。
- `/home/diu/anaconda3/envs/mediapipe_env` 的 Python 3.10 实测可 import `cv2`、`numpy`、`pyrealsense2`、`mediapipe`；这与脚本的 conda 运行意图一致：`src/handarm_moveit_demo/scripts/d455_conda_udp_sender_servo_v3.py:4-14`、`:33-36`。
- ROS 包 `realsense2_camera` 当前 `MISSING`，且 D455 未出现在 `lsusb`；真实相机 `NOT RUN`。
- P0 快照时仓库中不存在 HaMeR 安装、checkpoint、MANO_RIGHT.pkl、crop inference API 或显存基准代码；当时均为 `NOT RUN`。后续 P1 准备物的现状单独记录于 `docs/03_HAMER_INSTALL_AND_BENCHMARK.md`，不得回写成 P0 通过。

## 10. 已运行与未运行能力矩阵

| 能力/检查 | 状态 | 实际证据 |
|---|---|---|
| GPU 型号/显存/驱动 | PASS | RTX 2060 / 6144 MiB / 570.133.20；`docs/02_BASELINE_TEST_REPORT.md:28` |
| Ubuntu/ROS/Gazebo 版本 | PASS | Ubuntu 20.04.6 / Noetic 1.17.4 / Gazebo 11.15.1；`docs/02_BASELINE_TEST_REPORT.md:29` |
| `catkin_make` | PASS | 返回 0、增量构建插件；`docs/02_BASELINE_TEST_REPORT.md:30` |
| Catkin 自动测试 | NO TESTS | 0 tests；`docs/02_BASELINE_TEST_REPORT.md:35` |
| Python 静态编译 | STATIC PASS | 所有 `src/**/*.py` 可被 Python 3 解析；`docs/02_BASELINE_TEST_REPORT.md:33` |
| package/XML/launch 语法 | STATIC PASS | `xmllint`/launch 解析通过；`docs/02_BASELINE_TEST_REPORT.md:32` |
| 三份 URDF + 两个主 xacro | STATIC PASS | 本轮 `check_urdf`/`xacro` 返回 0；源码入口 `src/handarmtest1/xacro/arm.xacro:3-8`、`arm_g.xacro:3-7` |
| 速度版 Gazebo + MoveIt + Servo 启动 | PARTIAL | 进程共同启动，但不 headless；`docs/02_BASELINE_TEST_REPORT.md:36-47` |
| `/joint_states`、Servo status、控制器 running | PASS | `docs/02_BASELINE_TEST_REPORT.md:37-44` |
| 零 Twist 数值输出 | INCONCLUSIVE | 5 s 未收到速度 command 样本；`docs/02_BASELINE_TEST_REPORT.md:41` |
| 非零 X/Y/Z/Roll/Pitch/Yaw Servo | NOT RUN | `docs/02_BASELINE_TEST_REPORT.md:68` |
| MoveGroup ABB 轨迹执行（速度 controller 模式） | NOT RUN | 无 FollowJointTrajectory ABB controller；配置证据见第 6.3 节 |
| 手部命名动作执行 | NOT RUN | 仅有脚本与 controller，未执行 `05_move_hand_named.py` |
| D455 USB 枚举 | FAIL/NOT PRESENT | 本轮 `lsusb` 未出现 Intel RealSense 设备 |
| D455 RGB-D 对齐/深度质量 | NOT RUN | `docs/02_BASELINE_TEST_REPORT.md:63` |
| 旧 MediaPipe UDP 全链 | NOT RUN | `docs/02_BASELINE_TEST_REPORT.md:67-69` |
| HaMeR/MANO/显存/延迟 | NOT RUN | `docs/02_BASELINE_TEST_REPORT.md:64-65` |
| RGB-D KLT/Kabsch/SE(3) 融合 | NOT RUN | `docs/02_BASELINE_TEST_REPORT.md:66` |
| UDP 丢包/乱序/超时、clutch | NOT RUN | `docs/02_BASELINE_TEST_REPORT.md:69` |
| 三指手接触、自适应抓取、lift test | NOT RUN | `docs/02_BASELINE_TEST_REPORT.md:70` |
| Planning Scene 障碍/碰撞减速和停止 | NOT RUN；当前配置 FAIL | `docs/02_BASELINE_TEST_REPORT.md:71`；碰撞检查关闭见 Servo YAML |
| 真实 ABB | NOT RUN | 按安全要求未连接；`docs/02_BASELINE_TEST_REPORT.md:72` |
| 清洁源码压缩包 | NOT RUN | `docs/02_BASELINE_TEST_REPORT.md:73` |

## 11. P0 风险优先级与进入 P1 条件

### P0 阻塞风险

1. **安全阻塞：** Servo collision checking 关闭；SRDF 禁碰对过宽；Planning Scene 无传感器/障碍输入。不得连接真实 ABB。
2. **控制阻塞：** 非零六轴 Servo 从未实测；零 Twist 输出 inconclusive；Gazebo PID 缺失；速度模式下普通 MoveGroup ABB 轨迹执行未验证。
3. **模型阻塞：** xacro 与两份实际加载 Gazebo URDF 漂移，速度模型没有可生成的单一源文件；world/base 表达重复。
4. **感知阻塞：** D455 当前未连接；现有方案仍是禁止作为正式姿态来源的 MediaPipe + 每帧掌平面；HaMeR/KLT/Kabsch/融合代码不存在。
5. **抓取阻塞：** 只有四关节位置目标与 mimic，无接触传感、状态机、成功/失败判定。
6. **复现阻塞：** manifest 漏依赖、0 自动测试、根目录无 Git、交付目录不干净。

### P0 门槛状态与后续允许范围

P0 的**现状执行基线已记录，但 P0 验收为 `NOT PASS`**。阻断项包括 0 个自动测试、manifest 漏依赖、旧控制入口不拒绝 NaN/Inf、UDP 没有乱序/时间倒退防护，以及正式默认安全配置尚未建立。当前只允许准备离线 HaMeR crop inference、MANO 资源校验和 6 GB GPU 基准工具；这些准备工作不能记作 P1 通过，也不能用于掩盖 P0 缺口。现有 MediaPipe 姿态链、关闭碰撞的 Servo 和无接触手部控制均不得写成正式完成能力。
