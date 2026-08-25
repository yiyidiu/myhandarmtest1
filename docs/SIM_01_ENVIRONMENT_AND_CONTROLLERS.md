# SIM 01 — 环境、控制器与场景同步

日期：2026-08-14  
运行模式：Gazebo 11 + ROS Noetic + MoveIt 1，纯仿真；`simulation=true`、`use_real_robot=false`、`use_real_hand=false`。

## 阶段结论

阶段 1（独立包）、阶段 2（Gazebo 场景）与阶段 3（PlanningScene 同步）均已实际通过。没有启动真实 ABB 或真实手驱动，没有使用相机、感知、遥操作或既有 P2–P5 数据。

## 独立包与启动安全

- 新包：`src/handarm_sim_demo`。
- 主入口：`launch/simulation_baseline.launch`。
- required 进程 `simulation_guard.py` 对任一不安全组合 fail-closed：`simulation=false`、`use_real_robot=true` 或 `use_real_hand=true` 都会以非零码退出并关闭整套 launch。
- 实测安全默认组合可持续运行；实测 `simulation=false` 被拒绝，退出码为 2。
- `gazebo_gui=false` 实测没有 `gzclient` 进程，也没有 `/gazebo_gui` 节点。

## Gazebo 场景

统一坐标、尺寸与任务位姿保存在 `config/demo_scene.yaml`。默认 `double_obstacle` 场景实际包含：

| Gazebo model | type | position (m) | size (m) |
|---|---|---|---|
| work_table | static box | (0.65, 0.00, 0.33) | (0.80, 0.90, 0.08) |
| target_object | dynamic box | 约 (0.5003, 0.0503, 0.4200) | (0.05, 0.05, 0.10) |
| obstacle_a | static box | (0.36, 0.04, 0.545) | (0.08, 0.26, 0.35) |
| obstacle_b | static box | (0.54, -0.20, 0.51) | (0.08, 0.18, 0.28) |

目标物轻微位置差来自落到桌面后的 Gazebo 动力学稳定值；PlanningScene 使用该实时 Gazebo 位姿，不使用另一份硬编码位置。

SDF 由 `gz sdf -k` 实际校验通过。Gazebo 世界实测包含 `ground_plane`、上述四个场景模型与 `robot`。

## 启动顺序与主动等待

`startup_coordinator.py` 不使用固定 sleep 作为同步条件。它依次：

1. 等待仿真安全 guard；
2. 等待 Gazebo 服务与全部场景模型；
3. 在暂停状态加载三个控制器；
4. 解暂停并原子启动控制器；
5. 等待三个控制器均为 `running` 及 `/joint_states`；
6. 通过两个 FollowJointTrajectory action 同步设置确定性初始位姿；
7. 核验实际关节误差；
8. 发布 `/handarm_sim_demo/startup_ready` 与结构化状态。

若 `paused=true`，完成上述初始化后重新暂停物理。

三个实测运行控制器：

- `controller_gazebo`：六轴 `JointTrajectoryController`；
- `controller_gazebo_hand`：四个主动手关节 `JointTrajectoryController`；
- `joint_state_controller`。

初始化 action 实际成功；按 `joint_6` 的 `2π` 等价表示计算最短角距离后，最大初始误差为 `9.3883e-06 rad`。随后 2 秒的最大关节位置差约为 `1.78e-15 rad`。

## PID 与物理真实性

该 URDF 的 PositionJointInterface 在加载试验 PID 后表现出持续重力误差或启动瞬态，实际多次导致 `PATH_TOLERANCE_VIOLATED` / `GOAL_TOLERANCE_VIOLATED`。因此最终基线没有加载 Gazebo PID，使用 Gazebo ROS Control 的直接位置仿真路径。

结果是确定性的运动学/碰撞规划基线，但不是有效的关节力控，也不能据此声称动力学、接触力或物理抓取真实。Gazebo 会打印十条 `No p gain specified`，这是本基线已知且如实保留的限制，不应被误写为 PID 有效。

## PlanningScene 同步

`scene_manager.py`：

- 读取同一 `demo_scene.yaml`；
- 等待 startup ready、Gazebo 模型和 MoveIt 服务；
- 查询每个 Gazebo 模型的实际世界位姿；
- 向 PlanningScene 加入相同名称、盒体尺寸和实际位姿；
- 将 MoveIt 对象级 pose 与 primitive 局部 pose 组合后核验世界位姿；
- 发布 `/handarm_sim_demo/scene_ready` 和 JSON `/handarm_sim_demo/scene_status`。

默认场景实测同步结果：四个对象的位置误差均为 `0 m`，姿态误差均为 `0 rad`，尺寸逐项一致。`target_object` 在抓取前是 world collision object，attached objects 为空。

同步完成后调用 `/check_state_validity`：

- `abbarm`: `valid=true`，0 contacts；
- `hand`: `valid=true`，0 contacts。

## 三指手独立控制实测

统一接口 `hand_commander.py` 支持 `OPEN`、`CLOSE`、`PRE_SHAPE_A`、
`PRE_SHAPE_B`、`HOLD`、`STOP`。四个主动关节目标只存在
`hand_commands.yaml`，mimic 关系按运行 URDF 核验。Gazebo 将固定的
`tool0/handbase_link` 链折叠，因此 tip 位姿用 world frame 查询；用于开闭判定的
三指 tip 两两距离与参考坐标系无关。

实际执行 3 组序列、每组 3 个循环，共 27/27 条命令成功：

| command | samples | execution median (s) | active max error (rad) | mimic max error (rad) | mean tip spacing median (m) |
|---|---:|---:|---:|---:|---:|
| OPEN | 18 | 1.7802 | 9.3883e-06 | 9.3883e-06 | 0.154818 |
| CLOSE | 3 | 1.7802 | 9.3398e-06 | 4.6149e-06 | 0.056343 |
| PRE_SHAPE_A | 3 | 1.7785 | 8.7682e-06 | 5.5592e-06 | 0.056907 |
| PRE_SHAPE_B | 3 | 1.7995 | 7.3754e-06 | 7.3754e-06 | 0.129295 |

`PRE_SHAPE_A/B` 只标为来自 SRDF `grasp1/grasp2` 的中性命名状态，不推断其
物理抓取含义。CLOSE 明显缩小 tip 间距，但这只是运动学开闭证据，不是接触抓取。
机器结果见 `results/sim_baseline/hand_cycles_20260814T060847818164Z.json`。

## 本阶段之后仍未运行

- deterministic attachment：NOT AVAILABLE（无现有附着服务）；
- 物理接触抓取与力矩/摩擦验证：NOT RUN；
- GUI 截图和视频：本轮采用 headless 实测，NOT RUN。
