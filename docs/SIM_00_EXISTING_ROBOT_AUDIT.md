# SIM 00 — 现有机器人仿真审计

日期：2026-08-14（Asia/Shanghai）  
范围：只读检查既有 ABB IRB120、三指手、Gazebo、MoveIt 配置；未运行真实机器人或真实手驱动。

## 结论

现有模型可以作为纯仿真基线的几何与控制配置来源，但不能原样作为新自主抓取入口。实测 Gazebo、`move_group`、机械臂控制器和手控制器均能启动，当前初始状态在 `abbarm` 与 `hand` 两组中均无碰撞；同时发现 headless 参数失效、十个主动关节缺少 `gazebo_ros_control` PID、旧启动文件重复表达 `world -> base_link`、MoveIt 控制器默认项存在歧义等问题。

本任务将新增独立包 `handarm_sim_demo`，复用既有机器人描述及 MoveIt 配置，不修改既有包；新入口默认且仅允许仿真。

## 工作区与冻结范围

- 工作区根目录不是 Git 仓库，因此不能依赖根目录 `git status` 或 commit 回滚。
- 已在修改前归档 `src/`：`backups/sim_autonomous_baseline_single_agent_20260814/src_before.tar.gz`。
- 归档 SHA-256：`080a09f41ba05282d6a5be4651888a86500a0d563657a6bd74833030348598b9`。
- 归档排除了嵌套仓库元数据 `src/roboticsgroup_upatras_gazebo_plugins/.git`，未包含 `build/`、`devel/`、日志、数据集、模型或临时文件。
- 冻结目录 `perception_hamer/` 与 `datasets/` 的初始文件状态摘要：`194919a9a95814a9ce2b20d2985cdab8ca67e2a2968b1edcc7cb197c9e4fc111`。
- 冻结报告 SHA-256：
  - `docs/P3_P4_HAMER_PALM_STABILITY.md`：`613c853d2f6ba4a7d4328ac3b42c9d2dbcd2d6811728f1ab904f6722a2d67874`
  - `docs/P5_RGBD_RELATIVE_ORIENTATION.md`：`c52f67433dc33c6e6c55af64555e64a78a4427c6c0b06d34ab5f8dcb11ccb534`

## 实际运行模型

- 既有 Gazebo 入口加载的是静态文件 `src/abb120_moveit_config1/config/gazebo_handarm.urdf`，不是 `hand_g.xacro`。
- MoveIt 机器人模型名为 `handarm`，规划坐标系为 `world`。
- 机械臂规划组：`abbarm`，链为 `base_link -> tool0`，六个主动关节 `joint_1` 至 `joint_6`，TRAC-IK 可加载。
- 手规划组：`hand`，四个主动关节为 `f1j1`、`f1j2`、`f2j1`、`f3j2`；MoveIt 中没有配置手的 end-effector link。
- 手的四个 mimic 关系均为 multiplier `+1`、offset `0`：`f3j1 <- f1j1`、`f1j3 <- f1j2`、`f2j2 <- f2j1`、`f3j3 <- f3j2`。
- `handbase_link` 与 `tool0` 都通过固定结构连接在腕部附近；`tool0` 是机械臂规划末端，模型中不存在名为 `palm` 的 link。抓取位姿必须明确以 `tool0` 规划，并单独记录手掌几何偏置。

## 命名状态与关节范围

机械臂已有命名状态 `up`：

| joint | value (rad) | lower (rad) | upper (rad) |
|---|---:|---:|---:|
| joint_1 | 0.4675 | -2.87979 | 2.87979 |
| joint_2 | -0.4737 | -1.91986 | 1.91986 |
| joint_3 | -0.3083 | -1.91986 | 1.22173 |
| joint_4 | -0.7072 | -2.79253 | 2.79253 |
| joint_5 | -0.3808 | -2.094395 | 2.094395 |
| joint_6 | -0.4533 | -6.98132 | 6.98132 |

手已有命名状态：

| state | f1j1 | f1j2 | f2j1 | f3j2 |
|---|---:|---:|---:|---:|
| start1 | 0.0510 | 0.0317 | 0.0227 | 0.0363 |
| grasp1 | 3.1196 | 0.9520 | 0.9158 | 0.9067 |
| grasp2 | 1.5700 | 0.6664 | 0.6664 | 0.6664 |

四个手主动关节的下限均为 `0`；`f1j1` 上限为 `3.14`，其余三个上限为 `1.3963`，标称速度上限均为 `4 rad/s`。仅凭 URDF/SRDF 不能断言 `grasp1` 或 `grasp2` 对应哪种物理预抓取；新基线必须通过 Gazebo 外观、关节轨迹和接触表现验证后再命名。

## 控制器与 MoveIt

- `/controller_gazebo`：`position_controllers/JointTrajectoryController`，控制六个机械臂关节，运行态 action 为 `/controller_gazebo/follow_joint_trajectory`。
- `/controller_gazebo_hand`：同类型控制器，控制四个手主动关节，运行态 action 为 `/controller_gazebo_hand/follow_joint_trajectory`。
- `/joint_state_controller` 运行并发布关节状态。
- MoveIt 运行态识别到 `abbarm` 与 `hand` 两组；`abbarm` 的 end-effector link 为 `tool0`，`hand` 的 end-effector link 为空。
- `move_group`、`/apply_planning_scene`、`/get_planning_scene` 与两个轨迹 action 均实际可用。
- 当前 MoveIt 简单控制器配置中多个控制器被标为默认，存在选择歧义；新仿真入口将显式使用 Gazebo 的两个轨迹控制器。

## 阶段 0 实测

使用命令：

```bash
roslaunch abb120_moveit_config1 demo_gazebo.launch gazebo_gui:=false use_rviz:=false paused:=false
```

实际观察：

- Gazebo 成功生成 `robot`；世界中只有 `ground_plane` 与 `robot`。
- `move_group` 成功初始化 OMPL、CHOMP 与 Pilz pipeline，默认规划器为 OMPL。
- 三个控制器均报告 `running`。
- `abbarm` 与 `hand` 当前状态分别调用 `/check_state_validity`，结果均为 `valid=true`，接触数为 0。
- `tool0` 运行态位置约为 `(0.1166, 0.1222, 0.9976) m`；`handbase_link` 约为 `(0.0630, 0.0505, 0.8531) m`。这些仅是启动状态测量，不作为后续场景目标常量。
- 以 SIGINT 正常关闭审计实例后，ROS master、Gazebo、`move_group` 均无残留。

## 已确认问题

1. `gazebo_gui:=false` 没有被映射到 `gazebo_ros/empty_world.launch` 的 `gui` 参数，实测仍启动 `/gazebo_gui`（`gzclient`）。
2. `gazebo_ros_control` 实测报告十个主动关节均缺少 `/gazebo_ros_control/pid_gains/<joint>/p`。控制器虽进入 running，但该告警不能作为合格控制基线。
3. 旧 `demo.launch` 总是发布静态 `world -> base_link`，而当前 URDF 已含 world/base 固定结构；存在重复 TF 表达风险。
4. 现有 manifest 未完整声明 Gazebo 控制器运行依赖，且 Gazebo 相关依赖仍被注释。
5. 三指手没有接触传感器、抓取判定或闭环力控；mimic 插件的运动不等于物理抓取成功。
6. 旧世界没有桌面、目标物与障碍物，PlanningScene 也未同步这些对象。

## 阶段判定

阶段 0：**PASS（审计基线）**。

该 PASS 只表示现状、接口与已知风险已经通过静态和运行态审计，不表示避障、抓取或物理持物已通过。下一阶段允许创建独立 `handarm_sim_demo` 包；不得把上述旧启动入口直接包装成最终交付。
