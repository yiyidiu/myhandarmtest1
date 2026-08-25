# SIM 02 — 静态障碍规避基线

日期：2026-08-14  
规划组：`abbarm`；末端：`tool0`；pipeline：OMPL；planner：现有配置的 `RRTConnect`。

## 结论

阶段 4 实际通过。规定的 3/10/10 个可达试次全部完成规划、逐轨迹点碰撞验证、执行、末端误差核验和返回 home；不可达与完全阻挡场景均安全失败，未执行空轨迹或旧轨迹。

## 状态机与失效语义

实际状态序列：

`INIT -> WAIT_FOR_ROBOT -> WAIT_FOR_CONTROLLERS -> WAIT_FOR_SCENE -> MOVE_HOME -> SET_GOAL -> PLAN -> VALIDATE_PLAN -> EXECUTE -> VERIFY_GOAL -> RETURN_HOME -> DONE`

任何异常进入 `FAILED`，立即 `stop()`、清空 pose target 且不执行后续动作。每次规划都使用新的局部 trajectory 对象；规划失败或轨迹点为空时执行路径不可达。

每个非空计划还执行以下门禁：关节名已知、数值有限、时间严格递增、每个离散轨迹点调用 `/check_state_validity` 均无碰撞。执行后使用位置欧氏距离和四元数 SO(3) 测地角核验 `tool0`。

## 场景与目标

目标位姿来自 `demo_scene.yaml`：位置 `(0.44, -0.15, 0.80) m`，朝向与 home 时 `tool0` 朝向一致。无障碍场景只保留桌面与目标物；单障碍保留 `obstacle_a`；双障碍同时保留 A/B。`scene_manager` 同时从 Gazebo 删除不活动模型并从 PlanningScene 移除同名对象，避免两边场景不一致。

在单/双障碍场景中，起点到目标终态的直线关节插值实际有约 39–45/101 个无效样本；OMPL 生成的轨迹点全部有效。因此该测试确实需要绕障，不是障碍未参与规划。

不可达目标设为 `(1.50, 0.00, 1.50) m`。完全阻挡目标位于 `obstacle_a` 内部 `(0.36, 0.04, 0.545) m`。两者都预期在 PLAN 阶段安全失败；即使意外得到有效计划，脚本也明确禁止在 `safe_failure` 场景执行。

## 实际结果

| scenario | trials | plan success | execution success | return home | expected safe failure |
|---|---:|---:|---:|---:|---:|
| no_obstacle | 3 | 3/3 | 3/3 | 3/3 | n/a |
| single_obstacle | 10 | 10/10 | 10/10 | 10/10 | n/a |
| double_obstacle | 10 | 10/10 | 10/10 | 10/10 | n/a |
| unreachable | 2（含修复前后复测） | 0/2 | 0/2，未尝试执行 | n/a | 2/2 |
| fully_blocked | 2（含修复前后复测） | 0/2 | 0/2，未尝试执行 | n/a | 2/2 |

单障碍与双障碍的规划成功率、执行成功率均为 `100%`，高于 90% 目标。

可达场景指标：

| metric | no obstacle | single obstacle | double obstacle |
|---|---:|---:|---:|
| planning time median (s) | 0.1213 | 0.1263 | 0.1292 |
| planning wall time median (s) | 0.2213 | 0.2267 | 0.2272 |
| trajectory points median | 25 | 45 | 37 |
| straight-path invalid samples median / 101 | 0 | 41.5 | 42 |
| final position error max (m) | 0.01430 | 0.01457 | 0.01373 |
| final orientation error max (deg) | 3.803 | 3.932 | 3.660 |

失败规划的 MoveIt Python tuple 会携带未初始化的 `planning_time` 浮点值；脚本已修复为失败时写 `null`，真实耗时使用 `planning_wall_time_s`。最终不可达和完全阻挡复测的 wall time 均约 8.02 秒、轨迹点为 0、`execution_attempted=false`。

## 工件

- 逐试次 CSV：`results/sim_baseline/avoidance_trials.csv`
- 每次 launch 的 JSON：`results/sim_baseline/avoidance_<scenario>_<run_id>.json`
- 原始 ROS/Gazebo/MoveIt 日志：`results/sim_baseline/avoidance_*.log`

## 最终命令复核

2026-08-14 在所有旧 ROS/Gazebo 进程退出后重新串行执行：

| 场景 | 模式 | 轨迹点 | 直线无效采样 | 位置误差 | 姿态误差 | 结果 |
|---|---|---:|---:|---:|---:|---|
| single_obstacle | headless | 50 | 46/101 | 12.43 mm | 1.30° | DONE，return home |
| double_obstacle | headless | 39 | 42/101 | 14.44 mm | 1.43° | DONE，return home |
| double_obstacle | Gazebo + RViz | 43 | 42/101 | 10.66 mm | 2.55° | DONE，return home |

三次均 `trajectory_collision_free=true`、`execution_success=true`、
`return_home_success=true`。GUI 复核使用的正是最终用户命令，任务结束后窗口保持，
随后从同一终端 Ctrl-C 正常释放。对应最新证据为：

- `avoidance_single_obstacle_20260814T075257387617Z.json`
- `avoidance_double_obstacle_20260814T075407386209Z.json`
- `avoidance_double_obstacle_20260814T075608163739Z.json`

最终无冲突运行顺序见 `docs/SIM_FINAL_RUN_COMMANDS.md`。

## 运行中发现并修复的问题

1. 任务节点曾在 `move_group` 约 24 秒初始化完成前构造 commander；现显式等待 `/move_group` action 最多 90 秒。
2. `RRTConnectkConfigDefault` 不匹配本工程的 planner key；改为已有 `RRTConnect` 后不再回退默认配置。
3. 直接位置仿真存在偶发启动瞬态；协调器现在线控器启动后再次调用 `/gazebo/set_model_configuration` 复位初始关节，再执行保持轨迹。
4. 外层一次冷启动探针在 ROS master 出现前调用 `rostopic`，属于探针时序失败，未计入机器人测试；残留 rosmaster 已清理。

## 限制

这些结果证明静态已知盒体环境下的 MoveIt 碰撞规避与 Gazebo 轨迹执行基线，不证明动态避障、真实硬件安全、动力学精度或物理抓取。机械手与 scripted pick 的后续结果分别见 SIM 01 和 SIM 03。
