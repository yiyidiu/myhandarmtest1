# 仿真自主基线最终 Changelog

日期：2026-08-14  
执行：单 Agent 串行；工作区不是 Git 仓库。

## 工作区保护

- 修改前源码快照：`backups/sim_autonomous_baseline_single_agent_20260814/src_before.tar.gz`
- SHA-256：`080a09f41ba05282d6a5be4651888a86500a0d563657a6bd74833030348598b9`
- 冻结视觉目录初始内容哈希：`194919a9a95814a9ce2b20d2985cdab8ca67e2a2968b1edcc7cb197c9e4fc111`
- 冻结报告最终哈希：P3/P4 `613c853d...d67874`；P5 `c52f6743...ccb534`，与任务开始时一致。
- 本轮没有修改 `perception_hamer/`、`datasets/`、P2 recorder/verifier 或 P3–P5 报告。

## 逐文件变更

| 文件 | 修改前问题 | 修改后逻辑 | 实际测试 | 剩余风险 |
|---|---|---|---|---|
| `CMakeLists.txt` | 无独立包 | 声明依赖并安装全部脚本/配置/launch/world | catkin build PASS | 依赖 ROS Noetic |
| `package.xml` | 无 manifest | 完整声明 MoveIt/Gazebo/controller/ROS 依赖 | catkin 解析 PASS | Gazebo Classic 已弃用 |
| `README.md` | 无入口说明 | 写明仿真安全默认、四个入口和非物理边界 | 人工核对 | 不替代报告 |
| `worlds/handarm_pick_obstacle.world` | 无统一任务世界 | 地面、桌、目标、A/B 障碍集中定义 | `gz sdf -k` PASS；实际生成 PASS | 简单盒体 |
| `config/demo_scene.yaml` | 坐标散落/缺任务场景 | 集中对象、场景集合、避障目标和抓取距离 | YAML + Gazebo/PlanningScene 实测 | 已知静态位姿 |
| `config/startup_configuration.yaml` | 启动状态不确定 | 集中 arm/hand 初始关节与周期关节 | 初始最大误差 9.39e-06 rad | 直接位置仿真 |
| `config/trajectory_controllers.yaml` | 新包无 MoveIt controller 映射 | 映射 arm/hand FollowJointTrajectory | 两 controller running/executed | 无 controller-side 动力学证明 |
| `config/avoidance_demo.yaml` | 无统一规划验收参数 | RRTConnect、速度、误差和 home 阈值集中配置 | 3/10/10 成功 | 静态盒体限定 |
| `config/hand_commands.yaml` | 手角度会散落 | 公开接口仅 GRASP/RELEASE；两者掌型关节目标相同，只改变三路屈伸；限位和 mimic 关系集中配置 | 5 循环 15/15 空载稳定性 PASS | 不证明物理夹持 |
| `config/grasp_demo.yaml` | 无工具相对预抓取契约 | 定义 `T_object_pregrasp`、0.10 m approach、0.10 m lift 和固定附着契约 | approach 3/3；deterministic lift PASS | 非物理接触抓取 |
| `launch/simulation_baseline.launch` | 旧链 headless/安全/顺序不可靠 | guard、暂停加载、主动就绪、可选 MoveIt、任务完成默认保留 GUI | 安全正反例、headless 与 GUI 实测 | 同一时刻仅允许一套完整 launch |
| `launch/avoidance_demo.launch` | 无一键避障 | 封装 baseline 与场景/次数参数 | 批量实际运行 | 无动态障碍 |
| `launch/hand_demo.launch` | 无一键手循环 | 封装 3 组 × N 循环 | 3 cycles 实测 | 无接触验证 |
| `launch/scripted_pick_demo.launch` | 无一键抓取链 | 固定 no-obstacle known-pose 场景，默认确定性抬升 | 冷启动 DONE | 每次仅一个 lift trial |
| `scripts/simulation_guard.py` | 可能误接硬件 | 三个安全参数 fail-closed | 安全值 PASS；`simulation=false` 拒绝 | 仅保护本包 launch |
| `scripts/startup_coordinator.py` | 控制器/模型/初态竞态 | 服务等待、暂停加载、复位、action、10 个主动旋转关节最短角误差验证 | cold start PASS，覆盖偶发 ±2π 表示 | 无 Gazebo PID |
| `scripts/scene_manager.py` | Gazebo 与 MoveIt 可漂移 | 查询 Gazebo 实际 pose，组合 primitive pose，逐项核验 | 位置/姿态误差 0 | 只支持当前简单几何 |
| `scripts/scripted_avoidance_demo.py` | 无 fail-closed 自主绕障 | 状态机、逐点碰撞验证、执行后误差和 return home | 23/23 可达运行成功；2 个负例安全失败 | OMPL 随机性仍存在 |
| `scripts/hand_commander.py` | 无统一手接口 | FollowJointTrajectory + active/mimic 实际核验；拒绝公开构型命令 | GUI RELEASE/GRASP/RELEASE PASS | 尚未做物体接触闭合 |
| `scripts/run_hand_cycle_tests.py` | 无重复性/几何证据 | 三序列循环、world tip 查询、帧不变距离 | 27/27 PASS | 不测接触力 |
| `scripts/scripted_pick_demo.py` | 无 known-pose task chain | 保持初始手型、预抓取、接近、固定附着、attached collision、抬升和数值核验 | 物体实际抬升 99.978 mm | 非摩擦/接触抓取 |
| `src/deterministic_attach_world_plugin.cpp` | 无 attach 服务 | 仿真线程内创建/拆除跨模型 fixed joint，ROS 服务同步回执 | attach/detach smoke PASS，瞬移 0.0043 mm | 仅 Gazebo Classic 仿真 |
| `scripts/task_monitor.py` | 工件分散 | 选择通过的规范证据并生成 summary/CSV | 输出自校验 PASS | 仅离线聚合 |
| `test/test_sim_algorithms.py` | 无新包单测 | 场景、四元数、pose、手限位、GRASP 不改掌型、同步 URDF/xacro 动力学、后步心跳和异步服务伪影 fail-closed | 33/33 PASS | 不替代 Gazebo 接触实验 |

## 文档和结果

- `SIM_00_EXISTING_ROBOT_AUDIT.md`：修改前实际模型/控制器审计。
- `SIM_01_ENVIRONMENT_AND_CONTROLLERS.md`：场景、启动、同步与手循环。
- `SIM_02_AVOIDANCE_BASELINE.md`：3/10/10 成功与两个安全失败场景。
- `SIM_03_SCRIPTED_PICK_BASELINE.md`：3 次 approach-only、冷启动和附着拒绝。
- `SIM_KNOWN_LIMITATIONS.md`：非物理、非硬件、非视觉边界。
- `results/sim_baseline/summary.json`：机器可读汇总。
- `avoidance_results.csv`、`pick_results.csv`、逐运行 JSON 和 ROS 日志：实际原始证据。

截图和视频没有实际录制，均明确为 `NOT RUN_HEADLESS`；未生成占位媒体。
