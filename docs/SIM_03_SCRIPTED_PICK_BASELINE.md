# SIM 03 — 已知物体位姿预抓取与接近基线

日期：2026-08-14  
模式：默认 `deterministic_lift`，兼容 `approach_only`；纯仿真；没有视觉输入或物理抓取声明。

## 结论

Stage 6 与确定性仿真抬升均实际通过。默认一键 launch 从冷启动完成：Gazebo/机器人/
控制器/move_group/PlanningScene 就绪后，保持启动时手指角度不变，机械臂到达预抓取
位姿，执行完整 Cartesian 短距离接近，创建 Gazebo 固定关节，将目标同步转为 MoveIt
attached collision object，再向上抬升 0.10 m 并保持附着。

这是可重复、可见的仿真搬运基线，不是摩擦、接触力或夹持稳定性证明，不得写成实体
物理抓取。旧 `approach_only` 仍可显式选择。

## 几何与碰撞约定

目标物位姿从 `demo_scene.yaml` 的 `target_object` 已知配置读取。工具目标不是直接复制
物体位姿，而是：

`T_world_pregrasp = T_world_object * T_object_pregrasp`

其中 `T_object_pregrasp` 存在 `grasp_demo.yaml`：平移 `(0, 0, 0.36) m`，姿态为
经实际可达性验证的 `tool0` 抓取朝向。接近向量为 world frame 的
`(0, 0, -0.10) m`。这使 `tool0` 从 `z=0.78 m` 接近到 `z=0.68 m`；目标盒顶面
约为 `z=0.47 m`，其余距离对应实际手掌/手指几何。

正式参数冻结前实际探测显示：该 0.10 m 接近的碰撞感知 Cartesian fraction 为 1.0；
继续下降则路径开始被作为 world collision object 的目标物截断。因此没有关闭碰撞、
删除目标物或执行不完整路径。

## 状态机和 fail-closed 行为

默认 `deterministic_lift` 的成功状态序列：

`INIT -> WAIT_FOR_ROBOT -> WAIT_FOR_SCENE -> RESET_ROBOT -> HOLD_INITIAL_HAND ->`
`PLAN_PREGRASP -> EXECUTE_PREGRASP -> APPROACH -> ATTACH_OBJECT ->`
`PLAN_AND_EXECUTE_LIFT -> VERIFY_LIFT -> LIFT_OR_STOP -> DONE`

只有兼容模式 `approach_only` 才执行 `OPEN_HAND -> ... -> CLOSE_HAND ->`
`VERIFY_HAND_COMMAND -> LIFT_OR_STOP -> DONE`。

- 显式等待 startup/scene ready、两个 trajectory controller 和 move_group；
- 规划失败、空轨迹、非有限关节值、非递增时间或任一轨迹点碰撞时禁止执行；
- Cartesian `fraction < 0.95` 时禁止执行；
- 末端误差或手关节/mimic 核验失败时 STOP 并进入 FAILED；
- `approach_only` 从不抬升、不附着；`deterministic_lift` 使用固定关节抬升，但始终
  记录 `physical_grasp_claimed=false`。

## 确定性抬升实测

证据：`results/sim_baseline/pick_deterministic_lift_20260814T073643950118Z.json`。

- 状态完整到达 `ATTACH_OBJECT -> PLAN_AND_EXECUTE_LIFT -> VERIFY_LIFT -> DONE`；
- 手型为 `initial_configuration_held`，未执行 OPEN/CLOSE；
- 附着瞬时位移 `0.0101 mm`；
- lift Cartesian fraction `1.0`，22 个轨迹点，逐点碰撞校验通过；
- 工具抬升末端误差 `0.0091 mm`；
- 物体从 `z=0.420007 m` 到 `z=0.519985 m`，实际抬升 `99.978 mm`；
- 物体与工具位移差 `0.0155 mm`；
- `attachment_used=true`，`physical_grasp_claimed=false`。

首轮宽规划容差试验在预抓取产生 6.26° 姿态误差，脚本正确停止，未接近或闭合。
随后收紧 MoveIt 目标容差而没有放宽 6° 验收阈值，复测通过。

最终 GUI 命令复核证据：
`results/sim_baseline/pick_deterministic_lift_20260814T074027362134Z.json`。

- Gazebo 与 RViz 同时运行，任务到达 `DONE` 后保持窗口开放；
- pregrasp 与 approach 均成功，approach fraction `1.0`；
- 四个主动手指关节保持启动值，没有 OPEN/CLOSE 手型切换；
- 目标物现场查询高度为 `z=0.519985 m`，相对起点抬升约 `0.100 m`；
- 结束后以同一 launch 终端 Ctrl-C 正常释放全部 ROS/Gazebo 节点。

## 三次连续实际结果

证据：`results/sim_baseline/pick_approach_only_20260814T061716715510Z.json`。

| trial | pregrasp position error (mm) | pregrasp orientation error (deg) | Cartesian fraction | approach position error (mm) | approach orientation error (deg) | final |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.975 | 1.053 | 1.0 | 0.0080 | 0.00090 | DONE |
| 2 | 2.700 | 0.725 | 1.0 | 0.0080 | 0.00090 | DONE |
| 3 | 1.264 | 0.994 | 1.0 | 0.0080 | 0.00090 | DONE |

三次均 `pregrasp_plan_success=true`、逐点碰撞校验通过、OPEN/CLOSE 验证通过，
`attachment_used=false`、`physical_grasp_claimed=false`。另一次最终
`scripted_pick_demo.launch` 冷启动也完整 DONE，证据文件为
`pick_approach_only_20260814T061926804910Z.json`。
冷启动 ROS 日志已归档为 `pick_cold_launch_roslaunch.log` 与
`pick_cold_launch_node.log`。

## NOT RUN / NOT AVAILABLE

- deterministic fixed-joint attachment and 0.10 m lift：PASS；
- physical-contact grasp、摩擦/接触力与负载抬升：NOT RUN；
- RViz/Gazebo GUI 运行与动作：PASS；截图/视频归档：NOT RUN；
- 动态障碍、视觉识别、D455、HaMeR、KLT/Kabsch、遥操作、真实机器人：NOT RUN，且不属于本轮。

## 运行命令

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
roslaunch handarm_sim_demo scripted_pick_demo.launch \
  gazebo_gui:=true \
  rviz:=true \
  repetitions:=1
```

`deterministic_lift` 已是默认值，该模式只允许 `repetitions:=1`。旧行为可显式传
`grasp_mode:=approach_only`。完整的无冲突运行顺序见
`docs/SIM_FINAL_RUN_COMMANDS.md`。
