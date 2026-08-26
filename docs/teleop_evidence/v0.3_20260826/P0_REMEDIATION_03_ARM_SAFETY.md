# P0 remediation 03: constraint-complete simulated arm safety path

本阶段修复 2026-08-26 真人遥操作证据中暴露的机械臂安全链路缺失。结论只覆盖
Gazebo 中的机械臂自碰撞、关节限位、场景同步和状态失联闭锁；它不代表真人遥操作
已经合格，也不构成 ABB 真机安全认证。

## 原始问题

v0.3 的公开 EGM 启动入口默认使用自定义 Cartesian IK 到位置参考的路径，绕过了
MoveIt Servo；备用 Servo 配置又关闭了碰撞检查。SRDF 还以 `Never`/`Default`
为理由屏蔽了大量非相邻碰撞对，Gazebo 中存在的地面和物体也没有可靠地同步到
MoveIt PlanningScene。因此正式录像中碰撞速度缩放始终为 1，并不能说明整段动作
安全，只说明保护没有实际参与输出。

另一个容易遗漏的执行问题是：EGM 仿真使用位置参考控制。碰撞时仅将速度命令置零，
并不能消除已经领先于反馈的位置参考，也不能立即消除机械臂的惯性运动。

## 修复后的默认链路

公开入口保持不变：

```bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch
```

默认控制链现在为：

```text
human target
  -> relative-pose controller
  -> MoveIt Servo (scene/self collision, singularity, joint limits)
  -> raw arm qdot
  -> strict swept MoveIt/FCL gate
  -> bounded EGM position-reference adapter
  -> Gazebo arm position plant
```

关键改动如下：

- MoveIt Servo 默认启用 60 Hz 碰撞检查，并恢复可用的奇异性阈值和
  `0.08 rad` 关节限位余量；直接 IK 路径只保留为显式仿真回滚选项。
- SRDF 只保留 15 个真实运动学相邻对和 4 个有说明的结构近邻对。严格保护器会
  重新启用后 4 对的二值自碰撞检查，避免 Servo 的距离缩放被装配近邻永久压低。
- 新增完整机械臂—机械手状态保护器。它合并 6 个臂关节和 4 个独立手指关节，
  使用 MoveIt/FCL 检查当前状态、Servo 命令未来、手指参考未来和测量惯性滑行未来。
- 每条预测轨迹使用 `0.40 s` 前视，关节采样间距不超过 `0.01 rad`；命令、
  `/joint_states` 或必要安全监视器过期时均 fail closed。
- 掌部使用简化的 8 mm 碰撞网格，Gazebo 中 16 个臂/手 link 全部启用
  `selfCollide`，作为 MoveIt/FCL 之外的物理后备层。
- EGM 参考增加逐关节 following-error leash。进入硬安全停车时只捕获一次实测位置，
  随后冻结该参考并清零前馈；不再每个 4 ms 周期追随正在惯性滑行的反馈。
- 场景管理器将地面静态代理、目标物和左右障碍物同步到 PlanningScene。控制节点在
  `/handarm_sim_demo/scene_ready=true` 前保持锁定，场景就绪信号丢失会立即撤销控制。

## 定位过程中排除的错误方案

本阶段保留以下失败过程，防止以后重复引入同类缺陷：

1. 直接调用 `RobotState::satisfiesBounds()` 会同时检查保存在状态中的测量速度，
   因而把位置合法但瞬时速度估计偏大的状态误报为越界。最终改为只检查位置边界，
   测量速度只用于惯性滑行预测。
2. 单纯把预测前视从 `0.20 s` 加到 `0.40 s` 仍会发生实际接触。根因不是前视长度，
   而是适配器在安全停车期间每个周期都把位置参考重新锚定到移动反馈，等价于让目标
   跟着惯性运动，消除了用于制动的 P/D 误差。
3. 临时关闭位置控制器积分项后现象不变，因此排除积分饱和为主因，并恢复原有有限
   积分与 anti-windup 配置。
4. 对所有碰撞对统一增加 30 mm 几何余量会在安全边界前过早停车，并使反向恢复也
   无法通过，形成死锁。该启发式余量已完全移除，最终采用连续扫掠预测和允许恢复的
   固定安全参考。

## 动态验收结果

验收命令会从安全初始位姿驱动到一个已知自碰撞目标，再退回初始位姿；随后驱动
`joint_1` 接近上限并再次退回。最终运行结果：

```bash
roslaunch handarm_moveit_demo egm_servo_safety_acceptance.launch
```

| 验收项 | 结果 | 关键测量 |
|---|---|---|
| 自碰撞方向接近 | PASS | 实际运动 `0.695302 rad`；在接触前由 `base_link,handbase_link` 的第 `17/19` 个扫掠样本阻断 |
| 自碰撞后恢复 | PASS | 回到初始位姿的最大关节误差 `0.009748 rad`；最终状态 FCL 有效 |
| 关节上限方向接近 | PASS | 实际运动 `2.714425 rad`；距物理上限仍有 `0.174980 rad` 时预测到越界并阻断 |
| 关节上限后恢复 | PASS | 回到初始位姿的最大关节误差 `0.009654 rad`；最终状态有效 |
| 总体判定 | PASS | `DYNAMIC_ARM_SAFETY_ACCEPTANCE PASS` |

自碰撞试验中 MoveIt Servo 的距离缩放仍为 1，而严格扫掠保护器提前阻断命令。这一
结果说明二者是互补层，也证明不能再以单一 collision-scale 指标代替完整安全验收。

## 失联和公开入口验收

- 速度命令停止后，保护器发布 `RAW_VELOCITY_COMMAND_TIMEOUT`、
  `command_safe=false`；EGM 诊断进入 `SAFETY_HOLD`，六轴前馈速度全部为零。
- 运行中停止 `joint_state_controller` 后 `0.60 s` 内观察到
  `JOINT_STATE_TIMEOUT`、`safe=false` 和 `/shared_teleop/emergency_stop=true`。
- 公开真人入口无头启动时，`scene_ready=true`；地面、目标物、左右障碍物的
  PlanningScene 位置和姿态误差最大值均为 0。未收到相机数据且未按 C 时，控制诊断
  为 `WAITING_FOR_OPERATOR_C_REFERENCE`。

## 自动化验证

- `catkin_make -DCATKIN_ENABLE_TESTING=ON`：通过。
- `catkin_make run_tests && catkin_test_results --verbose`：107 项，0 error，
  0 failure，0 skipped。
- 完整 Python discovery：`handarm_moveit_demo` 104 项、`handarm_sim_demo`
  122 项，全部通过。
- 修改脚本的 Python 编译、launch XML/YAML 回归和 `git diff --check`：通过。

## 尚未完成的边界

- 尚未完成动态外部障碍物接近、奇异位形连续性和带载接触抓取的独立验收。
- 没有在 ABB 真机 EGM 上验证制动距离、网络延迟、驱动器状态或硬件急停。
- 本阶段没有提高 HaMeR 的约 `4.54 Hz` 输入频率，也没有实现人手手指到机器人手指
  的优化式 retargeting。
- 因此当前只能判定“P0 机械臂仿真安全底座通过已列验收”，不能判定“端到端真人
  遥操作合格”。下一阶段仍需先处理视觉时序/延迟和手指重定向，再执行新的 C-to-Q
  同步录像与数据验收。
