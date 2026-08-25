# hand_nan_explosion_isolation_followup

- Dialogue ID: `20260814T150525344232Z_hand_nan_explosion_isolation_followup`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

这是上一轮 hand_startup_and_three_distal_stability 的证据更新。请明确修正上一轮结论，并给出针对实际文件的第二版建议；仍然只读，不调用工具。

先纠正我上一轮提示中的一处错误：实际 mimic plugin 源码是：
```
angle = leader_position * multiplier + offset;
a = mimic_position;
error = angle - a;
velocity_error = leader_velocity * multiplier - mimic_velocity;
raw_effort = pid.computeCommand(error, velocity_error, period);
mimic_joint->SetForce(0, forceSign * clamp(raw_effort));
```
不是上一轮提示中的 `a-angle`。

新增实测证据（Gazebo 启动后、任何 GRASP/RELEASE 命令前）：
- model 初始以 spawn_model `-J` 传入 hand active/passive 目标，且 startup_coordinator 再调用 `/gazebo/set_model_configuration` 返回 success。
- 仿真从 paused 切到 unpaused，controller 启动后第一个记录时刻 sim t=0.024 s：
  - f1j1 pos=3.5868479, vel=-3.626e21
  - f1j2 pos=-2.6041965, vel=-5.110e21
  - f2j1 pos=4.6914469, vel=-5.601e21
  - f3j2 pos=2.6688806, vel=-1.871e21
- 轨迹首点正是上述已经爆炸的位置，末点 [0.051,0.20,0.20,0.20]；因此轨迹没有制造初始爆炸。
- sim t=2.101 s 时四个 active velocity 都变成 NaN，位置继续越界。
- /joint_states 只含四个有 transmission 的 active 手关节，mimic passive 关节未发布，所以之前缺少 passive 实时证据。
- 九个手部刚体 mass>0，惯量矩阵全部正定、condition number <6、满足惯量三角不等式。
- STL 尺寸合理（0.05~0.18 m），但碰撞网格很重：handbase 466230 triangles；f1/f3 proximal 各160098；中/远端各约20k/14k。
- 当前 world 初始 paused，max_step_size=0.001 s；startup 在 controller manager 可用、model 存在后 set configuration，然后 unpause、start controllers。日志显示 set_model_configuration 与 Gazebo model/plugin 完成 Load 的时间非常接近。

实际文件：
- `src/abb120_moveit_config1/config/gazebo_handarm.urdf`
- `src/roboticsgroup_upatras_gazebo_plugins/src/mimic_joint_plugin.cpp`
- `src/handarm_sim_demo/scripts/startup_coordinator.py`
- `src/handarm_sim_demo/config/gazebo_hand_position_pid.yaml`
- `src/handarm_sim_demo/launch/simulation_baseline.launch`

请回答：
1. 明确撤回或修正上一轮“轨迹时序是最可疑根因”的判断。
2. 评审以下单因素隔离顺序：A禁用四个 mimic plugin但保留 passive dynamics；B启用 f3j1 mimic only；C逐一启用三个 distal mimic；D在全部无爆炸后才运行轨迹与开合。
3. `velocity_error` D 项、SetVelocityLimit、SetForce 与 `forceSign` 哪些最可能导致数值爆炸？有限 maxEffort 是否足以排除插件导致爆炸？
4. 设计主动/被动关节有限值、限位、速度 fail-closed 门：启动后若任一非有限或越限应在发轨迹前中止并留下精确证据。
5. 设计获得 passive joint state 的可靠方式：是否应把 passive joints 增加到 joint_state 发布（不加 position controller transmission），或直接在独立 Gazebo verifier 使用 get_joint_properties？说明取舍。
6. 给出下一步最小代码补丁建议与自动隔离测试矩阵。不要建议放宽容差，不要把简化 collision mesh 与当前爆炸根因混为已证实。

输出：UPDATED SUMMARY、RETRACTION/CORRECTION、ISOLATION MATRIX、PATCH、TESTS、RISKS、NOT RUN。

## DeepSeek → Codex

# UPDATED SUMMARY

新证据推翻了上一轮"轨迹首点/目标时序是最可疑根因"的判断：爆炸发生在 unpause 后的前 24 ms 内、任何轨迹执行之前（t=0.024 s 四个 active 关节位置已为 3.59/−2.60/4.69/2.67 rad、速度 ~1e21 rad/s，t=2.1 s 速度 NaN）。轨迹首点只是**采样了已经爆炸的状态**，上一轮的 goal error 1.621 是同一爆炸的下游症状；且 f1j1 两次观测值 1.672 与 3.587 不一致，说明爆炸量级跨 run 不重复——这是求解器级失稳的典型特征，不是确定性时序 bug。当前首要目标是单因素隔离找到爆炸源，全部关节有限/在界内之前禁止运行任何轨迹与开合。远端三关节调稳（上一轮主题）必须**冻结**，待爆炸根因闭合后再恢复。

# RETRACTION / CORRECTION

1. **撤回**："reset 冲掉 teleport、首点采到 1.67、轨迹要求 1.62 rad/s 超限"作为最可疑根因的排序。保留其作为启动鲁棒性缺陷（teleport 后读回验证、settle 后采样、不信任 success 返回、禁止放宽容差仍全部有效），但它**不是本次爆炸的原因**。

2. **修正插件语义**：真实源码 `error = angle − a`（目标−当前）。正误差 ⇒ follower 低于目标 ⇒ 在 forceSign 与关节力矩方向映射正确时，力把 follower 推向目标。这翻转了上一轮基于 `a − angle` 的符号分析；正确性现在完全取决于 `forceSign × 关节安装方向（镜像 origin）` 的实际映射，必须用符号测试实测，不得假设。

3. **决定性逻辑（新）**：mimic 插件只对四个**被动**关节（f3j1/f1j3/f2j2/f3j3）调用 SetForce；四个爆炸的 active 关节是 leader，插件**从不向它们写力**。因此"插件正反馈"不可能直接解释 active 关节的 1e21。量级核算：maxEffort≤4 N·m，惯量正定且 cond<6（取保守 1e-5 kg·m²）→ 加速度 ≤4e5 rad/s² → 24 ms 内速度 ≤ ~1e4 rad/s，比观测值小约 17 个数量级；2.1 s 持续顶格也仅 ~8e5。**结论：在 clamp 有效的前提下，四个候选（velocity_error D 项、SetVelocityLimit、SetForce、forceSign）都无法解释观测量级**；1e21+NaN 指向全局物理失败。主要假设（按可证伪顺序）：(a) 启动竞态——set_model_configuration 与模型 Load 完成时间极近，teleport 可能作用于半加载模型后被 SDF 初始态覆盖，产生深穿透/状态损坏；(b) 重型碰撞网格（handbase 466k、f1/f3 proximal 各 160k 三角面）+ 初始穿透 → unpause 首个接触求解步产生巨大冲量；(c) URDF→SDF 转换后 SDF 仍含原生 `<mimic>` 运动学约束，与插件 SetForce 形成约束冲突；(d) paused/unpaused 或 real_time_update_rate/max_step_size 失配。**有限 maxEffort 不能开脱插件**——它仍可能以"有界力持续激励/约束冲突"方式参与，但不足以单独造成 active 关节 1e21；证据上它最多是诱因或放大器，不是能量来源。

# ISOLATION MATRIX

A–D 顺序本身正确，但需前置两个修正并扩充：

- **前置 0（必须先落地）**：显式"模型 Load 完成"门（订阅模型就绪/插件初始化完成信号）→ 之后才 teleport → 读回验证 → 再 unpause。不修竞态，矩阵结果跨 run 不可复现。
- **矩阵行（每行 N≥3 次全新 spawn，固定窗口时长，逐行 JSON 落盘）**：
  - **A0 纯物理基线**：无控制器、无插件，仅 `-J` 初始态 + teleport，unpause 观察。隔离"物理+接触+spawn 竞态"vs"控制器/插件"。
  - **A0'**：同 A0 但**取消 set_model_configuration**，只依赖 `-J` spawn（直接证伪竞态假设，零成本）。
  - **A**：保留 passive dynamics，禁用全部四个 mimic 插件。
  - **B**：仅启用 f3j1（镜像、forceSign、P=3/D=0.08/maxEffort=4.0）。
  - **C**：逐一启用三个 distal mimic。
  - **D**：全插件启用、全关节有限后才允许轨迹 + 开合。
- **通过判据（每行）**：窗口内 9 个关节位置/速度全有限、位置在 [min−δ, max+δ] 内、速度低于 sanity cap（候选 10 rad/s，TO_BE_CALIBRATED）；前 100 ms 以 ≥500 Hz 快照留档。
- **诊断专用行（单独标注、不当作修复）**：collision 换成简单图元重跑 A0——只用于证伪接触冲量假设；严禁把"简化碰撞网格"与爆炸根因混为已证实结论。
- 附加检查项：转换后的 SDF 中是否残留原生 `<mimic>` 元素（若是，A/B/C 全都要重新解释）。

# PATCH（最小，全部由 Codex 映射实际文件）

```diff
--- a/src/handarm_sim_demo/scripts/startup_coordinator.py
+++ b/src/handarm_sim_demo/scripts/startup_coordinator.py
@@
+INIT_READBACK_EPS = 0.01      # TO_BE_CALIBRATED, fixed before runs
+SANITY_VEL_CAP   = 10.0       # rad/s, sanity gate above which nothing is physical
+
+def wait_model_fully_loaded(model, timeout):      # NEW: load-complete gate
+    ...   # poll model info / plugin-ready signal; fail-closed on timeout
+
+def physics_sanity_gate(joint_names, window_s, hz):  # NEW: after unpause
+    for q in sample_all_joints(hz, window_s):
+        assert finite(q.pos) and finite(q.vel)        # abort with full snapshot
+        assert within_bounds(q.pos)                    # [min-d, max+d]
+        assert abs(q.vel) < SANITY_VEL_CAP
+    # any violation -> dump snapshot+timestamps+effort log -> abort BEFORE trajectory
+
 def startup(...):
+    wait_model_fully_loaded(HAND_MODEL, timeout=30.0)
     teleport_and_verify(HAND_MODEL, HAND_OPEN_TARGET)   # keep from prev round
     unpause()
     start_controllers()
+    physics_sanity_gate(ALL_9_HAND_JOINTS, window_s=0.5, hz=500)
```

```diff
--- a/src/abb120_moveit_config1/config/gazebo_handarm.urdf
+++ b/src/abb120_moveit_config1/config/gazebo_handarm.urdf
@@
+<!-- Diagnostics only: publish all 9 hand joints, read-only, no transmission,
+     no position controller; do NOT add transmissions for passive joints. -->
+<gazebo>
+  <plugin name="hand_joint_state_gt" filename="libgazebo_ros_joint_state_publisher.so">
+    <joint_name>f1j1</joint_name> ... <!-- all 9, including f3j1,f1j3,f2j2,f3j3 -->
+    <update_rate>100</update_rate>
+  </plugin>
+</gazebo>
```

```diff
--- a/src/roboticsgroup_upatras_gazebo_plugins/src/mimic_joint_plugin.cpp
+++ b/src/roboticsgroup_upatras_gazebo_plugins/src/mimic_joint_plugin.cpp
@@
+// Diagnostic publisher (param-gated, default OFF in production): 10-50 Hz,
+// per joint: leader_pos, mimic_pos, velocity_error, raw_effort, clamped_effort.
+// Purpose: verify the clamp actually applies (|effort| <= maxEffort) and
+//          enable the forceSign sign test. Read-only, no physics coupling.
```

新增脚本：`isolation_matrix_runner.py`（A0/A0'/A/B/C/D 逐行执行、JSON 通过/失败、每行全新 spawn）、`force_sign_test.py`（仅 f3j1、附近无接触，小步进指令，记录 error 符号与运动方向，断言 follower 向目标运动）。

# TESTS

- **T-race**：load-complete 门 + teleport + 读回验证，5 次全新 spawn，零轨迹前违规。
- **T-matrix**：A0/A0'/A/B/C/D 全行执行，定位第一个爆炸出现的行；每行 N≥3 且结果一致才采信。
- **T-clamp**：日志内所有 |clamped_effort| ≤ maxEffort；若出现超限即 clamp 破损（插件级 bug，优先修）。
- **T-sign**：每个镜像 mimic 的 forceSign 符号测试，实测运动方向与误差符号一致。
- **T-passive-coverage**：/joint_states 含全部 9 关节、有限、≥100 Hz。
- **T-gate-abort**：注入已知违规，验证 sanity gate 在轨迹下发前中止并留下完整证据。
- **T-regression**：根因闭合后重跑 D，再恢复远端三关节 RELEASE/GRASP/RELEASE 稳定性测试（沿用上一轮阈值，不放宽）。

# RISKS

- 碰撞网格简化**只能作为诊断行**；在爆炸根因未闭合前把它当修复 = 两个不可归因改动混包。
- 被动关节当前无实时证据，A/B 行的解释受限；先加 joint state publisher，但同一话题新增 5 关节可能影响 RSP/TF 消费者——需回归检查。
- NaN 会污染世界状态：矩阵每行必须全新 spawn，禁止行间复用同一 Gazebo 实例。
- 竞态门未落地前，矩阵结果不可复现、不可采信。
- "maxEffort 有限"目前是声明不是证据——没有 T-clamp 的 instrumentation，任何基于"插件力有界"的排除都无效。
- 原生 `<mimic>` 与 SetForce 冲突未排查：若 SDF 残留 mimic 元素，A–C 全部结论需重做。

# NOT RUN

未调用任何工具、未读写文件、未执行测试。所有量级核算、阈值与假设均为基于题面公开事实的分析，须由 Codex 通过 T-race/T-matrix/T-clamp/T-sign 实测证伪或证实后引用；上一轮远端 PID 调参结论整体冻结，待爆炸根因闭合后恢复。

## Codex review

PENDING

## Decision and applied diff

PENDING
