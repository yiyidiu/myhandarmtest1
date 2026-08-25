# physical_lift_slip_and_place_review

- Dialogue ID: `20260815T042208180073Z_physical_lift_slip_and_place_review`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

You are a read-only Gazebo/robot-hand consultant. Do not edit files or run tools.

We are validating a contact-only ABB IRB120 + fixed-preshape three-finger hand pick/place. Attachments, fixed joints, teleportation, hidden guides, collision-disable workarounds, and changing the palm configuration during GRASP are forbidden.

Measured facts from real Gazebo runs:
- The arm reaches a side grasp and closes flexion only. A target-mounted contact sensor proves stable target contacts from distinct finger families f1 and f2 for >0.30 s.
- f3 is unloaded; its CLOSE target was reduced from 1.30 to 0.90 rad because 1.30 produced a real MoveIt start-state self-collision f3link3 <-> handbase_link. After this change the lift start state is collision-free.
- Target is a dynamic 0.10 kg, 50x50x100 mm box resting directly on a table. No guide or pedestal.
- Target and table ODE friction currently mu=mu2=1.0. Hand links have no explicit Gazebo friction tags (defaults).
- Arm Cartesian lift request is [-0.08485, 0, +0.08485] m, planned fraction 1.0, 41 points, controller success, and executed in about 1.05 s despite the group settings velocity_scaling=.05, acceleration_scaling=.03.
- Tool final position error=11.5 mm. Object physically rose 49.4 mm rather than 84.85 mm. Object/tool displacement disagreement=29.2 mm. The run was correctly rejected.
- f1/f2 active joint effort limits are 2.0 Nm. Distal mimic joints use force-limited plugin maxEffort=.20 Nm. GRASP active positions observed around f1j2=.855, f2j1=.735, f3j2=.926. Distal f1j3/f2j2 can be physically blocked and need not equal the driving joint during contact.
- ODE step=1 ms, quick solver 100 iterations, contact kp=1e6, kd=10, max correcting velocity=.1.

Codex plans to explicitly retime approach/lift/place Cartesian trajectories at low velocity and acceleration, and to add finite physically plausible finger/object friction rather than weaken object-following acceptance. The object must then be carried and returned along the inverse path before RELEASE.

Give concise public advice:
1. Rank likely causes of the 29 mm slip.
2. Recommend the smallest one-variable-at-a-time test matrix.
3. Recommend conservative ranges for Cartesian duration/scaling and ODE mu/mu2 without pretending friction alone proves force closure.
4. State which evidence must remain fail-closed before accepting lift and place.
5. Flag any unsafe or misleading part of Codex's plan.
Do not provide hidden chain-of-thought and do not write code.

## DeepSeek → Codex

# REVIEW: 29 mm 滑移与 Codex 计划的只读审查

## SUMMARY

唯一带直接证据的缺陷是**时间参数化/缩放失效**：请求的 lift 位移仅 84.85 mm，却以约 1.05 s 执行完（等效 ~120 mm/s），而配置声称 velocity_scaling=.05、acceleration_scaling=.03。在 5% 缩放下这条路径应当跑 ~15–25 s。缩放设定未生效是事实缺陷，与其余观测完全自洽：快速抬升起始的加加速度冲击 + 位置控制的 P-only 弹簧手指（无力反馈、无重闭能力）→ 静摩擦一旦被打破，滑移不可恢复。**结论：先闭合时间参数化缺陷，再谈摩擦或抓取力；任何摩擦调参都不能替代 force closure 证据。** 运行被正确拒绝（物体/工具分歧 29.2 mm 独立于 11.5 mm 工具误差门），这本身就是体系健康的证明。

## 1. 29 mm 滑移原因排序

1. **时间参数化失效（唯一直接证据）**：1.05 s 执行 ≈ 全速 lift。快速起始产生大加加速度与动态载荷尖峰，击穿指-物静摩擦；且全程仅 ~1 s，接触求解器与手指弹簧都来不及建立/恢复握持。缩放为何没生效（作用在错误 group、参数未传播、未显式 retime）必须先诊断。
2. **握持力瓶颈 + 无力调节**：远端 mimic 为 P-only 弹簧，maxEffort=0.20 Nm 是瓶颈（早于主动关节 2.0 Nm）。加载后手指偏转、接触点下滑，位置环无法补力；P-only 弹簧在负载下稳态力由误差决定，不保证恒定的法向力。
3. **抓取几何余量小**：两个有效承重手指族 + f3 卸载（0.90 rad 仅维持预塑形）；侧向抓取接触法向接近水平，mu=1.0 的摩擦锥 45° 对重力+力矩余量有限。
4. **ODE 接触参数与快速运动的相互作用**：kp=1e6/kd=10 偏硬、max correcting velocity=.1 m/s 限制接触修正速度；在 #1 的快速相对运动下，求解器来不及建立稳定接触力。是放大器，不是主因。
5. **不可恢复性**：位置控制无滑移检测/重闭，一旦起始滑移发生，全过程失败是必然结果——这解释了"49.4 mm 而非中间值"的形态。

## 2. 最小单变量测试矩阵

- **T0 仪表基线（0 变量）**：≥100 Hz 记录物体/工具 6-DOF、各指 effort 与接触力、sim 时间与轨迹时间戳。验证计划时长 vs 实际执行时长。
- **T1 仅重定时**：同一路径显式慢参数化（lift ~8 s，见 §3），其余全部不动。预期直接消除大部分滑移；若仍有 → 继续。
- **T2 静态握持审计（0 变量）**：GRASP 后保持 5 s 不抬升，记录 effort vs maxEffort 裕量、接触力稳定性。**判饱和**：effort 顶格且误差增长 = 容量不足，不是容差问题。
- **T3 准静态探针（单变量：仅升降速率）**：lift 20 mm 极慢加速 → 保持 5 s → 测漂移。区分静态容量不足与动态冲击。
- **T4 摩擦变量（单变量）**：手指/物体显式 mu 设置（见 §3）。**先验证 Gazebo 默认 mu=1.0 是否已作用于手**——若是，此改动可能为 no-op，改做 mu2 或转 T5。
- **T5 接触求解器变量（单变量）**：kp 1e6→1e5，或 max_vel .1→.5（二选一），重跑 T3/T1。
- **T6 容量变量（单变量，仅当 T2 显示饱和）**：mimic maxEffort 0.20→0.30（能力变更，非放宽容差），重跑 T2/T1。
- 全部通过后，才允许携带逆向路径 + RELEASE（place 单独一阶段，见 §5）。

## 3. 保守参数范围（全部 TO_BE_MEASURED，开跑前写死）

- **Cartesian 时长**：85 mm 级 lift 显式重定时 6–12 s（峰值 ~25–50 mm/s）；approach/place 同类 8–15 s。必须显式调用时间参数化并验证执行时长与计划一致——**先修 §1 的缩放失效，否则任何慢设定都可能再次不生效**。
- **mu/mu2**：指-物（硅胶/橡胶 vs 塑料盒）mu≈0.6–0.9、mu2≈0.1–0.3；物-桌面 mu≈0.5–1.0。作为**材料模型一次设定**，禁止 per-run 调到通过；摩擦本身不构成 force closure 证据，必须由接触+漂移门共同背书。
- **ODE 接触**：kp 1e5–1e6、kd 1–10、max_vel .1–.5 为可接受带；改动后必须重过全部负例与漂移门。

## 4. 接受 lift/place 前必须保持 fail-closed 的证据

1. **全时程接触**：≥2 个不同手指族对目标的稳定接触从 GRASP 结束持续到 lift 结束（现有传感器扩展到整条轨迹，不只 0.3 s 窗口）。
2. **全时程漂移门**：物体-工具相对位移连续采样，瞬时 ≤ 5 mm、末态 ≤ 2 mm（TO_BE_MEASURED）；不接受"只查终点"。
3. **饱和签名**：任何 effort 顶格且误差继续增长 → 判容量不足失败，不通过调摩擦掩盖。
4. **零非手指接触**：物体 vs 掌/腕/臂接触事件必须为空。
5. **lift 高度取物体位姿**：≥ 84.85 mm（容差内），不得以工具位姿代替——本次 11.5 mm 工具误差"通过"而物体滑 29 mm 正是反面教材。
6. **时序验证**：执行时长与重定时计划一致（缩放缺陷闭合的硬证据）。
7. **hold ≥2 s + 漂移有界**后，place 才被允许；负例四件套仍为统计硬门禁。

## 5. Codex 计划的危险/误导点

- **"低速度重定时"方向正确，但必须首先诊断缩放为何失效**（作用对象、参数传播、是否真正 retime）；否则新设定可能再次静默无效。
- **摩擦补充**：Gazebo 默认 mu=1.0，手部 link 很可能已是 1.0——"添加有限摩擦"可能是 no-op 或已不现实；必须记录为材料标定，且**摩擦 ≠ force closure**，不能以它替代接触/漂移证据。
- **逆向携带 + RELEASE**：place 阶段自带释放/下落/桌面再接触的独立失败模式；不得把 lift+carry+place 打包成一次验收，必须分阶段（lift→hold 先闭合）。
- **f3 降到 0.90**：作为消除自碰撞的预塑形修正是合法的，但抓取实质变为"两指承重+一指预塑形"；验收判据必须写明承重接触集合（≥2 个手指族），不得暗示三指抓取。
- **工具误差门不能覆盖物体跟随**：11.5 mm 工具误差过门而物体滑移 29 mm 被拒，说明两门必须独立；任何"工具达标即可"的表述都必须禁止。

## PATCH

PATCH: NONE —— 本轮仅审查。

## NOT RUN

未调用任何工具、未读写文件、未执行测试。全部为基于题面实测事实的只读分析；所有数值（时长 6–12 s、mu 0.6–0.9、漂移 5/2 mm 等）为待标定候选，须由 Codex 实测后写死并引用。

## Codex review

PENDING

## Decision and applied diff

PENDING
