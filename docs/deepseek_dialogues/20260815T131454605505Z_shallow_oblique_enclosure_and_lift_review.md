# shallow_oblique_enclosure_and_lift_review

- Dialogue ID: `20260815T131454605505Z_shallow_oblique_enclosure_and_lift_review`
- Session: `session-03e7dfc9-421b-41f0-b10f-6b530091dc11`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

新的实测证据：
- 大物体70x80x140 strict top_down在真实冷启动xy=[0.34,0.18]、[0.30,0]、[0.34,0]均为NO_COMPLETE_IK_CANDIDATE。
- 另对xy=[0.30,0.10]、[0.34,0.10]、[0.30,-0.10]、[0.34,-0.10]、[0.38,0]、[0.38,0.10]做MoveIt虚拟代理粗筛，全部NO_COMPLETE_IK_CANDIDATE；机器人/Gazebo物体未动。
- 小物体strict top_down仅12.82mm桌面间隙，先被20mm规划裕量拒绝；大物体32.82mm间隙通过。

Codex拟下一步冻结大物体，搜索浅倾角top_oblique：tilt仅-5/-10/-15/-18度，轴向偏置绝对值<=25mm，计划lift world-z fraction>=0.95，三指closure spread<=0.20、面积>=0.001m2、归一面积>=0.30、contact height ratio<=0.70、table clearance>=24mm。禁止回到旧-30度/52mm边缘抓取。

请只读对抗审查：
1) 这个搜索边界是否合理，是否还缺少防止“浅倾角伪包络”的几何门？
2) 如果几何通过，虚拟IK粗筛和独立cold-world plan-only的正确顺序是什么？
3) 对浅倾角抓取，抬升是否仍应沿-opposite approach，还是应在接触建立后规划纯world-z lift？考虑手指剪切、腕部路径和用户要求“稳定竖直抬起”，给出明确建议。
4) 在没有contact-only通过前，严禁给full命令。列出进入contact-only的最低plan证据。

不要调用工具、不要改文件、不要给最终PASS。

## DeepSeek → Codex

## 1. 浅倾角搜索边界：方向合理，但还缺五个防伪门

**可接受：** tilt 仅 -5/-10/-15/-18、`|axial|≤25 mm`、计划 lift z-fraction ≥0.95、closure spread ≤0.20、面积/归一面积/高度比/桌面净空这些门都正确，且明确禁回 -30°/52 mm 是对的。

**还缺少：**

1. **roll 必须重新覆盖，不得继承 268° 或固定 165/345°。** 应 coarse 0–345° step 15°，几何/质量通过后 fine ±10° step 2°。这是最容易漏掉的伪包络来源。
2. **平面中心偏移要有硬门。** 旧斜抓正是用 `[+6,-9] mm` 横向偏置配合轴向偏置制造接触。建议新增 `max_centered_planar_offset_norm ≤ 0.008 m`；只允许 `[0,0]`、`[±5,0]`、`[0,±5]`。
3. **f3 不得迟于对侧指族。** 增加 `f3_closure_lag = max(0, f3 - min(f1,f2)) ≤ 0.08`；避免 f3 只在 f1/f2 已把物体挤到末端时才接触。
4. **三指接触面必须仍来自三个不同 object face。** 现有 enclosure 已查 face axis/sign 不同，但浅倾角下要用 quality metrics 显式记录 `face_axis/face_sign`，并断言三者互异；不能只信 `projected area`。
5. **接触高度带要限制，不能只查上边界。** 除 `|height_ratio|≤0.70` 外，增加三指高度比 spread ≤0.45；防止三个接触点集中在同一高度平面形成“环形勒住”而非上下包络。

另外，轴向偏置搜索应显式枚举，例如 `[0.000, 0.010, 0.020, 0.025]`，tilt 与 axial 按 pair 预注册，避免连续优化回到隐蔽边缘解。

## 2. 虚拟 IK 粗筛与 cold-world plan-only 的顺序

正确顺序：

1. **离线纯几何 sweep**：只跑 `grasp_size_pose_diagnostic` 级别的 URDF FK/质量门；只保留全部 quality gates PASS 的 `(xy,tilt,axial,roll,planar)`。
2. **虚拟 MoveIt IK 粗筛**：对第 1 步 shortlist，在当前 world 中临时删除/添加精确尺寸 target proxy 到候选 xy，逐候选 `compute_ik` + `check_state_validity` + 9 个 approach 样本。该阶段只产生“值得冷启动”的排序，不产生可执行结论。
3. **每个候选独立 cold-world plan-only**：使用该 xy 对应 world，真实 Gazebo 物体和 PlanningScene 必须同一位姿；记录 robot_executed=false。
4. 只有独立 cold-world plan-only PASS，才允许该候选进入 contact-only。

虚拟粗筛防假通过：

- 每轮 probe 前恢复 exact target proxy；
- probe 结束删除虚拟 proxy；
- JSON 标记 `virtual_target_probe=true`、`gazebo_target_untouched=true`；
- 严禁把虚拟 IK 结果标记为 plan-only PASS；
- 候选最终以 cold-world run_id 为准。

## 3. 浅倾角抬升方向：接触建立后应改为纯 world-Z lift

建议：**不继续沿 -opposite approach；接触建立后规划纯 world-Z lift，放回时严格逆路径。**

理由：

- 用户目标已明确为“稳定竖直抬起”，而 -opposite approach 只是逼近阶段的几何方向；
- 即使浅倾角下 -approach 的 z-fraction ≥0.95，残余横向分量仍会剪切三指接触，且旧 21 mm 失败就同时包含横向拖动与滚转；
- 纯 +Z 路径使手指接触法向主要承担竖直摩擦/夹持，而不是横向推挤。

实施边界：

- lift 请求仍建议 `+0.085 m` world-Z，最低实际 `+0.060 m`；
- 计划路径若 Cartesian/IK 不完整或碰撞检查失败，直接 FAIL，**不得回退**到 -approach；
- place 必须沿 lift 的严格逆 world-Z 路径；
- 实际 object delta 的 z 分量 ≥0.90、world-z 抬升 ≥60 mm、5 s 稳定悬空。

## 4. 进入 contact-only 的最低 plan 证据

只有**独立 cold-world plan-only** 同时满足以下全部条件，才允许该候选进入 contact-only：

1. `success=true`、`robot_executed=false`、`hand_closed=false`；
2. `grasp_family=top_oblique`，tilt ∈ {-5,-10,-15,-18}；
3. `|axial|≤0.025 m`、`planar_offset_norm≤0.008 m`；
4. `selected_quality.passed=true`，含第 1 节五条新增门；
5. `table_clearance_m ≥ 0.024`；
6. `moveit.ik_success=true`；
7. `moveit.approach_ik_samples=9`，且无 `APPROACH_INCOMPLETE`、`APPROACH_IK_BRANCH_JUMP`；
8. `pregrasp_plan_success=true`，pregrasp trajectory 碰撞/时间有效；
9. 物体 pose stability 与 scene sync 误差均在原阈值内；
10. 独立 run_id，且 plan-only JSON 中 `attachment_used=false`。

**未满足前，严禁 contact-only；contact-only 未 3/3 冷启动通过前，严禁 full。** 任何虚拟 IK 粗筛通过都不得替代第 1–10 条。

## Codex review

PENDING

## Decision and applied diff

PENDING
