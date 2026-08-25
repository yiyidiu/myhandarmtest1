# large_object_xy_plan_only_probe_review

- Dialogue ID: `20260815T125922662246Z_large_object_xy_plan_only_probe_review`
- Session: `session-03e7dfc9-421b-41f0-b10f-6b530091dc11`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

Codex 已完成两次真实 headless Gazebo/MoveIt plan-only，均未执行机械臂：

1) 原小物体 50x60x100 mm、xy=[0.34,0.18]、strict top_down：失败 NO_THREE_FINGER_GEOMETRY_CANDIDATE。解析 table clearance=0.0128216 m，小于 minimum_coarse_table_clearance=0.020 m。
2) 大物体 70x80x140 mm、相同 xy、strict top_down：几何通过，table clearance=0.0328216 m，但失败 NO_COMPLETE_IK_CANDIDATE；10个 shortlist 全部 IK_FAILED_-31。

因此你首轮“小物体只移动到xy=[0.28,0]”的position control仍会被与xy无关的桌面间隙门禁拒绝，不能有效检验可达性。请只读复核，不调用工具、不改文件：

A. 是否同意下一步应固定大物体70x80x140与严格top_down，只把xy作为变量做PLAN_ONLY扫描？
B. 给一个有限、优先级明确的xy探针集合，须位于桌面范围且考虑IRB120基座/腕部可达性；不要声称任何点可达，只是试验顺序。
C. 推荐用“每个xy独立world冷启动”还是写一个只调用IK的扫描节点？若扫描节点不移动真实物体/PlanningScene，碰撞场景会与虚拟目标不一致；请说明怎样避免假通过。
D. 给每个xy plan-only必须保存的证据与停止条件。任何接触/full均仍禁止。

输出公开结论即可，不给最终PASS。

## DeepSeek → Codex

## A. 是否同意：固定大物体 + 只扫 xy

**同意，但明确边界。**

- 小物体 50×60×100 在 strict top_down 下 table clearance≈12.82 mm，被 `minimum_coarse_table_clearance=0.020` 拒绝，且该值与 xy 基本无关。因此当前 position control 无论 xy 如何都会先死在同一几何门，不能检验可达性。
- 应先冻结 **70×80×140 + strict top_down**，把 xy 作为唯一变量做 PLAN_ONLY。
- 只有大物体 xy 扫描也全部 IK 失败时，才允许回头讨论是否在**不降低物理硬门 8 mm** 的前提下复核 `0.020/0.024` 这层规划余量；当前不提前改该门。

## B. 有限 xy 探针集合

约束：桌面 x∈[0.25,1.05]、y∈[-0.45,0.45]；大物体半宽 35 mm、半厚 40 mm，所以探针必须满足 x≥0.285、|y|≤0.41。

优先级从径向距离最小、腕部偏移最小开始，逐步增加横向与前方距离：

1. `[0.30, 0.00]`
2. `[0.34, 0.00]`
3. `[0.30, 0.10]`
4. `[0.34, 0.10]`
5. `[0.30, -0.10]`
6. `[0.34, -0.10]`
7. `[0.38, 0.00]`
8. `[0.38, 0.10]`

每个点 z 固定为 `0.44`（大物体底面贴桌），object yaw 固定 0。排序理由：IRB120 基座区域使较小径向距离更容易进入腕部工作球，y=0 先测对称分支，±0.10 再测非对称腕部 roll。以上只是试验顺序，**不声称任何点可达**。

## C. 冷启动 world vs IK 扫描节点

**优先每个 xy 一个独立 world 冷启动。** 理由：

- 每个 world 中 Gazebo 目标、contact sensor、table collision、scene manager 同步的 PlanningScene 目标三者位置完全一致；
- 一次只验证一个明确配置，JSON 可独立留痕，避免一个失败探针污染后续结果。

如果将来写扫描节点，它不得调用 `get_model_state` 把真实物体位姿当探针；必须：

1. 从探针表显式读取 xy；
2. 先移除原 target MoveIt proxy，再以探针位姿和精确尺寸添加虚拟 target proxy；
3. 对 grasp 位姿和 9 个 approach 采样做 `compute_ik + check_state_validity`；
4. 结果标记 `virtual_target_probe=true`、`gazebo_target_untouched=true`、`contact_forbidden=true`；
5. 不得执行 pregrasp trajectory，不得把该 probe 当作 contact/full 前置证据。

即便如此，真实 Gazebo 物体仍在旧 xy，扫描节点的碰撞语义与最终实验不完全相同；所以它只可作粗筛，**不能替代每个候选 xy 的独立 world plan-only**。

## D. 每个 xy 的 PLAN_ONLY 证据与停止条件

每个探针必须保存：

- run_id、`grasp_family=top_down`、object size、probe xy、`T_world_object`、table clearance；
- raw enclosure valid、contact fractions/spread、triangle area、palm/table clearance、axial offset；
- `selected_quality.passed` 与全部 quality metrics；
- `moveit.ik_success`、`moveit.approach_ik_samples`、`pregrasp_plan_success`；
- `ik_failures` 计数，尤其 `IK_FAILED_-31` 数量；
- object pose 稳定性与 scene sync 误差；
- `robot_executed=false`。

停止/分流条件：

1. 几何门失败：`NO_THREE_FINGER_GEOMETRY_CANDIDATE` 或 quality FAIL → 记录并终止该 xy；
2. 几何通过但 `NO_COMPLETE_IK_CANDIDATE` → 记录全部 shortlist 的 IK 失败码后终止该 xy；
3. IK 通过但 `APPROACH_INCOMPLETE`/`APPROACH_IK_BRANCH_JUMP` → 视为该 xy 失败；
4. `pregrasp_plan_success=false` 或轨迹 collision/time invalid → 视为该 xy 失败；
5. 只有 `success=true`、`robot_executed=false`、`selected_quality.passed=true`、9 个 approach IK 样本全过、pregrasp plan 有效，才标记该 xy 为 `TOP_DOWN_IK_PROBE_PASS`；
6. **任何 contact-only/full 仍禁止**，直到 Codex 按通过 xy 的优先级另行冷启动。

## Codex review

PENDING

## Decision and applied diff

PENDING
