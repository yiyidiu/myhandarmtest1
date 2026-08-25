# stage_a_multirun_home_bounds_review

- Dialogue ID: `20260814T114807428119Z_stage_a_multirun_home_bounds_review`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

Stage A continuous multi-angle avoidance already passed single-run calibration. A first independent 10-run attempt passed repetitions 1-3, then repetition 4 failed before motion because the home target code blindly added current-round multiples of 2*pi to wraparound joints, producing a target outside MoveIt finite bounds. Codex changed the logic to enumerate target+2*pi*k values that lie inside each joint actual MoveIt bounds and select the candidate nearest to the current joint value; if none exists it fails closed. Non-wrap targets are also explicitly checked against bounds. Unit tests cover finite bounds forcing the base angle, a wide joint-6 range selecting a nearby +2*pi equivalent, and no valid candidate raising. A fresh 10-run Gazebo test is now running from zero. Read-only review questions: (1) Is this correction logically sound for revolute joints represented with finite bounds? (2) What runtime invariants must be logged to prove no stale/planner-inconsistent start state across repetitions? (3) What failure must invalidate the entire 10-run result rather than permit stitching? Return concise public analysis only; do not use tools.

## DeepSeek → Codex

## 1. 修正逻辑是否成立（有限界 revolute + wraparound）

方向正确，但成立需满足五个条件，建议 Codex 逐条核验：

- **等价性前提**：只有真正绕 2π 等价的关节才允许 +2πk 枚举。必须显式维护"可环绕关节"白名单；非环绕关节走独立的 bounds 检查（题面已做），且要防止白名单误判（例如 URDF 为 bounded revolute 但 MoveIt 实际按 continuous 处理，或反之）。
- **候选距离须用圆周度量**：`|θ_current − θ_cand|` 应取 `min(|d|, 2π−|d|)` 而非原始差；否则当前值靠近缝（如 −3.1 rad）而候选在 +3.1 rad 附近时，"最近"选择可能失真。当前值本身也应先折入 bounds 再参与比较。
- **k 的枚举必须由实际 bounds 推导且有限**：`k ∈ [ceil((lo−t)/2π), floor((hi−t)/2π)]`；对 bounds 宽度异常（>2π 或多周期）的关节要有显式处理或拒绝，不能无限/盲目枚举。
- **与规划表示一致**：所选候选要成为规划起点的种子与 IK/轨迹生成唯一使用的表示；还需验证 MoveIt 在该 bounded 表示下不会在缝上走"最短圆周插值"——即轨迹逐点都必须在 bounds 内（可并入每点校验）。规划起点的 bounds 来源必须与枚举所用的 bounds 是同一份。
- **无候选 → fail-closed 正确但需归因**：若某目标在物理上可达、仅因参数化表示超出有限 bounds 而 fail，这是表示层限制而非不可达；须记录目标、bounds、k 区间，避免与真实不可达混淆。若该情况发生，说明该关节不应设为有限 bounds（连续关节应放宽），而非继续枚举。

结论：作为"在有限 bounds 表示内选最近等价候选"的修补，逻辑成立；作为"消除 wrap 问题根因"的方案，仅当上述一致性（度量、白名单、bounds 来源、轨迹表示）全部落实后才算完备。

## 2. 必须记录、以证明跨 repetition 无陈旧/不一致起点的运行时不变量

- **状态新鲜度**：每段规划前快照 `/joint_states` 的 header stamp 与接收时间；要求 stamp 有效且年龄 < 阈值（建议 ≤50 ms，值写死并记录）。
- **实际态 vs 规划起点**：记录实际关节向量与 MoveIt planner start state 全向量及逐关节最大偏差；不变量：< 0.01 rad（阈值开跑前写死），否则该段拒绝规划。
- **home 目标逐关节审计**：每个可环绕关节记录原始目标、实际 bounds、k 候选列表、所选 k、实际当前值、圆周距离；不变量：所选候选在 bounds 内、距离为候选中最小。
- **bounds 来源与一致性**：记录 bounds 向量来源（URDF/SRDF/joint_limits.yaml 合并结果）及哈希；不变量：10 次 run 内哈希不变，且枚举与规划使用同一哈希版本。
- **plan 首点一致性**：plan[0] vs planner start vs 实际态最大偏差 < ε；不满足即 fail-closed 而非执行。
- **段间连续性**：上一段末实际态 vs 本段起点实际态逐关节偏差 < ε（禁止段界 teleport）；连续 course 下此值必须记录。
- **fail-closed 全量留痕**：枚举为空时记录目标、bounds、k_min/k_max、白名单判定，形成可复现归因。
- 逐 run 单调计数（run id / segment id / 时间戳）与"每次规划种子 = 实际态（永不使用陈旧 home 快照）"的断言日志。

## 3. 何种失败必须使整个 10-run 批次作废（不得拼接）

必须整体作废并从头重跑的情形：

- **逻辑类失败复发**：任何与本次 wrap 修复同类的失败（越界目标、错误候选选择、枚举为空但目标本应可表示）——这是代码缺陷而非统计噪声，再跑多少次都是无效样本。
- **一致性证据缺失**：任一 run 的 §2 关键日志缺失/损坏，无法证明起点一致性；或规划所用 bounds 哈希与枚举所用不一致。
- **环境漂移**：批次中途障碍模型、物理参数、世界文件、仿真时钟/步长被改动或重置（含 physics 参数漂移）。
- **未归因失败**：fail-closed、controller abort、watchdog 超时、碰撞/接触事件发生后无根因分类的 run。
- **修复前样本混入**：修复前的 repetitions 与修复后的样本拼成一个统计——任何 pre-fix 数据不得并入（当前从零重跑已满足此点，须在报告中显式声明 pre-fix 3 次成功仅作校准记录、不进统计）。

允许的例外（仍需文档化）：单一、独立、已明确归因的**非逻辑**失败（如带清晰签名的瞬态控制器故障），可在归类后替换该单次 repetition，但批次须重新同质化；默认规则应为"逻辑类或未归因 → 整批作废"。当前正在跑的 fresh 10-run 只有在 §2 全部不变量逐 run 落盘且无上述任何事件时，其 10 次结果方可整体作为 Stage A 统计结论。

## Codex review

部分接受。DeepSeek正确识别了有限 bounds 来源一致、轨迹首点与实际状态一致、
逐轨迹点 bounds/碰撞复核以及失败整批作废的重要性。这些建议与当前实现的
`set_start_state_to_current_state()`、轨迹首点容差和逐点 PlanningScene 校验一致，
后续报告会明确修复前 3 次结果不进入正式 10 次统计。

拒绝“候选距离使用圆周度量”的建议：候选本身是同一物理角的 `2*pi` 等价数值表示，
圆周距离会使所有候选等距，无法选择在 MoveIt 有限坐标表示中让关节数值位移最小的候选。
这里使用 `abs(candidate-current)` 是有意设计。50 ms joint-state 新鲜度也不直接采纳为
本仿真的任意阈值；当前以实际状态快照、plan 首点误差 `<0.01 rad`、执行后连续低速样本
和末端误差共同 fail-closed，正式结论以新 10 次实际运行数据为准。

## Decision and applied diff

`PARTIAL_ACCEPT_NO_DEEPSEEK_PATCH`。DeepSeek未生成或应用补丁、未调用工具；Codex继续以
自己实现并已通过单元测试的有限 bounds 等价角算法进行全新 10 次验证。
