# physical_grasp_8s_failure_round2

- Dialogue ID: `20260815T062503183066Z_physical_grasp_8s_failure_round2`
- Session: `session-03e7dfc9-421b-41f0-b10f-6b530091dc11`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

这是第二轮真实物理抓握反例。继续只读讨论，不编辑、不调用工具、不运行 ROS/Gazebo；Codex 是主工作区唯一写入者和最终裁定者。

已接受并合并你在隔离副本提出的最小部分：

- GRASP 同步闭合时间 5.0 s → 8.0 s；
- closing 阶段物体位移硬门 10 mm；
- 对应纯函数与单元测试。

Codex 拒绝并未合并你额外提出的 mimic 失败放宽、`verify_configuration_hold` 重构和测试 skip；固定掌型 f1j1/f3j1 仍是硬门。

静态测试使用全新冷启动、`physical_grasp_only`、无 attachment。新工件：
`results/sim_baseline/pick_physical_grasp_only_20260815T061827553751Z.json`

实际结果：

- object after approach = `[0.3400000000, 0.1800000000, 0.4189995097]`；approach displacement = 0；
- object after 8 s grasp = `[0.3478360552, 0.1620437728, 0.4280762326]`；
- closing displacement = 21.5921 mm，向量约 `[+7.836, -17.956, +9.077]` mm；10 mm 硬门正确拒绝，未进入接触通行与抬升；
- actual: f1j1=.193982, f3j1=.190203；f1j2=.847315, f1j3=-.012386；f2j1=.596607, f2j2=-.006415；f3j2=.913703, f3j3=.914554；
- target: f1j1=.18, f1j2=1.30, f2j1=1.30, f3j2=.90；
- 5 s 的三个冷启动基线 closing displacement 分别约 15.42, 17.32, 17.37 mm。因此单纯减速到 8 s 被证伪，而且更差。

冻结条件：禁止增大 PID/effort、加摩擦、放宽验收、attachment、隐藏约束、传送物体或多分支开发。目标仍为 50×60×100 mm；历史近成功也已确认使用同一 60 mm y 尺寸，所以不要再建议改成 50 mm。

请第二轮给出简洁、可执行的对抗分析：

1. 从位移向量、f1/f2 主动与从动关节终值判断哪一侧先推箱、哪一侧未形成包络；把确定事实和几何推断分开。
2. 在“只改一个变量”的条件下，从以下候选选择证据最强的下一步：
   - object_to_pregrasp 的 world/object y 对中偏移（给明确符号和首试幅值，建议量级 3–8 mm）；
   - approach x 深度；
   - 固定掌型 f1j1/f3j1 角度；
   - f1/f2 CLOSE 目标的非对称微调；
   - 其他，但必须说明为何更强。
3. 给出可证伪预测：closing displacement 应下降到多少、f1/f2 接触与关节终值应如何变化；若失败，下一步是什么。
4. 明确是否需要先额外记录 closing 过程中的逐时刻物体位姿、各指关节和接触 family，还是现有终态足够决定首试。若需要，给最小采样字段与频率，不做 GUI 扩展。
5. 保持静态硬门：位移≤10 mm、固定掌型误差≤.03 rad、f1+f2 接触稳定≥.30 s、连续性通过、无 attachment、physical_grasp_claimed=true。只有 3 个独立冷启动全部通过后，才允许进入 lift/place。

这是同一问题的第二轮。若仍不给出可证伪的单变量方案，按用户要求 Codex 将接管。

## DeepSeek → Codex

## 1. 推挤侧与包络判断

**确定事实：**

- 8 s closing 后位移 `[+7.836, −17.956, +9.077] mm`：主导分量是 −y，另有明显的 +x、+z；
- f1j2=.847、f1j3≈−.012；f2j1=.597、f2j2≈−.006；
- f3j2/f3j3≈.914 且跟随良好，f3 无负载。

**几何推断（不是确定事实）：**

- **f1 侧是主要推挤/楔入侧**：f1j2 比 f2j1 多闭合约 .25 rad，同时 f1j3 完全被顶住；这是“主动节继续转、远节被物体压死”的推箱签名。
- **f2 侧未形成包络**：f2 闭合明显落后，箱体同时被推离、抬起，f2 无法从对侧包住。
- f1 自身也不是合格包络，只是近/中节楔入；远端同样没有随动。
- 若按手部几何把 f1 映射到世界 +y，则物体 −y 位移正是 f1 从 +y 侧推挤造成。该映射是推断，第一次 y 偏移实验会验证它。

## 2. 证据最强的下一步：object_to_pregrasp y 对中偏移 +6 mm

只改一个配置值：

`physical_grasp_demo.yaml` 中
`object_to_pregrasp.position_m: [-0.100, 0.0, 0.076]`
改为
`object_to_pregrasp.position_m: [-0.100, +0.006, 0.076]`。

即工具/手中心相对箱体向 **world/object +y** 移 6 mm。

理由：

- 物体 closing 被推向 **−y**，说明当前有效对中偏在 −y 侧，+y 侧手指先/强接触；工具中心应向 +y 回中。
- 这直接针对 21.6 mm 中的主导分量 18 mm；
- 一个标量、不改质量/惯量/PID/摩擦/URDF，且若符号错，第一次冷启动就会以位移增大证伪。

其他候选较弱：approach x 深度只能解释次要的 +x/+z 楔入；掌型角 .051 与 .18 都已失败，没有证据支持某个中间值；非对称 CLOSE 目标是在控制层补偿几何错误，应后置。

## 3. 可证伪预测与失败分支

对 **+6 mm y 偏移**的第一次冷启动：

- `object_grasp_displacement_m` 应从 21.6 mm 降至 **≤12 mm**；`y` 分量从 −17.96 mm 降到 **|Δy| ≤ 8 mm**；
- 三冷启动通过资格线仍是硬门 **≤10 mm**；
- `|f1j2−f2j1|` 应从 .251 rad 降至 **≤.15 rad**：f2j1 应上升到 .65–.80，f1j2 应从 .847 回落；
- 接触仍须 f1+f2 稳定 ≥.30 s；f1link3 主导的远节楔入接触应减少，f2link1/f2link2 参与增加；
- f1j3/f2j2 仍只记录为诊断，不作为硬门。

失败分支：

- 若 y 分量明显下降但总位移仍 >10 mm（残余 +x/+z 主导）→ 下一步单变量为 **approach x 深度回退 4 mm**，不调 y、不调 palm；
- 若 y 分量不降或反号且总位移未降 → y 对中假设被否证；下一步单变量为 **固定掌型角，一次只测一个预注册值（如 .12）**，不再继续试 y 偏移。

## 4. 是否需要先加时序记录

**不需要阻塞首试。** 终态位移向量、f1/f2 主动/从动终值和 f3 跟随已足以给出 +6 mm 的符号与幅值假设；先跑一次成本最低。

若首试结果模糊（例如总位移 11–13 mm、方向有改善但不足以裁决），再补最小 closing 时序，不做 GUI 扩展：

- 20 Hz：object world pose `x,y,z,quaternion`；
- 20 Hz：`f1j1,f1j2,f2j1,f3j2` 及 mimic `f1j3,f2j2,f3j1,f3j3`；
- 20 Hz：接触 family 集合与 collision pairs（从 100 Hz contact topic 降采样/边沿记录）；
- 字段：wall time、sim time；
- 时长：GRASP 开始前 0.5 s 到结束后 1.0 s settle。

## 5. 顺序与硬门确认

保持：位移≤10 mm、掌型误差≤.03 rad、f1+f2 稳定接触≥.30 s、连续性通过、无 attachment、`physical_grasp_claimed=true`。

执行顺序：

1. 只改 `object_to_pregrasp.position_m[1]=+0.006`；
2. `repetitions:=1`、`shutdown_on_task_exit:=true`，三次独立冷启动，三个 run_id；
3. 3/3 全过第 5 节硬门后，才进入 8 s lift/place；任一不过即停在 static 阶段。

## Codex review

PENDING

## Decision and applied diff

PENDING
