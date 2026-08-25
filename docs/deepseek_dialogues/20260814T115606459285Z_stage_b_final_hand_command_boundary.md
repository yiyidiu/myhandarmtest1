# stage_b_final_hand_command_boundary

- Dialogue ID: `20260814T115606459285Z_stage_b_final_hand_command_boundary`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

Final user boundary update for the Gazebo-only Stage B task: the upper-level hand interface only needs three command classes: grasp, release, and set_configuration. The real hand strain-gauge torque feedback is already deployed and must not be simulated, redesigned, connected, or observed in this task. If Gazebo collision contact data is retained, it may exist only inside an independent acceptance verifier to prove physical contact/lift; it must not enter the hand command interface or the grasp controller feedback path. Internally a high-level grasp command may issue bounded incremental joint targets for numerical stability, but it remains one upper-level command. Please give a concise read-only review of this separation and list the minimum evidence an independent verifier still needs to reject floating/fixed-attachment lifts. Do not use tools and do not propose sensor simulation.

## DeepSeek → Codex

# Stage B 接口/验证器分离最终审查

## SUMMARY

这次边界收紧在架构上是正确的，且是可审计的：手部接口收敛为三个动词（grasp / release / set_configuration），应变片被明确排除在任务之外，Gazebo 接触数据被降格为**只存在于独立验收验证器的地面真值**。这构成了一个清晰的单向数据流：验证器读物理世界 → 判定证据，但验证器数据**绝不回流**到手命令接口或抓取控制器。唯一需要补强的执行纪律是"独立"的可证明性，以及验证器必须携带能**证伪浮空/固定附着抬升**的证据链，而不是仅仅证成"看起来抬起来了"。

## 1. 对分离的审查

**成立的部分**：grasp 内部做有界增量关节目标是合法实现细节（数值稳定性），只要上层语义仍是一个命令、且增量限幅/冻结逻辑不读取接触话题——它只能读关节状态与自身状态机。这不违反"接触数据不进控制器反馈路径"。

**必须落地的执行纪律**：
- **进程/节点级隔离**：验证器必须是独立进程（独立 node 或独立可执行），直接订阅 Gazebo 物理原始话题（对象 pose/TF、contact、physics 参数、世界状态）；不通过手接口 API 获取证据，不信任任何"命令目标达成"的上报。日志中记录验证器话题订阅列表以自证。
- **grasp 控制器的输入白名单**：控制器可读的只有关节状态、内部状态机、set_configuration 参数。接触话题、应变片话题（若有）、验证器输出，一律禁止出现在其订阅/服务依赖中；建议静态检查订阅列表作为验收项。
- **应变片纪律**：不仿真、不建模、不订阅、不观察；手接口文档中明确"反馈路径为空/仅关节状态"。
- **一个命令的语义边界**：grasp 的增量子步不得暴露为上层接口的多次命令；验收时以"单次 grasp 调用后完成闭合"为准，子步只出现在内部日志。

## 2. 验证器拒绝浮空/固定附着抬升所需的最低证据

验证器必须**同时**满足以下证据链，任一缺失即判定该 run 无效（而不是失败重跑后拼接）：

1. **物体物理身份**：物体在 Gazebo 中非 static、非 kinematic，质量/惯量为题面 0.10 kg 值；验证器直接读取世界/模型属性，不信任启动配置声明。
2. **无附着审计**：物理世界内不存在 object↔手部/手掌/臂任何链接的 fixed joint 或 attachment 插件；`allow_nonphysical_attachment` 为 false/未启用；验证器读取世界状态而非代码注释。
3. **接触为唯一支撑**：抬升起始前，物体静止于桌面且速度≈0；抬升开始后，物体与手的全部接触事件**仅来自 ≥2 个不同手指 collision 与物体 collision 的碰撞对**（物体名过滤），且不存在物体与手掌/腕/臂/其他链接的接触。接触事件必须是原始碰撞引擎输出，不是控制器汇报。
4. **接触先行于抬升**：闭合阶段先出现稳定手指-物体接触（满足时长/阈值门），之后才出现物体竖向位移；时间线由原始话题时间戳重建。
5. **抬升量证据**：物体质心（或世界系位姿）竖向位移 ≥0.08 m，且此位移来自 Gazebo 物理输出的连续位姿轨迹（采样覆盖全程），不是命令目标值。
6. **保持与漂移**：≥2 s 保持期内接触不断、物体相对手掌漂移 ≤ 预设阈值；任何接触丢失而物体仍悬空 → 判定浮空/附着。
7. **释放即下落**：release 后物体必须自由下落——接触事件清零后，物体竖向加速度与 g 一致（在求解器噪声容差内）、无向上速度突变；若物体悬停、慢飘或保持在手中而接触已消失 → 判定存在未记录的约束。
8. **负向门禁前置**：张手抬升、无接触横移、打开掉落、单指不得成功四项负向测试全部通过且其接触/位姿原始数据可复核，作为本批统计的准入条件。

第 3/4/5/7 条组合起来即构成浮空/附着的最小证伪集：**浮空**会破坏"接触为唯一支撑 + 释放即下落"；**固定附着**会破坏"无附着审计 + 释放即下落 + 接触为唯一支撑"中的至少一条。验证器输出中这八项必须逐项布尔化落盘，任何一项为假则该 run 不得进入成功率统计。

## PATCH

PATCH: NONE —— 本轮仅审查，不生成补丁。

## NOT_RUN

未调用任何工具、未读取文件；全部为基于题面公开事实的设计审查意见，须由 Codex 独立验证后采用。

## Codex review

接受上层命令与验收器单向隔离：上层只保留 `grasp`、`release`、
`set_configuration`；内部有界子步不暴露为多个上层命令。控制器不订阅 contact、
strain 或 verifier 输出。真实应变片部署完全冻结且不在本轮仿真中复刻。

验收器会独立读取 Gazebo 世界位姿和 simulation-ground-truth collision 接触，只用于
证明无固定连接、接触先于抬升、连续物理抬升/保持以及松开后的自然下落。
不采用“释放加速度必须精确等于 g”作为单点硬门，因为物体释放后可能继续与手指或桌面
发生合法碰撞；改用接触时序、连续位姿、离手下降和最终支撑状态的组合证据。

## Decision and applied diff

`ACCEPT_WITH_PHYSICS_VERIFIER_REFINEMENT_NO_PATCH`。DeepSeek未生成补丁或调用工具；
Codex按小步可视化验收实现。
