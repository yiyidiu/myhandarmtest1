# stage_b_sim_contact_vs_real_strain_gauge_boundary

- Dialogue ID: `20260814T115333031615Z_stage_b_sim_contact_vs_real_strain_gauge_boundary`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

User hardware correction: the real three-finger hand has no fingertip or tactile sensors. It only has strain gauges at the base of the fingers. Update the prior Stage B contract boundary. This current task is Gazebo-only physical-contact grasp validation, so Gazebo contact sensors may be used strictly as simulation ground truth / acceptance instrumentation to prove which collision pairs touched the object. They must not be described as real hardware capabilities or required future real-hand inputs. The current task must not connect to real hardware. Please state concisely: (1) what Gazebo-only controller and verifier are allowed to use now, (2) what must be logged/documented so simulated contacts are not confused with hardware tactile sensing, (3) how a future real-hand interface would represent base strain-gauge signals without pretending to localize fingertip contact, and (4) which earlier suggestions must be narrowed or withdrawn. Do not use tools and do not propose starting real-hardware work.

## DeepSeek → Codex

# Stage B 契约边界更新（仅 Gazebo，真实手指基座应变片）

## SUMMARY

硬件事实变更不削弱 Gazebo 验证目标，但改变了接触证据的语义边界：**Gazebo contact sensor 在本任务中的唯一合法身份是"仿真验收仪器/地面真值"，用于证明哪对碰撞对接触了目标物体；它不是、也不得被描述为真实手部传感器，更不得作为未来实机抓取的输入需求。** 本任务严禁连接真实硬件，因此"实机接触证据"完全落在范围外。

## 1. Gazebo-only 允许使用范围

- **控制器**：PID mimic（hasPID 分支、有限 maxEffort、抗积分饱和）驱动四主动关节；增量限速闭合状态机；fail-closed 遥测门（力矩、位置、超时、NaN/Inf、误差-力矩一致性）。所有这些在纯 Gazebo 中运行，无任何实机接口。
- **验证器**：三指 bumper 接触话题**仅作仿真地面真值**——用于验收判定"≥2 个不同手指稳定接触目标盒体"、接触时序、抬升/保持/掉落期间的接触变化，以及负向测试中"无接触"的证伪。它们可以参与仿真内统计判据，但任何文档/日志中必须标注其身份。
- 应变片（若存在仿真模型）在 Gazebo 中可作为补充证据通道，但不得当作接触定位依据。

## 2. 必须记录的防混淆文档/日志

- **统一命名**：Gazebo 接触话题/插件命名为 `sim_ground_truth/contact_*` 或带 `GT_` 前缀；日志字段显式写 `sensor_role: simulation_ground_truth`。
- **能力声明**：每份报告/代码注释注明"该接触信息来自 Gazebo 碰撞引擎，是验收仪器；真实手部无指尖/触觉传感器，仅指基应变片"。
- **每 run 留痕**：记录哪些接触结论依赖 bumper GT；若未来有应变片模型输出，须分通道记录且明确"应变片非接触定位"。
- **禁止事项**：不得将接触点数/接触位置分布作为"实机可达证据"写入结论；不得在接口文档中把 bumper 列为未来实机输入。

## 3. 未来实机接口对指基应变片的表示

- 应变片只提供**每指基座的弯矩/法向力标量（或低维向量）**，不含接触位置信息；接口层应输出：每指应变信号时间序列（含标定系数、零点、饱和限），不输出接触点、接触法向或接触数。
- 抓取判定降级为**间接证据**：每指应变超阈 + 关节位置停滞 + 手指误差-力矩一致性；"≥2 指接触"在实机侧只能是"≥2 指应变信号满足阈值判据"，**不得声称定位接触点或指腹接触**。
- 若未来需要定位，须新增真实传感器或肌骨/几何估计，并在接口中显式声明估计性质与不确定性；当前任务不设计、不预留该能力。

## 4. 需收窄或撤回的早前建议

- **撤回**：任何把 Gazebo 接触位置/接触点分布当作实机能力或"真实手部将来可用输入"的表述；任何暗示"指腹接触定位"可迁移到当前实机手的描述。
- **收窄**：§5 的"接触证据主通道"必须标注为仅限 Gazebo 验收；其"接触法向/位置"项仅在 sim GT 语境保留，且不可写入未来实机需求。稳定接触判据在 Gazebo 中可用 bumper GT，在实机语境只能退化为"应变 + 位置停滞 + 力矩"间接判据。
- **收窄**：负向测试中"三路接触全空"的判定只对 Gazebo 成立；实机对应判据（若未来做）只能要求应变信号无上升、物体无位移，不做接触点断言。
- **收窄**：仿真统计判据中依赖 GT 接触的部分，必须与可迁移到实机的部分（力矩、位置、漂移、抬升量、掉落）分列记录，防止把"GT 接触成立"冒充"实机可测判据"。

## PATCH / NOT_RUN

PATCH: NONE。NOT_RUN：未调用任何工具、未读取文件；全部为基于题面公开事实的设计意见，须由 Codex 独立验证。

## Codex review

接受传感器能力边界：本轮 Gazebo bumper 只作为 `simulation_ground_truth` 验收仪器；
真实手仅有指根应变片，不能输出或声称指尖接触位置、接触法向或触觉分布。
未来实机接口不属于本轮范围，因此只在报告中记录该硬件事实，不实现也不连接实机。

## Decision and applied diff

`ACCEPT_SCOPE_BOUNDARY_NO_DEEPSEEK_PATCH`。DeepSeek未生成补丁或调用工具；Codex
将在仿真话题、结果字段和报告中显式标注 simulation-only ground truth。
