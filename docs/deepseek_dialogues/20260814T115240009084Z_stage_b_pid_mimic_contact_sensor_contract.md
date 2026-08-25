# stage_b_pid_mimic_contact_sensor_contract

- Dialogue ID: `20260814T115240009084Z_stage_b_pid_mimic_contact_sensor_contract`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

Read-only Stage B design review. Public code facts: the Gazebo hand has four active PositionJointInterface joints f1j1/f1j2/f2j1/f3j2 and four mimic joints f3j1<-f1j1, f1j3<-f1j2, f2j2<-f2j1, f3j3<-f3j2. The installed mimic plugin has two branches: without <hasPID> it calls SetPosition every simulation update; with <hasPID> it reads a ROS control_toolbox PID namespace, computes target-current error, clamps effort to maxEffort, and calls SetForce. Current URDF omits hasPID. The fingertip collision names after URDF-to-SDF conversion are f1link3_collision, f2link2_collision, f3link3_collision. gazebo_ros_bumper is installed. Target is a dynamic 0.10 kg 0.05x0.05x0.10 m box with gravity and friction. Requirements: no fixed/link attachment, at least two distinct fingers in stable contact, incremental close, >=0.08 m physical lift, >=2 s hold, bounded relative drift, and natural drop after open; four negative tests must pass before official trials. Questions: provide a minimal fail-closed contract for (1) configuring finite-effort PID mimic without claiming untested numeric gains, (2) three independent Gazebo contact sensors and object-name filtering, (3) incremental contact-aware closing and which telemetry gates must stop/fail, and (4) tests that specifically detect SetPosition-like nonphysical behavior or contact-topic false positives. Do not propose fixed joints, pose forcing, exaggerated collision boxes, or tools. Return concise public analysis, not code that assumes unverified link geometry.

## DeepSeek → Codex

# Stage B 最小化 fail-closed 契约审查

## SUMMARY

四个问题的共同原则：**契约只规定结构与可判定的失败不变量，不发布未标定的数值**；每项物理声明都需一条可自动判假的运行时检查（teleport 或假接触必产生可检测签名）。建议把所有增益、阈值列为"待标定参数"，由 Codex 用下述标定/探针实验确定后再进入统计，当前不得宣称任何数值。

## 1. 有限力 PID mimic 的 fail-closed 契约（不发布未验证增益）

- **分支可达性（启动级）**：physical_contact 世界的 URDF/SDF 必须显式含 `hasPID`；插件启动自检若在物理世界发现缺失 hasPID → 直接拒绝启动（fail-closed），任何路径都不得回落 SetPosition。同时记录插件版本/分支标识。
- **配置结构契约**：每个 mimic 关节绑定独立 PID namespace；error = target − current，target = leader 角度 × multiplier（允许带符号与 offset，符号由映射 `f3j1<-f1j1` 等显式声明并在启动时校验，防止反向跟随）；输出 clamp 到 maxEffort 后经 SetForce 下发。
- **有限力的运行时判据（核心）**：
  - maxEffort 必须有限、非零，逐 run 落盘；`|下发力矩| > maxEffort` 或 NaN/Inf 出现 → 立即停止。
  - **误差-力矩一致性**：SetForce 下若接触阻力存在，follower 位置误差必须非零增长；若"力矩饱和（≥maxEffort）且 |error| ≈ 0 持续 > T" → 判为刚度无穷大/teleport 签名，fail-closed。
  - 抗积分饱和：积分项必须有限幅/遇接触冻结，释放阶段禁止积分踢回（以"释放后 follower 力矩单调归零、无反向尖峰"为验证项）。
- **增益不做声明**：仅约束 kp≥0 有限、ki 有限且有 anti-windup、kd≥0；数值由标定程序产出——自由摆动手指的阶跃响应无持续振荡、上升时间有界；每 run 记录增益并要求跨 run 一致。**发布前任何数值都标注 NOT_CALIBRATED。**

## 2. 三个独立接触传感器与物体名过滤

- **独立实例**：在 f1link3_collision / f2link2_collision / f3link3_collision 各自挂 gazebo_ros_bumper，独立 namespace 与话题（如 `/contact_f1` 等），时间戳独立；门控逻辑**禁止合并**三路，逐指记账。
- **物体名过滤**：每路接触事件只承认碰撞对中含目标盒体确切 collision 名（转换后名称，显式前缀匹配列表）的事件；对 table/ground/他指的事件必须分类为非目标接触，绝不计数。
- **假阳性最小门**：连续接触时长 ≥ T_stable（如 200 ms，待标定）、法向力幅值高于无接触基线噪声（基线由负向测试实测，非假设）、滑动窗口内方差有界；单帧接触不成立。
- **诚实声明**：bumper 只提供 link 级接触，不给接触点位置分布；"哪部分指尖接触"的结论**不得**作为证据，除非另行标定几何（当前不假设未验证的 link 几何）。

## 3. 增量接触感知闭合与必须停止/失败的遥测门

- **闭合循环**：leader 关节以限幅增量（每控制周期 ≤ 有界步长，或等效有界角速度）下发；follower 经 §1 契约跟随。每步读取三路 bumper + 力矩 + 位置。
- **逐步冻结**：某指达到稳定接触（§2 过滤后）→ 该指 leader 目标冻结，不再前进；未接触指继续，直到：(a) ≥2 个**不同**手指稳定接触并持续 T_stable，或 (b) 剩余未接触指达到其允许最大前进量仍无接触（禁止继续推向盒/桌）。
- **必须停止/失败的遥测门（任一触发即 fail-closed，冻结目标→按有界增量张开→全量留痕）**：
  1. 任一手指 bumper 遥测缺失/超龄（>阈值）；
  2. 任一关节位置/力矩/接触力出现 NaN/Inf；
  3. 力矩饱和且误差≈0 持续超时（§1 teleport 签名）；
  4. 闭合开始前（手指仍在张开基线）出现任何与目标的接触事件；
  5. 指-指自接触或指-桌面接触（目标未参与）被错误计入；
  6. 单步下发的增量超过限幅（内部控制违规）；
  7. 抬升命令发出前，盒体相对初始静止位姿位移/转动超过漂移阈值（闭合推挤签名）；
  8. follower 实际运动方向与 multiplier 符号不一致（跟随反向）。
- 步数/时间预算超限 → 本次 run 计失败，不进入统计。

## 4. 检测 SetPosition 类非物理行为与接触假阳性的专项测试

**非物理行为探针：**
- **能量注入探针**：手指执行小幅闭合/张开循环，盒体置于**实测确认**远于任何可达指尖的位置（用 TF/关节实测保证，不用几何假设）；盒体位移/转动必须低于基线噪声，且三路接触全空。任何位移即 SetPosition 推挤/求解器伪力签名。
- **穿透探针**：将盒体置于实测已知位姿，命令指尖目标穿过盒体所在区域；诚实契约下必然出现"力矩饱和 + 跟踪误差增长 + 停在表面"。失败签名：接触已建立而 follower 误差≈0 且力矩有限（无穷刚度），或盒体被击飞而 follower 力矩近零（能量注入）。
- **误差-力矩一致性检查**：每次闭合接触建立时记录（接触力大 ⇔ 误差非零）相关性；违反即 fail。

**接触假阳性探针：**
- **无接触横移/悬停**：手在盒旁无接触运动，三路 bumper 必须全空（含瞬态）；任何偶发报告即阈值或噪声标定失败。
- **桌面假接触**：目标不在场时触桌，必须被分类为非目标、零计数。
- **指-指假接触**：无物体时指间相互接触，必须分类为自接触、目标接触计数为零。
- **弹跳/短接触**：快速触碰立即分离，单次接触不得满足 T_stable 稳定判据。
- **归属一致性审计**：bumper 声称 f_i 与盒接触时，用关节位置 + 盒体实测位姿做粗一致性核验；若 f_i 关节位置表明其远离盒体区域仍报接触 → 假阳性（此检查只依赖实测位姿，不依赖未验证几何）。

**配置审计（静态）**：physical_contact 世界中 hasPID 分支可达性自检；启动即校验四组 mimic 映射、PID namespace、maxEffort 有限性。

## 附：前置约束

- 统计前硬门禁不变：张手抬升、无接触横移、打开掉落、单指不得成功四项负向测试全部通过；§4 探针建议并入同一负向批次。
- 本审查不生成补丁；PATCH: NONE。
- NOT_RUN：未调用任何工具、未读取代码；全部结论仅基于题面公开事实，所有阈值/增益为待标定参数，未经验证不得引用。

## Codex review

部分接受。接受 physical-contact 模式必须显式进入有限力 PID mimic 分支、四个 mimic
命名空间与 effort 上限必须可审计、三指仿真接触必须独立且按目标 collision 名过滤、
闭合必须增量限速，以及四项负向测试先于正式统计。

不直接采用任何未实测 PID/时长/力阈值。也不把“接触力与 follower 误差相关”作为
SetPosition 行为的充分判据：主动关节的位置控制、接触求解器和 link 惯性同样影响该关系，
必须由 Gazebo 实际阶跃、阻挡和自然掉落实验综合判定。用户随后澄清真实手没有触觉，
因此 bumper 只能是仿真验收真值，不能成为未来实机接口假设。

## Decision and applied diff

`PARTIAL_ACCEPT_NO_DEEPSEEK_PATCH`。DeepSeek未生成补丁或调用工具；参数由 Codex
小步实测标定，每一步先展示结果再继续。
