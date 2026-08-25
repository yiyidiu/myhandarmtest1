# hand_collision_mesh_and_contact_stability_review

- Dialogue ID: `20260814T153854118387Z_hand_collision_mesh_and_contact_stability_review`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

请针对 ABB IRB120 + 三指手 Gazebo Classic/ODE 的剩余基础隐患做只读审查。不要假设你能读工作区，只依据以下实测事实给出可执行建议；不得建议通过放宽阈值掩盖问题。

已确认并实测：
- GRASP/RELEASE 只控制主动关节 `[f1j1,f1j2,f2j1,f3j2]`，其中 f1j1 是掌型/手指布置关节，两条命令都固定为 0.051 rad；三路屈伸分别从 0.20 到 1.30 rad。机械臂只在启动时初始化和保持，命令本身不控制臂。
- 被动耦合为 f3j1<-f1j1、f1j3<-f1j2、f2j2<-f2j1、f3j3<-f3j2；Gazebo mimic 插件走 PID/SetForce，不走 SetPosition。
- 三个远指节惯量轴约 2.45e-5 kg m^2。原阻尼 0.20 Nms/rad 在 1 ms 仿真步长下导致数值爆炸；改为 damping=0.01、friction=0.02，passive PID P=1,I=0,D=0，maxEffort=0.20、nearTargetEffort=0.05 后稳定。
- 已真实完成 5 次 RELEASE/GRASP/RELEASE，共 15 个独立 5 秒窗口，15/15 PASS。最坏被动尾段范围 0.001635 rad，最坏速度 0.00182 rad/s，均无 NaN/越限；关系误差最大 0.02850 rad。
- 当前 ROS 图只有单一 gazebo、robot_state_publisher、startup_coordinator、hand_commander；arm/hand 控制器资源无重叠。
- 所有质量/惯量矩阵已检查为正定，条件数<6。

尚存风险：hand visual mesh 同时被用作 collision mesh，动态 collision 三角面数很高：handbase 466230，两个近端布置指节各160098，中段约20266，远段约14192。物理抓取世界使用 ODE 1 ms、100 iterations、contact kp=1e6,kd=10,max_vel=.1,min_depth=.001；物体为50x50x100 mm、0.10 kg。当前还未把物体接触纳入这一轮空载稳定验收。

请回答：
1. 高面数动态三角网格对 ODE 接触、实时率、抖动/爆炸的具体风险；是否应在进入真实接触抓取前，将 collision-only 网格替换为每个 link 的低面数凸包，同时保留原 visual mesh？
2. 单凸包可能填平哪些凹区、导致什么假接触；对这类三节手指，单凸包、分解凸包、盒/胶囊哪个是最稳妥的第一版？给出明确选择与验证条件。
3. 当前 PID/mimic/关节阻尼还有哪些会导致空载抖动、接触时积分风up、关节散架或数值注能的风险？请区分必须现在修复和接触实验中再调。
4. 给出一个最小、分阶段、fail-closed 的验证顺序：空载、被动外力/碰撞、静态夹持、物理抬升；每一步记录哪些量，哪些现象必须立即停止。
5. 不允许用固定关节、attach插件或物体位姿脚本来冒充物理抓取。

Codex 将独立决定是否采用、生成碰撞网格并运行所有测试。请返回简明的审查结论，不要输出大段伪代码。

## DeepSeek → Codex

# 审查结论（Gazebo Classic / ODE 基础隐患，只读）

## SUMMARY

三处结论：① 高面数动态 trimesh 必须在进入接触实验**之前**替换为每 link 低面数凸碰撞体——这不是性能问题而是数值稳定性问题（ODE 对动-动 trimesh 接触本就不可靠，配合 kp=1e6 与 1 ms 步长，会把接触不确定性与边缘法向跳变直接变成抖动/爆炸源）；② 首选碰撞体是**指节用胶囊、掌基座用盒**，单凸包仅作诊断对照，不推荐作第一版；③ 当前 mimic 回路 I=0 恰好没有积分风up 风险，残余风险集中在**接触弹跳与 1 ms 采样率附近的高频能量泵**，属于接触期再调项，不在现在修。空载 15/15 的结论只在"碰撞体未变"的配置下有效，替换碰撞体后必须重跑同一门槛。

## 1. 高面数动态三角网格风险与替换决策

- **接触稳定性**：ODE 官方立场是 trimesh 面向静态几何；动-动 trimesh 接触在三角形边缘处法向跳变、接触点抖动，kp=1e6 的硬接触会把这些抖动放大成高频力，1 ms 步长下直接进入震荡/爆炸区间。
- **注能与穿透**：边缘处深穿透 + 硬恢复（kp=1e6、min_depth=0.001）对 0.10 kg 轻盒是现成的能量注入通道——轻物体被弹飞/穿刺的经典配方。
- **实时率**：handbase 466k + 两近端各 160k 三角面，接触阶段 RTF 会崩；RTF 崩又反哺步长/接触求解恶化，形成正反馈。
- **决策：是，必须替换**。每个 link 用低面数碰撞体（见 §2），visual mesh 原样保留；替换后**立即重跑空载 15 窗口门槛**，阈值不变、不得放宽，通过才算碰撞体替换完成。

## 2. 碰撞几何选择（第一版明确选择）

- **首选：指节用胶囊、掌基座用盒（或少量盒的并集）**，紧贴各自 STL 的实测包围几何拟合。胶囊法向连续、旋转对称，无边缘跳变；盒用于需要平面接触的基座。
- **单凸包的风险**：会把指节凹面/基座凹槽填平，产生真实几何不存在的假接触（"看得见碰不到"或提前接触）；只可作诊断对照（证明"换碰撞体后行为未变"时用），不作正式配置。
- **分解凸包（V-HACD）**：保留凹形但引入内部接缝与多重接触对，边缘不连续问题依旧存在——**延后**，只有当某指节存在对抓取关键的凹功能面时才启用。
- **验证条件**：(a) 重跑空载 15/15 门槛通过；(b) 实测非接触位姿下零接触事件、物体零位移；(c) 已知接触探针只在预期指尖区域报接触；(d) 全运动窗口 RTF 达标并留档。

## 3. PID / mimic / 阻尼残余风险（必须现在 vs 接触期再调）

- **积分风up：当前不存在**。mimic 回路 I=0，P-only + maxEffort 钳位本身即抗风up；**保持 I=0 进入物理世界**，任何未来加 I 都必须带抗风up与饱和证据。
- **P-only 弹性引起的稳态关系误差**：空载已见 0.0285 rad，接触负载下会更大。这是合规弹性，不是 bug，但必须设上界（候选：接触期 relation error ≤0.15 rad，TO_BE_CALIBRATED），超界即停止——防止从动指实际几何偏离设计抓取构型。
- **damping=0.01 的欠阻尼**：接触时指尖可能弹跳/极限环。这是**接触期再调**项；且历史证明 damping 上调曾引发爆炸，任何阻尼改动必须重跑空载门槛。
- **D=0 目前正确**：1 ms 步长下 velocity_error 差分噪声大，先不加 D；若接触弹跳需要阻尼，优先降 P 或 maxEffort，再考虑小 D。
- **残余注能通道（必须现在堵）**：SetForce 钳位目前是声明不是证据——必须加 effort 遥测（raw/clamped 双记录），并验证 |clamped| ≤ maxEffort 恒成立；9 个手关节全量有限/限位/速度门（见 §4 停止条件）；**接近 1 ms 采样率的高频振荡是数值泵的签名**，速度谱监控必须纳入。
- **"散架"**：有界力矩 + 正定惯量下不可能真散架；可见分离只来自 trimesh 穿透——由 §1 碰撞替换解决。

## 4. 最小分阶段 fail-closed 验证顺序

**P0 空载（已完成，碰撞替换后重跑）**：记录 9 关节位置/速度、三路 relation error、5 s range、f1j1 保持、RTF。**立即停止**：任何 NaN/Inf、越限、速度超 sanity cap（候选 10 rad/s，TO_BE_CALIBRATED）、f1j1 偏离 0.051 超阈。

**P1 被动外力/碰撞探针（无抓取）**：非接触扫描（物体位姿实测零位移、零接触事件）＋受控指尖触碰（接触只出现在预期指、力有界、无弹跳/爆炸、relation error 有界、effort 饱和合理）。记录：raw/clamped effort、接触事件、物体位姿、relation error、RTF、求解器参数哈希。**立即停止**：禁用接触发生时出现接触、无接触而物体移动、effort 饱和且误差≈0、任何 NaN/RTF 崩。

**P2 静态夹持（不抬升）**：GRASP 夹住静止物体并保持。验收（仅验证器）：≥2 个不同手指稳定接触、闭合期间物体位移低于阈、接触力振荡有界、保持 ≥2 s 漂移有界。**立即停止**：物体被推离/刺穿、保持期接触丢失、单指夹持、出现物体↔掌/腕/臂接触。

**P3 物理抬升**：抬升 ≥0.08 m → 保持 ≥2 s → RELEASE 自然下落；抬升/保持期间接触不断、相对漂移受限、释放后无向上速度尖峰。**前置**：四项负例（张手抬升、无接触横移、打开掉落、单指不得成功）全部通过后才开放官方统计。**立即停止**：悬空时接触丢失（浮空签名）、出现非手指接触、释放后不落或缓慢飘落。

## 5. 物理诚实性

不变约束：无固定关节、无 attach 插件、无物体位姿脚本；验收只采信 Gazebo 物理原始输出（物体位姿序列、原始接触事件、世界属性），不采信命令目标值或控制器自报。验证器必须逐 run 审计：物体 non-static、物体↔手/臂无 fixed joint、attachment 未激活、求解器参数哈希与配置一致。

## PATCH（最小改动，全部由 Codex 映射实现）

- 生成每 link 低面数碰撞体脚本（胶囊/盒，紧贴 STL 拟合），visual 不动；配置双世界隔离不变。
- mimic 插件加参数门控 effort 遥测发布（raw/clamped），默认关闭生产、诊断开启。
- startup_coordinator 增加：模型加载完成门、teleport 读回验证、unpause 后全 9 关节有限/限位/速度 sanity 门，任一违规在轨迹下发前中止并留完整快照。
- 被动关节状态覆盖（只读发布、无 transmission、无位置控制器），供验证器使用。
- 以上不改任何现有阈值；新增阈值一律 TO_BE_CALIBRATED 并在开跑前写死。

## TESTS

- T1：碰撞替换后重跑空载 15 窗口门槛（原阈值）。
- T2：非接触/已知接触探针（零假接触、接触位置正确、RTF 达标）。
- T3：effort 钳位验证（|clamped| ≤ maxEffort 全时段成立）。
- T4：P1→P2→P3 逐相执行，每相 N≥3 全新 spawn，逐相 JSON 落盘、禁止跨相缝合。
- T5：四项负例硬门禁。
- T6：sanity 门注入违规，验证在轨迹前中止并留证据。

## RISKS

- 碰撞替换若与任何 PID/阻尼改动同批提交，15/15 复测无法归因——必须分开验证。
- ODE trimesh 仍存在于 visual 或残余 collision 引用时，验证器的接触判定会被污染；替换后需确认无 link 仍以 trimesh 参与动-动接触。
- RTF 达标是接触实验的前置门，不是可选项；RTF < 1.0 时完成的接触实验结果不可采信。
- 任何"调参数解决接触弹跳"的改动都必须重过 P0 门槛，防回归爆炸。

## NOT RUN

未调用任何工具、未读写文件、未执行测试；全部为基于题面实测事实的审查意见。15/15 与爆炸数值均采信你方报告，未独立复现；所有候选阈值未经实测校准，须由 Codex 验证后引用。

## Codex review

PENDING

## Decision and applied diff

PENDING
