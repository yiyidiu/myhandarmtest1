# f3j1_single_velocity_spike_after_collision_proxy

- Dialogue ID: `20260814T163829341172Z_f3j1_single_velocity_spike_after_collision_proxy`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

这是上一轮 hand_collision_mesh_and_contact_stability_review 的实测跟进。请只读分析，不运行工具、不改文件。

Codex 将 9 个手部 link 的 collision 从高面数 STL 改为 primitives，visual 未改，PID/阻尼/mimic 未改。URDF 和 `gz sdf -p` 均确认：手部 collision 中无 mesh；handbase 1圆柱；两个近端各圆柱+定向盒；其余中/远节各一个圆柱+两球组成 capsule。生成 SDF 无 self_collide 标签（默认 false）。

随后原阈值跑 5 次 RELEASE/GRASP/RELEASE（15窗口）：前13个 PASS，第14个（第5轮GRASP）唯一失败，原因 `f3j1 settled velocity`。详细数据：
- f3j1 是 f1j1 的掌型从动关节，GRASP/RELEASE 两者 f1j1 目标都恒为0.051 rad，不参与屈伸。
- 此失败窗口 f3j1 位置范围0.0027635 rad；tail velocity P95=0.0009494 rad/s，但单点 max=0.4410558 rad/s，阈值0.05，所以 fail-closed。
- 同一窗口 f1j1 max速度0.0009439 rad/s，其余从动关节均正常。
- 替换前一轮15/15中，f3j1最坏max仅0.0009525 rad/s；替换后前4个GRASP的f3j1 max仍约0.000946~0.000954。
- f3j1当前 inertia轴约6.95e-5 kg m²，joint damping=.05, friction=.01；mimic PID P=3,I=0,D=.08,maxEffort=4,sensitiveness=.0005；未配置nearTargetEffort/nearTargetError（因此近目标仍可用完整上限）。
- 没有环境接触证据；新proximal proxy理论上不应因self_collide=false与掌座接触。

Codex 已增强评价器，下一次会记录最大速度样本的精确elapsed、signed velocity、position；阈值不放宽。

请回答：
1. 这个“P95正常但单点0.441”的最可能原因排序：真实f3j1动力学脉冲、碰撞代理/自碰撞、mimic deadband/PID切换、Gazebo service采样伪影或其他。
2. 下一步最小隔离方案是什么？是原配置重复5轮、临时恢复仅f3link1旧collision、还是先给f3j1加nearTarget limiter？要求一次只改一个变量。
3. 如果重复出现，f3j1怎样保守修改才能不改变掌型语义且避免掩盖故障？给出候选参数方向，但所有数值标明待Codex实测。
4. 是否应因1/14失败立即否决全部primitive collision，还是先完成可归因复现？

请明确撤回任何与新证据冲突的旧判断。

## DeepSeek → Codex

# UPDATED SUMMARY

先做一个只用题面数字的核算，它决定整轮判断：失败窗口 f3j1 位置总范围 0.0027635 rad，而速度单点 0.441 rad/s。若评价器以 100 Hz 差分求速度，0.441 rad/s 意味着单对样本 Δpos≈0.0044 rad——**大于整窗总范围，内部矛盾**；只有当该速度对应 ≈1 ms 尺度的单步事件（Δpos≈4.4e-4 rad）时数字才自洽。f3j1 阻尼时间常数 τ=J/b=6.95e-5/0.05≈1.39 ms，与"任何脉冲在 ~2 ms 内衰减、只留一个样本"完全吻合。**结论：这是单个 1–2 ms 物理步瞬态，不是持续振荡；P95 正常正是瞬态事件的期望形态，fail-closed 抓到的是一次真实异常而不是噪声。** 因此不应否决 primitives 方案，但必须先把事件归因再继续。

# RETRACTIONS（撤回与新证据冲突的旧判断）

1. **撤回**上一轮隐含的"只换 collision、不动 PID/阻尼/mimic ⇒ 空载门槛应无扰通过"的信心。碰撞管线替换本身改变了约束/接触求解路径，之前 15/15 安静的 f3j1 出现单样本尖峰正是此类扰动；上一轮风险清单只覆盖了接触期风险，**遗漏了"1 ms 步长下近静态 mimic 关节的单步瞬态"这一类**。
2. **撤回**对 self_collide 的含糊处理，明确 SDF 语义：默认 `self_collide=false` = **同模型 link 之间不互碰**。因此"新代理与掌座相碰"的假说在默认配置下先天关闭；副作用是**指-指、指-掌自碰探针在此配置下永远不会触发**，后续接触探针只能用"手 ↔ 环境/物体"对。
3. **撤回**"没有环境接触证据 ⇒ 接触不可能"的推理有效性：未订阅接触话题的缺席只是"未观测"，不是"不存在"；复现轮必须订阅接触话题取得正面证据。
4. **撤回**"f3j1 已有 15/15 清洁记录即充分稳定"的判断：0.441 表明该关节存在**低频、单步、快衰减**的瞬态模式，15 窗口只是样本不足未捕获，不是不可能发生。

# Q1 原因排序（按证据强度，全部待复现证实）

1. **单步动力学瞬态（摩擦/粘滑 breakout 或求解器约束校正弹出）**。量级核算：friction=0.01 的库仑摩擦反向切换产生 Δτ≈0.02 N·m，作用在 J=6.95e-5 上一仿真步（1 ms）即 Δv≈0.29 rad/s，与观测 0.44 同数量级；τ≈1.39 ms 保证只有单样本可见；leader f1j1 同窗 max 仅 0.00094 也符合"脉冲不来自 leader"。
2. **评价器采样/时标伪影**：RTF 停滞追赶、重复/近重复时间戳导致单对样本 elapsed 异常小，Δpos 量化噪声被放大。判别由增强评价器的 signed velocity / position / elapsed 三字段完成——这正是它们存在的意义。
3. **mimic deadband（sensitiveness=0.0005）/PID 切换**：量级核算否决其单独致因——穿越 deadband 的力矩步 ≈P×sens=0.0015 N·m → 单步 Δv≈0.02 rad/s，**比观测小约 20 倍**；除非与 D=0.08 对 follower 自身速度读数毛刺的放大叠加（0.08×0.44≈0.035 N·m → ~0.5 rad/s/步），但那是"读数毛刺在先"，鸡生蛋问题，故降级。
4. **碰撞代理/自碰撞**：被 self_collide=false 语义排除在默认路径之外；仅当手与臂为不同模型、或手 link 与环境存在一次未被记录的 graze 时才可能。当前证据最弱，但必须用接触订阅正面证伪。
5. **其他**：RTF 打嗝引发的 ODE 步长/迭代瞬态、单步"软爆自恢复"——与早期爆炸同族、量级小得多，需连续监视。

# Q2 最小隔离方案（一次一个变量）

**第一步（唯一零变量步骤）**：保持当前配置原样，重复 5 轮（15 窗口），同时升级观察：
- 增强评价器记录最大样本的精确 elapsed / signed velocity / position；
- **新增 ≥500 Hz（理想 1 kHz）触发式环形缓存**：阈值违规时把前 N 秒高速位置/速度转储。理由：事件仅 1–2 ms，100 Hz 差分采样根本无法分辨"真实单步脉冲"与"采样伪影"，100 Hz 下的 elapsed 字段只刻画采样对、不刻画事件本身；
- **本轮起订阅接触话题**，正面记录有无手↔环境接触。

**分支（只有复现后才允许动任何变量）**：
- 签名 A（高速缓存显示位置/速度/时标自洽的 1–2 ms 衰减脉冲，无接触事件）→ 单变量改 f3j1 friction 或 damping（二选一，见 Q3）。
- 签名 B（脉冲伴随 f3link1 与环境的接触事件）→ 才临时恢复**仅 f3link1** 的旧 collision 作诊断对照。
- 签名 C（脉冲伴随 mimic effort 步进/顶格）→ 才考虑给 f3j1 加 nearTarget 限制器。
- **5 轮不复发** → 保留 primitives，接触话题连续记录，该事件归档为低概率单事件，阈值不变。

**拒绝两个错误起点**：先恢复 f3link1 旧 collision（改动的变量针对的是当前证据最弱的假设）；先加 nearTarget 限制器（若真因是物理脉冲，这会降低力矩授权、掩盖症状）。

# Q3 复发时的保守修改（不改掌型语义、不掩蔽）

掌型语义 = f3j1 以 multiplier +1 镜像 f1j1、两条命令恒持 0.051。以下全部**一次只动一个变量、改后重跑 15 窗口原阈值**，数值全部 TO_BE_MEASURED：

- **候选 1（对应签名 A，首选方向）**：f3j1 friction 0.01 → 0.005。直接减半粘滑切换力矩跳变（0.02→0.01 N·m），位置保持由 PID 负责，镜像语义零改变。方向是**降**摩擦，不是升。
- **候选 2（对应签名 A，次选）**：f3j1 damping 0.05 → 0.075。τ 从 1.39→1.03 ms，更快吸收单步脉冲；语义不变。风险：历史上 damping 0.20 曾爆炸，任何上调必须留大余量并全部门槛重过。
- **候选 3（仅对应签名 C）**：f3j1 增加 nearTargetError/nearTargetEffort（error 取 sensitiveness 的数倍、effort 取 maxEffort 的一小部分量级），降低近目标力矩授权；不改变镜像关系。
- **掩蔽判定（三条件必须同时成立）**：新参数下 (i) 正常跟踪的 |effort| 仍有充分余量；(ii) 位置跟踪误差不恶化；(iii) 0.05 阈值与全部门槛不变。
- **明确禁止**：调大 maxEffort、放宽 0.05、改 multiplier、改 f1j1 目标、跳过任何门槛。

# Q4 是否否决 primitives

**不否决。** 13/14 通过且其余窗口 baseline max≈0.00095 与替换前一致，整体等价性成立；1/14 的失败是 fail-closed 门正确捕获的一次待归因单事件，不是系统性劣化。全盘回退会重新引入已证实的动-动 trimesh 接触风险（更糟的已知问题）。正确路径：**原配置 + 增强仪器复现 → 签名归因 → 局部单变量修复 → 重跑门槛**。只有复现稳定指向 f3link1 代理几何接触时，才只缩小/退让该 link 的代理，绝不全盘回退。

# RISKS

- 增强评价器若仍以 100 Hz 采样，"exact elapsed"只能刻画采样对、不能刻画 1–2 ms 事件本身；触发式高速缓存是归因成立的硬件前提，缺失则 Q2 分支无法执行。
- 接触话题必须在本轮就订阅；否则"无接触"仍是未观测状态，签名 B 无法证伪。
- 任何对 f3j1 的参数修改若与其它改动同批验证，等于再次制造不可归因实验；一次一个变量必须严格执行。
- friction/damping 都动过爆炸史，修改后除 15 窗口门槛外，应加"前 100 ms 高速快照"监视复发。

# NOT RUN

未调用任何工具、未读写文件、未执行测试；全部为基于题面公开事实与量级核算的只读审查意见。0.441 事件的归因（含 Q1 排序）必须经 Q2 复现与高速缓存数据证实后才能作为结论引用；所有候选数值待 Codex 实测标定。

## Codex review

PENDING

## Decision and applied diff

PENDING
