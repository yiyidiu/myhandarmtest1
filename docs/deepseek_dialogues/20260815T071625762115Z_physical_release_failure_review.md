# physical_release_failure_review

- Dialogue ID: `20260815T071625762115Z_physical_release_failure_review`
- Session: `session-03e7dfc9-421b-41f0-b10f-6b530091dc11`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

只读对抗审计：真实 Gazebo 物理抓取已经完成抓取、抬升、2 秒保持和放回桌面，但 RELEASE 失败。请只根据下列公开事实分析，不要调用工具，不要写代码或文件。

硬约束：固定掌型，GRASP/RELEASE 只改变三根手指屈伸；禁止 attachment、fixed joint、teleport、物体自运动、隐藏导向和关闭碰撞；释放必须有物理证据，不能仅放宽容差；Codex 是唯一主工作区写入者和最终裁定者。

当前命令：主动关节 [f1j1,f1j2,f2j1,f3j2]；f1j1 是掌型且 OPEN/CLOSE 都为 0.18；OPEN 屈伸为 [0.20,0.20,0.20]；CLOSE 为 [0.85,0.85,0.90]，8 秒同步轨迹。f1j3/f2j2/f3j3 分别 mimic 相应屈伸主动关节，有限力矩。目标物体 0.10kg、尺寸 0.05x0.06x0.10m，真实接触和摩擦。

本次独立完整运行：闭合物体位移 9.355mm（阈值10mm）；f1/f2 多指接触；物体真实上升43.672mm，工具上升42.656mm，二者增量分歧8.105mm（阈值10mm）；保持2.004s时掉落0.0147mm；放回后物体相对初始位姿误差13.532mm（阈值15mm），工具放回误差1.240mm。直到释放前，f1/f2接触连续26.99s。

第一次 RELEASE：轨迹 action 返回成功，但10个验证样本未全部满足主动关节目标±0.05，failure=active joint verification failed。第二次 RELEASE：FollowJointTrajectory 返回 -5 GOAL_TOLERANCE_VIOLATED，f2j1 goal error 0.065164rad。程序正确判整次失败。当前代码丢失了第一次 RELEASE 的逐关节 actual/error，也没有在 RELEASE 后独立验证：多指夹持是否解除、手撤离后物体是否留在桌面。

请回答：
1. 这组证据最可能区分哪些物理/控制原因，哪些不能据此断言？
2. 下一次运行前最小且 fail-closed 的诊断补丁应记录哪些字段？
3. 不改变碰撞/摩擦、不伪造释放的最小释放状态机应怎样设计？重点判断能否把“轨迹达到0.20rad”与“物体已物理释放”拆成两个门槛，并用撤手后物体留桌证明释放。
4. 是否可以把 OPEN 屈伸目标改到不与桌面/物体强迫冲突的更大角度（例如0.27~0.30rad）；若可以，必须配套哪些接触解除、指间净空和物体静止证据，避免只是承认卡住的位置？
5. 给出一次诊断、一次候选修复、最后两次冷启动完整运行的最小串行验收方案；禁止建议继续大范围调参。

输出简洁的公开工程结论，不给最终 PASS。

## DeepSeek → Codex

## 1. 这组证据能区分和不能区分什么

**可以确定的：**

- 失败是**单关节、小残差、持续未张开**型：第二次明确是 f2j1，goal error 0.065 rad；最可能 f2j1 实际停在约 .265 rad，仍偏向 CLOSE 侧，而不是全局控制器崩溃。
- 第一次“action 成功 + 后续 10 样本验证失败”说明控制器的“终点瞬时达标”与“张开状态持续保持”是两回事；现有代码正确判失败，但没有留证。
- 抓取、抬升、保持、放回的物理证据基本成立；问题集中在“放回后的手指退出”，不是掩盖早期抓取失败。

**不能据此断言的：**

- 不能断言 f2 是被目标物体、桌面、还是指-桌/指-物组合楔住；
- 不能断言第一次 RELEASE 中哪个关节先失败、是否曾短暂到 .20 后回弹，因为没有逐样本 actual/error；
- 不能断言是否力矩饱和、粘滑或几何卡死，因为没有 effort/velocity 序列；
- 不能断言 f1/f3 第一次已解除接触，只能推断第二次只有 f2j1 报 goal error；
- 不能断言撤手后物体会留在桌面——目前根本没有撤手证明。

## 2. 下一次运行前的最小 fail-closed 诊断字段

对每次 RELEASE attempt 保留，不得只存汇总布尔：

- 每个验证样本：wall/sim time、4 主动关节 target/actual/error/velocity/effort，4 mimic 关节 actual/relation error；
- FollowJointTrajectory：error_code、error_string、goal/path tolerance、是否 action success；
- 第一次 RELEASE 的 10 个样本全部逐样本落盘；
- RELEASE 前后及每次 attempt 后 settle：object pose(x,y,z,quat)、相对 placed pose 和 initial pose 位移；
- 每次 attempt 后 contact：f1/f2/f3 各 family 是否仍接触 target、具体 collision pairs、最后接触/解除时间戳；
- f2j1 的 effort 与 velocity 序列，至少 20–50 Hz，覆盖命令期 + 1 s settle；contact 从 100 Hz topic 记录边沿。

## 3. 最小释放状态机：把“关节到 .20”与“物体已释放”拆开

最终 PASS 必须同时过四道门，任何一道失败就是整次失败，不允许用第二次 RELEASE 掩盖：

1. **JOINT_OPEN_GATE**：一次 RELEASE 命令，action success 且所有主动关节在后续连续样本中 ≤0.05 rad；f1j1/f3j1 掌型保持 ≤0.03 rad。
2. **CONTACT_CLEARED_GATE**：关节门过后，target↔finger 接触对在 ≥0.5 s 内全部为零（f1/f2/f3 都清），且 contact topic 非 stale。关节到 .20 只是必要条件。
3. **OBJECT_STAY_DURING_RELEASE_GATE**：RELEASE 期间物体相对 placed pose 位移 ≤0.015 m，且无被手指带起/推移。
4. **RETREAT_PROOF_GATE**：接触清零后，执行一段慢速、已碰撞检查的逆 approach 撤手（例如 −x 40 mm）；撤手后 2 s 内物体相对 placed pose 位移 ≤0.015 m、仍在桌面、target↔finger 接触仍为零。

这样即使手指关节到达 .20，只要接触未清或撤手后物体跟手，就不算释放。最终验收不应把 `release_attempts=2` 当通过路径；第二次 attempt 只能用于诊断，不能缝合为 PASS。

## 4. 能否把 OPEN 目标改到 .27–.30

**不应采用，方向反了。** 当前 OPEN=.20、CLOSE=.85，数值越小越张开；f2j1 停在约 .265，正位于 .20 与 .85 之间，说明它“没开够”。把 OPEN 目标提高到 .27–.30 是把目标移向当前卡住值，等于用改目标名目放宽释放。

若几何证据证明必须更大净空，下一步应是更小角度（如 .15/.12，一次一个值），且必须配套：

- 空载 RELEASE/GRASP/RELEASE 9/9 重验；
- 候选 OPEN 位形无手部自碰撞（MoveIt state validity）；
- target↔finger 接触全部解除并持续清零；
- 释放期间物体相对 placed pose 位移 ≤.015 m；
- 撤手后物体留桌 2 s；
- f2j1 在候选目标上持续 ≤.05 rad。

若改为更小角度后接触仍不解除或物体被推动，立即否证，不得继续调角度。

## 5. 最小串行验收方案

- **D1 诊断冷启动**：当前配置 + 第 2 节日志补丁，完整 `physical_contact` 一次，只记录不修行为。用逐样本 f2j1 序列、effort、contact pairs 判定 f2 是目标接触楔住还是桌面/自碰。
- **F1 单一候选修复（最可能）**：只改 RELEASE 轨迹时序，最终目标仍全是 OPEN=.20。先让 f1j2/f3j2 张开，f2j1 保持 CLOSE 到后半段再张（8 s 内三段式）。依据是第二次失败已把故障孤立到 f2j1，先释放 f1/f3 可让物体在桌面沉降并解除 f2 侧楔入。可证伪预测：f2j1 首次且持续达到 .20；所有 target↔finger 接触清零；物体释放/撤手位移 ≤.015 m。
- **F1 前置门**：空载 3 轮 RELEASE/GRASP/RELEASE 9/9 通过后才进入物理运行。
- **E1、E2 最终验收**：F1 后两次独立冷启动完整 `physical_contact`。两次都必须通过全部静态门、抬升/保持/放回门和四个释放门；两次之间不得改任何参数。
- 任一最终 run 失败即回到诊断，不做大范围参数扫描，也不给最终 PASS。

## Codex review

PENDING

## Decision and applied diff

PENDING
