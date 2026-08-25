# hand_startup_and_three_distal_stability

- Dialogue ID: `20260814T145709172638Z_hand_startup_and_three_distal_stability`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

当前只审查 ABB IRB120 三指手 Gazebo 空载开合稳定性。请返回可直接审查的 unified diff 初稿和测试方案，不要调用工具。

用户约束：
- 公共命令只有 GRASP / RELEASE。
- GRASP 不得改变手掌构型；配置关节 f1j1 在 OPEN/CLOSE 都固定为 0.051 rad。
- 只有 f1j2、f2j1、f3j2 三个主动屈曲关节从 0.20 到 1.30 rad。
- 被动远端 f1j3<-f1j2、f2j2<-f2j1、f3j3<-f3j2，倍率 +1。
- 不得用 SetPosition 瞬移，不得靠放宽容差隐藏抖动。
- 必须三根远端关节都通过空载 RELEASE/GRASP/RELEASE 重复测试后才进入下一步。

当前实测：
1. 修改前 RELEASE 后 5 秒持续振荡：f1j3 range=0.01856 rad、f2j2 range=0.06144 rad、f3j3 range=0.01855 rad；f2j2 source f2j1 约 0.202 时仍在约 0.36~0.42；mimic relation errors 分别约 0.2259/0.1603/0.2359。
2. 已把三个被动远端 joint dynamics 调为 damping=0.20/friction=0.02；mimic PID P=1.0,I=0,D=0.25；插件 maxEffort=0.20、nearTargetEffort=0.05、nearTargetError=0.04、maxVelocity=0.8；三组配置对称。
3. Gazebo 日志确认上述参数全部加载。
4. 但新的 headless 启动在执行 hand initial trajectory [f1j1,f1j2,f2j1,f3j2]=[0.051,0.20,0.20,0.20] 时失败：FollowJointTrajectory error -5，`f1j1 goal error 1.621000`。启动前用 `/gazebo/set_model_configuration` 设同一目标并返回 success；controller 在 reset 后才启动；轨迹首点取启动 controller 后实际 joint state，末点为目标，duration=1.0s，path tolerance=3.2，goal tolerance=0.04。
5. f1j1 还通过 PID mimic 驱动 f3j1：f3j1 P=3,D=0.08,maxEffort=4.0；f1j1/f3j1 limits 均 [0,3.14]，multiplier=+1；f3j1 joint origin 相对镜像，插件 forceSign=+1。
6. 当前启动器即使 `initialize_arm_trajectory=false` 仍会初始化 hand 轨迹；这是为了 hand-only 测试。

插件 PID 核心语义：
```
angle = multiplier * leader.Position(0) + offset
error = mimic.Position(0) - angle
effort = pid.Update(error, dt)
effort = velocity/near-target/max-effort 限制后
mimic_joint->SetForce(0, forceSign * effort)
```

请重点判断：
1. f1j1 的 1.621 rad 启动失败更可能是 f1j1/f3j1 镜像耦合、碰撞、控制器默认命令竞态，还是启动轨迹首点/目标问题？给出可证伪的诊断顺序，不要建议直接放宽 tolerance。
2. 是否应在 hand-only 场景增加 initialize_hand_trajectory 开关，先仅由 set_model_configuration 初始化后采样稳定状态，再由 hand_commander 发 RELEASE；这种做法有什么风险？
3. 三个远端关节的 PID/阻尼策略如何以最小改动调稳？要求保持有限力 PID，不用 SetPosition。给出参数搜索顺序与验收阈值。
4. 给出最小 unified diff 初稿，优先修改启动器/launch 及自动测试脚本，加入三根远端关节逐一的 relation error、5秒 range、最大速度、f1j1 构型保持检查。若证据不足，不要猜 URDF 力方向。
5. 说明哪些建议必须拒绝或延后，避免把启动问题与 distal PID 调参混成一个不可归因的改动。

输出格式：SUMMARY、EVIDENCE-BASED DIAGNOSIS、PATCH、TESTS、RISKS、NOT RUN。

## DeepSeek → Codex

# 空载开合稳定性只读审查

## SUMMARY

启动失败（f1j1 goal error 1.621）与远端三关节调参是**两个不同的问题，必须分开归因**。数字 1.621 与"f1j1 实际停在 0.051+1.621≈1.672 rad、1.0 s 内几乎未动"完全自洽：轨迹首点与真实状态在创建时就已不一致。最可疑的因果链是 **set_model_configuration 成功 → 之后的 reset 把关节重新瞬移回 SDF 初始位（≈1.67）→ controller 启动后采样首点=1.67 → 轨迹要求 1.67→0.051 在 1 s 内完成（≈1.62 rad/s，几乎必然超速度/加速度限）→ path tolerance 3.2 rad 屏蔽所有中间检查 → 只有 goal 检查报 -5**。镜像耦合、碰撞、命令竞态都要按序证伪，但都不该优先于这个链。远端 PID 策略本身方向正确（有限力、有阻尼），但**当前参数加载后尚无任何实测数据**——先修启动、再测基线、再谈调参。任何放宽 goal/path tolerance 的建议一律拒绝。

## EVIDENCE-BASED DIAGNOSIS

**Q1：1.621 rad 启动失败的归因与可证伪诊断顺序**

先明确证据缺口：你只报了 f1j1 的 goal error，**其余三个关节（f1j2/f2j1/f3j2）当时的 goal error 是区分"全局启动问题"和"f1j1 特有问题"的最关键单一数据**。若三者都达标而只有 f1j1 失败，则矛头直指 f1j1 特有路径（它是唯一有 mimic follower 的 leader）；若四者全挂，则是重置/命令时序问题。

按可证伪顺序执行，每步产生书面结论，**任何一步都不许靠放宽容差"通过"**：

1. **补采全关节 goal error**（-5 事件时刻）：f1j2/f2j1/f3j2 是否在 0.04 内。→ 区分全局 vs 单关节。
2. **位置时间序列取证**：≥100 Hz 记录 f1j1（含 f3j1）从 set_model_configuration 之前到 goal 时间的位置。判读点：1.67 这个值**在哪一刻出现**——在 reset 边界跳变 ⇒ 瞬移/重置语义；在 controller 运行期间漂移 ⇒ 命令或负载问题。这一步直接证伪"重置把 0.051 冲掉"。
3. **轨迹消息与关节清单取证**：dump 实际下发的 trajectory（首点、末点、duration）与 controller 声明的 joints 列表；计算所需平均速度 |goal−first|/1.0s，与 f1j1 URDF velocity limit 比较。若 1.62 rad/s ≫ limit，则因果链闭合。
4. **镜像耦合隔离（仅诊断，测完恢复）**：临时移除 f3j1 的 mimic 对，重复同一启动序列。f1j1 若达标 ⇒ 镜像环（P=3、maxEffort=4.0 的反应力矩）参与；若仍挂 ⇒ 排除。注意：镜像环把 f1j1 踢出 1.6 rad 需要极大力矩，作为主因不太可信，但必须被排除。
5. **碰撞排除**：失败窗口内查 self-collision/contact 话题。
6. 上述全部有结论后，才允许**单因素**改动，且每次改动前先立可证伪假设。

**Q2：是否加 initialize_hand_trajectory 开关**

是，且是正确的最小改动，但它是**启动时序修复的载体**，不是掩盖 Q1 的手段。要点：

- 顺序必须显式固定：**先 reset（若需要）→ 再 set_model_configuration → 读回 /joint_states 验证 → 再启动 controller → 采样首点 → 发 RELEASE**。当前失败正是"teleport 被后续 reset 冲掉、首点采到 1.67"的经典形态。
- 读回验证 fail-closed：teleport 后 |q−target| > ε（候选 0.01 rad，待标定）立即拒绝进入下一步并留痕，**永远不信任服务返回的 success**。
- 采样前须有 settle 判据（速度 < 阈值持续 Ts，超时 fail-closed），否则瞬态被烤进首点，复发同类 bug。
- 风险 1：开关若默认 false 会改变整机行为——**默认 true，仅 hand-only 显式置 false**。
- 风险 2：把启动修复与 distal 调参混在一个 commit/一轮实验里 → 归因不可分（见 Q5）。
- 风险 3：即使启动修好，f3j1 镜像环若真有反力矩问题，会立刻在随后的 RELEASE 中复现——开关不能成为绕过它的藏身处。

**Q3：远端三关节最小调稳策略**

前提纪律：**当前参数（damping 0.20/friction 0.02/P=1.0/I=0/D=0.25/maxEffort 0.20/nearTarget 0.05/0.04/maxVelocity 0.8）只是"已加载"，不是"已通过"**；修好启动后先跑基线再谈搜索。

搜索顺序（每次只动一个因素，其余冻结）：

1. **基线复测**：当前参数下 RELEASE/GRASP/RELEASE，记录每指 relation error、5 s range、max |velocity|。达标即停。
2. 若仍有振荡且 D 贡献抖动（Gazebo 步长不匀时 D 是常见激励源）：D 粗降 0.25→0.10→0.05，P 不动。
3. 若稳态 relation error 不达标：P 小步升 1.0→1.5→2.0，每步核对 5 s range 不升、力矩直方图远低于 maxEffort。
4. **只有观察到"effort 顶格在 maxEffort 且 error 非零"的饱和签名时**，才允许讨论 maxEffort 大小——这是执行器容量问题，不是容差问题；必须带签名证据与日志，不得静默调。
5. nearTargetError/nearTargetEffort 的过渡带最后调；maxVelocity 0.8 是全局钳位，**禁止调大来掩盖抖动**（等同放松容差）。
6. 三组对称是约束；若 f2j2 仍最差，先查物理不对称（杆长/惯量/轴线朝向）留档，再谈单关节不对称增益，且不对称必须绑定物理原因，不得用作抖动遮罩。

验收阈值（**候选起点，开跑前由 Codex 写死，跑后不得再改**）：稳态 relation error ≤0.01 rad；RELEASE 后 5 s 窗末 3 s range ≤0.005 rad（比修前 f1j3 0.01855 严格约 4 倍、比 f2j2 0.06144 严格约 12 倍）；max |velocity| ≤0.05 rad/s；f1j1 保持 |f1j1−0.051| ≤0.005 rad 且速度有界；连续 ≥5 轮 RELEASE/GRASP/RELEASE 全过。

## PATCH

以下为 unified diff 初稿（路径/标识符为示意，须由 Codex 映射到实际仓库；不猜 URDF 力方向）：

```diff
diff --git a/launch/hand_only_test.launch b/launch/hand_only_test.launch
--- a/launch/hand_only_test.launch
+++ b/launch/hand_only_test.launch
@@
 <launch>
   <arg name="initialize_arm_trajectory" default="true"/>
+  <!-- Hand-only runs skip the hand init trajectory. State is teleported
+       via /gazebo/set_model_configuration AFTER any reset and BEFORE
+       controller start, then read back and verified. Default true
+       preserves full-system behavior. -->
+  <arg name="initialize_hand_trajectory" default="true"/>
   ...
   <node name="startup_commander" pkg="..." type="...">
     <param name="initialize_arm_trajectory"
            value="$(arg initialize_arm_trajectory)"/>
+    <param name="initialize_hand_trajectory"
+           value="$(arg initialize_hand_trajectory)"/>
   </node>
 </launch>
```

```diff
diff --git a/scripts/startup_commander.py b/scripts/startup_commander.py
--- a/scripts/startup_commander.py
+++ b/scripts/startup_commander.py
@@
 INIT_READBACK_EPS = 0.01   # TO_BE_CALIBRATED, fixed before runs
+
+def teleport_and_verify(model, target):
+    # Order is the fix: teleport AFTER any reset, BEFORE controller start.
+    set_model_configuration(model, target)
+    q = sample_joint_states(timeout=2.0)   # fail-closed on timeout
+    err = max(abs(q[j] - target[j]) for j in target)
+    if err > INIT_READBACK_EPS:
+        fail_closed("post-teleport readback err=%.4f" % err)
+    return q
+
 def run_hand_init_trajectory(...):
+    if not rospy.get_param("~initialize_hand_trajectory", True):
+        teleport_and_verify(HAND_MODEL, HAND_OPEN_TARGET)  # [0.051,0.20,0.20,0.20]
+        return
     # existing trajectory path unchanged, but ADD before publishing:
     first = sample_joint_states()
+    if max(abs(first[j] - target[j]) for j in target) > INIT_READBACK_EPS:
+        fail_closed("first-point/target mismatch; refusing bogus trajectory")
```

```diff
diff --git a/scripts/hand_no_load_cycle_test.py b/scripts/hand_no_load_cycle_test.py
new file mode 100755
--- /dev/null
+++ b/scripts/hand_no_load_cycle_test.py
@@
+#!/usr/bin/env python3
+# No-load RELEASE/GRASP/RELEASE stability gate for the three distal followers.
+# All thresholds are TO_BE_CALIBRATED and must be fixed BEFORE data collection.
+REL_ERR_STEADY  = 0.01    # rad, max |mimic - (mult*leader + offset)|
+RANGE_5S        = 0.005   # rad, range in last 3 s of 5 s post-RELEASE window
+MAX_VEL         = 0.05    # rad/s, candidate; prefer /joint_states velocity field
+F1J1_HOLD       = 0.005   # rad, |f1j1 - 0.051| during every cycle
+CYCLES          = 5
+
+FOLLOWERS = {("f1j3","f1j2"): 1.0, ("f2j2","f2j1"): 1.0, ("f3j3","f3j2"): 1.0}
+
+def check_cycle(rec):
+    fails = []
+    for (mimic, leader), mult in FOLLOWERS.items():
+        rel = np.abs(rec[mimic] - mult*rec[leader])          # offset=0
+        if rel[-3.0:].max() > REL_ERR_STEADY: fails.append((mimic,"rel"))
+        if np.ptp(rec[mimic][-3.0:]) > RANGE_5S:    fails.append((mimic,"range"))
+        if abs(rec.vel[mimic]).max() > MAX_VEL:     fails.append((mimic,"vel"))
+    if abs(rec["f1j1"] - 0.051).max() > F1J1_HOLD: fails.append(("f1j1","hold"))
+    return fails
+
+# protocol: for c in range(CYCLES): RELEASE -> settle 5 s -> GRASP -> settle
+#           -> RELEASE -> settle 5 s -> check_cycle -> write JSON -> nonzero exit on fail
```

## TESTS

- **T1 启动回归（Q1 的前置）**：`initialize_hand_trajectory=false` 下 teleport→读回验证→controller 启动→RELEASE，连续 5 次零 -5、零 readback 失败；留全关节 goal error 日志。
- **T2 归因测试（Q1 的 1–5 步）**：位置时间序列、轨迹 dump、URDF 速度限比对、f3j1 mimic 临时隔离、碰撞话题——每步书面结论，先归因、后修改。
- **T3 远端稳定性门槛（Q3）**：≥5 轮 RELEASE/GRASP/RELEASE，每轮对 f1j3/f2j2/f3j3 计算 steady relation error、5 s range、max velocity，并查 f1j1 保持；全指全轮通过才放行下一阶段。
- **T4 整机回归**：`initialize_hand_trajectory=true`（默认）下整机行为不变。

## RISKS

- path tolerance=3.2 rad 实际**禁用了所有中间点检查**，任何偏差都只会在 goal 处以 -5 爆出——这正是本次失败的表象来源；须记录 3.2 的理由，不在未归因前改动它。
- 启动修复与 distal 调参**同一 commit/同一轮实验 = 不可归因**，必须分两次独立验证。
- 首点采样若发生在 settle 前，瞬态被烤进轨迹，同类 bug 复发；必须有 settle 判据 + 超时。
- f3j1 的 multiplier/forceSign 在镜像几何下的正确性**证据不足**（你已提示勿猜 URDF 力方向）；在拿到误差增长方向与反力矩方向的实测前，禁止翻号——翻号可能用一个反向常力矩掩盖问题。
- 速度从位置差分估计噪声大；优先用 /joint_states 的 velocity 字段，缺则该字段记为验证器输入缺口。
- 读回验证的 0.01 rad 与全部验收阈值目前**未经实测校准**，只能在开跑前写死、跑后禁用事后调整。

## NOT RUN

未调用任何工具、未读写文件、未执行任何测试。所有诊断与阈值均为基于题面公开事实的审查意见；diff 为示意初稿，路径/标识符待 Codex 映射；1.621 归因链条必须经 T2 实测证伪后才能作为结论引用。

## Codex review

PENDING

## Decision and applied diff

PENDING
