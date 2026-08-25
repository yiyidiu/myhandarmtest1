# grasp_size_pose_patch_adversarial_review

- Dialogue ID: `20260815T124518903379Z_grasp_size_pose_patch_adversarial_review`
- Session: `session-03e7dfc9-421b-41f0-b10f-6b530091dc11`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

你上一轮已在隔离 worktree 提交了抓取尺寸/位姿诊断补丁。Codex 对抗审查发现以下问题，请只做公开技术复核，不调用工具、不修改文件；给出精确修订建议和测试断言，供 Codex 独立实现：

1. grasp_geometry.coarse_geometry_candidates 当前把配置中出现 auto 理解成“任何显式 family 请求都生成全部 family”。正确语义应是：配置列表限定允许的显式 family；调用 family=auto 时才展开所有允许的 top_down/top_oblique/side；调用 family=top_down 时只能生成 top_down。请确认并给伪代码/测试。
2. size/position control launch 当前一启动就运行完整 three_finger_pick_place_demo，绕过 plan-only -> contact-only -> full 的分阶段物理验收。请给最小安全改法，优先复用现有三个 stage launch，而不是复制状态机。
3. runtime 的 lift_distance_along_opposite_approach_m=0.060 与 minimum_object_lift_m=0.060 相等，任何真实跟随误差都会导致近乎必失败。请给建议的请求抬升、最低实际 world-z 抬升、world-z 占比和稳定保持时间，并说明为何。
4. position control 的 xy=[0.28,0] 没有 MoveIt IK 证据。是否应只保留为 PLAN_ONLY 假设，禁止直接接触/完整抓取？请给验收矩阵。
5. 当前 50x60x100 物体同位置的历史“成功”实际仅抬升 21.2~21.7 mm，横向约20 mm、物体滚转，并且图片显示一根手指未形成稳定包络。请明确它应被判失败，并列出新候选的硬几何门槛。
6. 70x80x140 大物体的惯量数值已核对正确；但请判断是先验证“同位置大物体 top-down”，还是同时改位置。要求单变量诊断，不能同时改尺寸和位置。

输出：A. 问题裁定；B. 最小补丁建议；C. 必须新增的纯Python断言；D. Codex应按什么顺序跑 plan-only、contact-only、full Gazebo。不要给最终PASS结论。

## DeepSeek → Codex

## A. 问题裁定

1. **成立，family 展开语义错误。** 配置中的 `auto` 应只表示“请求 `auto` 时展开所有已配置 family”；显式 `family=top_down` 必须只生成 top_down。当前逻辑因 `auto` 存在于默认配置中，导致所有显式请求都退化为全 family。

2. **成立，控制 launch 越级。** size/position 两个控制 launch 直接启动 `three_finger_pick_place_demo.py`，绕过了 plan-only → contact-only → full 的三阶段门，必须先改。

3. **成立，0.060=0.060 是必败设计。** 任何 arm 跟踪误差、接触滑移或测量残差都会使 `object_lift_m < minimum_object_lift_m`。请求距离必须大于最低实际抬升。

4. **成立，`xy=[0.28,0]` 只是假设。** 没有 IK/连续 approach 证据前，position control 只能作为 PLAN_ONLY 探针；禁止直接 contact/full。

5. **成立，历史 21.2–21.7 mm“成功”必须改判 FAIL。** 其实测横向约 20 mm、滚转约 17–18°、tool/object 分歧 7.6/7.9 mm，是滑移/滚动，不是三指稳定包络。

6. **必须先做“同位置大物体 top-down”，后做“同尺寸仅位置”。** 任何把 70×80×140 与新 xy 同时使用的运行都不能用于归因；两个对照必须独立冷启动。

## B. 最小补丁建议

### B1. `coarse_geometry_candidates` family 语义

```python
ALLOWED = {"top_down", "top_oblique", "side"}

configured = {f for f in search_config["grasp_families"] if f != "auto"}
if not configured:
    raise ValueError("at least one explicit grasp family required")

if family == "auto":
    requested = set(configured)
else:
    if family not in configured:
        raise ValueError("family not enabled")
    requested = {family}

# 只按 requested 生成 directions
```

默认配置 `[top_down, top_oblique, side, auto]` 下：`auto` 展开三类；`top_down` 只生成 top_down。控制配置 `[top_down]` 下：`auto` 也只生成 top_down；请求 `top_oblique` 直接失败。

### B2. 控制 launch 最小安全化

两个 control launch 改为 **stage-selector wrapper**，不再自带 pick/place node，也不重复 `simulation_baseline`：

```xml
<arg name="stage" default="plan_only"/>

<include if="$(eval stage == 'plan_only')"
         file="$(find handarm_sim_demo)/launch/three_finger_grasp_pose_demo.launch">
  <!-- 传入各自的 scene/geometry/world/expected_models/grasp_family -->
</include>

<include if="$(eval stage == 'contact_only')"
         file="$(find handarm_sim_demo)/launch/three_finger_grasp_contact_demo.launch">
  ...
</include>

<include if="$(eval stage == 'full')"
         file="$(find handarm_sim_demo)/launch/three_finger_pick_place_demo.launch">
  ...
</include>
```

`stage=plan_only` 为默认值。只有前序阶段独立 PASS 并留下 run_id 后，才允许人工启动下一 stage。禁止一个进程内从 plan 自动推进到 contact 或 full。

### B3. 抬升参数

建议写入 `runtime_acceptance`：

- `lift_distance_along_opposite_approach_m: 0.085`
- `minimum_object_lift_m: 0.060`
- `minimum_lift_world_z_fraction: 0.95`
- `minimum_actual_object_lift_z_fraction: 0.90`
- `physical_hold_duration_s: 5.0`

理由：

- 命令 85 mm、实际门槛 60 mm，给 arm 跟踪、接触滑移和落差值留 25 mm 余量；
- 计划方向世界 z 分量 0.95，只接受近乎竖直抬升；实际物体位移 z 分量 0.90，允许轻微测量/接触噪声但禁止横向拖动；
- 5 s 悬空门排除瞬时“抖起来”的 20 mm 假成功。

### B4. position control 阶段锁

- 当前只允许 `stage=plan_only`；
- plan-only JSON 必须同时满足：
  - `success=true`；
  - `selected_quality.passed=true`；
  - `moveit.ik_success=true`；
  - `moveit.approach_ik_samples==9`；
  - 无 `APPROACH_IK_BRANCH_JUMP`；
  - `selected_candidate.family=="top_down"`。
- 仅上述全过后，才解锁 contact-only；contact-only 3 次独立冷启动全过后，才解锁 full。position control 在获得 IK 证据前不得接触物体。

### B5. 历史结果改判与新几何门槛

历史 21.2–21.7 mm 结果按新规则必须标 `FAIL`，原因：实际 z 抬升 <60 mm、z 分量 ≈0.73 <0.90、横向位移约 20 mm、滚转角过大。

新候选的硬几何门：

- `|object_center_axial_offset| ≤ 0.025 m`；
- 三指 closure spread ≤ 0.20；
- 接触三角面积 ≥ 10 cm²；
- 归一化接触面积 ≥ 30% 物体横截面；
- 每个接触点高度比 ≤ 0.70；
- 计划抬升 world-z 分量 ≥ 0.95；
- 实际物体抬升 world-z 分量 ≥ 0.90；
- 实际 world-z 抬升 ≥ 60 mm；
- 稳定悬空 ≥ 5 s。

## C. 必须新增的纯 Python 断言

1. **family 语义测试**
   - 默认配置 + `family="top_down"`：生成的全部 candidate family 都是 `top_down`；
   - 默认配置 + `family="auto"`：candidate family 集合为 `{top_down, top_oblique, side}`；
   - 控制配置 `grasp_families=[top_down]` + `family="auto"`：只生成 `top_down`；
   - 控制配置 + `family="top_oblique"`：抛 `ValueError`。

2. **quality gate 历史反例测试**
   - `make_candidate(..., top_oblique, roll=268, tilt=-30, axial=0.052)`：
     - `enclosure.valid` 可为 true；
     - `quality.passed` 必须 false；
     - failures 同时含 `AXIAL_OFFSET_TOO_LARGE` 和 `CONTACT_TOO_CLOSE_TO_OBJECT_EDGE`。

3. **runtime lift 证据纯函数**
   建议新增 `evaluate_lift_evidence(object_delta, config)`，断言：
   - 历史 `[-0.0203, -0.0007, +0.0217]` → FAIL；
   - `[0.0, 0.0, +0.060]` → 抬升 PASS 但 z-fraction 边界按 `>=` 单独明确；
   - `[0.0, 0.0, +0.065]` → PASS。

4. **launch 阶段测试**
   - 两个 control launch 默认 `stage=plan_only`；
   - 文本中不得包含 `type="three_finger_pick_place_demo.py"` 的无条件启动；
   - `stage=full` 分支才引用 pick/place launch；
   - `stage=plan_only/contact_only/full` 三个 include 互斥。

5. **world/scene 一致性测试保持**
   - size control：`size=[0.07,0.08,0.14]`、pose `[0.34,0.18,0.44]`、质量 0.10 kg、三轴惯量与 `m/12` 盒体公式一致；
   - position control：`size=[0.05,0.06,0.10]`、pose `[0.28,0.0,0.42]`、质量与惯量不变；
   - 两个新 world 的 friction、ODE solver/constraints 与基线逐属性相等。

## D. Codex 实测顺序

全部为独立冷启动，禁止同进程 repetitions：

1. **纯 Python/YAML/XML 测试**：C 节全部通过后再进 Gazebo。
2. **原位置基线 plan-only**：原 50×60×100、xy `[0.34,0.18]`。验收不是“成功”，而是 JSON 必须显示 52 mm oblique 候选被 quality gates 拒绝；若仍被选中，则门禁未生效，停止。
3. **尺寸对照：同位置大物体 top-down**
   - `stage=plan_only`：验证 70×80×140 在 xy `[0.34,0.18]` 的 top_down IK/approach；
   - 仅 plan-only PASS 后 `stage=contact_only`，三次冷启动；
   - 仅 contact-only 3/3 PASS 后 `stage=full`，三次冷启动。
4. **位置对照：同尺寸仅位置 top-down**
   - `stage=plan_only`：验证 50×60×100 在 xy `[0.28,0]` 的 top_down IK/approach；
   - 仅 plan-only PASS 后 `stage=contact_only`，三次冷启动；
   - 仅 contact-only 3/3 PASS 后 `stage=full`，三次冷启动。
5. **归因规则**
   - 尺寸对照全程 PASS 而位置对照失败 → 尺寸因素是有效修复，但需说明大物体低密度边界；
   - 位置对照 PASS 而尺寸对照失败 → 原位置可达性/位姿是主因；
   - 两者都失败 → 单变量均不足，回到候选位姿优化，不得组合“大物体+新位置”冒充修复。
6. 任何 full run 必须满足 60 mm 实际 z 抬升、0.90 实际 z 占比、5 s 稳定悬空、三指接触连续、无 attachment，且不得用历史 20 mm 结果作为成功基线。

## Codex review

PENDING

## Decision and applied diff

PENDING
