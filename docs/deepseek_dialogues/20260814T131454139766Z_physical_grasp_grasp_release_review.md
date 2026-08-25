# physical_grasp_grasp_release_review

- Dialogue ID: `20260814T131454139766Z_physical_grasp_grasp_release_review`
- Session: `session-70fb820e-a6e1-4e34-bcf4-bf795d5628ac`
- Model: `deepseek-official/deepseek-v4-pro`
- Tool events: `NONE`

## Codex → DeepSeek

You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.

只读审查任务：当前Gazebo仿真中的ABB IRB120三指手已实测只需两个上层命令GRASP与RELEASE，禁止构型切换。默认物理world无固定关节附着插件，目标是仅靠接触、摩擦、重力完成接近、GRASP、抬升、保持、RELEASE、下落。真实手只有根部应变片且已经部署，本仿真不得设计或依赖应变反馈；Gazebo contact只能作为独立验收观测，不能闭环控制。请审查并给出最小状态机、每步fail-closed条件、必须做的四个负例、物理接触/物体随手/释放下落的机器验收字段，以及最容易产生假通过的实现错误。不要修改任何文件。

## DeepSeek → Codex

# Stage B 最终边界设计审查（仅 GRASP/RELEASE，接触仅作验收观测）

## SUMMARY

边界成立且可审计：上层仅 GRASP/RELEASE 两命令、无构型切换；控制回路只允许读关节状态与内部状态机；Gazebo 接触数据单向流入独立验收验证器，绝不回流控制。此设计下"抓取成功"的全部证明力来自验证器证据链，因此最小状态机必须把**每一次状态迁移都绑定可判假的物理不变量**，而最常见的假通过来源是接触话题泄漏进控制器、以及把"物体恰好悬停"当成"被接触力握持"。

## 1. 最小状态机（无构型切换）

状态仅七个，迁移只由内部条件/单次上层命令驱动：

```
IDLE → APPROACH(预置位，指开) → GRASPING(增量闭合)
     → LIFTING(抬升) → HOLDING(保持) → RELEASING(张开) → DROPPED(确认下落)
```

- APPROACH 与 LIFTING 由手臂执行；GRASPING 内允许有界增量关节子步（数值稳定性），但对外仍是一次 GRASP；RELEASE 同理一次命令完成 RELEASING。
- 任何状态失败 → FAILED，动作按序为：冻结手指目标 → 手指有界增量张开 → 手臂停止 → 全量留痕。禁止"重试缝合"跳过状态。

## 2. 每步 fail-closed 条件

- **IDLE→APPROACH**：目标物体存在且 non-static、手无任何接触报告、手/臂关节态无 NaN/Inf。失败：物体缺失/属性异常、起始即有接触。
- **GRASPING**：每子步限幅内；手指力矩有限且未饱和-零误差异常（teleport 签名）；闭合期间物体相对初始位姿位移/转动未超漂移阈（闭合推挤签名）；接触建立后手指目标冻结；指-指或指-桌面接触一律无效。失败：任一遥测超龄、力矩饱和+误差≈0 持续超时、推挤超阈、闭合步数超预算、单指接触后长时间无第二指。
- **LIFTING**：抬升前 ≥2 不同手指稳定接触（验收侧判定）、物体速度≈0；抬升中接触不断、无物体↔掌/腕/臂接触、位移与命令方向一致。失败：接触丢失、出现非手指接触、物体运动方向异常。
- **HOLDING**：≥2 s 内接触持续、相对漂移 ≤ 预设阈值。失败：任何接触断帧后物体仍悬空（浮空签名）。
- **RELEASING**：手指有界张开，物体速度在合理延迟内出现向下分量。失败：超时仍未下落、或接触已清零而物体悬停。
- **DROPPED**：确认自由下落（加速度与 g 一致、无向上速度尖峰）后终态。失败：回弹/粘滞异常。
- **全程**：物理世界时间戳单调有效、物体模型属性（mass/inertia/static 位）不变。

## 3. 必须做的四个负例（统计前硬门禁）

1. **张手抬升**：手指全开，手自物体上方下降至近距不接触再抬起 → 物体不得被吸附/提升。
2. **无接触横移**：手在物体旁横向移动，全程无接触 → 物体位移/转动必须为零（或低于标定噪声）。
3. **打开掉落**：一次完整握持后 RELEASE → 物体必须自由下落，时程与自由落体一致。
4. **单指不得成功**：仅一指闭合（另一指保持张开的初始条件），执行完整 GRASP→LIFT 流程 → 必须失败且**抬升量 < 0.08 m**。

四项全部通过才开放统计；任一失败即回炉参数/求解器标定，不得以"记录不阻断"通过。

## 4. 机器验收字段（验证器落盘，逐项布尔化）

**物理接触**：
- `contact_events[]`：原始碰撞对（物体名过滤后）、每指独立时间戳、持续时长、法向力幅值；
- `contact_count_per_finger`、`stable_contact_mask`（≥2 不同手指且持续 ≥ 阈值）；
- `non_finger_contact_events[]`（必须为空）。

**物体随手**：
- `object_pose_series`（世界系连续采样，含抬升前后）、`lift_height_m`（≥0.08 的判据值）、`drift_m/rad`（相对手掌系）、`hold_duration_s`（≥2.0）；
- `contact_during_lift/hold`（全程为真）。

**释放下落**：
- `release_time`、`contact_cleared_time`、`fall_start_time`；
- `accel_z_during_fall`（与 g 比对容差）、`upward_velocity_spike`（必须为无）；
- `final_rest_state`（落回支撑面/静止）。

**身份与附着审计**：`object_static_flag`、`world_fixed_joint_list`（物体↔手/臂任何链接必须为空）、`attachment_plugin_active=false`、`solver_params_hash`。

## 5. 最容易产生假通过的实现错误

1. **接触数据泄漏进控制器**：GRASP 内部读 contact 话题做"接触即停"（看似合理实则形成闭环，且难以与"独立验收"区分）——必须静态检查订阅列表为空的证据。
2. **浮空假通过**：接触丢失但物体因手下方几何/壁面支撑或求解器粘滞而悬停，HOLDING 只看时间不查接触 → 必须"接触断帧 + 仍悬空 = 立即失败"。
3. **SetPosition 残留**：物理 world 中 mimic 仍走无 PID 分支或 maxEffort 失效，表现为"力矩饱和且位置误差≈0"仍稳握 —— 须有 teleport 签名判据。
4. **固定/附着残留**：旧插件虽默认不连但被依赖链加载，物体随手动 → 必须由验证器读世界状态自证无 fixed joint。
5. **粘滞/伪摩擦假稳定**：求解器 kc/kd/min_depth 未标定，轻盒被"粘"在指尖上，负例 2/4 会被误通过 → 必须先用能量注入/无接触探针标定。
6. **抬升证据取命令值**：以"手臂命令抬了 0.08 m"代替"物体位姿实测抬了 0.08 m"——验收字段必须全部来自物理输出轨迹。
7. **接触话题假阳性**：单帧/瞬态接触被计入稳定接触；或物体名过滤失效把桌面/他指接触计入 → 必须时长门 + 碰撞对名称过滤 + 归属一致性审计。
8. **应变片误入**：任何对根部应变片的仿真、订阅或文档化需求都会违反边界，使验收依赖不存在的硬件能力 → 以"反馈路径为空"作为审计项。

## PATCH

PATCH: NONE —— 本轮仅审查。

## NOT_RUN

未调用任何工具、未读写文件；全部为基于题面公开事实的设计意见，须由 Codex 独立验证。

## Codex review

PENDING

## Decision and applied diff

PENDING
