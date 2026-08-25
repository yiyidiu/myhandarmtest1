# ABB IRB120—三指手—D455/HaMeR 工程交接总报告

生成时间：2026-08-15（Asia/Shanghai）  
工作区：`/home/diu/myhandarmtest1`  
用途：把本文件和同名 ZIP 直接发给新的 GPT/Codex。**本文件是当前状态的权威入口；若旧报告与本文件冲突，以本文件、当前源码和最新结果 JSON 为准。**

## 0. 给接手 GPT 的一句话结论

工程已经完成 HaMeR crop-only、D455 USB2 开发录制、ROI/HaMeR 实时实验、独立 RGB-D KLT–Kabsch 原型、Gazebo 静态避障、三指手空载稳定性，以及三次“机械臂接近物体并通过真实 Gazebo 接触闭合”的重复基线。

但用户最终需要的 **“手指物理夹住物体 → 抬起 → 放回桌面 → 松开 → 手臂撤离”仍为 NOT PASS**。目前物体确实由接触/摩擦抬起过，不是瞬移或固定关节；失败集中在放回位置偏差和接触负载下的松手。最新的“先打开 f2、再打开 f1/f3”松手顺序还没有在完整链路中真正执行到，因此不能宣称已解决。

## 1. 最终用户目标与不可违反的边界

当前唯一开发主线是 Gazebo 中的 ABB IRB120 + 三指手物理抓取、抬升、放回。用户的明确要求如下：

- `GRASP` 只改变三路手指屈伸；不得移动机械臂，不得夹带掌型变换。
- `RELEASE` 只张开手指；不得移动机械臂，不得夹带掌型变换。
- 掌型关节 `f1j1` 在 OPEN/CLOSE 中保持同一目标。
- 如果以后需要掌型变换，必须是独立命令，并与开合按顺序执行；当前公开接口只保留 `GRASP/RELEASE`。
- 物体必须由手指接触与摩擦抓起。禁止 fixed-joint attachment、瞬移物体、隐藏支架/导轨、关闭碰撞或让物体自行升起。
- Gazebo 接触传感器只能作为验收观测，不反向驱动抓握。实体手只有指根应变片，实体侧反馈已部署，本仿真不实现触觉/应变片闭环。
- 每个可视化增量都应给用户一条可复现命令；同一时刻只能启动一套顶层 roslaunch。
- 不自动进入真实机器人、MoveIt Servo、视觉遥操作或避障新开发分支。

## 2. 当前阶段总表

| 模块 | 已实现/实测状态 | 当前裁定 |
|---|---|---|
| P0 原工程审计 | 硬件、ROS/Gazebo、URDF、控制器和旧代码风险已审计；历史 P0 安全门禁并非正式通过 | 基线已记录；旧遥操作安全能力 NOT PASS |
| P1 HaMeR crop-only | 真实 checkpoint/config/mean/MANO 加载，FP16/FP32、RTX 2060、Gazebo headless 并存实测 | PASS（只代表推理与资源） |
| P2 D455 录制 | USB2.1 下 640×480@30，300 帧 RGB/raw depth/aligned depth、时间戳、异步写盘、离线校验与故障注入 | USB2 算法开发可用；P2 已冻结 |
| P3/P4 ROI + HaMeR 掌姿态 | 三组真实 D455+HaMeR 数据和 A/B/C 稳定性评价完成 | 实验完成，但三种 HaMeR 姿态全部不可控臂 |
| P5 独立 RGB-D KLT–Kabsch | 实现、单测和真实归档离线回放完成 | 未达到连续姿态最低条件；不接 MoveIt |
| Gazebo 启动/控制器 | 单一顶层 launch、启动协调、同名节点冲突规避、控制器/场景等待已实现 | PASS（当前仿真基线） |
| 静态避障 | no/single/double obstacle 与安全失败场景实测；多角度 10 次 headless、2 次 GUI | PASS（仅静态已知场景） |
| 三指手空载稳定性 | 凸碰撞代理、有限 PID/effort mimic、ODE 参数、后步诊断；多次冷启动循环 | PASS（不等于抓物） |
| 物理接触闭合 | 三次冷启动 `physical_grasp_only` 成功，物体闭合位移均 <10 mm | 局部基线 PASS |
| 完整物理抓取—抬升—放回—松开—撤离 | 物体实际随手抬升 36–44 mm；未使用 attachment；完整运行均失败 | **NOT PASS** |
| 真实 ABB/真实手 | 没有连接或下发命令 | NOT RUN |

## 3. 环境和硬件

- Ubuntu 20.04.6、ROS Noetic 1.17.4、Gazebo Classic 11.15.1、MoveIt 1。
- GPU：NVIDIA GeForce RTX 2060 6144 MiB，driver 570.133.20，compute capability 7.5。
- HaMeR 环境：conda `hamer_rtx2060`，Python 3.10.20，PyTorch 2.2.0+cu118，cuDNN 8.7。
- D455：serial `234322305987`，firmware `5.17.3.10`，librealsense `2.58.2.10647`。
- D455 当前通过 USB 2.1 / 480M 枚举，profile 为 RGB8 + Z16 640×480@30，depth align 到 color，depth scale `0.0010000000475 m/unit`。
- 当前没有 ROS/Gazebo/MoveIt 残留进程。
- 工作区根目录不是 Git 仓库；`src/roboticsgroup_upatras_gazebo_plugins/.git` 是历史嵌套仓库，不应随最终交付上传。

统一限制说明：

> 当前D455使用USB 2.1，本阶段结果用于算法开发。最终实时性能、正式数据集和长时间稳定性以后在USB3条件下重新测试。

## 4. 视觉与相对姿态链的完成情况

### 4.1 HaMeR crop-only

实际加载的资产：checkpoint、`model_config.yaml`、`mano_mean_params.npz`、`MANO_RIGHT.pkl`。真实输出包括：

- vertices `(778,3)`；
- joints `(21,3)`；
- `global_orient (3,3)`；
- `hand_pose (15,3,3)`；
- `betas (10,)`。

30 帧 batch=1 基准：

| 场景 | 精度 | 平均延迟 | FPS | 系统显存峰值 | OOM |
|---|---|---:|---:|---:|---|
| HaMeR only | FP16 | 41.00 ms | 24.39 | 4852 MiB | 否 |
| HaMeR only | FP32 | 71.87 ms | 13.91 | 3491 MiB | 否 |
| HaMeR + Gazebo headless | FP16 | 40.27 ms | 24.83 | 4962 MiB | 否 |
| HaMeR + Gazebo headless | FP32 | 70.19 ms | 14.25 | 3514 MiB | 否 |

### 4.2 D455 实时 ROI 与 HaMeR 稳定性实验

已实现 manual ROI、MediaPipe 仅作手存在/2D bbox/粗左右手、KLT ROI、latest-frame overwrite、FP16 crop-only、betas 中位冻结、joint palm frame、87 rigid-palm-vertex frame 和 OpenCV overlay。

三组约 25 秒真实数据均 100% 有效，HaMeR 约 14.85–14.97 Hz，P95 推理约 49–50 ms，RTX 2060 总显存峰值 4908–4959 MiB。

关键失败结论：

- 静止姿态 P95：raw/joint/rigid 约 `5.51/5.51/5.58°`；
- 纯平移假旋转 P95：约 `157.09/157.09/157.87°`；
- 张开握拳假旋转 P95：约 `139.63/139.63/139.34°`。

异常帧审计证明 MANO 网格本身随 HaMeR root 翻转，分类为 `HAMER_ROOT_ORIENTATION_FAILURE`，不是简单转置或评价错误。因此：

**`global_orient`、MANO joint palm frame、rigid palm vertex frame 均不得进入机械臂姿态通道；HaMeR 后续只允许用于 ROI、掌区、hand_pose、gesture 和重检测。**

### 4.3 独立 RGB-D KLT–Kabsch

已实现 Shi–Tomasi、双向 KLT、双帧 aligned depth、D455 反投影、3 点 RANSAC、无尺度 Kabsch、`det(R)=+1`、退化/残差/内点门控、真实 frame id/timestamp 和 `INITIALIZING/TRACKING/FROZEN/LOST`。

真实归档离线结果：

- STATIC：有效覆盖 99.20%，累计姿态 P95 `18.579°`；
- TRANSLATION：覆盖 34.14%，假旋转 P95 `11.083°`；
- GESTURE：覆盖 84.64%，累计变化 P95 `16.240°`；
- P5_ROTATION、真实 30 Hz 四组实验：NOT RUN。

最低标准要求静止 P95 <5°、平移 P95 <10°、覆盖 >90%；当前没有达到。建议后续若恢复视觉主线，优先稠密 RGB-D 掌部配准或 point-to-plane ICP，不再调 HaMeR 姿态滤波。

## 5. Gazebo 仿真与避障完成情况

新包 `handarm_sim_demo` 统一管理单套 Gazebo、robot model、controller、MoveIt、scene 和任务节点，解决了此前重复启动导致的 `/robot_state_publisher`、`/move_group`、`/gazebo_gui` 重名、`robot already exist` 和 controller already running。

静态避障已实测：

- no obstacle 3/3；single obstacle 10/10；double obstacle 10/10；
- unreachable 和 fully blocked 均规划失败且未执行旧/空轨迹；
- 多角度 5 waypoint course：10/10 headless，2/2 GUI，所有执行轨迹点 collision-free；
- 结果只证明静态已知盒体环境，不代表动态避障或真实硬件安全。

历史 `deterministic_lift` 使用 Gazebo fixed joint 抬升 100 mm，只是非物理演示，已与当前物理接触世界隔离，不能作为抓握成功证据。

## 6. 当前三指手实现

### 6.1 命令语义

当前权威配置是 `src/handarm_sim_demo/config/hand_commands.yaml`：

```text
active joints: [f1j1, f1j2, f2j1, f3j2]
OPEN:  [0.18, 0.20, 0.20, 0.20]
CLOSE: [0.18, 0.85, 0.85, 0.90]
duration: 8.0 s
configuration joint: f1j1
flexion joints: f1j2, f2j1, f3j2
current RELEASE order: first f2j1, then f1j2/f3j2
```

`f1j1` 在 OPEN/CLOSE 都是 0.18 rad，因此抓握只做屈伸。当前公开话题只接受 `GRASP/RELEASE`。

### 6.2 已修复的抖动/散架隐患

- 手部高面数 STL 仅作视觉，动态碰撞改为凸 primitive proxy；
- 主动关节使用有限 PID；四个 mimic 使用有限 effort PID，不再逐步 `SetPosition`；
- 被动关节改为有限 revolute，统一 limit/damping/inertia；
- 远端 mimic effort 约 0.2 N·m，近目标进一步限到 0.05 N·m；
- ODE：1 ms step、1000 Hz、quick 100 iterations、`contact_max_correcting_vel=0.1`；
- 1 kHz `WorldUpdateEnd` 后步诊断，避免把服务中间态尖峰误判为持续抖动；
- 当前空载 staged RELEASE 两种顺序均做过实际测试：9/9 和 3/3 稳态窗口 PASS。

这些结果只证明空载稳定，不能外推到物体把手指顶住时的 RELEASE。

## 7. 当前物理抓取链和真实结果

物理世界 `handarm_physical_grasp.world` 中：

- 目标为动态 0.10 kg、50×60×100 mm 盒体；
- 桌面与物体启用真实碰撞和摩擦；
- 没有 attachment plugin、hidden guide、固定关节或物体控制插件；
- contact sensor 只发布 `/handarm_sim_demo/target_contacts` 作验收证据。

当前运动参数：known pose 侧向预抓取、40 mm approach、52 mm 竖直 lift，lift/place 至少 8 s；物体闭合位移上限 10 mm、最小物体抬升 30 mm、物体/工具位移差上限 10 mm、放回误差上限 15 mm、只允许一次 RELEASE。

### 7.1 已通过的局部基线

三次独立冷启动 `physical_grasp_only`：

| 结果文件 | 闭合时物体位移 | 结果 |
|---|---:|---|
| `pick_physical_grasp_only_20260815T070615306637Z.json` | 9.370 mm | PASS |
| `pick_physical_grasp_only_20260815T070702855358Z.json` | 9.285 mm | PASS |
| `pick_physical_grasp_only_20260815T070750476579Z.json` | 9.268 mm | PASS |

这只证明“机械臂接近 + 手指通过接触闭合”可重复，不证明完成抬升和释放。结果 JSON 内局部字段 `physical_grasp_claimed=true` 不能被解释为整项 pick/place PASS。

### 7.2 物体确实被物理抬起过，但完整任务失败

| 结果文件 | 物体抬升 | 跟随差 | 放回误差 | 失败原因 |
|---|---:|---:|---:|---|
| `...T070938310141Z` | 43.672 mm | 8.105 mm | 13.532 mm | RELEASE 失败，f2j1 goal error 0.065164 |
| `...T075211202712Z` | 35.999 mm | 8.009 mm | 14.955 mm | RELEASE 失败，f2j1 被物体顶住，误差 -0.060862 |
| `...T075558461688Z` | 37.590 mm | 8.809 mm | 16.446 mm | 放回误差超过 15 mm，未执行 RELEASE |
| `...T075904469863Z` | 39.217 mm | 10.681 mm | — | 物体/工具跟随差超过 10 mm |
| `...T080104015000Z` | 38.609 mm | 9.341 mm | 18.007 mm | 放回误差超过 15 mm，未执行 RELEASE |

这些运行全部 `attachment_used=false`。物体抬升是接触/摩擦造成的真实 Gazebo 动力学结果；完整记录均 `all_success=false`、最终 `physical_grasp_claimed=false`。

### 7.3 当前未解决问题

1. **放置精度有随机滑移。** 工具回位误差只有约 0.8–2.1 mm，但物体回桌误差达到 16–18 mm，说明主要误差来自夹持中的相对滑移、桌面接触/拖曳或夹持相对位姿变化，而不是单纯机械臂没有返回。
2. **旧 RELEASE 顺序在负载下卡住 f2。** 先开 f1/f3 会使 f2 保持夹紧并被盒体楔住，最终无法到 0.20 rad。
3. **新 RELEASE 顺序尚无完整物理证据。** 当前代码已改成先开 f2、再开 f1/f3；两次空载验证通过，但之后三次完整运行均在更早的跟随/放置门限停止，所以还没有执行到这条路径。
4. **撤离验收尚未实际运行。** 代码已加入恢复 target planning proxy/ACM、沿 approach 反向慢撤、撤离后物体和接触检查，但尚未有一次完整物理 run 到达该状态。
5. **接触参数不是已辨识的真实手模型。** 当前摩擦、PID、collision proxy 和 effort 是 Gazebo 稳定开发参数；空载稳定不代表负载接触稳定。

## 8. 为什么现在不能宣称实现

最终成功必须在同一个冷启动 trial 内连续满足：

1. 预抓取/接近不推动物体超限；
2. 单次同步 GRASP 建立至少两指族稳定接触；
3. 闭合阶段物体位移 ≤10 mm；
4. 物体抬升 ≥30 mm；
5. 物体/工具抬升差 ≤10 mm，保持阶段不掉落；
6. 物体回桌位置误差 ≤15 mm；
7. 恰好一次 staged RELEASE 成功、接触持续清空、物体稳定在桌上；
8. 手臂撤离后物体仍不跟随且接触保持清空；
9. 全程 `attachment_used=false`；
10. 至少三次独立冷启动重复通过。

目前没有任何一条记录同时满足 1–10，所以最终结论必须是 NOT PASS。

## 9. 建议的下一步（严格串行）

不要放宽现有 10 mm 跟随门限和 15 mm 放置门限来制造 PASS。建议按下面顺序继续：

1. **先做证据定位，不改大组参数。** 在 grasp/lift/place/settle 全程记录物体—tool 相对位姿、三指族接触状态、主动/从动关节实际值和 controller error，区分空中滑移、桌面首次接触后的拖曳和 arm 跟踪误差。
2. **把放置目标从“机械反向 lift”校准为“让物体回到初始桌面 pose”。** 只能通过机械臂轨迹补偿，不能 set_model_state、attachment 或隐藏约束；每次只改一个几何/轨迹变量。
3. **确保回桌后先卸载再 RELEASE。** 要求物体已由桌面支撑、工具/物体速度低、接触稳定，再执行当前 f2-first 的单次 staged RELEASE。
4. **首次真正执行新 RELEASE 时保留完整诊断。** 若 f2 仍卡住，不允许把第二次 RELEASE 拼接成成功；应依据 f2 接触位置、目标误差、mimic error 和接触清空时序修复。
5. **完成撤离。** RELEASE 成功后恢复碰撞场景，沿反 approach 低速撤离，验证物体没有跟手、没有穿透、没有再次接触。
6. **最后做三次冷启动重复验收。** 任一 trial 失败即整体 NOT PASS，不拼接不同运行的局部成功。

## 10. 当前可视化和测试命令

每次只启动一个顶层 launch；旧 launch 必须先在原终端 `Ctrl-C` 并等待 `done`。

### 10.1 查看完整物理抓取开发链

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch handarm_sim_demo scripted_pick_demo.launch \
  gazebo_gui:=true rviz:=true grasp_mode:=physical_contact \
  repetitions:=1 shutdown_on_task_exit:=false
```

预期：窗口保留供观察。当前不保证 DONE；日志安全停止即代表某门限真实失败，不应马上重复启动第二套。

### 10.2 只查看“接近 + 接触闭合”局部基线

```bash
roslaunch handarm_sim_demo scripted_pick_demo.launch \
  gazebo_gui:=true rviz:=true grasp_mode:=physical_grasp_only \
  repetitions:=1 shutdown_on_task_exit:=false
```

### 10.3 查看空载手指稳定性

```bash
roslaunch handarm_sim_demo hand_stability_demo.launch \
  gazebo_gui:=true cycles:=1 shutdown_on_task_exit:=false
```

### 10.4 双障碍避障

```bash
roslaunch handarm_sim_demo avoidance_demo.launch \
  gazebo_gui:=true rviz:=true scenario:=double_obstacle repetitions:=1
```

### 10.5 回归测试

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
python3 -m compileall -q \
  perception_hamer/src perception_hamer/scripts perception_hamer/tests \
  src/handarm_sim_demo/scripts src/handarm_sim_demo/test
python3 -m unittest discover -s src/handarm_sim_demo/test -p 'test_*.py' -v
python3 -m unittest discover -s perception_hamer/tests -p 'test_*.py' -v
```

2026-08-15 最新实际结果：仿真 63/63 PASS；视觉/几何 129/129 PASS；YAML 解析 PASS。单元测试通过不替代 Gazebo 物理验收。

## 11. 关键文件导航

### 当前物理抓取主线

- `src/handarm_sim_demo/scripts/scripted_pick_demo.py`：完整状态机、运动、接触与验收。
- `src/handarm_sim_demo/scripts/hand_commander.py`：GRASP/RELEASE 轨迹与 staged RELEASE。
- `src/handarm_sim_demo/config/physical_grasp_demo.yaml`：运动与验收门限。
- `src/handarm_sim_demo/config/hand_commands.yaml`：当前手角度、顺序和稳定性参数。
- `src/handarm_sim_demo/config/physical_grasp_scene.yaml`：桌面/物体 PlanningScene 几何。
- `src/handarm_sim_demo/worlds/handarm_physical_grasp.world`：真实接触物理世界。
- `src/handarm_sim_demo/launch/scripted_pick_demo.launch`：唯一一键入口。
- `src/handarm_sim_demo/test/test_sim_algorithms.py`：关键契约测试。
- `src/abb120_moveit_config1/config/gazebo_handarm.urdf`：实际运行的 position URDF。
- `src/roboticsgroup_upatras_gazebo_plugins/`：mimic plugin 源码。

### 视觉主线

- `perception_hamer/src/hamer_crop_inference.py`
- `perception_hamer/src/roi_provider.py`
- `perception_hamer/src/palm_frame.py`
- `perception_hamer/src/rgbd_rigid_tracker.py`
- `perception_hamer/src/p5_async_runtime.py`
- `perception_hamer/scripts/run_d455_hamer_crop.py`
- `perception_hamer/scripts/evaluate_hamer_palm_stability.py`
- `perception_hamer/scripts/run_d455_rgbd_relative_tracker_async.py`
- `perception_hamer/scripts/replay_rgbd_relative_tracker.py`
- `perception_hamer/scripts/evaluate_rgbd_relative_orientation.py`

## 12. 旧文档冲突警告

以下文件包含当时正确、但已被后续物理抓取开发取代的历史值：

- `docs/HAND_GRASP_RELEASE_STABILITY.md` 仍写 `f1j1=0.051`、CLOSE 1.30 rad；当前是 0.18 和 0.85/0.85/0.90。
- `docs/SIM_03_SCRIPTED_PICK_BASELINE.md` 主要描述 fixed-joint `deterministic_lift`；当前默认 launch 是 `physical_contact`。
- `docs/SIM_KNOWN_LIMITATIONS.md` 仍写 physical contact NOT RUN；现在已运行，但完整链仍 NOT PASS。
- `docs/SIM_FINAL_CHANGELOG.md` 是更早仿真基线的变更记录，不覆盖 2026-08-15 的接触抓取迭代。

这些历史文件保留用于追溯，不能覆盖本报告的当前结论。

## 13. DeepSeek 协作边界与已采纳/拒绝项

DeepSeek 仅在独立临时目录提出审计与补丁，主工作区由 Codex 单独写入和验收。已采纳：单次 RELEASE 语义、接触清空、放置/撤离验收、释放诊断和恢复 PlanningScene 后再撤离。已拒绝或被实测推翻：

- 依靠 broad mimic/PID/掌型大改而没有物理证据；
- 先开 f1/f3、延后 f2 的释放顺序（实测 f2 被楔住）；
- 第二次 RELEASE 补救后拼接为同一次 PASS；
- 在未恢复 target proxy/ACM 前直接规划撤离；
- 让 DeepSeek 或任何审查者替代最终 Gazebo PASS 裁定。

主要对话在：

- `docs/deepseek_dialogues/20260815T062503183066Z_physical_grasp_8s_failure_round2.md`
- `docs/deepseek_dialogues/20260815T071625762115Z_physical_release_failure_review.md`

## 14. 压缩包内容与刻意排除项

同名 ZIP 是面向 GPT 的文本/小型证据包，包含：本报告、项目文档、主要源码/config/launch/world/URDF、单测、精选真实结果 JSON/CSV、P3/P4/P5 指标与数据索引、DeepSeek 对话。

刻意不包含：

- HaMeR checkpoint、6 GB 官方 tar、MANO 许可模型和任何其他模型权重；
- RGB/depth 帧、rosbag、视频、PNG/STL 大二进制；
- `build/`、`devel/`、conda 环境、ROS logs、`__pycache__`；
- 嵌套 `.git` 和临时 DeepSeek worktree。

因此 ZIP 用于代码/设计/结果审阅，不能独立离线运行完整 Gazebo 或 HaMeR。完整运行仍需原工作区、ROS 依赖、模型资产和 mesh 文件。
