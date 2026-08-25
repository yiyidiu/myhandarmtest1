# D455–HaMeR 掌部坐标系与稳定性开发报告

日期：2026-08-13

当前D455使用USB 2.1，本阶段结果用于算法开发。最终实时性能、正式数据集和长时间稳定性以后在USB3条件下重新测试。

## 本轮实现

实时链路已经实现为：

```text
D455 RGB8 + aligned Z16, 640x480@30
  -> 人工鼠标框选或 MediaPipe 一次性存在性/2D bbox/粗左右手
  -> KLT + forward/backward check + RANSAC ROI 跟踪与尺度平滑
  -> 容量为 1 的 latest-frame overwrite mailbox
  -> HaMeR crop-only, batch=1, FP16, torch.inference_mode()
  -> MANO joints/vertices/pose/betas
  -> raw global_orient / joint palm / rigid-vertex palm 三路输出
  -> OpenCV RGB 叠加与显式实验录制
```

未使用 ViTDet、ViTPose、Pyrender 常驻窗口、OBJ 导出或普通实时模式逐帧保存。
MediaPipe sidecar 只输出手存在、二维 bbox 与粗左右手，不输出关键点、world landmarks 或 z，
不参与掌部姿态定义。

ROI 输出字段为 `bbox/source/valid/age/center_jump/scale_change/lost/reinitialized`；
KLT 丢失时不复用旧框。HaMeR 输出包含 capture timestamp、bbox、778 MANO vertices、21
MANO joints、`global_orient (3,3)`、`hand_pose (15,3,3)`、`betas (10,)`、推理耗时、
valid 与 failure reason。30 个有效结果后对 betas 逐坐标取中位数，后续通过 MANO 层以
`betas_user` 重算 joints/vertices。

掌坐标系 B 严格使用：

```text
x = normalize(index_mcp - little_mcp)
y_raw = middle_mcp - wrist
y = normalize(y_raw - dot(y_raw,x)x)
z = normalize(cross(x,y))
y = normalize(cross(z,x))
R_palm = [x,y,z]
```

有限值、轴长、单位长度、正交性和 `det(R)=+1` 均检查；invalid 不输出单位矩阵冒充
有效姿态。四元数使用 `q/-q` 连续约定。`global_orient` 只作 A 路对照，禁止当作最终
机械臂姿态。C 路可选方法使用 87 个 root/wrist 权重大于 0.98 的冻结刚性掌部顶点。

## HaMeR 真实推理

以下资产实际加载：

- `_DATA/hamer_ckpts/checkpoints/hamer.ckpt`（约 2.6 GiB）；
- `_DATA/hamer_ckpts/model_config.yaml`；
- `_DATA/data/mano_mean_params.npz`；
- `_DATA/data/mano/MANO_RIGHT.pkl`。

在已有 D455 归档真实左手图像 `p2_smoke/rgb/000000.png` 上实测输出：

| 输出 | 实际形状/结果 |
|---|---|
| MANO vertices | `(778,3)` |
| MANO joints | `(21,3)` |
| global_orient | `(3,3)` |
| hand_pose | `(15,3,3)` |
| betas | `(10,)` |
| A/B/C palm valid | 全部 true |
| A/B/C det(R) | 均约 1.0 |

实时不落盘冒烟 2.03 秒得到 41/41 有效 HaMeR 结果，实际 20.22 Hz，推理中位
40.14 ms、P95 43.76 ms，系统显存峰值 4714 MiB。30 个 betas 样本已成功中位冻结；
相机发布 78 帧时 latest-frame scheduler 丢弃 19 个旧帧，证明无推理队列积压。

## 三组真实手开发录制

三段最终可用 session 均为约 25 秒，MediaPipe 只在开始时确认真实手与二维框，随后
KLT 接管；画面首/中/末抽查分别确认张掌静止、空间平移、张开/握拳动作实际发生。

| 实验 | 有效帧 | 覆盖率 | HaMeR Hz | 延迟 mean / P50 / P95 (ms) | RTX 2060 系统显存峰值 |
|---|---:|---:|---:|---:|---:|
| STATIC | 374/374 | 100% | 14.97 | 47.54 / 47.45 / 49.76 | 4959 MiB |
| TRANSLATION | 372/372 | 100% | 14.90 | 48.59 / 48.21 / 49.54 | 4908 MiB |
| OPEN_CLOSE | 371/371 | 100% | 14.85 | 48.37 / 47.92 / 49.38 | 4910 MiB |

每段前 30 个 betas 校准样本不参与稳定性统计。进入评价窗口后依次保留 345、343、342
帧；每段 `betas_user` 均已冻结。RGB、aligned depth、JSONL、叠加视频逐段数量严格一致，
三个视频均已逐帧解码验证。

## A/B/C 稳定性结果

主指标为相对该段 SO(3) chordal reference 的测地角，单位为度，不使用欧拉角差值。

| 实验 | 方法 | P50 | P95 | 最大值 | ROI变化–相邻姿态相关性 |
|---|---|---:|---:|---:|---:|
| 静止 | A raw global_orient | 2.934 | 5.511 | 9.188 | 0.147 |
| 静止 | B MANO joint palm | 2.935 | 5.511 | 9.189 | 0.145 |
| 静止 | C rigid palm vertices | 2.951 | 5.582 | 9.328 | 0.143 |
| 纯平移 | A raw global_orient | 55.698 | 157.090 | 171.967 | 0.101 |
| 纯平移 | B MANO joint palm | 55.697 | 157.089 | 171.971 | 0.101 |
| 纯平移 | C rigid palm vertices | 55.783 | 157.867 | 172.789 | 0.101 |
| 张开握拳 | A raw global_orient | 25.194 | 139.628 | 179.316 | -0.385 |
| 张开握拳 | B MANO joint palm | 25.193 | 139.629 | 179.316 | -0.385 |
| 张开握拳 | C rigid palm vertices | 24.761 | 139.341 | 179.686 | -0.385 |

静止段 ROI 中心相邻变化 P50/P95/最大值为 0.031/0.108/0.656 px；平移段为
0.118/0.703/2.378 px，ROI 尺度相邻比 P50/P95/最大值为 1.00065/1.00531/1.04584；
开合段中心变化为 0.059/0.169/0.238 px。相关性按相邻实际推理帧重新计算，已包含
latest-frame scheduler 丢帧后的真实帧间隔。

### 判定

A/B/C 在三个实验中几乎同步变化，B、C 没有相对 raw `global_orient` 获得有意义的稳定性
提升。原因是 MANO joints/vertices 已经包含 HaMeR 估计的根部 global rotation，从这些
点集重新构造掌轴不能独立消除同一个根姿态错误。纯平移和开合时出现了远超静止基线的
假旋转，P95 分别约 157° 和 139°；不能用
低通滤波把这一现象掩盖，也不能将 `global_orient` 直接用于机械臂控制。

**需要进入 RGB-D Kabsch 融合。** 下一步应使用掌部 ROI 内的 KLT 对应、两帧 aligned
depth 反投影、RANSAC 和无尺度 Kabsch 估计刚体旋转增量。现有 P5 合成代码与测试保留，
但本轮没有继续扩展或运行真实 P5 链路。

## 被拒绝的早期录制

第一次执行生成了三个 25 秒目录及 RGB、aligned depth、JSONL 和叠加视频：

- `DEV_HAMER_STATIC_20260813T184508`：409/409 数值输出；
- `DEV_HAMER_TRANSLATION_20260813T184556`：415/415 数值输出；
- `DEV_HAMER_OPEN_CLOSE_20260813T184640`：409/409 数值输出。

视觉抽查证明三段画面均为无人实验室与椅背，ROI 中没有手。HaMeR 在背景上仍产生了
有限数值，因此这些目录已明确列入 `REJECTED_BACKGROUND_SESSIONS.md`，不能作为三类
稳定性数据，之前基于它们产生的指标 JSON/CSV 也不是有效实验结论。评价器现只选择
`roi_seed_hand_presence_validated=true` 的 session；当前运行会拒绝这些旧目录。

为防止再次发生，实时脚本已加入 MediaPipe 一次性 hand-presence/2D-bbox preflight，
本轮随后实际运行时正确报告 `no_hand_detected` 并在 HaMeR 启动前停止。人工鼠标框选也
通过带 Qt 的轻量 sidecar 提供，不受 HaMeR 环境无 GUI OpenCV 的影响。

另有 `DEV_HAMER_STATIC_20260813T193921`：存在性预检成功但旧版 KLT 初始化时序错误，
0/750 有效，已单独标为 tracker initialization regression sample。两个拒绝类型都不会
被当前评价器选入。

## 自动测试与交付物

`python3 -m unittest discover -s perception_hamer/tests -p 'test_*.py' -q` 实测
`115/115 PASS`。覆盖 ROI clip/lost/reinitialize/尺度平滑、单槽旧帧覆盖、HaMeR shape、
SO(3)、左右手、`q/-q`、betas 中位冻结、无效不返回单位姿态、A/B/C 评价与数据索引。

主要交付：

- `perception_hamer/src/roi_provider.py`
- `perception_hamer/src/hamer_crop_inference.py`
- `perception_hamer/src/hamer_palm_frame.py`
- `perception_hamer/src/realtime_hamer_pipeline.py`
- `perception_hamer/scripts/run_d455_hamer_crop.py`
- `perception_hamer/scripts/evaluate_hamer_palm_stability.py`
- `perception_hamer/scripts/mediapipe_detect_roi_once.py`
- `perception_hamer/scripts/manual_select_roi_once.py`
- 对应单元测试

本轮没有连接 MoveIt Servo、控制 ABB、开发三指手抓取或避障，也没有用低通滤波改变
姿态定义。P2 已冻结；没有修改 `record_rgbd_session.py` 或 `verify_rgbd_session.py`，
没有继续长录、depth bias、彩深对齐实验或 USB3 认证开发。

两次 fx/fy 失败只保留诊断，不继续放宽容差。只读探测确认 raw depth 为
`depth/index0/profile ID0/Z16`，目标 color 为 `color/index0/profile ID3/RGB8`，aligned
depth 为 `depth/index0/profile ID10/Z16`；三者均为 640×480@30。aligned depth 的
fx/fy/ppx/ppy 和 distortion 与目标 color profile 一致，比较对象不是 raw depth 内参。
完整记录见 `perception_hamer/test_results/p3p4_realsense_profile_diagnostic.json`。
