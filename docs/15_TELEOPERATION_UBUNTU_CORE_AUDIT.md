# `teleoperation_ubuntu_core.tar.gz` 审查与迁移记录

审查对象：`/home/diu/teleoperation_ubuntu_core.tar.gz`，SHA-256
`87fa1fd27adb67a07e7aaf97509837e49a831260434fa0d2bfe62d61bb783bc9`。
归档共 456 个条目。先完成静态审查，之后在隔离环境复现了观察器；原始现场运行的
前臂有效率为 0/1973，因此该独立脚本只保留作审计参考，没有覆盖当前主程序。

## 结论

归档中的完整实时 HaMeR/MANO 显示链现已迁入当前工程，并成为默认显示实现：使用
HaMeR 的 778 个 `pred_vertices`、裁剪相机参数和 MANO 的 1538 个三角面，在产生
该推理结果的同一张 RGB 帧上投影，再按归档形式显示“原始推理帧/完整网格”双栏。
旧的“把上一帧网格按 KLT 框缩放到最新 RGB”显示不再是默认，只能通过
`--mesh-renderer legacy-depth` 显式启用。

归档 V9 的 16 点 MANO 手腕开口参考也已迁移并成为控制默认值：位置为 16 点中心
投影后从 D455 对齐深度取得的公制点；姿态为中性手腕环到当前手腕环的 IRLS/Huber
稳健相似 Kabsch 拟合，再进入现有因果 SO(3) 滤波。旧的 `joint 0 + MCP` 方法保留为
`--control-reference mano-joint-palm` 回退项。V5 的局部 RGB-D 前臂原则已迁入为低
权重纵轴锚点：最大融合权重 0.20，MANO 继续负责完整腕姿态和横滚；前臂无效时精确
退回 MANO。没有把单根前臂轴冒充完整三轴姿态，也没有迁入 V9 的 0.55 s 姿态预测保持。

没有整体接入 V8.3 控制模型，因为它输出 `TRANSLATION/YAW/PITCH/ROLL` 四个互斥
类别，而当前系统必须允许六个速度分量同时存在。渲染迁移和 V8.3 控制模型是否
适用是两个独立问题。

归档与当前工程的 `MANO_RIGHT.pkl`、`mano_mean_params.npz` 和
`model_config.yaml` SHA-256 完全一致。归档没有 `hamer.ckpt`，当前工程已有完整
权重；归档的 HaMeR model/head/geometry 源码与当前 conda 包在忽略 Windows CRLF
换行后也完全一致，所以复制模型或网络源码不会改善推理或渲染效果。

## 实时 HaMeR/MANO 对比

归档 `live_hamer_realsense_mesh_windows.py` 的优点：

- 推理请求队列容量为 1，旧帧被覆盖；
- MANO 三角形使用一次 `fillPoly` 批量绘制；
- MediaPipe 框使用新帧权重 `0.55` 的因果 EMA；
- 显式记录相机频率、HaMeR 频率和端到端耗时。

当前工程原显示实现将旧 HaMeR 网格按最新 KLT 框做中心/尺度重映射。虽然相机画面
保持 30 Hz，但网格的手指关节和三维姿态仍来自旧推理帧，快速运动时会产生网格
悬空、拉动或“框跟上而手模型没跟上”的错觉。默认链已丢弃该跨帧重映射效果。

归档的投影、完整面片绘制、颜色、稀疏边线和双栏布局已完整迁入
`perception_hamer/src/teleoperation_core_mano_renderer.py`。归档主循环的旧结果保持
行为没有迁入：当前外围仍保留连续真实手检测、单活动手自动切换、ROI/检测框空间
一致性检查、推理完成后二次 presence 检查和无手 UDP 截止。快速运动时只容忍一个
不超过 0.08 s 的孤立检测漏帧；第二个连续漏帧后 presence 无效，双栏退回当前真实
相机图像且两侧均不画 MANO。

本轮同时保留了归档的 `0.55` ROI 平滑权重。旧值 `0.35` 在一步 12 px 合成平移
测试中更滞后；但 ROI 现在只决定下一次 HaMeR 裁剪，不再用于把旧网格伪装成新姿态。

确定性 640×480、778 顶点、1538 局部三角面的 CPU 显示基准中，本机本次运行的
归档完整面片路径 median/P95 为 `2.267/2.357 ms`，旧深度分层路径为
`5.343/5.483 ms`。这是合成拓扑的绘制耗时比较，不包含 HaMeR 推理、JPEG sidecar
显示或 D455 采集。

此外，已对现有真实 D455 录制
`DEV_HAMER_STATIC_20260813T194133` 的第 27 帧重新运行当前检查点：模型预热
本轮复跑 `274.077 ms`，预热后该帧 HaMeR 推理 `40.468 ms`，输出 778 顶点/1538
面并成功生成 1280×480 精确双栏 PNG。真实 MANO 拓扑自动发现的腕环顶点索引为
`[38,92,234,239,279,215,214,121,78,79,108,120,119,117,118,122]`，数量严格为 16；
该帧 D455 公制腕环中心为 `[-0.01419,0.18601,0.42100] m`，几何/深度置信度分别为
`0.873/0.717`，深度中位数 `0.421 m`、MAD `0.001 m`。它证明模型、腕环参考、对齐
深度和完整渲染器的离线整链可执行，不等于在线连续帧率或姿态真值测试。

迁移代码的归档 MIT 许可和版权声明已保存在
`perception_hamer/THIRD_PARTY_NOTICES.md`。

## SO(3)、裁剪质量与跟踪间断

从归档 V9 诊断代码迁移并按遥操作安全语义收紧了以下内容：

- `perception_hamer/src/causal_wrist_so3_filter.py`：按真实 `dt` 在 SO(3) 上更新，
  不做矩阵元素或欧拉角平均；质量差时降低测量增益；单帧大姿态创新直接无效，
  连续三次一致测量后才重新初始化。
- `perception_hamer/src/crop_quality.py`：只使用当前框和上一个已接受框，依据尺寸、
  边界截断、中心跳动和尺度跳动给出因果质量；质量进入六轴置信度。
- 六维趋势估计器在无效输入或长时间戳间断后清空局部导数窗口，但保留操作者确认的
  零位。恢复帧速度为零，丢失期间的运动不会稍后作为“追赶速度”发出。

观察版原实现会在短时缺失时对屏幕保持旧姿态。当前遥操作版本将位置和姿态通道拆开：
MANO 姿态创新超过 35° 时保持最后可信 SO(3)、旋转置信度置零，但 D455 公制腕位置
继续更新；真实手/ROI/深度整体失效时才停止新报文。Gazebo 使用 V3 HOLD_LAST，实体
输出仍保留独立 watchdog。

两段已有 USB2 开发静止录制的只读配对回放结果如下：

| 录制 | 有效帧 | 原始帧间角 median/P95 | 滤波后 median/P95 | 滤波耗时 P95 |
|---|---:|---:|---:|---:|
| `DEV_HAMER_STATIC_20260813T184508` | 409 | 3.866° / 8.234° | 0.916° / 2.081° | 0.372 ms |
| `DEV_HAMER_STATIC_20260813T194133` | 374 | 1.132° / 2.978° | 0.379° / 1.110° | 0.392 ms |

两段均无滤波拒绝帧。该指标只说明已有“静止”录制的帧间抖动下降，不是姿态真值
误差，也不能证明快速动作延迟合格；完整 JSON 可用下方回放命令重新生成。

2026-08-20 又录制了现场 8.04 s USB2 开发段
`DEV_HAMER_STATIC_20260820T231136`：101 个处理帧、88 个有效 HaMeR 帧，实际有效
HaMeR 10.94 Hz，推理均值/P95 为 42.58/45.96 ms，显存峰值 4440 MiB。原始宽深度
带会把手臂和身体连成约 0.52 m 假组件；加入“距独立 MANO/D455 腕中心不超过
0.19 m”的公制局部门后，该段只读重放前臂轴 88/88 有效，轴帧间角变化中位/P95
为 2.26°/6.88°，置信度中位 0.889。随后在线抓取 50 个 UDP 包，37 个直接融合、
其余为短保持或 MANO 回退；融合权重中位 0.155，姿态修正中位/最大
3.24°/3.81°，前臂处理耗时中位/P95 为 15.12/19.08 ms。以上是稳定性和运行数据，
不是姿态真值误差，也不代表所有动作/背景均已验证。

## V8.3 意图模型为何不进入控制链

归档报告明确标记该模型为 `development_only`、`observer_only`、
`robot_control_authorized=false`，训练/开发参与者只有 P01、P02。其留一参与者开发
结果为：普通动作 Macro-F1 约 `0.947`，慢动作 Macro-F1 约 `0.840`；但慢动作
切换 P95 延迟约 `3.07 s`，P01 折达到 `5.65 s`。选择性输出约拒绝 `9.90%` 样本。
这些指标不能支持 50 Hz 机械臂实时控制，也没有独立第三参与者验证。

此外，V8.3 的 0.40/0.85/1.60 s 多尺度证据和 0.22 s 概率滤波服务于互斥类别稳定，
会显著增加当前连续六轴速度链的切换延迟。本轮只采用其“真实时间窗、因果处理、
间断不补发”的原则，不加载 `multiscale_intent_v83.joblib`，不把四类概率接到 Servo。

## 可复现检查

```bash
cd /home/diu/myhandarmtest1
python3 -m unittest discover -s perception_hamer/tests -v
PYTHONPATH=src/handarm_moveit_demo/src \
  python3 -m unittest -v src/handarm_moveit_demo/test/test_shared_teleop_core.py

python3 perception_hamer/scripts/evaluate_causal_wrist_filter.py \
  datasets/development_usb2/hamer_palm_stability/DEV_HAMER_STATIC_20260813T184508/frames.jsonl \
  datasets/development_usb2/hamer_palm_stability/DEV_HAMER_STATIC_20260813T194133/frames.jsonl

python3 perception_hamer/scripts/benchmark_mano_renderers.py --iterations 40

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n hamer_rtx2060 \
  python perception_hamer/scripts/render_recorded_hamer_frame.py \
  datasets/development_usb2/hamer_palm_stability/DEV_HAMER_STATIC_20260813T194133 \
  --index 27 \
  --output /tmp/teleoperation_core_mano_frame27.png \
  --overwrite
```

安全实时观察/UDP 命令仍只启动一个 HaMeR；不需要按 `C` 选择 MediaPipe 活动手：

```bash
cd /home/diu/myhandarmtest1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n hamer_rtx2060 \
  python perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe \
  --mesh-renderer teleoperation-core \
  --control-reference mano-wrist-ring \
  --roi-smoothing-alpha 0.55 \
  --orientation-filter-time-constant-s 0.10 \
  --orientation-filter-large-angle-mode follow \
  --orientation-filter-max-gain 1.0 \
  --orientation-filter-soft-deg 25 \
  --orientation-filter-hard-deg 60 \
  --disable-forearm-fusion \
  --hand-presence-timeout-s 0.25 \
  --teleop-udp-host 127.0.0.1 \
  --teleop-udp-port 5010
```

## 尚未验证

- 完整渲染链已通过投影、左右手仿射、全部面片、精确推理帧配对和无人手清屏测试，
  但尚未完成现场 D455/USB3 肉眼验收；不能声称实际 MANO 贴合精度已经提高。
- 精确帧配对意味着左右两栏只在新的 HaMeR 结果产生时更新。它不会假装网格有
  30 Hz；相机仍是约 30 Hz，网格更新率取决于本机实际 HaMeR Hz。
- 当前前臂轴只以 0--0.20 连续权重辅助 MANO 纵轴，并随 UDP 六维位姿发送完整审计
  字段；它不能观测绕自身的横滚，质量失败时会显示 `FOREARM MANO-ONLY`。尚无
  外部姿态真值，不能把上述稳定性指标解释为绝对精度。
- 没有进行实体 ABB 测试，实体输出默认仍关闭。
