# P5 独立 RGB-D 相对刚体姿态跟踪

日期：2026-08-13

## 结论

本轮完成了独立的 D455 RGB-D 相对刚体姿态跟踪器、异步实时运行程序、离线回放、
SO(3) 评价、状态机、单元测试和三组真实归档回放。当前结果**未达到**进入低速 Gazebo
姿态测试的最低条件，不建议进入 MoveIt；下一步应优先评估稠密 RGB-D 掌部配准或
point-to-plane ICP。

HaMeR 姿态完全未参与 Kabsch 输出。本报告不称为“HaMeR 姿态融合”，也没有使用
`global_orient`、MANO joint palm frame 或 rigid palm vertex frame 作低频校正。HaMeR
异步线程仅允许提供 ROI、`hand_pose` 和 gesture context；上下文对象会拒绝任何
`global_orient`、`rotation` 或 `orientation` 键。

> 当前D455使用USB 2.1，本阶段结果用于算法开发。最终实时性能、正式数据集和长时间稳定性以后在USB3条件下重新测试。

## 实现

- `perception_hamer/src/rgbd_rigid_tracker.py`：中央腐蚀掌区、Shi–Tomasi、双向 KLT、
  两帧 aligned depth、D455 彩色内参反投影、3 点 RANSAC、无尺度 Kabsch、质量门限、
  `INITIALIZING/TRACKING/FROZEN/LOST` 状态和 clutch 零位累计。
- `perception_hamer/src/p5_async_runtime.py`：Kabsch 小型连续 FIFO、HaMeR 单元素
  latest-only mailbox，以及禁止 HaMeR 姿态字段的上下文。
- `perception_hamer/scripts/run_d455_rgbd_relative_tracker_async.py`：D455 采集线程、
  连续 RGB-D Kabsch、独立 HaMeR latest-only 线程、异步写盘和叠加视频。
- `perception_hamer/scripts/replay_rgbd_relative_tracker.py`：只读真实归档的 RGB、aligned
  Z16、内参、帧号和时间戳，重新运行同一 RGB-D 算法。
- `perception_hamer/scripts/evaluate_rgbd_relative_orientation.py`：只用 SO(3) 测地角评价。

变换约定为：

```text
p_current = R_increment @ p_previous + t_increment
det(R_increment) = +1
translation unit = metre
similarity scale estimation = disabled
orientation source = RGBD_KLT_RANSAC_KABSCH_ONLY
```

每帧记录 `tracked_2d_points`、`forward_backward_valid_points`、`valid_3d_pairs`、
`ransac_inliers`、`inlier_ratio`、`kabsch_rms_m`、`rotation_increment_deg`、
`translation_increment_m`、`spatial_span_m`、`covariance_singular_values`、`frame_gap`、
`tracker_age`、`valid` 和 `failure_reason`。无效帧的 R/t 是 `null`，不以 I/0 冒充。

失效语义：可靠帧才累计；单帧失败进入 FROZEN，连续失败超过 0.25 s 进入 LOST；
LOST 清除累计姿态。ROI 重捕获后仍保持 LOST，必须再次 clutch 才把当前手姿定义为 I。

## 180° 翻转有限审计

现有 P3/P4 异常不是坐标轴转换或评价错误，而是
`HAMER_ROOT_ORIENTATION_FAILURE`：

- TRANSLATION 首个有效窗口内 >60° 跳变：index 105→106，D455 frame 381→383，
  dt 0.0666373 s；raw/joint/rigid 分别为 64.9964°/64.9967°/65.2143°。
- OPEN_CLOSE：index 39→40，frame 249→251，dt 0.0666361 s；三者为
  165.7149°/165.7162°/165.4987°。
- 两处均是右手、ROI/affine 连续、旋转 det=+1、正交误差约 1e-15。
- 同索引 87 个 rigid palm vertices 的 Kabsch 也得到 65.1047°/165.6000°；去掉
  HaMeR root 后只剩 0.1184°/0.1259°。因此整个 MANO 网格随错误 root 旋转。

结论：HaMeR 的任何姿态字段都不能进入后续机械臂姿态通道。

## 实际运行与数据边界

D455 实测设备：serial `234322305987`，FW `5.17.3.10`，librealsense
`2.58.2.10647`，USB descriptor `2.1`。原始 Profile 是 RGB8 + Z16
640×480@30，depth align 到 color；depth scale 0.0010000000475 m/unit。

当前镜头实时画面四次检查均无人手，MediaPipe 2-D presence 均返回
`no_hand_detected`；鼠标 ROI sidecar 也在 120 s 后超时。因此未伪造 30 Hz 实时人体
测试。本轮只读复用了被冻结 P3/P4 归档的 RGB、aligned Z16、内参、真实 frame id 和
timestamp。旧归档是 HaMeR latest-only 约 14.9 Hz 保存帧，不代表相机只有 15 Hz。

离线 ROI 只使用允许范围内的二维手/掌部区域：由 wrist 与四个 MCP 的二维投影生成
中央掌区，再经 bbox 腐蚀与深度连续门限筛选；完全不读取原归档的任何姿态字段。

| 测试 | 真实输入 | 实际频率 | 有效覆盖 | FROZEN | LOST | 状态 |
|---|---|---:|---:|---:|---:|---|
| P5_STATIC | 374 RGB-D 帧 | 14.973 Hz | 99.20% | 1 | 0 | 实际离线回放 |
| P5_TRANSLATION | 372 RGB-D 帧 | 14.896 Hz | 34.14% | 4 | 355 | 实际离线回放，不通过 |
| P5_GESTURE | 371 RGB-D 帧 | 14.853 Hz | 84.64% | 59 | 265 | 实际离线回放，不通过 |
| P5_ROTATION | 无对应真实归档 | — | — | — | — | NOT RUN |
| 30 Hz 实时四组 | 镜头无人手 | — | — | — | — | NOT RUN |

## SO(3) 指标

主要指标均为 `acos((trace(R_ref.T @ R)-1)/2)`，不是欧拉角差。

| 测试 | 相邻增量 P50/P95/max | 累计变化 P50/P95/max | inlier P50 | RMS P95 | span P50 |
|---|---|---|---:|---:|---:|
| STATIC | 0.625° / 1.517° / 18.032° | 3.548° / 18.579° / 19.925° | 1.000 | 1.345 mm | 0.101 m |
| TRANSLATION | 1.606° / 10.986° / 14.677° | 0.000° / 11.083° / 11.083° | 1.000 | 3.931 mm | 0.080 m |
| GESTURE | 2.073° / 9.370° / 14.915° | 9.939° / 16.240° / 18.586° | 1.000 | 3.039 mm | 0.088 m |

ROI 变化与相邻旋转增量的 Pearson 相关系数：STATIC 0.369、TRANSLATION 0.057、
GESTURE 0.439。未出现“已接受的单帧 >30°”跳变，但静止累计漂移与平移假旋转均超标。

Kabsch 离线单帧处理耗时 P95：STATIC 24.09 ms、TRANSLATION 26.10 ms、GESTURE
27.44 ms。进程 CPU 利用率分别为 126.9%、132.3%、133.1%，峰值 RSS 为
209.4/209.4/212.8 MiB（OpenCV 可使用多核）。30 Hz 实时 CPU、内存和端到端频率因为
镜头无人手而 NOT RUN。

三轴小幅旋转方向/幅值响应、真实 re-detection/re-clutch、30 Hz 实时 frame-gap 分布均
NOT RUN，不能从 15 Hz 抽帧回放外推。

## 判定

最低条件逐项：

- 静止累计姿态 P95 <5°：**失败**，18.579°。
- 纯平移假旋转 P95 <10°：**失败**，11.083°，且仅 17 帧保持可用累计姿态。
- 无未拒绝单帧 >30°：当前三组已运行数据通过，但 P5_ROTATION 未运行。
- 有效覆盖 >90%：**失败**，平移 34.14%，手势 84.64%。
- 失败进入 FROZEN/LOST：实际回放和单元测试通过。
- 重捕获不自动续接：单元测试通过；真实重捕获 NOT RUN。

综合判定：`minimum_gazebo_criteria_pass=false`，不得接 MoveIt 或 ABB。单 D455 裸手
KLT–Kabsch 当前对静止相邻增量可用，但作为连续相对姿态源不可用。建议下一步改为稠密
RGB-D 掌部配准或 point-to-plane ICP；不再调 HaMeR 姿态或使用低通滤波掩盖定义问题。

## 验证与交付物

P5 定向测试覆盖无尺度 Kabsch、det=+1/反射修正、SVD/退化、RANSAC final consensus、
KLT FB、双帧深度、15/30 Hz、不规则 dt、跳帧、无深度、状态机、clutch、latest-only
和离线评价。本轮结束前全仓 `perception_hamer/tests` 回归与全 Python `py_compile`
均通过；冻结的 P2、P3/P4、HaMeR 核心和 ROS 文件 SHA256 全部一致。

机器可读结果与数据索引：

- `datasets/development_usb2/p5_rgbd_relative_orientation/development_dataset_index.json`
- `datasets/development_usb2/p5_rgbd_relative_orientation/p5_rgbd_relative_orientation_metrics.json`
- `datasets/development_usb2/p5_rgbd_relative_orientation/p5_rgbd_relative_orientation_metrics.csv`
- 每组 session 的 `frames.jsonl`、`summary.json` 和 `tracking_overlay.mp4`

本轮在此停止，未启动 ROS、Gazebo、MoveIt、ABB、三指手或避障。
