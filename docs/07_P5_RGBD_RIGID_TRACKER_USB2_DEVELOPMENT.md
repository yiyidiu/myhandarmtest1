# P5 RGB-D 刚体跟踪器（USB2 开发范围）

> 本文是早期合成测试快照，已由
> [`docs/P5_RGBD_RELATIVE_ORIENTATION.md`](P5_RGBD_RELATIVE_ORIENTATION.md)
> 的真实 RGB-D 离线回放、状态机、资源和判定结果取代。以下 `NOT RUN` 只描述当时
> 的历史状态，不代表当前工作区结论。

日期：2026-08-13

## 结论与证据边界

```text
P5_OFFLINE_RIGID_MATH_GATE          PASS
P5_SYNTHETIC_RGBD_TRACKER_GATE      PASS
P5_DEVELOPMENT_IMPLEMENTATION       PASS_SYNTHETIC_ONLY
P5_REAL_USB2_REPLAY                 NOT RUN
P5_LIVE_USB2                        NOT RUN
P5_FORMAL_USB3                      DEFERRED_USB3_UNAVAILABLE
formal_P2_pass                      false
formal_dataset_eligible             false
```

本阶段完成了掌部 ROI 到成对 SE(3) 增量的独立实现和合成测试。本轮没有占用 D455，
没有运行真实 USB2 手部回放或实时流，也没有生成正式 USB3/G00～G09 结论。P2 的
正式 USB3 门禁没有被删除或绕过。

实现文件：

- `perception_hamer/src/rgbd_rigid_tracker.py`
- `perception_hamer/tests/test_rgbd_rigid_tracker.py`

## 几何和时间契约

输入深度必须是已对齐到彩色像素网格的原始 Z16，使用每帧记录的
`depth_scale_m_per_unit` 转换为米。三维点位于 D455 彩色相机坐标系。增量定义为：

```text
p_current = R_increment @ p_previous + t_increment
t_increment unit = metre
det(R_increment) = +1
similarity scale = disabled
```

输入 `timestamp_s` 是实际设备时间戳的秒值；接 D455 API 时应使用
`color_timestamp_ms / 1000`，并传入原始 `timestamp_domain`。算法不使用名义 FPS
构造时间：

- 15 Hz、30 Hz 和门限内不规则 `dt` 均按实际差值输出；
- timestamp domain 改变、时间戳不递增会立即重初始化；
- 默认 `maximum_dt_s=0.12`；超过门限不递推；
- 默认 `maximum_frame_gap=1`；检测到跳帧不跨越缺帧估计运动，而以当前帧重初始化。

第一帧、重初始化帧和所有失败帧都满足：

```text
valid = false
rotation_increment = null
translation_increment = null
failure_reason != "NONE"
```

因此 Kabsch 失败不会用单位旋转、零平移伪装成有效静止运动。

## 算法链

```text
palm bbox (continuous half-open xyxy)
  -> previous ROI Shi-Tomasi
  -> pyramidal KLT previous -> current
  -> KLT current -> previous
  -> forward/backward pixel error gate
  -> current palm ROI gate
  -> previous/current aligned-depth median + MAD gate
  -> calibrated color-grid deprojection
  -> metric 3-D correspondences
  -> 3-point RANSAC rigid hypotheses
  -> inlier Kabsch refinement
  -> SO(3), RMS and consensus gates
  -> pairwise SE(3) increment
```

Kabsch 只计算旋转和平移。SVD 反射分支被修正为 `det(R)=+1`；源点或目标点接近
共线时拒绝求解。平面上的非共线掌部点可以求解，不要求不现实的满三维体积散布。

D455 开发会话的 aligned-depth 使用 `inverse_brown_conrady` 彩色内参。实现使用
Brown-Conrady 去畸变后反投影，并用本机 librealsense 2.58.2 的固定参考值测试。
RealSense 对 deprojection/畸变模型的定义见：
<https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0>。

## 输出契约

`RigidTrackResult.as_dict()` 至少输出提示词要求的字段：

| 字段 | 含义 |
|---|---|
| `valid_3d_pairs` | 两帧都具有有效深度的 FB 一致特征数 |
| `ransac_inliers` | 最终刚体模型内点数 |
| `inlier_ratio` | `ransac_inliers / valid_3d_pairs` |
| `kabsch_rms` | 最终内点三维残差 RMS，单位米；无效时为 null |
| `rotation_increment` | row-major 3×3 SO(3)；无效时为 null |
| `translation_increment` | 三维米制平移；无效时为 null |
| `frame_gap` | 当前帧号减上一帧号 |
| `tracker_age` | 最近一次初始化后的连续有效增量数 |
| `failure_reason` | 有效时为 `NONE`，否则为明确原因 |

附加输出包括 `dt_s`、`reinitialized`、Shi-Tomasi 候选数、forward KLT 数、FB
一致数、变换约定、平移单位和 `scale_estimation=DISABLED`。

主要失败原因包括：

```text
INITIALIZING
TIMESTAMP_DOMAIN_CHANGED
TIMESTAMP_NON_INCREASING
FRAME_NUMBER_NON_INCREASING
FRAME_GAP_EXCEEDS_MAXIMUM
DT_EXCEEDS_MAXIMUM
INSUFFICIENT_FB_TRACKS
INSUFFICIENT_VALID_DEPTH_PAIRS
DEGENERATE_3D_GEOMETRY
RANSAC_NO_CONSENSUS
KABSCH_RMS_EXCEEDS_LIMIT
```

失败后当前帧成为新的基准帧，`tracker_age` 清零；不会继续使用失败前的陈旧对应。

## 默认开发门限

| 门限 | 默认值 |
|---|---:|
| Shi-Tomasi maximum corners | 160 |
| KLT FB error | 1.0 px |
| minimum FB tracks | 8 |
| depth patch radius | 1 px |
| depth patch MAD | 0.02 m |
| valid depth range | 0.10～3.0 m |
| minimum 3-D pairs | 6 |
| RANSAC iterations | 64 |
| RANSAC residual | 0.012 m |
| minimum inlier ratio | 0.50 |
| maximum Kabsch RMS | 0.008 m |
| maximum dt | 0.12 s |
| maximum frame gap | 1 |

这些值是安全的开发初值，不是 USB3 正式调参结果。真实掌部数据必须记录阈值扫描、
内点率、RMS、重初始化率和错误运动分布后再冻结部署配置。

## 已运行测试

执行：

```bash
cd /home/diu/myhandarmtest1
python3 -m unittest perception_hamer.tests.test_rgbd_rigid_tracker -v
python3 -m unittest discover -s perception_hamer/tests -v
python3 -m py_compile \
  perception_hamer/src/rgbd_rigid_tracker.py \
  perception_hamer/tests/test_rgbd_rigid_tracker.py
```

结果：

```text
P5 targeted tests       18/18 PASS
full current regression 100/100 PASS
static syntax           PASS
```

P5 测试覆盖：

- 纯平移，并严格验证 `R_increment ~= I`；
- 纯旋转和旋转加平移；
- 高斯噪声；
- 30% 错误对应与 RANSAC；
- 接近共线退化；
- 反射矩阵修正；
- 禁止自由尺度拟合；
- 真实 OpenCV Shi-Tomasi、forward/backward KLT；
- 两帧深度均有效的门禁；
- D455 inverse-Brown 反投影参考值；
- 实际 15 Hz、30 Hz、不规则 `dt`；
- 跳帧、超时距、时间戳倒退和时间域切换重初始化；
- 深度全无效；
- 无效结果不能携带单位旋转/零平移替代值。

## 集成要求与剩余风险

P3 应传入“掌部刚性 ROI”，不能把明显形变的整只手和指尖作为默认 KLT 区域。
与 P2 对接时字段映射为：

```text
rgb                         <- D455Frame.rgb
aligned_depth_raw           <- D455Frame.aligned_depth_raw
color_intrinsics            <- D455Frame.color_intrinsics
depth_scale_m_per_unit      <- D455Frame.depth_scale_m_per_unit
timestamp_s                 <- D455Frame.color_timestamp_ms / 1000
frame_number                <- D455Frame.color_frame_number
timestamp_domain            <- D455Frame.color_timestamp_domain
palm_bbox_xyxy              <- P3 palm ROI
```

当前模块只输出相邻帧增量，不累积全局姿态；累计、HaMeR 低频修正和质量感知融合属于
P6。真实数据仍需确认动态手指不会污染掌部刚体假设，并统计深度边缘、遮挡、运动模糊、
ROI 跳变和大角速度下的失败率。

## NOT RUN / 等待 USB3

```text
真实 development_usb2 手部序列离线运行      NOT RUN
实时 USB2 D455 + P3 ROI                    NOT RUN
真实掌部 RANSAC 内点率/RMS/重初始化率       NOT RUN
遮挡、深度边缘和运动模糊实测                NOT RUN
DEV_USB2_G00～G09 P5评价                   NOT RUN
USB3短录/长录上的正式P5评价                 WAITING FOR USB3
正式G00～G09和最终端到端频率/延迟            WAITING FOR USB3
```
