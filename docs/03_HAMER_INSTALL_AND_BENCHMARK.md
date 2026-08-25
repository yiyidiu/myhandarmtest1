# P1 HaMeR 安装与显存基准

状态：**P1 crop-only 验收 PASS；ROI 联合格 DEFERRED TO P3**  
日期：2026-08-13  
GPU：NVIDIA GeForce RTX 2060，6144 MiB，compute capability 7.5  
驱动：570.133.20

## 当前结论

P1 的 crop-only 验收条件已经满足：独立环境、固定 HaMeR 提交、checkpoint、
`MANO_RIGHT.pkl`、真实左右手 MANO 输出、batch 1 FP32/FP16、HaMeR-only 及 Gazebo 真
headless 并存测试均已实际运行；没有 OOM、没有构造 renderer，四个正式 30 帧场景的
`nvidia-smi` 总峰值均低于 5222.4 MiB。HaMeR+ROI 仍按 RD-007 延后到 P3：当前没有
ROI provider，脚本会拒绝把固定 bbox 冒充联合测试。

官方完整 `demo.py` 因依赖并常驻 detector/ViTPose/renderer，未作为正式链运行；本轮
改用官方提交中的 `example_data/test1.jpg` 和人工 bbox 实测相同 checkpoint/MANO 的
crop-only 输出。该样例验证只证明模型能够对真实图像产生有限 MANO 输出，不构成姿态
精度或数据集泛化结论。

## 官方来源复核

官方仓库：`https://github.com/geopavlakos/hamer`  
审计提交：`3a01849f4148352e9260b69bf28b65d1671a4905`

官方 `demo.py` 会组合 HaMeR、人体检测器、ViTPose 和渲染器，不适合 6 GB 常驻链。
本工程的 `perception_hamer/src/hamer_crop_inference.py` 只加载 HaMeR + MANO，构造
模型时传入 `init_renderer=False`，bbox 由外部 ROI 模块提供。

官方模型归档 HTTP `Content-Length` 与下载文件大小均为 6,037,554,929 bytes；归档
SHA-256 为 `ccfb70abd672b64c3ea90891c808d4499cc36a37dd6cf86c561a665113aef11e`。
完整遍历 12 个 tar 路径，绝对路径和 `..` 穿越项为 0。归档还包含 3.8 GB ViTPose
权重，本 crop-only 路径未将其解出。MANO 模型需用户
在 MANO 官方网站注册并按许可证取得，不能由本工程伪造或替代。
下载归档暂保存在 `perception_hamer/_DATA/hamer_demo_data.tar.gz` 以便复核与续跑；
`_DATA/` 已被 `.gitignore` 排除，不得进入最终清洁源码包。

## P0 环境实测

```text
conda 25.11.1
hamer_rtx2060: Python 3.10.20
torch 2.2.0+cu118 / torchvision 0.17.0+cu118
torch built CUDA 11.8 / cuDNN 8.7
torch.cuda.is_available(): true
HaMeR commit 3a01849f... installed crop-only with --no-deps
nvcc: NOT INSTALLED
CUDA Toolkit/cuDNN system packages: NOT FOUND
HaMeR checkpoint: READY / SHA-256 e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53
model_config.yaml: READY / SHA-256 0e5eeb82752e47dfd01db8e13ccc4c5eba9bf83f53da8285523b8d3e87247aa3
mano_mean_params.npz: READY / SHA-256 efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988
MANO_RIGHT.pkl: READY / 3,821,356 bytes / SHA-256 45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767
```

`nvidia-smi` 显示的 CUDA 12.8 是驱动支持上限，并不表示 CUDA Toolkit 已安装。
预编译 cu118 PyTorch wheel 可不依赖本地 nvcc；若将来某依赖必须本地编译 CUDA
扩展，缺少 Toolkit 将成为额外阻塞。

CUDA 烟雾测试实际结果：RTX 2060 compute capability 7.5；FP16 256×256 矩阵结果为
有限值；torch peak allocated 8.8125 MiB、reserved 22 MiB。该测试只验证运行时和
autocast，不是 HaMeR 模型显存或速度结果。

官方 `hamer` 元数据将完整 demo 的 detectron2/mmcv/opencv-python/pandas/xtcocotools
列为必需。低显存 crop-only 环境刻意省略这些依赖并使用 headless OpenCV，所以
`pip check` 当前报告这 5 项缺失；`from hamer.models import HAMER` 已实测导入成功。
该取舍见 `docs/REVIEW_DECISIONS.md` RD-003，不能把 `pip check` 写成 PASS。

## 显存门槛

```text
实际总显存                6144 MiB
85% 上限                  5222.4 MiB
审计期间桌面/GUI占用       620–981 MiB
按981 MiB基线剩余至上限    4241.4 MiB
```

最终判据使用 `nvidia-smi` 的系统总峰值，不只看 torch allocated。建议关闭浏览器、
gzclient、RViz 和所有渲染后再测。不得根据网络结构推测 30 Hz。

## 实现文件

- `perception_hamer/src/hamer_crop_inference.py`：RGB+bbox+handedness crop API；
- `perception_hamer/scripts/benchmark_hamer.py`：allocated/reserved/nvidia-smi/延迟/OOM；
- `perception_hamer/configs/rtx2060_realtime.yaml`：6 GB 默认配置；
- `perception_hamer/environment/hamer_rtx2060.yml`：独立 Python 3.10 依赖规格；
- `perception_hamer/tests/test_hamer_crop_inference.py`：输入、非对称左手 affine、边界
  padding、NaN/Inf、SO(3) 与资产门控。

## Crop API 几何边界

交叉审查后，接口固定了以下不可隐式转换的约定：

- bbox 是原 RGB 图上的连续半开区间 `[x1,y1,x2,y2)`；请求框用于 crop，裁剪到图像
  的 visible bbox 只用于质量门禁，两者分别返回；
- `affine_original_to_crop` 始终从原图映射到 crop，左手时已复合整图 x 镜像；
- 左手 vertices/joints 同时保留 MANO_RIGHT canonical 原值，并提供恢复 x 轴后的
  source-camera-axes 点集；反射绝不单边应用到旋转矩阵；
- `global_orient` 固定为 `(3,3)`、`hand_pose` 固定为 `(15,3,3)`，逐块检查正交性和
  `det(R)=+1`；两者只是 MANO native prior，不是 D455 掌姿态；
- HaMeR camera/translation/focal 字段机器可读地标为 crop projection only，既不是
  D455 内参，也不得作为公制掌心位置；
- timestamp 必须由调用方传入 RGB capture time；接口同时携带 clock domain 与 RGB
  source frame 标识，不能用推理完成时刻冒充采集时刻。

系统 Python 3.8 与 `hamer_rtx2060` Python 3.10 的单元测试均为
`12/12 PASS`，包含 dummy-model `infer()` 成功返回契约。这仍只覆盖接口/预处理契约、
返回结构和 fail-closed 资产门禁。真实 checkpoint/MANO 的左右手成功路径随后独立
运行并通过：vertices `(778,3)`、joints `(21,3)`、global orientation `(3,3)`、hand pose
`(15,3,3)`、betas `(10,)` 均有限；全部旋转块 `det(R)` 接近 +1。模型内部
`renderer is None` 且 `mesh_renderer is None`。

## 实测验收矩阵

所有正式计时均为 batch 1、warmup 5、30 次，性能输入为 synthetic constant crop；它只
用于稳定测量性能和显存，不可用于精度结论。

| 场景 | 精度 | 成功 | 系统峰值 | allocated / reserved | mean / median / P95 | FPS | OOM | 85% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HaMeR only | FP16 | 30/30 | 4852 MiB | 3926.2 / 4162 MiB | 41.00 / 40.96 / 41.81 ms | 24.39 | 否 | PASS |
| HaMeR only | FP32 | 30/30 | 3491 MiB | 2653.4 / 2716 MiB | 71.87 / 71.79 / 72.82 ms | 13.91 | 否 | PASS |
| HaMeR + Gazebo headless | FP16 | 30/30 | 4962 MiB | 3926.2 / 4162 MiB | 40.27 / 40.18 / 41.06 ms | 24.83 | 否 | PASS |
| HaMeR + Gazebo headless | FP32 | 30/30 | 3514 MiB | 2653.4 / 2716 MiB | 70.19 / 69.91 / 71.62 ms | 14.25 | 否 | PASS |
| HaMeR + ROI | FP16/FP32 | 0 | — | — | — | — | — | DEFERRED TO P3 |

Gazebo 测试由 `gazebo_ros empty_world.launch gui:=false headless:=true` 启动，确认
`gzserver` 和 `/clock` 存在，同时确认无 gzclient/RViz；两种精度完成后正常关闭，未留
ROS/Gazebo 进程。FP16+Gazebo 是最接近门槛的实测场景：4962 MiB，占物理显存
80.76%，距离 85% 门槛仅 260.4 MiB。因此正式运行仍禁止同时打开 Gazebo GUI、RViz、
浏览器 GPU 页面或 GPU detector。配置继续保守采用 5 Hz HaMeR 调度，不承诺 30 Hz。

额外真实图像检查使用官方固定提交 `example_data/test1.jpg`，SHA-256
`8ee090a3f1521367e7bdc320a6ed2cbae1f08dff842cb319d816aabafaab9263`，人工右手 bbox
`[245,145,390,345]`。FP16 10/10 成功，mean 41.44 ms、P95 42.18 ms、24.13 FPS、
系统峰值 4939 MiB、无 OOM。该 bbox 为人工离线标注，不是 ROI provider 输出。

## P1 门禁与下一步

P1 crop-only 门禁判定为 `PASS`：真实 MANO 参数已输出、batch 1 无 OOM、renderer 未
构造、实际显存与延迟已记录且低于 85%。ROI 联合基准不是固定 bbox 的别名，将在 P3
实现真实 ROI provider 后补测。

P1 完成后 D455 已连接并完成短流/录制 smoke，但相机当前经 480M Hub 以 USB 2.1
连接，且 `realsense2_camera` ROS 包仍缺失。P2 现状及阻塞见
`docs/04_D455_CAPTURE_AND_RECORDING.md`。未连接真实 ABB。
