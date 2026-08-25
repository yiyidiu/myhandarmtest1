# 交叉评审决策

更新日期：2026-08-13（Asia/Shanghai）

## RD-001：P0 的状态措辞

**争议：** 工程能够构建并启动，是否可写成“P0 通过”。

**决策：** 只写“P0 现状执行基线已记录，P0 验收 `NOT PASS`”。构建、XML、URDF、
节点和控制器的局部 PASS 保留，但不能外推为正式遥操作或安全通过。独立对抗审查确认
0 个 Catkin 自动测试、旧控制链 NaN/Inf 可穿透、UDP 无乱序/时间倒退防护、manifest
漏依赖和碰撞关闭。详见 `docs/00_CURRENT_SYSTEM_AUDIT.md` 和
`docs/02_BASELINE_TEST_REPORT.md`。

## RD-002：旧 MediaPipe 姿态链

**争议：** 是否可继续调整死区/低通/RANSAC，使旧脚本成为正式感知。

**决策：** 否。旧脚本只保留为 A 方案基线。其
`R_delta=(A R0)^T(A Rn)=R0^T Rn` 会代数抵消映射矩阵；正式链必须使用手相对旋转和
hand-to-tool 共轭映射。证据：
`src/handarm_moveit_demo/scripts/d455_conda_udp_sender_servo_v3.py:514-528`。

## RD-003：HaMeR 6 GB 运行架构

**争议：** 是否安装/常驻官方完整 demo 依赖以换取 `pip check` 无警告。

**决策：** 正式运行路径仅加载 HaMeR + MANO，bbox 外部提供，模型构造使用
`init_renderer=False`。不安装/不常驻 Detectron2、ViTDet、ViTPose、mmcv 和
xtcocotools；用 `opencv-python-headless` 替代 GUI OpenCV。官方 `hamer` 包元数据仍把
这些完整 demo 依赖列为必需，所以 `pip check` 会报告 5 个缺项：detectron2、mmcv、
opencv-python、pandas、xtcocotools。这是已知元数据差异，不应隐瞒，也不能被写成环境
完整性 PASS。crop-only 导入和单元测试需独立作为实测证据。

## RD-004：PyTorch/CUDA 锁定

**争议：** 驱动显示 CUDA 12.8，是否应安装最新 PyTorch/CUDA 13 依赖。

**决策：** 否。第一次解析确实尝试下载 CUDA 13/cuDNN 9.20，已中止并删除不完整环境。
重建环境锁定官方 HaMeR Docker 使用的 torch 2.2.0、torchvision 0.17.0 和 cu118。
RTX 2060 上实测 CUDA 可用、cuDNN 8.7、FP16 有限值测试通过。

## RD-005：显存验收口径

**争议：** torch 报告 5731.125 MiB，而 `nvidia-smi` 报告 6144 MiB，85% 应取谁。

**决策：** 按用户要求，以 `nvidia-smi` 的物理总显存 6144 MiB 为正式分母，上限为
5222.4 MiB，并以 `nvidia-smi` 系统总峰值判定；同时单独记录 torch reported total、
allocated 和 reserved，便于诊断但不替代总占用门槛。

## RD-006：MANO 和 P1 阶段门槛

**争议：** 缺少 `MANO_RIGHT.pkl` 时能否用占位或替代模型跑通接口。

**决策：** 否。该文件需用户从 MANO 官方渠道按许可证取得。crop API 对 checkpoint、
model config、mean parameters 和 MANO 逐项硬校验；缺任一项必须 fail closed。实际
MANO 输出、FP32/FP16 HaMeR 显存、FPS 和 OOM 结果保持 `NOT RUN`，P1 不得通过。

## RD-007：HaMeR+ROI 基准标签

**争议：** 传入固定外部 bbox 能否标记为“HaMeR + ROI 模块”。

**决策：** 否。在 P3 ROI provider 实现前，基准脚本主动拒绝 `hamer_roi` 场景，避免
把预计算 bbox 冒充联合模块测试。该场景保持 `NOT RUN`。

## RD-008：左手 HaMeR 几何契约

**争议：** 左手整图镜像后的 affine、MANO 点集和 `global_orient` 能否都直接称为原
D455 相机坐标输出。

**决策：** 不能。crop API 返回的 affine 必须复合原图到镜像图的变换，并固定为
`affine_original_to_crop`；请求 bbox 与可见 bbox 分开返回，越界 padding 不得静默改变
crop 中心/尺度。左手的 vertices/joints 另以 x 反射恢复到 source-camera axes，同时保留
MANO_RIGHT canonical 原始点集。反射不得单边乘到旋转矩阵上；`global_orient` 和
`hand_pose` 只标记为 MANO native prior，严格校验 shape、正交性及 `det(R)=+1`，不得
作为 D455 掌姿态或机器人控制姿态。P3 必须从掌部刚性点集构造有明确定义的
`R_C_H`。

## RD-009：Servo safe profile 与输出端 watchdog

**争议：** 仅把 MoveIt Servo 的 `check_collisions` 改为 true，是否足以形成 safe
profile。

**决策：** 否。当前 `JointGroupVelocityController` 会持续执行 realtime buffer 中最后
一条速度命令，Servo 进程失联不等同于 controller 收到零速。正式 safe 链除碰撞检查、
合理奇异阈值、0.10 s 输入超时、状态码锁存和 supervisor 限速外，还必须有 controller
侧或独立 watchdog/停机路径，并执行 Servo `SIGKILL`/发布者消失故障测试。在该测试
通过前，现有速度链仅为 Gazebo 基线，真实 ABB 禁止连接。

## RD-010：三指手模型单一事实源

**争议：** `hand_g.xacro` 能否视为当前 Gazebo 手模型的生成源，命名的 `grasp*` 姿态
能否视为抓取能力。

**决策：** 均不能。Gazebo 实际直接加载两份静态 URDF；它们含四个 mimic 插件，而
`hand_g.xacro` 不含，模型已经漂移。命名姿态只证明四个主动关节的位置目标存在，
不证明张开/闭合语义、接触、保持或 lift 成功。P11 前必须统一生成源、单轴点动确认
四主动/四 mimic 物理方向，并在正常重力下建立多指接触、抬升、滑移及失败超时
oracle。无 PID 的 mimic `SetPosition` 路径必须先通过接触穿透与能量稳定性测试。

## RD-011：P1 通过口径与 FP16 显存余量

**争议：** HaMeR+ROI 尚未实现时，P1 是否必须保持整体阻塞；FP16 较快是否意味着它
也必然占用更少显存。

**决策：** P1 的 crop-only 核心验收可独立判定：真实 checkpoint/MANO 输出、batch 1、
FP32/FP16、renderer 禁用、HaMeR-only 和 Gazebo headless 显存均已实测通过。ROI provider
属于 P3，联合格明确记为 `DEFERRED TO P3`，不得用固定 bbox 冒充，也不反向否定已通过
的 crop-only 模型门禁。

本机实测 FP16 比 FP32 快，但 CUDA reserved 和系统峰值更高；最坏 FP16+Gazebo 为
4962/6144 MiB，仅比 85% 门槛低 260.4 MiB。结论采用实测而非“FP16 必然省显存”的
假设：默认仍选 FP16 获得约 24 FPS 的模型吞吐，但 HaMeR 调度先固定 5 Hz，并禁止
Gazebo GUI、RViz、GPU detector 或其他图形负载并存。任何新增 GPU 模块必须重新测总
峰值。

## RD-012：D455 USB2 短流的通过口径

**争议：** 640×480@30 在 USB 2.1 Hub 上短时无丢帧，是否可以判定 P2 通过。

**决策：** 否。设备枚举、对齐和 recorder 只记 `SMOKE PASS`。当前 SDK 明确报告 USB
2.1/480M，且内核持续出现 UVC `GET_CUR -32`；固定几十帧无丢帧不能外推到 G00–G09
或 10–30 分钟正式录制。正式数据采集必须 USB 3.x 直连，重新验证内核日志和长录。

为保留可重放性，数据集同时保存原始 Z16 与 aligned-to-color Z16，以及两套内参、
depth→color 外参、深度尺度、双设备时间戳/domain、host monotonic/wall、传感器选项和
逐文件哈希。`global_time` 不等同 host monotonic/ROS time；尚未实现的 ROI/HaMeR/KLT
字段必须显式 `NOT_RUN`。

用户确认线长限制后，补充允许显式 USB2 降级开发模式：默认 recorder 仍要求
SuperSpeed，只有 `--allow-usb2` 才能在当前链路启动。manifest 分开记录
`data_integrity_pass` 与 `deployment_link_pass`；短录数据完整时可标记
`DEGRADED_USB2_ACCEPTED_FOR_DEVELOPMENT`，但总 `quality_pass=false`，不得用于宣称
USB3 部署门禁或完整 P2 已通过。
