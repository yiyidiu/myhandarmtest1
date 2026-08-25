# P2 D455 采集与数据录制

状态：**P2_DEVELOPMENT_GATE=PASS；P2_FORMAL_USB3_GATE=DEFERRED_USB3_UNAVAILABLE**  
日期：2026-08-13  
设备：Intel RealSense D455，serial `234322305987`，firmware `5.17.3.10`

## 当前结论

D455 已枚举，Python SDK 能够启动 640×480@30 的 RGB8+Z16，深度对齐到彩色成功。
实现了严格采集 API 和可回放 recorder，并完成 30/60/300 帧真实落盘回读以及 900 帧
纯采集稳定性测试。稳定窗口后没有帧号跳变；900 帧最大同步偏差低于 0.05 ms。相机
仍经 480M Hub 以 USB 2.1 连接，内核持续报告 UVC `GET_CUR -32`，因此允许继续开发和
短录，但只能通过显式 `--allow-usb2` 进入降级模式，不能替代 USB3 部署门禁或完整
G00–G09 数据集。

## 设备与软件实测

```text
USB ID                         8086:0b5c
USB path                       1-6.4，经 Genesys Logic 480M Hub
SDK descriptor                 2.1（不是 SuperSpeed）
pyrealsense2                   2.58.2.10647，仅 mediapipe_env
realsense2_camera ROS package  NOT INSTALLED
depth scale                    0.0010000000474974513 m/unit
active profile                 depth Z16 640x480@30 + color RGB8 640x480@30
timestamp domain               global_time（不得冒充 host monotonic/ROS time）
```

相机含 Stereo Module、RGB Camera、Motion Module；正式连续位姿不会使用 IMU。Session
manifest 会保存所有可读 sensor options，但本轮未写相机选项、未升级固件。

## 新增实现

- `perception_hamer/src/d455_capture.py`
  - 强制选择唯一 D455/可指定 serial；
  - 同时返回原始 RGB、原始 Z16 和 align-to-color Z16；
  - 保存 RGB/depth 各自 frame number、设备 timestamp/domain，以及 host monotonic/wall；
  - 保存 raw-depth/color intrinsics、depth→color SE(3)、depth scale、设备/SDK/FW/USB；
  - 连续帧号、时间戳和 RGB-depth skew 的稳定窗口门禁；
  - recorder 默认要求 SuperSpeed；当前连接只有显式 `--allow-usb2` 才接受并标记降级；
- `perception_hamer/scripts/record_rgbd_session.py`
  - 场景白名单 G00–G09；
  - RGB、raw Z16、aligned Z16 均用无损 PNG；
  - 有界异步 writer 保持采集线程 30 Hz；队列溢出/worker 错误直接失败，不静默丢帧；
  - 每个文件保存 SHA-256，逐帧 JSONL 与 session manifest 原子写入；
  - 设备/主机节拍、时间域恒定、计划帧数、writer 完整性均为硬门禁；
  - 尚未运行的 ROI/HaMeR/KLT 显式写 `NOT_RUN`；
  - 中断/失败保留 `INCOMPLETE` manifest，拒绝覆盖已有 session。
- `perception_hamer/scripts/verify_rgbd_session.py`
  - 离线逐帧复核 canonical/唯一文件路径、SHA-256、PNG dtype/shape、标定与常量；
  - 从原始字段重建派生元数据并重算 summary，拒绝伪造 USB 标志、复用路径和孤儿文件；
  - `verification.json` 绑定 `session.json` 与 `frames.jsonl` 的 SHA-256；
  - USB3 候选只有离线验证后才可能获得正式数据资格。

## 真实短录结果

直接流探针在丢弃预热后连续采集 90 帧：

```text
depth/color frame drops       0 / 0
period median                 ~33.28 ms
period P95                    ~33.32 ms
max RGB-depth skew            0.0491 ms
aligned intrinsics match      PASS
mean valid-depth fraction     0.8686
```

完整 recorder contract 的 30 帧 G00 smoke：

```text
status                         COMPLETE
RGB/raw depth/aligned depth    30 / 30 / 30
file checksum spot check       PASS
depth/color frame drops        0 / 0
period median                  ~33.09 ms
period P95                     ~33.31 ms
max RGB-depth skew             0.0491 ms
mean/min valid-depth fraction  0.8600 / 0.8578
USB SuperSpeed                 false
```

临时验证数据位于 `/tmp/d455_p2_owftUK/`，不属于正式 G00 数据集。早期 manifest 的
`quality_pass=true` 只表示当时实现中的帧连续性和同步门禁；新 schema 已将数据完整性
与 USB 部署门禁分开，USB2 总 `quality_pass=false`。两者都不覆盖深度准确度、KLT
亮度一致性或长录可靠性。

## USB2 约束下的重复测试

在用户确认线长限制后，按 USB2 已知约束重新执行：

- start/stop 3 次，每次稳定窗口后 30 帧：三次均 0 丢帧，max skew 约 0.049 ms；
- 完整三流 recorder 300 帧：RGB/raw/aligned 各 300 张，0 帧号跳变，抽样哈希通过，
  max skew 1.148 ms；周期中位约 34.42 ms（约 29.1 Hz），期间新增 74 条 UVC 告警；
- 无写盘纯采集 900 帧/30.01 s：0 丢帧、0 时间倒退、29.987 FPS，host period P95
  34.105 ms，max skew 0.0493 ms；期间新增 53 条 UVC 告警，无 USB reset/disconnect。

300 帧测试时场景变化导致 valid-depth fraction 约 0.258，这不是 USB 丢帧指标，也不
代表手掌深度质量通过；后续必须在固定已知距离和平面/手部 ROI 内评价深度 bias、MAD
与空洞率。

新 manifest 将两个概念分开：`data_integrity_pass` 表示帧连续/同步，
`deployment_link_pass` 表示 USB SuperSpeed。USB2 数据完整但仅用于开发时写
`session_acceptance=DEGRADED_USB2_ACCEPTED_FOR_DEVELOPMENT`，总 `quality_pass=false`。

300 帧临时 session 位于 `/tmp/d455_usb2_retry_fri1ET/static_300_retry2`（约 201 MiB）；
新 schema 的 30 帧门禁复测位于 `/tmp/d455_usb2_schema_YwSUnl/allowed_30`。两者均为
硬件/recorder 验证数据，不冒充包含静止手势的正式 `G00_STATIC`。

当前链路的明确运行方式：

```bash
cd /home/diu/myhandarmtest1
env -u PYTHONPATH -u ROS_PACKAGE_PATH \
  conda run --no-capture-output -n mediapipe_env \
  python perception_hamer/scripts/record_rgbd_session.py \
    --scenario G00_STATIC \
    --session-name G00_STATIC_take01 \
    --frames 300 \
    --allow-usb2
```

### 最终 schema v2 异步录制实测

在当前 USB2 连接上用最终字段和离线 verifier 重新执行 300 帧：

```text
session status                  CAPTURE_COMPLETE_UNVERIFIED
offline verification           PASS（300/300，逐文件 hash + decode）
depth/color frame drops         0 / 0
max RGB-depth skew              0.049316 ms
device frame-rate gate          PASS
host delivery cadence gate      PASS
writer queue max/capacity       1 / 64
writer queue overflows          0
writer service mean/P95         31.196 / 32.892 ms
capture data_integrity_pass     true
deployment_link_pass            false
formal_dataset_eligible         false
acceptance                      DEGRADED_USB2_ACCEPTED_FOR_DEVELOPMENT
```

临时会话：`/tmp/d455_async_gate_20260813_c`。其 `verification.json` 同时记录 manifest
和 JSONL 哈希。上一轮使用容量 16 且逐帧 JSONL fsync 的故障注入在第 145 帧触发 queue
overflow，正确留下 `INCOMPLETE`；它证明溢出门禁有效，但不是通过数据。

最终 schema 的回放命令：

```bash
conda run -n mediapipe_env \
  python perception_hamer/scripts/verify_rgbd_session.py \
    /path/to/session --write-result
```

去掉 `--allow-usb2` 时当前连接会返回非零并留下 `INCOMPLETE` manifest，这是预期的
SuperSpeed 默认门禁。

## 当前阻塞与剩余验收

1. 将 D455 用合格 SuperSpeed 线直接连接主机 USB 3.x 端口，绕过当前 480M Hub；
2. 确认 SDK `usb_type_descriptor` 为 3.x，且内核不再刷 UVC `GET_CUR -32`；
3. 执行 start/stop 10 次和 10–30 分钟 640×480@30 长录；掉帧率目标 `<0.1%`；
4. 用已知距离平面测 depth bias、MAD、P95，并做对齐边缘 P50/P95；
5. 录制 G00–G09，每场景独立 manifest、重复次数和质量报告；
6. 硬件无关故障注入 24/24 PASS：frameset timeout、SIGINT/SIGTERM、writer、mock
   ENOSPC、队列溢出、PNG/manifest 失败、时钟/帧号倒退、彩深不同步、路径冲突和异常
   退出；详见 `docs/04_P2_FAULT_INJECTION_REPORT.md`。物理拔线、真实 ENOSPC 仍 NOT RUN；
7. 如采用 ROS 相机话题，再安装并锁定 `realsense2_camera` 后独立验收。

USB2 开发数据允许继续 P3 及后续算法/仿真开发；所有正式 USB3 数据、性能和部署结论
继续冻结，且不连接真实 ABB。
