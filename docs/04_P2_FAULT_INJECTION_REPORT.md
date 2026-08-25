# P2 安全故障注入报告

日期：2026-08-13（Asia/Shanghai）  
范围：D455 recorder、异步 writer、离线 verifier 和 P2 双门禁  
执行约束：**未打开相机、未启动 Gazebo、未填充真实文件系统**

## 结论

硬件无关故障注入套件实际运行 **24/24 PASS**；当前全仓 Python 测试实际运行
**100/100 PASS**。全部可捕获 recorder 失败都会留下 `status=INCOMPLETE` 和结构化
`failure_reason`，且离线 verifier 拒绝该 session。不可捕获的 `os._exit(23)` 由录制
开始前原子写入的 `PROCESS_DID_NOT_COMPLETE` 默认失败原因覆盖。

本结果只证明软件 fail-closed 行为，不代表物理拔线、真实磁盘耗尽、USB3 或长录通过。
机器可读结果位于 `perception_hamer/test_results/p2_fault_injection.json`。

## 实际执行

```bash
cd /home/diu/myhandarmtest1
python3 perception_hamer/scripts/run_p2_fault_injection.py \
  --json-report perception_hamer/test_results/p2_fault_injection.json

python3 -m unittest discover \
  -s perception_hamer/tests -p 'test_*.py' -q

python3 perception_hamer/scripts/verify_rgbd_session.py \
  datasets/development_usb2/p2_smoke
python3 perception_hamer/scripts/verify_rgbd_session.py \
  datasets/development_usb2/p2_gate
```

结果：故障套件 24 tests / OK；全仓 100 tests / OK；永久归档的 30 帧 smoke 和
300 帧 gate 均重新验证为 `verification=PASS`、`p2_development_gate=PASS`、
`p2_formal_usb3_gate=DEFERRED_USB3_UNAVAILABLE`、`formal_P2_pass=false`。

## 覆盖矩阵

| 故障 | 方法 | 结果 | 核心断言/证据 |
|---|---|---|---|
| frameset timeout | mock pipeline 抛出 timeout；不导入 SDK 设备 | PASS | `test_p2_fault_injection.py:174-197`；manifest 为 `INCOMPLETE/FRAMESET_TIMEOUT` |
| 相机拔线 | 注入 `device disconnected` | MOCK PASS | `test_p2_fault_injection.py:198-211`；**物理拔线 NOT RUN** |
| SIGINT / SIGTERM | 向只含 mock capture 的真实子进程发送信号 | PASS | `test_p2_fault_injection.py:635-650`；`p2_fault_signal_child.py:83-124` |
| writer 异常 | writer 线程注入 `RuntimeError` | PASS | `test_p2_fault_injection.py:215-248`；worker 错误传播并拒绝 session |
| ENOSPC | writer 显式抛 `OSError(errno.ENOSPC)` | MOCK PASS | `test_p2_fault_injection.py:250-271`；测试目录写入量小于 100 KB；真实 ENOSPC NOT RUN |
| 队列溢出 | 容量 1 的 writer + 可控阻塞 save | PASS | `test_p2_fault_injection.py:273-329`；overflow 计数大于零，原因 `WRITER_QUEUE_OVERFLOW` |
| PNG 写入失败 | mock `cv2.imwrite=False` | PASS | `test_p2_fault_injection.py:331-353`；原因 `PNG_WRITE_FAILED` |
| manifest 写入失败 | 分别注入初始/最终 manifest 写异常 | PASS | `test_p2_fault_injection.py:357-399`；best-effort `INCOMPLETE` 标记保留 |
| 时间戳倒退 | 修改第二帧设备时间戳 | PASS | `test_p2_fault_injection.py:437-443`；data integrity=false |
| 帧号倒退 | 修改第二帧 RGB/depth frame number | PASS | `test_p2_fault_injection.py:429-435`；data integrity=false |
| 彩深不同步 | 注入 3 ms skew，门限 2 ms | PASS | `test_p2_fault_injection.py:445-455`；data integrity=false |
| 重复输出路径 | 篡改 JSONL，使两帧复用同一 RGB 路径 | PASS | `test_p2_fault_injection.py:457-518`；offline verifier 拒绝 |
| 已有帧文件冲突 | 预置 sentinel PNG | PASS | `test_p2_fault_injection.py:401-411`；sentinel 未被覆盖 |
| 已有 session 冲突 | 预置 session 目录和 sentinel | PASS | `test_p2_fault_injection.py:413-422`；目录内容未被修改 |
| 进程异常退出 | mock 子进程执行 `os._exit(23)` | PASS | `p2_fault_signal_child.py:116-119`、`test_p2_fault_injection.py:652-666`；预置失败原因保留 |
| 文件哈希篡改 | 修改已录 PNG | PASS（回归） | `test_d455_capture.py:371-456`；offline verifier 拒绝 |

## Recorder 加固

- `RecordingSignalGuard` 把 SIGINT/SIGTERM 转成可收尾的 `RecordingInterrupted`：
  `record_rgbd_session.py:42-81`。
- 结构化失败码覆盖 signal、ENOSPC、manifest、已有文件、队列、PNG、writer 和 timeout：
  `record_rgbd_session.py:90-121`。
- 每帧写入前检查三个目标 PNG，不覆盖已有文件：
  `record_rgbd_session.py:133-149`。
- `record()` 支持注入 capture/writer/manifest writer，生产默认值不变：
  `record_rgbd_session.py:588-595`。
- 初始 manifest 预置 `PROCESS_DID_NOT_COMPLETE`；正常完成时移除；捕获失败时更新明确原因：
  `record_rgbd_session.py:611-628`、`:700-748`。

## 双门禁对抗审查

1. USB2 即使 `data_integrity_pass=true`，也只能得到开发门禁 PASS；测试断言
   `deployment_link_pass=false`、`formal_dataset_eligible=false`、
   `formal_P2_pass=false`：`test_p2_fault_injection.py:520-536`。
2. 单个模拟 USB3 session 即使数据完整且 offline verification PASS，也只得到
   `USB3_SESSION_CANDIDATE`，全局 formal gate 保持 pending：
   `test_p2_fault_injection.py:538-562`。
3. 把 manifest 中 `formal_P2_pass` 和 `formal_dataset_eligible` 篡改为 true 会被拒绝：
   `test_p2_fault_injection.py:593-606`。
4. verifier 兼容不可变的早期 schema-v2 真实归档，但只允许旧记录成为新版重算结果的
   严格投影；已有字段、布尔门禁、整数、路径和列表不能变化，仅容忍浮点跨 NumPy 的微小
   重算差以及新增派生字段：`verify_rgbd_session.py:31-117`、`:323-335`。
5. 全局正式门禁不会由单 session verifier 置 PASS：
   `record_rgbd_session.py:555-568`、`verify_rgbd_session.py:337-372`。

## NOT RUN 与剩余风险

- **物理相机拔线：NOT RUN。** 按要求未占用或拔除 D455；仅完成等价异常路径 mock。
- **真实 ENOSPC/受限挂载：NOT RUN。** 当前环境未建立独立受限文件系统；采用显式
  `errno.ENOSPC` 注入，未向根分区或用户目录写填充数据。
- **SIGKILL：NOT RUN。** `os._exit(23)` 已覆盖不可捕获异常退出后的磁盘状态；未对任意
  非测试进程发送 SIGKILL。
- **USB2 10/30 分钟 soak：NOT RUN（本任务未占用相机）。**
- **USB3 故障回归：等待 USB3。**
- 真实存储设备在 manifest 自身写入期间发生 ENOSPC 时，任何软件都可能无法追加最新
  错误详情；因此 recorder 在采集开始前先落盘默认失败原因，保证残留 session 不会被
  误认为完成。
