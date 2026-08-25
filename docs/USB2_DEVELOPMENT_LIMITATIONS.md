# USB2 开发证据边界

状态：`P2_DEVELOPMENT_GATE=PASS`，`P2_FORMAL_USB3_GATE=DEFERRED_USB3_UNAVAILABLE`。

当前 D455 通过 480 Mbit/s Hub 以 descriptor `2.1` 枚举。已验证数据完整性、彩深同步、
时间戳、写盘和离线回放，因此 `development_dataset_eligible=true`、
`phase_progression_allowed=true`，证据范围固定为 `USB2_DEVELOPMENT_ONLY`。

这些证据允许开发 P3 以后算法、离线回放、ROS/Gazebo 和故障注入；不支持 USB3 部署、
正式数据集、最终 30 Hz/延迟、正式深度精度、正式彩深边缘对齐或真实部署稳定性结论。
所有 USB2 数据位于 `datasets/development_usb2`，禁止复制或重命名为未来的
`datasets/formal_usb3` 数据。

永久索引：[DATASET_INDEX.json](../datasets/development_usb2/DATASET_INDEX.json)。
Profile 探测：[D455_PROFILE_PROBE.json](../datasets/development_usb2/D455_PROFILE_PROBE.json)。

已知限制：USB2 内核曾报告 UVC `GET_CUR -32`；当前 Profile 短时可靠不等价于长时或
USB3 可靠。ROI/掌坐标/Kabsch 的开发通过也必须在 USB3 正式数据上重新认证。
