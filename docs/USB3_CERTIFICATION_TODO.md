# USB3 恢复后的正式认证

当前状态：`DEFERRED_USB3_UNAVAILABLE`，未运行，未伪造结果。

入口脚本：[run_usb3_certification.sh](../scripts/run_usb3_certification.sh)。脚本默认拒绝
运行；只有显式设置 `RUN_USB3_CERTIFICATION=YES` 且实时 descriptor 为 3.x 才能进入。

认证顺序：

1. 确认 USB descriptor 和物理拓扑为 SuperSpeed；
2. 枚举并逐一短测 USB3 候选 Profile；
3. 300 帧正式短录；
4. 10～30 分钟正式长录；
5. 正式深度 bias/MAD/P95；
6. 正式彩深边缘对齐；
7. 重新录制正式 G00～G09；
8. 正式端到端频率和延迟；
9. USB3 故障回归；
10. 离线验证和正式数据集验收。

任一项不满足都不得设置 `deployment_link_pass=true`、
`formal_dataset_eligible=true` 或 `P2_FORMAL_USB3_GATE=PASS`。
