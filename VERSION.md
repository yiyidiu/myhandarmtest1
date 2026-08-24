# v0.3.0-egm-stable-hold

重建自 2026-08-24 18:31:22（Asia/Shanghai）的稳定 EGM 位置参考检查点。

## 冻结时证据

- 87 项 Python 测试通过；
- ROS launch 解析和 profile 参数注入通过；
- 无头 Gazebo 外环 50 Hz、位置参考 250 Hz；
- 丢失输入后六轴命令参考跨度为 0 rad；
- 重新获取输入时从 `POSITION_HOLD` 回到 `TRACKING`，无参考跳变。

## 原始冻结包

`egm_stable_stiff_hold_passed_20260824_183122.tar.gz`

SHA-256：`7b9ea411fddf5beaf5a427ac81576f499396dce17d6c343a21eba154871cc860`

原包只有 15 个最终变更文件。本 Git 版本以最近的完整 EGM 源码快照为基础叠加这些文件，并补齐共同依赖。
