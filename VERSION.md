# 当前保存基线：2026-09-14

控制与视觉实现固定于 `dac3518`。之后的保存工作仅修正启动环境选择、从全新终端加载 ROS 以及旧版说明，不调整运动或姿态参数。

本机独立运行目录：`/home/dongtian/teleop_versions/20260914_dac3518/`。

```bash
bash /home/dongtian/teleop_versions/20260914_dac3518/run.sh
```

确切源码提交和环境清单见保存目录内的 `manifest.json`；完整恢复包仅在本机，未上传 GitHub。当前版本仍有真人遥操作待验收问题，保存不等于控制效果已经全部合格。

以下为历史重建记录，不是当前版本标识。

## 历史：v0.3.0-egm-stable-hold

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
