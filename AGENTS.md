# 2026-09-21 整体腕姿与四场景保存版

此版本对应 GitHub 标签 `teleop-whole-wrist-scenes-20260921`。
在本机 `/home/dongtian/myhandarmtest1-whole-wrist-scenes-20260921` 完成本次保存后，
默认只读取或运行，不在保存目录继续调试、更新依赖、切换开发分支或执行 git pull。

后续开发仍使用 `/home/dongtian/myhandarmtest1-mediapipe-comparison-20260915` 的 `codex/` 分支。
用户明确要求保存新版本时另建目录和标签，保留此版本及9月17日保存版。

四个场景入口共用 HaMeR 16点腕口、整体腕姿映射和当前控制器，仅场景不同。
不得通过切换场景顺带切换手部模型、坐标定义或控制参数。
