# v0.1.0-ground-workspace

重建自 2026-08-24 00:27:37（Asia/Shanghai）的地面工作空间通过检查点。

## 冻结时证据

- Catkin 构建通过；
- 61 项测试，0 错误、0 失败、0 跳过；
- 离线 6 个平移边界和 6 个姿态边界通过；
- 合成 UDP → Servo → Gazebo 验收通过；
- 相机标定仍为仿真临时参数。

## 原始冻结包

`teleop_ground_workspace_passed_20260824_002737.tar.gz`

SHA-256：`157865c9db15dee09daa2739ad5a5feaa894e1c4eef1edf4a3fc757aca3c1504`

该 Git 版本补入了冻结包未包含、但独立 Catkin 构建所需的共同机器人描述和第三方依赖。原始事实以 Release 附件为准。
