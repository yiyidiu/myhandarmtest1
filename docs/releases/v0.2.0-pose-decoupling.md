# v0.2.0-pose-decoupling

重建自 2026-08-24 04:43（Asia/Shanghai）的平移/姿态解耦通过检查点。

## 冻结时证据

- Catkin 构建通过；
- 62 项测试，0 错误、0 失败、0 跳过；
- 离线六方向平移和六方向姿态检查通过；
- 隔离 Gazebo 六方向检查通过，Servo 只出现状态 0；
- 回零误差约 0.091 mm / 0.031°。

## 原始冻结包

`teleop_small_stiffness_pose_decoupling_passed_20260824_044327.tar.gz`

SHA-256：`daa8aa9544802b2ec4f6dad051451fe312b0e3a3f944ecc774c18df99447a651`

真人自由六维输入仍可能因感知轴耦合接近奇异位形。该检查点不是实体机器人验收。
