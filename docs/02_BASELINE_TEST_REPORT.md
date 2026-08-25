# P0 基线测试报告

测试时间：2026-08-13（Asia/Shanghai）  
工作空间：`/home/diu/myhandarmtest1`  
范围：只建立现有系统基线；未修改核心控制算法。

## 结论

现有工程可增量构建，Gazebo 11、MoveIt 1、MoveIt Servo、ABB 六轴速度控制器和
三指手四主动关节位置轨迹控制器能够共同启动。该结果仅证明基线进程和 ROS 接口
存在，不代表正式遥操作、碰撞安全、D455、HaMeR 或抓取闭环已经通过。

P0 发现的阻塞风险：

1. RTX 2060 只有 6144 MiB，必须执行低显存 HaMeR 架构及真实峰值测试。
2. 正式候选 Servo 配置仍为 `check_collisions: false`，不可作为安全配置。
3. `gazebo_gui:=false` 仍启动 gzclient 和 RViz，不是真正 headless。
4. Gazebo 报告十个关节缺少 `/gazebo_ros_control/pid_gains/*` 的 P gain。
5. 现有 Catkin 测试目标为 0 tests，尚无自动回归保护。
6. 零 Twist 探测没有观测到 `/abbarm_velocity_controller/command` 样本；非零六轴链路
   必须在 P10 单独测试。
7. 工作空间根不是 Git 仓库，现有 `build/`、`devel/` 只能视为本地生成物，最终包需排除。
8. `JointGroupVelocityController` 无输出端 timeout；Servo 进程崩溃时可能持续执行最后一条
   非零速度。独立 watchdog 与进程消失故障注入均为 `NOT RUN`。

## 实测结果

| ID | 检查 | 结果 | 证据/备注 |
|---|---|---|---|
| P0-HW-01 | GPU 型号、显存、驱动 | PASS | `NVIDIA GeForce RTX 2060, 6144 MiB, 570.133.20` |
| P0-ENV-01 | Ubuntu/ROS/Gazebo | PASS | Ubuntu 20.04.6、Noetic 1.17.4、Gazebo 11.15.1 |
| P0-BUILD-01 | `catkin_make` | PASS | 返回 0；增量构建插件目标 |
| P0-DEPS-01 | `rosdep check --from-paths src --ignore-src` | PASS | `All system dependencies have been satisfied`；只覆盖已声明依赖 |
| P0-XML-01 | launch/XML/package.xml 语法 | PASS | `xmllint --noout` 返回 0 |
| P0-PY-01 | 现有 Python 文件静态编译 | PASS | 所有 `src/**/*.py` 由 Python 3 `compile()` 解析成功 |
| P0-URDF-01 | 速度版 Gazebo URDF | PASS | `check_urdf` 成功解析，根 link 为 `world` |
| P0-TEST-01 | Catkin 自动测试 | NO TESTS | `0 tests, 0 errors, 0 failures, 0 skipped` |
| P0-LAUNCH-01 | headless 参数启动 | PARTIAL | 系统启动，但仍有 `/gazebo_gui` 和 RViz 进程 |
| P0-ROS-01 | `/joint_states` | PASS | 收到 10 个主动关节的位置/速度样本 |
| P0-ROS-02 | `/servo_server/status` | PASS | 类型 `std_msgs/Int8`，零 Twist 时样本 `data: 0` |
| P0-ROS-03 | Servo Twist 输入 | PASS | 类型正确，subscriber 为 `/servo_server` |
| P0-ROS-04 | Servo 速度输出连接 | PASS | publisher `/servo_server`、subscriber `/gazebo` |
| P0-ROS-05 | 零 Twist 数值输出 | INCONCLUSIVE | 5 秒内未收到输出样本；不能记为通过 |
| P0-CTRL-01 | ABB 速度控制器 | PASS | 六个 ABB joints，状态 running |
| P0-CTRL-02 | 三指手轨迹控制器 | PASS | `f1j1/f1j2/f2j1/f3j2`，状态 running |
| P0-GZ-01 | 模型生成 | PASS | `SpawnModel: Successfully spawned entity` |
| P0-GZ-02 | gazebo_ros_control PID | FAIL | 十个主动关节均报告 `No p gain specified` |
| P0-SAFE-01 | Servo 碰撞检查 | FAIL | 运行参数 `/servo_server/check_collisions=false` |
| P0-SHUTDOWN-01 | 关闭所有进程 | PASS WITH WARNING | 控制器卸载阶段需要 roslaunch 升级 SIGTERM，最终无残留 |

## 已确认的 ROS 接口

```text
/joint_states                                      sensor_msgs/JointState
/servo_server/status                               std_msgs/Int8
/servo_server/delta_twist_cmds                     geometry_msgs/TwistStamped
/abbarm_velocity_controller/command                std_msgs/Float64MultiArray
/controller_gazebo_hand/follow_joint_trajectory/*  actionlib action topics
```

## 未运行

以下项目均为 `NOT RUN`：

- D455 设备枚举、RGB-D 对齐和真实深度质量；
- HaMeR 官方示例、裁剪推理、FP32/FP16 显存与延迟；
- MANO_RIGHT.pkl 加载和 MANO 掌部坐标系；
- RGB-D KLT、RANSAC-Kabsch、SO(3)/SE(3) 融合；
- 旧 MediaPipe、raw HaMeR、融合方案 A/B/C 数据集对比；
- 非零 X/Y/Z、Roll/Pitch/Yaw Servo 合成输入；
- clutch、UDP 乱序/丢包/超时测试；
- Servo 进程崩溃、速度发布者消失与 controller-side watchdog；
- 三指手接触传感器、自适应闭合、lift test 和抓取成败判定；
- Planning Scene 障碍、碰撞减速/停止、奇异性和工作空间边界；
- 真实 ABB 机械臂（按要求默认禁止连接）；
- 清洁交付压缩包。

## P0 门槛判定

P0 的现状执行基线已经记录，但 **P0 验收为 `NOT PASS`**。除碰撞关闭、PID、输出端
watchdog 和测试缺口外，对抗审查还确认旧控制入口不拒绝 NaN/Inf、UDP 不防乱序/
时间倒退且 manifest
依赖不闭合。当前仅可准备离线 HaMeR 裁剪接口和基准工具；不得把准备工作记作 P1
通过，也不能用后续功能掩盖这些 P0 阻断项。
