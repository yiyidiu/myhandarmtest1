# 仿真自主基线已知限制

日期：2026-08-14

1. 仅为 Gazebo 11 / ROS Noetic / MoveIt 1 的已知静态场景基线，未连接真实 ABB 或真实三指手。
2. 运行 URDF 的 PositionJointInterface 已加载有限 PID；四个 Gazebo mimic 关节也使用有限力 PID，不再走逐步 `SetPosition`。这些参数只证明当前空载 Gazebo 稳定性，不能外推到实体动力学或力矩控制。
3. 三指手有四个主动位置关节与四个 Gazebo mimic 关节；实体根部应变片不在仿真闭环中。Gazebo 接触以后只允许用于独立验收观测，不能反向驱动抓握控制。
4. 公开接口只允许 `GRASP/RELEASE`，两者保持同一掌型。`PRE_SHAPE_A/B` 仅保留为内部历史配置，其物理含义未确认且公开话题会拒绝调用。
5. `approach_only` 只证明 OPEN、预抓取、碰撞感知短接近、CLOSE、STOP；不证明物体被夹住。
6. `deterministic_lift` 使用新增 Gazebo 固定关节服务搬运已知目标，证明的是确定性仿真附着和 0.10 m 抬升，不证明摩擦夹持。
7. physical-contact grasp 仍为 NOT RUN；已通过的 15/15 空载窗口不能证明夹持或抬升，结果必须继续记录 `physical_grasp_claimed=false`。
8. 障碍物是静态已知盒体，没有动态障碍感知或在线场景更新。
9. 目标物位姿由 YAML 提供，没有 D455、HaMeR、MediaPipe、KLT/Kabsch 或视觉识别。
10. 核心回归包含 headless 5 循环 15/15 实测；GUI 链路也实际下发过 `RELEASE -> GRASP -> RELEASE` 并读取全部 8 个关节，但尚未归档近景视频。
11. Gazebo Classic 11 已进入弃用状态；迁移到新 Gazebo 尚未测试。
12. 当前接口允许未来替换 YAML 目标源，但接入遥操作或真实硬件前必须重新建立安全监督、标定、watchdog 和硬件验收。
