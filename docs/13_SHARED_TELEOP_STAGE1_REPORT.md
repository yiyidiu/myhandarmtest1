# 最小干预共享遥操作第一阶段交付报告

日期：2026-08-20。本报告只描述本轮实际检查、实现和运行过的内容。未连接实体 ABB、真实三指手控制器，也没有进行新的 D455 在线采集。

## 1. 开始前审查得到的真实基线

- 原 D455 遥操作基线不是 HaMeR：`d455_conda_udp_sender_servo_v3.py` 使用 MediaPipe、深度和平面估计，通过 UDP 5005 发送 JSON `delta=[dx,dy,dz,droll_deg,dpitch_deg,dyaw_deg]`。
- 原 ROS 链为 `ros_udp_target_pose_receiver_servo_local.py -> /target_ee_pose (PoseStamped) -> servo_pose_tracking_node_v2.py -> /servo_server/delta_twist_cmds (TwistStamped) -> MoveIt Servo -> /abbarm_velocity_controller/command (Float64MultiArray)`。
- HaMeR 只有 JSONL 归档，没有 ROS/UDP 控制输出。每帧含 D455 global-time 时间戳、`mano_joints`、腕部二维投影、对齐深度路径、D455 内参，以及 `palm_frames.mano_joint_palm_frame.rotation/quaternion_xyzw`。MANO 三维关节和 `pred_cam_t` 不是 D455 公制坐标，不能直接作机械臂平移。
- 相机光学系为 `camera_color_optical_frame`（x 右、y 下、z 前）；机器人基座为 `base_link`；法兰为 `flange`；当前手掌链接为 `handbase_link`；Servo 控制链接为 `tool0`。
- 控制使用相对运动速度方向映射，不使用相机中的绝对手位姿目标，因此不需要也不查询 `camera_color_optical_frame -> base_link` 的完整 TF/外参；只需确认三轴对应、符号和比例。
- 当前 URDF 只有 `flange -> handbase_link` 固定变换和 `flange -> tool0` 的 `[0.17,0,0] + Ry(90°)`；没有实测、独立的掌心/抓取中心标定。
- 原 Servo 配置的 `check_collisions:false` 没有被修改；本轮新增独立的安全配置并启用碰撞检查，以免破坏旧基线。
- 工作区根目录不是 Git 仓库；没有执行 reset、checkout、删除或覆盖无关文件。

## 2. 本轮数据流和节点

```text
D455 + HaMeR
  -> HaMeR腕部像素 + D455 aligned Z16/内参：相机系公制腕位置
  -> MANO joint palm frame：粗略 SO(3) 掌姿态
  -> UDP handarm_hamer_pose_v1 / HaMeR记录回放 / 合成发布器
  -> /shared_teleop/hamer_pose (HamerHandPose)
  -> six_dof_trend：相对零位、多帧6D趋势、跳变拒绝、逐轴置信度
  -> /shared_teleop/raw_hand_command (HandCommand)
  -> gesture_isolation：0.3 s稳定/释放防抖，手臂保持，机械手单次事件
  -> /shared_teleop/operator_command
  -> shared_control：top/side最近可行候选、连续辅助、反向降权
  -> /shared_teleop/assisted_command
  -> moveit_servo_output_adapter：最新时间戳桥、50 Hz、速度/加速度/超时/工作区/急停
  -> /shared_teleop/safe_twist（始终可监视）
  -> /servo_server/delta_twist_cmds（仅通过输出门时）
  -> MoveIt Servo -> /abbarm_velocity_controller/command（本轮仅 Gazebo）
```

`HamerHandPose.msg` 表达公制腕位姿、六维置信度、有效性和手势；`HandCommand.msg` 表达带 Header 的 Twist、六维置信度、有效性和手势。`HamerHandPose.header.stamp` 使用 ROS 接收时钟供 watchdog 使用，`source_timestamp` 保留 D455/HaMeR 原始时钟供趋势估计，避免 Gazebo `/clock` 与 D455 global-time 混算。内部姿态全部用旋转矩阵、四元数和 SO(3) 旋转向量计算，没有用欧拉角相减。

默认位置参考已迁移为 MANO 手腕开口 16 个边界顶点的中心投影加 D455 对齐公制深度；默认旋转参考为中性手腕环到当前手腕环的 IRLS/Huber 稳健 Kabsch 坐标架，再经因果 SO(3) 滤波。MANO 第 0 号 wrist/root 与 MCP 掌坐标架只作为 `--control-reference mano-joint-palm` 回退基线。真人 UDP 入口打开后默认锁定，必须在摄像头窗口按 `C`，带新参考令牌的下一有效帧才会同时锁定手零位和机器人零位；合成/回放测试仍可显式调用 `/shared_teleop/confirm_hand_reference`。控制方向始终使用固定 `camera_color_optical_frame`，不随掌局部坐标旋转；跳变拒绝和重锚趋势窗口都不会自动改写零位。压缩包的前臂纵轴不能观测绕自身的横滚，本阶段不把它当作完整姿态或位置参考。

旧 UDP 5005 协议原样保留。新适配器只接受 UDP 5010 上 `handarm_hamer_pose_v1`，并拒绝旧 schema、重复序号和乱序包。

## 3. 坐标和参数

权威参数为 `src/handarm_moveit_demo/config/shared_teleop.yaml`。其中包含：

- 相机/基座/法兰/掌心/抓取中心/Servo 链接名；
- 带符号的三轴映射矩阵、平移和旋转增益；
- 逐轴速度与加速度上限、平移/旋转死区、跳变阈值、因果平滑、时间间隔范围；
- 真人低帧率链使用 0.40 s 输入超时和 0.55 s 强制全零截止；
- 明确标注为预设边界、不是未知障碍物避障的工作空间；
- 法兰到掌心、法兰到抓取中心、Servo 控制点到抓取中心的固定变换；
- top 倾角范围、side 仰角范围、桌面法向、左右前后接近方向；
- 辅助速度/增益、连续强度上升下降、反向输入判定；
- 0.3 s 手势稳定和释放时间。

当前轴映射和所有工具固定变换均标记为 `TEMPORARY/UNCALIBRATED`。实体输出同时要求非仿真、`enable_robot=true`、实测标定确认和精确授权令牌；任一缺失都不会发布实体 Servo 命令。新增 launch 默认 `enable_robot=false`。

## 4. 实际验证结果

### 自动测试

- `./scripts/run_stage1_tests.sh`：退出码 0。
- 阶段一核心/安全：22 tests，0 error，0 failure。
- 原 `perception_hamer` 与新增公制腕转换/MANO 网格叠加：133 tests，全部通过。
- 正确 source ROS 环境后单独回归原 `handarm_sim_demo`：111 tests，47.087 s，全部通过。输出中的 `Unknown tag material/hardwareInterface` 是原 URDF 解析器警告，不是失败。

22 项阶段一测试覆盖六轴同时存在、非主轴保留、四元数正负号等价、跳变拒绝、150 ms 回零、跨时钟域未来时间戳拒绝、逐轴置信度缩放、top/side 最近候选、固定抓取中心、反向降权、短手势不触发、稳定手势保持和单次动作、50 Hz 统计，以及显式固定相机参考、轴映射、速度仿真接口、launch/碰撞/未标定/实体输出门的静态安全检查。

### 纯离线演示

`./scripts/run_stage1_offline_demo.sh` 的最终复跑结果：301 个控制 tick，30 Hz 合成输入、实际控制频率 50.0 Hz，检测到六轴同时非零；处理时间均值 0.400 ms、P95 0.757 ms、最大 1.092 ms；断开输入后 0.15 s 全零验证通过；实体命令数 0。

### ROS/Gazebo 安全演示

完整运行日志 `/tmp/handarm_shared_teleop_logs/shared_teleop_20260820T160634.csv` 共 6236 行。去除主动断开输入后的超时段：

| 指标 | 均值 | P95 | 最大值 |
|---|---:|---:|---:|
| 控制频率 | 50.002 Hz | 50.000 Hz | 58.824 Hz |
| 输入年龄/输入到输出延迟 | 39.6 ms | 53.0 ms | 73.0 ms |
| 六维趋势节点 | 0.519 ms | 2.000 ms | 7.000 ms |
| 共享姿态辅助节点 | 1.965 ms | 3.225 ms | 9.179 ms |
| 输出安全适配器 | 0.655 ms | 1.127 ms | 2.555 ms |

主动停止输入后记录到 `INPUT_TIMEOUT` 和 `INPUT_TIMEOUT_ZERO`，安全 Twist 六轴为零；急停锁存试验也得到 `EMERGENCY_STOP_LATCHED` 和全零。稳定 CLOSE 手势只发出一次 mock 机械手命令。短时再次启动生成 642 行 CSV 并正常关闭，没有定时器发布到已关闭话题或 CSV 关闭竞态。

新增 `./scripts/run_stage1_gazebo_direction_validation.sh` 会从安全初始关节位形启动，自动确认手参考并逐轴往返。该位形经 `/check_state_validity=true`、`tool0=[0.3114,0.0101,0.4672] m` 和雅可比条件数 `10.55` 筛选。2026-08-20 实测：`base +X/+Y/+Z` 平移主分量为 `24.31/24.31/19.89 mm`；绕 `base +X/+Y/+Z` 的横滚/俯仰/偏航为 `0.1602/0.1614/0.1395 rad`；各交叉轴远小于主轴，六项均通过。Servo 危险状态为空，碰撞缩放全程 1.0，往返后位置误差约 `[−0.158,0.114,0.197] mm`、旋转误差约 `[−0.00113,−0.00189,0.00056] rad`。结果写入脚本打印的 JSON 路径。

首次排查还发现速度接口错误继承了位置 PID，零命令时机械臂会在重力下移动。现已将手部 PID 单独配置；方向验证专用速度 URDF 对可动链接关闭重力，以表达“ABB 驱动已补偿重力”的理想速度植物。该设置只验证坐标/Servo 运动，不是实际机械臂动力学证明。`check_collisions:true` 保持开启，自碰撞减速阈值设为 MoveIt Servo 示例采用的 10 mm；未关闭碰撞检查。top/side 姿态辅助的 Gazebo 到位仍未在本次方向验收中验证。

### 现有 HaMeR 记录回放

回放 `DEV_HAMER_TRANSLATION_20260813T184556`。归档摘要为 415 个有效 HaMeR 帧、实际 HaMeR 16.585 Hz；其中 161 帧在腕像素附近有足够 aligned-depth 样本并发布为公制腕位姿，254 帧以 `insufficient aligned depth near HaMeR wrist pixel` 跳过。回放链可运行，但该 USB2 开发记录不等于 30 Hz、连续、正式硬件输入验收。

## 5. 文件清单

主要新增：

- `handarm_moveit_demo/msg/{HamerHandPose,HandCommand}.msg`
- `handarm_moveit_demo/src/handarm_moveit_demo/shared_teleop_core.py`
- `handarm_moveit_demo/scripts/{hamer_input_adapter,six_dof_trend_node,gesture_isolation_node,shared_control_node,moveit_servo_output_adapter}.py`
- `handarm_moveit_demo/scripts/{synthetic_hamer_pose_publisher,teleop_pose_replay,hamer_recording_replay,teleop_csv_logger,offline_shared_teleop_demo}.py`
- `handarm_moveit_demo/scripts/{synthetic_direction_sequence,gazebo_direction_validator}.py`
- `handarm_moveit_demo/scripts/{mock_three_finger_hand_adapter,grasp_tolerance_data_collector}.py`
- `handarm_moveit_demo/config/shared_teleop.yaml`
- `handarm_moveit_demo/launch/{shared_teleop_core,shared_teleop_safe_demo}.launch`
- `abb120_moveit_config1/config/servo_abbarm_velocity_safe.yaml`
- `abb120_moveit_config1/launch/abbarm_servo_velocity_safe.launch`
- `perception_hamer/src/teleop_pose_packet.py` 及其测试
- `scripts/run_stage1_tests.sh`、`scripts/run_stage1_offline_demo.sh`、`scripts/run_stage1_safe_demo.sh`、`scripts/run_stage1_gazebo_direction_validation.sh`
- `docs/12_GRASP_TOLERANCE_DATA_COLLECTION.md` 和本报告。

修改：`handarm_moveit_demo/CMakeLists.txt`、`package.xml`、`README.md`、`setup.py`，`perception_hamer/scripts/run_d455_hamer_crop.py`，以及 `abb120_moveit_config1/launch/gazebo_velocity.launch` 的 headless GUI/速度 PID 配置。新增 `handarm_sim_demo/config/gazebo_hand_only_pid.yaml`，速度专用 URDF 增加明确的零重力方向验证语义。旧 5005 UDP、旧位姿跟踪和旧 Servo 配置均保留。

## 6. 尚未完成和用户需补充

1. 确认相机运动方向到机器人速度轴的对应、正负号和每轴比例；不需要完整相机外参。
2. 实测 `flange -> 三指手掌心` 和 `flange -> 抓取中心`，并确认抓取中心局部接近轴。
3. 确认 top/side 允许角度。
4. 提供三指手现有 ROS 高层接口的真实消息/服务合同；当前只有明确标注的 mock 发布器，没有 CAN/EtherCAT 实连声明。
5. 六轴方向 Gazebo 验收已通过；仍需单独做 top/side 集成姿态修正到位和抓取中心保持的 Gazebo 测量。
6. 进行 USB3 下 D455/HaMeR 在线频率、深度覆盖和长时稳定性验证。
7. 经用户单独授权后，按低速分级流程测试实体 ABB；本轮未执行。
8. 没有未知环境感知、未知物体识别或未知障碍物自主避障。

下一阶段字段合同与只记录、不控制的试验脚本见 `docs/12_GRASP_TOLERANCE_DATA_COLLECTION.md`。脚本只接受明确标记为真实测量或显式仿真的完整记录，不生成默认值或伪造实验数据。
