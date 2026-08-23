# handarm_moveit_demo

这是给 `myhandarmtest1` 工作空间补充的 MoveIt 编程接口入门示例包。

## 推荐运行顺序

### 1. 编译

```bash
cd ~/myhandarmtest1
catkin_make
source devel/setup.bash
```

### 2. 只看 RViz 中的 MoveIt 假执行

终端 1：

```bash
roslaunch abb120_moveit_config1 demo.launch
```

终端 2：

```bash
source ~/myhandarmtest1/devel/setup.bash
rosrun handarm_moveit_demo 01_print_moveit_info.py
rosrun handarm_moveit_demo 02_move_arm_named.py
rosrun handarm_moveit_demo 03_move_arm_joint.py
rosrun handarm_moveit_demo 04_move_arm_relative_pose.py
rosrun handarm_moveit_demo 05_move_hand_named.py
```

### 3. 让 Gazebo 中的机械臂执行

终端 1：

```bash
roslaunch abb120_moveit_config1 demo_gazebo.launch
```

终端 2：

```bash
source ~/myhandarmtest1/devel/setup.bash
rosrun handarm_moveit_demo 02_move_arm_named.py
rosrun handarm_moveit_demo 03_move_arm_joint.py
rosrun handarm_moveit_demo 04_move_arm_relative_pose.py
rosrun handarm_moveit_demo 05_move_hand_named.py
```

## 规划组名字

你的 SRDF 中有两个规划组：

- `abbarm`：机械臂，链为 `base_link -> handbase_link`
- `hand`：灵巧手，主动关节为 `f1j1、f1j2、f2j1、f3j2`

## 命名姿态

- `abbarm/up`
- `hand/start1`
- `hand/grasp1`
- `hand/grasp2`

---

## 第一阶段：最小干预共享遥操作（安全仿真）

本阶段使用“显式零位下的六维相对位姿跟踪 + 连续姿态辅助”。确认参考时同时锁定人手零位和当前 `tool0` 初始位姿；手的相对位姿生成机器人目标位姿，因此手回到零位后机械臂也闭环回到记录的初始位姿。六个平移/旋转方向可以同时存在；姿态差使用 SO(3)/四元数计算，欧拉角不进入控制链。

### 数据流

```text
D455 640x480@30 + HaMeR
  -> MANO手腕开口16点中心投影 + D455对齐深度 = 相机系公制腕位置（负责平移）
  -> 中性手腕环到当前手腕环的稳健Kabsch + 因果SO(3) = 腕姿态（负责旋转）
  -> UDP handarm_hamer_pose_v1 或记录回放
  -> /shared_teleop/hamer_pose (HamerHandPose)
  -> six_dof_trend_node（手/robot零位、直接相对目标位姿、50 Hz闭环误差Twist、HOLD_LAST）
  -> /shared_teleop/raw_hand_command (HandCommand)
  -> gesture_isolation_node（0.3 s防抖、手臂保持、机械手动作单次事件）
  -> shared_control_node（俯抓/四向侧抓最近可行姿态、连续辅助强度）
  -> moveit_servo_output_adapter（50 Hz、工作空间/超时/急停）
  -> /servo_server/delta_twist_cmds
  -> MoveIt Servo（真人 Gazebo 响应 profile 不做低通/碰撞距离缩放）
  -> /abbarm_velocity_controller/command（Gazebo）
```

HaMeR 的 `pred_cam_t`/弱透视相机量不是 D455 公制位置，本实现不会使用它们驱动机器人。默认平移参考点是 MANO 开放腕口 16 个边界顶点的中心，不是手心，也不再是单独的第 0 号手腕关节点；默认旋转轴由同一 16 点环稳健拟合，因此实时手指张合不参与控制轴构造。压缩包中的前臂纵轴不能单独观测绕自身的横滚，当前不把它当作完整位置/姿态参考。旧第 0 点和 MCP 方法可用 `--control-reference mano-joint-palm` 回退。当前 `run_d455_hamer_crop.py` 可选地输出新 UDP 合同；旧 MediaPipe `delta + Euler°` 的 5005 端口基线保留但不进入本链。

### 构建和一条命令测试

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y
./scripts/run_stage1_tests.sh
```

测试覆盖：六轴同时输出、相对位姿回零、AprilTag V3 的 0.6 倍平移与 1:1 工具局部旋转、滤波/限幅组件边界、真人 profile 的直接位姿配置、绿色轴符号、C 键控制门、HOLD_LAST、四元数正负号、姿态失效时位置通道继续、650 ms 内超时停机、逐轴置信度、组合速度范数、俯/侧抓最近姿态、抓取中心保持、人工反向输入降权、手势防抖/单次命令和 50 Hz 循环统计。

### 一条命令运行纯离线演示

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_offline_demo.sh
```

脚本以带真实时间戳的 30 Hz 合成 HaMeR 位姿驱动 50 Hz 控制循环，最后主动停止输入以验证超时回零；输出 CSV 和 summary JSON 到命令打印的临时目录。它不启动 ROS，不发送任何机器人命令。

### 一条命令运行安全 Gazebo 演示

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_safe_demo.sh
```

默认 `enable_robot:=false`、`simulation:=true`、无 Gazebo GUI/RViz，并使用 `servo_abbarm_velocity_safe.yaml`（`check_collisions:true`、20 ms 发布周期、0.10 s Servo watchdog）。合成源约 30 Hz，安全 Twist 目标 50 Hz。停止使用 `Ctrl-C`。

启动后必须先建立人手参考，否则六维命令保持无效全零：

```bash
rosservice call /shared_teleop/confirm_hand_reference
```

服务调用后的下一帧有效“16 点手腕环公制中心 + 手腕环姿态”会成为人手锁定零位，同时记录当时 `base_link -> tool0` 为机器人零位。平移轴始终是 D455 固定光学轴；旋转使用记录时的 MANO/tool0 局部轴右乘关系。跳变拒绝不会自动改写零位。安全方向仿真模型使用零重力，因为 Gazebo 理想速度关节不包含 ABB 驱动器重力补偿；这只用于运动方向/Servo 验证，不代表真实动力学验证。碰撞检查仍保持开启。

### 一条命令验证 Gazebo 三轴平移和横滚/俯仰/偏航

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_gazebo_direction_validation.sh
```

脚本自动启动无 GUI 响应优先仿真、确认人手与机器人零位、逐轴发出小幅“偏移—保持—回零”动作、从 `base_link -> tool0` TF 测量实际运动并关闭仿真。2026-08-21 实测：2.5 cm 手位移得到 `14.886/14.862/15.021 mm`；0.22 rad 手旋转得到 `0.219499/0.219276/0.219394 rad`；三姿态轴比例为 `0.9977/0.9967/0.9972`，整套往返后回零误差为 `0.180 mm / 0.034°`。正反 90° X 轴（含 Y/Z 姿态耦合、120 mm 平移和单帧回零）专项可运行 `./scripts/run_gazebo_bidirectional_x90_validation.sh`；当前稳态 `+90.010°/-90.011°`，两个方向的 5°/80° 响应时间均为 `0.167/0.500 s`。

真人 UDP 的 C 键门控可单独回归：

```bash
./scripts/run_c_gate_gazebo_validation.sh
```

该脚本在按 C 前持续发送变化的人手位姿并确认机械臂不动，然后发送新 C 参考令牌，
验证画面向右对应 `base -Y` 的 0.6 倍平移关系以及回零。中途还会故意停止 UDP
1.2 秒，验收 V3 `HOLD_LAST` 是否持续刷新下游目标且不触发 `INPUT_TIMEOUT_ZERO`。

### 使用当前 HaMeR + D455 实时输出（仍仅仿真）

ROS 端：

```bash
roslaunch handarm_moveit_demo live_human_gazebo_teleop.launch gazebo_gui:=true
```

HaMeR conda 端：

```bash
cd /home/diu/myhandarmtest1
conda run --no-capture-output -n hamer_rtx2060 \
  python perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe \
  --mesh-renderer teleoperation-core \
  --control-reference mano-wrist-ring \
  --roi-smoothing-alpha 1.0 \
  --orientation-filter-large-angle-mode follow \
  --orientation-filter-max-gain 1.0 \
  --disable-forearm-fusion \
  --hand-presence-timeout-s 0.25 \
  --teleop-udp-host 127.0.0.1 \
  --teleop-udp-port 5010
```

摄像头启动后控制保持锁定；手保持中性并在摄像头窗口按一次 `C`，才会同时记录
手零位和当前 `tool0` 位姿并开始跟随。也可运行
`./scripts/run_live_human_gazebo_acceptance.sh --duration-s 30`，在倒计时后完成至少
3 cm 平移和 15 度旋转。验收器不发布合成输入，只检查真人位姿、ROS 安全命令和
Gazebo `base_link -> tool0` 实测运动，并输出带 `passed` 字段的 JSON。完整分终端命令、
判据和故障边界见 `docs/17_LIVE_HUMAN_GAZEBO_TELEOP_ACCEPTANCE.md`。

HaMeR 窗口默认使用 `teleoperation_ubuntu_core.tar.gz` 的完整实时渲染链：左侧是
产生该 HaMeR 结果的原始推理帧，右侧是在同一帧上投影全部 778 个顶点和 1538 个
MANO 三角面的网格。默认不再用 KLT 将旧网格缩放到最新相机帧，因此不会用“框已
跟上、旧手姿态被搬过去”的方式伪造 30 Hz 网格。网格真实刷新率就是实际 HaMeR Hz。

`hamer_rtx2060` 中的 OpenCV 是无 GUI
版本，程序会自动把显示帧交给带 Qt5 的 `mediapipe_env`，推理和 UDP 不会再因
`cv2.imshow` 崩溃。终端应打印 `using the MediaPipe display sidecar`。实时遥操作
必须保留窗口，因为 `C` 是唯一的启用/重设零位入口；同时指定 UDP 和
`--no-display` 会被拒绝。打开窗口后先保持手稳定，再按一次 `C`；按 C 前 Gazebo
始终不跟随，再次按 C 会同时重设手零位和当前机器人零位。

启动时先显示 D455 原始彩色画面。真实手、手腕和五指稳定出现在画面中后会自动
选择活动手并启动 HaMeR，不需要按 `C` 选择手；但必须随后按 `C` 建立 Gazebo
控制零位。MediaPipe 可以同时看到左右手，但只有
一个自动活动手进入单个 HaMeR/MANO；另一只手同时出现时会被忽略。活动手消失且
另一只手连续稳定出现后，系统自动清除旧跟踪并判定换手。
裁剪框使用 `1.0` 新帧权重；发送给遥操作链的手掌姿态使用质量/运动自适应
SO(3) 滤波：静止时抑制 MANO 抖动，明确运动时增益升到最多 `0.95`。裁剪靠近
边缘或突然跳动仍会降低对应置信度。跟踪失效后
局部趋势窗口会重建，但不会改写已确认的人手零位和机器人零位；有效输入恢复后，会重新按当前相对手位姿闭环追踪，而不是补发失效期间的速度。
单帧姿态创新达到 `70°` 会保持最后可信旋转；正常连续的大幅人手转动会进入高增益
而不会被低通拖慢。姿态通道无效时 D455 公制腕位置仍可继续发送。
同一时刻只能运行一个 `run_d455_hamer_crop.py`；程序有单实例锁，重复启动会在
加载模型前报告现有 PID，避免 RTX 2060 同时加载两份 HaMeR 后显存溢出。必须在
D455 画面中保持完整手掌；只有显式加入 `--require-hand-confirmation` 时才需要在
窗口中按 `c`、Enter、空格或双击。

确认后 MediaPipe 仍逐帧独立检查真实手是否存在，KLT 只负责检测帧之间的快速
裁剪框插值。首次 `no_hand_detected` 或 0.25 秒无检测结果时，窗口应立即显示
`REAL HAND presence=NO`、`ROI ... valid=False`、`MANO mesh=OFF`；程序同时清空
KLT，并停止新的 HaMeR 结果和 UDP 输出。Gazebo 控制链此时按 V3 语义继续闭环保持
最后一个有效目标，不会把目标清零或因输入 watchdog 反复刹车。重新伸手后需要连续两帧检测一致才重建
裁剪框，确认前的旧 MANO 网格不会重新出现。

UDP v1 必填：`schema/session_id/sequence/stamp/frame_id/wrist_position_m`、旋转矩阵或 `xyzw` 四元数、六维置信度、valid、手势以及
`control_enabled/control_reference_epoch/control_reference_token`。缺少有效 C 参考令牌的
旧发送器会被默认视为锁定；接收器同时拒绝乱序和旧版协议。

`HamerHandPose.header.stamp` 是 ROS 时钟域的适配器接收时间；`source_timestamp` 保留 D455/HaMeR 原始时钟，供多帧趋势估计和 CSV 追溯。两者分离可避免 Gazebo `/clock` 与 D455 global-time 混算。在仿真中，已确认 C 零位后会像 V3 `publish_repeated` 一样以 50 Hz 刷新最后目标的 ROS 时间戳；实体输出仍保留 0.40 s 输入 watchdog 和 0.65 s 强制归零，不会把仿真的无限 HOLD 策略带到真机。

2026-08-21 真人运行中正常帧间隔约 0.19--0.30 s，且手腕环偶发无效会让发送端短时没有 UDP。仿真链现采用 V3 目标保持：感知帧只更新目标，控制器始终以 50 Hz 向最后有效目标闭环收敛；恢复后从最后滤波位姿继续，不补发丢帧期间的虚构速度。自动验收中故意断流 1.2 秒，下游输入年龄最大仅 0.04 秒且没有超时刹车。USB3 和最终真人感知质量仍需操作者现场验收。

### 记录与回放

安全演示默认在 `/tmp/handarm_shared_teleop_logs/` 新建 CSV。回放该 CSV：

```bash
roslaunch handarm_moveit_demo shared_teleop_safe_demo.launch \
  input_source:=replay \
  input_csv:=/absolute/path/shared_teleop_YYYYMMDDTHHMMSS.csv
```

也可回放已有 `perception_hamer` 会话；回放器读取 HaMeR 腕像素、对齐 Z16、D455 内参/深度比例和 MANO joint palm frame：

```bash
roslaunch handarm_moveit_demo shared_teleop_safe_demo.launch \
  input_source:=hamer_recording \
  hamer_session:=/absolute/path/to/DEV_HAMER_SESSION \
  hamer_replay_speed:=1.0
```

CSV 包含原始/相对手位姿、原始/安全六维速度、六维置信度、辅助强度/候选/目标、实际末端位姿、输入年龄、循环频率和处理时间、手势，以及超时/跳变/工作空间原因。

### 姿态辅助、取消和急停

```bash
rostopic pub -1 /shared_teleop/assist_request std_msgs/String "data: 'top'"
rostopic pub -1 /shared_teleop/assist_request std_msgs/String "data: 'side'"
rostopic pub -1 /shared_teleop/assist_request std_msgs/String "data: 'cancel'"
rostopic pub -1 /shared_teleop/emergency_stop std_msgs/Bool "data: true"
```

俯抓将抓取中心接近轴对齐桌面法向反方向，并用最小旋转保留水平朝向。侧抓比较 left/right/front/back，调用 `/compute_ik` 且启用碰撞检查，只选距离最小的可行候选；无候选时拒绝辅助。持续反向角速度会把辅助强度连续降到零。急停锁存；输入恢复前先发布 `false`，再调用：

```bash
rosservice call /shared_teleop/reset_emergency_stop
```

### 坐标、工具中心和参数

权威配置是 `config/shared_teleop.yaml`：

- D455：`camera_color_optical_frame`，x右/y下/z前；机器人基座：`base_link`；法兰：`flange`；现有手掌基座：`handbase_link`；Servo/临时抓取中心：`tool0`。
- 平移点为 MANO 开放手腕口 16 个边界顶点中心的 D455 公制位置；旋转主体由同一手腕环稳健拟合得到。默认允许 D455 深度前臂纵轴以最高 20% 权重校正纵向轴，MANO 仍提供完整姿态与横滚；加 `--disable-forearm-fusion` 才是纯 MANO 模式。
- `reference.frame=camera_color_optical_frame`、`direction_basis=FIXED_CAMERA_TRANSLATION_AND_C_ZERO_LOCAL_ROTATION`、`allow_automatic_rezero=false`；平移使用固定相机轴，姿态使用 C 零位局部轴。
- 本系统把相机中人手相对零位的位移/旋转映射成机器人相对初始 `tool0` 的目标位姿，不把相机绝对坐标直接当机器人绝对坐标。因此不要求 `camera_color_optical_frame -> base_link` 的完整外参；`mapping.translation_matrix`、`mapping.rotation_matrix` 和两个 gain 数组定义轴对应、正负号与位姿比例。
- 当前采用 AprilTag V3 虚拟手腕链路的关系：手靠近相机→`base +X`，画面向右→绿色轴 `base -Y`，画面向上→`base +Z`。平移统一为 0.6 倍，即手移 5 cm 生成末端 3 cm 目标。姿态使用 `R_delta=R_hand_zero^T R_hand_now` 与 `R_target=R_tool_zero R_delta`，MANO 手腕局部 RGB 轴直接对应记录零位时的 `tool0` 局部 XYZ 轴，旋转角度 1:1（手转 20° 生成 20° 目标）。
- 当 C 零位局部 X 旋转超过 30°并成为主分量时，侧抓投影会在 30°--55°范围内平滑消除 MANO 附带的 Y/Z 耦合，但 X 角度始终保持 1:1；正负输入严格使用同一奇对称关系。IRB120 第 6 轴模型已按官方工作范围改为 ±400°，不再用错误的 ±180°提前触发关节边界。
- 真人 Gazebo profile 为实时优先且静止自适应稳态：ROI 直接更新，感知和 ROS 位姿滤波在静止噪声带使用低增益、明确运动时接近直接更新；速度前馈只在静止噪声带归零，绝对位姿反馈始终保留。死区为 0，普通人手范围内的软件速度整形为 1 m/s、10 rad/s、50 m/s²、500 rad/s²；目标几何仍是 1:1，不是 3 倍增益。该 profile 关闭碰撞距离缩放，只允许仿真。
- 轴映射/符号、平移与旋转增益、速度/加速度、小运动死区、跳变阈值、超时、工作空间、俯/侧抓方向范围均在 YAML 中。
- 法兰到手掌来自当前 URDF；法兰到临时抓取中心沿用当前 `tool0` 的 `[0.17,0,0] + Ry(90°)`。二者均未实测，不能作为真实标定。
- `servo_control_to_grasp_center` 当前为单位变换，因为临时把 `tool0` 当抓取中心。填入实测值后，姿态辅助会求保持抓取中心不动的控制点位姿。

实体 ABB 输出同时要求：`simulation:=false`、`enable_robot:=true`、实测标定确认和精确授权字符串。默认 YAML 将标定状态设为 false，本阶段没有提供实体启动命令。

### 明确限制

- 未连接真实 ABB、D455 在线链和真实三指手 ROS 接口；机械手动作只发布到 mock 适配器，没有编造 CAN/EtherCAT 状态。
- 没有未知物体/障碍物感知。工作空间包络只是预设边界；真人 Gazebo 响应 profile 已按本轮要求关闭碰撞检查，不能用于实体机器人或抓取安全验证；安全 profile 仍会检查已知机器人/Planning Scene。
- HaMeR 姿态已知有漂移和根姿态翻转；本实现只做实验性 Gazebo 跟随、跳变拒绝和通道隔离，不声称解决姿态感知精度。按 `docs/10_TELEOP_TECHNICAL_ROADMAP.md`，正式姿态源仍需稠密 RGB-D point-to-plane ICP 通过门禁，AprilTag 可作真值/兜底。
- 当前固定变换和轴映射都是明显标记的仿真临时值；真实使用前必须实测。
- 下一阶段真实抓取容错字段和只记录不控制的脚本见 `docs/12_GRASP_TOLERANCE_DATA_COLLECTION.md`。
- 本轮逐项审查、性能数据、文件清单和未完成项见 `docs/13_SHARED_TELEOP_STAGE1_REPORT.md`。
- 供人工检查的逐终端运行命令和预期现象见 `docs/14_STAGE1_MANUAL_CHECK.md`。
