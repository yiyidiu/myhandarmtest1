# 当前人手位姿控制 Gazebo 机械臂末端：运行与验收

本入口只控制 Gazebo 中的 ABB IRB120，不允许输出到实体机器人。数据链为：

```text
D455 + HaMeR/MANO wrist-ring pose
  -> UDP handarm_hamer_pose_v1, localhost:5010
  -> /shared_teleop/hamer_pose
  -> locked hand/tool0 zero + AprilTag V3 relative-pose relation
  -> motion-adaptive pose filter + symmetric local-X side projection
  -> 50 Hz target-vs-current tool0 error controller
  -> response-first workspace/emergency-stop gate
  -> /servo_server/delta_twist_cmds
  -> unfiltered MoveIt Servo (Gazebo response profile)
  -> Gazebo base_link -> tool0
```

## 0. 构建和确定性前置验收

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_tests.sh
./scripts/run_stage1_gazebo_direction_validation.sh
./scripts/run_gazebo_bidirectional_x90_validation.sh
./scripts/run_c_gate_gazebo_validation.sh
```

四条命令都必须退出码为 0。第二条证明六轴映射，第三条用带 Y/Z 耦合的输入专门证明
正、反两个 90° X 姿态响应，
第四条从 UDP 入口证明按 C 前锁定、按 C 后跟随和回零；它们都不代替真人感知验收。

2026-08-21 当前响应优先配置实测：2.5 cm 合成手部位移使 `tool0` 沿
`base X/Y/Z` 分别移动 14.886/14.862/15.021 mm（目标 15 mm）；0.22 rad
合成手部旋转使末端绕记录零位时的 tool0 X/Y/Z 局部轴分别转动
0.219499/0.219276/0.219394 rad，三个姿态比例均约 1.0。整套六轴往返后的
回零误差为 0.180 mm/0.034°。带 120 mm 平移并单帧回零的正、反 90° X 专项
稳态角为 +90.010°/-90.011°，首次越过 5°
为 0.167 s、越过 80° 为 0.500 s；自适应滤波下瞬时峰值约为 103.6°，随后
回到严格 1:1 稳态。数值允许因 Gazebo 步进有小幅波动，验收以脚本退出码和 JSON
顶层 `passed` 为准。

## 1. 终端一：启动仿真、Servo 和 UDP 接收链

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch handarm_moveit_demo live_human_gazebo_teleop.launch gazebo_gui:=true
```

无桌面环境时将 `gazebo_gui:=false`。该 launch 使用 `input_source=udp`、
`simulation=true`、`enable_robot=false` 和 `response_first=true`；实体输出没有
开放参数。响应优先配置关闭 MoveIt 碰撞距离缩放、关节速度低通和奇异邻域降速，
只用于 Gazebo。安全配置仍保留，可用 `shared_teleop_safe_demo.launch` 的默认
`response_first:=false` 单独回归。

## 2. 终端二：启动当前 D455 + HaMeR/MANO 位姿发送器

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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

画面中只保留一只完整手和手腕。未检测到真实手、ROI 丢失或公制腕深度无效时，
发送器不再更新目标；Gazebo 响应链保持最后目标，不虚构新运动。实体输出路径仍保留
0.40 s/0.65 s 的失效归零，但本 launch 不允许打开实体输出。不要同时启动第二份 HaMeR。

摄像头窗口打开后 Gazebo 控制始终为 `LOCKED`，人手运动只更新识别画面，不会驱动
机械臂。把手放到固定中立位置并保持稳定，然后在摄像头窗口按一次 `C`。这次按键
生成新的控制参考令牌；随后第一帧有效人手位姿会同时锁定手零位和当时的 `tool0`
初始位姿并启用跟随。不要在启动摄像头前调用 ROS 零位服务。

## 3A. 重新记录零位与人工观察

运行中如果需要重新记录零位，把手放到新的中立位置并保持稳定，然后在摄像头窗口
再次按 `C`。不需要第三个终端，也不需要调用 ROS 服务。

确认时会同时记录手零位和当时的 `tool0` 初始位姿。随后小幅、缓慢移动：靠近相机
对应 `base +X`，画面向右对应绿色轴 `base -Y`，画面向上对应 `base +Z`。相对
平移比例统一为 `0.6`，即手移 5 cm 生成机械臂 3 cm 目标；姿态采用工具局部
右乘且角度 `1:1`，即 MANO 手腕绕局部轴转 20° 生成机械臂绕记录时 tool0 对应
局部轴转 20° 的目标。手保持在偏移位置时机械臂
继续收敛并保持目标；手回到记录零位时，机械臂必须回到记录的初始 `tool0` 位姿。
本轮设置为：ROI 新帧权重 1.0；感知 SO(3) 在静止时保留自适应滤波，明确运动及
大角度有效 MANO 姿态以增益 1.0 直接跟随，不做软衰减或硬拒绝；本轮关闭前臂姿态
融合以单独验收 MANO 腕口；静止噪声带只关闭速度前馈，绝对位姿反馈持续工作；平移/旋转死区为 0，
Servo 关节速度低通为 0。常规人手
运动范围内的软件速度/加速度上限提高到 1 m/s、10 rad/s、50 m/s²、500 rad/s²，
目标角度仍保持 V3 的严格 1:1。Gazebo 仍受 URDF 关节硬范围和工作空间边界约束。
如果窗口显示 `CONTROL: LOCKED`，说明尚未成功按 C；显示
`GAZEBO CONTROL: ENABLED` 才表示控制已启用。当前 profile 不做碰撞减速，因此只应
用于空场景响应测试；需要恢复碰撞保护时退出此 launch，使用安全 profile 重启。

可观察：

```bash
rostopic hz /shared_teleop/hamer_pose
rostopic echo /shared_teleop/safe_twist
rosrun tf tf_echo base_link tool0
```

## 3B. 机器判定真人跟随验收

先在摄像头窗口按 `C` 并确认 Gazebo 已开始跟随，再回到该零位保持中性手势，然后执行：

```bash
cd /home/diu/myhandarmtest1
./scripts/run_live_human_gazebo_acceptance.sh --duration-s 30
```

验收器默认只读现有摄像头 C 参考，不再调用 ROS 服务重采零位。只有合成测试明确加入
`--confirm-reference` 时才允许它建立新参考，避免在机械臂已偏转时误把危险姿态设为零位。

倒计时结束后，在 30 s 内至少完成一次大于 3 cm 的平移和一次大于 15 度的腕部旋转。
脚本不会发布合成输入或 Servo 命令，只观察真人输入链。退出码为 0 且 JSON 顶层
`passed=true` 才通过。

验收器同时检查：

- 有效、递增序号的人手位姿和最低输入频率；
- 平移与旋转均产生非零安全命令；
- `tool0` 平移和旋转均超过最小量；
- 安全命令与实测末端速度的方向余弦达到门限；
- 仿真输出门开启，且无急停、TF 故障或危险 Servo 状态。

结果保存在脚本最后打印的 `acceptance_json=...`。若只需单独排查一个通道，可临时用
`--allow-translation-only` 或 `--allow-rotation-only`，但这两种降级结果不算完整六维
真人验收。

## 当前硬件限制

本机 D455 当前枚举在 480 Mbit/s USB2 hub 下，可用于降级开发，但不构成技术路线的
正式 USB3 验收。当前入口使用的是现有 HaMeR wrist-ring 粗位姿；Gazebo 链打通不表示
RGB-D KLT/Kabsch + SO(3)/SE(3) 最终感知门禁已经完成。
