# 第一阶段人工检查命令

以下命令均为仿真或离线检查，不启用实体 ABB。每个新终端都先执行：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

## 1. 一键自动测试

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_tests.sh
```

预期控制核心至少显示 `31 tests, 0 errors, 0 failures`，并且完整
`perception_hamer` 测试集全部通过。

## 2. 最简单的纯离线六维验证

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_offline_demo.sh
```

预期 JSON 中包含：

```text
control_loop_actual_hz: 50.0
six_axes_simultaneous_seen: true
timeout_zero_verified: true
real_robot_commands_sent: 0
```

命令会打印 CSV 的临时路径，可直接用表格软件查看六轴原始/处理后速度和原因字段。

## 3. 启动安全 Gazebo 演示

终端 A：

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_safe_demo.sh
```

保持终端 A 运行。终端 B 检查 50 Hz 输出：

```bash
rosservice call /shared_teleop/confirm_hand_reference
rostopic hz /shared_teleop/safe_twist
```

确认参考前必须把手保持稳定；服务调用后的下一帧有效 HaMeR 位姿会成为锁定零位。需要重设零位时再次保持手不动并调用同一服务。清除参考并立即停止跟随：

```bash
rosservice call /shared_teleop/clear_hand_reference
```

按 `Ctrl-C` 结束 `rostopic hz`，再分别查看六维命令和安全诊断：

```bash
rostopic echo -n 1 /shared_teleop/raw_hand_command
rostopic echo -n 1 /shared_teleop/safe_twist
rostopic echo -n 1 /shared_teleop/output_diagnostics
```

打印当前方向映射表：

```bash
rosrun handarm_moveit_demo check_direction_mapping.py
```

连续显示手运动经映射后的基座六维速度：

```bash
rosrun handarm_moveit_demo direction_mapping_monitor.py
```

合成输入会同时改变三轴平移和三轴转动。安全方向仿真使用零重力速度接口以避免把无 ABB 重力补偿的下坠误认为手命令；碰撞检查仍开启。

一条命令完成实际 Gazebo `tool0` 六轴方向测量：

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_gazebo_direction_validation.sh
```

预期 JSON 为 `passed: true`，并分别列出 `translation_base_x/y/z` 与 `rotation_tool_x/y/z` 的实际 TF 位移/旋转向量。任何方向反向、交叉轴过大或 Servo 急停都会以非零状态退出。

## 4. 检查手势隔离和单次机械手命令

默认合成源每 10 秒在第 6.0～6.7 秒稳定输出一次 CLOSE。终端 B：

```bash
rostopic echo /shared_teleop/gesture_diagnostics
```

另一个终端：

```bash
rostopic echo /shared_teleop/mock_hand_command
```

预期先看到约 0.3 秒 `GESTURE_DEBOUNCE`，随后 `GESTURE_TRIGGERED/GESTURE_HOLD`；每次稳定手势只出现一条 `CLOSE` mock 命令。这里没有连接真实三指手。

## 5. 检查俯抓、侧抓和取消

```bash
rostopic pub -1 /shared_teleop/assist_request std_msgs/String "data: 'top'"
rostopic echo -n 1 /shared_teleop/assist_diagnostics

rostopic pub -1 /shared_teleop/assist_request std_msgs/String "data: 'side'"
rostopic echo -n 1 /shared_teleop/assist_diagnostics

rostopic pub -1 /shared_teleop/assist_request std_msgs/String "data: 'cancel'"
```

诊断中的 `candidates` 会列出候选角距离和 `ik_feasible`。最近候选和抓取中心保持由自动测试验证；本轮六轴方向验收不等于 top/side 姿态辅助已经在 Gazebo 到位。

## 6. 检查输入超时停止

在安全演示运行时：

```bash
rosnode kill /synthetic_hamer_pose_publisher
sleep 1
rostopic echo -n 1 /shared_teleop/output_diagnostics
rostopic echo -n 1 /shared_teleop/safe_twist
```

预期诊断包含 `INPUT_TIMEOUT_ZERO`，Twist 六个分量均为零。完成后在终端 A 按 `Ctrl-C`，重新运行安全演示即可恢复合成输入。

## 7. 检查急停锁存

```bash
rostopic pub -1 /shared_teleop/emergency_stop std_msgs/Bool "data: true"
rostopic echo -n 1 /shared_teleop/output_diagnostics
rostopic echo -n 1 /shared_teleop/safe_twist
```

预期包含 `EMERGENCY_STOP_LATCHED` 且六轴全零。解除时必须先撤销急停输入，再复位锁存：

```bash
rostopic pub -1 /shared_teleop/emergency_stop std_msgs/Bool "data: false"
rosservice call /shared_teleop/reset_emergency_stop
```

## 8. 回放现有 HaMeR 记录

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_safe_demo.sh \
  input_source:=hamer_recording \
  hamer_session:=$(pwd)/datasets/development_usb2/hamer_palm_stability/DEV_HAMER_TRANSLATION_20260813T184556 \
  hamer_replay_speed:=1.0
```

该记录的实际结果是 415 个有效 HaMeR 帧中发布 161 个公制腕位姿，另外 254 帧因腕像素附近深度不足被跳过。

## 9. D455 + HaMeR 在线输入，但仍只驱动 Gazebo

终端 A：

```bash
cd /home/diu/myhandarmtest1
./scripts/run_stage1_safe_demo.sh gazebo_gui:=true input_source:=udp
```

终端 B：

```bash
cd /home/diu/myhandarmtest1
conda run --no-capture-output -n hamer_rtx2060 \
  python perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe \
  --hand-presence-timeout-s 0.25 \
  --teleop-udp-host 127.0.0.1 \
  --teleop-udp-port 5010
```

实时 MANO 画面由 `mediapipe_env` 的 Qt5 OpenCV sidecar 显示；这是因为 HaMeR
环境固定使用无 GUI OpenCV。窗口按 `q/Esc` 退出，按 `r` 重新选择 ROI。若窗口
被单独关闭，HaMeR 推理和 UDP 安全链仍继续，终端会明确记录显示进程退出。
启动预检阶段必须先在彩色画面中看到完整真实手；连续检测稳定后会自动选择一个
活动手并启用 HaMeR，不需要按 `c`。另一只手同时进入画面时只作为候选，不得
抢走当前裁剪框；活动手离开、另一只手连续稳定出现后才会自动换手。

启动后把手完全移出画面，最多经过当前 MediaPipe 检测延迟（检测结果超过 0.25
秒也会超时）应看到 `REAL HAND presence=NO`、`ROI ... valid=False` 和
`MANO mesh=OFF`，Gazebo 随后由输入 watchdog 平滑停住。此时画面不得保留黄色
裁剪框或旧手模型。重新伸手需连续两次检测一致才恢复新的裁剪框和 MANO；旧
模型不会跨越“无手区间”复用。

只需要在 YAML 中调整方向映射，不需要相机到机械臂的完整外参：

```text
mapping.translation_matrix
mapping.rotation_matrix
mapping.translation_gain
mapping.rotation_gain
```

所有演示使用 `Ctrl-C` 停止。不要启动或连接任何实体 ABB 输出 launch；本阶段没有提供实体运行命令。
