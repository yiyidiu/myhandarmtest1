# MediaPipe 对照副本

当前实时入口与六组实验均固定为左手，检测器的原始左右手分类仅作诊断记录。
固定左手后重新计算的结果见 [固定左手六组对比报告](docs/six_group_fixed_left_20260917/固定左手六组对比报告.md)。
该轮实验继续关闭时间滤波与掌面辅助，旧自动手别结果单独保留。

2026-09-17：新增 [A—F 六组无滤波对比实验报告](docs/six_group_unfiltered_20260917/六组无滤波对比实验报告.md)。
该实验独立重放缓存录像，关闭掌面辅助与手部时间滤波；F 组仅使用真实深度。
六组定义、指标、原始结果、参考限制及复现命令均在报告中。下文为原实时入口说明。

从 `/home/dongtian/myhandarmtest1-teleop-repair` 的 `a0cae6c` 复制。
原工程和 `/home/dongtian/teleop_versions/20260914_dac3518` 未修改。
源码、场景、机械臂模型和控制配置已复制，catkin 在副本内重新编译。
历史录像只读使用原目录；没有再复制原工程约 314 GB 的历史运行数据。

## 运行

进入本目录后，MediaPipe 三维关键点模式（默认，不录制）：

```bash
bash scripts/run_grasp_feedback_recording.sh --no-recording
```

二维关键点结合真实 RGB-D 的模式：

```bash
TELEOP_HAND_BACKEND=mediapipe bash scripts/run_grasp_feedback_recording.sh --no-recording
```

需要同时录屏、IMU、RGB-D 和 ROS 数据时，去掉 `--no-recording`。
两次运行之间先按 Q 退出相机，等待仿真结束。不要同时运行两个工程。
打开后先张手、放稳、按 C；按 C 前不发送有效控制位姿。Q 退出。
按 C 后窗口显示 `C RECEIVED`，成功后显示 `CONTROL ON - zero set`，
终端同时提示“零位已建立”。未完成时终端会说明是位姿不可用、张手读数或稳定性要求未满足。
重新标定会在下一帧新观测到达时重建姿态滤波历史，避免旧姿态使新零位被大角度保护持续拒绝；
正常运动中的跳变保护、标定稳定性要求与机器人控制配置不变。
保存路径由原录制入口打印，在本副本的 `.runtime/teleop_feedback_*` 下。

## 两种模式的定义

- `mediapipe`：同一帧的二维腕点、四个掌指关节根部，加对齐后的真实深度，构建掌部坐标系。
- `mediapipe3d`：由 `multi_hand_world_landmarks` 的腕点和四个掌指关节根部构建掌部坐标系。
  这些坐标的原点在手附近，不是相机原点，所以腕点的绝对位置仍来自 RGB-D。
- 两种模式使用同一套 MediaPipe Hands Full 模型；不存在本实验另外训练的“MediaPipe 3D 网络”。
- 位置参考改为 MediaPipe 关键点 0，姿态参考为腕点与掌指根部组成的刚体近似。
  坐标列依次为小指指根指向食指指根、腕点指向指根中心、两者叉积。
  在相机光学坐标下输出右手坐标系；C 的相对旋转仍由原映射变换到机器人。
- 完全不加载 HaMeR 网络、MANO 权重、旧掌面辅助、IMU 拟合校正或前臂融合。
  RGB-D 模式的深度测量是自身位姿构建的一部分，不是原来的 HaMeR 掌面辅助混合。
- 三指开合改用各手指内部的弯曲角，排除“腕点到指根”的夹角。
  长指使用 PIP/DIP，拇指使用 MCP/IP；输入定义为 `mediapipe_digit_internal_bend_over_pi_v1`。
  当二维关键点明显伸直、预测三维指节却明显弯曲时，该帧只保持已有开合指令，腕部位姿仍可更新。
  原机械手开闭角度、映射参数、执行控制和自碰撞配置不变。

## 保留的控制链

保留 C 确认与同帧参考、原 SO(3) 滤波参数、ROS 接口、位姿映射及增益、QP/Servo 切换、
自碰撞保护、250 Hz 参考生成、EGM、场景、抓取视角和录制入口。
Gazebo 环境真值进入控制器的通道仍关闭。
ROS 消息和 UDP schema 中历史上的 `hamer` 名字保留以兼容接口；实际来源明确标为 `MEDIAPIPE_*`。
新前端不因接口名字而加载 HaMeR。

## 环境与复现

MediaPipe Python 默认 `/home/dongtian/anaconda3/envs/mediapipe_env/bin/python`；
可用 `TELEOP_MEDIAPIPE_PYTHON` 指定其它 Python 3.10 环境，依赖见 `requirements-mediapipe.txt`。
HaMeR 仅作为离线对照保留原代码和资产路径，MediaPipe 运行不需要其检查点。

对照程序按记录的 RGB 帧号及时间取图，再检查真实深度与 RGB 相差不超过半个相机帧周期。
数据库读取使用 `rosbags` 解码和原 RealSense `rs.align`，不改变相机几何算法。
已有标注、IMU 时间偏移直接沿用，没有对新模型拟合或调参。

```bash
# 依次执行，避免两个模型争用计算资源。
TELEOP_PY=/home/dongtian/anaconda3/envs/mediapipe_env/bin/python
OPENBLAS_NUM_THREADS=1 "$TELEOP_PY" perception_mediapipe/compare_recordings.py mp --period 0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 8-11 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/compare_recordings.py hamer
"$TELEOP_PY" perception_mediapipe/summarize.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/measure_speed.py
"$TELEOP_PY" perception_mediapipe/render_report.py
```

本轮图表与逐动作结果也已存入版本控制，见 [对比报告](docs/mediapipe_comparison_20260915/对比报告.md)。
重新计算时的中间结果位于 `.runtime/comparison/report/`。IMU 比较的是跨时刻**总转角**，
不等于逐轴姿态真值；位置指标是静止散布，不是绝对定位误差。模型处理耗时也不等于机器人闭环延迟。

参考：
- [MediaPipe Hands 输出定义](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)
- [RealSense 官方软件设备对齐示例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-with-software-device.py)
