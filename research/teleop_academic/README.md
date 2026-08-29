# Teleoperation academic track

本目录保存“单手腕 6D 位姿到机械臂末端”的学术主线。当前完成的是里程碑 1：确认位置—姿态串扰的第一处实际传播链并选择研究问题；尚未证明解决方法，也尚未完成抓取、抬升、移动和放置。

- `M1_BASELINE_AND_DECISION_ZH.md`：实际结果、证据边界与路线选择。
- `LITERATURE_EVIDENCE_MATRIX.md`：一次统一文献证据矩阵。
- `references.bib`：矩阵对应的唯一 BibTeX 库。
- `evidence/m1/`：pose-only 输入、Gazebo 数值摘要、边缘门控元数据和运行清单。
- `scripts/`：从已有记录生成 pose-only 输入、分析 ROS bag、探测边缘门控。

## 复现 M1 Gazebo 因果回放

环境是 Ubuntu 20.04、ROS Noetic 和 Gazebo 11。先在仓库根目录构建并加载工作区：

```bash
source /opt/ros/noetic/setup.bash
catkin_make --pkg handarm_moveit_demo
source devel/setup.bash
```

终端 A 启动无 GUI、安全链激活的 Gazebo 基线：

```bash
roslaunch handarm_moveit_demo shared_teleop_safe_demo.launch \
  gazebo_gui:=false \
  input_source:=external \
  enable_logger:=false \
  enable_gesture_demo:=false \
  mapping_profile:=camera_ground_axis_decoupled \
  live_human_velocity_profile:=true \
  world_name:="$(rospack find handarm_sim_demo)/worlds/handarm_ground_grasp.world"
```

等待终端显示完整自碰撞合同已激活。终端 B 创建一个明确的运行目录并开始记录：

```bash
mkdir -p /tmp/handarm_m1_reproduction
rosbag record \
  -O /tmp/handarm_m1_reproduction/ry_stage0_causal.bag \
  /shared_teleop/hamer_pose \
  /shared_teleop/trend_diagnostics \
  /shared_teleop/safe_twist \
  /joint_states \
  /gazebo/model_states \
  /handarm_sim_demo/target_contacts \
  /controller_gazebo_hand/command \
  /controller_gazebo_hand/follow_joint_trajectory/goal \
  /shared_teleop/hand_action \
  /servo_server/status
```

终端 C 发布同一段带任务标签的腕位姿：

```bash
source devel/setup.bash
rosservice call /shared_teleop/confirm_hand_reference "{}"
rosrun handarm_moveit_demo teleop_pose_replay.py \
  _input_csv:="$(pwd)/research/teleop_academic/evidence/m1/ry_stage0_labelled_wrist_input.csv" \
  _speed:=1.0 \
  _loop:=false \
  _start_delay_s:=1.0
```

服务调用把“下一帧有效手位姿”明确指定为人手和 `tool0` 的共同零参考；1 秒延迟只用于让 ROS publisher 与 subscriber 建立连接，避免第一帧在连接建立前丢失。缺少任一步时，映射器会按设计保持停控，不能重现本基线。

回放退出后停止终端 B，再生成因果摘要：

```bash
python3 research/teleop_academic/scripts/analyze_m1_causal_bag.py \
  /tmp/handarm_m1_reproduction/ry_stage0_causal.bag \
  --labelled-input-csv research/teleop_academic/evidence/m1/ry_stage0_labelled_wrist_input.csv \
  --endpoint-window-size 6 \
  --output /tmp/handarm_m1_reproduction/ry_stage0_gazebo_causal_summary.json
```

首次实测 bag 没有提交：它可由已提交的 pose-only CSV 和上述命令重新生成。原 bag 的哈希、大小、消息数、话题计数和代码哈希固定在 `evidence/m1/ry_stage0_gazebo_run_manifest.json`，用于区分“重新运行得到相近物理结果”和“原始证据文件完全相同”。独立复跑的派生结果保存在 `evidence/m1/ry_stage0_gazebo_reproduction_summary.json`。

## 数据与隐私边界

提交的 `ry_stage0_labelled_wrist_input.csv` 不含 RGB、depth、MANO 网格或身份字段，且清单明确写有 `control_authorized=false`。任务标签表示操作者被要求主要完成某类运动，不表示系统读取了脑内意图。

边缘探针需要本机已有的两帧 RGB 和离线 observer 记录，因此 Git 中只发布检测元数据、源文件哈希和探针脚本，不发布真人图像。它只能复现图像平移条件下的 presence 门控，不能替代新的真人边缘实验。没有冻结实验卡和操作者明确同意前，不采集新的真人数据。
