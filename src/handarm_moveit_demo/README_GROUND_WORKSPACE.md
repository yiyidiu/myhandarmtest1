# D455 人手范围到 IRB120 地面工作空间映射

## 当前实现

新入口 `live_human_ground_gazebo_teleop.launch` 独立于原来的
`live_human_gazebo_teleop.launch`。原入口仍默认使用 `current_linear`，因此
这轮工作空间实验不需要覆盖旧方案。

新入口使用以下关系：

- D455 光学坐标 `+X`（画面向右）映射到机器人基座 `-Y`。
- D455 光学坐标 `+Y`（画面向下）映射到机器人基座 `-Z`。
- D455 光学坐标 `+Z`（远离相机）映射到机器人基座 `-X`。
- 每个相机正、负方向分别标定。手从 C 零位到稳定识别边缘的位移被归一化为
  `[-1,+1]`，再沿相应机器人方向映射到前方、左右和地面截断后的可用边界。
- 姿态不拆成欧拉角累加，而是在 C 零位局部坐标中使用 SO(3) 相对旋转。
  人手各正、负方向的最大可识别角分别映射到机器人该方向的可达角。
- 平移和旋转同时发生时组成一个六维方向，共享同一个归一化半径。在线
  `/compute_ik` 沿这条六维射线投影，避免分别达到上限后组合成不可达目标。
- 手回到 C 零位时，目标严格返回按 C 时捕获的机器人 `tool0` 位姿。

IRB120 的原始 FK 点云外包络来自 100 万组关节样本；实际遥操作只开放
`base_link +X` 前方区域。地面场景中考虑了三指手爪尺寸：C 零位姿态下的
有效 `tool0` 下界为 0.110 m（配置值 0.100 m 加 10 mm 边界余量），不会再让
目标持续压入 Gazebo 地面。

## 场景

- `with_ground_object:=true`：只有地面、机器人和三个落地物体；没有桌子。
- `with_ground_object:=false`：只有地面和机器人，用于观察工作空间。
- 机器人基座位于世界坐标原点、地面 `z=0`。
- 三个物体中心分别为 `(0.48,0,0.051)`、`(0.40,0.28,0.041)`、
  `(0.40,-0.28,0.061)` m，物体底面与地面之间保留 1 mm 沉降间隙。

## 首次相机范围标定

定量验收前必须生成操作者/相机实测文件。此步骤不能同时启动 Gazebo，因为
标定器需要独占 UDP 5010。

终端 1：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun handarm_moveit_demo calibrate_camera_workspace_udp.py \
  --neutral-s 3 \
  --explore-s 30 \
  --output /home/diu/myhandarmtest1/camera_workspace_measured.yaml
```

终端 2：先把手保持在之后要按 C 的中立位置，再启动相机。前三秒保持不动；
看到 `EXPLORE` 后，依次充分探索左右、上下、前后平移以及绕腕部 X/Y/Z 的
正反向旋转，始终保持整只手可见。

```bash
cd /home/diu/myhandarmtest1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n hamer_rtx2060 \
python perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe \
  --mesh-renderer teleoperation-core \
  --control-reference mano-wrist-ring \
  --hand-presence-timeout-s 0.25 \
  --teleop-udp-host 127.0.0.1 \
  --teleop-udp-port 5010
```

默认的 `camera_workspace_calibration.yaml` 只是可运行的仿真占位范围，不应作为
真人最大范围的定量结论。

## 正式运行：两个终端

终端 1（Gazebo、MoveIt Servo、UDP 接收和地面物体一次启动）：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_teleop.launch \
  camera_workspace_calibration_file:=/home/diu/myhandarmtest1/camera_workspace_measured.yaml \
  with_ground_object:=true
```

终端 2使用上面的相机命令。Gazebo 完全启动且相机显示手部测量有效后，把手放在
固定中立位，按一次 `C`；在按 C 前机器人不会跟随。

若只是验证结构、尚未生成实测范围，可省略
`camera_workspace_calibration_file:=...`，但此时终端会明确提示 provisional。

## 可重复验收

离线检查六个平移方向是否到达配置边界：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun handarm_moveit_demo check_ground_workspace_mapping.py
```

Gazebo 已启动、相机进程已退出时，可用确定性 UDP 输入检查整条控制链。它会主动
运动仿真机器人，且会拒绝在非 Gazebo 环境运行：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun handarm_moveit_demo validate_ground_workspace_gazebo.py \
  --calibration /home/diu/myhandarmtest1/camera_workspace_measured.yaml \
  --output /tmp/handarm_ground_workspace_gazebo.json
```

真人链路验收：先在相机窗口按 C，然后另开终端运行：

```bash
cd /home/diu/myhandarmtest1
./scripts/run_live_human_gazebo_acceptance.sh --duration-s 30
```

## 回退点

修改前快照：

`/home/diu/myhandarmtest1/backups/teleop_before_workspace_mapping_20260823_204342.tar.gz`

SHA256：

`44da6db40e1f0ed18d706a93c07d93299cf891cd9e676a049f4a990864837884`

先验证：

```bash
cd /home/diu/myhandarmtest1/backups
sha256sum -c teleop_before_workspace_mapping_20260823_204342.sha256
```

不要直接覆盖当前工程。需要回退时，先解压到单独目录进行比较：

```bash
rollback_review="$(mktemp -d /tmp/handarm_rollback_review.XXXXXX)"
tar -xzf /home/diu/myhandarmtest1/backups/teleop_before_workspace_mapping_20260823_204342.tar.gz \
  -C "$rollback_review"
echo "$rollback_review"
```
