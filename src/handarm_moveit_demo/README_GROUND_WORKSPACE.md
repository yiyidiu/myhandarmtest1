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
- 平移和旋转当前独立映射：平移使用相机可见范围到机器人边界的比例，姿态保持
  1:1 并在当前配置的方向关节余量处截断，不再让二者互相压缩量程。
- 手回到 C 零位时，目标严格返回按 C 时捕获的机器人 `tool0` 位姿。
- Gazebo 每次重新启动后必须重新按一次 `C`。即使相机仍在发送上一轮的 C-token，
  机械臂也保持启动关节位，不会沿用旧参考突然运动。

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

默认的 `camera_workspace_calibration.yaml` 已使用本次真人日志得到的左右范围
`0.16/0.30 m`；上下和前后仍是临时范围 `0.18/0.25 m`。因此当前已经实现
“配置的人手边界映射到机器人前方工作空间边界”，但只有左右方向来自真人数据，
尚不能把六个方向都称为最终实测最大距离。

## 正式运行：两个终端

终端 1（推荐的新 EGM 式位置参考 profile；Gazebo、MoveIt Servo、UDP 接收和
地面物体一次启动）：

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch \
  with_ground_object:=true
```

它只替换 Gazebo 执行层：MoveIt Servo 的最新关节速度在 250 Hz 下积分成持续位置
参考，停手后保持该参考，并用开启重力的有限力矩 PI-D 关节伺服恢复扰动。HaMeR/
MANO、C 零位和相机范围映射全部沿用当前实现。这是 Gazebo 的 EGM 行为仿真，
不是 ABB 真机 EGM UDP 驱动。原速度方案没有删除，立即回退命令为：

```bash
roslaunch handarm_moveit_demo live_human_ground_gazebo_teleop.launch \
  with_ground_object:=true
```

终端 2使用上面的相机命令。Gazebo 完全启动且相机显示手部测量有效后，把手放在
固定中立位，按一次 `C`；在按 C 前机器人不会跟随。

生成 `/home/diu/myhandarmtest1/camera_workspace_measured.yaml` 后，再在命令中追加：

```bash
camera_workspace_calibration_file:=/home/diu/myhandarmtest1/camera_workspace_measured.yaml
```

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

新执行层的锁位/抗扰/回零验收见
[`docs/18_EGM_POSITION_REFERENCE_PROFILE.md`](docs/18_EGM_POSITION_REFERENCE_PROFILE.md)。

## 2026-08-24 小幅刚性与六维解耦试验

实时日志确认旧的联合六维映射把平移和姿态放在同一个归一化半径内，并继续用
一个全姿态 IK 比例同时缩放两者。故障帧中人手已经达到配置范围的 2.099 倍，
但机器人目标只保留 0.195，导致侧方位置和 Y 轴姿态同时被截短。

当前 Gazebo profile 做了以下可回退调整：

- 平移和姿态独立映射；关闭会同步截短两者的全姿态射线投影。
- 姿态保持 1:1，达到当前 C-zero 的方向关节余量后才截断。初始
  `[0,0,0,0,90,0]` 下，局部 Y 负/正方向约为 `76.2/30.8 deg`。
- 2026-08-24 真人日志的 3213 个独立有效帧给出相机 X 稳定负/正范围约
  `0.153/0.306 m`；试验配置取保守的 `0.16/0.30 m`，Y/Z 暂不改变。
- Gazebo 腕部速度 PID 的 P 值从 `5/2/3` 小幅调至 `6/2.5/3.5`，I/D
  保持为零。该参数只有重启 Gazebo 后才生效。

验收结果：62 项自动测试全部通过；独立 Gazebo 六方向验收中 X/Y/Z 姿态
响应比为 `0.998/0.996/0.998`，Servo 危险状态为空，回零误差为
`0.091 mm / 0.031 deg`。

补充现场结论：固定 C-zero 位置的纯局部 Y 轴 Gazebo 测试中，目标
`-60/+25 deg` 分别到达 `-59.97/+24.96 deg`，全过程 Servo 状态为 0。
因此当前 Y 轴量程和执行器增益本身已经生效。真人日志中，操作者认为的“大幅
绕 Y”并不是纯 Y 输入：在 `|Y| >= 45 deg` 的帧里，姿态 X/Y/Z 绝对值中位数
约为 `33/56/114 deg`，同时还带有约 `15/67/200 mm` 的三轴平移。这个混合
六维目标可能沿另一条关节路径接近奇异位，触发 Servo 状态 1/2，表现为姿态
变慢或停止。下一阶段若要消除该现象，应增加可关闭的姿态意图/单轴锁定模式，
而不是继续提高 PID 或关闭奇异保护；自由 6D 模式目前保持不变。

## 回退点

修改前快照（返回本轮工作开始前）：

`/home/diu/myhandarmtest1/backups/teleop_before_workspace_mapping_20260823_204342.tar.gz`

SHA256：

`44da6db40e1f0ed18d706a93c07d93299cf891cd9e676a049f4a990864837884`

本轮全部验收通过后的快照：

`/home/diu/myhandarmtest1/backups/teleop_ground_workspace_passed_20260824_002737.tar.gz`

SHA256：

`157865c9db15dee09daa2739ad5a5feaa894e1c4eef1edf4a3fc757aca3c1504`

本次小幅调整前快照：

`/home/diu/myhandarmtest1/backups/teleop_before_stiffness_pose_decoupling_20260824_043042.tar.gz`

SHA256：

`ad54e4f0f4dd43cb7714fd178eaa75065a44ee7afa313b1d0228c5eb9b662a45`

本次小幅调整验收通过后的快照：

`/home/diu/myhandarmtest1/backups/teleop_small_stiffness_pose_decoupling_passed_20260824_044327.tar.gz`

SHA256：

`daa8aa9544802b2ec4f6dad051451fe312b0e3a3f944ecc774c18df99447a651`

启动零位保护和工作空间边界映射验收通过后的快照：

`/home/diu/myhandarmtest1/backups/teleop_startup_zero_workspace_mapping_passed_20260824_050420.tar.gz`

SHA256：

`daa59f27ddd38ab9b42bb656ef7e55ffd2038b937bce07bcbcfe567ed6f86ac3`

新增 EGM 式位置参考层前、包含完整旧速度 profile 的快照：

`/home/diu/myhandarmtest1/backups/teleop_before_egm_position_reference_20260824_154746.tar.gz`

SHA256：

`47258b6966f683754e3243ee60ba3b429f83eb05e1b7631facf40617ce5a04ea`

先验证：

```bash
cd /home/diu/myhandarmtest1/backups
sha256sum -c teleop_before_workspace_mapping_20260823_204342.sha256
sha256sum -c teleop_ground_workspace_passed_20260824_002737.sha256
sha256sum -c teleop_before_stiffness_pose_decoupling_20260824_043042.sha256
sha256sum -c teleop_small_stiffness_pose_decoupling_passed_20260824_044327.sha256
sha256sum -c teleop_startup_zero_workspace_mapping_passed_20260824_050420.sha256
sha256sum -c teleop_before_egm_position_reference_20260824_154746.sha256
```

不要直接覆盖当前工程。需要回退时，先解压到单独目录进行比较：

```bash
rollback_review="$(mktemp -d /tmp/handarm_rollback_review.XXXXXX)"
tar -xzf /home/diu/myhandarmtest1/backups/teleop_before_workspace_mapping_20260823_204342.tar.gz \
  -C "$rollback_review"
echo "$rollback_review"
```
