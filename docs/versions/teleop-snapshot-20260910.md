# 2026-09-10 遥操作代码快照

分支 `codex/teleop-snapshot-20260910`；固定标签 `teleop-snapshot-20260910`。快照包含截至 9 月 10 日当前工作目录的完整源码、配置、Gazebo 场景、必要模型与第三方源码。为完成版本封装，本次仅补齐 IMU 记录依赖、可配置的本机资产路径和运行说明，没有新增控制算法调参。

## 固定配置

| 项目 | 快照设置 |
|---|---|
| 控制器 | `TELEOP_MOTION_CONTROLLER=constrained` |
| 机械臂初始关节角 | `[0,0,0,0,π/2,0]` |
| 三指根初始角度、OPEN 终点 | 均为 URDF 数值 5° |
| 平移增益 | 前后 1.6、左右 1.6、上下 1.2 |
| 左偏航 | 人手约 35° → 末端名义90° |
| 右偏航 | 人手约 70° → 末端名义90° |
| 偏航目标限幅 | ±88°，保留原有 2° 超调余量 |
| 横滚增益 | 3 倍 |
| 前方纸盒 | 6.9 × 7.6 × 11.67 cm，中心 X=0.425 m、Y=0 |
| 左侧圆柱 | 直径6 cm，高15.67 cm，中心 X=0.24 m、Y=−0.41 m |
| IMU | 记录与分析，不进入控制闭环 |

## 已观察到的结果与限制

- 同一段原始 RGB-D 的张手观测，在旧置信度逻辑下持续保留 CLOSE；修正背景 padding 的重复扣分后可触发三指同步释放。
- 实际 Gazebo 驱动已完成三指释放，5° OPEN 目标下稳定角度约5.64–5.72°，彼此差约0.08°。这不是带物体抓取验收。
- 在 `[0.38,±0.16,0.35] m` 两个可达测试点完成平移、左右偏航、转回和返回 C，末端停稳位置误差约0.20–0.39 mm；这些数值不是动态跟踪误差。
- 缩短后的纸盒上方位姿 `[0.425,0,0.35815] m` 已在实际仿真到达并返回，末端停稳误差约0.32 mm、姿态误差约0.05°。
- 多处历史“卡住”包含超出位姿可达范围、腕部与掌座接近保留间距、接近伸直奇异位形。没有通过关闭自碰撞保护绕过这些限制。
- 左侧圆柱完整从竖直正上方进入仍略超出连杆伸展条件；侧向进入是不同的位姿条件，尚未完成真人抓取验收。
- 这些结果不能证明所有真人卡顿、视觉误差或抓取问题均已消失。详细时间片段见 [诊断记录](teleop_diagnosis_20260910.md)。

## 首次检出与依赖

本机已具备 Ubuntu20.04、ROS Noetic、Gazebo、NVIDIA/CUDA、相机 SDK 和已有 Conda 环境。新机器仍须配置这些运行依赖；此快照不是环境镜像。

```bash
git clone --branch teleop-snapshot-20260910 --single-branch https://github.com/yiyidiu/myhandarmtest1.git myhandarmtest1-teleop-snapshot-20260910
cd myhandarmtest1-teleop-snapshot-20260910
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release -j3 -l3
source devel/setup.bash
```

HaMeR / MANO 模型资产及原始录像保留在本机，未纳入 Git。默认资产目录沿用当前机器，启动脚本允许下列环境变量覆盖路径：

```bash
export TELEOP_CONDA_EXECUTABLE=/实际路径/anaconda3/bin/conda
export TELEOP_HAMER_ASSET_ROOT=/实际路径/perception_hamer/_DATA
export TELEOP_IMU_PYTHON=/实际路径/带numpy和pyserial的Python解释器
```

HaMeR Conda 环境名沿用 `hamer_rtx2060`；环境记录位于 `perception_hamer/environment/hamer_rtx2060.yml`。HaMeR 资产目录应包含 `hamer_ckpts/checkpoints/hamer.ckpt` 和 `data/mano/MANO_RIGHT.pkl`。录屏使用系统 Python 的 GStreamer / GI 与 X11；IMU 接收使用 WT901 BLE USB 接收器、串口权限及 Python 的 numpy、pyserial。

IMU 读取与时间重建脚本已纳入 `scripts/feedback_capture/`，录制入口不再引用另一个工程的研究脚本。时间重建输出仍是估计采样时刻，不是真实设备时间戳或已完成的相机外参标定。

## 运行、固定版本与恢复

```bash
TELEOP_MOTION_CONTROLLER=constrained ./scripts/run_teleop_feedback_recording.sh
```

当前本机运行目录仍为 `/home/dongtian/myhandarmtest1-teleop-repair`。录像保存于 `.runtime/teleop_feedback_日期T时间_随机后缀/`；`commit.txt` 记录本次运行使用的提交，`code_changes.patch` 记录相对提交的本地改动。

后续修改建议在新分支进行；该固定标签保持不变。若需重新取得本快照，可创建独立目录，避免覆盖正在修改的工作区：

```bash
git fetch origin tag teleop-snapshot-20260910
git worktree add --detach ../myhandarmtest1-snapshot-20260910 teleop-snapshot-20260910
cd ../myhandarmtest1-snapshot-20260910
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release -j3 -l3
source devel/setup.bash
TELEOP_MOTION_CONTROLLER=constrained ./scripts/run_teleop_feedback_recording.sh
```

以上恢复入口须在退出当前相机和遥操作后运行。固定标签保存的是源码与配置，切换后必须重新编译，不能复用其他版本的旧构建目录。

用户此前指定保留的 `before_wrist_recovery_20260909T174835` 本地备份仍保留；它与本次固定标签是不同的版本。`legacy` 参数只切换控制器路径，不等于恢复某个历史源码版本。

## 第三方来源

- qpOASES 3.2.1：见 `src/handarm_moveit_demo/third_party/qpOASES/SOURCE.md` 和其 LICENSE。
- MoveIt Servo 1.1.16 及局部修改：见 `src/moveit_servo/UPSTREAM_LOCAL_PATCH.md`，源文件保留上游许可说明。
- Gazebo 控制后端来源许可：`src/handarm_sim_demo/LICENSE.gazebo_ros_control`。
