# 当前遥操作完整工程源码 · 2026-09-14

这份快照固定当前实际运行的工程代码、有效配置、启动命令和依赖来源，供下载后继续使用同一版。控制实现基线为 `dac3518`，本机保存基线为 `da8c63d`。

**固定标签：[teleop-complete-code-20260914](https://github.com/yiyidiu/myhandarmtest1/releases/tag/teleop-complete-code-20260914)。** 不要用旧 main 或 9 月 12 日的标签替代这个快照。

控制器、视觉处理、滤波、映射、碰撞几何、物体布局和 CPU 调度逻辑保持本机保存版。工程未引入新的安装器或新的 CPU 调度方案。本次补齐外部 HaMeR 实际源码、运行配置记录、依赖版本，以及换电脑时的路径配置和源码 ZIP 录制兼容。

## 下载当前工程

```bash
git clone --branch teleop-complete-code-20260914 --single-branch https://github.com/yiyidiu/myhandarmtest1.git
cd myhandarmtest1
```

也可以在该标签的发布页下载 Source code ZIP 并完整解压。Release 的环境备份附件是可选依赖备份，工程源码以此标签的 Source code 为准。

## 运行命令保持原入口

在已配置对应 ROS、Python 环境和模型文件的电脑上：

```bash
bash scripts/run_grasp_feedback_recording.sh
```

相机、Gazebo、两个抓取观察窗、视频录像、RGB-D、控制数据和 IMU 记录一起启动。张手放稳后按 C 才启用控制；按 Q 退出并等待录制完成。数据保存在本工程 `.runtime/teleop_feedback_日期T时间_随机后缀/`。

## 换电脑时只配置本机路径并编译

目标平台是 Ubuntu 20.04 x86_64 + NVIDIA 显卡、ROS Noetic、Gazebo 11。在新电脑第一次使用，复制路径模板并按实际模型与环境位置编辑：

```bash
cp teleop_machine.env.example teleop_machine.env
```

模板只配置 Python、模型和 SDK 路径，不调整控制参数。它默认使用本人的 `$HOME/anaconda3/envs/hamer_rtx2060`、`mediapipe_env` 和本工程 `perception_hamer/_DATA`；你的依赖不在那里时修改对应路径即可。新电脑没有对应依赖时，先按 [完整性与复现说明](reproduction/20260914/README.md) 还原固定环境，不要直接装一套未锁版本的依赖。

重新下载或切换源码后，必须在当前工程编译，不能复制旧目录的 devel：

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release -j3 -l3
```

随后仍运行上面的原入口。当前电脑原有的固定保存版及开发版继续保留。

## 本次保存得更完整的部分

- 全部机械臂/三指手控制源码、自定义 MoveIt Servo、qpOASES 和 Gazebo 插件源码。
- 相机处理、坐标映射、滤波、场景、机器人模型、碰撞形状和标定配置。
- 实际安装环境里的 HaMeR 源码副本及 MIT 许可，位于 `third_party/hamer_runtime`。
- 实际 ROS 参数、关键系统包版本、两个 Python 环境版本和模型内容校验，位于 `reproduction/20260914`。
- Release 中保存两个原运行环境的可迁移备份分卷，避免在另一台电脑上凭感觉重新安装依赖。

MANO 授权模型无法放入公开仓库，SMPL-X 也不重新分发。它们的原始来源、导入方法和版本列在复现说明中。训练权重按原始来源提供，不把历史录屏、临时实验和账号凭据混入源码。

相同代码不能消除硬件和系统负载差异。本次固定的是代码、配置和运行依赖，不能据此承诺不同电脑有完全相同的延迟，也不表示此前所有控制问题均已修好。
