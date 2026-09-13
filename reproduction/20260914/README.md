# 本次完整保存的内容与使用边界

这里记录当前实际运行版，而不是新控制方案。原来重新下载的工程停在 9 月 12 日，缺少之后的延迟和控制修复；旧 README 又指向历史入口。本次通过固定标签和统一原入口避免再次混用。

## 当前源码和实际配置

- 机械臂、视觉与控制参数保持 da8c63d 基线；其中控制/视觉实现为 dac3518。
- 当前执行的 HaMeR SDK 副本随源码保存，文件内容与原环境中的副本一致。
- observed_ros_parameters.yaml 是实际启动后读取的参数记录。路径已替换为 `${WORKSPACE}`，供核对；**不要把整个文件直接 rosparam load**，参数仍由原 launch/config 生成。
- system_versions.tsv 记录关键 ROS、Gazebo、MoveIt 版本；packages.json 记录各 Python 环境中的实际版本。
- exact_original_binaries.json 用于追溯原电脑执行程序。新电脑需从同一源码 Release 编译，不直接使用带旧绝对路径的 devel。
- cpu.json/gpu.txt 记录原机硬件。当前原入口仍保留原有 CPU 调度逻辑，本发布没有偷偷更换它。其他 CPU 的拓扑和负载不一定与原机相同。

## 已有相同依赖的电脑

仅将 teleop_machine.env.example 复制为 teleop_machine.env，并填写本机环境/模型路径。随后按根目录 README 编译，运行原来的 scripts/run_grasp_feedback_recording.sh。

## 缺少原 Python 环境时的备份还原

Release 附件只是原环境备份，不是新程序或新安装器。两个环境各有三个分卷，总约 7 GB。先下载全部六卷与 SHA256SUMS，进入分卷所在目录后核对：

```bash
sha256sum -c SHA256SUMS
```

下面以本人目录 teleop_envs 为例。目标目录必须是新建的，不要覆盖现有环境：

```bash
mkdir -p "$HOME/teleop_envs/hamer_rtx2060" "$HOME/teleop_envs/mediapipe_env"
cat hamer_rtx2060.tar.gz.part* | tar -xz -C "$HOME/teleop_envs/hamer_rtx2060"
cat mediapipe_env.tar.gz.part* | tar -xz -C "$HOME/teleop_envs/mediapipe_env"
"$HOME/teleop_envs/hamer_rtx2060/bin/python" "$HOME/teleop_envs/hamer_rtx2060/bin/conda-unpack"
"$HOME/teleop_envs/mediapipe_env/bin/python" "$HOME/teleop_envs/mediapipe_env/bin/conda-unpack"
"$HOME/teleop_envs/hamer_rtx2060/bin/python" -m pip install --no-deps smplx==0.1.28
"$HOME/teleop_envs/mediapipe_env/bin/python" -m pip install --no-deps smplx==0.1.28
```

在 teleop_machine.env 中，将两个 Python 环境路径填写为这些位置。环境展开后不能随意移动，再换电脑/目录时应从原分卷重建内部路径。以上环境来自相同基线，并已在独立目录完成重定位、模型加载和完整启动检查。

[Conda-pack 迁移说明](https://conda.github.io/conda-pack/)；[SMPL-X 原始许可](https://github.com/vchoutas/smplx/blob/main/LICENSE)。SMPL-X 受分发限制，不在公开附件内，从原始 PyPI 分发源安装；精确 wheel URL 和 SHA-256 在 runtime_reference.json 中。

## 模型文件

当前代码需要 `_DATA/hamer_ckpts/checkpoints/hamer.ckpt`、`_DATA/hamer_ckpts/model_config.yaml`、`_DATA/data/mano_mean_params.npz` 和 `_DATA/data/mano/MANO_RIGHT.pkl`。内容校验记录在 runtime_reference.json。

HaMeR 权重和配置可按 [原作者说明](https://github.com/geopavlakos/hamer/blob/3a01849f4148352e9260b69bf28b65d1671a4905/README.md) 获取，或复用自己的现有模型文件。

MANO_RIGHT.pkl 受 [MANO 官方许可](https://mano.is.tue.mpg.de/license.html) 限制，不能放进公开仓库。你可将原机自己的授权文件带到自己控制的新电脑使用。原机文件位于：

`/home/dongtian/teleop_versions/20260914_dac3518/assets/_DATA/data/mano/MANO_RIGHT.pkl`

把整个现有 `_DATA` 放到新工程 perception_hamer 下面，或者在 teleop_machine.env 指向实际模型目录即可。不要把另一版本模型当成本次同一版本。

## 系统和性能

目标是 Ubuntu 20.04 x86_64 + NVIDIA、ROS Noetic、Gazebo 11.15.1，关键包版本有记录。备份不包含操作系统、显卡内核驱动、USB 权限或账号组设置；其他电脑仍需这些基础条件。

固定源码、编译方式、依赖和有效参数能避免下载遗漏或版本混用，但不能消除 CPU/GPU 性能、拓扑、驱动、相机 USB 和后台负载差异。独立目录能启动不代表另一台电脑已经通过跟随延迟验收。

## 本轮验证范围

- 原算法源码目录 src 和 perception_hamer 与 da8c63d 保持一致。
- 原环境备份在独立目录重定位后，HaMeR/MANO 能在 GPU 加载。
- 相机、Gazebo、两个观察窗和录制均已在独立目录启动；过程中未按 C。
- 去掉 Git 元数据后也能启动并封装录像，支持源码 ZIP 的录制入口。
- 本轮没有进行抓取验收，也没有宣称全部视觉或运动问题已经修复。
