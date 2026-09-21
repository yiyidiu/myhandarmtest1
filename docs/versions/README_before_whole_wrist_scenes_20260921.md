# 2026-09-17 五点掌面坐标遥操作固定版

固定标签：**`teleop-palm-axes-20260917`**。请下载这个标签或同名 Release；旧的9月12日、9月14日版本与本版效果不同。

本版保存当前实际运行的源码、模型接口、机器人网格、场景和配置，并提供固定版本的依赖恢复入口。**GitHub 源码 ZIP 本身不含 Python 环境和授权 MANO 模型。**首次恢复后运行 `bash run.sh`，不再依赖原电脑另一份工程的路径。

## 本版运行内容

- 固定左手，HaMeR关节点0、5、9、13、17建立坐标系；X朝掌心外、Y食指侧到小指侧、Z手腕到四个指根中心。
- 掌面辅助姿态修正、前臂融合关闭；原实时滤波保留。
- 平移保持手向前→机械臂向前、手向上→机械臂向上。
- 手掌顺/逆时针对应的末端转向已交换，其他旋转方向和原增益保留；横滚5°输入→15°目标。
- 前桌移远3 cm，左右台面由12 cm降到9 cm，桌上物体同步调整。
- 默认带抓取场景、正面和侧面两个观察小窗；Gazebo中使用约束速度控制和QP求解。
- 本入口仅运行Gazebo仿真，不启用实物ABB输出。

控制与场景保存点是 `a4ba647`；后续封装只补依赖、可移植路径、恢复入口和准确的启动方法标识。详细坐标/历史验证见 [技术说明](docs/palm_axes_teleop_20260917/README.md)。

## 下载固定版本

```bash
git clone --branch teleop-palm-axes-20260917 --single-branch \
  https://github.com/yiyidiu/myhandarmtest1.git myhandarmtest1-palm-20260917
cd myhandarmtest1-palm-20260917
```

也可以从同名 Release 下载源码压缩包并解压；安装和运行不要求目录中存在 `.git`。

## 原电脑恢复：复用已有Python环境和授权模型

以下路径适用于原电脑。安装程序会把本版所需模型复制进新工程，记录所选Python路径，并在新目录重新编译：

```bash
bash scripts/setup_current_release.sh --skip-apt \
  --existing-env-root "$HOME/anaconda3/envs" \
  --assets-from "$HOME/myhandarmtest1-v1.0.1-ubuntu2004/perception_hamer/_DATA"
bash run.sh
```

已有模型也可来自自己备份的其他 `_DATA` 目录。模型内容与本版本不一致时安装会明确报错，避免默默混用。

## 新电脑恢复

系统条件：**Ubuntu20.04 x86_64、ROS Noetic desktop-full、Gazebo11、支持CUDA11.8的NVIDIA显卡/驱动、X11桌面、D435/D455的USB3连接及相机访问权限**。记录的原机是RTX2070、驱动535.230.02；其他机器的实时速度没有因此得到保证。系统/驱动的首次安装不包含在源码包内。

准备自己已授权的 `MANO_RIGHT.pkl` 后执行：

```bash
bash scripts/setup_current_release.sh --mano /实际路径/MANO_RIGHT.pkl
bash run.sh
```

安装程序将：

1. 安装声明的Ubuntu/ROS依赖（sudo可能要求输入密码）。
2. 从项目已发布的固定分卷恢复HaMeR、MediaPipe两个Python环境，并完成路径重定位；从SMPL-X原始分发源安装固定版本。
3. 从HaMeR原作者下载权重、配置和平均手型；导入自己提供的MANO模型。
4. 校验本版依赖文件，生成本机 `teleop_machine.env`，从源码进行Release编译。

环境分卷约7 GB；若仅提供MANO，还需下载原作者约6 GB的模型档案，只提取本程序需要的文件。建议准备至少35 GB空闲空间。已有完整 `_DATA` 时用 `--assets-from` 代替 `--mano`，避免再次下载权重。已具备系统依赖时可加 `--skip-apt`。

环境恢复在本项目的 `.venvs/` 下，不替换其他项目的环境。环境完成重定位后不要直接移动整个目录；换目录时重新运行恢复步骤，或通过 `--existing-env-root` 明确复用原位置的环境。

### MANO必须由本人提供

[MANO原始许可](https://mano.is.tue.mpg.de/license.html)不允许公开再分发，因此本公开仓库和附件不含 `MANO_RIGHT.pkl`。原电脑已有的授权文件可以用于本人控制的电脑；没有模型时需先按 [MANO官网](https://mano.is.tue.mpg.de/) 的注册和授权流程取得。**只下载GitHub源码、没有授权模型及运行依赖，不能运行HaMeR。**

精确模型校验值、环境分卷及原始下载地址见 [依赖清单](reproduction/20260917/runtime_reference.json)。环境附件复用已发布的 `teleop-complete-code-20260914` 固定分卷；它们只是依赖，控制代码始终使用本标签。

## 使用

```bash
bash run.sh
```

左手张开放稳后，在相机窗口按 **C** 设零位并启用控制；按 **Q** 退出。正面/侧面观察小窗支持滚轮放大，Esc关闭观察窗。运行日志保存在当前工程 `.runtime/` 中。

需要恢复本次交换之前的偏航方向时，不改源码，运行：

```bash
TELEOP_EMPTY_SCENE=false TELEOP_ALIGNMENT_VIEWS=true \
TELEOP_REVERSE_PALM_TURN=false bash scripts/run_five_point_axes_teleop.sh
```

## 版本控制与验证范围

本次已从GitHub重新克隆、恢复固定依赖、独立编译，并实际启动相机、Gazebo和两个观察窗，见 [恢复与启动记录](reproduction/20260917/VALIDATION.md)。

固定标签不随以后开发变动；新改动应使用新提交和新标签。`teleop_machine.env`、模型、环境、编译产物及运行录像不提交到Git。公开发布不包含个人回放截图和原始实验录像，它们不影响在线运行。

场景调整后已核对实际桌面位置、两个观察窗，以及5°输入生成15°横滚目标（短时仿真实际约14.97°）。这些是特定输入和本机仿真结果，不等于新的真人抓取、视觉精度或不同硬件的延迟验收。

HaMeR运行SDK源码随仓库保存，并保留其MIT许可；其余第三方代码许可在各自目录中。历史入口和说明见 [旧README](docs/versions/README_before_palm_release_20260917.md)，其中的旧路径和旧参数不代表本标签。
