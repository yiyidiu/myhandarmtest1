# 视角修正与关节速度控制基线 · 2026-10-06

固定标签：**`teleop-view-velocity-baseline-20261006`**。本版以 9 月 21 日完整工程为基础，合入后来修正过的 **原始 HaMeR＋纯视角几何处理＋16 点腕环**，并将机械臂执行接口换为 **直接关节速度控制**。作为后续论文实验的明确代码基线；完整任务效果仍须实际测量。

## 运行

本机已准备好的保存目录：

```bash
bash /home/dongtian/myhandarmtest1-view-velocity-baseline-20261006/run.sh
```

默认倒水场景，**不录屏、不保存 RGB-D、不读取 IMU**。使用左手，在相机窗口按 **C**，静止等待两秒倒计时及标定完成，然后开始控制 Gazebo。按 **Q** 退出。窗口标题包含 `PURE VIEW CORRECTION - WRIST RING 16`；终端显示机械臂底层使用关节速度指令。

在工程目录内可选场景；四个场景共用相同感知、映射与控制参数：

| 场景 | 命令 |
|---|---|
| 倒水 | `bash run.sh` |
| 方块堆叠 | `bash run.sh --stacking-scene` |
| 三物体抓取 | `bash run.sh --grasp-scene` |
| 空场景动作观察 | `bash run.sh --empty-scene` |

需要保存试验时加 `--record`，例如 `bash run.sh --stacking-scene --record`。数据保存在本工程 `.runtime/` 下，启动时打印路径，不上传到 GitHub。各场景原有 `scripts/run_*_observation.sh` 入口也使用新感知和速度接口，保留其默认录制行为。控制诊断日志仍保存在本机。

## 本版到底使用什么

1. **图像处理**：结合真实 RGB 相机内参及手部框中心，构造同光心、朝向手中心的虚拟视角，变换图像后送入原始 HaMeR；将预测三维点恢复到真实相机坐标。没有额外学习修正权重，不使用 9 月 28 日 v1 或公开训练候选。
2. **腕姿与位置**：使用 16 个 MANO 腕口点与中立腕环配准求朝向；C 标定后固定手形。五个掌部点仅用于掌面皮肤深度采样。RGB-D 深度用于位置；这一层纯视角腕姿修正没有利用深度拟合朝向。
3. **映射与控制**：沿用 9 月 21 日的整体腕姿映射、滤波、关节与自碰撞约束、原 QP 求解和三指手控制。整体转角为 1.3 倍，基座 X/Y/Z 平移增益为 1.6/2.0/1.2。未合入后续 NEO 对照或受限 6D 软协调研究原型。
4. **机械臂底层**：`velocity_controllers/JointGroupVelocityController` 接收六关节速度，250 Hz 发布；不启动外部关节位置积分器。Gazebo 内部采用速度 PI、模型重力补偿和有界驱动力，保留质量、惯量、重力与接触模型。速度指令经过原加速度过渡及失效归零处理。

原 9 月 21 日链路虽然上层求解速度，底层接收的仍是积分后关节位置。本版改变的是这项执行接口。历史文件名中的 `egm_position` 不代表新入口仍使用位置控制；以实际加载的控制器、硬件接口和发布指令为准。

## 下载和恢复

```bash
git clone --branch teleop-view-velocity-baseline-20261006 --single-branch \
  https://github.com/yiyidiu/myhandarmtest1.git myhandarmtest1-view-velocity-baseline-20261006
cd myhandarmtest1-view-velocity-baseline-20261006
bash scripts/setup_current_release.sh --mano /实际路径/MANO_RIGHT.pkl
bash run.sh
```

目标环境：Ubuntu 20.04、ROS Noetic、Gazebo 11、X11、NVIDIA CUDA 11.8 兼容显卡与 RealSense RGB-D 相机。已有依赖及两个 Python 环境时：

```bash
bash scripts/setup_current_release.sh --skip-apt \
  --existing-env-root "$HOME/anaconda3/envs" \
  --assets-from /已有授权模型路径/_DATA
```

依赖和原始模型版本继续使用 [9 月 21 日依赖清单](reproduction/20260921/runtime_reference.json)。设置程序生成当前目录的 `teleop_machine.env` 并重新编译。不要复制其他版本的 `build/`、`devel/`。源码发布包含模型接口、机器人网格、仿真插件和场景；模型权重、MANO 授权资产、Python 环境、个人录像及本机路径不随新源码上传。

## 论文使用与证据边界

将本版称为“2026-10-06 纯视角修正、16 点腕环与直接关节速度执行基线”，同时记录 Git 标签及提交。9 月 21 日基线作为旧版对照保留；两项改动的各自收益需用统一条件分别评价，不能把已有单模块结果当作本组合的完整任务效果。

既有腕环录像对照中，纯视角修正减少部分横向平移误转，但真实转腕存在欠跟；没有证明在所有条件下更准确或响应不变。本文新增速度执行器是 Gazebo 近似，未标定实体 ABB 电机响应，也未验证实体 EGM 速度通信。

倒水场景沿用简化水流和水量显示，不模拟真实液体动力学。控制器仍不读取仿真桌子或物体坐标。

本次实际核对、源码来源与验证范围见 [新基线验证说明](reproduction/20261006/VALIDATION.md)。固定保存目录不继续开发；后续变化另存版本，保留旧版。
