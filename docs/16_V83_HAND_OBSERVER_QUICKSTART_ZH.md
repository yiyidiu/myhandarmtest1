# V8.3 人手平移与姿态观察器：快速验收

> 状态：仅保留作归档复现/审计，不是当前主运行入口。该独立路线在本机现场记录中
> 前臂有效率为 0/1973，且 V8.3 输出四类互斥意图，不满足六轴同时控制。当前请使用
> `perception_hamer/README.md` 中唯一的
> `perception_hamer/scripts/run_d455_hamer_crop.py` 单 HaMeR 命令。

本入口直接运行用户提供的 `teleoperation_ubuntu_core.tar.gz` 中的完整
MANO 腕口、前臂 V5.3、V8.3 意图模型和腕部位姿面板。它只观察、显示和
记录，不导入 ROS，不发送 UDP，也不会启动 Gazebo 或控制机械臂。

## 第一次安装

```bash
cd /home/diu/myhandarmtest1
./scripts/setup_teleoperation_core_v83_env.sh
./scripts/test_teleoperation_core_v83.sh
```

正常测试结果应包含：

```text
22 passed, 1 skipped
V8.3 MODEL OK: V8.3 causal multiscale low-speed intent
```

## 每次运行

先在旧的 HaMeR/Gazebo 终端按 `Ctrl-C`。新入口有单实例锁；旧 HaMeR
没有退出时会拒绝加载第二份模型，防止 RTX 2060 显存溢出。

```bash
cd /home/diu/myhandarmtest1
./scripts/run_teleoperation_core_v83_observer.sh
```

只会出现一个主观察窗口。默认画面包括：

- D455 的 640x480 实时 RGB；
- 与当前有效帧对应的完整 MANO 网格；
- MANO 16 顶点腕口中心及三轴坐标；
- RGB-D 前臂中心线/纵轴；
- `TRANSLATION / YAW / PITCH / ROLL` 四类 V8.3 概率；
- 手腕绝对 `XYZ`、相对 `dXYZ`、相对 `YAW/PITCH/ROLL` 和 SO(3) 总转角。

## 手应该怎样放、怎样动

冻结 V8.3 检查点的真实数据合同是**物理左手**。把左手、完整手腕和至少
约 15 cm 前臂放进画面；另一只手出现时不会被选作 V8.3 控制手。这里没有
按键锁手或换手：MediaPipe 自动寻找物理左手。

识别稳定后，保持自然中立姿态并按一次 `C`。此时：

- 平移参考点：MANO 手腕开口的 16 个边界顶点中心，经 D455 对齐深度得到
  相机坐标中的米制位置；
- 姿态参考：按 `C` 时的完整 MANO 腕口旋转矩阵；
- 相机坐标：`+X` 画面向右，`+Y` 画面向下，`+Z` 远离相机；
- 相对旋转先用 `R_current * R_zero.T` 在 SO(3) 上计算，最后才分解为
  `Rz(yaw) * Ry(pitch) * Rx(roll)` 供画面显示。

验收动作：

1. 手和前臂整体左右、上下、前后移动，观察 `dXYZ` 与 `TRANSLATION`；
2. 尽量固定前臂，绕三个方向转动手腕，分别观察 `YAW/PITCH/ROLL`；
3. 把手完全移出画面，MANO 和意图必须失效，不能保留凭空的旧手；
4. `R` 清除零位，`C` 重新设零；`Q`/`Esc` 或终端 `Ctrl-C` 退出。

不要用 `C` 选择手。它只设平移和姿态零位。

## 频率和显存的真实结果

参考包的分类特征按 10 Hz 训练，因此观察器保持 10 Hz 采样；D455 仍为
30 Hz。2026-08-20 在本机 D455 + RTX 2060 的 20 秒无人画面测试中：

- 观察频率逐步稳定到 9.64 Hz；
- HaMeR Python 进程约 3987 MiB；
- 整卡约 4427/6144 MiB；
- 200 帧始终为 `INVALID`，无手时没有伪造意图；
- 退出码为 0。

为适配 6 GiB 显存，入口只把原包的 HaMeR 加载顺序改为 CPU 反序列化后
再移动到 GPU，并对网络前向启用 FP16 autocast。MANO 模型、V5.3 特征、
V8.3 三个 ExtraTrees、概率滤波、阈值和训练采样率均未修改。

## 日志

默认日志会写到运行时目录下：

```text
.runtime/teleoperation_core_v83/<archive-sha>/teleoperation_ubuntu_core/data/live_observer_logs/<timestamp>/
```

包括 `v83_robust_live.csv`、逐帧复核视频和 `observer_manifest.json`。位姿
数值面板是压缩包的显示扩展；原包 CSV 保持原字段，不声称已加入相对
`dXYZ/YPR` 字段。

## 当前明确限制

- 只验证压缩包训练合同内的物理左手；右手归一化尚未训练或验证；
- V8.3 只有两名参与者，是开发候选模型，不是跨人群结论；
- V8.3 输出是四类互斥观察结果，当前不接入六轴机械臂命令；
- 本次无人冒烟测试验证了启动、频率、显存和无手失效，没有伪装成已经
  验证操作者的具体平移、俯仰、偏航和横滚动作。
