# 2026-09-13 画面与控制输入延迟修复

本轮目标：实时手部画面更及时，减少输入超时造成的机械臂间歇保持；不改变位姿映射、C 零位、速度/加速度限制或碰撞判据。

已观察到的改善来自 `teleop_feedback_20260913T161134_uqTj` 的原始 RGB-D 回放。仿真、相机窗口、录屏和 ROS 数据记录同时运行。视觉主程序基线为提交 `02b3d21`；两次视觉对照使用相同控制侧程序。每次只在原记录的 C 事件之后请求标定，随后由同一条视觉链产生机械臂输入。

| 指标 | 视觉基线 | 本次修复 |
|---|---:|---:|
| RGB-D 帧就绪至姿态输出，中位数 | 106.8 ms | 94.1 ms |
| 同一延迟，95% 分位 | 231.3 ms | 141.4 ms |
| 同一延迟，最大值 | 499.3 ms | 265.4 ms |
| 姿态处理频率 | 11.0 Hz | 12.9 Hz |
| 有效姿态输出间隔，95% 分位 | 177.0 ms | 113.9 ms |
| 有效姿态输出最大间隔 | 475.0 ms | 231.3 ms |
| 进入预览绘制时的图像年龄，中位数 | 144.9 ms | 11.4 ms |
| 约 51 秒启用期间，保持状态的诊断采样点 | 15 | 2 |

图像年龄是帧就绪至进入绘制的时间，尚不包括 JPEG、窗口显示和显示器延迟。姿态延迟从 RGB-D 对齐完成开始计时，不是从人手开始运动到机械臂到达的总延迟。保持状态采样点也不是独立停顿次数。两次 C 接受的帧可能略有不同，这组实验不用于毫米级轨迹精度或抓取成功率比较。真实 USB 采集和同时写入原始 RGB-D 的长时间真人运行仍由用户验收。

已落实的改动：

- 左侧显示最新相机画面，右侧保留与姿态估计匹配的图像。旧姿态轴不会投影到新的相机帧上；左下方仍显示原始腕口轴。
- 缩短同一 Python 进程的线程交接等待，显卡状态监控改为复用一个 `nvidia-smi` 进程，保持原有监控频率。
- 逐帧完整记录仍写入 JSONL；写盘后不再把所有历史网格记录重复保留在内存里。
- 修复向检测、显示进程写入图像时没有处理短写的问题。性能采样时复现过 JPEG 帧边界损坏及显示进程退出；现在会把每帧完整写完，窗口进程失败时结束相机控制循环。
- 控制消息相关连接启用 TCP_NODELAY。固定原始指令的单独机械臂回放中，位置误差中位数为 10.64→10.59 mm，95% 分位为 39.11→39.38 mm；此轻负载对照未复现计时保持，不能单独宣称 TCP 修改消除了延迟。

未引入新的意图模型。独立相机进程的对照反而增加总延迟，未加入运行版；融合注意力的小幅性能收益也不足以支持替换模型计算。RealSense 2.58.3 的对齐 Python 绑定未释放 GIL，可解释观察到的一部分线程等待，但不能据此把所有延迟归因于深度对齐。[官方绑定源码](https://github.com/realsenseai/librealsense/blob/v2.58.3/wrappers/python/pyrs_processing.cpp)

回放启动错误已纠正：最初把人手画面与已记录指令分别启动，造成机械臂先动。`before_loaded`、`after_loaded` 不能用于同步跟随验收。后续 `before_synced_linked`、`after_monitor_linked` 均由同一视觉输入控制，在 C 前最后五秒的六关节控制参考变化均为零。性能采样运行及未成功标定的尝试不用于上表。

原始对照与程序日志在 `.runtime/latency_repair_20260913/`，上表使用 `perception_before_synced_linked`、`perception_after_monitor_linked` 的 `summary.json`、对应 `display_*.json` 及 `linked_controller_comparison.json`。修复后的回放录屏位于该目录的 `after_monitor_linked/screen/screen.mp4`。

正常运行并录屏、尝试接收 IMU：

```bash
bash /home/dongtian/myhandarmtest1-teleop-repair/scripts/run_grasp_feedback_recording.sh
```

仍需用户在相机窗口按 C 后才能控制；Q 退出并等待录像写完。数据自动保存到 `/home/dongtian/myhandarmtest1-teleop-repair/.runtime/teleop_feedback_日期时间_随机后缀/`，终端会打印具体路径与 IMU 接收状态。

退出运行后，恢复本轮修改前的源码与配套控制程序：

```bash
bash /home/dongtian/myhandarmtest1-teleop-repair/.runtime/version_backups/before_latency_repair_20260913/restore.sh
```

备份对应 `02b3d21`，数据不会被覆盖。当前修复保存于 `codex/latency-repair-20260913` 分支。
