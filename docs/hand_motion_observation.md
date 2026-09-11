# 手部运动观察采集

运行：

```bash
cd /home/dongtian/myhandarmtest1-teleop-repair
./scripts/run_hand_motion_observation_recording.sh
```

这是当前遥操作的可选采集入口。Gazebo和MoveIt同时使用只有地面、机械臂及三指手的场景；关闭桌子、目标物体和抓取视角小窗，保留自身碰撞、地面及关节限制。常规`run_teleop_feedback_recording.sh`仍使用原物体布局。

相机左图显示原始16点腕口姿态，右图显示滤波和前臂融合后的姿态，红X、绿Y、蓝Z。箭头画在对应推理源帧上。底部固定位置的坐标轴只随估计朝向变化；手上箭头是三维轴的透视投影，可能随手的位置变化，不能仅凭箭头在画面里的角度判断真转动。

自然张手放稳后按一次C；可以依次做上下、左右、前后平移，俯仰、偏航、横滚。动作之间停稳2—3秒便于标注，通常无需每组重新按C。按Q退出，等待终端打印保存结束。

输出位于：

`/home/dongtian/myhandarmtest1-teleop-repair/.runtime/hand_motion_observation_日期T时间_随机后缀/`

- `screen.mp4`：屏幕回放，反馈直接使用播放器从00:00开始的时间，不扣启动等待。
- `rgbd.db3`：原始RGB-D；不另外重复保存逐帧PNG。
- `imu_raw.csv`、`imu_time_estimated.csv`：原始IMU和接收时间重建结果。接收时间重建不等于完成物理同步或方向标定。
- `control_chain.bag`、`camera_hand_pose.csv`：实际控制链和接收的手部姿态。
- `perception/DEV_HAMER_6DOF_OBSERVATION_日期T时间/frames.jsonl`：源帧、裁剪框、原始MANO、腕口求解、处理后姿态、发布包及时间数据。

最近一次本模式采集的文件夹：

```bash
cat /home/dongtian/myhandarmtest1-teleop-repair/.runtime/latest_hand_motion_observation.txt
```

人工标注用于说明动作意图；精确比较使用记录的源时间。切换边界和快速转向需要检查时间偏移，不能用人工标注直接计算瞬时姿态误差或链路延迟。

## 近腕区深度关联的对照运行

使用新增的位置估计，并保留录屏、RGB-D、IMU及空场景：

```bash
cd /home/dongtian/myhandarmtest1-teleop-repair
TELEOP_WRIST_DEPTH_MODE=proximal-surface ./scripts/run_hand_motion_observation_recording.sh
```

恢复之前的位置估计：

```bash
cd /home/dongtian/myhandarmtest1-teleop-repair
TELEOP_WRIST_DEPTH_MODE=visible-surface ./scripts/run_hand_motion_observation_recording.sh
```

不设置该变量时仍运行原来的 `visible-surface`。两种方式都需要重新启动并按一次C；
不在同一次控制中热切换估计方法。`session.json` 和相机运行摘要记录所选模式。

新模式只用确认手部的2D掌部区域选择近腕处的深度，输出位置仍在16点腕口中心的相机射线上。
测得的手面深度变化直接传递，只对模型的手面到腕口深度偏移进行因果平滑；
短暂缺少局部支持时连续接入原估计的增量，避免两种估计的偏差形成一次跳变。
这个模式不改变姿态估计、姿态或平移增益、机器人C零位、关节控制和碰撞设置。

已有回放用于验证特定深度跳变与输入敏感性的改善。位置绝对精度与实际操作体验仍需真人对照，
不能以图像取样稳定代替遥操作验收。此前尝试的统一视角旋转补偿存在返回动作退化，没有启用。

本轮修改前的完整源码保存在本地标签 `teleop-before-perception-geometry-20260911`。

### 2026-09-11回放结果与限制

使用 `hand_motion_observation_20260911T153019_6Wdm` 的原始RGB-D及对应MANO输出，
没有启动机器人或重新采集。局部裁剪对照使用84帧；连续位置回放覆盖3:13至8:40的4655帧。

| 比较内容 | 原位置算法 | 新位置算法 | 能说明什么 |
| --- | ---: | ---: | --- |
| 俯仰片段：同一RGB-D帧仅扩大裁剪框10%，最大位置差异 | 25.46 mm | 4.27 mm | 对这类裁剪变化更稳定，不是绝对定位精度 |
| 横滚片段：同一裁剪对照的位置差异95分位 | 4.06 mm | 6.11 mm | 仍存在退化，不能推广为所有动作都更准 |
| 连续回放：相邻帧位移99分位 | 21.18 mm | 21.62 mm | 包含真实运动，整段并未显示全面改善 |
| 4:23—4:27：前移动作的相机Z净变化 | −35.21 cm | −35.18 cm | 这段前后运动幅度没有被强行压平 |

直接切换深度取样算法曾在约5:57产生约5 cm跳变；当前实现保持两种估计间的偏移连续，
并仅平滑模型深度偏移。这个切换问题已在连续回放中检查，不能据此声称遥操作卡顿已全部解决。

原算法回放与原记录位置差的99分位为0.067 mm，最大4.85 mm；离散像素取样存在重建差异。
旧、新深度计算的离线中位耗时分别约0.59、0.81 ms；这不是相机到机械臂的实时延迟测量。
现有IMU不能提供位置真值，因此不能把上述数值写成毫米级定位精度。

统一视角补偿、局部深度法向和另一视觉模型的姿态对照均出现动作间退化，
没有加入在线姿态控制。假转动问题尚未修好。当前候选仅用于真人比较位置控制是否受益，
观察纯平移是否新增前后跳动、主动前后运动是否仍可控；不以抓取成功与否验收本次位置改动。
