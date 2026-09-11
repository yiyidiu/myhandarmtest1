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
