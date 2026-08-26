# P0 修复 06：真人 C 门控、映射与实时链路

## 结论边界

本阶段修复的是“相机按 C 后 Gazebo 机械臂不产生可用运动”的确定性链路故障，并用
合成 UDP、完整 PlanningScene 和 Gazebo 实际 `base_link -> tool0` 位姿验证。它不把
尚未完成的新一轮真人 C-to-Q 录像宣称为合格，也不代表 ABB 真机验收。

## 复现出的独立故障

1. EGM 真人入口把示例 `camera_ground_workspace` 标定当成默认值。一次 25 mm 相机
   X 位移产生约 58 mm 末端目标变化，随后碰撞速度比例从 1.0 快速下降并进入保持；
   这不是当前操作者的实测标定，不能作为默认映射。
2. 相机显示线程会用一帧陈旧的显示状态销毁主推理线程刚建立的 C 令牌；短暂的
   MediaPipe 阴性结果也会永久销毁令牌。操作者看到的结果是按 C 后链路偶发或很快
   重新锁定。
3. 旧的 Gazebo GUI 默认约 62 Hz 渲染，加上全分辨率屏幕采集，与 HaMeR 抢占 CPU。
   故障录像中 HaMeR 平均模型推理为 204 ms、输出约 2.79 Hz，已经贴近或超过 ROS
   端 200 ms 的生产者延迟上限。
4. 原测试入口可以要求 PlanningScene 就绪却不启动场景同步节点，导致所谓六轴测试
   实际一直被安全门锁住。完整地面入口现已允许仅测试使用 `direction_test` 输入。

## 修复

- 真人 EGM 入口默认改为 `current_linear`，即平移 0.6 倍的相对 C-zero 映射；只有
  操作者完成测量后才允许显式选用归一化工作空间。
- 显示线程不再修改控制门。主推理线程仍逐包校验检测、ROI、手身份、方向和时间戳。
- MediaPipe 短暂丢失采用 8 帧且最多 0.35 s 的有界宽限：期间发送 INVALID 心跳并
  将机械臂保持为零输出，但保留同一手身份和 C；超过任一界限仍 fail-closed，并要求
  新 C。
- 地面 Gazebo 世界通过 GUI 插件把渲染限制为 15 Hz。正式采集关闭同步 OpenCV 网格
  录像，屏幕录像缩放到 1920x1080，并使用两线程 ultrafast x264。
- C 和 Q/ESC 由相机进程原子写入持久化 JSON 标记。正式 HaMeR JSONL 从有效 C 的
  同一控制包开始记录；视频和 rosbag 使用预录覆盖边界，分析时按标记裁切。
- 单命令验证新增三层终端证明：ROS 接受并捕获 C、Servo 产生非零指令、Gazebo
  `tool0` 实际移动至少 2 mm。

## 自动验收

- 严格 UDP C 门控与完整地面链路：PASS。25 mm 相机 X 位移得到 14.75 mm 左右的
  `base -Y` 实际位移；短暂 0.25 s INVALID 期间保持，恢复时不需要第二次 C；1.0 s
  UDP 静默后旧令牌被拒绝；回零误差约 0.20 mm；危险 Servo 状态为空。
- 完整 PlanningScene 六轴：PASS。三个 25 mm 平移的工具响应比例为
  0.5705–0.5725；三个 0.22 rad 旋转的响应比例为 0.8562–0.9586；总回零误差
  0.251 mm / 0.045°。腕部 Y 旋转最低碰撞速度比例 0.305，未进入硬停并成功回零。
- GUI 负载 A/B（同机同进程）：默认约 62 Hz 渲染时 HaMeR 平均 52.73 ms、
  18.96 Hz；15 Hz 渲染时平均 46.36 ms、21.57 Hz。启用新的缩放屏幕录像时平均
  39.52 ms、25.30 Hz，仍满足严格 200 ms 生产者延迟上限。
- 回归：`perception_hamer` 216 项测试通过；`handarm_moveit_demo` 111 项测试通过；
  两个修改包完整构建通过。

运行时测量文件保存在工作树的 `.runtime/` 下，包括
`live_ground_c_gate_current_linear.json`、`current_linear_six_axis_full_ground.json`
和 `gui_render_rate_ab/`。这些大体积/机器相关证据不纳入 Git 源码提交。

## 操作者命令

快速跟手验证：

```bash
cd /home/dongtian/myhandarmtest1-v0.3.0
./scripts/run_live_human_teleop_validation.sh
```

正式 C-to-Q 证据采集：

```bash
cd /home/dongtian/myhandarmtest1-v0.3.0
./scripts/run_live_human_evidence_session.sh
```
