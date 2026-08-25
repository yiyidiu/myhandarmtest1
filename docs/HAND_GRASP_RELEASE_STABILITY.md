# 三指手 GRASP/RELEASE 语义与 Gazebo 稳定性复核

日期：2026-08-15

## 已冻结的命令语义

公开话题 `/handarm_sim_demo/hand_command` 只接受 `GRASP` 和 `RELEASE`：

- `GRASP` 只把屈伸关节 `f1j2/f2j1/f3j2` 从 0.20 rad 移向 1.30 rad；
- `RELEASE` 只把三个屈伸关节移向 0.20 rad；
- 掌型关节 `f1j1` 在两条命令中的目标均为 0.051 rad；
- 两条命令均不发布机械臂轨迹，也不切换掌型；
- `PRE_SHAPE_A/B` 和任意构型命令不属于公开操作接口，发布后 fail-closed 拒绝。

启动协调器会在冷启动时把机械臂和手送到同一个已知初态并保持。这是一次性的仿真
初始化，不是 `GRASP` 或 `RELEASE` 的一部分。

空载开合只验证命令语义、关节方向、限位、从动关系和稳态，不证明能夹住物体。
后续物理抓取必须按 `机械臂接近 -> RELEASE 手型保持 -> GRASP 接触闭合 -> 机械臂
抬升 -> 保持 -> RELEASE 自然下落` 验证；禁止固定关节附着、瞬移或直接搬动物体。

## 已发现并修复的基础隐患

| 隐患 | 修复与验证 |
|---|---|
| 手部高面数视觉 STL 同时用作动态碰撞体 | 视觉网格保留；9 个手部 link 的碰撞改为圆柱、盒体和球体组成的凸基元代理，避免三角网格接触抖动 |
| 位置 URDF、速度 URDF 和旧 xacro 物理参数漂移 | 三套模型统一有限 revolute 限位、质量/惯量、阻尼、零关节内部库仑摩擦和碰撞代理；两套 xacro 展开后均通过 `check_urdf` |
| 速度 URDF 的被动关节为 continuous、mimic 无 PID，可能逐步 `SetPosition` | 改为有限 revolute；四个 mimic 插件全部使用有限力 PID，禁用逐步 SetPosition 路径 |
| 手部控制器和从动关节力矩过大 | 主动关节使用有限 PID；掌型从动最大 4 N·m，三个远端从动最大 0.2 N·m，近目标进一步限到 0.05 N·m |
| 关节库仑摩擦与小 PID 力矩相当，造成 stick-slip | 关节内部 Coulomb friction 置零，保留粘性阻尼和物体表面接触摩擦；尾窗位置范围由约 1e-3 rad 降到 1e-7 rad 量级 |
| world 接触修正速度 100 m/s，深穿透时可能爆炸 | 三个 handarm world 统一 ODE quick 100 iterations、`contact_max_correcting_vel=0.1 m/s`、`contact_surface_layer=0.001 m` |
| 旧 launch 可重新生成不受控模型 | `handarmtest1/launch/irb120_gazebo.launch` 改为安全基线的兼容包装，不再直接生成旧 xacro |
| 只看 20 Hz Gazebo 服务速度会偶发读到 ODE 步中间态 | mimic 插件在 `WorldUpdateEnd` 发布 1 kHz 后步状态的 0.1 s 心跳和越阈事件；缺心跳、任何后步速度大于 0.05 rad/s、无法交叉证明的服务尖峰均失败 |
| 服务中间态被无限忽略的风险 | 每关节每个稳态尾窗最多接受 1 次、且必须由覆盖其调用时间区间的后步心跳证伪；更多次数 fail-closed |

静态审计同时确认：位置/速度 URDF 和两套 xacro 的 9 个手部 link 惯量矩阵均正定，
最小质量 0.057879 kg，最小惯量特征值约 `4.81e-6 kg m²`；没有手部 continuous
关节、`self_collide=true` 或关闭重力的 link。

## 实际运行结果

### 五循环空载回归

执行 5 次 `RELEASE -> GRASP -> RELEASE`，共 15 个独立 5 秒窗口：15/15 PASS。
机器结果为
`results/sim_baseline/hand_stability_20260814T173439819381Z.json`。

- 最坏稳态尾窗位置范围：`3.91e-7 rad`；
- 最坏 20 Hz 服务速度 P95：`2.02e-6 rad/s`；
- 最坏 1 kHz 物理步末速度：`2.64e-6 rad/s`；
- 最坏 mimic 关系误差：`0.02370 rad`，门限 `0.03 rad`；
- 最坏主动目标误差：`0.02884 rad`，门限 `0.05 rad`；
- 仅有一次服务读数为 `0.3068 rad/s`；调用时间区间完全落在一个 100 步的
  WorldUpdateEnd 心跳窗中，该窗对应关节最大步末速度仅 `4.02e-7 rad/s`，因此记录为
  `ASYNC_SERVICE_MID_UPDATE_ARTIFACT_SUPPORTED`，没有把它删除或伪装成零。

每个从动关节每个尾窗均有 30--31 个心跳，心跳间隔约 0.1 s、每窗 100 个物理
update。正常开合阶段能观测到真实越阈事件，证明诊断链路不是静默失效。

加入“每关节每尾窗最多 1 次被证伪的服务尖峰”硬上限后，又从冷启动实际跑了
1 次完整 `RELEASE -> GRASP -> RELEASE`，3/3 PASS，结果为
`results/sim_baseline/hand_stability_20260814T175542223443Z.json`。三个窗口均无服务
尖峰、无未解析异常；最坏尾窗位置范围 `3.98e-7 rad`，证明最终代码路径已运行，
不是只通过合成单测。

### 2026-08-15 GUI 命令核对

在 `hand_commands_only.launch gazebo_gui:=true` 冷启动中实际下发
`RELEASE -> GRASP -> RELEASE`，三条命令均 `success=true`：

- 第一次 RELEASE：主动关节实际位置
  `f1j1=0.07176, f1j2=0.17118, f2j1=0.20003, f3j2=0.17270 rad`；
- GRASP：`f1j1=0.06277, f1j2=1.31150, f2j1=1.32884, f3j2=1.31239 rad`；
- 最后 RELEASE：`f1j1=0.07176, f1j2=0.17119, f2j1=0.20002, f3j2=0.17270 rad`。

`f1j1` 在三次命令中的目标都为 0.051 rad；三个屈伸关节发生约 1.13--1.14 rad
开合。四个 mimic 关系和短稳态检查全部通过。GUI 退出使用 Ctrl-C，ROS master、
Gazebo 和 controller 正常结束，没有遗留第二套同名节点。

## 当前结论和停止线

空载 `GRASP/RELEASE` 语义与关节稳定性通过，可以进入固定掌型的接触抓取调试。
这不等于物理抓取通过：当前尚未用真实接触证明物体被至少两指夹住、随手抬升、保持
并在 RELEASE 后自然下落。在该证据完成前，不得宣称“抓住并抬起物体”。实体手指
根部应变片已经在硬件侧部署，本仿真不设计也不依赖应变片闭环。
