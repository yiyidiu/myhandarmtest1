# Franka FR3 ＋ Allegro 四指左手遥操作

2026 年 11 月 25 日完整仿真基线。用 D455 观察左手，张手按 C 标定，手腕驱动 FR3 七轴，拇指、食指、中指、无名指分别驱动 Allegro V4 左手的 16 个关节。默认场景只有装配好的臂、手、地面和光照。默认姿态为掌心朝下。

## 在新电脑上运行

适用环境为 **Ubuntu 20.04 x86_64**。实时模式使用 D455、USB 3、X11 桌面及支持 CUDA 11.8 的 NVIDIA 驱动，本版在 RTX 2060 上验证。首次使用需要联网和安装 Ubuntu 软件包的权限，建议为工程、依赖、编译与缓存预留至少 45 GB，实验录制另计。后续普通启动使用本地已准备好的环境。

下载固定版本，可以从本页的 **Releases** 解压源码包，也可以运行以下命令。

```bash
git clone --depth 1 --branch fr3-allegro-teleop-v2026.11.25 https://github.com/yiyidiu/myhandarmtest1.git fr3-allegro-teleop
cd fr3-allegro-teleop
```

**先看仿真。** 无需相机、显卡推理环境或 MANO。以下命令自动安装所需系统软件、编译模型与插件，再用归档腕部和手指数据驱动完整控制链。

```bash
./run.sh --replay
```

**用自己的 D455 遥操作。** 将本人获授权的 `MANO_RIGHT.pkl` 放到 `run.sh` 同一目录。启动器会下载并校验本版本的两套视觉环境、HaMeR 权重和模型配置。也可以将自己电脑上的 `fr3-owner.tar` 直接复制到 `run.sh` 旁边。启动器会自动读取里面的授权模型和预先下载的视觉依赖，省去大型模型与环境的网络下载。授权文件也可以通过 `FR3_MANO_FILE` 指定。 [MANO 许可与申请](https://mano.is.tue.mpg.de/license.html) · [HaMeR 项目](https://github.com/geopavlakos/hamer)

```bash
./run.sh --record
```

相机预览中显示 **C = SET ZERO** 后，把自然张开的左手和腕部完整放入画面，点击相机窗口，按一次 **C**，保持到 **GAZEBO CONTROL: ENABLED**。按 **Q** 或在终端按一次 **Ctrl+C** 结束，等到打印录制结果。每次录制独立保存到 `~/handarm/results/fr3-teleop-recordings/`，包含 `screen.mp4`、`robot.bag`、实时采集的 `rgbd.db3`、感知与时钟记录。用 `session.json` 的 `recording_complete` 和 `missing` 核对本次是否录全。

| 需要的操作 | 在工程目录执行 |
|---|---|
| 预先装好实时环境，完成后退出 | `./run.sh --prepare` |
| 仅准备仿真及回放环境 | `./run.sh --prepare --replay` |
| 在后台运行归档回放 | `./run.sh --replay --headless` |
| 切换到腕环深度补偿观测，实时录制 | `./run.sh --qb0 --record` |
| 使用掌面中心作为机械端控制点 | `FR3_CONTROL_POINT=palm ./run.sh --record` |
| 固定机械臂，仅遥操作四指 | `./run.sh --fingers-only --record` |
| 观看单线轴阻抗抓起、悬停和放回的合成输入演示 | `./run.sh --impedance-grasp --replay` |
| 实时遥操作抓取同一物体 | `./run.sh --impedance-grasp --record` |

## 本版运行的代码

| 环节 | 位置与含义 |
|---|---|
| 整机 | [FR3 与 Allegro 左手 URDF](reproduction/fr3/fr3_allegro_sim/urdf/fr3_allegro_left.urdf)、[模型网格和许可证](reproduction/fr3/fr3_allegro_sim/meshes)、[两个轨迹控制器](reproduction/fr3/fr3_allegro_sim/config/controllers.yaml)、[Gazebo 关节插件](reproduction/fr3/fr3_allegro_sim/src/bounded_revolute_hw_sim.cpp) |
| 人手感知与标定 | [D455 与 HaMeR 入口](perception_hamer/scripts/run_d455_hamer_crop.py)、[张手 C 标定](perception_hamer/src/teleop_control_gate.py)、[实际启动参数](src/allegro_finger_teleop/scripts/fingers_camera.sh) |
| 腕部到七轴 | 默认使用 9 月 21 日的整体腕姿映射和掌面实测深度，控制点在 Allegro 安装口。可用 `--qb0` 选择相对中性模板的姿态与腕环中心深度补偿。[腕部控制链](src/allegro_finger_teleop/launch/baseline_arm.launch) · [FR3 参数](src/allegro_finger_teleop/config/fr3_whole_wrist.yaml) |
| 手指到四指 | 在掌部局部坐标中处理关节点，通过指尖、活动指节及指对接触关系求解手指角度。使用 11 月 25 日侧转腕时的指对确认和拇指外壳间隙检查。严重遮挡时，新指对可能显示“待确认”，转回能分辨各指的角度后可重新确认。[手形观测](src/allegro_finger_teleop/scripts/hand_shape.py) · [重定向求解](src/allegro_finger_teleop/scripts/retarget.py) · [Allegro V4 限位](src/allegro_finger_teleop/scripts/allegro_v4_limits.py) |
| 抓取 | 物体受重力、摩擦及接触约束。双侧指腹接触后施加关节弹簧预载。内置样例是食指与拇指抓放一只线轴。[阻抗参数](src/allegro_finger_teleop/config/grasp_impedance.yaml) · [控制代码](src/allegro_finger_teleop/scripts/object_impedance.py) · [抓放验收](src/allegro_finger_teleop/tests/check_gazebo_impedance_pick.py) |

[FRoGGeR 整机模型来源](https://github.com/alberthli/frogger/blob/f91157f96a235fc3b070d7f7ec9f52c3140c2cba/models/fr3_algr/fr3_algr_lh.sdf) · [Franka 官方模型版本](https://github.com/frankarobotics/franka_description/tree/7aeeddc449edf8d62b594f9e36a81da53e7796f9) · [Wonik V4 手官方模型](https://github.com/simlabrobotics/allegro_hand_ros_v4/tree/b19b18ea130c4fbe343f2c178eec6cb55186a4ac) · [9 月 21 日原始腕部基线](https://github.com/yiyidiu/myhandarmtest1/tree/teleop-whole-wrist-scenes-20260921)
