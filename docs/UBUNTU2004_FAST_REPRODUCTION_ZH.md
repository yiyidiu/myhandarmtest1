# Ubuntu 20.04 最快复现手册

本手册面向只安装了 ROS、没有 GitHub 使用经验的新电脑。目标是先用一个 Release 压缩包恢复当前完整源码，自动安装依赖，编译六个 Catkin 包，并运行当前完整的 439 项测试。默认流程不会连接或驱动实体 ABB。

## 一、下载哪两个文件

登录 GitHub 账号后打开：

<https://github.com/yiyidiu/myhandarmtest1/releases/tag/v1.0.1-ubuntu2004-reproduction>

仓库是私有仓库，必须先登录并且账号具有访问权限。在页面下方 `Assets` 中下载：

```text
myhandarmtest1-v1.0.1-ubuntu2004.tar.gz
myhandarmtest1-v1.0.1-ubuntu2004.tar.gz.sha256
```

第二个小文件用于检查下载是否完整，不是另一个工程包。

## 二、校验、解压、一键复现

打开新电脑的终端，逐行复制下面的命令：

```bash
cd ~/Downloads
sha256sum -c myhandarmtest1-v1.0.1-ubuntu2004.tar.gz.sha256
tar -xzf myhandarmtest1-v1.0.1-ubuntu2004.tar.gz -C ~
cd ~/myhandarmtest1-v1.0.1-ubuntu2004
./scripts/bootstrap_ubuntu2004.sh
```

校验应显示 `OK`。一键脚本会做四件事：

1. 确认系统是 64 位 Ubuntu 20.04，并找到 `/opt/ros/noetic/setup.bash`；
2. 用 `apt` 和 `rosdep` 安装 MoveIt 1、MoveIt Servo、Gazebo 11、ros_control 和 Python 基础依赖；
3. 用 Release 模式编译六个 Catkin 包；
4. 运行 439 项离线/安全测试（感知 201、仿真 120、遥操作/Catkin 118），并检查六个包都能被 ROS 找到。

安装期间终端可能显示：

```text
[sudo] password for 你的用户名:
```

这里输入的是新电脑的 Ubuntu 登录密码。输入时屏幕不会显示星号，这是 Linux 的正常行为；输完按回车即可。整个过程需要联网，建议预留至少 6 GiB 可用磁盘空间。

最后看到下面这行才表示基础复现成功：

```text
[完成] Ubuntu 20.04 + ROS Noetic 安全基线已经可用。
```

如果失败，不要反复乱装软件。把终端最后的报错和脚本提示的 `.runtime/bootstrap_*.log` 发回来即可定位。

## 三、启动安全仿真

复现脚本结束后可以直接运行：

```bash
source ~/myhandarmtest1-v1.0.1-ubuntu2004/devel/setup.bash
cd ~/myhandarmtest1-v1.0.1-ubuntu2004
./scripts/run_stage1_safe_demo.sh
```

以后每开一个新终端，都要先执行上面的 `source .../devel/setup.bash`。安全演示默认关闭实体输出门；不要把仿真通过理解成实体机器人已经安全可用。

## 四、为什么压缩包没有 HaMeR 大模型

源码基线和 439 项测试不需要大模型。真正运行 `D455 + HaMeR/MANO` 还取决于新电脑的硬件和许可，至少需要：

- 支持 CUDA 11.8 路线的 NVIDIA 显卡与驱动；
- Intel RealSense D455、USB 3.x 接口和 `pyrealsense2`；
- 官方 HaMeR 模型包（原始下载约 6.04 GB）；
- 需要单独同意许可的 `MANO_RIGHT.pkl`。

MANO 许可不允许把文件随工程 Release 随意转发，因此本压缩包不会包含本机的 `perception_hamer/_DATA/`。这不是漏传。基础复现成功后，再按 [HaMeR 安装与显存验收](03_HAMER_INSTALL_AND_BENCHMARK.md) 配置感知环境；所需文件位置见 [perception_hamer 说明](../perception_hamer/README.md)。

实体 ABB、真实三指手和相机标定必须在仿真基线之外单独验收，不可通过这个一键脚本自动打开。

## 五、以后用 GitHub 更新（可选）

只想最快复现时，下载 Release 压缩包即可，不需要先学 Git。以后需要学习用 Git 管理版本时，再安装 Git：

```bash
sudo apt update
sudo apt install git
```

GitHub 不使用账号密码执行 `git push`。账号密码也不应写进命令、脚本或工程文件；详细步骤见 [GitHub 新手手册](GITHUB_BEGINNER_GUIDE_ZH.md)。

## 六、常用的重复运行选项

依赖已经安装好，仅需重新编译和测试：

```bash
./scripts/bootstrap_ubuntu2004.sh --skip-apt
```

只想尽快编译，暂时跳过测试：

```bash
./scripts/bootstrap_ubuntu2004.sh --skip-apt --skip-tests
```

跳过测试只能说明编译完成，不能代替 439 项当前测试通过的验收结论。
