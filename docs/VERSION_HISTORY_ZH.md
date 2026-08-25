# 冻结检查点版本历史

本仓库把本机历史冻结包整理为同一项目的可检出版本。每个版本由三部分组成：

- Git 提交：可独立检出的源码树；
- Git 标签：稳定的版本名称；
- GitHub Release：版本说明、原始冻结包和校验文件的下载入口。

## 版本表

| 标签 | 原检查点时间（Asia/Shanghai） | 内容 | 原冻结包性质 |
| --- | --- | --- | --- |
| `v0.1.0-ground-workspace` | 2026-08-24 00:27 | 地面工作空间、速度 Servo 和确定性 Gazebo 验收 | 较完整的遥操作工程快照 |
| `v0.2.0-pose-decoupling` | 2026-08-24 04:43 | 平移/姿态解耦与小幅 Gazebo 刚度调整 | 较完整的遥操作工程快照 |
| `v0.3.0-egm-stable-hold` | 2026-08-24 18:31 | 稳定 EGM 位置参考与刚性保持 | 15 个最终变更文件，需叠加前序源码 |
| `v0.4.0-self-collision-safe` | 2026-08-25 21:03 | 全机器人 MoveIt/FCL 自碰撞失败关闭链 | 三个相关 Catkin 包的检查点 |
| `v1.0.0-clean-baseline` | 2026-08-25 22:04 | 清理后统一基线，269 项测试通过 | 当前完整 Git 工作树 |

## “重建版本”的含义

历史冻结包创建时，工作空间还不是 Git 仓库，并且部分压缩包只保存了当次发生变化的文件。因此 `v0.1.0` 至 `v0.4.0` 是根据带时间和 SHA-256 的冻结包，按时间顺序补齐未变化依赖后形成的可独立检出版本。它们不是对当时整个磁盘目录的字节级复制。

GitHub Release 保留原始压缩包、README、SHA-256 文件和验收证据。需要审计原始事实时以这些附件为准；需要编译和运行时使用对应 Git 标签。

## 检出为独立工程

不要覆盖正在使用的工作空间。以 `v0.3.0` 为例：

```bash
cd /home/diu/myhandarmtest1
git fetch --tags
git worktree add --detach \
  /home/diu/myhandarmtest1-v0.3.0 \
  v0.3.0-egm-stable-hold
cd /home/diu/myhandarmtest1-v0.3.0
catkin_make
```

每个 worktree 都应使用自己的 `build/` 和 `devel/`。不要在同一个终端里同时 `source` 两个版本的 `devel/setup.bash`。

## 安全边界

历史通过记录主要覆盖离线测试和 Gazebo。它们不等于 D455 USB3 正式认证、实体 ABB 授权、真实三指手标定或生产安全评审。任何实体机器人输出都必须另行完成低速验收和现场授权。
