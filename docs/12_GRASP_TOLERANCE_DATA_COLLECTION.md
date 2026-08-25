# 下一阶段：真实抓取容错数据采集框架

本文件只定义数据合同和记录器，不包含、生成或暗示任何真实抓取结果。记录器不发布机械臂或三指手命令。

## CSV 字段

字段分为：试次/操作者/物体元数据，俯抓或四类侧抓标签，计划与实测抓取中心位姿，位置与 SO(3) 旋转向量误差，机械臂和手关节状态，三指接触/应变片原始与标定值，闭合和接触稳定时间，提升/滑移量，人工取消和反向覆盖，Servo 碰撞/关节/工作空间/急停状态，成功标签、失败原因、源日志以及硬件标定编号。

权威字段顺序在 `src/handarm_moveit_demo/scripts/grasp_tolerance_data_collector.py` 的 `FIELDS` 中。至少必须提供：`trial_id`、`object_id`、`grasp_type`、`actual_grasp_center_pose_base`、`hand_joint_positions_rad`、`grasp_success` 和 `hardware_calibration_id`。

## 启动记录器

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun handarm_moveit_demo grasp_tolerance_data_collector.py \
  _output_csv:=/absolute/new/path/grasp_tolerance.csv
```

向 `/shared_teleop/grasp_trial_observation` 发布 JSON。记录器只有在
`measurement_status` 明确等于 `MEASURED_REAL_OR_EXPLICIT_SIM`、必填字段齐全且没有未知字段时才写一行；否则拒绝且不会用默认值伪造试次。

真实实验开始前仍需完成：实测法兰到掌心/抓取中心固定变换、确认相机运动轴到机器人速度轴的方向/符号/比例、三指手真实 ROS 高层命令适配器、应变片标定、抓取成功/失败判据冻结、实体 ABB 授权和逐级低速安全检查。这里不要求相机到机器人基座的完整位姿外参。
