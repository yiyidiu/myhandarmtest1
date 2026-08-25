#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dynamic_arm_controller.py

功能：
使用 MoveIt 编程接口动态控制机械臂 abbarm。

支持接口：

1. /abbarm/joint_target_deg
   类型：std_msgs/Float64MultiArray
   数据：[q1, q2, q3, q4, q5, q6]
   含义：机械臂 6 个关节的绝对目标角度
   单位：deg

2. /abbarm/joint_delta_deg
   类型：std_msgs/Float64MultiArray
   数据：[dq1, dq2, dq3, dq4, dq5, dq6]
   含义：机械臂 6 个关节的相对角度增量
   单位：deg

3. /abbarm/ee_target_xyzrpy_deg
   类型：std_msgs/Float64MultiArray
   数据：[x, y, z, roll, pitch, yaw]
   含义：末端 handbase_link 的绝对目标位姿
   单位：x/y/z 为 m，roll/pitch/yaw 为 deg

4. /abbarm/ee_delta_xyzrpy_deg
   类型：std_msgs/Float64MultiArray
   数据：[dx, dy, dz, droll, dpitch, dyaw]
   含义：末端 handbase_link 在当前位姿基础上的相对增量
   单位：dx/dy/dz 为 m，droll/dpitch/dyaw 为 deg

说明：
- 末端姿态使用 RPY 欧拉角输入。
- 程序内部会将 RPY 转换为四元数。
- MoveIt 中真正执行的是 geometry_msgs/Pose，即位置 + 四元数姿态。
"""

import sys
import math
import copy
import threading

import rospy
import moveit_commander
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class DynamicArmController:
    def __init__(self):
        # 初始化 MoveIt Commander
        moveit_commander.roscpp_initialize(sys.argv)

        # MoveIt 规划组名称，你的 SRDF 里机械臂规划组叫 abbarm
        self.group_name = rospy.get_param("~group_name", "abbarm")

        # 创建 MoveGroupCommander 对象
        self.arm = moveit_commander.MoveGroupCommander(self.group_name)

        # 规划参数
        self.arm.set_max_velocity_scaling_factor(
            rospy.get_param("~velocity_scale", 0.15)
        )
        self.arm.set_max_acceleration_scaling_factor(
            rospy.get_param("~acceleration_scale", 0.15)
        )
        self.arm.set_planning_time(
            rospy.get_param("~planning_time", 8.0)
        )
        self.arm.set_num_planning_attempts(
            rospy.get_param("~planning_attempts", 10)
        )

        # 获取规划坐标系
        self.planning_frame = self.arm.get_planning_frame()

        # 获取末端 link
        # 你的 abbarm 规划链是 base_link 到 handbase_link
        param_end_link = rospy.get_param("~end_link", "")
        moveit_end_link = self.arm.get_end_effector_link()

        if param_end_link:
            self.end_link = param_end_link
        elif moveit_end_link:
            self.end_link = moveit_end_link
        else:
            self.end_link = "tool0"

        # 加锁，避免多个话题同时触发 MoveIt 执行
        self.lock = threading.Lock()

        rospy.loginfo("========== Dynamic MoveIt Arm Controller ==========")
        rospy.loginfo("Planning group: %s", self.group_name)
        rospy.loginfo("Planning frame: %s", self.planning_frame)
        rospy.loginfo("End link: %s", self.end_link)
        rospy.loginfo("Active joints: %s", self.arm.get_active_joints())
        rospy.loginfo("Current joints(rad): %s", self.arm.get_current_joint_values())
        rospy.loginfo("===================================================")

        # 关节控制接口
        rospy.Subscriber(
            "/abbarm/joint_target_deg",
            Float64MultiArray,
            self.joint_target_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/abbarm/joint_delta_deg",
            Float64MultiArray,
            self.joint_delta_callback,
            queue_size=1
        )

        # 末端位姿控制接口：位置 + 姿态
        rospy.Subscriber(
            "/abbarm/ee_target_xyzrpy_deg",
            Float64MultiArray,
            self.ee_target_xyzrpy_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/abbarm/ee_delta_xyzrpy_deg",
            Float64MultiArray,
            self.ee_delta_xyzrpy_callback,
            queue_size=1
        )

    def execute_joint_goal(self, joint_goal_rad):
        """
        执行关节空间目标。
        输入：
            joint_goal_rad：关节目标角，单位 rad
        """
        with self.lock:
            joint_num = len(self.arm.get_active_joints())

            if len(joint_goal_rad) != joint_num:
                rospy.logerr(
                    "关节目标数量错误：收到 %d 个，但规划组 %s 需要 %d 个。",
                    len(joint_goal_rad),
                    self.group_name,
                    joint_num
                )
                return

            rospy.loginfo("设置关节目标(rad): %s", [round(v, 4) for v in joint_goal_rad])

            self.arm.set_joint_value_target(joint_goal_rad)

            success = self.arm.go(wait=True)

            self.arm.stop()
            self.arm.clear_pose_targets()

            rospy.loginfo("关节目标执行结果: %s", success)

    def execute_pose_goal(self, target_pose):
        """
        执行末端位姿目标。
        输入：
            target_pose：geometry_msgs/Pose
        """
        with self.lock:
            rospy.loginfo(
                "设置末端位姿目标：位置 x=%.4f, y=%.4f, z=%.4f",
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z
            )

            rospy.loginfo(
                "设置末端位姿目标：四元数 qx=%.4f, qy=%.4f, qz=%.4f, qw=%.4f",
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w
            )

            self.arm.set_pose_target(target_pose, self.end_link)

            success = self.arm.go(wait=True)

            self.arm.stop()
            self.arm.clear_pose_targets()

            rospy.loginfo("末端位姿目标执行结果: %s", success)

    def joint_target_callback(self, msg):
        """
        绝对关节角控制。
        话题：
            /abbarm/joint_target_deg

        数据：
            [q1, q2, q3, q4, q5, q6]
            单位：deg
        """
        joint_goal_deg = list(msg.data)

        rospy.loginfo("收到绝对关节角目标(deg): %s", joint_goal_deg)

        joint_goal_rad = [math.radians(v) for v in joint_goal_deg]

        self.execute_joint_goal(joint_goal_rad)

    def joint_delta_callback(self, msg):
        """
        相对关节角控制。
        话题：
            /abbarm/joint_delta_deg

        数据：
            [dq1, dq2, dq3, dq4, dq5, dq6]
            单位：deg
        """
        delta_deg = list(msg.data)

        rospy.loginfo("收到关节角增量(deg): %s", delta_deg)

        current = self.arm.get_current_joint_values()

        if len(delta_deg) != len(current):
            rospy.logerr(
                "关节增量数量错误：收到 %d 个，但当前规划组需要 %d 个。",
                len(delta_deg),
                len(current)
            )
            return

        joint_goal_rad = []
        for q_current, dq_deg in zip(current, delta_deg):
            joint_goal_rad.append(q_current + math.radians(dq_deg))

        self.execute_joint_goal(joint_goal_rad)

    def ee_target_xyzrpy_callback(self, msg):
        """
        末端绝对位姿控制。
        话题：
            /abbarm/ee_target_xyzrpy_deg

        数据：
            [x, y, z, roll, pitch, yaw]

        单位：
            x/y/z：m
            roll/pitch/yaw：deg

        含义：
            将末端 handbase_link 控制到规划坐标系下的绝对位姿。
        """
        data = list(msg.data)

        if len(data) != 6:
            rospy.logerr(
                "末端绝对位姿目标需要 6 个数：[x, y, z, roll, pitch, yaw]"
            )
            return

        x, y, z, roll_deg, pitch_deg, yaw_deg = data

        rospy.loginfo(
            "收到末端绝对位姿目标：x=%.4f, y=%.4f, z=%.4f, roll=%.2f, pitch=%.2f, yaw=%.2f",
            x, y, z, roll_deg, pitch_deg, yaw_deg
        )

        # RPY 角度转弧度
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        # RPY 转四元数
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        target_pose.orientation.x = qx
        target_pose.orientation.y = qy
        target_pose.orientation.z = qz
        target_pose.orientation.w = qw

        self.execute_pose_goal(target_pose)

    def ee_delta_xyzrpy_callback(self, msg):
        """
        末端相对位姿控制。
        话题：
            /abbarm/ee_delta_xyzrpy_deg

        数据：
            [dx, dy, dz, droll, dpitch, dyaw]

        单位：
            dx/dy/dz：m
            droll/dpitch/dyaw：deg

        含义：
            在当前末端位姿基础上叠加位置增量和 RPY 姿态增量。

        注意：
            这里的姿态增量采用“当前 RPY + 增量 RPY”的方式，
            适合小角度动态调节和入门验证。
        """
        data = list(msg.data)

        if len(data) != 6:
            rospy.logerr(
                "末端相对位姿增量需要 6 个数：[dx, dy, dz, droll, dpitch, dyaw]"
            )
            return

        dx, dy, dz, droll_deg, dpitch_deg, dyaw_deg = data

        rospy.loginfo(
            "收到末端相对位姿增量：dx=%.4f, dy=%.4f, dz=%.4f, droll=%.2f, dpitch=%.2f, dyaw=%.2f",
            dx, dy, dz, droll_deg, dpitch_deg, dyaw_deg
        )

        # 获取当前末端位姿
        current_pose = self.arm.get_current_pose(self.end_link).pose
        target_pose = copy.deepcopy(current_pose)

        # 位置增量
        target_pose.position.x += dx
        target_pose.position.y += dy
        target_pose.position.z += dz

        # 当前四元数
        current_q = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]

        # 当前四元数转 RPY
        current_roll, current_pitch, current_yaw = euler_from_quaternion(current_q)

        # 增量角度转弧度
        droll = math.radians(droll_deg)
        dpitch = math.radians(dpitch_deg)
        dyaw = math.radians(dyaw_deg)

        # 当前 RPY + 增量 RPY
        target_roll = current_roll + droll
        target_pitch = current_pitch + dpitch
        target_yaw = current_yaw + dyaw

        # 新 RPY 转四元数
        qx, qy, qz, qw = quaternion_from_euler(
            target_roll,
            target_pitch,
            target_yaw
        )

        target_pose.orientation.x = qx
        target_pose.orientation.y = qy
        target_pose.orientation.z = qz
        target_pose.orientation.w = qw

        self.execute_pose_goal(target_pose)


def main():
    rospy.init_node("dynamic_arm_controller", anonymous=False)

    controller = DynamicArmController()

    rospy.loginfo("动态机械臂控制节点已启动。")
    rospy.loginfo("可发布以下话题进行控制：")
    rospy.loginfo("  /abbarm/joint_target_deg")
    rospy.loginfo("  /abbarm/joint_delta_deg")
    rospy.loginfo("  /abbarm/ee_target_xyzrpy_deg")
    rospy.loginfo("  /abbarm/ee_delta_xyzrpy_deg")

    rospy.spin()

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()