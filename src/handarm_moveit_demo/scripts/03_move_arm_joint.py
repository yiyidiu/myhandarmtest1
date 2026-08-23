#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 6 个关节角直接控制机械臂 abbarm。
单位：弧度 rad。关节顺序必须与 MoveIt 中 abbarm 的 active_joints 一致。
"""
import sys
import math
import rospy
import moveit_commander


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_arm_joint", anonymous=True)

    arm = moveit_commander.MoveGroupCommander("abbarm")
    arm.set_max_velocity_scaling_factor(0.1)
    arm.set_max_acceleration_scaling_factor(0.1)
    arm.set_planning_time(5.0)

    joints = arm.get_current_joint_values()
    rospy.loginfo("Active joints: %s", arm.get_active_joints())
    rospy.loginfo("Current joints: %s", [round(v, 4) for v in joints])

    # 示例：从当前姿态出发，只让 joint_1 和 joint_2 小幅改变。
    # 这样比随便给一个末端位姿更不容易规划失败。
    target = list(joints)
    # target[0] += math.radians(10.0)
    target[1] += math.radians(30.0)

    arm.set_joint_value_target(target)
    success = arm.go(wait=True)
    arm.stop()

    rospy.loginfo("Move joint target: %s", success)
    rospy.loginfo("Target joints: %s", [round(v, 4) for v in target])
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
