#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
让机械臂规划组 abbarm 运动到 SRDF 中定义的命名姿态 up。
适合第一步验证：MoveIt 编程接口是否能驱动 RViz / Gazebo 中的机械臂。
"""
import sys
import rospy
import moveit_commander


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_arm_named", anonymous=True)

    arm = moveit_commander.MoveGroupCommander("abbarm")
    arm.set_max_velocity_scaling_factor(0.15)
    arm.set_max_acceleration_scaling_factor(0.15)
    arm.set_planning_time(5.0)

    rospy.loginfo("Planning group: abbarm")
    rospy.loginfo("Named targets: %s", arm.get_named_targets())

    arm.set_named_target("up")
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()

    rospy.loginfo("Move to named target 'up': %s", success)
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
