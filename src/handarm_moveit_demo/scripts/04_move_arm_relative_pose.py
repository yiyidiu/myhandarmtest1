#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按末端位姿控制机械臂 abbarm。
这里不是直接给一个绝对抓取点，而是在当前末端位姿基础上移动一点点，
更适合初学者验证“笛卡尔空间目标 -> MoveIt 规划 -> 执行”。
"""
import sys
import copy
import rospy
import moveit_commander


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_arm_relative_pose", anonymous=True)

    arm = moveit_commander.MoveGroupCommander("abbarm")
    arm.set_max_velocity_scaling_factor(0.10)
    arm.set_max_acceleration_scaling_factor(0.10)
    arm.set_planning_time(8.0)
    arm.set_num_planning_attempts(10)

    end_link = arm.get_end_effector_link()
    planning_frame = arm.get_planning_frame()
    rospy.loginfo("Planning frame: %s", planning_frame)
    rospy.loginfo("End link: %s", end_link)

    current_pose = arm.get_current_pose(end_link).pose
    target_pose = copy.deepcopy(current_pose)

    # 示例：保持当前姿态不变，只让末端沿 planning frame 的 z 方向上移 3 cm。
    # 如果你的模型方向不符合预期，可以先改成 x/y 小幅移动。
    target_pose.position.z -= 0.2

    arm.set_pose_target(target_pose, end_link)
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()

    rospy.loginfo("Move relative pose: %s", success)
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
