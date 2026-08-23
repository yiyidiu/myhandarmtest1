#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
控制手部规划组 hand 的命名姿态。
SRDF 中已有 start1、grasp1、grasp2。
注意：这里走 MoveIt 的手部规划组，不是你的下位机 CAN 自适应抓取逻辑。
"""
import sys
import rospy
import moveit_commander


def move_to(hand, name):
    hand.set_named_target(name)
    ok = hand.go(wait=True)
    hand.stop()
    hand.clear_pose_targets()
    rospy.loginfo("Move hand to %s: %s", name, ok)
    rospy.sleep(1.0)


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_hand_named", anonymous=True)

    hand = moveit_commander.MoveGroupCommander("hand")
    hand.set_max_velocity_scaling_factor(0.20)
    hand.set_max_acceleration_scaling_factor(0.20)
    hand.set_planning_time(5.0)

    rospy.loginfo("Named targets: %s", hand.get_named_targets())
    move_to(hand, "start1")
    move_to(hand, "grasp2")
    move_to(hand, "start1")

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
