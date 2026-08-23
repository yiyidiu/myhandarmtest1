#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印 MoveIt 当前能看到的机器人信息。
先运行：roslaunch abb120_moveit_config1 demo.launch
或者：roslaunch abb120_moveit_config1 demo_gazebo.launch
再运行：rosrun handarm_moveit_demo 01_print_moveit_info.py
"""
import sys
import rospy
import moveit_commander


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("print_moveit_info", anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    print("\n====== MoveIt groups ======")
    print(robot.get_group_names())

    print("\n====== Current robot state ======")
    print(robot.get_current_state())

    for group_name in robot.get_group_names():
        group = moveit_commander.MoveGroupCommander(group_name)
        print("\n====== Group: {} ======".format(group_name))
        print("planning frame:", group.get_planning_frame())
        print("end effector link:", group.get_end_effector_link())
        print("active joints:", group.get_active_joints())
        print("current joint values:", group.get_current_joint_values())
        try:
            print("named targets:", group.get_named_targets())
        except Exception as exc:
            print("named targets read failed:", exc)

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
