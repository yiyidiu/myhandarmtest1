#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import termios
import tty

import rospy
from std_msgs.msg import Float64MultiArray


class KeyboardEEDeltaControl:
    def __init__(self):
        rospy.init_node("keyboard_ee_delta_control", anonymous=False)

        self.topic_name = rospy.get_param(
            "~topic_name",
            "/abbarm/ee_delta_xyzrpy_deg"
        )

        # 位置步长，单位 m
        self.step_xyz = rospy.get_param("~step_xyz", 0.01)

        # 姿态步长，单位 deg
        self.step_rpy = rospy.get_param("~step_rpy", 2.0)

        self.pub = rospy.Publisher(
            self.topic_name,
            Float64MultiArray,
            queue_size=1
        )

        self.print_help()

    def print_help(self):
        print("")
        print("========== 末端相对位姿键盘控制 ==========")
        print("")
        print("发布话题：{}".format(self.topic_name))
        print("")
        print("数据格式：")
        print("  [dx, dy, dz, droll, dpitch, dyaw]")
        print("")
        print("平移控制：")
        print("  R / F  -> X轴 + / -")
        print("  A / D  -> Y轴 + / -")
        print("  W / S  -> Z轴 + / -")
        print("")
        print("姿态控制：")
        print("  I / K  -> Roll  + / -")
        print("  J / L  -> Pitch + / -")
        print("  U / O  -> Yaw   + / -")
        print("")
        print("其他：")
        print("  Space  -> 发送零增量")
        print("  Q / ESC -> 退出")
        print("")
        print("当前步长：")
        print("  位置步长：{} m".format(self.step_xyz))
        print("  姿态步长：{} deg".format(self.step_rpy))
        print("")
        print("=========================================")
        print("")

    def get_key(self):
        """
        阻塞读取一个按键。
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return key.lower()

    def publish_delta(self, dx, dy, dz, droll, dpitch, dyaw):
        """
        发布末端相对位姿增量。
        """
        msg = Float64MultiArray()
        msg.data = [dx, dy, dz, droll, dpitch, dyaw]
        self.pub.publish(msg)

        rospy.loginfo(
            "发布末端增量: dx=%.4f dy=%.4f dz=%.4f | droll=%.2f dpitch=%.2f dyaw=%.2f",
            dx, dy, dz, droll, dpitch, dyaw
        )

    def run(self):
        while not rospy.is_shutdown():
            key = self.get_key()

            dx = dy = dz = 0.0
            droll = dpitch = dyaw = 0.0

            # ESC 退出
            if key == "\x1b":
                print("\n退出键盘控制。")
                break

            # Ctrl+C 退出
            if key == "\x03":
                print("\n退出键盘控制。")
                break

            # q 退出
            if key == "q":
                print("\n退出键盘控制。")
                break

            # 空格发送零增量
            if key == " ":
                self.publish_delta(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                continue

            # =========================
            # 平移控制
            # =========================

            # X 轴
            if key == "r":
                dx = self.step_xyz
            elif key == "f":
                dx = -self.step_xyz

            # Y 轴
            elif key == "a":
                dy = self.step_xyz
            elif key == "d":
                dy = -self.step_xyz

            # Z 轴
            elif key == "w":
                dz = self.step_xyz
            elif key == "s":
                dz = -self.step_xyz

            # =========================
            # 姿态控制
            # =========================

            # Roll
            elif key == "i":
                droll = self.step_rpy
            elif key == "k":
                droll = -self.step_rpy

            # Pitch
            elif key == "j":
                dpitch = self.step_rpy
            elif key == "l":
                dpitch = -self.step_rpy

            # Yaw
            elif key == "u":
                dyaw = self.step_rpy
            elif key == "o":
                dyaw = -self.step_rpy

            else:
                continue

            self.publish_delta(dx, dy, dz, droll, dpitch, dyaw)


if __name__ == "__main__":
    node = KeyboardEEDeltaControl()
    node.run()