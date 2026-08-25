#!/usr/bin/env python3
"""Simulation-only placeholder for a real three-finger-hand command adapter."""

import rospy
from std_msgs.msg import String, UInt8

from handarm_moveit_demo.shared_teleop_core import GESTURE_NAMES


class MockHandAdapter:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        topics = config.get("topics", {})
        self.publisher = rospy.Publisher(topics.get("mock_hand_command", "/shared_teleop/mock_hand_command"),
                                         String, queue_size=10)
        rospy.Subscriber(topics.get("hand_action", "/shared_teleop/hand_action"),
                         UInt8, self.callback, queue_size=10)
        rospy.logwarn("Using mock hand adapter only; no CAN/EtherCAT/real hand interface is connected")

    def callback(self, message):
        name = GESTURE_NAMES.get(int(message.data), "UNKNOWN")
        self.publisher.publish(String(data=name))
        rospy.loginfo("MOCK three-finger-hand command: %s", name)


def main():
    rospy.init_node("mock_three_finger_hand_adapter")
    MockHandAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
