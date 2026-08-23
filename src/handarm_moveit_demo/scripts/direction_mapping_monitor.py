#!/usr/bin/env python3
"""Show the live mapped base-frame directions produced from hand motion."""

import threading

import rospy

from handarm_moveit_demo.msg import HandCommand


NAMES = ("base +X", "base +Y", "base +Z", "roll +X", "pitch +Y", "yaw +Z")


class DirectionMappingMonitor:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        topic = config.get("topics", {}).get(
            "raw_command", "/shared_teleop/raw_hand_command")
        self.threshold = float(rospy.get_param("~display_threshold", 1.0e-4))
        self.lock = threading.Lock()
        self.latest = None
        rospy.Subscriber(topic, HandCommand, self.callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.tick)

    def callback(self, message):
        with self.lock:
            self.latest = message

    def tick(self, _event):
        with self.lock:
            message = self.latest
        if message is None:
            rospy.loginfo_throttle(1.0, "waiting for mapped hand command")
            return
        values = (message.twist.linear.x, message.twist.linear.y,
                  message.twist.linear.z, message.twist.angular.x,
                  message.twist.angular.y, message.twist.angular.z)
        active = []
        for name, value in zip(NAMES, values):
            if abs(value) > self.threshold:
                base = name[:-2]
                active.append("{}{}={:+.4f}".format(
                    base, "+" if value > 0.0 else "-", value))
        rospy.loginfo("mapped base twist [%s] confidence=%s valid=%s active=%s",
                      " ".join("{:+.4f}".format(value) for value in values),
                      " ".join("{:.2f}".format(value) for value in message.confidence),
                      message.valid, ", ".join(active) if active else "deadband/zero")


def main():
    rospy.init_node("direction_mapping_monitor")
    DirectionMappingMonitor()
    rospy.spin()


if __name__ == "__main__":
    main()
