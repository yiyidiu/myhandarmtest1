#!/usr/bin/env python3
"""Debounce hand gestures, hold arm motion, and emit one hand action event."""

import json

import rospy
from std_msgs.msg import String, UInt8

from handarm_moveit_demo.msg import HandCommand
from handarm_moveit_demo.shared_teleop_core import GestureIsolationGate


class GestureIsolationNode:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        gesture = config.get("gesture", {})
        topics = config.get("topics", {})
        self.gate = GestureIsolationGate(
            gesture.get("stable_duration_s", 0.30),
            gesture.get("release_duration_s", 0.30),
            gesture.get("confidence_threshold", 0.75),
        )
        self.publisher = rospy.Publisher(topics.get("operator_command", "/shared_teleop/operator_command"),
                                         HandCommand, queue_size=1)
        self.action_publisher = rospy.Publisher(topics.get("hand_action", "/shared_teleop/hand_action"),
                                                UInt8, queue_size=1)
        self.diagnostics = rospy.Publisher(topics.get("gesture_diagnostics", "/shared_teleop/gesture_diagnostics"),
                                           String, queue_size=1)
        rospy.Subscriber(topics.get("raw_command", "/shared_teleop/raw_hand_command"),
                         HandCommand, self.callback, queue_size=1)

    def callback(self, message):
        result = self.gate.update(message.header.stamp.to_sec(), message.gesture,
                                  message.gesture_confidence)
        output = HandCommand()
        output.header = message.header
        output.confidence = message.confidence
        output.valid = message.valid
        output.gesture = message.gesture
        output.gesture_confidence = message.gesture_confidence
        if not result.hold_arm:
            output.twist = message.twist
        self.publisher.publish(output)
        if result.action is not None:
            self.action_publisher.publish(UInt8(data=result.action))
            rospy.loginfo("Stable hand gesture emitted once: %d", result.action)
        self.diagnostics.publish(String(data=json.dumps({
            "stamp": message.header.stamp.to_sec(), "hold_arm": result.hold_arm,
            "active_gesture": result.active_gesture, "reason": result.reason,
            "action_emitted": result.action,
        }, separators=(",", ":"))))


def main():
    rospy.init_node("gesture_isolation")
    GestureIsolationNode()
    rospy.spin()


if __name__ == "__main__":
    main()
