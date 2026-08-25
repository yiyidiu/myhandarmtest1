#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MoveIt Servo test: publish a short Cartesian velocity pulse.
This tests whether /servo_server/delta_twist_cmds -> MoveIt Servo -> /controller_gazebo/command works.
"""
import rospy
from geometry_msgs.msg import TwistStamped


def main():
    rospy.init_node("servo_twist_pulse_test")

    topic = rospy.get_param("~topic", "/servo_server/delta_twist_cmds")
    frame_id = rospy.get_param("~frame_id", "base_link")
    axis = rospy.get_param("~axis", "x")       # x, y, z, rx, ry, rz
    speed = float(rospy.get_param("~speed", 0.03))
    duration = float(rospy.get_param("~duration", 2.0))
    rate_hz = float(rospy.get_param("~rate", 50.0))

    pub = rospy.Publisher(topic, TwistStamped, queue_size=10)
    rate = rospy.Rate(rate_hz)
    start = rospy.Time.now()

    rospy.loginfo("Publishing Servo pulse: axis=%s speed=%.4f duration=%.2fs frame=%s topic=%s",
                  axis, speed, duration, frame_id, topic)

    while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < duration:
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        if axis == "x":
            msg.twist.linear.x = speed
        elif axis == "y":
            msg.twist.linear.y = speed
        elif axis == "z":
            msg.twist.linear.z = speed
        elif axis == "rx":
            msg.twist.angular.x = speed
        elif axis == "ry":
            msg.twist.angular.y = speed
        elif axis == "rz":
            msg.twist.angular.z = speed
        else:
            rospy.logerr("Unsupported axis: %s. Use x/y/z/rx/ry/rz", axis)
            return
        pub.publish(msg)
        rate.sleep()

    # Publish several zero commands to stop cleanly.
    for _ in range(10):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        pub.publish(msg)
        rate.sleep()

    rospy.loginfo("Servo pulse finished.")


if __name__ == "__main__":
    main()
