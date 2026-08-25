#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position-error Cartesian servo demo for ABB IRB120.
It reads the current tool0 pose from TF, creates a small relative target, and publishes
TwistStamped commands using a simple proportional law:
    v = Kp * (p_des - p_now)
The first version controls position only and keeps angular velocity zero.
"""
import math
import rospy
import tf
from geometry_msgs.msg import TwistStamped


def clamp(value, limit):
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def main():
    rospy.init_node("servo_pose_step_test")

    base_frame = rospy.get_param("~base_frame", "base_link")
    ee_frame = rospy.get_param("~ee_frame", "tool0")
    topic = rospy.get_param("~topic", "/servo_server/delta_twist_cmds")

    # Relative target from current tool0 pose, in base_link frame.
    dx = float(rospy.get_param("~dx", 0.05))
    dy = float(rospy.get_param("~dy", 0.00))
    dz = float(rospy.get_param("~dz", 0.00))

    kp = float(rospy.get_param("~kp", 0.8))
    max_speed = float(rospy.get_param("~max_speed", 0.04))
    pos_tolerance = float(rospy.get_param("~pos_tolerance", 0.003))
    rate_hz = float(rospy.get_param("~rate", 50.0))
    timeout = float(rospy.get_param("~timeout", 8.0))

    listener = tf.TransformListener()
    pub = rospy.Publisher(topic, TwistStamped, queue_size=10)
    rate = rospy.Rate(rate_hz)

    rospy.loginfo("Waiting for TF %s -> %s ...", base_frame, ee_frame)
    listener.waitForTransform(base_frame, ee_frame, rospy.Time(0), rospy.Duration(10.0))
    trans, _ = listener.lookupTransform(base_frame, ee_frame, rospy.Time(0))

    target = [trans[0] + dx, trans[1] + dy, trans[2] + dz]
    rospy.loginfo("Initial position: [%.4f, %.4f, %.4f]", trans[0], trans[1], trans[2])
    rospy.loginfo("Target position : [%.4f, %.4f, %.4f]", target[0], target[1], target[2])

    start_time = rospy.Time.now()
    last_dist = None

    while not rospy.is_shutdown():
        if (rospy.Time.now() - start_time).to_sec() > timeout:
            rospy.logwarn("Timeout reached before target tolerance.")
            break

        try:
            trans, _ = listener.lookupTransform(base_frame, ee_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            rospy.logwarn_throttle(1.0, "TF lookup failed: %s", str(exc))
            rate.sleep()
            continue

        err = [target[0] - trans[0], target[1] - trans[1], target[2] - trans[2]]
        dist = math.sqrt(err[0]**2 + err[1]**2 + err[2]**2)
        last_dist = dist
        if dist < pos_tolerance:
            rospy.loginfo("Target reached, position error %.4f m", dist)
            break

        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = clamp(kp * err[0], max_speed)
        msg.twist.linear.y = clamp(kp * err[1], max_speed)
        msg.twist.linear.z = clamp(kp * err[2], max_speed)
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        pub.publish(msg)
        rate.sleep()

    # Stop.
    for _ in range(10):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = base_frame
        pub.publish(msg)
        rate.sleep()

    if last_dist is not None:
        rospy.loginfo("Final position error: %.4f m", last_dist)


if __name__ == "__main__":
    main()
