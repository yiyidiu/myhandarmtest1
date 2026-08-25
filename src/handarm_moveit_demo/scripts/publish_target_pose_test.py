#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import tf
from geometry_msgs.msg import PoseStamped


def normalize_quat(q):
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0]/n, q[1]/n, q[2]/n, q[3]/n]


if __name__ == "__main__":
    rospy.init_node("publish_target_pose_test")

    base_frame = rospy.get_param("~base_frame", "base_link")
    ee_frame = rospy.get_param("~ee_frame", "tool0")
    topic = rospy.get_param("~topic", "/target_ee_pose")

    dx = float(rospy.get_param("~dx", 0.02))
    dy = float(rospy.get_param("~dy", 0.0))
    dz = float(rospy.get_param("~dz", 0.0))

    droll = math.radians(float(rospy.get_param("~droll", 0.0)))
    dpitch = math.radians(float(rospy.get_param("~dpitch", 0.0)))
    dyaw = math.radians(float(rospy.get_param("~dyaw", 0.0)))

    rate_hz = float(rospy.get_param("~rate", 50.0))

    listener = tf.TransformListener()
    pub = rospy.Publisher(topic, PoseStamped, queue_size=1)

    rospy.loginfo("Waiting for TF %s -> %s ...", base_frame, ee_frame)
    listener.waitForTransform(base_frame, ee_frame, rospy.Time(0), rospy.Duration(10.0))

    trans, rot = listener.lookupTransform(base_frame, ee_frame, rospy.Time(0))

    q_current = normalize_quat([rot[0], rot[1], rot[2], rot[3]])
    q_delta = tf.transformations.quaternion_from_euler(droll, dpitch, dyaw)

    q_target = tf.transformations.quaternion_multiply(q_delta, q_current)
    q_target = normalize_quat(q_target)

    target = PoseStamped()
    target.header.frame_id = base_frame
    target.pose.position.x = trans[0] + dx
    target.pose.position.y = trans[1] + dy
    target.pose.position.z = trans[2] + dz
    target.pose.orientation.x = q_target[0]
    target.pose.orientation.y = q_target[1]
    target.pose.orientation.z = q_target[2]
    target.pose.orientation.w = q_target[3]

    rospy.loginfo("Initial current pose:")
    rospy.loginfo("  p=[%.4f %.4f %.4f]", trans[0], trans[1], trans[2])
    rospy.loginfo("Target pose:")
    rospy.loginfo("  p=[%.4f %.4f %.4f]", target.pose.position.x, target.pose.position.y, target.pose.position.z)
    rospy.loginfo("  q=[%.4f %.4f %.4f %.4f]",
                  target.pose.orientation.x,
                  target.pose.orientation.y,
                  target.pose.orientation.z,
                  target.pose.orientation.w)

    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        target.header.stamp = rospy.Time.now()
        pub.publish(target)
        rate.sleep()