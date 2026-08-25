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
    rospy.init_node("publish_dynamic_target_pose_test")

    base_frame = rospy.get_param("~base_frame", "base_link")
    ee_frame = rospy.get_param("~ee_frame", "tool0")
    topic = rospy.get_param("~topic", "/target_ee_pose")

    mode = rospy.get_param("~mode", "sine_z")

    pos_amp = float(rospy.get_param("~pos_amp", 0.01))          # m
    pos_freq = float(rospy.get_param("~pos_freq", 0.03))        # Hz

    roll_amp_deg = float(rospy.get_param("~roll_amp", 0.0))     # deg
    pitch_amp_deg = float(rospy.get_param("~pitch_amp", 0.0))   # deg
    yaw_amp_deg = float(rospy.get_param("~yaw_amp", 0.0))       # deg
    rot_freq = float(rospy.get_param("~rot_freq", 0.03))        # Hz

    duration = float(rospy.get_param("~duration", 30.0))
    rate_hz = float(rospy.get_param("~rate", 50.0))

    listener = tf.TransformListener()
    pub = rospy.Publisher(topic, PoseStamped, queue_size=1)

    rospy.loginfo("Waiting for TF %s -> %s ...", base_frame, ee_frame)
    listener.waitForTransform(base_frame, ee_frame, rospy.Time(0), rospy.Duration(10.0))

    trans, rot = listener.lookupTransform(base_frame, ee_frame, rospy.Time(0))

    p0 = [trans[0], trans[1], trans[2]]
    q0 = normalize_quat([rot[0], rot[1], rot[2], rot[3]])

    rospy.loginfo("Dynamic target pose publisher started.")
    rospy.loginfo("p0=[%.4f %.4f %.4f]", p0[0], p0[1], p0[2])
    rospy.loginfo("q0=[%.4f %.4f %.4f %.4f]", q0[0], q0[1], q0[2], q0[3])
    rospy.loginfo("mode=%s pos_amp=%.4f pos_freq=%.4f", mode, pos_amp, pos_freq)
    rospy.loginfo("roll_amp=%.2f pitch_amp=%.2f yaw_amp=%.2f rot_freq=%.4f",
                  roll_amp_deg, pitch_amp_deg, yaw_amp_deg, rot_freq)

    start_time = rospy.Time.now()
    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        t = (rospy.Time.now() - start_time).to_sec()
        if duration > 0.0 and t > duration:
            break

        sp = math.sin(2.0 * math.pi * pos_freq * t)
        cp = math.cos(2.0 * math.pi * pos_freq * t)

        dx = dy = dz = 0.0

        if mode == "sine_x":
            dx = pos_amp * sp
        elif mode == "sine_y":
            dy = pos_amp * sp
        elif mode == "sine_z":
            dz = pos_amp * sp
        elif mode == "circle_xz":
            dx = pos_amp * cp
            dz = pos_amp * sp
        elif mode == "circle_xy":
            dx = pos_amp * cp
            dy = pos_amp * sp
        elif mode == "circle_yz":
            dy = pos_amp * cp
            dz = pos_amp * sp
        else:
            dz = pos_amp * sp

        sr = math.sin(2.0 * math.pi * rot_freq * t)

        droll = math.radians(roll_amp_deg * sr)
        dpitch = math.radians(pitch_amp_deg * sr)
        dyaw = math.radians(yaw_amp_deg * sr)

        q_delta = tf.transformations.quaternion_from_euler(droll, dpitch, dyaw)

        # Base-frame rotation increment.
        q_target = tf.transformations.quaternion_multiply(q_delta, q0)
        q_target = normalize_quat(q_target)

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = base_frame

        msg.pose.position.x = p0[0] + dx
        msg.pose.position.y = p0[1] + dy
        msg.pose.position.z = p0[2] + dz

        msg.pose.orientation.x = q_target[0]
        msg.pose.orientation.y = q_target[1]
        msg.pose.orientation.z = q_target[2]
        msg.pose.orientation.w = q_target[3]

        pub.publish(msg)

        rospy.loginfo_throttle(
            1.0,
            "target p=[%.3f %.3f %.3f] rpy_delta=[%.2f %.2f %.2f] deg",
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            math.degrees(droll),
            math.degrees(dpitch),
            math.degrees(dyaw)
        )

        rate.sleep()

    rospy.loginfo("Dynamic target pose publisher finished.")