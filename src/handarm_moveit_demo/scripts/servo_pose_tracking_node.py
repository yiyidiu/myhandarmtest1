#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
servo_pose_tracking_node.py

Purpose:
- Track a time-varying target end-effector pose, not only position.
- Subscribe to /target_ee_pose as geometry_msgs/PoseStamped.
- Read current end-effector pose from TF.
- Compute position error and orientation error.
- Publish geometry_msgs/TwistStamped to /servo_server/delta_twist_cmds.

Control law:
    v_cmd     = Kp_pos * (p_d - p)
    omega_cmd = Kp_rot * e_R

where:
    p_d : desired end-effector position in base_frame
    p   : current end-effector position in base_frame
    e_R : orientation error vector from current orientation to desired orientation

This node does not call MoveIt plan/execute.
"""

import math
import threading

import rospy
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped


def norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def clamp_vector3(x, y, z, max_norm):
    n = norm3(x, y, z)
    if n < 1e-12 or n <= max_norm:
        return x, y, z

    scale = max_norm / n
    return x * scale, y * scale, z * scale


def normalize_quat(q):
    """
    q format: [x, y, z, w]
    """
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def quat_dot(q1, q2):
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]


def shortest_quat(q_des, q_cur):
    """
    q and -q represent the same orientation.
    Choose the representation of q_des that gives the shortest rotation
    from q_cur to q_des.
    """
    if quat_dot(q_des, q_cur) < 0.0:
        return [-q_des[0], -q_des[1], -q_des[2], -q_des[3]]
    return q_des


def quat_error_to_rotvec(q_des, q_cur):
    """
    Compute orientation error vector e_R.

    q_des, q_cur format: [x, y, z, w]

    Error quaternion:
        q_e = q_des * inverse(q_cur)

    Then convert q_e to axis-angle:
        e_R = theta * axis

    The result is approximately expressed in the command frame when
    both q_des and q_cur are expressed in the same base frame.
    """

    q_des = normalize_quat(q_des)
    q_cur = normalize_quat(q_cur)
    q_des = shortest_quat(q_des, q_cur)

    q_cur_inv = tf.transformations.quaternion_inverse(q_cur)
    q_err = tf.transformations.quaternion_multiply(q_des, q_cur_inv)
    q_err = normalize_quat(q_err)

    # Use shortest rotation.
    if q_err[3] < 0.0:
        q_err = [-q_err[0], -q_err[1], -q_err[2], -q_err[3]]

    vx = q_err[0]
    vy = q_err[1]
    vz = q_err[2]
    w = q_err[3]

    sin_half = norm3(vx, vy, vz)

    if sin_half < 1e-9:
        return 0.0, 0.0, 0.0

    angle = 2.0 * math.atan2(sin_half, w)

    # Since q_err.w is forced positive, angle should be within [0, pi].
    axis_x = vx / sin_half
    axis_y = vy / sin_half
    axis_z = vz / sin_half

    return angle * axis_x, angle * axis_y, angle * axis_z


class ServoPoseTrackingNode:
    def __init__(self):
        rospy.init_node("servo_pose_tracking_node")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_frame = rospy.get_param("~ee_frame", "tool0")

        self.target_topic = rospy.get_param("~target_topic", "/target_ee_pose")
        self.twist_topic = rospy.get_param("~twist_topic", "/servo_server/delta_twist_cmds")

        self.kp_pos = float(rospy.get_param("~kp_pos", 0.8))
        self.kp_rot = float(rospy.get_param("~kp_rot", 0.6))

        self.max_linear_speed = float(rospy.get_param("~max_linear_speed", 0.04))     # m/s
        self.max_angular_speed = float(rospy.get_param("~max_angular_speed", 0.40))   # rad/s

        self.pos_deadband = float(rospy.get_param("~pos_deadband", 0.001))            # m
        self.rot_deadband = float(rospy.get_param("~rot_deadband", 0.01))             # rad

        self.rate_hz = float(rospy.get_param("~rate", 50.0))
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.5))          # s

        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=1)

        self.lock = threading.Lock()
        self.target_pose = None
        self.target_stamp = None

        self.sub = rospy.Subscriber(self.target_topic, PoseStamped, self.target_callback, queue_size=1)

        rospy.loginfo("========== Servo Pose Tracking Node ==========")
        rospy.loginfo("base_frame: %s", self.base_frame)
        rospy.loginfo("ee_frame: %s", self.ee_frame)
        rospy.loginfo("target_topic: %s", self.target_topic)
        rospy.loginfo("twist_topic: %s", self.twist_topic)
        rospy.loginfo("kp_pos=%.3f kp_rot=%.3f", self.kp_pos, self.kp_rot)
        rospy.loginfo("max_linear_speed=%.3f m/s max_angular_speed=%.3f rad/s",
                      self.max_linear_speed, self.max_angular_speed)

        self.wait_for_tf()

    def wait_for_tf(self):
        rospy.loginfo("Waiting for TF %s -> %s ...", self.base_frame, self.ee_frame)
        self.listener.waitForTransform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(10.0))
        rospy.loginfo("TF is ready.")

    def target_callback(self, msg):
        """
        Receive target pose.

        The target pose can be published in base_frame directly.
        If it is published in another TF frame, this node tries to transform it
        into base_frame.
        """

        try:
            if msg.header.frame_id == "":
                msg.header.frame_id = self.base_frame

            if msg.header.frame_id != self.base_frame:
                # Use latest available transform.
                msg.header.stamp = rospy.Time(0)
                target_in_base = self.listener.transformPose(self.base_frame, msg)
            else:
                target_in_base = msg

            q = target_in_base.pose.orientation
            q_list = normalize_quat([q.x, q.y, q.z, q.w])
            target_in_base.pose.orientation.x = q_list[0]
            target_in_base.pose.orientation.y = q_list[1]
            target_in_base.pose.orientation.z = q_list[2]
            target_in_base.pose.orientation.w = q_list[3]

            with self.lock:
                self.target_pose = target_in_base
                self.target_stamp = rospy.Time.now()

        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            rospy.logwarn_throttle(1.0, "Failed to transform target pose to %s: %s",
                                   self.base_frame, str(exc))

    def lookup_current_pose(self):
        self.listener.waitForTransform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(1.0))
        trans, rot = self.listener.lookupTransform(self.base_frame, self.ee_frame, rospy.Time(0))

        p = [trans[0], trans[1], trans[2]]
        q = normalize_quat([rot[0], rot[1], rot[2], rot[3]])

        return p, q

    def get_target_pose(self):
        with self.lock:
            if self.target_pose is None:
                return None, None

            age = (rospy.Time.now() - self.target_stamp).to_sec()
            if self.target_timeout > 0.0 and age > self.target_timeout:
                return None, age

            pose = self.target_pose
            return pose, age

    def publish_zero_twist(self):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        self.pub.publish(msg)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        last_log_time = rospy.Time(0)

        while not rospy.is_shutdown():
            target, age = self.get_target_pose()

            if target is None:
                rospy.logwarn_throttle(2.0, "No valid target pose. Publishing zero twist.")
                self.publish_zero_twist()
                rate.sleep()
                continue

            try:
                p_cur, q_cur = self.lookup_current_pose()
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                rospy.logwarn_throttle(1.0, "TF lookup failed: %s", str(exc))
                self.publish_zero_twist()
                rate.sleep()
                continue

            p_des = [
                target.pose.position.x,
                target.pose.position.y,
                target.pose.position.z
            ]

            q_des = normalize_quat([
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
                target.pose.orientation.w
            ])

            # Position error.
            ex = p_des[0] - p_cur[0]
            ey = p_des[1] - p_cur[1]
            ez = p_des[2] - p_cur[2]

            if abs(ex) < self.pos_deadband:
                ex = 0.0
            if abs(ey) < self.pos_deadband:
                ey = 0.0
            if abs(ez) < self.pos_deadband:
                ez = 0.0

            # Orientation error.
            erx, ery, erz = quat_error_to_rotvec(q_des, q_cur)

            if abs(erx) < self.rot_deadband:
                erx = 0.0
            if abs(ery) < self.rot_deadband:
                ery = 0.0
            if abs(erz) < self.rot_deadband:
                erz = 0.0

            # Control law.
            vx = self.kp_pos * ex
            vy = self.kp_pos * ey
            vz = self.kp_pos * ez

            wx = self.kp_rot * erx
            wy = self.kp_rot * ery
            wz = self.kp_rot * erz

            vx, vy, vz = clamp_vector3(vx, vy, vz, self.max_linear_speed)
            wx, wy, wz = clamp_vector3(wx, wy, wz, self.max_angular_speed)

            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.base_frame

            msg.twist.linear.x = vx
            msg.twist.linear.y = vy
            msg.twist.linear.z = vz

            msg.twist.angular.x = wx
            msg.twist.angular.y = wy
            msg.twist.angular.z = wz

            self.pub.publish(msg)

            now = rospy.Time.now()
            if (now - last_log_time).to_sec() > 1.0:
                pos_err = norm3(ex, ey, ez)
                rot_err = norm3(erx, ery, erz)
                rospy.loginfo(
                    "target_age=%.3f | ep=%.4f m er=%.4f rad | v=[%.3f %.3f %.3f] w=[%.3f %.3f %.3f]",
                    age if age is not None else -1.0,
                    pos_err,
                    rot_err,
                    vx, vy, vz,
                    wx, wy, wz
                )
                last_log_time = now

            rate.sleep()

        # Stop at shutdown.
        for _ in range(10):
            self.publish_zero_twist()
            rate.sleep()


if __name__ == "__main__":
    node = ServoPoseTrackingNode()
    node.run()