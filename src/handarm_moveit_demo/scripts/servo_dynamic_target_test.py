#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Cartesian target tracking test for ABB IRB120 + MoveIt Servo.

Purpose:
- This node does NOT call MoveIt plan/execute.
- It creates a time-varying desired end-effector position p_d(t).
- It reads the current end-effector position p(t) from TF.
- It publishes a TwistStamped command:
      v_cmd = Kp * (p_d - p) + feedforward * p_dot_d
  to /servo_server/delta_twist_cmds.

Recommended first use:
  1) Start Gazebo + MoveIt + Servo + bridge.
  2) Move the arm to a non-singular ready pose.
  3) Disable rotational control dimensions in Servo.
  4) Run this node with a small z-axis sinusoid.
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


def clamp_vector(vx, vy, vz, max_norm):
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm <= max_norm or norm < 1e-12:
        return vx, vy, vz
    scale = max_norm / norm
    return vx * scale, vy * scale, vz * scale


class ServoDynamicTargetTest:
    def __init__(self):
        rospy.init_node("servo_dynamic_target_test")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_frame = rospy.get_param("~ee_frame", "tool0")
        self.topic = rospy.get_param("~topic", "/servo_server/delta_twist_cmds")

        # Motion mode: sine_z, sine_x, sine_y, circle_xy, circle_xz, circle_yz
        self.mode = rospy.get_param("~mode", "sine_z")

        # Dynamic target parameters.
        self.amplitude = float(rospy.get_param("~amplitude", 0.015))  # m
        self.frequency = float(rospy.get_param("~frequency", 0.05))   # Hz
        self.duration = float(rospy.get_param("~duration", 30.0))      # s, <=0 means run forever

        # Servo law parameters.
        self.kp = float(rospy.get_param("~kp", 0.8))
        self.feedforward = float(rospy.get_param("~feedforward", 0.5))
        self.max_speed = float(rospy.get_param("~max_speed", 0.02))
        self.deadband = float(rospy.get_param("~deadband", 0.001))
        self.rate_hz = float(rospy.get_param("~rate", 50.0))

        # Smoothly grow dynamic target amplitude at startup to avoid sudden commands.
        self.ramp_time = float(rospy.get_param("~ramp_time", 3.0))

        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher(self.topic, TwistStamped, queue_size=1)

        rospy.loginfo("Dynamic target test starting.")
        rospy.loginfo("  mode=%s amplitude=%.4f m frequency=%.4f Hz", self.mode, self.amplitude, self.frequency)
        rospy.loginfo("  kp=%.3f feedforward=%.3f max_speed=%.4f m/s", self.kp, self.feedforward, self.max_speed)
        rospy.loginfo("  base_frame=%s ee_frame=%s topic=%s", self.base_frame, self.ee_frame, self.topic)

        self._wait_for_tf()
        self.p0 = self._lookup_position()
        rospy.loginfo("Initial target center p0 = [%.4f, %.4f, %.4f]", self.p0[0], self.p0[1], self.p0[2])

    def _wait_for_tf(self):
        rospy.loginfo("Waiting for TF %s -> %s ...", self.base_frame, self.ee_frame)
        self.listener.waitForTransform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(10.0))

    def _lookup_position(self):
        self.listener.waitForTransform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(1.0))
        trans, _ = self.listener.lookupTransform(self.base_frame, self.ee_frame, rospy.Time(0))
        return [trans[0], trans[1], trans[2]]

    def _target_and_velocity(self, t):
        """Return p_d(t) and p_dot_d(t) in base_frame."""
        omega = 2.0 * math.pi * self.frequency
        s = math.sin(omega * t)
        c = math.cos(omega * t)

        # Smooth amplitude ramp.
        if self.ramp_time > 1e-6:
            ramp = min(1.0, max(0.0, t / self.ramp_time))
        else:
            ramp = 1.0
        A = self.amplitude * ramp

        # Do not add feedforward from the ramp derivative; keep it conservative.
        vx = vy = vz = 0.0
        dx = dy = dz = 0.0

        if self.mode == "sine_x":
            dx = A * s
            vx = A * omega * c
        elif self.mode == "sine_y":
            dy = A * s
            vy = A * omega * c
        elif self.mode == "sine_z":
            dz = A * s
            vz = A * omega * c
        elif self.mode == "circle_xy":
            dx = A * c
            dy = A * s
            vx = -A * omega * s
            vy = A * omega * c
        elif self.mode == "circle_xz":
            dx = A * c
            dz = A * s
            vx = -A * omega * s
            vz = A * omega * c
        elif self.mode == "circle_yz":
            dy = A * c
            dz = A * s
            vy = -A * omega * s
            vz = A * omega * c
        else:
            rospy.logwarn_throttle(2.0, "Unknown mode '%s'. Falling back to sine_z.", self.mode)
            dz = A * s
            vz = A * omega * c

        pd = [self.p0[0] + dx, self.p0[1] + dy, self.p0[2] + dz]
        vd = [vx, vy, vz]
        return pd, vd

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        start_time = rospy.Time.now()
        last_log_time = rospy.Time(0)

        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_time).to_sec()
            if self.duration > 0.0 and t > self.duration:
                break

            try:
                p = self._lookup_position()
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                rospy.logwarn_throttle(1.0, "TF lookup failed: %s", str(exc))
                rate.sleep()
                continue

            pd, vd = self._target_and_velocity(t)

            ex = pd[0] - p[0]
            ey = pd[1] - p[1]
            ez = pd[2] - p[2]

            # Deadband avoids tiny oscillatory commands.
            if abs(ex) < self.deadband:
                ex = 0.0
            if abs(ey) < self.deadband:
                ey = 0.0
            if abs(ez) < self.deadband:
                ez = 0.0

            vx = self.kp * ex + self.feedforward * vd[0]
            vy = self.kp * ey + self.feedforward * vd[1]
            vz = self.kp * ez + self.feedforward * vd[2]
            vx, vy, vz = clamp_vector(vx, vy, vz, self.max_speed)

            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.base_frame
            msg.twist.linear.x = vx
            msg.twist.linear.y = vy
            msg.twist.linear.z = vz
            msg.twist.angular.x = 0.0
            msg.twist.angular.y = 0.0
            msg.twist.angular.z = 0.0
            self.pub.publish(msg)

            now = rospy.Time.now()
            if (now - last_log_time).to_sec() > 1.0:
                err_norm = math.sqrt(ex * ex + ey * ey + ez * ez)
                rospy.loginfo("t=%.1f pd=[%.3f %.3f %.3f] p=[%.3f %.3f %.3f] |e|=%.4f v=[%.3f %.3f %.3f]",
                              t, pd[0], pd[1], pd[2], p[0], p[1], p[2], err_norm, vx, vy, vz)
                last_log_time = now

            rate.sleep()

        # Publish a few zero commands at the end.
        for _ in range(10):
            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.base_frame
            self.pub.publish(msg)
            rate.sleep()

        rospy.loginfo("Dynamic target test finished.")


if __name__ == "__main__":
    node = ServoDynamicTargetTest()
    node.run()
