#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import JointJog


def clamp(x, limit):
    if x > limit:
        return limit
    if x < -limit:
        return -limit
    return x


class JointTargetServoController:
    def __init__(self):
        rospy.init_node("joint_target_servo_controller")

        self.joint_names = rospy.get_param("~joint_names", [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ])

        self.joint_state_topic = rospy.get_param("~joint_state_topic", "/joint_states")
        self.target_topic = rospy.get_param("~target_topic", "/abbarm/joint_target_deg")
        self.joint_jog_topic = rospy.get_param("~joint_jog_topic", "/servo_server/delta_joint_cmds")

        self.kp = float(rospy.get_param("~kp", 0.8))
        self.max_joint_speed = float(rospy.get_param("~max_joint_speed", 0.25))  # rad/s
        self.max_joint_accel = float(rospy.get_param("~max_joint_accel", 1.0))   # rad/s^2
        self.tolerance = math.radians(float(rospy.get_param("~tolerance_deg", 0.5)))
        self.rate_hz = float(rospy.get_param("~rate", 50.0))
        self.max_duration = float(rospy.get_param("~max_duration", 20.0))

        self.lock = threading.Lock()
        self.current_pos = {}
        self.target_rad = None
        self.active = False
        self.start_time = None
        self.last_cmd = [0.0] * len(self.joint_names)

        self.js_sub = rospy.Subscriber(
            self.joint_state_topic, JointState, self.joint_state_cb, queue_size=1
        )

        self.target_sub = rospy.Subscriber(
            self.target_topic, Float64MultiArray, self.target_cb, queue_size=1
        )

        self.joint_jog_pub = rospy.Publisher(
            self.joint_jog_topic, JointJog, queue_size=1
        )

        rospy.loginfo("========== Joint Target Servo Controller ==========")
        rospy.loginfo("target_topic: %s", self.target_topic)
        rospy.loginfo("joint_jog_topic: %s", self.joint_jog_topic)
        rospy.loginfo("joint_names: %s", self.joint_names)
        rospy.loginfo(
            "kp=%.3f max_joint_speed=%.3f rad/s max_joint_accel=%.3f rad/s^2 tolerance=%.3f deg",
            self.kp,
            self.max_joint_speed,
            self.max_joint_accel,
            math.degrees(self.tolerance),
        )

    def joint_state_cb(self, msg):
        with self.lock:
            for name, pos in zip(msg.name, msg.position):
                self.current_pos[name] = pos

    def target_cb(self, msg):
        if len(msg.data) != len(self.joint_names):
            rospy.logerr(
                "Target length error. Expected %d values, got %d.",
                len(self.joint_names),
                len(msg.data),
            )
            return

        target = [math.radians(x) for x in msg.data]

        with self.lock:
            self.target_rad = target
            self.active = True
            self.start_time = rospy.Time.now()
            self.last_cmd = [0.0] * len(self.joint_names)

        rospy.loginfo("Received joint target deg: %s", ["%.2f" % x for x in msg.data])

    def get_current_vector(self):
        with self.lock:
            if not all(name in self.current_pos for name in self.joint_names):
                return None
            return [self.current_pos[name] for name in self.joint_names]

    def get_target_state(self):
        with self.lock:
            if self.target_rad is None:
                return None, False, None
            return list(self.target_rad), self.active, self.start_time

    def stop_active_motion(self):
        with self.lock:
            self.active = False
            self.last_cmd = [0.0] * len(self.joint_names)

    def publish_joint_velocity(self, velocities):
        msg = JointJog()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ""
        msg.joint_names = list(self.joint_names)
        msg.velocities = list(velocities)
        msg.displacements = []
        msg.duration = 1.0 / self.rate_hz
        self.joint_jog_pub.publish(msg)

    def publish_zero_for_a_moment(self):
        zero = [0.0] * len(self.joint_names)
        r = rospy.Rate(self.rate_hz)
        for _ in range(10):
            if rospy.is_shutdown():
                break
            self.publish_joint_velocity(zero)
            r.sleep()

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        last_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - last_time).to_sec()
            last_time = now

            if dt <= 1e-4:
                dt = 1.0 / self.rate_hz
            if dt > 0.1:
                dt = 0.1

            q = self.get_current_vector()
            target, active, start_time = self.get_target_state()

            if q is None:
                rospy.logwarn_throttle(2.0, "Waiting for joint_states...")
                rate.sleep()
                continue

            if not active or target is None:
                # 空闲时不持续发布，避免干扰 Twist Servo 跟踪
                rate.sleep()
                continue

            if self.max_duration > 0.0 and (now - start_time).to_sec() > self.max_duration:
                rospy.logwarn("Joint target timeout. Stop joint jog command.")
                self.publish_zero_for_a_moment()
                self.stop_active_motion()
                rate.sleep()
                continue

            err = [target[i] - q[i] for i in range(len(self.joint_names))]
            max_abs_err = max(abs(e) for e in err)

            if max_abs_err < self.tolerance:
                rospy.loginfo("Joint target reached. max_error=%.3f deg", math.degrees(max_abs_err))
                self.publish_zero_for_a_moment()
                self.stop_active_motion()
                rate.sleep()
                continue

            raw_cmd = [clamp(self.kp * e, self.max_joint_speed) for e in err]

            max_delta = self.max_joint_accel * dt
            cmd = []
            for i in range(len(raw_cmd)):
                delta = raw_cmd[i] - self.last_cmd[i]
                delta = clamp(delta, max_delta)
                cmd.append(self.last_cmd[i] + delta)

            self.last_cmd = list(cmd)

            self.publish_joint_velocity(cmd)

            rospy.loginfo_throttle(
                1.0,
                "moving | max_err=%.2f deg | joint_vel=[%s]",
                math.degrees(max_abs_err),
                " ".join(["%.3f" % c for c in cmd]),
            )

            rate.sleep()

        self.publish_zero_for_a_moment()


if __name__ == "__main__":
    node = JointTargetServoController()
    node.run()