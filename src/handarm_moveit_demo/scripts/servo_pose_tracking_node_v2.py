#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
servo_pose_tracking_node_v2.py

改进版视觉位姿跟踪节点。

功能：
1. 订阅 /target_ee_pose，模拟真实视觉输出的目标末端位姿。
2. 对目标位置和姿态进行 α-β 状态估计。
3. 估计目标线速度和角速度。
4. 使用“目标速度前馈 + 位姿误差反馈”生成 TwistStamped。
5. 支持视觉延迟预测。
6. 支持速度限幅和加速度限幅，减小突变和抖动。

输入：
    /target_ee_pose     geometry_msgs/PoseStamped

输出：
    /servo_server/delta_twist_cmds     geometry_msgs/TwistStamped
"""

import math
import threading

import rospy
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped


def norm3(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def add3(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub3(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def scale3(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def clamp_vector3(v, max_norm):
    n = norm3(v)
    if max_norm <= 0.0:
        return [0.0, 0.0, 0.0]
    if n < 1e-12 or n <= max_norm:
        return v
    s = max_norm / n
    return scale3(v, s)


def limit_delta_vector3(target, last, max_delta):
    """
    限制指令变化量，用于实现加速度限幅。
    """
    if max_delta <= 0.0:
        return target

    delta = sub3(target, last)
    delta = clamp_vector3(delta, max_delta)
    return add3(last, delta)


def low_pass_vector3(new_value, last_value, alpha):
    """
    alpha=1 表示不过滤；
    alpha 越小越平滑，但滞后越大。
    """
    if alpha >= 1.0:
        return new_value
    if alpha <= 0.0:
        return last_value

    return [
        alpha * new_value[0] + (1.0 - alpha) * last_value[0],
        alpha * new_value[1] + (1.0 - alpha) * last_value[1],
        alpha * new_value[2] + (1.0 - alpha) * last_value[2]
    ]


def normalize_quat(q):
    """
    四元数格式：[x, y, z, w]
    """
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def quat_dot(q1, q2):
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]


def shortest_quat(q_des, q_ref):
    """
    q 和 -q 表示同一个姿态。
    这里选择与参考姿态更接近的四元数表示，避免姿态跳变。
    """
    if quat_dot(q_des, q_ref) < 0.0:
        return [-q_des[0], -q_des[1], -q_des[2], -q_des[3]]
    return q_des


def rotvec_to_quat(rotvec):
    """
    旋转向量转四元数。
    rotvec = theta * axis
    """
    theta = norm3(rotvec)
    if theta < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]

    axis = scale3(rotvec, 1.0 / theta)
    half = 0.5 * theta
    s = math.sin(half)

    return normalize_quat([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        math.cos(half)
    ])


def quat_error_to_rotvec(q_des, q_cur):
    """
    计算从当前姿态 q_cur 到目标姿态 q_des 的旋转误差向量。

    q_des, q_cur 格式均为 [x, y, z, w]

    q_err = q_des * inverse(q_cur)

    输出：
        e_R = theta * axis
    """

    q_des = normalize_quat(q_des)
    q_cur = normalize_quat(q_cur)
    q_des = shortest_quat(q_des, q_cur)

    q_cur_inv = tf.transformations.quaternion_inverse(q_cur)
    q_err = tf.transformations.quaternion_multiply(q_des, q_cur_inv)
    q_err = normalize_quat(q_err)

    if q_err[3] < 0.0:
        q_err = [-q_err[0], -q_err[1], -q_err[2], -q_err[3]]

    v = [q_err[0], q_err[1], q_err[2]]
    sin_half = norm3(v)
    w = q_err[3]

    if sin_half < 1e-12:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(sin_half, w)
    axis = scale3(v, 1.0 / sin_half)

    return scale3(axis, angle)


def integrate_quat_by_omega(q, omega, dt):
    """
    根据角速度 omega 对四元数做一小步积分。

    这里认为 omega 表达在 base_frame 下，因此使用左乘：
        q_next = dq * q
    """
    if dt <= 0.0:
        return normalize_quat(q)

    rotvec = scale3(omega, dt)
    dq = rotvec_to_quat(rotvec)
    q_next = tf.transformations.quaternion_multiply(dq, q)
    return normalize_quat(q_next)


class ServoPoseTrackingNodeV2:
    def __init__(self):
        rospy.init_node("servo_pose_tracking_node_v2")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_frame = rospy.get_param("~ee_frame", "tool0")

        self.target_topic = rospy.get_param("~target_topic", "/target_ee_pose")
        self.twist_topic = rospy.get_param("~twist_topic", "/servo_server/delta_twist_cmds")

        # 反馈增益
        self.kp_pos = float(rospy.get_param("~kp_pos", 0.8))
        self.kp_rot = float(rospy.get_param("~kp_rot", 0.5))

        # 前馈增益
        self.kff_pos = float(rospy.get_param("~kff_pos", 0.8))
        self.kff_rot = float(rospy.get_param("~kff_rot", 0.5))

        # 目标状态估计 α-β 参数
        self.alpha_pos = float(rospy.get_param("~alpha_pos", 0.35))
        self.beta_pos = float(rospy.get_param("~beta_pos", 0.08))

        self.alpha_rot = float(rospy.get_param("~alpha_rot", 0.35))
        self.beta_rot = float(rospy.get_param("~beta_rot", 0.08))

        # 视觉延迟预测时间，单位 s
        self.delay_comp = float(rospy.get_param("~delay_comp", 0.05))

        # 指令限幅
        self.max_linear_speed = float(rospy.get_param("~max_linear_speed", 0.04))
        self.max_angular_speed = float(rospy.get_param("~max_angular_speed", 0.25))

        # 加速度限幅
        self.max_linear_accel = float(rospy.get_param("~max_linear_accel", 0.20))
        self.max_angular_accel = float(rospy.get_param("~max_angular_accel", 1.00))

        # 目标估计速度限幅，防止视觉跳变导致速度估计爆炸
        self.target_linear_speed_limit = float(rospy.get_param("~target_linear_speed_limit", 0.30))
        self.target_angular_speed_limit = float(rospy.get_param("~target_angular_speed_limit", 2.00))

        # 误差死区
        self.pos_deadband = float(rospy.get_param("~pos_deadband", 0.001))
        self.rot_deadband = float(rospy.get_param("~rot_deadband", 0.01))

        # 指令低通，1.0 表示不启用额外低通
        self.cmd_lpf_alpha = float(rospy.get_param("~cmd_lpf_alpha", 1.0))

        # 目标超时
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.5))

        # 如果目标突然跳得太远，重置滤波器
        self.target_jump_threshold = float(rospy.get_param("~target_jump_threshold", 0.20))
        self.rot_jump_threshold = float(rospy.get_param("~rot_jump_threshold", 1.00))

        self.rate_hz = float(rospy.get_param("~rate", 50.0))

        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=1)

        self.lock = threading.Lock()

        self.filter_initialized = False

        self.p_hat = [0.0, 0.0, 0.0]
        self.v_hat = [0.0, 0.0, 0.0]

        self.q_hat = [0.0, 0.0, 0.0, 1.0]
        self.w_hat = [0.0, 0.0, 0.0]

        self.last_filter_time = None
        self.last_receive_time = None

        self.last_v_cmd = [0.0, 0.0, 0.0]
        self.last_w_cmd = [0.0, 0.0, 0.0]

        self.sub = rospy.Subscriber(
            self.target_topic,
            PoseStamped,
            self.target_callback,
            queue_size=1
        )

        rospy.loginfo("========== Servo Pose Tracking Node V2 ==========")
        rospy.loginfo("base_frame: %s", self.base_frame)
        rospy.loginfo("ee_frame: %s", self.ee_frame)
        rospy.loginfo("target_topic: %s", self.target_topic)
        rospy.loginfo("twist_topic: %s", self.twist_topic)
        rospy.loginfo("kp_pos=%.3f kp_rot=%.3f", self.kp_pos, self.kp_rot)
        rospy.loginfo("kff_pos=%.3f kff_rot=%.3f", self.kff_pos, self.kff_rot)
        rospy.loginfo("alpha_pos=%.3f beta_pos=%.3f", self.alpha_pos, self.beta_pos)
        rospy.loginfo("alpha_rot=%.3f beta_rot=%.3f", self.alpha_rot, self.beta_rot)
        rospy.loginfo("delay_comp=%.3f s", self.delay_comp)
        rospy.loginfo("max_linear_speed=%.3f m/s max_angular_speed=%.3f rad/s",
                      self.max_linear_speed, self.max_angular_speed)
        rospy.loginfo("max_linear_accel=%.3f m/s^2 max_angular_accel=%.3f rad/s^2",
                      self.max_linear_accel, self.max_angular_accel)

        self.wait_for_tf()

    def wait_for_tf(self):
        rospy.loginfo("Waiting for TF %s -> %s ...", self.base_frame, self.ee_frame)
        self.listener.waitForTransform(
            self.base_frame,
            self.ee_frame,
            rospy.Time(0),
            rospy.Duration(10.0)
        )
        rospy.loginfo("TF is ready.")

    def target_callback(self, msg):
        try:
            if msg.header.frame_id == "":
                msg.header.frame_id = self.base_frame

            if msg.header.frame_id != self.base_frame:
                msg.header.stamp = rospy.Time(0)
                target_in_base = self.listener.transformPose(self.base_frame, msg)
            else:
                target_in_base = msg

            p_meas = [
                target_in_base.pose.position.x,
                target_in_base.pose.position.y,
                target_in_base.pose.position.z
            ]

            q_meas = normalize_quat([
                target_in_base.pose.orientation.x,
                target_in_base.pose.orientation.y,
                target_in_base.pose.orientation.z,
                target_in_base.pose.orientation.w
            ])

            now = rospy.Time.now()

            with self.lock:
                self.last_receive_time = now

                if not self.filter_initialized:
                    self.p_hat = p_meas
                    self.v_hat = [0.0, 0.0, 0.0]
                    self.q_hat = q_meas
                    self.w_hat = [0.0, 0.0, 0.0]
                    self.last_filter_time = now
                    self.filter_initialized = True
                    return

                dt = (now - self.last_filter_time).to_sec()
                if dt <= 1e-4:
                    return

                # 限制异常 dt
                if dt > 0.2:
                    dt = 0.2

                # 位置 α-β 滤波
                p_pred = add3(self.p_hat, scale3(self.v_hat, dt))
                pos_res = sub3(p_meas, p_pred)

                # 目标突变过大时重置，避免速度估计爆炸
                if norm3(pos_res) > self.target_jump_threshold:
                    rospy.logwarn_throttle(
                        1.0,
                        "Target position jump detected. Resetting target filter."
                    )
                    self.p_hat = p_meas
                    self.v_hat = [0.0, 0.0, 0.0]
                    self.q_hat = q_meas
                    self.w_hat = [0.0, 0.0, 0.0]
                    self.last_filter_time = now
                    return

                self.p_hat = add3(p_pred, scale3(pos_res, self.alpha_pos))
                self.v_hat = add3(self.v_hat, scale3(pos_res, self.beta_pos / dt))
                self.v_hat = clamp_vector3(self.v_hat, self.target_linear_speed_limit)

                # 姿态 α-β 滤波
                q_pred = integrate_quat_by_omega(self.q_hat, self.w_hat, dt)
                q_meas = shortest_quat(q_meas, q_pred)

                rot_res = quat_error_to_rotvec(q_meas, q_pred)

                if norm3(rot_res) > self.rot_jump_threshold:
                    rospy.logwarn_throttle(
                        1.0,
                        "Target orientation jump detected. Resetting orientation filter."
                    )
                    self.q_hat = q_meas
                    self.w_hat = [0.0, 0.0, 0.0]
                    self.last_filter_time = now
                    return

                dq_corr = rotvec_to_quat(scale3(rot_res, self.alpha_rot))
                self.q_hat = tf.transformations.quaternion_multiply(dq_corr, q_pred)
                self.q_hat = normalize_quat(self.q_hat)

                self.w_hat = add3(self.w_hat, scale3(rot_res, self.beta_rot / dt))
                self.w_hat = clamp_vector3(self.w_hat, self.target_angular_speed_limit)

                self.last_filter_time = now

        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                1.0,
                "Failed to process target pose: %s",
                str(exc)
            )

    def lookup_current_pose(self):
        self.listener.waitForTransform(
            self.base_frame,
            self.ee_frame,
            rospy.Time(0),
            rospy.Duration(1.0)
        )

        trans, rot = self.listener.lookupTransform(
            self.base_frame,
            self.ee_frame,
            rospy.Time(0)
        )

        p_cur = [trans[0], trans[1], trans[2]]
        q_cur = normalize_quat([rot[0], rot[1], rot[2], rot[3]])

        return p_cur, q_cur

    def get_predicted_target_state(self):
        now = rospy.Time.now()

        with self.lock:
            if not self.filter_initialized or self.last_receive_time is None:
                return None

            age = (now - self.last_receive_time).to_sec()
            if self.target_timeout > 0.0 and age > self.target_timeout:
                return None

            p_hat = list(self.p_hat)
            v_hat = list(self.v_hat)
            q_hat = list(self.q_hat)
            w_hat = list(self.w_hat)
            last_filter_time = self.last_filter_time

        dt_now = (now - last_filter_time).to_sec()
        if dt_now < 0.0:
            dt_now = 0.0
        if dt_now > 0.2:
            dt_now = 0.2

        # 预测到当前时刻
        p_now = add3(p_hat, scale3(v_hat, dt_now))
        q_now = integrate_quat_by_omega(q_hat, w_hat, dt_now)

        # 视觉延迟预测
        if self.delay_comp > 0.0:
            p_ref = add3(p_now, scale3(v_hat, self.delay_comp))
            q_ref = integrate_quat_by_omega(q_now, w_hat, self.delay_comp)
        else:
            p_ref = p_now
            q_ref = q_now

        return {
            "p_ref": p_ref,
            "q_ref": q_ref,
            "v_ff": v_hat,
            "w_ff": w_hat,
            "age": age
        }

    def publish_zero_twist(self):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        self.pub.publish(msg)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        last_loop_time = rospy.Time.now()
        last_log_time = rospy.Time(0)

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt_loop = (now - last_loop_time).to_sec()
            if dt_loop <= 1e-4:
                dt_loop = 1.0 / self.rate_hz
            if dt_loop > 0.1:
                dt_loop = 0.1
            last_loop_time = now

            target_state = self.get_predicted_target_state()

            if target_state is None:
                rospy.logwarn_throttle(
                    2.0,
                    "No valid target pose. Publishing zero twist."
                )
                self.publish_zero_twist()
                self.last_v_cmd = [0.0, 0.0, 0.0]
                self.last_w_cmd = [0.0, 0.0, 0.0]
                rate.sleep()
                continue

            try:
                p_cur, q_cur = self.lookup_current_pose()
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                rospy.logwarn_throttle(1.0, "TF lookup failed: %s", str(exc))
                self.publish_zero_twist()
                rate.sleep()
                continue

            p_ref = target_state["p_ref"]
            q_ref = target_state["q_ref"]
            v_ff = target_state["v_ff"]
            w_ff = target_state["w_ff"]
            age = target_state["age"]

            # 位置误差
            ep = sub3(p_ref, p_cur)

            if abs(ep[0]) < self.pos_deadband:
                ep[0] = 0.0
            if abs(ep[1]) < self.pos_deadband:
                ep[1] = 0.0
            if abs(ep[2]) < self.pos_deadband:
                ep[2] = 0.0

            # 姿态误差
            er = quat_error_to_rotvec(q_ref, q_cur)

            if abs(er[0]) < self.rot_deadband:
                er[0] = 0.0
            if abs(er[1]) < self.rot_deadband:
                er[1] = 0.0
            if abs(er[2]) < self.rot_deadband:
                er[2] = 0.0

            # 前馈 + 反馈
            v_cmd = add3(
                scale3(v_ff, self.kff_pos),
                scale3(ep, self.kp_pos)
            )

            w_cmd = add3(
                scale3(w_ff, self.kff_rot),
                scale3(er, self.kp_rot)
            )

            # 速度限幅
            v_cmd = clamp_vector3(v_cmd, self.max_linear_speed)
            w_cmd = clamp_vector3(w_cmd, self.max_angular_speed)

            # 加速度限幅
            v_cmd = limit_delta_vector3(
                v_cmd,
                self.last_v_cmd,
                self.max_linear_accel * dt_loop
            )

            w_cmd = limit_delta_vector3(
                w_cmd,
                self.last_w_cmd,
                self.max_angular_accel * dt_loop
            )

            # 指令低通
            v_cmd = low_pass_vector3(v_cmd, self.last_v_cmd, self.cmd_lpf_alpha)
            w_cmd = low_pass_vector3(w_cmd, self.last_w_cmd, self.cmd_lpf_alpha)

            # 再次限幅，确保安全
            v_cmd = clamp_vector3(v_cmd, self.max_linear_speed)
            w_cmd = clamp_vector3(w_cmd, self.max_angular_speed)

            self.last_v_cmd = list(v_cmd)
            self.last_w_cmd = list(w_cmd)

            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.base_frame

            msg.twist.linear.x = v_cmd[0]
            msg.twist.linear.y = v_cmd[1]
            msg.twist.linear.z = v_cmd[2]

            msg.twist.angular.x = w_cmd[0]
            msg.twist.angular.y = w_cmd[1]
            msg.twist.angular.z = w_cmd[2]

            self.pub.publish(msg)

            if (now - last_log_time).to_sec() > 1.0:
                rospy.loginfo(
                    "age=%.3f | ep=%.4f m er=%.4f rad | "
                    "vff=[%.3f %.3f %.3f] wff=[%.3f %.3f %.3f] | "
                    "v=[%.3f %.3f %.3f] w=[%.3f %.3f %.3f]",
                    age,
                    norm3(ep),
                    norm3(er),
                    v_ff[0], v_ff[1], v_ff[2],
                    w_ff[0], w_ff[1], w_ff[2],
                    v_cmd[0], v_cmd[1], v_cmd[2],
                    w_cmd[0], w_cmd[1], w_cmd[2]
                )
                last_log_time = now

            rate.sleep()

        for _ in range(10):
            self.publish_zero_twist()
            rate.sleep()


if __name__ == "__main__":
    node = ServoPoseTrackingNodeV2()
    node.run()