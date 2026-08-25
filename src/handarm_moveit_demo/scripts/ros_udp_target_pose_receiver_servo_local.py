#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 端运行：专门配合 servo_pose_tracking_node_v3_tool_rotation.py 的 UDP 目标位姿接收节点。

和旧版 ros_udp_target_pose_receiver.py 的区别：
1. 旧版会 50 Hz 重复发布同一个 /target_ee_pose。
2. 这个速度伺服优化版默认只在收到 conda 端新目标时发布一次。

原因：
servo_pose_tracking_node_v2.py 内部已经 50 Hz 查询当前机械臂 TF、做 α-β 滤波、预测、前馈+反馈，并持续发布 TwistStamped。
因此 /target_ee_pose 不需要重复灌 50 Hz，否则会让目标速度估计 v_hat/w_hat 变钝。

输入：
UDP JSON from d455_conda_udp_sender_servo_v3.py

输出：
/target_ee_pose geometry_msgs/PoseStamped
"""

import json
import math
import socket
import time

import rospy
import tf
from geometry_msgs.msg import PoseStamped


def normalize_quat(q):
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0]/n, q[1]/n, q[2]/n, q[3]/n]


def build_target_pose(base_frame, robot_zero_trans, robot_zero_rot, delta):
    dx, dy, dz, droll_deg, dpitch_deg, dyaw_deg = delta

    droll = math.radians(float(droll_deg))
    dpitch = math.radians(float(dpitch_deg))
    dyaw = math.radians(float(dyaw_deg))

    q_zero = normalize_quat([
        robot_zero_rot[0],
        robot_zero_rot[1],
        robot_zero_rot[2],
        robot_zero_rot[3]
    ])

    q_delta = tf.transformations.quaternion_from_euler(droll, dpitch, dyaw)

    # 工具坐标系局部姿态增量：q_target = q_zero * q_delta
    # 含义：droll/dpitch/dyaw 绕按 c 时 tool0 自己的 x/y/z 轴转，而不是绕 base_link 的轴转。
    q_target = tf.transformations.quaternion_multiply(q_zero, q_delta)
    q_target = normalize_quat(q_target)

    target = PoseStamped()
    target.header.stamp = rospy.Time.now()
    target.header.frame_id = base_frame

    target.pose.position.x = robot_zero_trans[0] + float(dx)
    target.pose.position.y = robot_zero_trans[1] + float(dy)
    target.pose.position.z = robot_zero_trans[2] + float(dz)

    target.pose.orientation.x = q_target[0]
    target.pose.orientation.y = q_target[1]
    target.pose.orientation.z = q_target[2]
    target.pose.orientation.w = q_target[3]

    return target


def main():
    rospy.init_node("ros_udp_target_pose_receiver_servo_local")

    base_frame = rospy.get_param("~base_frame", "base_link")
    ee_frame = rospy.get_param("~ee_frame", "tool0")
    topic = rospy.get_param("~topic", "/target_ee_pose")
    port = int(rospy.get_param("~port", 5005))
    bind_ip = rospy.get_param("~bind_ip", "127.0.0.1")
    rate_hz = float(rospy.get_param("~rate", 100.0))

    # 默认 false：只在收到新 UDP 数据时发布一次，推荐给 servo_pose_tracking_node_v2.py。
    # 如果你后续不用 servo_pose_tracking_node_v2，而是某个节点必须连续收 /target_ee_pose，再改成 true。
    publish_repeated = bool(rospy.get_param("~publish_repeated", False))

    # 默认 false：质量差时不更新目标，让 servo_pose_tracking_node_v2.py 依靠 target_timeout 停止。
    accept_bad_quality = bool(rospy.get_param("~accept_bad_quality", False))

    listener = tf.TransformListener()
    pub = rospy.Publisher(topic, PoseStamped, queue_size=1)

    rospy.loginfo("Waiting for TF %s -> %s ...", base_frame, ee_frame)
    listener.waitForTransform(base_frame, ee_frame, rospy.Time(0), rospy.Duration(10.0))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_ip, port))
    sock.setblocking(False)

    rospy.loginfo("========== ROS UDP Target Pose Receiver Servo Local Rotation ==========")
    rospy.loginfo("UDP listening on %s:%d", bind_ip, port)
    rospy.loginfo("Publishing target pose to %s", topic)
    rospy.loginfo("publish_repeated=%s accept_bad_quality=%s", publish_repeated, accept_bad_quality)

    robot_zero_trans = None
    robot_zero_rot = None
    last_target = None
    last_delta = None
    last_packet_time = 0.0

    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        newest_packet = None

        # 读完 UDP 缓冲区，只处理最新包，避免旧视觉帧堆积造成延迟。
        while True:
            try:
                data, addr = sock.recvfrom(65535)
                newest_packet = json.loads(data.decode("utf-8"))
            except BlockingIOError:
                break
            except Exception as e:
                rospy.logwarn_throttle(1.0, "UDP parse error: %s", str(e))
                break

        published_this_loop = False

        if newest_packet is not None:
            cmd = newest_packet.get("cmd", "data")
            last_packet_time = time.time()

            if cmd == "zero":
                try:
                    trans, rot = listener.lookupTransform(base_frame, ee_frame, rospy.Time(0))
                    robot_zero_trans = [trans[0], trans[1], trans[2]]
                    robot_zero_rot = normalize_quat([rot[0], rot[1], rot[2], rot[3]])

                    last_delta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    last_target = build_target_pose(base_frame, robot_zero_trans, robot_zero_rot, last_delta)
                    pub.publish(last_target)
                    published_this_loop = True

                    rospy.loginfo("========== ROBOT ZERO SET ==========")
                    rospy.loginfo("robot_zero p=[%.4f %.4f %.4f]",
                                  robot_zero_trans[0], robot_zero_trans[1], robot_zero_trans[2])

                except Exception as e:
                    rospy.logwarn("Cannot set robot zero: %s", str(e))

            elif cmd == "reset":
                robot_zero_trans = None
                robot_zero_rot = None
                last_target = None
                last_delta = None
                rospy.loginfo("========== RECEIVER RESET ==========")

            elif cmd == "data":
                if robot_zero_trans is not None and robot_zero_rot is not None:
                    delta = newest_packet.get("delta", None)
                    calibrated = bool(newest_packet.get("calibrated", False))
                    quality_ok = bool(newest_packet.get("quality_ok", False))
                    pos_ok = bool(newest_packet.get("pos_ok", False))
                    rot_ok = bool(newest_packet.get("rot_ok", False))

                    if calibrated and delta is not None and len(delta) == 6 and (quality_ok or accept_bad_quality):
                        last_delta = [float(x) for x in delta]
                        last_target = build_target_pose(base_frame, robot_zero_trans, robot_zero_rot, last_delta)
                        pub.publish(last_target)
                        published_this_loop = True

                        rospy.loginfo_throttle(
                            0.5,
                            "target update | pos_ok=%s rot_ok=%s delta=[%.4f %.4f %.4f %.1f %.1f %.1f]",
                            pos_ok, rot_ok,
                            last_delta[0], last_delta[1], last_delta[2],
                            last_delta[3], last_delta[4], last_delta[5]
                        )

        if publish_repeated and last_target is not None and not published_this_loop:
            last_target.header.stamp = rospy.Time.now()
            pub.publish(last_target)

        if last_packet_time > 0.0 and time.time() - last_packet_time > 0.5:
            rospy.logwarn_throttle(1.0, "No fresh UDP packet from camera sender.")

        rate.sleep()


if __name__ == "__main__":
    main()
