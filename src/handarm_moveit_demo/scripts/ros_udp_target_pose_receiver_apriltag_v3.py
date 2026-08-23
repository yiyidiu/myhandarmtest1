#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS UDP Target Pose Receiver for AprilTag V3

与 V2 相比：
1. 接受 HOLD_LAST 数据包，继续发布最后一次有效目标；
2. 不要求每次 tracking_update 都为 true；
3. 只要 quality_ok=true 且 delta_pos/delta_quat 有效，就更新 /target_ee_pose；
4. 姿态主通道：q_target = q_zero * q_delta。
"""

import json
import math
import socket
import time

import rospy
import tf
from geometry_msgs.msg import PoseStamped


def normalize_quat(q):
    q = [float(x) for x in q]
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def build_target_pose(base_frame, robot_zero_trans, robot_zero_rot, delta_pos, delta_quat):
    q_zero = normalize_quat(robot_zero_rot)
    q_delta = normalize_quat(delta_quat)

    # 局部工具坐标系相对姿态：
    # R_target = R_zero * R_delta
    q_target = tf.transformations.quaternion_multiply(q_zero, q_delta)
    q_target = normalize_quat(q_target)

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = base_frame

    msg.pose.position.x = robot_zero_trans[0] + float(delta_pos[0])
    msg.pose.position.y = robot_zero_trans[1] + float(delta_pos[1])
    msg.pose.position.z = robot_zero_trans[2] + float(delta_pos[2])

    msg.pose.orientation.x = q_target[0]
    msg.pose.orientation.y = q_target[1]
    msg.pose.orientation.z = q_target[2]
    msg.pose.orientation.w = q_target[3]

    return msg


def main():
    rospy.init_node("ros_udp_target_pose_receiver_apriltag_v3")

    base_frame = rospy.get_param("~base_frame", "base_link")
    ee_frame = rospy.get_param("~ee_frame", "tool0")
    topic = rospy.get_param("~topic", "/target_ee_pose")
    port = int(rospy.get_param("~port", 5005))
    bind_ip = rospy.get_param("~bind_ip", "127.0.0.1")
    rate_hz = float(rospy.get_param("~rate", 100.0))

    publish_repeated = bool(rospy.get_param("~publish_repeated", False))
    accept_bad_quality = bool(rospy.get_param("~accept_bad_quality", False))
    require_seq = bool(rospy.get_param("~require_seq", True))

    listener = tf.TransformListener()
    pub = rospy.Publisher(topic, PoseStamped, queue_size=1)

    rospy.loginfo("Waiting for TF %s -> %s ...", base_frame, ee_frame)
    listener.waitForTransform(base_frame, ee_frame, rospy.Time(0), rospy.Duration(10.0))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_ip, port))
    sock.setblocking(False)

    rospy.loginfo("========== ROS UDP Target Pose Receiver AprilTag V3 ==========")
    rospy.loginfo("UDP listening on %s:%d", bind_ip, port)
    rospy.loginfo("Publishing target pose to %s", topic)
    rospy.loginfo("publish_repeated=%s accept_bad_quality=%s require_seq=%s",
                  publish_repeated, accept_bad_quality, require_seq)

    robot_zero_trans = None
    robot_zero_rot = None
    last_target = None
    last_packet_time = 0.0
    last_seq_by_sender = {}

    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        newest_packet = None

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

            sender_id = newest_packet.get("sender_id", "default")
            seq = newest_packet.get("seq", None)

            if require_seq and cmd == "data":
                if seq is None:
                    rospy.logwarn_throttle(1.0, "Packet has no seq, ignored.")
                    rate.sleep()
                    continue

                seq = int(seq)
                last_seq = last_seq_by_sender.get(sender_id, None)

                if last_seq is not None and seq <= last_seq:
                    rate.sleep()
                    continue

                last_seq_by_sender[sender_id] = seq

            if cmd == "zero":
                try:
                    trans, rot = listener.lookupTransform(base_frame, ee_frame, rospy.Time(0))
                    robot_zero_trans = [float(trans[0]), float(trans[1]), float(trans[2])]
                    robot_zero_rot = normalize_quat([rot[0], rot[1], rot[2], rot[3]])

                    last_target = build_target_pose(
                        base_frame,
                        robot_zero_trans,
                        robot_zero_rot,
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    )
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
                last_seq_by_sender = {}
                rospy.loginfo("========== RECEIVER RESET ==========")

            elif cmd == "data":
                if robot_zero_trans is None or robot_zero_rot is None:
                    rospy.logwarn_throttle(1.0, "Robot zero not set. Press c in camera sender.")
                    rate.sleep()
                    continue

                calibrated = bool(newest_packet.get("calibrated", False))
                quality_ok = bool(newest_packet.get("quality_ok", False))

                if not calibrated:
                    rospy.logwarn_throttle(1.0, "Camera sender not calibrated yet. Press c in sender.")
                    rate.sleep()
                    continue

                if not (quality_ok or accept_bad_quality):
                    rospy.logwarn_throttle(
                        1.0,
                        "Camera quality false, hold target. status=%s",
                        str(newest_packet.get("status", ""))
                    )
                    rate.sleep()
                    continue

                delta_pos = newest_packet.get("delta_pos", None)
                delta_quat = newest_packet.get("delta_quat", None)

                if delta_pos is None or len(delta_pos) != 3:
                    rospy.logwarn_throttle(1.0, "Bad packet: invalid delta_pos.")
                    rate.sleep()
                    continue

                if delta_quat is None or len(delta_quat) != 4:
                    rospy.logwarn_throttle(1.0, "Bad packet: invalid delta_quat.")
                    rate.sleep()
                    continue

                delta_pos = [float(x) for x in delta_pos]
                delta_quat = [float(x) for x in delta_quat]

                last_target = build_target_pose(
                    base_frame,
                    robot_zero_trans,
                    robot_zero_rot,
                    delta_pos,
                    delta_quat
                )

                pub.publish(last_target)
                published_this_loop = True

                rospy.loginfo_throttle(
                    0.5,
                    "target update | tag=%s status=%s tracking=%s dp=[%.4f %.4f %.4f]",
                    str(newest_packet.get("tag_id", None)),
                    str(newest_packet.get("status", "")),
                    str(newest_packet.get("tracking_update", "")),
                    delta_pos[0],
                    delta_pos[1],
                    delta_pos[2]
                )

        if publish_repeated and last_target is not None and not published_this_loop:
            last_target.header.stamp = rospy.Time.now()
            pub.publish(last_target)

        if last_packet_time > 0.0 and time.time() - last_packet_time > 0.5:
            rospy.logwarn_throttle(1.0, "No fresh UDP packet from AprilTag camera sender.")

        rate.sleep()


if __name__ == "__main__":
    main()
