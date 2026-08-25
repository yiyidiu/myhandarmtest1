#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 端运行：AprilTag UDP 目标位姿接收节点。

作用：
1. 接收 conda 端 d455_apriltag_udp_sender_servo.py 发来的 UDP JSON。
2. 按收到的相对位姿增量生成 /target_ee_pose。
3. 姿态优先使用 delta_quat 四元数，不再依赖 RPY 拆角，避免欧拉角耦合。
4. 默认只在收到新的 UDP seq 时发布一次，适配 MoveIt Servo + 速度跟踪节点。

推荐配合：
- servo_pose_tracking_node_v3_tool_rotation.py
- d455_apriltag_udp_sender_servo.py

UDP 协议：
cmd == "zero":
    ROS 端记录当前 base_link -> tool0 作为机械臂零位。

cmd == "reset":
    ROS 端清除零位。

cmd == "data":
    {
      "seq": int,
      "calibrated": bool,
      "quality_ok": bool,
      "delta_pos": [dx, dy, dz],                 # 单位 m，base_link 方向下的相对平移
      "delta_quat": [qx, qy, qz, qw],            # 相对按 c 时 tool0 局部坐标系的姿态增量
      "delta": [dx,dy,dz,droll,dpitch,dyaw]      # 兼容旧版，可选
    }

注意：
- 姿态控制使用 q_target = q_zero * q_delta。
- 这表示 delta_quat 是绕按 c 时 tool0 自身坐标系的局部旋转。
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
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0]/n, q[1]/n, q[2]/n, q[3]/n]


def build_target_pose(base_frame, robot_zero_trans, robot_zero_rot,
                      delta_pos, delta_quat=None, delta_rpy_deg=None):
    q_zero = normalize_quat(robot_zero_rot)

    if delta_quat is not None and len(delta_quat) == 4:
        q_delta = normalize_quat(delta_quat)
    else:
        if delta_rpy_deg is None:
            delta_rpy_deg = [0.0, 0.0, 0.0]
        droll = math.radians(float(delta_rpy_deg[0]))
        dpitch = math.radians(float(delta_rpy_deg[1]))
        dyaw = math.radians(float(delta_rpy_deg[2]))
        q_delta = tf.transformations.quaternion_from_euler(droll, dpitch, dyaw)
        q_delta = normalize_quat(q_delta)

    # 局部工具坐标系姿态增量：q_target = q_zero * q_delta
    q_target = tf.transformations.quaternion_multiply(q_zero, q_delta)
    q_target = normalize_quat(q_target)

    target = PoseStamped()
    target.header.stamp = rospy.Time.now()
    target.header.frame_id = base_frame

    target.pose.position.x = robot_zero_trans[0] + float(delta_pos[0])
    target.pose.position.y = robot_zero_trans[1] + float(delta_pos[1])
    target.pose.position.z = robot_zero_trans[2] + float(delta_pos[2])

    target.pose.orientation.x = q_target[0]
    target.pose.orientation.y = q_target[1]
    target.pose.orientation.z = q_target[2]
    target.pose.orientation.w = q_target[3]

    return target


def main():
    rospy.init_node("ros_udp_target_pose_receiver_apriltag")

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

    rospy.loginfo("========== ROS UDP Target Pose Receiver AprilTag ==========")
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

            sender_id = newest_packet.get("sender_id", "default")
            seq = newest_packet.get("seq", None)

            if require_seq and cmd == "data":
                if seq is None:
                    rospy.logwarn_throttle(1.0, "Packet has no seq, ignored. Set _require_seq:=false to accept.")
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
                    robot_zero_trans = [trans[0], trans[1], trans[2]]
                    robot_zero_rot = normalize_quat([rot[0], rot[1], rot[2], rot[3]])

                    delta_pos = [0.0, 0.0, 0.0]
                    delta_quat = [0.0, 0.0, 0.0, 1.0]
                    last_target = build_target_pose(
                        base_frame,
                        robot_zero_trans,
                        robot_zero_rot,
                        delta_pos,
                        delta_quat=delta_quat
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
                if robot_zero_trans is not None and robot_zero_rot is not None:
                    calibrated = bool(newest_packet.get("calibrated", False))
                    quality_ok = bool(newest_packet.get("quality_ok", False))
                    pos_ok = bool(newest_packet.get("pos_ok", False))
                    rot_ok = bool(newest_packet.get("rot_ok", False))

                    if not calibrated:
                        rospy.logwarn_throttle(1.0, "Camera sender not calibrated yet. Press c in sender window.")
                    elif quality_ok or accept_bad_quality:
                        delta_pos = newest_packet.get("delta_pos", None)
                        delta_quat = newest_packet.get("delta_quat", None)

                        # 兼容旧版 delta=[dx,dy,dz,droll,dpitch,dyaw]
                        delta = newest_packet.get("delta", None)
                        delta_rpy = None
                        if delta_pos is None and delta is not None and len(delta) >= 3:
                            delta_pos = [float(delta[0]), float(delta[1]), float(delta[2])]
                        if delta is not None and len(delta) >= 6:
                            delta_rpy = [float(delta[3]), float(delta[4]), float(delta[5])]

                        if delta_pos is not None and len(delta_pos) == 3:
                            delta_pos = [float(x) for x in delta_pos]

                            if delta_quat is not None and len(delta_quat) == 4:
                                delta_quat = [float(x) for x in delta_quat]
                                last_target = build_target_pose(
                                    base_frame,
                                    robot_zero_trans,
                                    robot_zero_rot,
                                    delta_pos,
                                    delta_quat=delta_quat
                                )
                            else:
                                last_target = build_target_pose(
                                    base_frame,
                                    robot_zero_trans,
                                    robot_zero_rot,
                                    delta_pos,
                                    delta_quat=None,
                                    delta_rpy_deg=delta_rpy
                                )

                            pub.publish(last_target)
                            published_this_loop = True

                            rospy.loginfo_throttle(
                                0.5,
                                "target update | tag=%s pos_ok=%s rot_ok=%s dp=[%.4f %.4f %.4f] quat=%s",
                                str(newest_packet.get("tag_id", None)),
                                pos_ok,
                                rot_ok,
                                delta_pos[0], delta_pos[1], delta_pos[2],
                                "yes" if delta_quat is not None else "no"
                            )

        if publish_repeated and last_target is not None and not published_this_loop:
            last_target.header.stamp = rospy.Time.now()
            pub.publish(last_target)

        if last_packet_time > 0.0 and time.time() - last_packet_time > 0.5:
            rospy.logwarn_throttle(1.0, "No fresh UDP packet from AprilTag camera sender.")

        rate.sleep()


if __name__ == "__main__":
    main()
