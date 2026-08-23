#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 conda 环境运行：D455 + MediaPipe 识别人手，计算 6D 增量，通过 UDP 发给 ROS。

V2 修改：
1. 最大姿态角默认改为 ±91°。
2. 修正死区和平滑逻辑：accepted_delta 负责目标，last_delta_out 负责平滑输出，避免输出永远到不了目标。

运行示例：
conda activate mediapipe_env
python d455_conda_udp_sender_v2.py --host 127.0.0.1 --port 5005 --pos-scale 0.5 \
  --max-droll 91 --max-dpitch 91 --max-dyaw 91

按键：
c：当前人手位姿设为零位，同时通知 ROS 端记录当前机械臂末端位姿为零位
r：重置
ESC：退出

默认映射：
人手向前伸/靠近相机 -> dx 正
人手向左移动        -> dy 正
人手向上移动        -> dz 正
"""

import argparse
import json
import math
import socket
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp


WIDTH = 640
HEIGHT = 480
FPS = 30
USE_DEPTH_FILTER = True

FULL_HAND_DILATE_KERNEL = 19
PALM_DILATE_KERNEL = 23
FULL_HAND_Z_THRESHOLD = 0.22
PALM_Z_THRESHOLD = 0.12
POINT_STEP = 2

MIN_PALM_POINTS = 80
RANSAC_ITER = 120
RANSAC_DIST_TH = 0.010
RANSAC_MAX_POINTS = 1800


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def rotation_matrix_to_rpy(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)], dtype=np.float32)


def unwrap_angle_deg(last_angle, now_angle):
    if last_angle is None:
        return now_angle
    while now_angle - last_angle > 180.0:
        now_angle -= 360.0
    while now_angle - last_angle < -180.0:
        now_angle += 360.0
    return now_angle


def unwrap_rpy_deg(last_rpy, now_rpy):
    if now_rpy is None:
        return last_rpy
    if last_rpy is None:
        return now_rpy
    return np.array([
        unwrap_angle_deg(float(last_rpy[0]), float(now_rpy[0])),
        unwrap_angle_deg(float(last_rpy[1]), float(now_rpy[1])),
        unwrap_angle_deg(float(last_rpy[2]), float(now_rpy[2]))
    ], dtype=np.float32)


def clamp_vec(v, limits):
    out = np.array(v, dtype=np.float32).copy()
    for i in range(len(out)):
        out[i] = max(-limits[i], min(limits[i], out[i]))
    return out


def accept_delta_by_deadband(accepted_delta, raw_delta, pos_deadband, rot_deadband_deg):
    """
    死区判断只更新 accepted_delta，不直接卡住平滑输出 last_delta_out。

    accepted_delta:
        已经被接受的目标增量，相当于“目标值”。

    raw_delta:
        当前识别映射出来的新目标增量。

    逻辑：
        raw_delta 相对 accepted_delta 的变化超过死区，才更新 accepted_delta；
        如果没超过死区，accepted_delta 保持不变；
        last_delta_out 会继续平滑逼近 accepted_delta。

    这样可以避免：
        目标是 6°，输出平滑到 1.3° 后被死区卡住，永远到不了 6°。
    """
    if raw_delta is None:
        return accepted_delta

    if accepted_delta is None:
        return raw_delta.copy()

    new_accepted = accepted_delta.copy()

    # 位置三轴，单位 m
    for i in range(3):
        if abs(raw_delta[i] - accepted_delta[i]) >= pos_deadband:
            new_accepted[i] = raw_delta[i]

    # 姿态三轴，单位 degree
    for i in range(3, 6):
        if abs(raw_delta[i] - accepted_delta[i]) >= rot_deadband_deg:
            new_accepted[i] = raw_delta[i]

    return new_accepted


def hold_small_change(last_out, new_out, pos_deadband, rot_deadband_deg):
    if last_out is None:
        return new_out

    out = np.array(new_out, dtype=np.float32).copy()

    for i in range(3):
        if abs(out[i] - last_out[i]) < pos_deadband:
            out[i] = last_out[i]

    for i in range(3, 6):
        if abs(out[i] - last_out[i]) < rot_deadband_deg:
            out[i] = last_out[i]

    return out


def smooth_delta(last_out, target_delta, alpha):
    if target_delta is None:
        return last_out
    if last_out is None:
        return target_delta.copy()
    return (1.0 - alpha) * last_out + alpha * target_delta


def limit_delta_step(last_out, new_out, max_step):
    if last_out is None:
        return new_out

    out = np.array(new_out, dtype=np.float32).copy()
    diff = out - last_out

    for i in range(len(out)):
        diff[i] = max(-max_step[i], min(max_step[i], diff[i]))

    return last_out + diff


def get_valid_depth(depth_frame, u, v, kernel=5):
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    half = kernel // 2
    depths = []

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x = int(u + dx)
            y = int(v + dy)
            if 0 <= x < w and 0 <= y < h:
                d = depth_frame.get_distance(x, y)
                if 0.20 < d < 4.00:
                    depths.append(d)

    if len(depths) == 0:
        return None

    return float(np.median(depths))


def pixel_to_3d(depth_frame, intrinsics, u, v, kernel=5):
    depth = get_valid_depth(depth_frame, u, v, kernel=kernel)
    if depth is None:
        return None

    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(depth))
    return np.array(point, dtype=np.float32)


def project_point(intrinsics, point_3d):
    pixel = rs.rs2_project_point_to_pixel(
        intrinsics,
        [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]
    )
    return int(pixel[0]), int(pixel[1])


def draw_axes(image, intrinsics, center, R, axis_len=0.08):
    if center is None or R is None:
        return

    try:
        c = project_point(intrinsics, center)
        x_end = project_point(intrinsics, center + R[:, 0] * axis_len)
        y_end = project_point(intrinsics, center + R[:, 1] * axis_len)
        z_end = project_point(intrinsics, center + R[:, 2] * axis_len)

        cv2.line(image, c, x_end, (0, 0, 255), 3)
        cv2.line(image, c, y_end, (0, 255, 0), 3)
        cv2.line(image, c, z_end, (255, 0, 0), 3)
        cv2.putText(image, "x", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, "z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    except Exception:
        pass


def landmarks_to_pixels(hand_landmarks, width, height):
    pixels = []
    for lm in hand_landmarks.landmark:
        u = int(lm.x * width)
        v = int(lm.y * height)
        u = max(0, min(width - 1, u))
        v = max(0, min(height - 1, v))
        pixels.append((u, v))
    return pixels


def make_mask_from_ids(pixels, ids, width, height, dilate_kernel):
    pts = []
    for idx in ids:
        u, v = pixels[idx]
        pts.append([u, v])

    pts = np.array(pts, dtype=np.int32)
    if len(pts) < 3:
        return None, None

    hull = cv2.convexHull(pts)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    if dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, hull


def extract_point_cloud_from_mask(depth_frame, intrinsics, mask, z_ref=None, z_threshold=0.12, step=2):
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    points = []

    for i in range(0, len(xs), step):
        u = int(xs[i])
        v = int(ys[i])
        d = depth_frame.get_distance(u, v)

        if d <= 0.20 or d >= 4.00:
            continue

        if z_ref is not None and abs(d - z_ref) > z_threshold:
            continue

        p = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(d))
        points.append(p)

    if len(points) == 0:
        return None

    return np.array(points, dtype=np.float32)


def fit_plane_ransac(points, iterations=RANSAC_ITER, dist_th=RANSAC_DIST_TH, max_points=RANSAC_MAX_POINTS):
    if points is None or len(points) < MIN_PALM_POINTS:
        return None, None, None, 0.0, 0

    pts_all = np.asarray(points, dtype=np.float32)

    if len(pts_all) > max_points:
        idx = np.random.choice(len(pts_all), max_points, replace=False)
        pts = pts_all[idx]
    else:
        pts = pts_all

    n = len(pts)
    if n < MIN_PALM_POINTS:
        return None, None, None, 0.0, 0

    best_inliers = None
    best_count = 0

    for _ in range(iterations):
        ids = np.random.choice(n, 3, replace=False)
        p1 = pts[ids[0]]
        p2 = pts[ids[1]]
        p3 = pts[ids[2]]

        normal = np.cross(p2 - p1, p3 - p1)
        normal = normalize(normal)
        if normal is None:
            continue

        dists = np.abs((pts - p1) @ normal)
        inliers = dists < dist_th
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < MIN_PALM_POINTS:
        return None, None, None, 0.0, 0

    inlier_pts = pts[best_inliers]
    center = np.mean(inlier_pts, axis=0)
    centered = inlier_pts - center
    cov = centered.T @ centered / len(inlier_pts)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    normal = eig_vecs[:, 0]
    normal = normalize(normal)
    if normal is None:
        return None, None, None, 0.0, 0

    distances = np.abs(centered @ normal)
    plane_error = float(np.mean(distances))
    inlier_ratio = float(best_count / n)

    return center, normal, plane_error, inlier_ratio, best_count


def range_score(value, low, high, soft_margin):
    if value is None:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, 1.0 - (low - value) / soft_margin)
    return max(0.0, 1.0 - (value - high) / soft_margin)


def compute_kp_quality(P0, P5, P9, P17):
    if P0 is None or P5 is None or P9 is None or P17 is None:
        return 0.0

    palm_width = np.linalg.norm(P5 - P17)
    palm_length = np.linalg.norm(P9 - P0)
    score_width = range_score(palm_width, 0.04, 0.15, 0.06)
    score_length = range_score(palm_length, 0.04, 0.18, 0.07)

    x_hint = normalize(P5 - P17)
    y_hint = normalize(P9 - P0)

    if x_hint is None or y_hint is None:
        score_ortho = 0.0
    else:
        ortho_error = abs(float(np.dot(x_hint, y_hint)))
        score_ortho = max(0.0, 1.0 - ortho_error / 0.4)

    z_values = np.array([P0[2], P5[2], P9[2], P17[2]], dtype=np.float32)
    z_span = float(np.max(z_values) - np.min(z_values))
    score_depth = max(0.0, 1.0 - z_span / 0.25)

    quality = 0.30 * score_width + 0.30 * score_length + 0.25 * score_ortho + 0.15 * score_depth
    return float(max(0.0, min(1.0, quality)))


def compute_plane_quality(palm_pts_num, plane_error, inlier_ratio, inlier_count):
    if palm_pts_num is None or plane_error is None:
        return 0.0

    score_pts = min(1.0, float(palm_pts_num) / 1600.0)
    score_inlier_count = min(1.0, float(inlier_count) / 800.0)
    score_inlier_ratio = max(0.0, min(1.0, float(inlier_ratio) / 0.65))
    score_err = math.exp(-float(plane_error) / 0.008)

    quality = 0.25 * score_pts + 0.25 * score_inlier_count + 0.25 * score_inlier_ratio + 0.25 * score_err
    return float(max(0.0, min(1.0, quality)))


def build_plane_hand_frame(P0, P5, P9, P17, plane_normal, last_R=None):
    if plane_normal is None:
        return None

    y_hint = normalize(P9 - P0)
    x_hint = normalize(P5 - P17)
    if y_hint is None or x_hint is None:
        return None

    z_hint = normalize(np.cross(x_hint, y_hint))
    z_h = plane_normal.copy()

    if z_hint is not None and np.dot(z_h, z_hint) < 0:
        z_h = -z_h

    if last_R is not None and np.dot(z_h, last_R[:, 2]) < 0:
        z_h = -z_h

    y_h = y_hint - np.dot(y_hint, z_h) * z_h
    y_h = normalize(y_h)
    if y_h is None:
        return None

    if last_R is not None and np.dot(y_h, last_R[:, 1]) < 0:
        y_h = -y_h

    x_h = normalize(np.cross(y_h, z_h))
    if x_h is None:
        return None

    y_h = normalize(np.cross(z_h, x_h))
    if y_h is None:
        return None

    return np.column_stack((x_h, y_h, z_h))


def enforce_rotation_continuity(R_now, R_last):
    if R_now is None or R_last is None:
        return R_now

    y = R_now[:, 1].copy()
    z = R_now[:, 2].copy()

    if np.dot(z, R_last[:, 2]) < 0:
        z = -z

    if np.dot(y, R_last[:, 1]) < 0:
        y = -y

    y = y - np.dot(y, z) * z
    y = normalize(y)
    if y is None:
        return R_now

    x = normalize(np.cross(y, z))
    if x is None:
        return R_now

    y = normalize(np.cross(z, x))
    if y is None:
        return R_now

    return np.column_stack((x, y, z))


def blend_rotation_by_axes(R_last, R_now, alpha):
    if R_now is None:
        return R_last
    if R_last is None:
        return R_now

    z = normalize((1.0 - alpha) * R_last[:, 2] + alpha * R_now[:, 2])
    y_hint = normalize((1.0 - alpha) * R_last[:, 1] + alpha * R_now[:, 1])
    if z is None or y_hint is None:
        return R_now

    y = y_hint - np.dot(y_hint, z) * z
    y = normalize(y)
    if y is None:
        return R_now

    x = normalize(np.cross(y, z))
    if x is None:
        return R_now

    y = normalize(np.cross(z, x))
    if y is None:
        return R_now

    return np.column_stack((x, y, z))


def camera_position_delta_to_robot_delta(dp_cam, pos_scale):
    dx = -pos_scale * dp_cam[2]
    dy =  pos_scale * dp_cam[0]
    dz = -pos_scale * dp_cam[1]
    return np.array([dx, dy, dz], dtype=np.float32)


def get_camera_to_control_matrix():
    return np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ], dtype=np.float32)


def map_hand_rotation_to_robot_rpy(R_hand_cam0, R_hand_cam_now, last_rpy_out=None):
    A = get_camera_to_control_matrix()
    R0_control = A @ R_hand_cam0
    Rn_control = A @ R_hand_cam_now
    R_delta = R0_control.T @ Rn_control
    rpy = rotation_matrix_to_rpy(R_delta)
    return unwrap_rpy_deg(last_rpy_out, rpy)


def format_vec(v):
    if v is None:
        return "None"
    return "[{:.3f}, {:.3f}, {:.3f}]".format(v[0], v[1], v[2])


def format_delta(v):
    if v is None:
        return "None"
    return "[dx={:.4f}, dy={:.4f}, dz={:.4f}, droll={:.1f}, dpitch={:.1f}, dyaw={:.1f}]".format(
        v[0], v[1], v[2], v[3], v[4], v[5]
    )


def send_packet(sock, host, port, packet):
    data = json.dumps(packet).encode("utf-8")
    sock.sendto(data, (host, port))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--pos-scale", type=float, default=0.6)
    parser.add_argument("--pos-deadband-hand", type=float, default=0.0015)
    parser.add_argument("--rot-deadband-deg", type=float, default=1.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.35)
    parser.add_argument("--max-dx", type=float, default=0.25)
    parser.add_argument("--max-dy", type=float, default=0.25)
    parser.add_argument("--max-dz", type=float, default=0.25)
    parser.add_argument("--max-droll", type=float, default=91.0)
    parser.add_argument("--max-dpitch", type=float, default=91.0)
    parser.add_argument("--max-dyaw", type=float, default=91.0)
    parser.add_argument("--max-step-pos", type=float, default=0.008)
    parser.add_argument("--max-step-rot", type=float, default=4.0)
    parser.add_argument("--position-only", action="store_true", help="只发送 xyz，姿态增量置零")
    parser.add_argument("--min-kp-quality", type=float, default=0.20)
    parser.add_argument("--min-plane-quality", type=float, default=0.35)
    parser.add_argument("--min-palm-pts", type=int, default=120)
    parser.add_argument("--no-quality-gate", action="store_true", help="关闭质量门控，仅调试用")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pos_deadband_robot = args.pos_deadband_hand * args.pos_scale
    max_limits = np.array([args.max_dx, args.max_dy, args.max_dz, args.max_droll, args.max_dpitch, args.max_dyaw], dtype=np.float32)
    max_step = np.array([args.max_step_pos, args.max_step_pos, args.max_step_pos, args.max_step_rot, args.max_step_rot, args.max_step_rot], dtype=np.float32)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    calibrated = False
    p_hand0 = None
    R_hand0 = None

    last_R_hand = None
    last_rpy_delta = None
    last_delta_out = None
    accepted_delta = None

    last_valid_p = None
    last_valid_R = None

    frame_count = 0

    print("====================================================")
    print("Conda D455 UDP sender SERVO V3 started.")
    print("UDP target: {}:{}".format(args.host, args.port))
    print("c: set zero | r: reset | ESC: exit")
    print("position_only:", args.position_only)
    print("pos_scale:", args.pos_scale)
    print("pos_deadband_hand:", args.pos_deadband_hand)
    print("rot_deadband_deg:", args.rot_deadband_deg)
    print("smooth_alpha:", args.smooth_alpha)
    print("max angles: +/-{} deg".format(args.max_droll))
    print("====================================================")

    try:
        while True:
            frame_count += 1

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            if USE_DEPTH_FILTER:
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_frame = hole_filling.process(depth_frame)
                depth_frame = depth_frame.as_depth_frame()
                if not depth_frame:
                    continue

            color_image = np.asanyarray(color_frame.get_data())
            h, w, _ = color_image.shape
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results_hands = hands.process(rgb_image)

            status = "NO_HAND"
            p_hand = None
            R_hand = None

            kp_quality = 0.0
            plane_quality = 0.0
            palm_pts_num = 0
            full_pts_num = 0
            plane_error = None
            inlier_ratio = 0.0
            inlier_count = 0

            if results_hands.multi_hand_landmarks:
                hand_landmarks = results_hands.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                pixels = landmarks_to_pixels(hand_landmarks, w, h)

                P0 = pixel_to_3d(depth_frame, intrinsics, *pixels[0], kernel=5)
                P5 = pixel_to_3d(depth_frame, intrinsics, *pixels[5], kernel=5)
                P9 = pixel_to_3d(depth_frame, intrinsics, *pixels[9], kernel=5)
                P13 = pixel_to_3d(depth_frame, intrinsics, *pixels[13], kernel=5)
                P17 = pixel_to_3d(depth_frame, intrinsics, *pixels[17], kernel=5)

                for idx in [0, 5, 9, 13, 17]:
                    u, v = pixels[idx]
                    cv2.circle(color_image, (u, v), 6, (0, 0, 255), -1)
                    cv2.putText(color_image, str(idx), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                if P0 is not None and P5 is not None and P9 is not None and P17 is not None:
                    center_points = [P0, P5, P9, P17]
                    if P13 is not None:
                        center_points.append(P13)

                    p_hand = np.mean(np.array(center_points), axis=0)
                    kp_quality = compute_kp_quality(P0, P5, P9, P17)

                    z_values = [P0[2], P5[2], P9[2], P17[2]]
                    if P13 is not None:
                        z_values.append(P13[2])
                    z_ref = float(np.median(z_values))

                    full_mask, full_hull = make_mask_from_ids(pixels, list(range(21)), w, h, FULL_HAND_DILATE_KERNEL)
                    full_points = extract_point_cloud_from_mask(depth_frame, intrinsics, full_mask, z_ref=z_ref, z_threshold=FULL_HAND_Z_THRESHOLD, step=POINT_STEP)
                    if full_points is not None:
                        full_pts_num = len(full_points)

                    palm_mask, palm_hull = make_mask_from_ids(pixels, [0, 1, 5, 9, 13, 17], w, h, PALM_DILATE_KERNEL)
                    palm_points = extract_point_cloud_from_mask(depth_frame, intrinsics, palm_mask, z_ref=z_ref, z_threshold=PALM_Z_THRESHOLD, step=POINT_STEP)
                    if palm_points is not None:
                        palm_pts_num = len(palm_points)

                    if full_mask is not None:
                        overlay = color_image.copy()
                        overlay[full_mask > 0] = (80, 80, 255)
                        color_image = cv2.addWeighted(overlay, 0.10, color_image, 0.90, 0)

                    if palm_mask is not None:
                        overlay = color_image.copy()
                        overlay[palm_mask > 0] = (0, 255, 255)
                        color_image = cv2.addWeighted(overlay, 0.22, color_image, 0.78, 0)

                    if full_hull is not None:
                        cv2.polylines(color_image, [full_hull], True, (80, 80, 255), 1)
                    if palm_hull is not None:
                        cv2.polylines(color_image, [palm_hull], True, (0, 255, 255), 2)

                    plane_center = None
                    plane_normal = None
                    if palm_points is not None and len(palm_points) >= MIN_PALM_POINTS:
                        plane_center, plane_normal, plane_error, inlier_ratio, inlier_count = fit_plane_ransac(palm_points)

                    plane_quality = compute_plane_quality(palm_pts_num, plane_error, inlier_ratio, inlier_count)

                    if plane_normal is not None:
                        R_now = build_plane_hand_frame(P0, P5, P9, P17, plane_normal, last_R=last_R_hand)
                        if R_now is not None:
                            R_now = enforce_rotation_continuity(R_now, last_R_hand)
                            quality_for_filter = 0.70 * plane_quality + 0.30 * kp_quality
                            alpha = 0.03 + (0.22 - 0.03) * quality_for_filter
                            alpha = max(0.03, min(0.22, alpha))
                            R_hand = blend_rotation_by_axes(last_R_hand, R_now, alpha)
                            last_R_hand = R_hand
                            draw_axes(color_image, intrinsics, plane_center if plane_center is not None else p_hand, R_hand)
                            status = "TRACKING"
                        else:
                            status = "HAND_POS_ONLY"
                    else:
                        status = "HAND_POS_ONLY"

            if p_hand is not None:
                last_valid_p = p_hand
            if R_hand is not None:
                last_valid_R = R_hand

            # 速度伺服优化：位置质量和姿态质量分开判断。
            # 手掌平面姿态失败时，仍允许 xyz 继续更新；只冻结姿态。
            pos_ok = (p_hand is not None and kp_quality >= args.min_kp_quality)
            rot_ok = (
                R_hand is not None and
                status == "TRACKING" and
                plane_quality >= args.min_plane_quality and
                palm_pts_num >= args.min_palm_pts
            )

            if args.no_quality_gate:
                pos_ok = (p_hand is not None)
                rot_ok = (R_hand is not None)

            quality_ok = bool(pos_ok or rot_ok)

            raw_delta = None

            if calibrated and quality_ok:
                if accepted_delta is None:
                    raw_delta = np.zeros(6, dtype=np.float32)
                else:
                    raw_delta = accepted_delta.copy()

                if pos_ok:
                    dp_cam = p_hand - p_hand0
                    dp_robot = camera_position_delta_to_robot_delta(dp_cam, args.pos_scale)
                    raw_delta[0] = dp_robot[0]
                    raw_delta[1] = dp_robot[1]
                    raw_delta[2] = dp_robot[2]

                if args.position_only:
                    raw_delta[3] = 0.0
                    raw_delta[4] = 0.0
                    raw_delta[5] = 0.0
                elif rot_ok:
                    rpy_delta = map_hand_rotation_to_robot_rpy(
                        R_hand0,
                        R_hand,
                        last_rpy_out=last_rpy_delta
                    )
                    last_rpy_delta = rpy_delta
                    raw_delta[3] = rpy_delta[0]
                    raw_delta[4] = rpy_delta[1]
                    raw_delta[5] = rpy_delta[2]

                raw_delta = clamp_vec(raw_delta, max_limits)

                # 修正后的死区逻辑：只更新 accepted_delta，不直接卡住平滑输出。
                accepted_delta = accept_delta_by_deadband(
                    accepted_delta,
                    raw_delta,
                    pos_deadband=pos_deadband_robot,
                    rot_deadband_deg=args.rot_deadband_deg
                )

            # 即使当前帧没有新 raw_delta，只要 accepted_delta 存在，输出继续逼近它
            if calibrated and accepted_delta is not None:
                smoothed_delta = smooth_delta(last_delta_out, accepted_delta, args.smooth_alpha)
                stepped_delta = limit_delta_step(last_delta_out, smoothed_delta, max_step)
                last_delta_out = clamp_vec(stepped_delta, max_limits)

            packet = {
                "cmd": "data",
                "stamp": time.time(),
                "calibrated": calibrated,
                "quality_ok": quality_ok,
                "pos_ok": bool(pos_ok),
                "rot_ok": bool(rot_ok),
                "status": status,
                "delta": None if last_delta_out is None else [float(x) for x in last_delta_out],
                "accepted_delta": None if accepted_delta is None else [float(x) for x in accepted_delta],
                "raw_delta": None if raw_delta is None else [float(x) for x in raw_delta],
                "kp_quality": float(kp_quality),
                "plane_quality": float(plane_quality),
                "palm_pts": int(palm_pts_num)
            }
            send_packet(sock, args.host, args.port, packet)

            y0 = 30
            dy = 27
            cv2.putText(color_image, "status:{} cal:{} pos_ok:{} rot_ok:{}".format(status, calibrated, pos_ok, rot_ok), (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(color_image, "hand_xyz: {}".format(format_vec(last_valid_p)), (20, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(color_image, "out: {}".format(format_delta(last_delta_out)), (20, y0 + 2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
            cv2.putText(color_image, "accepted: {}".format(format_delta(accepted_delta)), (20, y0 + 3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
            cv2.putText(color_image, "kp={:.2f} plane={:.2f} palmPts={}".format(kp_quality, plane_quality, palm_pts_num), (20, y0 + 4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
            cv2.putText(color_image, "c: zero | r: reset | ESC: exit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("Conda D455 UDP Sender", color_image)

            if frame_count % 20 == 0:
                print("status:", status, "calibrated:", calibrated, "pos_ok:", pos_ok, "rot_ok:", rot_ok)
                print("raw     :", format_delta(raw_delta))
                print("accepted:", format_delta(accepted_delta))
                print("out     :", format_delta(last_delta_out))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if key == ord("c"):
                if last_valid_p is not None and (args.position_only or last_valid_R is not None):
                    p_hand0 = last_valid_p.copy()
                    if last_valid_R is not None:
                        R_hand0 = last_valid_R.copy()
                    else:
                        R_hand0 = np.eye(3, dtype=np.float32)
                    calibrated = True
                    last_delta_out = np.zeros(6, dtype=np.float32)
                    accepted_delta = np.zeros(6, dtype=np.float32)
                    last_rpy_delta = np.zeros(3, dtype=np.float32)

                    send_packet(sock, args.host, args.port, {
                        "cmd": "zero",
                        "stamp": time.time(),
                        "delta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    })

                    print("========== HAND ZERO SET, ZERO SENT TO ROS ==========")
                else:
                    print("Cannot set zero: no valid hand pose.")

            if key == ord("r"):
                calibrated = False
                p_hand0 = None
                R_hand0 = None
                last_delta_out = None
                accepted_delta = None
                last_rpy_delta = None

                send_packet(sock, args.host, args.port, {
                    "cmd": "reset",
                    "stamp": time.time()
                })
                print("========== RESET SENT ==========")

    finally:
        hands.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
