#!/usr/bin/env python3
"""Measure the stable visible D455 hand envelope without starting the robot.

Run this receiver before the camera process. Hold the intended C-zero pose for
the neutral interval, then deliberately visit left/right/up/down/near/far and
rotate about all three wrist axes in both directions while keeping the complete
hand visible. The generated YAML can be passed directly to
``live_human_ground_gazebo_teleop.launch``.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import yaml


def project_to_so3(value):
    rotation = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation is non-finite")
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def so3_log(rotation):
    value = project_to_so3(rotation)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-8:
        return 0.5 * np.array([
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ])
    if math.pi - angle < 1.0e-5:
        eigenvalues, eigenvectors = np.linalg.eig(value)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        axis /= max(np.linalg.norm(axis), 1.0e-12)
        return angle * axis
    scale = angle / (2.0 * math.sin(angle))
    return scale * np.array([
        value[2, 1] - value[1, 2],
        value[0, 2] - value[2, 0],
        value[1, 0] - value[0, 1],
    ])


def mean_rotation(rotations):
    average = np.mean(np.asarray(rotations, dtype=np.float64), axis=0)
    return project_to_so3(average)


def directional_extents(values, quantile, minimum_samples, label):
    samples = np.asarray(values, dtype=np.float64)
    negative = []
    positive = []
    for axis in range(3):
        below = -samples[samples[:, axis] < 0.0, axis]
        above = samples[samples[:, axis] > 0.0, axis]
        if len(below) < minimum_samples or len(above) < minimum_samples:
            raise RuntimeError(
                "{} axis {} was not explored in both directions "
                "(negative={}, positive={})".format(
                    label, axis, len(below), len(above)))
        negative.append(float(np.quantile(below, quantile)))
        positive.append(float(np.quantile(above, quantile)))
    return negative, positive


def perspective_decoupled_translation_delta(positions, neutral_position,
                                             minimum_depth_m=0.12):
    """Express image-plane motion on the C-zero depth plane.

    Raw D455 metric coordinates obey X=(u-cx)Z/fx and Y=(v-cy)Z/fy.  Using
    ``position-neutral`` therefore creates a false lateral displacement when
    the hand only changes depth near an image edge.  Keep the legacy metric
    delta for rollback, but also calibrate the control coordinates consumed by
    ``camera_ground_axis_decoupled``.
    """
    samples = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    zero = np.asarray(neutral_position, dtype=np.float64).reshape(3)
    if (not np.all(np.isfinite(samples)) or not np.all(np.isfinite(zero)) or
            zero[2] < minimum_depth_m or
            np.any(samples[:, 2] < minimum_depth_m)):
        raise ValueError("perspective decoupling requires valid positive depth")
    zero_ray = zero[:2] / zero[2]
    result = np.empty_like(samples)
    result[:, :2] = zero[2] * (
        samples[:, :2] / samples[:, 2:3] - zero_ray)
    result[:, 2] = samples[:, 2] - zero[2]
    return result


def packet_pose(packet, minimum_confidence):
    if packet.get("schema") != "handarm_hamer_pose_v1" or not packet.get(
            "valid", False):
        raise ValueError("not a valid handarm_hamer_pose_v1 packet")
    position = np.asarray(packet["wrist_position_m"], dtype=np.float64)
    rotation = project_to_so3(packet["palm_rotation_row_major"])
    confidence = np.asarray(packet.get("confidence", [0.0] * 6), dtype=np.float64)
    if (position.shape != (3,) or confidence.shape != (6,) or
            not np.all(np.isfinite(position)) or
            float(np.min(confidence[:3])) < minimum_confidence):
        raise ValueError("metric wrist confidence is below calibration threshold")
    return position, rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--neutral-s", type=float, default=3.0)
    parser.add_argument("--explore-s", type=float, default=20.0)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--minimum-confidence", type=float, default=0.45)
    parser.add_argument("--minimum-direction-samples", type=int, default=5)
    parser.add_argument("--minimum-translation-extent-m", type=float, default=0.03)
    parser.add_argument("--minimum-orientation-extent-deg", type=float,
                        default=10.0)
    parser.add_argument(
        "--output", default="/tmp/handarm_camera_workspace_calibration.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.neutral_s <= 0.0 or args.explore_s <= 0.0 or
            not 0.5 < args.quantile < 1.0 or
            not 0.0 <= args.minimum_confidence <= 1.0 or
            args.minimum_direction_samples <= 0 or
            args.minimum_translation_extent_m <= 0.0 or
            args.minimum_orientation_extent_deg <= 0.0):
        raise SystemExit("invalid calibration arguments")

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind((args.bind_host, args.port))
    receiver.settimeout(0.25)
    print("Camera workspace calibration listening on udp://{}:{}".format(
        args.bind_host, args.port), flush=True)
    print("Do NOT start Gazebo. Hold the neutral C pose until prompted.", flush=True)

    neutral_positions = []
    neutral_rotations = []
    exploration_positions = []
    exploration_rotations = []
    started = None
    last_sequence = None
    last_session = None
    last_report_second = -1
    rejected = 0
    total_duration = args.neutral_s + args.explore_s

    try:
        while True:
            try:
                payload, _ = receiver.recvfrom(65535)
                packet = json.loads(payload.decode("utf-8"))
                session = str(packet.get("session_id", ""))
                sequence = int(packet.get("sequence", -1))
                if session == last_session and last_sequence is not None and sequence <= last_sequence:
                    continue
                position, rotation = packet_pose(packet, args.minimum_confidence)
                last_session, last_sequence = session, sequence
            except socket.timeout:
                continue
            except Exception:
                rejected += 1
                continue

            now = time.monotonic()
            if started is None:
                started = now
            elapsed = now - started
            if elapsed < args.neutral_s:
                neutral_positions.append(position)
                neutral_rotations.append(rotation)
                phase = "HOLD NEUTRAL"
                remaining = args.neutral_s - elapsed
            elif elapsed < total_duration:
                exploration_positions.append(position)
                exploration_rotations.append(rotation)
                phase = "EXPLORE POSITION + ROTATION +/-X +/-Y +/-Z"
                remaining = total_duration - elapsed
            else:
                break
            report_second = int(elapsed)
            if report_second != last_report_second:
                print("{}: {:.1f}s remaining (valid neutral={}, explore={})".format(
                    phase, remaining, len(neutral_positions),
                    len(exploration_positions)), flush=True)
                last_report_second = report_second
    finally:
        receiver.close()

    if len(neutral_positions) < 10 or len(exploration_positions) < 30:
        raise SystemExit(
            "insufficient valid samples: neutral={}, exploration={}".format(
                len(neutral_positions), len(exploration_positions)))
    neutral_position = np.median(np.asarray(neutral_positions), axis=0)
    neutral_rotation = mean_rotation(neutral_rotations)
    translation_delta = np.asarray(exploration_positions) - neutral_position
    decoupled_translation_delta = perspective_decoupled_translation_delta(
        exploration_positions, neutral_position)
    rotation_delta = np.asarray([
        so3_log(neutral_rotation.T @ rotation)
        for rotation in exploration_rotations
    ])
    negative_m, positive_m = directional_extents(
        translation_delta, args.quantile,
        args.minimum_direction_samples, "translation")
    decoupled_negative_m, decoupled_positive_m = directional_extents(
        decoupled_translation_delta, args.quantile,
        args.minimum_direction_samples, "perspective-decoupled translation")
    if min(negative_m + positive_m) < args.minimum_translation_extent_m:
        raise SystemExit(
            "one or more camera directions were explored less than {:.3f} m: "
            "negative={}, positive={}".format(
                args.minimum_translation_extent_m, negative_m, positive_m))
    if min(decoupled_negative_m + decoupled_positive_m) < (
            args.minimum_translation_extent_m):
        raise SystemExit(
            "one or more perspective-decoupled camera directions were "
            "explored less than {:.3f} m: negative={}, positive={}".format(
                args.minimum_translation_extent_m,
                decoupled_negative_m, decoupled_positive_m))
    negative_rotation, positive_rotation = directional_extents(
        np.degrees(rotation_delta), args.quantile,
        args.minimum_direction_samples, "orientation")
    if min(negative_rotation + positive_rotation) < (
            args.minimum_orientation_extent_deg):
        raise SystemExit(
            "one or more wrist directions were explored less than "
            "{:.1f} deg: negative={}, positive={}".format(
                args.minimum_orientation_extent_deg,
                negative_rotation, positive_rotation))

    result = {
        "schema_version": 1,
        "status": "MEASURED_CAMERA_WORKSPACE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frame_id": "camera_color_optical_frame",
        "units": "m",
        "human_workspace": {
            # Kept for camera_ground_workspace rollback compatibility.
            "negative_extent_m": negative_m,
            "positive_extent_m": positive_m,
            # Used by camera_ground_axis_decoupled.  X/Y are image rays
            # expressed on the C-zero depth plane; Z is independent depth.
            "perspective_decoupled_negative_extent_m": decoupled_negative_m,
            "perspective_decoupled_positive_extent_m": decoupled_positive_m,
            "perspective_decoupling_mode": (
                "C_ZERO_REFERENCE_PLANE_PLUS_INDEPENDENT_DEPTH"),
            "neutral_position_m": neutral_position.tolist(),
            "sample_count": len(exploration_positions),
            "neutral_sample_count": len(neutral_positions),
            "quantile": args.quantile,
            "rejected_packet_count": rejected,
        },
        "human_orientation": {
            "negative_extent_deg": negative_rotation,
            "positive_extent_deg": positive_rotation,
            "note": "ACTIVE_FOR_NORMALIZED_6D_CAMERA_POSE_MAPPING",
        },
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(result, stream, sort_keys=False, allow_unicode=True)
    print("Calibration written: {}".format(destination), flush=True)
    print("negative_extent_m={}".format(negative_m), flush=True)
    print("positive_extent_m={}".format(positive_m), flush=True)
    print("perspective_decoupled_negative_extent_m={}".format(
        decoupled_negative_m), flush=True)
    print("perspective_decoupled_positive_extent_m={}".format(
        decoupled_positive_m), flush=True)


if __name__ == "__main__":
    main()
