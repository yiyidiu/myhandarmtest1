#!/usr/bin/env python3
"""Enumerate and rank D455 stream capabilities without silently starting a fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


class CapabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoProfile:
    sensor: str
    stream: str
    format: str
    width: int
    height: int
    fps: int
    stream_index: int

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MotionProfile:
    sensor: str
    stream: str
    format: str
    fps: int
    stream_index: int

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def enumerate_device_profiles(serial: Optional[str] = None) -> Dict[str, Any]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CapabilityError("pyrealsense2 is unavailable") from exc
    devices = list(rs.context().query_devices())
    if serial:
        devices = [
            item
            for item in devices
            if item.get_info(rs.camera_info.serial_number) == serial
        ]
    if len(devices) != 1:
        raise CapabilityError(f"expected exactly one selected device, found {len(devices)}")
    device = devices[0]
    if "D455" not in device.get_info(rs.camera_info.name):
        raise CapabilityError("selected device is not a D455")
    videos: List[VideoProfile] = []
    motions: List[MotionProfile] = []
    sensor_names: List[str] = []
    for sensor in device.query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name)
        sensor_names.append(sensor_name)
        for profile in sensor.get_stream_profiles():
            stream = str(profile.stream_type())
            common = {
                "sensor": sensor_name,
                "stream": stream,
                "format": str(profile.format()),
                "fps": int(profile.fps()),
                "stream_index": int(profile.stream_index()),
            }
            if "accel" in stream.lower() or "gyro" in stream.lower():
                motions.append(MotionProfile(**common))
                continue
            try:
                video = profile.as_video_stream_profile()
                videos.append(
                    VideoProfile(
                        **common,
                        width=int(video.width()),
                        height=int(video.height()),
                    )
                )
            except RuntimeError:
                motions.append(MotionProfile(**common))
    return {
        "device": {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware_version": device.get_info(rs.camera_info.firmware_version),
            "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
        },
        "sensors": sensor_names,
        "video_profiles": [item.as_dict() for item in videos],
        "motion_profiles": [item.as_dict() for item in motions],
    }


def _candidate_score(item: Dict[str, Any], purpose: str) -> Tuple[int, ...]:
    color = item["color"]
    depth = item["depth"]
    exact_640_480 = int(
        color["width"] == depth["width"] == 640
        and color["height"] == depth["height"] == 480
    )
    matched_fps = int(color["fps"] == depth["fps"])
    fps = min(color["fps"], depth["fps"])
    if purpose == "live_algorithm":
        return matched_fps, exact_640_480, min(fps, 30), -abs(fps - 30)
    pixels = min(
        color["width"] * color["height"], depth["width"] * depth["height"]
    )
    return matched_fps, exact_640_480, min(fps, 30), pixels


def rank_rgbd_candidates(
    capability: Dict[str, Any], purpose: str
) -> List[Dict[str, Any]]:
    if purpose not in ("live_algorithm", "recording"):
        raise ValueError("purpose must be live_algorithm or recording")
    color = [
        item
        for item in capability["video_profiles"]
        if "color" in item["stream"].lower() and "rgb8" in item["format"].lower()
    ]
    depth = [
        item
        for item in capability["video_profiles"]
        if "depth" in item["stream"].lower() and "z16" in item["format"].lower()
    ]
    candidates = []
    for color_profile in color:
        for depth_profile in depth:
            if color_profile["fps"] != depth_profile["fps"]:
                continue
            candidate = {"color": color_profile, "depth": depth_profile}
            score = _candidate_score(candidate, purpose)
            candidate["score"] = list(score)
            candidate["selection_reason"] = (
                "matched RGB8/Z16 FPS; prefer enumerated 640x480 and up to 30 Hz; "
                f"ranked for {purpose}"
            )
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: tuple(item["score"]), reverse=True)
