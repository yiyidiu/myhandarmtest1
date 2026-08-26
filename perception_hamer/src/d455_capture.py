#!/usr/bin/env python3
"""Strict, aligned Intel RealSense D435i/D455 RGB-D capture.

The public frame carries both device and host timestamps, the calibrated color
intrinsics used by the aligned depth image, the native depth scale, and USB
link metadata.  It deliberately contains no MediaPipe pose computation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .realsense_capability import (
    match_supported_device_model,
    normalize_device_models,
)


class D455CaptureError(RuntimeError):
    """Raised when the requested calibrated RealSense stream cannot be trusted."""


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise D455CaptureError(f"{name} is not finite")
    return number


def intrinsics_to_dict(intrinsics: Any) -> Dict[str, Any]:
    """Convert librealsense intrinsics into JSON-safe calibrated metadata."""

    values = {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": _finite(intrinsics.fx, "fx"),
        "fy": _finite(intrinsics.fy, "fy"),
        "ppx": _finite(intrinsics.ppx, "ppx"),
        "ppy": _finite(intrinsics.ppy, "ppy"),
        "distortion_model": str(intrinsics.model),
        "coeffs": [_finite(value, "distortion coefficient") for value in intrinsics.coeffs],
    }
    if values["width"] <= 0 or values["height"] <= 0:
        raise D455CaptureError("intrinsics have invalid dimensions")
    if values["fx"] <= 0.0 or values["fy"] <= 0.0:
        raise D455CaptureError("intrinsics have invalid focal length")
    return values


def extrinsics_to_dict(extrinsics: Any) -> Dict[str, Any]:
    """Convert librealsense column-major rotation storage to a row-major JSON SE(3)."""

    rotation = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(
        3, 3, order="F"
    )
    translation = np.asarray(extrinsics.translation, dtype=np.float64).reshape(-1)
    if (
        translation.shape != (3,)
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5)
    ):
        raise D455CaptureError("depth-to-color extrinsics are not a finite SE(3)")
    return {
        "rotation_row_major": rotation.reshape(-1, order="C").tolist(),
        "translation_m": translation.tolist(),
        "det_rotation": float(np.linalg.det(rotation)),
        "source_storage": "librealsense_column_major",
        "convention": "p_color = R_depth_to_color @ p_depth + t_depth_to_color",
    }


def _validate_intrinsics_dimensions(
    intrinsics: Dict[str, Any], shape: tuple, name: str
) -> None:
    required = {"width", "height", "fx", "fy", "ppx", "ppy"}
    if not required.issubset(intrinsics):
        raise D455CaptureError(f"{name} intrinsics are incomplete")
    expected = (int(intrinsics["height"]), int(intrinsics["width"]))
    if tuple(shape) != expected:
        raise D455CaptureError(
            f"{name} shape {tuple(shape)} does not match intrinsics {expected}"
        )
    for field in ("fx", "fy", "ppx", "ppy"):
        _finite(intrinsics[field], f"{name} {field}")
    if float(intrinsics["fx"]) <= 0.0 or float(intrinsics["fy"]) <= 0.0:
        raise D455CaptureError(f"{name} focal length must be positive")


def _validate_extrinsics_dict(extrinsics: Dict[str, Any]) -> None:
    try:
        rotation = np.asarray(extrinsics["rotation_row_major"], dtype=np.float64).reshape(
            3, 3
        )
        translation = np.asarray(extrinsics["translation_m"], dtype=np.float64).reshape(
            3
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise D455CaptureError("depth-to-color extrinsics schema is invalid") from exc
    if (
        not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5)
    ):
        raise D455CaptureError("depth-to-color extrinsics are not a finite SE(3)")


@dataclass(frozen=True)
class D455Frame:
    """One color frame and depth frame aligned into the color pixel grid."""

    rgb: np.ndarray
    raw_depth_raw: np.ndarray
    aligned_depth_raw: np.ndarray
    depth_scale_m_per_unit: float
    raw_depth_intrinsics: Dict[str, Any]
    color_intrinsics: Dict[str, Any]
    depth_to_color_extrinsics: Dict[str, Any]
    aligned_to_color_intrinsics_differences: Dict[str, float]
    color_frame_number: int
    depth_frame_number: int
    raw_color_frame_number: int
    raw_depth_frame_number: int
    color_timestamp_ms: float
    depth_timestamp_ms: float
    raw_color_timestamp_ms: float
    raw_depth_timestamp_ms: float
    color_timestamp_domain: str
    depth_timestamp_domain: str
    raw_color_timestamp_domain: str
    raw_depth_timestamp_domain: str
    host_monotonic_ns_before_wait: int
    host_monotonic_ns_frameset_received: int
    host_monotonic_ns_alignment_completed: int
    host_wall_time_ns_alignment_completed: int
    device_serial: str
    firmware_version: str
    usb_type_descriptor: str

    def __post_init__(self) -> None:
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise D455CaptureError("rgb must be uint8 HxWx3")
        if (
            self.raw_depth_raw.dtype != np.uint16
            or self.raw_depth_raw.ndim != 2
            or self.aligned_depth_raw.dtype != np.uint16
            or self.aligned_depth_raw.ndim != 2
        ):
            raise D455CaptureError("raw and aligned depth must be uint16 HxW")
        if self.rgb.shape[:2] != self.aligned_depth_raw.shape:
            raise D455CaptureError("aligned depth and RGB dimensions differ")
        _validate_intrinsics_dimensions(
            self.raw_depth_intrinsics, self.raw_depth_raw.shape, "raw depth"
        )
        _validate_intrinsics_dimensions(
            self.color_intrinsics, self.rgb.shape[:2], "color/aligned depth"
        )
        _validate_extrinsics_dict(self.depth_to_color_extrinsics)
        if set(self.aligned_to_color_intrinsics_differences) != {
            "width",
            "height",
            "fx",
            "fy",
            "ppx",
            "ppy",
        } or not all(
            math.isfinite(float(value))
            for value in self.aligned_to_color_intrinsics_differences.values()
        ):
            raise D455CaptureError("aligned/color intrinsics differences are invalid")
        if (
            not self.rgb.flags.c_contiguous
            or not self.raw_depth_raw.flags.c_contiguous
            or not self.aligned_depth_raw.flags.c_contiguous
            or not self.rgb.flags.owndata
            or not self.raw_depth_raw.flags.owndata
            or not self.aligned_depth_raw.flags.owndata
        ):
            raise D455CaptureError("frame arrays must be contiguous owned snapshots")
        _finite(self.depth_scale_m_per_unit, "depth scale")
        if self.depth_scale_m_per_unit <= 0.0:
            raise D455CaptureError("depth scale must be positive")
        for name in (
            "color_timestamp_ms",
            "depth_timestamp_ms",
            "raw_color_timestamp_ms",
            "raw_depth_timestamp_ms",
        ):
            _finite(getattr(self, name), name)
        if self.color_frame_number < 0 or self.depth_frame_number < 0:
            raise D455CaptureError("frame numbers must be nonnegative")
        if (
            self.raw_color_frame_number != self.color_frame_number
            or self.raw_depth_frame_number != self.depth_frame_number
        ):
            raise D455CaptureError("raw and aligned frames do not share frame numbers")
        if (
            not math.isclose(
                self.raw_color_timestamp_ms,
                self.color_timestamp_ms,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                self.raw_depth_timestamp_ms,
                self.depth_timestamp_ms,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or self.raw_color_timestamp_domain != self.color_timestamp_domain
            or self.raw_depth_timestamp_domain != self.depth_timestamp_domain
        ):
            raise D455CaptureError("raw and aligned frames do not share timestamp identity")
        if (
            self.host_monotonic_ns_frameset_received
            < self.host_monotonic_ns_before_wait
            or self.host_monotonic_ns_alignment_completed
            < self.host_monotonic_ns_frameset_received
        ):
            raise D455CaptureError("host monotonic timestamps moved backwards")
        if not all(
            (
                self.color_timestamp_domain,
                self.depth_timestamp_domain,
                self.raw_color_timestamp_domain,
                self.raw_depth_timestamp_domain,
            )
        ):
            raise D455CaptureError("timestamp domains must be nonempty")

    @property
    def aligned_depth_m(self) -> np.ndarray:
        return self.aligned_depth_raw.astype(np.float32) * self.depth_scale_m_per_unit

    @property
    def raw_depth_m(self) -> np.ndarray:
        return self.raw_depth_raw.astype(np.float32) * self.depth_scale_m_per_unit

    @property
    def device_timestamp_skew_ms(self) -> float:
        return abs(self.depth_timestamp_ms - self.color_timestamp_ms)

    @property
    def valid_depth_fraction(self) -> float:
        return float(np.count_nonzero(self.aligned_depth_raw)) / float(
            self.aligned_depth_raw.size
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "color_frame_number": self.color_frame_number,
            "depth_frame_number": self.depth_frame_number,
            "raw_color_frame_number": self.raw_color_frame_number,
            "raw_depth_frame_number": self.raw_depth_frame_number,
            "color_timestamp_ms": self.color_timestamp_ms,
            "depth_timestamp_ms": self.depth_timestamp_ms,
            "raw_color_timestamp_ms": self.raw_color_timestamp_ms,
            "raw_depth_timestamp_ms": self.raw_depth_timestamp_ms,
            "color_timestamp_domain": self.color_timestamp_domain,
            "depth_timestamp_domain": self.depth_timestamp_domain,
            "raw_color_timestamp_domain": self.raw_color_timestamp_domain,
            "raw_depth_timestamp_domain": self.raw_depth_timestamp_domain,
            "device_timestamp_skew_ms": self.device_timestamp_skew_ms,
            "host_monotonic_ns_before_wait": self.host_monotonic_ns_before_wait,
            "host_monotonic_ns_frameset_received": (
                self.host_monotonic_ns_frameset_received
            ),
            "host_monotonic_ns_alignment_completed": (
                self.host_monotonic_ns_alignment_completed
            ),
            "host_wall_time_ns_alignment_completed": (
                self.host_wall_time_ns_alignment_completed
            ),
            "depth_scale_m_per_unit": self.depth_scale_m_per_unit,
            "raw_depth_intrinsics": dict(self.raw_depth_intrinsics),
            "color_intrinsics": dict(self.color_intrinsics),
            "depth_to_color_extrinsics": dict(self.depth_to_color_extrinsics),
            "aligned_to_color_intrinsics_differences": dict(
                self.aligned_to_color_intrinsics_differences
            ),
            "valid_depth_fraction": self.valid_depth_fraction,
            "device_serial": self.device_serial,
            "firmware_version": self.firmware_version,
            "usb_type_descriptor": self.usb_type_descriptor,
            "usb_superspeed": self.usb_type_descriptor.startswith("3"),
            "depth_alignment_target": "color",
            "rgb_encoding": "rgb8",
            "raw_depth_encoding": "z16_raw_device_units_depth_pixel_grid",
            "aligned_depth_encoding": "z16_raw_device_units_color_pixel_grid",
        }


class D455Capture:
    """Backward-compatible D455 name for calibrated D435i/D455 capture."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
        timeout_ms: int = 3000,
        require_superspeed: bool = False,
        device_models: Optional[Iterable[str]] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = str(serial) if serial else None
        self.timeout_ms = int(timeout_ms)
        self.require_superspeed = bool(require_superspeed)
        self.device_models = normalize_device_models(device_models)
        if min(self.width, self.height, self.fps, self.timeout_ms) <= 0:
            raise ValueError("capture dimensions, fps, and timeout must be positive")
        self._rs: Any = None
        self._pipeline: Any = None
        self._align: Any = None
        self._profile: Any = None
        self._metadata: Dict[str, str] = {}
        self._session_metadata: Dict[str, Any] = {}
        self._depth_scale = 0.0

    @property
    def running(self) -> bool:
        return self._profile is not None

    @property
    def device_metadata(self) -> Dict[str, Any]:
        if not self.running:
            raise D455CaptureError("capture is not running")
        return {
            **self._metadata,
            **self._session_metadata,
            "depth_scale_m_per_unit": self._depth_scale,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "depth_alignment_target": "color",
            "usb_superspeed": self._metadata["usb_type_descriptor"].startswith("3"),
        }

    def start(self) -> "D455Capture":
        if self.running:
            return self
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise D455CaptureError("pyrealsense2 is unavailable") from exc
        context = rs.context()
        devices = list(context.query_devices())
        if self.serial:
            devices = [
                device
                for device in devices
                if device.get_info(rs.camera_info.serial_number) == self.serial
            ]
        if len(devices) != 1:
            raise D455CaptureError(
                f"expected exactly one selected RealSense device, found {len(devices)}"
            )
        selected = devices[0]
        name = selected.get_info(rs.camera_info.name)
        device_model = match_supported_device_model(name, self.device_models)
        if device_model is None:
            raise D455CaptureError(
                "selected RealSense device is not supported: "
                f"{name}; allowed={','.join(self.device_models)}"
            )
        usb = selected.get_info(rs.camera_info.usb_type_descriptor)
        if self.require_superspeed and not usb.startswith("3"):
            raise D455CaptureError(
                f"{device_model} is not on SuperSpeed USB: descriptor={usb}"
            )

        pipeline = rs.pipeline(context)
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        try:
            profile = pipeline.start(config)
            device = profile.get_device()
            depth_scale = float(device.first_depth_sensor().get_depth_scale())
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise D455CaptureError(f"failed to start aligned RGB-D profile: {exc}") from exc
        self._rs = rs
        self._pipeline = pipeline
        self._profile = profile
        try:
            self._align = rs.align(rs.stream.color)
            self._depth_scale = _finite(depth_scale, "depth scale")
            self._metadata = {
                "device_name": device.get_info(rs.camera_info.name),
                "device_model": device_model,
                "device_serial": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(
                    rs.camera_info.usb_type_descriptor
                ),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "product_id": device.get_info(rs.camera_info.product_id),
                "product_line": device.get_info(rs.camera_info.product_line),
            }
            depth_profile = profile.get_stream(
                rs.stream.depth
            ).as_video_stream_profile()
            color_profile = profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            extrinsics = depth_profile.get_extrinsics_to(color_profile)
            extrinsics_metadata = extrinsics_to_dict(extrinsics)
            sensor_options: Dict[str, Any] = {}
            for sensor in device.query_sensors():
                sensor_name = sensor.get_info(rs.camera_info.name)
                option_values: Dict[str, Any] = {}
                for option in sensor.get_supported_options():
                    try:
                        option_values[str(option)] = {
                            "value": float(sensor.get_option(option)),
                            "read_only": bool(sensor.is_option_read_only(option)),
                        }
                    except Exception as exc:
                        option_values[str(option)] = {"error": type(exc).__name__}
                sensor_options[sensor_name] = option_values
            try:
                from importlib.metadata import version

                sdk_version = version("pyrealsense2")
            except Exception:
                sdk_version = "unknown"
            self._session_metadata = {
                "pyrealsense2_version": sdk_version,
                "active_depth_profile": {
                    "stream": str(depth_profile.stream_type()),
                    "format": str(depth_profile.format()),
                    "width": depth_profile.width(),
                    "height": depth_profile.height(),
                    "fps": depth_profile.fps(),
                },
                "active_color_profile": {
                    "stream": str(color_profile.stream_type()),
                    "format": str(color_profile.format()),
                    "width": color_profile.width(),
                    "height": color_profile.height(),
                    "fps": color_profile.fps(),
                },
                "raw_depth_intrinsics": intrinsics_to_dict(depth_profile.intrinsics),
                "color_intrinsics": intrinsics_to_dict(color_profile.intrinsics),
                "depth_to_color_extrinsics": extrinsics_metadata,
                "sensor_options": sensor_options,
            }
        except Exception as exc:
            self.stop()
            if isinstance(exc, D455CaptureError):
                raise
            raise D455CaptureError(
                f"failed to initialize RealSense session metadata: {exc}"
            ) from exc
        return self

    def stop(self) -> None:
        pipeline = self._pipeline
        self._profile = None
        self._pipeline = None
        self._align = None
        self._session_metadata = {}
        if pipeline is not None:
            pipeline.stop()

    def __enter__(self) -> "D455Capture":
        return self.start()

    def __exit__(self, *_args: Any) -> None:
        self.stop()

    def wait_for_frame(self) -> D455Frame:
        if not self.running:
            raise D455CaptureError("capture is not running")
        host_before = time.monotonic_ns()
        try:
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
            host_frameset_received = time.monotonic_ns()
            raw_depth = frames.get_depth_frame()
            raw_color = frames.get_color_frame()
            aligned = self._align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
        except Exception as exc:
            raise D455CaptureError(f"frame wait/alignment failed: {exc}") from exc
        host_alignment_completed = time.monotonic_ns()
        wall_time_alignment_completed = time.time_ns()
        if not raw_depth or not raw_color or not depth or not color:
            raise D455CaptureError("frameset is missing raw or aligned color/depth")
        # Copy immediately: librealsense frame buffers can be reused after return.
        rgb = np.ascontiguousarray(np.asanyarray(color.get_data()).copy())
        raw_depth_array = np.ascontiguousarray(
            np.asanyarray(raw_depth.get_data()).copy()
        )
        depth_raw = np.ascontiguousarray(np.asanyarray(depth.get_data()).copy())
        intrinsics = intrinsics_to_dict(
            depth.profile.as_video_stream_profile().intrinsics
        )
        color_intrinsics = intrinsics_to_dict(
            color.profile.as_video_stream_profile().intrinsics
        )
        calibration_fields = ("width", "height", "fx", "fy", "ppx", "ppy")
        calibration_differences = {
            key: float(intrinsics[key]) - float(color_intrinsics[key])
            for key in calibration_fields
        }
        dimensions_match = all(
            int(intrinsics[key]) == int(color_intrinsics[key])
            for key in ("width", "height")
        )
        focal_match = all(
            abs(calibration_differences[key])
            <= max(0.5, 0.001 * abs(float(color_intrinsics[key])))
            for key in ("fx", "fy")
        )
        principal_point_match = all(
            abs(calibration_differences[key]) <= 0.5 for key in ("ppx", "ppy")
        )
        if not (dimensions_match and focal_match and principal_point_match):
            raise D455CaptureError(
                "aligned depth intrinsics do not match color intrinsics: "
                + repr(calibration_differences)
            )
        if (
            intrinsics["distortion_model"] != color_intrinsics["distortion_model"]
            or not np.allclose(
                intrinsics["coeffs"], color_intrinsics["coeffs"], rtol=0.0, atol=1e-9
            )
        ):
            raise D455CaptureError(
                "aligned depth distortion metadata does not match color intrinsics"
            )
        return D455Frame(
            rgb=rgb,
            raw_depth_raw=raw_depth_array,
            aligned_depth_raw=depth_raw,
            depth_scale_m_per_unit=self._depth_scale,
            raw_depth_intrinsics=dict(self._session_metadata["raw_depth_intrinsics"]),
            color_intrinsics=color_intrinsics,
            depth_to_color_extrinsics=dict(
                self._session_metadata["depth_to_color_extrinsics"]
            ),
            aligned_to_color_intrinsics_differences=calibration_differences,
            color_frame_number=int(color.get_frame_number()),
            depth_frame_number=int(depth.get_frame_number()),
            raw_color_frame_number=int(raw_color.get_frame_number()),
            raw_depth_frame_number=int(raw_depth.get_frame_number()),
            color_timestamp_ms=_finite(color.get_timestamp(), "color timestamp"),
            depth_timestamp_ms=_finite(depth.get_timestamp(), "depth timestamp"),
            raw_color_timestamp_ms=_finite(
                raw_color.get_timestamp(), "raw color timestamp"
            ),
            raw_depth_timestamp_ms=_finite(
                raw_depth.get_timestamp(), "raw depth timestamp"
            ),
            color_timestamp_domain=str(color.get_frame_timestamp_domain()),
            depth_timestamp_domain=str(depth.get_frame_timestamp_domain()),
            raw_color_timestamp_domain=str(raw_color.get_frame_timestamp_domain()),
            raw_depth_timestamp_domain=str(raw_depth.get_frame_timestamp_domain()),
            host_monotonic_ns_before_wait=host_before,
            host_monotonic_ns_frameset_received=host_frameset_received,
            host_monotonic_ns_alignment_completed=host_alignment_completed,
            host_wall_time_ns_alignment_completed=wall_time_alignment_completed,
            device_serial=self._metadata["device_serial"],
            firmware_version=self._metadata["firmware_version"],
            usb_type_descriptor=self._metadata["usb_type_descriptor"],
        )

    def wait_for_stable_frames(
        self, consecutive: int = 15, maximum_skew_ms: float = 2.0
    ) -> D455Frame:
        """Wait for a strictly monotonic synchronized startup window."""

        if consecutive <= 0:
            raise ValueError("consecutive must be positive")
        maximum_skew_ms = _finite(maximum_skew_ms, "maximum skew")
        if maximum_skew_ms < 0.0:
            raise ValueError("maximum skew must be nonnegative")
        previous: Optional[D455Frame] = None
        stable = 0
        attempts = 0
        maximum_attempts = max(60, 10 * consecutive)
        while attempts < maximum_attempts:
            current = self.wait_for_frame()
            attempts += 1
            monotonic = previous is not None and (
                current.depth_frame_number == previous.depth_frame_number + 1
                and current.color_frame_number == previous.color_frame_number + 1
                and current.depth_timestamp_ms > previous.depth_timestamp_ms
                and current.color_timestamp_ms > previous.color_timestamp_ms
            )
            synchronized = (
                current.device_timestamp_skew_ms <= maximum_skew_ms
                and current.depth_timestamp_domain == current.color_timestamp_domain
            )
            stable = stable + 1 if monotonic and synchronized else 0
            previous = current
            if stable >= consecutive:
                return current
        raise D455CaptureError(
            f"stream did not reach {consecutive} stable synchronized frames"
        )


# Generic names for new code; legacy imports remain source-compatible.
RealSenseCaptureError = D455CaptureError
RealSenseFrame = D455Frame
RealSenseCapture = D455Capture
