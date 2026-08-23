#!/usr/bin/env python3
"""Subprocess-only harness for safe recorder signal/abnormal-exit tests.

It never imports pyrealsense2 and never opens a camera.  The parent test sends a
real SIGINT/SIGTERM after ``ready`` is created, or asks the fake capture to call
``os._exit`` to model an uncatchable process failure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.record_rgbd_session import record  # noqa: E402
from src.d455_capture import D455Frame  # noqa: E402


def _intrinsics():
    return {
        "width": 6,
        "height": 4,
        "fx": 390.0,
        "fy": 391.0,
        "ppx": 3.0,
        "ppy": 2.0,
        "distortion_model": "none",
        "coeffs": [0.0] * 5,
    }


def _frame(index: int) -> D455Frame:
    timestamp_ms = 100.0 + index * (1000.0 / 30.0)
    host_ns = 1_000_000_000 + index * 33_333_333
    return D455Frame(
        rgb=np.zeros((4, 6, 3), dtype=np.uint8),
        raw_depth_raw=np.full((4, 6), 1000 + index, dtype=np.uint16),
        aligned_depth_raw=np.full((4, 6), 1000 + index, dtype=np.uint16),
        depth_scale_m_per_unit=0.001,
        raw_depth_intrinsics=_intrinsics(),
        color_intrinsics=_intrinsics(),
        depth_to_color_extrinsics={
            "rotation_row_major": np.eye(3).reshape(-1).tolist(),
            "translation_m": [0.0, 0.0, 0.0],
        },
        aligned_to_color_intrinsics_differences={
            "width": 0.0,
            "height": 0.0,
            "fx": 0.0,
            "fy": 0.0,
            "ppx": 0.0,
            "ppy": 0.0,
        },
        color_frame_number=100 + index,
        depth_frame_number=100 + index,
        raw_color_frame_number=100 + index,
        raw_depth_frame_number=100 + index,
        color_timestamp_ms=timestamp_ms + 0.01,
        depth_timestamp_ms=timestamp_ms,
        raw_color_timestamp_ms=timestamp_ms + 0.01,
        raw_depth_timestamp_ms=timestamp_ms,
        color_timestamp_domain="global_time",
        depth_timestamp_domain="global_time",
        raw_color_timestamp_domain="global_time",
        raw_depth_timestamp_domain="global_time",
        host_monotonic_ns_before_wait=host_ns - 100,
        host_monotonic_ns_frameset_received=host_ns,
        host_monotonic_ns_alignment_completed=host_ns + 100,
        host_wall_time_ns_alignment_completed=host_ns + 200,
        device_serial="MOCK_NO_CAMERA",
        firmware_version="MOCK",
        usb_type_descriptor="2.1",
    )


class BlockingMockCapture:
    def __init__(self, *, ready: Path, mode: str, **_kwargs):
        self.ready = ready
        self.mode = mode
        self.index = 0
        self._frame0 = _frame(0)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    @property
    def device_metadata(self):
        metadata = self._frame0.metadata()
        return {
            key: metadata[key]
            for key in (
                "device_serial",
                "firmware_version",
                "usb_type_descriptor",
                "depth_scale_m_per_unit",
                "raw_depth_intrinsics",
                "color_intrinsics",
                "depth_to_color_extrinsics",
            )
        }

    def wait_for_stable_frames(self, **_kwargs):
        return self._frame0

    def wait_for_frame(self):
        self.ready.touch(exist_ok=True)
        if self.mode == "abrupt":
            os._exit(23)
        # Python signal handlers run while sleeping, so the production guard is
        # exercised without touching librealsense or hardware.
        while True:
            time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--mode", choices=("signal", "abrupt"), required=True)
    args = parser.parse_args()
    record_args = argparse.Namespace(
        output_root=args.output_root,
        session_name="fault_session",
        scenario="G00_STATIC",
        frames=10,
        duration_s=None,
        fps=30,
        warmup=0,
        stable_frames=1,
        maximum_skew_ms=2.0,
        allow_usb2=True,
        width=6,
        height=4,
        serial=None,
        timeout_ms=100,
        writer_queue_depth=4,
    )

    def capture_factory(**kwargs):
        return BlockingMockCapture(
            ready=Path(args.ready), mode=args.mode, **kwargs
        )

    try:
        record(record_args, capture_factory=capture_factory)
    except BaseException as exc:
        print(f"controlled child failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
