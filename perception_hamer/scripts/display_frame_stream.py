#!/usr/bin/env python3
"""Display length-prefixed JPEG frames received on stdin."""

from __future__ import annotations

import argparse
import struct
import sys
from typing import Optional

import cv2
import numpy as np


def _read_exact(size: int) -> Optional[bytes]:
    chunks = []
    remaining = size
    while remaining:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="D455 HaMeR live MANO mesh")
    parser.add_argument("--window-width", type=int, default=1120)
    parser.add_argument("--window-height", type=int, default=600)
    parser.add_argument("--window-x", type=int, default=20)
    parser.add_argument("--window-y", type=int, default=70)
    args = parser.parse_args()
    if args.window_width <= 0 or args.window_height <= 0:
        raise SystemExit("window dimensions must be positive")
    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.title, args.window_width, args.window_height)
    cv2.moveWindow(args.title, args.window_x, args.window_y)

    def mouse_callback(event, _x, _y, _flags, _parameter):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("CONFIRM", flush=True)

    cv2.setMouseCallback(args.title, mouse_callback)
    try:
        while True:
            header = _read_exact(4)
            if header is None:
                break
            size = struct.unpack("!I", header)[0]
            if size <= 0 or size > 16 * 1024 * 1024:
                raise RuntimeError("invalid display frame size: {}".format(size))
            payload = _read_exact(size)
            if payload is None:
                break
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            cv2.imshow(args.title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("QUIT", flush=True)
                break
            if key in (ord("r"), ord("R")):
                print("REINITIALIZE", flush=True)
            if key in (ord("c"), ord("C"), 10, 13, 32):
                print("CONFIRM", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
