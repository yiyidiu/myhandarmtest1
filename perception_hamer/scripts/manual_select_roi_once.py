#!/usr/bin/env python3
"""GUI sidecar for one human mouse ROI selection from a raw RGB frame."""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    expected = args.width * args.height * 3
    raw = sys.stdin.buffer.read(expected)
    if len(raw) != expected:
        raise SystemExit("incomplete RGB frame")
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(args.height, args.width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x, y, width, height = cv2.selectROI(
        "Select hand ROI, then ENTER", bgr, fromCenter=False, showCrosshair=True
    )
    cv2.destroyAllWindows()
    valid = width > 1 and height > 1
    print(json.dumps({
        "valid": valid,
        "bbox": None if not valid else [x, y, x + width, y + height],
        "source": "human_mouse_selection",
        "hand_presence_validated": valid,
    }))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
