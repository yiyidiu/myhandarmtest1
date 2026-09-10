#!/usr/bin/env python3
"""Run the existing raw IMU reader with graceful CSV closure on both signals."""
import signal
from read_wt9011dcl_bt50 import main


def stop(*_args):
    raise KeyboardInterrupt()


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
if __name__ == "__main__":
    raise SystemExit(main())
