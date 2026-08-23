#!/usr/bin/env python3
"""Threading primitives for independent RGB-D Kabsch and latest-only HaMeR."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class P5CapturePacket:
    frame: Any
    palm_roi: Any
    hand_roi: Any
    sequence: int


class SequentialCaptureQueue:
    """Small FIFO for consecutive RGB-D frames; overflow is explicit."""

    def __init__(self, capacity: int = 4) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=int(capacity))
        self.dropped = 0
        self.maximum_size = 0

    def publish(self, packet: P5CapturePacket) -> bool:
        try:
            self._queue.put_nowait(packet)
            self.maximum_size = max(self.maximum_size, self._queue.qsize())
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def get(self, timeout_s: float = 3.0) -> P5CapturePacket:
        return self._queue.get(timeout=float(timeout_s))


class LatestOnlySlot:
    """One-element overwrite mailbox used exclusively by asynchronous HaMeR."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._version = 0
        self._value: Any = None
        self._closed = False
        self.published = 0
        self.overwritten = 0
        self.consumed = 0

    def publish(self, value: Any) -> None:
        with self._condition:
            if self._closed:
                return
            if self.published > self.consumed:
                self.overwritten += 1
            self._value = value
            self._version += 1
            self.published += 1
            self._condition.notify_all()

    def get_after(self, version: int, timeout_s: float = 1.0) -> Tuple[int, Any]:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while self._version <= version and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError
                self._condition.wait(remaining)
            if self._version <= version:
                return version, None
            self.consumed = self.published
            return self._version, self._value

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def stats(self) -> dict:
        return {
            "capacity": 1,
            "policy": "overwrite_old_keep_latest",
            "published": self.published,
            "consumed": self.consumed,
            "overwritten": self.overwritten,
        }


class HamerContextState:
    """Lock-protected non-orientation HaMeR context."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload = {
            "valid": False,
            "hand_pose": None,
            "gesture_changing": False,
            "timestamp": None,
            "inference_ms": None,
            "failure_reason": "NOT_RUN",
        }

    def update(self, payload: dict) -> None:
        forbidden = {"global_orient", "rotation", "orientation"} & set(payload)
        if forbidden:
            raise ValueError("HaMeR orientation fields are forbidden in P5 context")
        with self._lock:
            self._payload = dict(payload)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._payload)


__all__ = [
    "HamerContextState",
    "LatestOnlySlot",
    "P5CapturePacket",
    "SequentialCaptureQueue",
]
