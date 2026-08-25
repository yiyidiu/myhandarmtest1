"""Thread-safe operator gate for the live HaMeR teleoperation stream.

Opening the camera must never arm robot motion.  A successful explicit C-key
confirmation creates a new reference epoch.  Every UDP pose carries that epoch
so the ROS adapter can atomically reset both the hand and robot references
before it publishes the first enabled pose.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Mapping, Optional


class TeleopControlGate:
    """Decorate live pose packets with an explicit, repeatable C-key gate."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._enabled = False
        self._reference_epoch = 0

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def reference_epoch(self) -> int:
        with self._lock:
            return self._reference_epoch

    def disable(self) -> None:
        """Lock output without changing the most recent reference epoch."""

        with self._lock:
            self._enabled = False

    def confirm(self) -> int:
        """Enable output and return a new monotonically increasing epoch."""

        with self._lock:
            self._reference_epoch += 1
            self._enabled = True
            return self._reference_epoch

    def decorate(
        self, packet: Mapping[str, Any], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return a packet carrying the current fail-closed control state."""

        with self._lock:
            enabled = self._enabled
            epoch = self._reference_epoch
        output = dict(packet)
        session = str(
            output.get("session_id", "") if session_id is None else session_id
        )
        output["control_enabled"] = bool(enabled)
        output["control_reference_epoch"] = int(epoch)
        output["control_reference_token"] = (
            "{}:{}".format(session, epoch) if enabled and session and epoch > 0 else ""
        )
        return output


__all__ = ["TeleopControlGate"]
