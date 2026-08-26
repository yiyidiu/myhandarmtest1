"""Thread-safe operator gate for the live HaMeR teleoperation stream.

Opening the camera must never arm robot motion.  A successful explicit C-key
confirmation creates a new reference epoch.  Every UDP pose carries that epoch
so the ROS adapter can atomically reset both the hand and robot references
before it publishes the first enabled pose.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Mapping, Optional, Tuple


class TeleopControlGate:
    """Decorate live pose packets with an explicit, repeatable C-key gate."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._enabled = False
        self._reference_epoch = 0
        self._bound_identity: Optional[Tuple[int, int, bool]] = None
        self._lock_reason = "WAITING_FOR_OPERATOR_C_REFERENCE"

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def reference_epoch(self) -> int:
        with self._lock:
            return self._reference_epoch

    @property
    def lock_reason(self) -> str:
        with self._lock:
            return self._lock_reason

    @property
    def bound_identity(self) -> Optional[Tuple[int, int, bool]]:
        with self._lock:
            return self._bound_identity

    @staticmethod
    def _identity(
        presence_generation: Any,
        active_hand_generation: Any,
        hand_is_right: Any,
    ) -> Tuple[int, int, bool]:
        if not isinstance(hand_is_right, bool):
            raise ValueError("hand_is_right must be boolean")
        presence = int(presence_generation)
        active_hand = int(active_hand_generation)
        if presence < 0 or active_hand < 0:
            raise ValueError("hand identity generations must be non-negative")
        return presence, active_hand, hand_is_right

    @staticmethod
    def reference_token(
        session_id: str,
        reference_epoch: int,
        identity: Tuple[int, int, bool],
    ) -> str:
        presence, active_hand, is_right = identity
        return "{}:{}:p{}:h{}:{}".format(
            session_id,
            int(reference_epoch),
            int(presence),
            int(active_hand),
            "R" if is_right else "L",
        )

    @staticmethod
    def _invalidate_dependent_observations(
        output: Dict[str, Any], reason: str
    ) -> None:
        """Keep nested observations consistent when the outer pose is locked."""

        fingers = output.get("finger_observation")
        if isinstance(fingers, Mapping):
            fingers = dict(fingers)
            fingers["valid"] = False
            fingers["flexion"] = [0.0] * 5
            fingers["confidence"] = 0.0
            fingers["invalid_reason"] = str(
                reason or "CONTROL_GATE_LOCKED"
            )
            output["finger_observation"] = fingers

    def disable(self, reason: str = "WAITING_FOR_OPERATOR_C_REFERENCE") -> None:
        """Lock output without changing the most recent reference epoch."""

        with self._lock:
            self._enabled = False
            self._bound_identity = None
            self._lock_reason = str(reason or "CONTROL_GATE_DISABLED")

    def confirm(
        self,
        presence_generation: int,
        active_hand_generation: int,
        hand_is_right: bool,
    ) -> int:
        """Bind C-zero to one observed hand identity and return a new epoch."""

        identity = self._identity(
            presence_generation, active_hand_generation, hand_is_right
        )
        with self._lock:
            self._reference_epoch += 1
            self._bound_identity = identity
            self._enabled = True
            self._lock_reason = ""
            return self._reference_epoch

    def observe_identity(
        self,
        presence_generation: int,
        active_hand_generation: int,
        hand_is_right: bool,
    ) -> bool:
        """Fail closed if a live C session no longer observes its bound hand."""

        identity = self._identity(
            presence_generation, active_hand_generation, hand_is_right
        )
        with self._lock:
            if self._enabled and identity != self._bound_identity:
                self._enabled = False
                self._bound_identity = None
                self._lock_reason = "HAND_IDENTITY_CHANGED_REQUIRES_NEW_C"
            return self._enabled

    def invalidate(self, reason: str) -> None:
        """Fail closed after hand loss/reacquisition or another safety event."""

        self.disable(reason or "HAND_IDENTITY_INVALID_REQUIRES_NEW_C")

    def decorate(
        self, packet: Mapping[str, Any], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return a packet carrying the current fail-closed control state."""

        output = dict(packet)
        session = str(
            output.get("session_id", "") if session_id is None else session_id
        )
        identity_present = output.get("hand_identity_present") is True
        observed_identity = None
        if identity_present:
            observed_identity = self._identity(
                output.get("presence_generation", -1),
                output.get("active_hand_generation", -1),
                output.get("hand_is_right"),
            )
        with self._lock:
            if self._enabled and observed_identity != self._bound_identity:
                self._enabled = False
                self._bound_identity = None
                self._lock_reason = "HAND_IDENTITY_CHANGED_REQUIRES_NEW_C"
                output["valid"] = False
                output["invalid_reason"] = self._lock_reason
            enabled = self._enabled
            epoch = self._reference_epoch
            bound_identity = self._bound_identity
            lock_reason = self._lock_reason
        if not enabled:
            output["valid"] = False
            output["invalid_reason"] = str(
                lock_reason or output.get("invalid_reason", "")
                or "WAITING_FOR_OPERATOR_C_REFERENCE"
            )
        if output.get("valid") is not True:
            self._invalidate_dependent_observations(
                output, str(output.get("invalid_reason", ""))
            )
        output["control_enabled"] = bool(enabled)
        output["control_reference_epoch"] = int(epoch)
        output["control_identity_present"] = bool(
            enabled and bound_identity is not None
        )
        output["control_presence_generation"] = int(
            0 if bound_identity is None else bound_identity[0]
        )
        output["control_active_hand_generation"] = int(
            0 if bound_identity is None else bound_identity[1]
        )
        output["control_hand_is_right"] = bool(
            False if bound_identity is None else bound_identity[2]
        )
        output["control_reference_token"] = (
            self.reference_token(session, epoch, bound_identity)
            if enabled and session and epoch > 0 and bound_identity is not None
            else ""
        )
        output["control_gate_reason"] = str(lock_reason)
        return output


__all__ = ["TeleopControlGate"]
