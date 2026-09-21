"""Thread-safe operator gate for the live HaMeR teleoperation stream.

Opening the camera must never arm robot motion.  A successful explicit C-key
confirmation creates a new reference epoch.  Every UDP pose carries that epoch
so the ROS adapter can atomically reset both the hand and robot references
before it publishes the first enabled pose.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


class TeleopControlGate:
    """Decorate live pose packets with an explicit, repeatable C-key gate."""

    def __init__(
        self, retain_reference_until_operator_action: bool = False,
        startup_rotation_noise_deg: float = 0.0,
        confirmation_delay_s: float = 0.0,
    ) -> None:
        self._lock = threading.RLock()
        self._retain_reference_until_operator_action = bool(
            retain_reference_until_operator_action
        )
        self._enabled = False
        self._reference_epoch = 0
        self._bound_identity: Optional[Tuple[int, int, bool]] = None
        self._lock_reason = "WAITING_FOR_OPERATOR_C_REFERENCE"
        self._confirmation_request = None
        self._confirmation_samples = []
        self._confirmation_delay_s = float(confirmation_delay_s)
        if not np.isfinite(self._confirmation_delay_s) or self._confirmation_delay_s < 0.0:
            raise ValueError("confirmation delay must be finite and non-negative")
        self._reference_pose = None
        self._startup_rotation_noise_deg = float(startup_rotation_noise_deg)
        if not np.isfinite(self._startup_rotation_noise_deg) or not 0.0 <= self._startup_rotation_noise_deg < 5.0:
            raise ValueError("startup rotation noise bound must be in [0, 5) degrees")
        self._startup_rotation_pending = False
        self._startup_rotation_count = 0
        self._startup_rotation_sequence = -1

    @property
    def confirmation_pending(self) -> bool:
        with self._lock:
            return self._confirmation_request is not None

    def confirmation_countdown_s(self, now=None):
        """Return time left to settle, zero while sampling, or None if idle."""
        with self._lock:
            if self._confirmation_request is None:
                return None
            began, _ = self._confirmation_request
            current = time.monotonic() if now is None else float(now)
            return max(0.0, began + self._confirmation_delay_s - current)

    @property
    def reference_pose(self):
        with self._lock:
            return None if self._reference_pose is None else dict(self._reference_pose)

    def request_confirmation(self, after_sequence: int, now=None) -> None:
        """The display requests C; only inference may choose its pose frame."""
        with self._lock:
            self.disable("OPERATOR_REQUESTED_NEW_C_REFERENCE")
            self._confirmation_request = (
                time.monotonic() if now is None else float(now),
                int(after_sequence),
            )
            self._confirmation_samples = []

    def confirm_packet(self, packet: Mapping[str, Any], calibrate_display, now=None,
                       source_monotonic_s=None):
        """Bind display and outgoing first enabled packet to the SAME frame.

        After the operator settling delay, four quiet observations reject a
        one-frame depth outlier as C-zero. Source acquisition must also start
        after the delay; a queued pre-delay frame cannot become the new zero.
        This runs in the inference thread immediately after updating the
        display pose, before decorating/sending this very packet.
        """
        with self._lock:
            if self._confirmation_request is None:
                return None
            began, after_sequence = self._confirmation_request
            current = time.monotonic() if now is None else float(now)
            capture_after = began + self._confirmation_delay_s
            # Preserve the existing two-second valid/stable-pose search
            # window, but start it AFTER the operator's settling countdown.
            if current - capture_after > 2.0:
                self._confirmation_request = None
                self._confirmation_samples = []
                self._lock_reason = "C_REFERENCE_NOT_STABLE_RETRY"
                return None
            if current < capture_after or (self._confirmation_delay_s > 0.0 and (
                    source_monotonic_s is None or
                    not np.isfinite(float(source_monotonic_s)) or
                    float(source_monotonic_s) < capture_after)):
                self._confirmation_samples = []
                return None
            position = np.asarray(packet.get("wrist_position_m", []), dtype=float)
            rotation = np.asarray(packet.get("palm_rotation_row_major", []), dtype=float)
            confidence = np.asarray(packet.get("confidence", []), dtype=float)
            fingers = packet.get("finger_observation") or {}
            flexion = np.asarray(fingers.get("flexion", []), dtype=float)
            if (packet.get("valid") is not True or
                    int(packet.get("sequence", -1)) <= after_sequence or
                    position.shape != (3,) or rotation.size != 9 or
                    confidence.shape != (6,) or not np.all(confidence > 0.0) or
                    not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)) or
                    packet.get("orientation_held", False) or
                    not packet.get("orientation_channel_valid", True) or
                    fingers.get("valid") is not True or flexion.shape != (5,) or
                    not np.all(np.isfinite(flexion)) or not np.all(flexion[1:] <= 0.40)):
                self._confirmation_samples = []
                return None
            sequence = int(packet["sequence"])
            if self._confirmation_samples and sequence <= self._confirmation_samples[-1][2]:
                return None
            self._confirmation_samples.append((position.copy(), rotation.reshape(3, 3).copy(), sequence))
            self._confirmation_samples = self._confirmation_samples[-4:]
            if len(self._confirmation_samples) < 4:
                return None
            positions = np.asarray([x[0] for x in self._confirmation_samples])
            rotations = np.asarray([x[1] for x in self._confirmation_samples])
            if (np.max(np.linalg.norm(positions - np.median(positions, axis=0), axis=1)) > 0.006 or
                    np.max(np.linalg.norm(rotations - rotations[-1], axis=(1, 2))) > 0.074):
                return None
            if not calibrate_display():
                return None
            epoch = self.confirm(packet["presence_generation"],
                                 packet["active_hand_generation"], packet["hand_is_right"])
            self._reference_pose = {
                "sequence": sequence, "stamp": float(packet["stamp"]),
                "wrist_position_m": position.tolist(),
                "palm_rotation_row_major": rotation.reshape(-1).tolist(),
            }
            surface = packet.get("palm_surface_observation") or {}
            if surface.get("valid"):
                # Auxiliary geometry belongs to the very same C frame.
                # Its absence does not add a C requirement or a motion hold.
                self._reference_pose["palm_surface_observation"] = {
                    "valid": True,
                    "rotation_row_major": list(surface["rotation_row_major"]),
                }
            self._startup_rotation_pending = self._startup_rotation_noise_deg > 0.0
            self._startup_rotation_count = 0
            self._startup_rotation_sequence = sequence
            self._confirmation_request = None
            self._confirmation_samples = []
            return epoch

    def filter_startup_orientation(self, packet: Mapping[str, Any]) -> Dict[str, Any]:
        """Do not turn calibration noise into the first robot rotation.

        Only the initial stationary interval after an explicit C is covered.
        Three distinct observations beyond the measured quiet range release
        the unchanged absolute pose map. There is no moving zero, angle-gain
        reduction, or translation/rotation intent classification.
        """
        output = dict(packet)
        with self._lock:
            if not self._enabled or not self._startup_rotation_pending or self._reference_pose is None:
                return output
            if (packet.get("valid") is not True or packet.get("orientation_held", False)
                    or not packet.get("orientation_channel_valid", True)):
                self._startup_rotation_count = 0
                return output
            reference = np.asarray(self._reference_pose["palm_rotation_row_major"]).reshape(3, 3)
            observed = np.asarray(packet["palm_rotation_row_major"]).reshape(3, 3)
            angle = float(np.rad2deg(np.arccos(np.clip(
                (np.trace(observed @ reference.T)-1.0)*0.5, -1.0, 1.0))))
            sequence = int(packet["sequence"])
            if sequence > self._startup_rotation_sequence:
                self._startup_rotation_count = (self._startup_rotation_count + 1
                    if angle > self._startup_rotation_noise_deg else 0)
                self._startup_rotation_sequence = sequence
            if self._startup_rotation_count >= 3:
                self._startup_rotation_pending = False
            output["reference_startup_rotation"] = {
                "held": self._startup_rotation_pending, "observed_angle_deg": angle,
                "quiet_bound_deg": self._startup_rotation_noise_deg,
                "consecutive_clear_observations": self._startup_rotation_count,
            }
            if self._startup_rotation_pending:
                output["palm_rotation_row_major"] = reference.reshape(-1).tolist()
                output["orientation_held"] = True
                output["confidence"] = list(packet["confidence"][:3]) + [0.0]*3
            return output

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
            self._reference_pose = None
            self._confirmation_request = None
            self._confirmation_samples = []
            self._startup_rotation_pending = False

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
            self._reference_pose = None
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
                if not self._retain_reference_until_operator_action:
                    self._enabled = False
                    self._bound_identity = None
                    self._lock_reason = (
                        "HAND_IDENTITY_CHANGED_REQUIRES_NEW_C"
                    )
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
            if (
                self._enabled
                and not self._retain_reference_until_operator_action
                and observed_identity != self._bound_identity
            ):
                self._enabled = False
                self._bound_identity = None
                self._lock_reason = "HAND_IDENTITY_CHANGED_REQUIRES_NEW_C"
                output["valid"] = False
                output["invalid_reason"] = self._lock_reason
            enabled = self._enabled
            epoch = self._reference_epoch
            bound_identity = self._bound_identity
            lock_reason = self._lock_reason
            reference_pose = self._reference_pose
        if (
            enabled
            and self._retain_reference_until_operator_action
            and bound_identity is not None
        ):
            # In live Gazebo operation C is an operator clutch, not a fragile
            # detector lease.  Missing/reacquired detections may invalidate
            # geometry, but they cannot rewrite or cancel the C-bound robot
            # reference.  Keep one stable identity token on every heartbeat;
            # the active-hand selector still guarantees a single pose source.
            if output.get("valid") is True and observed_identity is None:
                output["valid"] = False
                output["invalid_reason"] = (
                    "HAND_IDENTITY_UNAVAILABLE_HOLDING_C_REFERENCE"
                )
            output["hand_identity_present"] = True
            output["presence_generation"] = int(bound_identity[0])
            output["active_hand_generation"] = int(bound_identity[1])
            output["hand_is_right"] = bool(bound_identity[2])
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
        # Repeat the immutable C pose with its token. A dropped/rejected
        # first enabled UDP packet must not make ROS choose a different zero.
        if enabled and reference_pose is not None:
            output["control_reference_pose"] = dict(reference_pose)
        return output


__all__ = ["TeleopControlGate"]
