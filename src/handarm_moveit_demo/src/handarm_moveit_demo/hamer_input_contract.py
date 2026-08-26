"""Pure validation and watchdog state for the live HaMeR UDP contract."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .shared_teleop_core import matrix_to_quaternion_xyzw, project_to_so3


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError("{} must be an integer".format(name))
    parsed = int(value)
    if parsed < 0 or float(parsed) != float(value):
        raise ValueError("{} must be a non-negative integer".format(name))
    return parsed


def _unit_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("{} must be finite".format(name))
    return float(np.clip(parsed, 0.0, 1.0))


def identity_reference_token(
    session_id: str,
    reference_epoch: int,
    presence_generation: int,
    active_hand_generation: int,
    hand_is_right: bool,
) -> str:
    """Return the camera/ROS token bound to one explicit hand identity."""

    if not isinstance(hand_is_right, bool):
        raise ValueError("hand_is_right must be boolean")
    return "{}:{}:p{}:h{}:{}".format(
        str(session_id),
        _nonnegative_int(reference_epoch, "reference_epoch"),
        _nonnegative_int(presence_generation, "presence_generation"),
        _nonnegative_int(active_hand_generation, "active_hand_generation"),
        "R" if hand_is_right else "L",
    )


class HamerPacketContract:
    """Validate packets before committing their session/sequence state."""

    def __init__(self, default_frame: str) -> None:
        self.default_frame = str(default_frame)
        self.session_id: Optional[str] = None
        self.last_sequence: Optional[int] = None

    @staticmethod
    def _source_stamp(value: Any) -> float:
        stamp = float(value)
        if not math.isfinite(stamp) or stamp <= 0.0:
            raise ValueError("stamp must be a positive wall-clock second value")
        return stamp

    @staticmethod
    def _identity(packet: Mapping[str, Any]) -> Dict[str, Any]:
        present = packet.get("hand_identity_present") is True
        if not present:
            return {
                "hand_identity_present": False,
                "hand_is_right": False,
                "presence_generation": _nonnegative_int(
                    packet.get("presence_generation", 0),
                    "presence_generation",
                ),
                "active_hand_generation": _nonnegative_int(
                    packet.get("active_hand_generation", 0),
                    "active_hand_generation",
                ),
            }
        hand_is_right = packet.get("hand_is_right")
        if not isinstance(hand_is_right, bool):
            raise ValueError("hand_is_right must be boolean when identity is present")
        return {
            "hand_identity_present": True,
            "hand_is_right": hand_is_right,
            "presence_generation": _nonnegative_int(
                packet.get("presence_generation"), "presence_generation"
            ),
            "active_hand_generation": _nonnegative_int(
                packet.get("active_hand_generation"),
                "active_hand_generation",
            ),
        }

    @staticmethod
    def _pose(packet: Mapping[str, Any]) -> Dict[str, Any]:
        position = np.asarray(packet["wrist_position_m"], dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("wrist_position_m must be a finite 3-vector")
        if "palm_rotation_row_major" in packet:
            rotation = project_to_so3(
                np.asarray(
                    packet["palm_rotation_row_major"], dtype=float
                ).reshape(3, 3)
            )
            quaternion = matrix_to_quaternion_xyzw(rotation)
        else:
            quaternion = np.asarray(
                packet["palm_quaternion_xyzw"], dtype=float
            )
            if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
                raise ValueError("palm_quaternion_xyzw must be finite xyzw")
            norm = float(np.linalg.norm(quaternion))
            if norm < 1.0e-10:
                raise ValueError("zero quaternion")
            quaternion = quaternion / norm
        confidence = np.asarray(packet.get("confidence", [1.0] * 6), dtype=float)
        if confidence.shape != (6,) or not np.all(np.isfinite(confidence)):
            raise ValueError("confidence must be a finite six-vector")
        return {
            "position": position,
            "quaternion": quaternion,
            "confidence": np.clip(confidence, 0.0, 1.0),
        }

    def validate(self, packet: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(packet, Mapping):
            raise ValueError("packet must be a mapping")
        if packet.get("schema") != "handarm_hamer_pose_v1":
            raise ValueError("unsupported schema")
        session = str(packet.get("session_id", ""))
        if not session:
            raise ValueError("session_id is required")
        sequence = _nonnegative_int(packet.get("sequence"), "sequence")
        if (
            self.session_id == session
            and self.last_sequence is not None
            and sequence <= self.last_sequence
        ):
            raise ValueError("duplicate_or_out_of_order_sequence")
        source_stamp = self._source_stamp(packet.get("stamp"))
        frame_id = str(packet.get("frame_id", self.default_frame))
        if frame_id != self.default_frame:
            raise ValueError("unexpected_camera_frame:{}".format(frame_id))

        observation_valid = packet.get("valid") is True
        identity = self._identity(packet)
        control_enabled = packet.get("control_enabled") is True
        reference_epoch = _nonnegative_int(
            packet.get("control_reference_epoch", 0),
            "control_reference_epoch",
        )
        reference_token = str(packet.get("control_reference_token", ""))
        if control_enabled:
            if not identity["hand_identity_present"]:
                raise ValueError("enabled_control_requires_hand_identity")
            if packet.get("control_identity_present") is not True:
                raise ValueError("enabled_control_requires_bound_identity")
            control_presence = _nonnegative_int(
                packet.get("control_presence_generation"),
                "control_presence_generation",
            )
            control_active_hand = _nonnegative_int(
                packet.get("control_active_hand_generation"),
                "control_active_hand_generation",
            )
            control_is_right = packet.get("control_hand_is_right")
            if not isinstance(control_is_right, bool):
                raise ValueError("control_hand_is_right must be boolean")
            observed = (
                identity["presence_generation"],
                identity["active_hand_generation"],
                identity["hand_is_right"],
            )
            bound = (control_presence, control_active_hand, control_is_right)
            if observed != bound:
                raise ValueError("control_identity_does_not_match_observation")
            expected_token = identity_reference_token(
                session,
                reference_epoch,
                control_presence,
                control_active_hand,
                control_is_right,
            )
            if reference_epoch <= 0 or reference_token != expected_token:
                raise ValueError("invalid_control_reference_token")
        elif reference_token:
            raise ValueError("disabled_control_must_not_carry_reference_token")

        if observation_valid:
            pose = self._pose(packet)
        else:
            pose = {
                "position": np.zeros(3, dtype=float),
                "quaternion": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "confidence": np.zeros(6, dtype=float),
            }

        gesture = _nonnegative_int(packet.get("gesture", 0), "gesture")
        if gesture > 255:
            raise ValueError("gesture must fit uint8")
        gesture_confidence = _unit_float(
            packet.get("gesture_confidence", 0.0), "gesture_confidence"
        )
        normalized = {
            "session_id": session,
            "sequence": sequence,
            "source_stamp": source_stamp,
            "frame_id": frame_id,
            "observation_valid": observation_valid,
            "invalid_reason": str(
                packet.get("invalid_reason", "")
                or ("" if observation_valid else "HAMER_POSE_INVALID")
            ),
            "gesture": gesture,
            "gesture_confidence": gesture_confidence,
            "control_enabled": control_enabled,
            "control_reference_epoch": reference_epoch,
            "control_reference_token": reference_token,
            **identity,
            **pose,
        }
        # Commit ordering state only after every field and identity check passes.
        self.session_id = session
        self.last_sequence = sequence
        return normalized


class InputWatchdog:
    """Schedule explicit fail-closed publications while UDP input is absent."""

    def __init__(
        self,
        timeout_s: float,
        repeat_s: float,
        start_monotonic: Optional[float] = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.repeat_s = float(repeat_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if not math.isfinite(self.repeat_s) or self.repeat_s <= 0.0:
            raise ValueError("repeat_s must be finite and positive")
        now = time.monotonic() if start_monotonic is None else float(
            start_monotonic
        )
        if not math.isfinite(now):
            raise ValueError("start_monotonic must be finite")
        self.last_accepted_monotonic = now
        self.last_timeout_publish_monotonic: Optional[float] = None

    def mark_accepted(self, now_monotonic: Optional[float] = None) -> None:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not math.isfinite(now):
            raise ValueError("now_monotonic must be finite")
        self.last_accepted_monotonic = now
        self.last_timeout_publish_monotonic = None

    def timeout_due(self, now_monotonic: Optional[float] = None) -> bool:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not math.isfinite(now):
            raise ValueError("now_monotonic must be finite")
        if now - self.last_accepted_monotonic + 1.0e-12 < self.timeout_s:
            return False
        if (
            self.last_timeout_publish_monotonic is not None
            and now - self.last_timeout_publish_monotonic + 1.0e-12
            < self.repeat_s
        ):
            return False
        self.last_timeout_publish_monotonic = now
        return True


__all__ = [
    "HamerPacketContract",
    "InputWatchdog",
    "identity_reference_token",
]
