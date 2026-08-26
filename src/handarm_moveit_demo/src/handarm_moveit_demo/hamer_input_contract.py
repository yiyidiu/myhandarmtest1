"""Pure validation and watchdog state for the live HaMeR UDP contract."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .shared_teleop_core import matrix_to_quaternion_xyzw, project_to_so3


FINGER_FEATURE_DEFINITION = "mano_openpose_chain_total_bend_over_pi_v1"


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


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return parsed


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


def authoritative_camera_c_gate_lock(
    control_enabled: Any,
    timing_contract_present: Any,
) -> bool:
    """Distinguish a camera C-up packet from adapter-generated lock status."""

    return control_enabled is False and timing_contract_present is True


class HamerPacketContract:
    """Validate packets before committing their session/sequence state."""

    def __init__(
        self,
        default_frame: str,
        maximum_pipeline_latency_s: Optional[float] = None,
        require_timing_contract: bool = False,
        require_finger_contract: bool = False,
    ) -> None:
        self.default_frame = str(default_frame)
        self.require_timing_contract = bool(require_timing_contract)
        self.require_finger_contract = bool(require_finger_contract)
        self.maximum_pipeline_latency_s = (
            None
            if maximum_pipeline_latency_s is None
            else float(maximum_pipeline_latency_s)
        )
        if (
            self.maximum_pipeline_latency_s is not None
            and (
                not math.isfinite(self.maximum_pipeline_latency_s)
                or self.maximum_pipeline_latency_s <= 0.0
            )
        ):
            raise ValueError(
                "maximum_pipeline_latency_s must be finite and positive"
            )
        self.session_id: Optional[str] = None
        self.last_sequence: Optional[int] = None

    def _fingers(
        self, packet: Mapping[str, Any], observation_valid: bool
    ) -> Dict[str, Any]:
        raw = packet.get("finger_observation")
        if raw is None:
            if self.require_finger_contract:
                raise ValueError("live packet requires finger observation contract")
            return {
                "finger_tracking_present": False,
                "finger_tracking_valid": False,
                "finger_flexion": np.zeros(5, dtype=float),
                "finger_tracking_confidence": 0.0,
                "finger_invalid_reason": "FINGER_CONTRACT_ABSENT",
            }
        if not isinstance(raw, Mapping):
            raise ValueError("finger_observation must be a mapping")
        if _nonnegative_int(
            raw.get("contract_version"),
            "finger_observation.contract_version",
        ) != 1:
            raise ValueError("unsupported finger observation contract version")
        if raw.get("feature_definition") != FINGER_FEATURE_DEFINITION:
            raise ValueError("unsupported finger feature definition")
        valid = raw.get("valid") is True
        if valid and not observation_valid:
            raise ValueError("invalid pose cannot carry valid finger tracking")
        flexion = np.asarray(raw.get("flexion"), dtype=float)
        if flexion.shape != (5,) or not np.all(np.isfinite(flexion)):
            raise ValueError("finger flexion must be a finite five-vector")
        if np.any(flexion < 0.0) or np.any(flexion > 1.0):
            raise ValueError("finger flexion must remain in [0,1]")
        confidence = _unit_float(
            raw.get("confidence", 0.0), "finger_observation.confidence"
        )
        return {
            "finger_tracking_present": True,
            "finger_tracking_valid": valid,
            "finger_flexion": flexion,
            "finger_tracking_confidence": confidence,
            "finger_invalid_reason": str(
                raw.get("invalid_reason", "")
                or ("" if valid else "FINGER_OBSERVATION_INVALID")
            ),
        }

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

    def _timing(
        self, packet: Mapping[str, Any], observation_valid: bool
    ) -> Dict[str, Any]:
        raw = packet.get("timing")
        if raw is None:
            if self.require_timing_contract:
                raise ValueError("live packet requires timing contract")
            return {
                "timing_contract_present": False,
                "source_capture_sequence": 0,
                "dropped_capture_frames": 0,
                "capture_to_publish_s": 0.0,
                "inference_executed": False,
                "inference_call_s": 0.0,
                "model_inference_s": 0.0,
                "postprocess_s": 0.0,
            }
        if not isinstance(raw, Mapping):
            raise ValueError("timing must be a mapping")
        if _nonnegative_int(
            raw.get("contract_version"), "timing.contract_version"
        ) != 1:
            raise ValueError("unsupported timing contract version")
        capture_sequence = _nonnegative_int(
            raw.get("capture_sequence"), "timing.capture_sequence"
        )
        dropped = _nonnegative_int(
            raw.get("dropped_capture_frames"),
            "timing.dropped_capture_frames",
        )
        if capture_sequence > 2**64 - 1 or dropped > 2**32 - 1:
            raise ValueError("timing sequence/drop count exceeds ROS field width")
        capture_to_publish = _nonnegative_float(
            raw.get("capture_to_publish_s"),
            "timing.capture_to_publish_s",
        )
        inference_executed = raw.get("inference_executed") is True
        inference_call = _nonnegative_float(
            raw.get("inference_call_s"), "timing.inference_call_s"
        )
        model_inference = _nonnegative_float(
            raw.get("model_inference_s"), "timing.model_inference_s"
        )
        postprocess = _nonnegative_float(
            raw.get("postprocess_s"), "timing.postprocess_s"
        )
        if inference_executed and inference_call <= 0.0:
            raise ValueError("executed inference must have positive call time")
        if not inference_executed and (
            inference_call > 0.0 or model_inference > 0.0
        ):
            raise ValueError("non-executed inference must have zero timings")
        if model_inference > inference_call + 1.0e-6:
            raise ValueError("model inference time exceeds total inference call")
        if observation_valid and (
            not inference_executed or model_inference <= 0.0
        ):
            raise ValueError("valid observation requires measured inference timing")
        if (
            observation_valid
            and self.maximum_pipeline_latency_s is not None
            and capture_to_publish > self.maximum_pipeline_latency_s
        ):
            raise ValueError(
                "source_pipeline_latency_exceeded:{:.6f}>{:.6f}".format(
                    capture_to_publish, self.maximum_pipeline_latency_s
                )
            )
        return {
            "timing_contract_present": True,
            "source_capture_sequence": capture_sequence,
            "dropped_capture_frames": dropped,
            "capture_to_publish_s": capture_to_publish,
            "inference_executed": inference_executed,
            "inference_call_s": inference_call,
            "model_inference_s": model_inference,
            "postprocess_s": postprocess,
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
        timing = self._timing(packet, observation_valid)
        fingers = self._fingers(packet, observation_valid)
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
            **timing,
            **fingers,
            **identity,
            **pose,
        }
        # Commit ordering state only after every field and identity check passes.
        self.session_id = session
        self.last_sequence = sequence
        return normalized


class ReferenceTokenInterlock:
    """Latch a failed live C token until the camera creates a new epoch."""

    def __init__(self) -> None:
        self.last_accepted_token: Optional[str] = None
        self._blocked_tokens = set()

    @property
    def blocked_token(self) -> Optional[str]:
        """Backward-compatible single-value view for diagnostics/tests."""

        if not self._blocked_tokens:
            return None
        if self.last_accepted_token in self._blocked_tokens:
            return self.last_accepted_token
        return sorted(self._blocked_tokens)[0]

    def accept(self, normalized: Mapping[str, Any]) -> None:
        enabled = normalized.get("control_enabled") is True
        token = str(normalized.get("control_reference_token", ""))
        if not enabled:
            return
        if not token:
            raise ValueError("enabled control has no reference token")
        if token in self._blocked_tokens:
            raise ValueError("blocked_reference_token_requires_new_c")
        if self._blocked_tokens:
            self._blocked_tokens.clear()
        self.last_accepted_token = token

    def require_new_reference(
        self, rejected_token: Optional[str] = None
    ) -> Optional[str]:
        candidate = str(rejected_token or "")
        if self.last_accepted_token:
            self._blocked_tokens.add(self.last_accepted_token)
        if candidate:
            self._blocked_tokens.add(candidate)
            return candidate
        return self.last_accepted_token


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
    "authoritative_camera_c_gate_lock",
    "HamerPacketContract",
    "InputWatchdog",
    "ReferenceTokenInterlock",
    "identity_reference_token",
]
