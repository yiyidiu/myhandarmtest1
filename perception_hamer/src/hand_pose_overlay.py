#!/usr/bin/env python3
"""Thread-safe display-only 6-D wrist pose derived from the live HaMeR packet.

The pose definition follows the supplied Teleoperation Core archive: metric
translation is in the RealSense colour-camera frame, rotation differences are
computed on SO(3), and ZYX Euler angles are produced only for the operator
display.  This module does not publish ROS or UDP commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class WristPoseDelta:
    valid: bool
    calibrated: bool
    reason: str
    center_m: Optional[np.ndarray]
    delta_m: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    rotation_geodesic_deg: float
    sequence: int
    stamp: float
    presence_generation: int
    orientation_source: str = ""
    forearm_applied: bool = False
    forearm_confidence: float = 0.0
    forearm_fusion_weight: float = 0.0
    forearm_status: str = "unavailable"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "calibrated": bool(self.calibrated),
            "reason": str(self.reason),
            "center_m": None if self.center_m is None else self.center_m.tolist(),
            "delta_m": None if self.delta_m is None else self.delta_m.tolist(),
            "confidence": (
                None if self.confidence is None else self.confidence.tolist()
            ),
            "yaw_deg": float(self.yaw_deg),
            "pitch_deg": float(self.pitch_deg),
            "roll_deg": float(self.roll_deg),
            "rotation_geodesic_deg": float(self.rotation_geodesic_deg),
            "sequence": int(self.sequence),
            "stamp": float(self.stamp),
            "presence_generation": int(self.presence_generation),
            "orientation_source": str(self.orientation_source),
            "forearm_applied": bool(self.forearm_applied),
            "forearm_confidence": float(self.forearm_confidence),
            "forearm_fusion_weight": float(self.forearm_fusion_weight),
            "forearm_status": str(self.forearm_status),
            "robot_output": False,
        }


def project_to_so3(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    u, _singular_values, vt = np.linalg.svd(value)
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("rotation projection did not produce SO(3)")
    return rotation


def matrix_to_euler_zyx_deg(rotation: Any) -> Tuple[float, float, float]:
    """Return yaw(Z), pitch(Y), roll(X) for R=Rz(yaw)Ry(pitch)Rx(roll)."""

    value = project_to_so3(rotation)
    horizontal = math.hypot(float(value[0, 0]), float(value[1, 0]))
    if horizontal > 1.0e-9:
        roll = math.atan2(float(value[2, 1]), float(value[2, 2]))
        pitch = math.atan2(-float(value[2, 0]), horizontal)
        yaw = math.atan2(float(value[1, 0]), float(value[0, 0]))
    else:
        roll = math.atan2(-float(value[1, 2]), float(value[1, 1]))
        pitch = math.atan2(-float(value[2, 0]), horizontal)
        yaw = 0.0
    return tuple(math.degrees(item) for item in (yaw, pitch, roll))


def rotation_geodesic_deg(rotation: Any) -> float:
    value = project_to_so3(rotation)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class RelativeWristPoseDisplay:
    """Causal C-zero pose state shared by inference and display threads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._zero_center_m: Optional[np.ndarray] = None
        self._zero_frame: Optional[np.ndarray] = None
        self._latest_center_m: Optional[np.ndarray] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_confidence: Optional[np.ndarray] = None
        self._latest_valid = False
        self._latest_reason = "waiting_for_metric_wrist_pose"
        self._sequence = -1
        self._stamp = float("nan")
        self._presence_generation = -1
        self._orientation_source = ""
        self._forearm_applied = False
        self._forearm_confidence = 0.0
        self._forearm_fusion_weight = 0.0
        self._forearm_status = "unavailable"
        self._delta = self._invalid_locked(self._latest_reason)

    @property
    def calibrated(self) -> bool:
        with self._lock:
            return self._zero_center_m is not None and self._zero_frame is not None

    def clear_zero(self, reason: str = "zero_cleared") -> None:
        with self._lock:
            self._zero_center_m = None
            self._zero_frame = None
            self._latest_reason = str(reason)
            self._delta = self._compute_locked()

    def calibrate_from_latest(
        self, expected_presence_generation: Optional[int] = None
    ) -> bool:
        with self._lock:
            if (
                not self._latest_valid
                or self._latest_center_m is None
                or self._latest_frame is None
                or (
                    expected_presence_generation is not None
                    and int(expected_presence_generation)
                    != self._presence_generation
                )
            ):
                return False
            self._zero_center_m = self._latest_center_m.copy()
            self._zero_frame = self._latest_frame.copy()
            self._latest_reason = "ok"
            self._delta = self._compute_locked()
            return True

    def update_from_packet(
        self, packet: Dict[str, Any], presence_generation: int
    ) -> WristPoseDelta:
        try:
            center = np.asarray(packet["wrist_position_m"], dtype=np.float64)
            frame = np.asarray(
                packet["palm_rotation_row_major"], dtype=np.float64
            ).reshape(3, 3)
            confidence = np.asarray(packet["confidence"], dtype=np.float64)
            sequence = int(packet["sequence"])
            stamp = float(packet["stamp"])
            orientation_source = str(packet.get("orientation_source", ""))
            fusion = packet.get("forearm_fusion") or {}
            forearm = fusion.get("forearm") or {}
            forearm_applied = bool(fusion.get("applied", False))
            forearm_confidence = float(forearm.get("confidence", 0.0))
            forearm_fusion_weight = float(fusion.get("fusion_weight", 0.0))
            forearm_status = str(
                forearm.get("status", fusion.get("fallback", "unavailable"))
            )
        except Exception as exc:
            return self.invalidate(
                "invalid_pose_packet:{}:{}".format(type(exc).__name__, exc),
                presence_generation,
            )
        if (
            center.shape != (3,)
            or confidence.shape != (6,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(confidence))
            or not math.isfinite(stamp)
        ):
            return self.invalidate(
                "pose_packet_contains_non_finite_values", presence_generation
            )
        try:
            rotation = project_to_so3(frame)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return self.invalidate(
                "invalid_wrist_rotation:{}".format(exc), presence_generation
            )

        with self._lock:
            generation = int(presence_generation)
            # Presence generation identifies tracking continuity, not the
            # operator's explicit C-zero contract.  A temporary no-hand gap
            # invalidates current measurements but must not force re-zeroing.
            self._presence_generation = generation
            self._latest_center_m = center.copy()
            self._latest_frame = rotation.copy()
            self._latest_confidence = np.clip(confidence, 0.0, 1.0)
            self._latest_valid = True
            self._latest_reason = "ok"
            self._sequence = sequence
            self._stamp = stamp
            self._orientation_source = orientation_source
            self._forearm_applied = forearm_applied
            self._forearm_confidence = float(
                np.clip(forearm_confidence, 0.0, 1.0)
            )
            self._forearm_fusion_weight = float(
                np.clip(forearm_fusion_weight, 0.0, 1.0)
            )
            self._forearm_status = forearm_status
            self._delta = self._compute_locked()
            return self._copy_delta(self._delta)

    def invalidate(
        self, reason: str, presence_generation: Optional[int] = None
    ) -> WristPoseDelta:
        with self._lock:
            if (
                presence_generation is not None
                and int(presence_generation) != self._presence_generation
            ):
                self._presence_generation = int(presence_generation)
            self._latest_valid = False
            self._latest_center_m = None
            self._latest_frame = None
            self._latest_confidence = None
            self._latest_reason = str(reason or "invalid_wrist_measurement")
            self._delta = self._invalid_locked(self._latest_reason)
            return self._copy_delta(self._delta)

    def snapshot(self, force_invalid_reason: str = "") -> WristPoseDelta:
        with self._lock:
            if force_invalid_reason:
                return self._copy_delta(self._invalid_locked(force_invalid_reason))
            return self._copy_delta(self._delta)

    def _compute_locked(self) -> WristPoseDelta:
        if (
            not self._latest_valid
            or self._latest_center_m is None
            or self._latest_frame is None
        ):
            return self._invalid_locked(self._latest_reason)
        calibrated = self._zero_center_m is not None and self._zero_frame is not None
        if not calibrated:
            return WristPoseDelta(
                True,
                False,
                "press_c_to_set_zero",
                self._latest_center_m.copy(),
                None,
                None if self._latest_confidence is None else self._latest_confidence.copy(),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                self._sequence,
                self._stamp,
                self._presence_generation,
                self._orientation_source,
                self._forearm_applied,
                self._forearm_confidence,
                self._forearm_fusion_weight,
                self._forearm_status,
            )
        assert self._zero_center_m is not None
        assert self._zero_frame is not None
        delta = self._latest_center_m - self._zero_center_m
        relative = project_to_so3(self._latest_frame @ self._zero_frame.T)
        yaw, pitch, roll = matrix_to_euler_zyx_deg(relative)
        return WristPoseDelta(
            True,
            True,
            "ok",
            self._latest_center_m.copy(),
            delta,
            None if self._latest_confidence is None else self._latest_confidence.copy(),
            float(yaw),
            float(pitch),
            float(roll),
            rotation_geodesic_deg(relative),
            self._sequence,
            self._stamp,
            self._presence_generation,
            self._orientation_source,
            self._forearm_applied,
            self._forearm_confidence,
            self._forearm_fusion_weight,
            self._forearm_status,
        )

    def _invalid_locked(self, reason: str) -> WristPoseDelta:
        calibrated = self._zero_center_m is not None and self._zero_frame is not None
        return WristPoseDelta(
            False,
            calibrated,
            str(reason),
            None,
            None,
            None,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            self._sequence,
            self._stamp,
            self._presence_generation,
            self._orientation_source,
            False,
            0.0,
            0.0,
            "measurement_invalid",
        )

    @staticmethod
    def _copy_delta(delta: WristPoseDelta) -> WristPoseDelta:
        return WristPoseDelta(
            delta.valid,
            delta.calibrated,
            delta.reason,
            None if delta.center_m is None else delta.center_m.copy(),
            None if delta.delta_m is None else delta.delta_m.copy(),
            None if delta.confidence is None else delta.confidence.copy(),
            delta.yaw_deg,
            delta.pitch_deg,
            delta.roll_deg,
            delta.rotation_geodesic_deg,
            delta.sequence,
            delta.stamp,
            delta.presence_generation,
            delta.orientation_source,
            delta.forearm_applied,
            delta.forearm_confidence,
            delta.forearm_fusion_weight,
            delta.forearm_status,
        )


def draw_hand_pose_panel(
    image: np.ndarray, delta: WristPoseDelta, panel_width: int = 410
) -> np.ndarray:
    """Append an ASCII diagnostics panel without covering the MANO rendering."""

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("pose panel requires a uint8 BGR image")
    width_extra = max(360, int(panel_width))
    height, width = source.shape[:2]
    output = np.full((height, width + width_extra, 3), (18, 18, 18), dtype=np.uint8)
    output[:, :width] = source
    cv2.line(output, (width, 0), (width, height - 1), (90, 90, 90), 1)
    x = width + 12
    y = 25

    def line(text: str, color: Sequence[int] = (220, 220, 220), gap: int = 23) -> None:
        nonlocal y
        cv2.putText(
            output,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            tuple(int(item) for item in color),
            1,
            cv2.LINE_AA,
        )
        y += gap

    line("HAND 6D POSE + GAZEBO CONTROL", (80, 230, 255), 25)
    line(
        (
            "REF: MANO WRIST + RGB-D FOREARM"
            if delta.forearm_applied
            else "REF: MANO WRIST (forearm fallback)"
        ),
        (190, 190, 190),
        21,
    )
    line("CAM: +X RIGHT  +Y DOWN  +Z FORWARD", (190, 190, 190), 28)
    if not delta.valid:
        line("MEASUREMENT: INVALID", (60, 80, 255), 24)
        line("reason: " + delta.reason[:43], (150, 150, 150), 24)
        line("No stale XYZ or rotation is reused.", (150, 150, 150), 38)
        line(
            "CONTROL: PAUSED (C-ZERO PRESERVED)"
            if delta.calibrated
            else "CONTROL: LOCKED - C-ZERO NOT SET",
            (90, 245, 150) if delta.calibrated else (60, 210, 255),
            24,
        )
        line("Show a complete hand and wrist.", (60, 210, 255), 23)
        line(
            "C = re-zero after pose becomes valid"
            if delta.calibrated
            else "C = set zero after pose becomes valid",
            (200, 200, 200),
            23,
        )
        line("R = reacquire ROI   Q/Esc = exit", (200, 200, 200), 23)
        return output

    assert delta.center_m is not None
    absolute_mm = 1000.0 * delta.center_m
    line("MEASUREMENT: VALID", (80, 255, 120), 24)
    line(
        "ABS XYZ mm  %+7.1f  %+7.1f  %+7.1f" % tuple(absolute_mm.tolist()),
        (235, 235, 235),
        24,
    )
    if delta.confidence is not None:
        line(
            "CONF pos %.2f  rot %.2f"
            % (float(np.min(delta.confidence[:3])), float(np.min(delta.confidence[3:]))),
            (160, 220, 220),
            28,
        )
    line(
        (
            "FOREARM FUSED conf %.2f weight %.2f"
            % (delta.forearm_confidence, delta.forearm_fusion_weight)
            if delta.forearm_applied
            else "FOREARM MANO-ONLY  " + delta.forearm_status[:24]
        ),
        (90, 245, 150) if delta.forearm_applied else (80, 190, 255),
        24,
    )
    if not delta.calibrated or delta.delta_m is None:
        line("GAZEBO CONTROL: LOCKED", (60, 80, 255), 25)
        line("RELATIVE POSE: ZERO NOT SET", (60, 210, 255), 25)
        line("Hold wrist + OPEN fingers steady; press C", (60, 210, 255), 38)
        line("C = SET ZERO", (230, 230, 230), 24)
        line("R = reacquire ROI   Q/Esc = exit", (190, 190, 190), 24)
        return output

    relative_mm = 1000.0 * delta.delta_m
    line("GAZEBO CONTROL: ENABLED", (80, 255, 120), 25)
    line("RELATIVE TO C-ZERO", (80, 230, 255), 25)
    line("dX RIGHT(+)/LEFT(-)    %+8.1f mm" % relative_mm[0], gap=22)
    line("dY DOWN(+)/UP(-)       %+8.1f mm" % relative_mm[1], gap=22)
    line("dZ FORWARD(+)          %+8.1f mm" % relative_mm[2], gap=28)
    line("PITCH about camera Y   %+8.2f deg" % delta.pitch_deg, (100, 245, 160), 22)
    line("YAW   about camera Z   %+8.2f deg" % delta.yaw_deg, (100, 245, 160), 22)
    line("ROLL  about camera X   %+8.2f deg" % delta.roll_deg, (100, 245, 160), 22)
    line("SO(3) total rotation   %8.2f deg" % delta.rotation_geodesic_deg, (100, 245, 160), 28)
    line("C = RE-ZERO (OPEN steady fingers)", (230, 230, 230), 22)
    line("R = reacquire ROI   Q/Esc = exit", (190, 190, 190), 22)
    return output


__all__ = [
    "RelativeWristPoseDisplay",
    "WristPoseDelta",
    "draw_hand_pose_panel",
    "matrix_to_euler_zyx_deg",
    "project_to_so3",
    "rotation_geodesic_deg",
]
