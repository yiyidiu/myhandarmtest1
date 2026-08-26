#!/usr/bin/env python3
"""Live D455 RGB -> ROI -> HaMeR -> MANO mesh display and palm frames.

The capture worker owns a one-element overwrite mailbox.  If HaMeR is slower
than the D455, old frames are dropped before inference instead of queued.
Normal live mode never saves images.  ``--experiment`` explicitly enables the
requested RGB/aligned-depth/JSONL/overlay-video development recording.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = PACKAGE_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MEDIAPIPE_ENV_PREFIX = Path(os.environ.get(
    "MEDIAPIPE_ENV_PREFIX",
    Path.home() / "anaconda3/envs/mediapipe_env",
))
DEFAULT_MEDIAPIPE_PYTHON = os.environ.get(
    "MEDIAPIPE_PYTHON", str(MEDIAPIPE_ENV_PREFIX / "bin/python")
)
DEFAULT_REALSENSE_SITE_PACKAGES = os.environ.get(
    "REALSENSE_SITE_PACKAGES",
    str(MEDIAPIPE_ENV_PREFIX / "lib/python3.10/site-packages"),
)

from perception_hamer.src.d455_capture import D455Capture
from perception_hamer.src.active_hand_selector import AutomaticActiveHandSelector
from perception_hamer.src.causal_wrist_so3_filter import (
    CausalWristSO3Filter,
    CausalWristSO3FilterConfig,
)
from perception_hamer.src.crop_quality import bbox_crop_quality
from perception_hamer.src.forearm_fusion import (
    CausalForearmEstimator,
    ForearmFusionConfig,
    apply_forearm_fusion_to_packet,
)
from perception_hamer.src.hamer_crop_inference import HamerCropInference
from perception_hamer.src.hand_detection_gate import (
    ConsecutiveHandDetectionGate,
    ContinuousHandPresenceGate,
)
from perception_hamer.src.hand_pose_overlay import (
    RelativeWristPoseDisplay,
    draw_hand_pose_panel,
)
from perception_hamer.src.hamer_palm_frame import RobustBetasCalibrator
from perception_hamer.src.live_display import LiveDisplay
from perception_hamer.src.mano_wrist_reference import (
    build_mano_wrist_definition,
    estimate_mano_wrist_frame,
)
from perception_hamer.src.realtime_hamer_pipeline import (
    LatestFrameSlot,
    LiveFramePacket,
    build_live_palm_estimates,
    draw_mano_mesh_overlay,
    hand_bbox_alignment,
    normalized_crop_points_to_original,
    project_hamer_vertices_to_original,
    remap_points_between_bboxes,
)
from perception_hamer.src.roi_provider import KLTTrackerROIProvider
from perception_hamer.src.teleop_pose_packet import (
    build_invalid_teleop_packet,
    build_live_teleop_packet,
)
from perception_hamer.src.teleop_control_gate import TeleopControlGate
from perception_hamer.src.teleoperation_core_mano_renderer import (
    TeleoperationCoreRenderFrame,
    render_inference_frame,
)


EXPERIMENTS = (
    "DEV_HAMER_STATIC",
    "DEV_HAMER_TRANSLATION",
    "DEV_HAMER_OPEN_CLOSE",
)
SKELETON = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


class ProcessSingleton:
    """Prevent a second HaMeR checkpoint from exhausting the 6 GiB GPU."""

    def __init__(self, name: str = "handarm_d455_hamer_live") -> None:
        self.path = Path("/tmp") / "{}_uid{}.lock".format(name, os.getuid())
        self._file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.seek(0)
            existing_pid = self._file.read().strip() or "unknown"
            self._file.close()
            raise RuntimeError(
                "another D455/HaMeR live process is already running (pid={}); "
                "use its existing window/terminal instead of loading a second model".format(
                    existing_pid
                )
            )
        self._file.seek(0)
        self._file.truncate()
        self._file.write(str(os.getpid()))
        self._file.flush()

    def close(self) -> None:
        if self._file.closed:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hand_preflight_timeout_diagnostics(
    last_reason: str,
    attempts: int,
    detection_results: int,
    wait_for_hand_s: float,
) -> Dict[str, Any]:
    """Describe a bounded hand-presence wait without relying on a seed ROI."""
    return {
        "valid": False,
        "reason": str(last_reason),
        "attempts": int(attempts),
        "detection_results": int(detection_results),
        "wait_for_hand_s": float(wait_for_hand_s),
    }


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if len(finite) == 0 else float(np.percentile(finite, percentile))


class GPUMemorySampler:
    def __init__(self, period_s: float = 0.2) -> None:
        self.period_s = float(period_s)
        self.peak_used_mib: Optional[int] = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2.0,
                )
                used = int(output.strip().splitlines()[0])
                self.peak_used_mib = (
                    used if self.peak_used_mib is None else max(self.peak_used_mib, used)
                )
                self.samples += 1
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class PendingReinitialization:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[tuple] = None

    def set(self, bbox: Sequence[float], is_right: Optional[bool] = None) -> None:
        with self._lock:
            self._value = (
                "seed",
                np.asarray(bbox, dtype=np.float64).copy(),
                None if is_right is None else bool(is_right),
            )

    def request_reset(self) -> None:
        """Make the next camera frame explicitly lose its ROI."""

        with self._lock:
            self._value = ("reset", None, None)

    def pop(self) -> Optional[tuple]:
        with self._lock:
            value = self._value
            self._value = None
            return value


class LatestDisplayOverlay:
    """Thread-safe latest HaMeR result and its exact-frame rendered pair."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = (
            None, None, 0.0, "HaMeR initializing", None, None, None, 0,
        )

    def update(
        self, result: Any, estimates: Any, processed_fps: float,
        failure_reason: str, mano_faces: Any, projection_source_bbox: Any,
        teleoperation_core_render: Optional[TeleoperationCoreRenderFrame],
        presence_generation: int,
    ) -> None:
        with self._lock:
            self._value = (
                result, estimates, float(processed_fps), str(failure_reason), mano_faces,
                (
                    None
                    if projection_source_bbox is None
                    else np.asarray(projection_source_bbox, dtype=np.float64).copy()
                ),
                teleoperation_core_render,
                int(presence_generation),
            )

    def snapshot(self) -> tuple:
        with self._lock:
            return self._value


def _capture_worker(
    capture: D455Capture,
    tracker: KLTTrackerROIProvider,
    slot: LatestFrameSlot,
    preview_slot: Optional[LatestFrameSlot],
    pending: PendingReinitialization,
    async_detector: Optional["AsyncMediaPipeDetection"],
    stop: threading.Event,
) -> None:
    sequence = 0
    last_detector_confirmation_serial = -1
    try:
        while not stop.is_set():
            frame = capture.wait_for_frame()
            presence = None
            if async_detector is not None:
                async_detector.submit(frame.rgb)
                presence = async_detector.presence_snapshot()
            fresh_seed = pending.pop()
            if presence is not None and not presence["valid"]:
                # A KLT box is never evidence of a hand.  Fail closed even if
                # an older main-loop reinitialization request races with the
                # newest MediaPipe no-hand result.
                tracker.reset()
                roi = tracker.update(frame.rgb)
            elif fresh_seed is not None:
                action, fresh_bbox, fresh_is_right = fresh_seed
                if action == "reset":
                    tracker.reset()
                    roi = tracker.update(frame.rgb)
                else:
                    try:
                        roi = tracker.reinitialize(
                            frame.rgb, fresh_bbox, fresh_is_right
                        )
                    except Exception:
                        # Reacquisition must fail closed without killing the D455
                        # capture thread.  The main loop will run the detector again.
                        roi = tracker.update(frame.rgb)
            elif (
                presence is not None
                and presence.get("confirmed_detection") is not None
                and int(presence.get("confirmation_serial", -1))
                > last_detector_confirmation_serial
            ):
                # Continuously anchor optical flow to the independently
                # detected hand.  Without this correction KLT can remain
                # numerically valid while drifting to a face/background, which
                # made HaMeR draw a hand in empty image space.
                confirmed = presence["confirmed_detection"]
                last_detector_confirmation_serial = int(
                    presence["confirmation_serial"]
                )
                try:
                    roi = tracker.reinitialize(
                        frame.rgb,
                        confirmed["bbox"],
                        bool(confirmed["is_right"]),
                    )
                except Exception:
                    roi = tracker.update(frame.rgb)
            else:
                roi = tracker.update(frame.rgb)
            packet = LiveFramePacket(frame=frame, roi=roi, capture_sequence=sequence)
            slot.publish(packet)
            if preview_slot is not None:
                preview_slot.publish(packet)
            sequence += 1
    except BaseException as exc:
        if not stop.is_set():
            slot.close(exc)
            if preview_slot is not None:
                preview_slot.close(exc)
    finally:
        slot.close()
        if preview_slot is not None:
            preview_slot.close()


def _select_bbox(rgb: np.ndarray) -> np.ndarray:
    helper = SCRIPT_DIR / "manual_select_roi_once.py"
    completed = subprocess.run(
        [DEFAULT_MEDIAPIPE_PYTHON, str(helper), "--width", str(rgb.shape[1]),
         "--height", str(rgb.shape[0])],
        input=rgb.tobytes(), capture_output=True, timeout=120.0,
    )
    try:
        payload = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError("manual ROI sidecar returned invalid output") from exc
    if completed.returncode != 0 or not payload.get("valid"):
        raise RuntimeError("manual ROI selection was cancelled")
    return np.asarray(payload["bbox"], dtype=np.float64)


def _detect_bbox_with_mediapipe_sidecar(
    rgb: np.ndarray, minimum_detection_confidence: float = 0.45
) -> Dict[str, Any]:
    helper = SCRIPT_DIR / "mediapipe_detect_roi_once.py"
    completed = subprocess.run(
        [DEFAULT_MEDIAPIPE_PYTHON, str(helper), "--width", str(rgb.shape[1]),
         "--height", str(rgb.shape[0]), "--min-detection-confidence",
         str(float(minimum_detection_confidence))],
        input=rgb.tobytes(), capture_output=True, timeout=15.0,
    )
    try:
        payload = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe ROI sidecar returned invalid output: "
            + completed.stderr.decode("utf-8", errors="replace")[-500:]
        ) from exc
    if completed.returncode not in (0, 2):
        raise RuntimeError("MediaPipe ROI sidecar failed: " + repr(payload))
    return payload


class MediaPipeDetectionSidecar:
    """Persistent 2-D detector; avoids reloading MediaPipe for every frame."""

    def __init__(self, width: int, height: int, minimum_confidence: float) -> None:
        helper = SCRIPT_DIR / "mediapipe_detect_roi_once.py"
        self.width = int(width)
        self.height = int(height)
        self._process = subprocess.Popen(
            [
                DEFAULT_MEDIAPIPE_PYTHON, "-u", str(helper), "--width", str(self.width),
                "--height", str(self.height), "--min-detection-confidence",
                str(float(minimum_confidence)), "--stream",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def detect(self, rgb: np.ndarray) -> Dict[str, Any]:
        frame = np.asarray(rgb)
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError("MediaPipe sidecar requires uint8 RGB frame")
        process = self._process
        if process.poll() is not None or process.stdin is None or process.stdout is None:
            raise RuntimeError("persistent MediaPipe sidecar exited")
        try:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            line = process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("persistent MediaPipe sidecar pipe failed") from exc
        if not line:
            raise RuntimeError("persistent MediaPipe sidecar returned EOF")
        return json.loads(line.decode("utf-8"))

    def close(self) -> None:
        process = self._process
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)


class AsyncMediaPipeDetection:
    """Latest-only worker so 2-D detection never blocks the camera preview."""

    def __init__(
        self,
        sidecar: MediaPipeDetectionSidecar,
        hand_presence_timeout_s: float = 0.25,
        confirmation_frames: int = 2,
        negative_grace_frames: int = 1,
    ) -> None:
        self._sidecar = sidecar
        self._condition = threading.Condition()
        self._frame = None
        self._input_version = 0
        self._consumed_version = 0
        self._result_version = 0
        self._result = None
        self._result_monotonic: Optional[float] = None
        self._presence_timeout_s = float(hand_presence_timeout_s)
        self._confirmation_frames = max(2, int(confirmation_frames))
        self._presence = ContinuousHandPresenceGate(
            required_frames=self._confirmation_frames,
            minimum_iou=0.25,
            timeout_s=self._presence_timeout_s,
            negative_grace_frames=int(negative_grace_frames),
            negative_grace_s=min(0.08, 0.5 * self._presence_timeout_s),
        )
        self._active_hand_selector = AutomaticActiveHandSelector(
            switch_frames=3
        )
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="mediapipe-latest-only", daemon=True
        )
        self._thread.start()

    def submit(self, rgb: np.ndarray) -> int:
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("MediaPipe input must be uint8 RGB")
        with self._condition:
            # RealSense-backed arrays must not be read after their next SDK
            # frame replaces the underlying buffer.
            self._frame = np.ascontiguousarray(frame).copy()
            self._input_version += 1
            self._condition.notify_all()
            return self._input_version

    def latest_after(self, previous_version: int) -> tuple:
        with self._condition:
            if self._result_version <= int(previous_version):
                return int(previous_version), None, None
            return (
                self._result_version,
                self._result,
                self._result_monotonic,
            )

    def presence_snapshot(self) -> Dict[str, Any]:
        """Return current fail-closed presence state, applying timeout."""

        with self._condition:
            state = self._presence.snapshot()
            state["detector_result_version"] = int(self._result_version)
            state["detector_result_monotonic"] = self._result_monotonic
            state["active_hand_is_right"] = (
                self._active_hand_selector.active_is_right
            )
            state["active_hand_generation"] = int(
                self._active_hand_selector.active_hand_generation
            )
            return state

    def _run(self) -> None:
        while True:
            with self._condition:
                while (self._input_version <= self._consumed_version
                       and not self._stopping):
                    self._condition.wait()
                if self._stopping:
                    return
                frame = self._frame
                version = self._input_version
                self._consumed_version = version
            try:
                result = self._sidecar.detect(frame)
            except Exception as exc:
                result = {
                    "valid": False,
                    "reason": "detector_sidecar_error:{}:{}".format(
                        type(exc).__name__, exc
                    ),
                }
            result_monotonic = time.monotonic()
            with self._condition:
                result = self._active_hand_selector.select(result)
                self._result = result
                self._result_version = version
                self._result_monotonic = result_monotonic
                self._presence.observe(result, result_monotonic)

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=3.0)


def _draw_axes(
    image: np.ndarray, origin: Sequence[float], rotation: Any, label: str, y_offset: int
) -> None:
    matrix = np.asarray(rotation, dtype=np.float64)
    point = np.asarray(origin, dtype=np.float64)
    if matrix.shape != (3, 3) or point.shape != (2,) or not np.all(np.isfinite(matrix)):
        return
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    start = tuple(np.rint(point).astype(int))
    for axis, color in enumerate(colors):
        direction = np.array([matrix[0, axis], matrix[1, axis]])
        norm = float(np.linalg.norm(direction))
        if norm > 1e-8:
            end = tuple(np.rint(point + 32.0 * direction / norm).astype(int))
            cv2.arrowedLine(image, start, end, color, 2, tipLength=0.25)
    cv2.putText(image, label, (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)


def _roi_detector_alignment(roi: Any, presence: Dict[str, Any]) -> dict:
    if not presence.get("valid"):
        return {
            "valid": False,
            "iou": 0.0,
            "normalized_center_distance": float("inf"),
            "reason": "no_real_hand",
        }
    confirmed = presence.get("confirmed_detection")
    if roi.lost or roi.bbox is None or confirmed is None:
        return {
            "valid": False,
            "iou": 0.0,
            "normalized_center_distance": float("inf"),
            "reason": "roi_or_confirmed_detection_missing",
        }
    return hand_bbox_alignment(
        roi.bbox,
        confirmed.get("bbox"),
        minimum_iou=0.05,
        maximum_normalized_center_distance=0.75,
    )


def _make_teleoperation_core_overlay(
    rgb: np.ndarray,
    roi: Any,
    result: Any,
    estimates: Optional[Dict[str, Any]],
    processed_fps: float,
    failure_reason: str,
    mano_faces: Optional[np.ndarray],
    reference_render: Optional[TeleoperationCoreRenderFrame],
    hand_presence_valid: bool,
    roi_matches_detected_hand: bool,
    active_hand_is_right: Optional[bool],
    ignored_non_active_hand_count: int,
) -> np.ndarray:
    """Compose the archive's exact inference-frame source/MANO pair.

    Both panels come from the same frame that produced ``pred_vertices``.
    Before the first valid inference, and immediately after the independent
    presence gate becomes invalid, two copies of the current D455 image are
    shown with no synthetic hand.  No old mesh is warped onto a new frame.
    """

    live_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    can_show_reference = bool(
        hand_presence_valid
        and roi_matches_detected_hand
        and result is not None
        and reference_render is not None
    )
    if can_show_reference:
        canvas = reference_render.side_by_side_bgr().copy()
    else:
        source_panel = live_bgr.copy()
        overlay_panel = live_bgr.copy()
        if (
            hand_presence_valid
            and roi_matches_detected_hand
            and not roi.lost
            and roi.bbox is not None
        ):
            x1, y1, x2, y2 = np.rint(roi.bbox).astype(int)
            for panel in (source_panel, overlay_panel):
                cv2.rectangle(panel, (x1, y1), (x2, y2), (70, 255, 70), 2)
        canvas = np.hstack((source_panel, overlay_panel))

    control_reference_label = "UNAVAILABLE"
    if estimates is not None:
        control = estimates.get("control_wrist_frame")
        if control is not None:
            control_reference_label = str(
                control.get("reference_kind", "MANO_JOINT_0_PALM_FRAME_LEGACY")
            )
        if (
            can_show_reference
            and control is not None
            and control.get("valid")
            and control_reference_label == "MANO_WRIST_RING_16"
        ):
            try:
                loop = np.asarray(
                    control["quality"]["wrist_loop_vertex_indices"],
                    dtype=np.int64,
                )
                canonical_ring = result.pred_vertices_mano_right_canonical[loop]
                center_pixel, _ = project_hamer_vertices_to_original(
                    canonical_ring.mean(axis=0, keepdims=True),
                    result.hamer_crop_projection_translation,
                    result.hamer_nominal_crop_focal_length,
                    result.quality["affine_original_to_crop"],
                    256,
                )
                center = np.asarray(center_pixel[0], dtype=np.float64)
                rotation = np.asarray(control["rotation"], dtype=np.float64)
                colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
                for panel_offset in (0, int(rgb.shape[1])):
                    origin = center + np.asarray([panel_offset, 0.0])
                    start = tuple(np.rint(origin).astype(int))
                    cv2.circle(canvas, start, 5, (255, 255, 255), -1)
                    cv2.circle(canvas, start, 7, (0, 0, 0), 1)
                    for axis, color in enumerate(colors):
                        direction = rotation[:2, axis]
                        norm = float(np.linalg.norm(direction))
                        if norm > 1.0e-8:
                            end = tuple(np.rint(
                                origin + 35.0 * direction / norm
                            ).astype(int))
                            cv2.arrowedLine(
                                canvas, start, end, color, 2, tipLength=0.24
                            )
                    cv2.putText(
                        canvas,
                        "16-point wrist centre",
                        (start[0] + 8, start[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            except Exception:
                # Display diagnostics never bypass the control packet's strict
                # validation and never keep a stale reference marker.
                pass

    panel_width = int(rgb.shape[1])
    cv2.putText(
        canvas,
        "HaMeR inference RGB",
        (8, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "teleoperation_ubuntu_core MANO 778 vertices / 1538 faces",
        (panel_width + 8, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    inference_ms = (
        0.0 if result is None else 1000.0 * float(result.inference_time_s)
    )
    status_lines = [
        "renderer=TELEOPERATION_CORE_EXACT_FRAME (no KLT mesh warp)",
        "REAL HAND presence={}  ROI/detector match={}".format(
            "YES" if hand_presence_valid else "NO",
            "YES" if roi_matches_detected_hand else "NO",
        ),
        "ACTIVE HAND={} other_hand_ignored={}".format(
            (
                "AUTO/UNKNOWN"
                if active_hand_is_right is None
                else ("RIGHT" if active_hand_is_right else "LEFT")
            ),
            int(ignored_non_active_hand_count),
        ),
        "HaMeR valid={} {:.1f} ms  mesh Hz={:.2f}".format(
            result is not None, inference_ms, float(processed_fps)
        ),
        "MANO mesh={} vertices={} faces={}".format(
            "ON" if can_show_reference else "OFF",
            0 if result is None else len(result.pred_vertices_mano_right_canonical),
            0 if mano_faces is None else len(mano_faces),
        ),
        "CONTROL REF={} (white point; RGB axes)".format(
            control_reference_label
        ),
    ]
    if estimates is not None:
        filter_diagnostics = estimates.get("palm_orientation_filter") or {}
        innovation = filter_diagnostics.get("innovation_deg")
        status_lines.append(
            "SO3 {} status={} innovation={} gain={:.2f}".format(
                str(filter_diagnostics.get(
                    "large_angle_mode", "unknown"
                )).upper(),
                str(filter_diagnostics.get("status", "unavailable")),
                (
                    "n/a"
                    if innovation is None
                    else "{:.1f}deg".format(float(innovation))
                ),
                float(filter_diagnostics.get("gain", 0.0)),
            )
        )
    if failure_reason:
        status_lines.append("failure=" + failure_reason[:100])
    for index, line in enumerate(status_lines):
        error_line = (
            "mesh=OFF" in line
            or line.startswith("failure=")
            or "presence=NO" in line
            or "match=NO" in line
        )
        cv2.putText(
            canvas,
            line,
            (8, 22 + index * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 100, 255) if error_line else (80, 255, 80),
            1,
            cv2.LINE_AA,
        )
    return canvas


def make_overlay(
    rgb: np.ndarray,
    roi: Any,
    result: Any,
    estimates: Optional[Dict[str, Any]],
    processed_fps: float,
    failure_reason: str,
    mano_faces: Optional[np.ndarray] = None,
    projection_source_bbox: Optional[np.ndarray] = None,
    hand_presence_valid: bool = True,
    roi_matches_detected_hand: bool = True,
    active_hand_is_right: Optional[bool] = None,
    ignored_non_active_hand_count: int = 0,
    teleoperation_core_render: Optional[TeleoperationCoreRenderFrame] = None,
    mesh_renderer: str = "teleoperation-core",
) -> np.ndarray:
    if not hand_presence_valid or not roi_matches_detected_hand:
        # Never reuse a previously inferred mesh or an optical-flow crop when
        # the independent MediaPipe presence detector says the image has no
        # real hand.
        result = None
        estimates = None
        teleoperation_core_render = None
    if mesh_renderer == "teleoperation-core":
        return _make_teleoperation_core_overlay(
            rgb,
            roi,
            result,
            estimates,
            processed_fps,
            failure_reason,
            mano_faces,
            teleoperation_core_render,
            hand_presence_valid,
            roi_matches_detected_hand,
            active_hand_is_right,
            ignored_non_active_hand_count,
        )
    if mesh_renderer != "legacy-depth":
        raise ValueError("unknown mesh renderer: " + str(mesh_renderer))

    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if hand_presence_valid and roi_matches_detected_hand and roi.bbox is not None:
        x1, y1, x2, y2 = np.rint(roi.bbox).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
    joints_px = None
    mesh_drawn = False
    render_failure = ""
    if result is not None:
        try:
            if mano_faces is not None:
                vertices_px, vertex_depth = project_hamer_vertices_to_original(
                    result.pred_vertices_mano_right_canonical,
                    result.hamer_crop_projection_translation,
                    result.hamer_nominal_crop_focal_length,
                    result.quality["affine_original_to_crop"],
                    256,
                )
                if projection_source_bbox is not None and roi.bbox is not None:
                    vertices_px = remap_points_between_bboxes(
                        vertices_px, projection_source_bbox, roi.bbox
                    )
                canvas = draw_mano_mesh_overlay(
                    canvas, vertices_px, vertex_depth, mano_faces)
                mesh_drawn = True
            joints_px = normalized_crop_points_to_original(
                result.pred_keypoints_2d_crop_normalized,
                result.quality["affine_original_to_crop"],
                256,
            )
            if projection_source_bbox is not None and roi.bbox is not None:
                joints_px = remap_points_between_bboxes(
                    joints_px, projection_source_bbox, roi.bbox
                )
            for first, second in SKELETON:
                cv2.line(canvas, tuple(np.rint(joints_px[first]).astype(int)),
                         tuple(np.rint(joints_px[second]).astype(int)), (255, 210, 0), 1)
            for point in joints_px:
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (255, 255, 0), -1)
        except Exception as exc:
            joints_px = None
            render_failure = "{}:{}".format(type(exc).__name__, exc)
    if estimates is not None and joints_px is not None:
        origin = joints_px[0]
        raw = estimates["raw_global_orient"]
        control = estimates.get(
            "control_wrist_frame", estimates["mano_joint_palm_frame"]
        )
        if control.get("reference_kind") == "MANO_WRIST_RING_16":
            try:
                loop = np.asarray(
                    control["quality"]["wrist_loop_vertex_indices"],
                    dtype=np.int64,
                )
                ring = result.pred_vertices_mano_right_canonical[loop]
                projected_center, _ = project_hamer_vertices_to_original(
                    ring.mean(axis=0, keepdims=True),
                    result.hamer_crop_projection_translation,
                    result.hamer_nominal_crop_focal_length,
                    result.quality["affine_original_to_crop"],
                    256,
                )
                origin = projected_center[0]
            except Exception:
                pass
        if raw.get("valid"):
            _draw_axes(
                canvas, origin, raw["rotation"], "raw global_orient axes",
                canvas.shape[0] - 58,
            )
        if control.get("valid"):
            shifted = origin + np.array([0.0, 45.0])
            _draw_axes(
                canvas,
                shifted,
                control["rotation"],
                "CONTROL " + str(control.get("reference_kind", "legacy")),
                canvas.shape[0] - 37,
            )
    inference_ms = 0.0 if result is None else 1000.0 * result.inference_time_s
    status_lines = [
        "REAL HAND presence={}".format("YES" if hand_presence_valid else "NO"),
        "ACTIVE HAND={} other_hand_ignored={}".format(
            (
                "AUTO/UNKNOWN"
                if active_hand_is_right is None
                else ("RIGHT" if active_hand_is_right else "LEFT")
            ),
            int(ignored_non_active_hand_count),
        ),
        "ROI matches detected hand={}".format(
            "YES" if roi_matches_detected_hand else "NO"
        ),
        f"ROI {roi.source} valid={hand_presence_valid and roi_matches_detected_hand and not roi.lost} age={roi.age} conf={roi.confidence:.2f}",
        f"HaMeR valid={result is not None} {inference_ms:.1f} ms  FPS={processed_fps:.2f}",
        "MANO mesh={} vertices={} faces={}".format(
            "ON" if mesh_drawn else "OFF",
            0 if result is None else len(result.pred_vertices_mano_right_canonical),
            0 if mano_faces is None else len(mano_faces),
        ),
    ]
    if render_failure:
        status_lines.append("MANO render error=" + render_failure[:70])
    if failure_reason:
        status_lines.append("failure=" + failure_reason[:75])
    for index, line in enumerate(status_lines):
        line_reports_error = (
            "mesh=OFF" in line or "error=" in line or line.startswith("failure=")
        )
        color = (
            (80, 255, 80)
            if not failure_reason and not render_failure and not line_reports_error
            else (0, 100, 255)
        )
        cv2.putText(canvas, line, (8, 22 + index * 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return canvas


def _display_worker(
    display: LiveDisplay,
    preview_slot: LatestFrameSlot,
    overlay_state: LatestDisplayOverlay,
    pose_display: RelativeWristPoseDisplay,
    teleop_control_gate: TeleopControlGate,
    async_detector: Optional[AsyncMediaPipeDetection],
    stop: threading.Event,
    mesh_renderer: str,
) -> None:
    last_version = 0
    displayed = 0
    started = time.monotonic()
    while not stop.is_set():
        try:
            last_version, packet = preview_slot.get_after(last_version, timeout_s=1.0)
        except TimeoutError:
            continue
        except RuntimeError as exc:
            print("HaMeR preview stopped: {}".format(exc), file=sys.stderr, flush=True)
            break
        if packet is None:
            break
        (
            result, estimates, processed_fps, failure_reason, mano_faces,
            projection_source_bbox, teleoperation_core_render,
            overlay_presence_generation,
        ) = overlay_state.snapshot()
        hand_presence_valid = True
        roi_matches_detected_hand = True
        active_hand_is_right = None
        active_hand_generation = 0
        ignored_non_active_hand_count = 0
        if async_detector is not None:
            presence = async_detector.presence_snapshot()
            hand_presence_valid = bool(presence["valid"])
            active_hand_is_right = presence.get("active_hand_is_right")
            active_hand_generation = int(
                presence.get("active_hand_generation", 0)
            )
            confirmed_for_status = presence.get("confirmed_detection") or {}
            ignored_non_active_hand_count = int(
                confirmed_for_status.get("ignored_non_active_hand_count", 0)
            )
            alignment = _roi_detector_alignment(packet.roi, presence)
            roi_matches_detected_hand = bool(alignment["valid"])
            if not hand_presence_valid:
                result = None
                estimates = None
                projection_source_bbox = None
                teleoperation_core_render = None
                failure_reason = "no_real_hand:" + str(presence["reason"])
            elif int(overlay_presence_generation) != int(presence["generation"]):
                # The mesh belongs to an earlier continuous hand-presence
                # interval.  Keep it hidden until a new HaMeR result arrives.
                result = None
                estimates = None
                projection_source_bbox = None
                teleoperation_core_render = None
                failure_reason = "hand_reacquiring:new_presence_generation"
            elif not roi_matches_detected_hand:
                result = None
                estimates = None
                projection_source_bbox = None
                teleoperation_core_render = None
                failure_reason = (
                    "roi_detector_mismatch:iou={:.3f},center={:.3f}".format(
                        alignment["iou"],
                        alignment["normalized_center_distance"],
                    )
                )
        canvas = make_overlay(
            packet.frame.rgb, packet.roi, result, estimates,
            processed_fps, failure_reason, mano_faces,
            projection_source_bbox, hand_presence_valid,
            roi_matches_detected_hand,
            active_hand_is_right,
            ignored_non_active_hand_count,
            teleoperation_core_render,
            mesh_renderer,
        )
        pose_matches_visible_hand = bool(
            result is not None
            and estimates is not None
            and hand_presence_valid
            and roi_matches_detected_hand
        )
        if not pose_matches_visible_hand:
            teleop_control_gate.invalidate(
                "HAND_TRACKING_INVALID_REQUIRES_NEW_C"
            )
        else:
            teleop_control_gate.observe_identity(
                int(overlay_presence_generation),
                int(active_hand_generation),
                bool(result.is_right),
            )
        if display.pop_confirm_request():
            # C is the only live-teleoperation enable/re-zero action. Lock
            # first so an invalid visible pose cannot leave an older robot
            # reference active.
            teleop_control_gate.disable("OPERATOR_REQUESTED_NEW_C_REFERENCE")
            pose_display.clear_zero("operator_reference_requested")
            zero_set = bool(
                pose_matches_visible_hand
                and pose_display.calibrate_from_latest(
                    expected_presence_generation=overlay_presence_generation
                )
            )
            reference_epoch = (
                teleop_control_gate.confirm(
                    int(overlay_presence_generation),
                    int(active_hand_generation),
                    bool(result.is_right),
                )
                if zero_set else None
            )
            print(
                (
                    "6D wrist/robot control reference requested by C "
                    "(epoch {}); hold the hand steady until Gazebo starts "
                    "following.".format(reference_epoch)
                    if zero_set
                    else
                    "C reference rejected and robot control remains LOCKED: "
                    "current MANO/D455 pose is invalid."
                ),
                flush=True,
            )
        pose_delta = pose_display.snapshot(
            "display_pose_not_current:" + str(failure_reason or "invalid")
            if not pose_matches_visible_hand
            else ""
        )
        # Append, rather than overlay, so the existing exact-frame RGB/MANO
        # rendering remains pixel-for-pixel visible.
        canvas = draw_hand_pose_panel(canvas, pose_delta)
        displayed += 1
        display_hz = displayed/max(1.0e-9, time.monotonic()-started)
        cv2.putText(
            canvas, (
                "exact-pair display loop={:.1f} Hz (new MANO={:.1f} Hz)"
                if mesh_renderer == "teleoperation-core"
                else "D455 display={:.1f} Hz (HaMeR mesh={:.1f} Hz)"
            ).format(display_hz, processed_fps),
            (8, canvas.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 0), 1, cv2.LINE_AA,
        )
        if not display.show(canvas):
            print("HaMeR display sidecar exited.", file=sys.stderr, flush=True)
            break
        if display.stop_requested:
            stop.set()
            break


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--auto-roi-mediapipe", action="store_true",
                        help="continuous presence plus bbox/handedness preflight")
    parser.add_argument("--wait-for-hand-s", type=float, default=0.0,
                        help="maximum preflight wait; 0 waits until q/Esc")
    parser.add_argument(
        "--mediapipe-min-detection-confidence", type=float, default=0.65,
        help="continuous 2-D hand detector threshold",
    )
    parser.add_argument(
        "--hand-confirm-frames", type=int, default=3,
        help="consistent MediaPipe detections required before HaMeR is enabled",
    )
    parser.add_argument(
        "--hand-presence-timeout-s", type=float, default=0.25,
        help="hide ROI/MANO and stop output when MediaPipe has no fresh result",
    )
    parser.add_argument(
        "--hand-miss-grace-frames",
        type=int,
        default=1,
        help=(
            "isolated MediaPipe misses tolerated during fast motion; default "
            "1, the next miss hides MANO"
        ),
    )
    parser.add_argument(
        "--auto-confirm-hand", action="store_true",
        help="legacy alias; stable hand detection now starts automatically",
    )
    parser.add_argument(
        "--require-hand-confirmation", action="store_true",
        help="optional manual C/Enter gate; default automatically selects one active hand",
    )
    parser.add_argument(
        "--roi-reacquire-period-s", type=float, default=0.60,
        help="minimum time between MediaPipe ROI reacquisition attempts",
    )
    parser.add_argument(
        "--roi-smoothing-alpha", type=float, default=1.0,
        help=(
            "new-frame weight for causal KLT bbox smoothing; live teleoperation "
            "defaults to 1.0 (direct, no ROI smoothing)"
        ),
    )
    parser.add_argument(
        "--orientation-filter-time-constant-s", type=float, default=0.08,
        help="stationary causal SO(3) time constant; motion-adaptive gain preserves response",
    )
    parser.add_argument(
        "--orientation-filter-min-gain", type=float, default=0.08,
        help="minimum new-frame gain for noisy/low-quality stationary orientation",
    )
    parser.add_argument(
        "--orientation-filter-max-gain", type=float, default=1.0,
        help="maximum new-frame gain during clear intentional rotation",
    )
    parser.add_argument(
        "--orientation-filter-motion-start-deg", type=float, default=1.5,
        help="innovation where responsive motion gain starts",
    )
    parser.add_argument(
        "--orientation-filter-motion-full-deg", type=float, default=8.0,
        help="innovation where responsive motion gain reaches its quality-scaled maximum",
    )
    parser.add_argument(
        "--orientation-filter-soft-deg", type=float, default=25.0,
        help="SO(3) innovation above this angle receives a smaller gain",
    )
    parser.add_argument(
        "--orientation-filter-hard-deg", type=float, default=45.0,
        help=(
            "large-angle diagnostic threshold; follow mode passes it through, "
            "reject mode restores the former hard rejection"
        ),
    )
    parser.add_argument(
        "--orientation-filter-large-angle-mode",
        choices=("follow", "reject"),
        default="reject",
        help=(
            "reject fails closed on discontinuous MANO orientation jumps "
            "(default); follow is a diagnostic-only passthrough"
        ),
    )
    parser.add_argument(
        "--control-reference",
        choices=("mano-wrist-ring", "mano-joint-palm"),
        default="mano-wrist-ring",
        help=(
            "mano-wrist-ring uses the archive's robust 16-vertex wrist-opening "
            "centre/frame (default); mano-joint-palm keeps the former joint-0 "
            "and MCP-axis baseline"
        ),
    )
    parser.add_argument(
        "--forearm-fusion-max-weight",
        type=float,
        default=0.20,
        help=(
            "maximum low-weight RGB-D forearm anchor applied to the MANO "
            "longitudinal axis; roll always remains MANO-derived"
        ),
    )
    parser.add_argument(
        "--disable-forearm-fusion",
        action="store_true",
        help="diagnostic MANO-only mode; normal single-HaMeR command enables fusion",
    )
    parser.add_argument("--left-hand", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--display-backend", choices=("auto", "local", "sidecar"), default="auto",
        help="auto uses a GUI sidecar when HaMeR OpenCV is headless",
    )
    parser.add_argument(
        "--display-helper-python",
        default=DEFAULT_MEDIAPIPE_PYTHON,
        help="Python with a GUI-enabled OpenCV build for the display sidecar",
    )
    parser.add_argument("--no-mesh-overlay", action="store_true",
                        help="disable the MANO mesh overlay")
    parser.add_argument(
        "--mesh-renderer",
        choices=("teleoperation-core", "legacy-depth"),
        default="teleoperation-core",
        help=(
            "teleoperation-core uses the supplied archive's complete exact-frame "
            "MANO pair (default); legacy-depth keeps the former cross-frame display"
        ),
    )
    parser.add_argument("--duration-s", type=float, default=0.0,
                        help="0 means run until q/ESC outside experiment mode")
    parser.add_argument("--countdown-s", type=float, default=1.0)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--output-root", default=str(
        REPOSITORY_ROOT / "datasets/development_usb2/hamer_palm_stability"))
    parser.add_argument("--checkpoint", default=str(
        PACKAGE_DIR / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"))
    parser.add_argument("--data-root", default=str(PACKAGE_DIR / "_DATA"))
    parser.add_argument(
        "--realsense-sdk-site-packages",
        default=DEFAULT_REALSENSE_SITE_PACKAGES,
    )
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--teleop-udp-host", default="",
                        help="optional handarm_hamer_pose_v1 destination; empty disables UDP")
    parser.add_argument("--teleop-udp-port", type=int, default=5010)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.roi_smoothing_alpha <= 1.0:
        raise SystemExit("--roi-smoothing-alpha must be in (0, 1]")
    if args.orientation_filter_time_constant_s <= 0.0:
        raise SystemExit("--orientation-filter-time-constant-s must be > 0")
    if not (
        0.0 <= args.orientation_filter_min_gain
        <= args.orientation_filter_max_gain <= 1.0
    ):
        raise SystemExit(
            "orientation filter gains must satisfy 0 <= min <= max <= 1")
    if not (
        0.0 < args.orientation_filter_soft_deg
        < args.orientation_filter_hard_deg
        < 180.0
    ):
        raise SystemExit(
            "orientation filter thresholds must satisfy "
            "0 < soft-deg < hard-deg < 180"
        )
    if not (
        0.0 <= args.orientation_filter_motion_start_deg
        < args.orientation_filter_motion_full_deg
        < args.orientation_filter_hard_deg
    ):
        raise SystemExit(
            "orientation motion thresholds must satisfy "
            "0 <= start < full < hard-deg")
    if not 0.0 <= args.forearm_fusion_max_weight <= 0.35:
        raise SystemExit("--forearm-fusion-max-weight must be in [0, 0.35]")
    if not 0 <= args.hand_miss_grace_frames <= 2:
        raise SystemExit("--hand-miss-grace-frames must be 0, 1, or 2")
    if args.teleop_udp_host and args.no_display:
        raise SystemExit(
            "live teleoperation requires the camera window because C is the "
            "only control-enable/reference action"
        )
    try:
        process_singleton = ProcessSingleton()
    except RuntimeError as exc:
        raise SystemExit("HaMeR live startup refused: {}".format(exc))
    sdk_path = Path(args.realsense_sdk_site_packages)
    if sdk_path.is_dir() and str(sdk_path) not in sys.path:
        sys.path.append(str(sdk_path))
    if args.experiment and args.duration_s <= 0.0:
        args.duration_s = 25.0
    is_right = not args.left_hand
    runner = HamerCropInference(
        args.checkpoint,
        data_root=args.data_root,
        device="cuda:0",
        precision=args.precision,
        freeze_betas=False,
        source_frame="camera_color_optical_frame",
        timestamp_clock_domain="d455_device_global_time_ms",
    )
    asset_status = runner.asset_status()
    if not all(asset_status[key] for key in (
        "checkpoint_exists", "model_config_exists", "mano_right_exists",
        "mano_mean_params_exists")):
        raise SystemExit("HaMeR assets are incomplete: " + repr(asset_status))

    capture = D455Capture(width=640, height=480, fps=30, require_superspeed=False)
    output_dir: Optional[Path] = None
    records_file = None
    video = None
    stop = threading.Event()
    worker: Optional[threading.Thread] = None
    display_worker: Optional[threading.Thread] = None
    display: Optional[LiveDisplay] = None
    detector_sidecar: Optional[MediaPipeDetectionSidecar] = None
    async_detector: Optional[AsyncMediaPipeDetection] = None
    gpu_sampler = GPUMemorySampler()
    slot = LatestFrameSlot()
    preview_slot = LatestFrameSlot()
    overlay_state = LatestDisplayOverlay()
    pose_display = RelativeWristPoseDisplay()
    teleop_control_gate = TeleopControlGate()
    pending = PendingReinitialization()
    session_records = []
    previous_joint_quaternion = None
    previous_control_quaternion = None
    # Keep translation alive when only the monocular MANO orientation channel
    # fails its innovation gate.  This state is cleared on every real
    # no-hand/ROI discontinuity, so a held rotation never crosses a reacquire.
    last_valid_control_rotation = None
    previous_control_depth_m = None
    previous_control_depth_monotonic = None
    previous_quality_bbox = None
    orientation_filter = CausalWristSO3Filter(
        CausalWristSO3FilterConfig(
            time_constant_s=args.orientation_filter_time_constant_s,
            minimum_gain=args.orientation_filter_min_gain,
            maximum_gain=args.orientation_filter_max_gain,
            motion_gain_start_deg=args.orientation_filter_motion_start_deg,
            motion_gain_full_deg=args.orientation_filter_motion_full_deg,
            innovation_soft_deg=args.orientation_filter_soft_deg,
            innovation_hard_deg=args.orientation_filter_hard_deg,
            large_angle_mode=args.orientation_filter_large_angle_mode,
        )
    )
    print(
        "MANO wrist SO(3) large-angle mode: {}{}".format(
            args.orientation_filter_large_angle_mode.upper(),
            (
                " (valid large rotations are not rejected or softened)"
                if args.orientation_filter_large_angle_mode == "follow"
                else " (former hard rejection restored)"
            ),
        ),
        flush=True,
    )
    forearm_fusion_config = ForearmFusionConfig(
        maximum_fusion_weight=float(args.forearm_fusion_max_weight)
    )
    forearm_estimator = CausalForearmEstimator(forearm_fusion_config)
    calibrator = RobustBetasCalibrator(required_samples=30, maximum_samples=60)
    processed = 0
    valid_count = 0
    started_monotonic = 0.0
    teleop_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.teleop_udp_host else None
    teleop_session_id = "hamer-live-{}-{}".format(os.getpid(), int(time.time()))
    if teleop_socket is not None:
        print(
            "Gazebo teleoperation is LOCKED. Hold the hand at the desired "
            "zero pose and press C in the camera window to enable control.",
            flush=True,
        )
    teleop_skipped_since_report = 0
    teleop_last_warning_at = 0.0
    last_roi_reacquire_at = -float("inf")
    last_preflight_reason = "starting"
    warmup_s = 0.0
    try:
        # Load before ROI initialization: KLT must not sit idle while a 2.6 GiB
        # checkpoint is loaded and then compare a stale seed against a new frame.
        runner.load()
        warmup_s = runner.warmup()
        print(
            "HaMeR/MANO warm-up complete in {:.1f} ms".format(1000.0 * warmup_s),
            flush=True,
        )
        mano_topology = runner.mano_faces()
        wrist_definitions = {}
        if args.control_reference == "mano-wrist-ring":
            for physical_is_right in (False, True):
                neutral_vertices, neutral_joints = runner.neutral_mano_geometry(
                    physical_is_right
                )
                wrist_definitions[physical_is_right] = build_mano_wrist_definition(
                    neutral_vertices,
                    neutral_joints,
                    mano_topology,
                    physical_is_right,
                )
            print(
                "control reference: MANO wrist-opening ring "
                "(16 vertices, robust Kabsch; joint 0 is diagnostic only)",
                flush=True,
            )
        else:
            print(
                "control reference: legacy MANO joint 0 + MCP palm axes",
                flush=True,
            )
        mano_faces = None if args.no_mesh_overlay else mano_topology
        capture.start()
        seed_frame = capture.wait_for_stable_frames(consecutive=8)
        if not args.no_display:
            display = LiveDisplay(
                "D455 HaMeR/MANO - teleoperation_ubuntu_core renderer",
                args.display_helper_python,
                SCRIPT_DIR / "display_frame_stream.py",
                backend=args.display_backend,
            )
        seed_validation: Dict[str, Any]
        if args.auto_roi_mediapipe:
            detector_sidecar = MediaPipeDetectionSidecar(
                seed_frame.rgb.shape[1], seed_frame.rgb.shape[0],
                args.mediapipe_min_detection_confidence,
            )
            async_detector = AsyncMediaPipeDetection(
                detector_sidecar,
                hand_presence_timeout_s=args.hand_presence_timeout_s,
                confirmation_frames=2,
                negative_grace_frames=args.hand_miss_grace_frames,
            )
            preflight_gate = ConsecutiveHandDetectionGate(
                required_frames=max(2, int(args.hand_confirm_frames)),
                minimum_iou=0.40,
            )
            wait_for_hand_s = float(args.wait_for_hand_s)
            deadline = None if wait_for_hand_s <= 0.0 else time.monotonic() + wait_for_hand_s
            attempts = 0
            detection_results = 0
            last_detection_version = 0
            stable_detection = None
            preflight_started_at = time.monotonic()
            last_wait_log_at = -float("inf")
            last_preflight_reason = "waiting_for_first_detection"
            while True:
                attempts += 1
                async_detector.submit(seed_frame.rgb)
                (
                    last_detection_version, detection, _detection_monotonic,
                ) = async_detector.latest_after(
                    last_detection_version
                )
                if detection is not None:
                    detection_results += 1
                    confirmed_detection = preflight_gate.observe(detection)
                    if detection.get("valid"):
                        last_preflight_reason = "confirming {}/{}".format(
                            preflight_gate.count, preflight_gate.required_frames
                        )
                    else:
                        last_preflight_reason = str(
                            detection.get("reason", "not_valid")
                        )
                        stable_detection = None
                    if confirmed_detection is not None:
                        stable_detection = confirmed_detection
                        last_preflight_reason = "stable active hand - auto starting"
                if display is not None:
                    preview = cv2.cvtColor(seed_frame.rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        preview,
                        "Show the complete hand and wrist inside the image",
                        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview,
                        "Recommended distance: 0.45 - 0.80 m",
                        (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview,
                        "Detector: {}".format(last_preflight_reason),
                        (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, 180, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview,
                        "Preflight display/detect loop: {:.1f} Hz".format(
                            attempts/max(1.0e-9, time.monotonic()-preflight_started_at)
                        ),
                        (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (255, 255, 0), 1, cv2.LINE_AA,
                    )
                    if stable_detection is not None:
                        x1, y1, x2, y2 = np.rint(
                            stable_detection["bbox"]
                        ).astype(int)
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (80, 255, 80), 2)
                        cv2.putText(
                            preview,
                            (
                                "C / Enter / Space / double-click to confirm"
                                if args.require_hand_confirmation
                                else "AUTO START - active hand selected"
                            ),
                            (18, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (80, 255, 80), 2, cv2.LINE_AA,
                        )
                    cv2.putText(
                        preview, "PREFLIGHT ONLY - HaMeR / MANO IS OFF",
                        (18, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0, 0, 255), 2, cv2.LINE_AA,
                    )
                    display.show(preview)
                    if display.stop_requested:
                        return 0
                confirmed_by_user = bool(
                    args.auto_confirm_hand or not args.require_hand_confirmation
                )
                if display is not None and display.pop_confirm_request():
                    confirmed_by_user = True
                if confirmed_by_user and stable_detection is not None:
                    seed_validation = dict(stable_detection)
                    seed_validation["attempts"] = attempts
                    seed_validation["detection_results"] = detection_results
                    seed_validation["explicit_user_confirmation"] = not bool(
                        args.auto_confirm_hand or not args.require_hand_confirmation
                    )
                    seed_validation["automatic_active_hand_selection"] = not bool(
                        args.require_hand_confirmation
                    )
                    break
                if display is None and args.require_hand_confirmation:
                    raise RuntimeError(
                        "headless preflight cannot use --require-hand-confirmation"
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError(
                        "MediaPipe hand-presence preflight timed out: "
                        + repr(_hand_preflight_timeout_diagnostics(
                            last_preflight_reason,
                            attempts,
                            detection_results,
                            wait_for_hand_s,
                        ))
                    )
                log_now = time.monotonic()
                if log_now - last_wait_log_at >= 2.0:
                    print("Waiting for a visible hand in the D455 image...",
                          file=sys.stderr, flush=True)
                    last_wait_log_at = log_now
                seed_frame = capture.wait_for_frame()
            bbox = np.asarray(seed_validation["bbox"], dtype=np.float64)
            is_right = bool(seed_validation["is_right"])
        elif args.bbox:
            bbox = np.asarray(args.bbox, dtype=np.float64)
            seed_validation = {
                "valid": True,
                "hand_presence_validated": False,
                "source": "unverified_cli_bbox",
            }
        else:
            bbox = _select_bbox(seed_frame.rgb)
            seed_validation = {
                "valid": True,
                "hand_presence_validated": True,
                "source": "human_mouse_selection",
            }
        tracker = KLTTrackerROIProvider(
            initial_bbox=bbox,
            is_right=is_right,
            bbox_smoothing_alpha=args.roi_smoothing_alpha,
            minimum_visible_fraction=0.35,
        )
        # Initialize inside the capture worker on the exact frame it publishes.
        # The sidecar can take hundreds of milliseconds; using its older input
        # as KLT's previous frame caused an immediate loss followed by permanent
        # not_initialized results.
        pending.set(bbox, is_right)
        worker = threading.Thread(
            target=_capture_worker,
            args=(capture, tracker, slot, preview_slot if display is not None else None,
                  pending, async_detector, stop),
            daemon=True,
        )
        worker.start()
        gpu_sampler.start()
        if display is not None:
            display_worker = threading.Thread(
                target=_display_worker,
                args=(
                    display,
                    preview_slot,
                    overlay_state,
                    pose_display,
                    teleop_control_gate,
                    async_detector,
                    stop,
                    args.mesh_renderer,
                ),
                name="hamer-live-preview",
                daemon=True,
            )
            display_worker.start()

        if args.experiment:
            output_root = Path(args.output_root).resolve()
            suffix = time.strftime("%Y%m%dT%H%M%S")
            output_dir = output_root / f"{args.experiment}_{suffix}"
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "rgb").mkdir()
            (output_dir / "aligned_depth").mkdir()
            records_file = (output_dir / "frames.jsonl").open("x", encoding="utf-8")
            video = cv2.VideoWriter(
                str(output_dir / "axes_overlay.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (1280, 480) if args.mesh_renderer == "teleoperation-core"
                else (640, 480),
            )
            if not video.isOpened():
                raise RuntimeError("failed to open OpenCV overlay video")

        last_version = 0
        presence_state: Dict[str, Any] = {
            "valid": True,
            "reason": "manual_presence_contract",
            "generation": 0,
            "active_hand_generation": 0,
            "active_hand_is_right": bool(is_right),
            "detection_age_s": 0.0,
            "confirmed_detection": None,
        }
        if async_detector is not None:
            presence_state = async_detector.presence_snapshot()
        last_presence_valid = bool(presence_state["valid"])
        last_presence_generation = int(presence_state["generation"])
        countdown_started = time.monotonic()
        last_packet = None
        while time.monotonic() - countdown_started < args.countdown_s:
            last_version, packet = slot.get_after(last_version)
            if packet is None:
                continue
            last_packet = packet
            if async_detector is not None:
                presence_state = async_detector.presence_snapshot()
                if not presence_state["valid"] and not packet.roi.lost:
                    pending.request_reset()
                last_presence_valid = bool(presence_state["valid"])
                last_presence_generation = int(presence_state["generation"])
            remaining = max(0.0, args.countdown_s - (time.monotonic() - countdown_started))
            overlay_state.update(
                None, None, 0.0,
                "{} starts in {:.1f}s".format(args.experiment or "LIVE", remaining),
                mano_faces, None, None, int(presence_state["generation"]),
            )
            pose_display.invalidate(
                "live_countdown", int(presence_state["generation"])
            )
            if display is not None and display.stop_requested:
                break
        started_monotonic = time.monotonic()

        while args.duration_s <= 0.0 or time.monotonic() - started_monotonic < args.duration_s:
            if display is not None and display.stop_requested:
                break
            last_version, packet = slot.get_after(last_version)
            if packet is None:
                break
            last_packet = packet
            processed += 1
            result = None
            estimates = None
            failure_reason = ""
            projection_source_bbox = None
            teleoperation_core_render = None
            pose_packet_payload = None
            pose_failure_reason = ""
            forearm_observation = None
            crop_quality_score = 0.0
            timestamp_s = packet.frame.color_timestamp_ms / 1000.0
            if async_detector is not None:
                presence_state = async_detector.presence_snapshot()
            else:
                presence_state = {
                    "valid": True,
                    "reason": "manual_presence_contract",
                    "generation": 0,
                    "active_hand_generation": 0,
                    "active_hand_is_right": bool(is_right),
                    "detection_age_s": 0.0,
                    "confirmed_detection": None,
                }
            presence_valid = bool(presence_state["valid"])
            presence_generation = int(presence_state["generation"])
            active_hand_generation = int(
                presence_state.get("active_hand_generation", 0)
            )
            presence_interval_changed = (
                presence_generation != last_presence_generation
            )
            if presence_interval_changed:
                orientation_filter.reset()
                previous_joint_quaternion = None
                previous_control_quaternion = None
                last_valid_control_rotation = None
                previous_control_depth_m = None
                previous_control_depth_monotonic = None
                previous_quality_bbox = None
                forearm_estimator.reset()
            confirmed_detection = presence_state.get("confirmed_detection")
            roi_alignment = (
                {"valid": True, "iou": 1.0,
                 "normalized_center_distance": 0.0, "reason": "manual"}
                if async_detector is None
                else _roi_detector_alignment(packet.roi, presence_state)
            )
            if not presence_valid:
                failure_reason = "no_real_hand:" + str(presence_state["reason"])
                # KLT is only a fast interpolation mechanism.  It must not
                # survive independent evidence that the real hand disappeared.
                if last_presence_valid or not packet.roi.lost:
                    pending.request_reset()
            elif presence_interval_changed:
                # Never resume from the old background/hand track.  A new
                # continuous presence interval always starts from MediaPipe's
                # newly confirmed bbox and produces a new HaMeR result.
                if confirmed_detection is not None:
                    detected_is_right = bool(confirmed_detection["is_right"])
                    if detected_is_right != is_right:
                        is_right = detected_is_right
                        orientation_filter.reset()
                        previous_joint_quaternion = None
                        previous_control_quaternion = None
                        last_valid_control_rotation = None
                        previous_control_depth_m = None
                        previous_control_depth_monotonic = None
                        calibrator.reset()
                        runner.reset_shape()
                        runner.freeze_betas = False
                    pending.set(confirmed_detection["bbox"], detected_is_right)
                    failure_reason = "hand_reacquiring:new_presence_interval"
                else:
                    pending.request_reset()
                    failure_reason = "hand_reacquiring:awaiting_confirmed_bbox"
            elif packet.roi.lost or packet.roi.bbox is None:
                previous_control_depth_m = None
                previous_control_depth_monotonic = None
                previous_quality_bbox = None
                failure_reason = "roi_lost:" + packet.roi.reason
                reacquire_now = time.monotonic()
                if reacquire_now - last_roi_reacquire_at >= max(
                    0.2, float(args.roi_reacquire_period_s)
                ):
                    last_roi_reacquire_at = reacquire_now
                    if confirmed_detection is not None:
                        detected_is_right = bool(confirmed_detection["is_right"])
                        if detected_is_right != is_right:
                            is_right = detected_is_right
                            orientation_filter.reset()
                            previous_joint_quaternion = None
                            previous_control_quaternion = None
                            last_valid_control_rotation = None
                            previous_control_depth_m = None
                            previous_control_depth_monotonic = None
                            calibrator.reset()
                            runner.reset_shape()
                            runner.freeze_betas = False
                        pending.set(confirmed_detection["bbox"], detected_is_right)
                        failure_reason = "roi_reacquiring:mediapipe_bbox"
            elif not roi_alignment["valid"]:
                # A real hand elsewhere in the image does not validate this
                # crop.  Block HaMeR and force a detector-anchored correction.
                previous_control_depth_m = None
                previous_control_depth_monotonic = None
                previous_quality_bbox = None
                if confirmed_detection is not None:
                    pending.set(
                        confirmed_detection["bbox"],
                        bool(confirmed_detection["is_right"]),
                    )
                failure_reason = (
                    "roi_detector_mismatch:iou={:.3f},center={:.3f}".format(
                        roi_alignment["iou"],
                        roi_alignment["normalized_center_distance"],
                    )
                )
            else:
                try:
                    candidate_result = runner.infer(
                        packet.frame.rgb, packet.roi.bbox, is_right, timestamp_s
                    )
                    post_presence = (
                        None
                        if async_detector is None
                        else async_detector.presence_snapshot()
                    )
                    if (
                        post_presence is not None
                        and (
                            not post_presence["valid"]
                            or int(post_presence["generation"])
                            != presence_generation
                        )
                    ):
                        # HaMeR is slower than MediaPipe.  A disappearance can
                        # therefore be detected while inference is in flight;
                        # discard that result before calibration, display and
                        # UDP rather than leaking one final ghost-hand packet.
                        presence_state = post_presence
                        presence_valid = bool(post_presence["valid"])
                        presence_generation = int(post_presence["generation"])
                        active_hand_generation = int(
                            post_presence.get("active_hand_generation", 0)
                        )
                        presence_interval_changed = True
                        forearm_estimator.reset()
                        previous_control_depth_m = None
                        previous_control_depth_monotonic = None
                        previous_quality_bbox = None
                        confirmed_detection = post_presence.get(
                            "confirmed_detection"
                        )
                        if not presence_valid:
                            pending.request_reset()
                            failure_reason = "no_real_hand:" + str(
                                post_presence["reason"]
                            )
                        elif confirmed_detection is not None:
                            detected_is_right = bool(
                                confirmed_detection["is_right"]
                            )
                            if detected_is_right != is_right:
                                is_right = detected_is_right
                                orientation_filter.reset()
                                previous_joint_quaternion = None
                                previous_control_quaternion = None
                                last_valid_control_rotation = None
                                previous_control_depth_m = None
                                previous_control_depth_monotonic = None
                                calibrator.reset()
                                runner.reset_shape()
                                runner.freeze_betas = False
                            pending.set(
                                confirmed_detection["bbox"], detected_is_right
                            )
                            failure_reason = (
                                "hand_reacquiring:presence_changed_during_inference"
                            )
                        else:
                            pending.request_reset()
                            failure_reason = (
                                "hand_reacquiring:no_confirmed_bbox_after_inference"
                            )
                    else:
                        result = candidate_result
                        valid_count += 1
                        just_frozen = calibrator.add(result.betas, timestamp_s)
                        if just_frozen:
                            runner.set_frozen_betas(
                                is_right, calibrator.betas_user
                            )
                        estimates = build_live_palm_estimates(
                            result, previous_joint_quaternion
                        )
                        raw_joint = dict(estimates["mano_joint_palm_frame"])
                        estimates["mano_joint_palm_frame_raw"] = raw_joint
                        if raw_joint.get("valid"):
                            previous_joint_quaternion = np.asarray(
                                raw_joint["quaternion_xyzw"], dtype=np.float64
                            )
                        else:
                            previous_joint_quaternion = None
                        if args.control_reference == "mano-wrist-ring":
                            raw_control = estimate_mano_wrist_frame(
                                result.pred_vertices_source_camera_axes,
                                wrist_definitions[bool(result.is_right)],
                                previous_control_quaternion,
                            ).as_dict()
                        else:
                            raw_control = {
                                **raw_joint,
                                "reference_kind": "MANO_JOINT_0_PALM_FRAME_LEGACY",
                            }
                        estimates["control_wrist_frame_raw"] = dict(raw_control)
                        crop_quality_score = bbox_crop_quality(
                            packet.roi.bbox,
                            previous_quality_bbox,
                            packet.frame.rgb.shape[1],
                            packet.frame.rgb.shape[0],
                        )
                        previous_quality_bbox = np.asarray(
                            packet.roi.bbox, dtype=np.float64
                        ).copy()
                        roi_quality = float(np.clip(
                            getattr(packet.roi, "confidence", 0.0), 0.0, 1.0
                        ))
                        visible_quality = float(np.clip(
                            result.quality.get("bbox_visible_fraction", 0.0),
                            0.0,
                            1.0,
                        ))
                        measurement_quality = (
                            roi_quality * visible_quality * crop_quality_score
                        )
                        measurement_quality *= float(np.clip(
                            raw_control.get("quality", {}).get(
                                "geometric_confidence", 1.0
                            ),
                            0.0,
                            1.0,
                        ))
                        if raw_control.get("valid"):
                            filtered_control = orientation_filter.update(
                                timestamp_s,
                                raw_control["rotation"],
                                measurement_quality,
                            )
                            estimates["palm_orientation_filter"] = (
                                filtered_control.as_dict()
                            )
                            estimates["teleop_crop_quality"] = crop_quality_score
                            if filtered_control.valid:
                                control = dict(raw_control)
                                control["rotation"] = filtered_control.rotation.tolist()
                                control["quaternion_xyzw"] = (
                                    filtered_control.quaternion_xyzw.tolist()
                                )
                                control["orientation_source"] = (
                                    "HAMER_MANO_WRIST_RING_16_IRLS_KABSCH_"
                                    "QUALITY_ADAPTIVE_CAUSAL_SO3"
                                    if args.control_reference == "mano-wrist-ring"
                                    else
                                    "HAMER_MANO_JOINT_PALM_FRAME_"
                                    "QUALITY_ADAPTIVE_CAUSAL_SO3"
                                )
                                control["raw_rotation_preserved_in"] = (
                                    "control_wrist_frame_raw"
                                )
                                control["filter_confidence"] = float(
                                    filtered_control.confidence
                                )
                                control["filter_status"] = filtered_control.status
                                control["quality"] = {
                                    **dict(raw_control.get("quality", {})),
                                    "crop_quality": crop_quality_score,
                                    "measurement_quality": measurement_quality,
                                    "filter_gain": filtered_control.gain,
                                    "filter_innovation_deg": (
                                        filtered_control.innovation_deg
                                    ),
                                }
                                estimates["control_wrist_frame"] = control
                                last_valid_control_rotation = (
                                    filtered_control.rotation.copy()
                                )
                                previous_control_quaternion = (
                                    filtered_control.quaternion_xyzw
                                )
                                if args.control_reference == "mano-joint-palm":
                                    estimates["mano_joint_palm_frame"] = control
                            else:
                                filter_reason = (
                                    "causal_so3_filter:"
                                    + filtered_control.reason
                                )
                                if last_valid_control_rotation is not None:
                                    # A MANO root-orientation failure must not
                                    # suppress an otherwise valid D455 metric
                                    # wrist position.  Emit the last trusted
                                    # SO(3) value with zero rotational
                                    # confidence; ROS therefore keeps the
                                    # orientation target fixed while position
                                    # continues to update.
                                    held_rotation = (
                                        last_valid_control_rotation.copy()
                                    )
                                    held_quaternion = (
                                        rotation_matrix_to_quaternion_xyzw(
                                            held_rotation
                                        )
                                    )
                                    estimates["control_wrist_frame"] = {
                                        **raw_control,
                                        "valid": True,
                                        "rotation": held_rotation.tolist(),
                                        "quaternion_xyzw": (
                                            held_quaternion.tolist()
                                        ),
                                        "failure_reason": filter_reason,
                                        "filter_confidence": 0.0,
                                        "filter_status": (
                                            filtered_control.status
                                        ),
                                        "orientation_channel_valid": False,
                                        "orientation_held": True,
                                        "orientation_source": (
                                            "HELD_LAST_TRUSTED_MANO_"
                                            "ORIENTATION"
                                        ),
                                    }
                                    previous_control_quaternion = (
                                        held_quaternion
                                    )
                                else:
                                    estimates["control_wrist_frame"] = {
                                        **raw_control,
                                        "valid": False,
                                        "rotation": None,
                                        "quaternion_xyzw": None,
                                        "origin": None,
                                        "failure_reason": filter_reason,
                                        "filter_confidence": 0.0,
                                        "filter_status": (
                                            filtered_control.status
                                        ),
                                        "orientation_channel_valid": False,
                                        "orientation_held": False,
                                    }
                                    previous_control_quaternion = None
                                if args.control_reference == "mano-joint-palm":
                                    estimates["mano_joint_palm_frame"] = dict(
                                        estimates["control_wrist_frame"]
                                    )
                                failure_reason = (
                                    "orientation_filter:"
                                    + filtered_control.reason
                                )
                        else:
                            previous_control_depth_m = None
                            previous_control_depth_monotonic = None
                            estimates["control_wrist_frame"] = dict(raw_control)
                            estimates["palm_orientation_filter"] = {
                                "valid": False,
                                "status": "raw_control_frame_invalid",
                                "reason": raw_control.get(
                                    "failure_reason", "invalid_raw_control_frame"
                                ),
                            }
                            estimates["teleop_crop_quality"] = crop_quality_score
                            failure_reason = "control_reference:" + str(
                                raw_control.get(
                                    "failure_reason", "invalid_raw_control_frame"
                                )
                            )
                        projection_source_bbox = np.asarray(
                            packet.roi.bbox, dtype=np.float64
                        ).copy()
                except Exception as exc:
                    failure_reason = f"{type(exc).__name__}:{exc}"
            if (
                args.mesh_renderer == "teleoperation-core"
                and result is not None
                and mano_faces is not None
            ):
                try:
                    teleoperation_core_render = render_inference_frame(
                        packet.frame.rgb,
                        result,
                        mano_faces,
                    )
                except Exception as exc:
                    render_reason = (
                        "teleoperation_core_render:"
                        + type(exc).__name__
                        + ":"
                        + str(exc)
                    )
                    failure_reason = (
                        render_reason
                        if not failure_reason
                        else failure_reason + ";" + render_reason
                    )
            confirmed_identity = presence_state.get("confirmed_detection")
            identity_hand_is_right = (
                bool(is_right)
                if async_detector is None
                else (
                    None
                    if confirmed_identity is None
                    else bool(confirmed_identity["is_right"])
                )
            )
            if not presence_valid or presence_interval_changed:
                teleop_control_gate.invalidate(
                    "HAND_PRESENCE_CHANGED_REQUIRES_NEW_C"
                )
            elif identity_hand_is_right is None:
                teleop_control_gate.invalidate(
                    "HAND_IDENTITY_UNAVAILABLE_REQUIRES_NEW_C"
                )
            else:
                teleop_control_gate.observe_identity(
                    presence_generation,
                    active_hand_generation,
                    identity_hand_is_right,
                )
            orientation_filter_state = (
                {} if estimates is None
                else estimates.get("palm_orientation_filter") or {}
            )
            if orientation_filter_state.get("status") == "jump_rejected":
                teleop_control_gate.invalidate(
                    "ORIENTATION_JUMP_REQUIRES_NEW_C"
                )
            if (
                result is not None
                and identity_hand_is_right is not None
                and bool(result.is_right) != identity_hand_is_right
            ):
                failure_reason = "hamer_result_identity_mismatch"
                pose_failure_reason = failure_reason
                pose_delta = pose_display.invalidate(
                    pose_failure_reason, presence_generation
                )
                result = None
                estimates = None

            if result is not None and estimates is not None:
                try:
                    depth_now_monotonic = time.monotonic()
                    reference_depth_age_s = (
                        None
                        if previous_control_depth_monotonic is None
                        else max(
                            0.0,
                            depth_now_monotonic
                            - previous_control_depth_monotonic,
                        )
                    )
                    pose_packet_payload = build_live_teleop_packet(
                        result,
                        estimates,
                        packet.frame,
                        packet.roi,
                        teleop_session_id,
                        processed - 1,
                        presence_generation,
                        active_hand_generation,
                        reference_depth_m=previous_control_depth_m,
                        reference_depth_age_s=reference_depth_age_s,
                    )
                    diagnostics = pose_packet_payload.get(
                        "position_diagnostics"
                    )
                    if (
                        diagnostics is not None
                        and not diagnostics.get(
                            "depth_reference_hold_used", False
                        )
                    ):
                        previous_control_depth_m = float(
                            diagnostics["depth_m"]
                        )
                        previous_control_depth_monotonic = depth_now_monotonic
                    if not args.disable_forearm_fusion:
                        forearm_observation = forearm_estimator.update(
                            packet.frame.aligned_depth_raw,
                            packet.frame.depth_scale_m_per_unit,
                            packet.frame.color_intrinsics,
                            pose_packet_payload["wrist_position_m"],
                            presence_state.get("confirmed_detection"),
                            detection_age_s=float(
                                presence_state.get("detection_age_s", float("inf"))
                            ),
                        )
                    pose_packet_payload = apply_forearm_fusion_to_packet(
                        pose_packet_payload,
                        forearm_observation,
                        forearm_fusion_config,
                    )
                    estimates["forearm_fusion"] = pose_packet_payload.get(
                        "forearm_fusion"
                    )
                    pose_delta = pose_display.update_from_packet(
                        pose_packet_payload, presence_generation
                    )
                except Exception as exc:
                    pose_failure_reason = "metric_6d_pose:{}:{}".format(
                        type(exc).__name__, exc
                    )
                    pose_delta = pose_display.invalidate(
                        pose_failure_reason, presence_generation
                    )
            else:
                pose_failure_reason = failure_reason or "hamer_pose_unavailable"
                pose_delta = pose_display.invalidate(
                    pose_failure_reason, presence_generation
                )
            last_presence_valid = presence_valid
            last_presence_generation = presence_generation
            elapsed = max(1e-9, time.monotonic() - started_monotonic)
            hamer_fps = valid_count / elapsed
            overlay = None
            if output_dir is not None:
                # The teleoperation-core pair has already been rendered once
                # on the exact inference frame; this only adds the HUD/video.
                overlay = make_overlay(
                    packet.frame.rgb, packet.roi, result, estimates,
                    hamer_fps, failure_reason, mano_faces,
                    projection_source_bbox, presence_valid,
                    bool(roi_alignment["valid"]),
                    presence_state.get("active_hand_is_right"),
                    int((presence_state.get("confirmed_detection") or {}).get(
                        "ignored_non_active_hand_count", 0
                    )),
                    teleoperation_core_render,
                    args.mesh_renderer,
                )
            overlay_state.update(
                result, estimates, hamer_fps, failure_reason, mano_faces,
                projection_source_bbox, teleoperation_core_render,
                presence_generation,
            )
            record: Dict[str, Any] = {
                "index": processed - 1,
                "capture_sequence": packet.capture_sequence,
                "timestamp": timestamp_s,
                "timestamp_ms": packet.frame.color_timestamp_ms,
                "timestamp_domain": packet.frame.color_timestamp_domain,
                "frame_number": packet.frame.color_frame_number,
                "roi": packet.roi.as_dict(),
                "roi_detector_alignment": _json_safe(roi_alignment),
                "crop_quality": float(crop_quality_score),
                "mano_renderer": args.mesh_renderer,
                "mano_render_exact_inference_frame": bool(
                    teleoperation_core_render is not None
                ),
                "hand_presence": _json_safe(presence_state),
                "valid": result is not None and estimates is not None,
                "failure_reason": failure_reason,
                "hand_pose_6d": pose_delta.as_dict(),
                "hand_pose_6d_failure_reason": pose_failure_reason,
                "forearm_fusion": (
                    None
                    if pose_packet_payload is None
                    else _json_safe(pose_packet_payload.get("forearm_fusion"))
                ),
                "inference_ms": None if result is None else 1000.0 * result.inference_time_s,
                "betas_calibration": calibrator.as_dict(),
                "gpu_system_peak_used_mib_so_far": gpu_sampler.peak_used_mib,
            }
            if result is not None:
                record.update({
                    "bbox": result.requested_bbox_xyxy,
                    "mano_vertices": result.pred_vertices_source_camera_axes,
                    "mano_joints": result.pred_keypoints_3d_source_camera_axes,
                    "mano_joints_2d_crop_normalized": result.pred_keypoints_2d_crop_normalized,
                    "global_orient": result.global_orient,
                    "hand_pose": result.hand_pose,
                    "betas": result.betas,
                    "betas_user": calibrator.betas_user,
                    "palm_frames": estimates,
                    "hamer_quality": result.quality,
                })
            if teleop_socket is not None:
                try:
                    udp_payload = pose_packet_payload
                    if udp_payload is None:
                        udp_payload = build_invalid_teleop_packet(
                            teleop_session_id,
                            processed - 1,
                            timestamp_s,
                            pose_failure_reason
                            or failure_reason
                            or pose_delta.reason
                            or "HAMER_POSE_UNAVAILABLE",
                            presence_generation,
                            active_hand_generation,
                            hand_is_right=(
                                identity_hand_is_right
                                if presence_valid else None
                            ),
                        )
                    teleop_packet = teleop_control_gate.decorate(
                        udp_payload, teleop_session_id
                    )
                    teleop_socket.sendto(
                        json.dumps(
                            _json_safe(teleop_packet),
                            separators=(",", ":"),
                        ).encode("utf-8"),
                        (args.teleop_udp_host, int(args.teleop_udp_port)),
                    )
                    record["teleop_udp"] = {
                        "valid": bool(teleop_packet.get("valid", False)),
                        "control_enabled": bool(
                            teleop_packet.get("control_enabled", False)
                        ),
                        "control_reference_token": str(
                            teleop_packet.get("control_reference_token", "")
                        ),
                        "invalid_reason": str(
                            teleop_packet.get("invalid_reason", "")
                        ),
                    }
                except Exception as exc:
                    teleop_skipped_since_report += 1
                    warning_now = time.monotonic()
                    if warning_now - teleop_last_warning_at >= 1.0:
                        print(
                            "teleop UDP heartbeat failed: {} ({} frame(s) since last report)".format(
                                exc, teleop_skipped_since_report
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        teleop_skipped_since_report = 0
                        teleop_last_warning_at = warning_now
            if output_dir is not None:
                rgb_name = f"rgb/{processed - 1:06d}.png"
                depth_name = f"aligned_depth/{processed - 1:06d}.png"
                if not cv2.imwrite(str(output_dir / rgb_name), cv2.cvtColor(
                    packet.frame.rgb, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError("failed to write RGB frame")
                if not cv2.imwrite(str(output_dir / depth_name), packet.frame.aligned_depth_raw):
                    raise RuntimeError("failed to write aligned depth frame")
                record["rgb_path"] = rgb_name
                record["aligned_depth_path"] = depth_name
                records_file.write(json.dumps(_json_safe(record), separators=(",", ":")) + "\n")
                records_file.flush()
                if overlay is None:
                    raise RuntimeError("experiment overlay was not rendered")
                video.write(overlay)
            session_records.append(record)
            if (display is not None and display.pop_reinitialize_request()
                    and last_packet is not None):
                pending.set(_select_bbox(last_packet.frame.rgb), is_right)
        elapsed = max(1e-9, time.monotonic() - started_monotonic)
        inference_times = [record["inference_ms"] for record in session_records
                           if record["inference_ms"] is not None]
        summary = {
            "schema_version": 1,
            "experiment": args.experiment,
            "profile": "D455 color RGB8 + aligned depth Z16, 640x480@30",
            "usb_type_descriptor": capture.device_metadata["usb_type_descriptor"],
            "device": capture.device_metadata,
            "gpu": "NVIDIA GeForce RTX 2060 6144 MiB",
            "precision": args.precision,
            "batch_size": 1,
            "vitdet_enabled": False,
            "vitpose_enabled": False,
            "pyrender_enabled": False,
            "mano_renderer": args.mesh_renderer,
            "mano_render_contract": (
                "same_hamer_inference_rgb_no_cross_frame_mesh_warp"
                if args.mesh_renderer == "teleoperation-core"
                else "legacy_latest_rgb_with_klt_bbox_mesh_remap"
            ),
            "opencv_mano_mesh_overlay_enabled": not args.no_mesh_overlay,
            "obj_export_enabled": False,
            "hand_pose_6d_display_enabled": not args.no_display,
            "hand_pose_6d_reference": (
                "C_ZERO_MANO_WRIST_RING_16_D455_METRIC_SO3_"
                "PLUS_LOW_WEIGHT_RGBD_FOREARM_AXIS"
            ),
            "forearm_fusion_enabled": not args.disable_forearm_fusion,
            "forearm_fusion_max_weight": float(
                args.forearm_fusion_max_weight
            ),
            "orientation_filter_large_angle_mode": (
                args.orientation_filter_large_angle_mode
            ),
            "orientation_filter_maximum_gain": float(
                args.orientation_filter_max_gain
            ),
            "duration_s": elapsed,
            "processed_frames": processed,
            "valid_frames": valid_count,
            "valid_coverage": valid_count / max(processed, 1),
            "experiment_usable": bool(
                processed > 0
                and valid_count / processed >= 0.80
                and calibrator.frozen
                and seed_validation.get("valid")
            ),
            "actual_hamer_hz": valid_count / elapsed,
            "inference_ms": {
                "mean": None if not inference_times else float(np.mean(inference_times)),
                "median": _percentile(inference_times, 50),
                "p95": _percentile(inference_times, 95),
            },
            "hamer_warmup_ms": 1000.0 * warmup_s,
            "latest_frame_scheduler": slot.statistics,
            "roi_seed": _json_safe(seed_validation),
            "roi_seed_hand_presence_validated": bool(
                seed_validation.get("hand_presence_validated",
                                    seed_validation.get("valid", False)
                                    and args.auto_roi_mediapipe)
            ),
            "betas_calibration": calibrator.as_dict(),
            "gpu_system_peak_used_mib": gpu_sampler.peak_used_mib,
            "gpu_memory_samples": gpu_sampler.samples,
            "development_limitation": (
                "当前D455使用USB 2.1，本阶段结果用于算法开发。"
                "最终实时性能、正式数据集和长时间稳定性以后在USB3条件下重新测试。"
            ),
        }
        if output_dir is not None:
            (output_dir / "summary.json").write_text(
                json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False))
        else:
            print(json.dumps(summary, ensure_ascii=False))
        return 0 if valid_count > 0 else 2
    finally:
        if teleop_socket is not None:
            try:
                teleop_control_gate.invalidate(
                    "CAMERA_STREAM_STOPPED_REQUIRES_NEW_C"
                )
                shutdown_packet = teleop_control_gate.decorate(
                    build_invalid_teleop_packet(
                        teleop_session_id,
                        processed,
                        time.time(),
                        "CAMERA_STREAM_STOPPED_REQUIRES_NEW_C",
                        0,
                        0,
                    ),
                    teleop_session_id,
                )
                teleop_socket.sendto(
                    json.dumps(
                        _json_safe(shutdown_packet), separators=(",", ":")
                    ).encode("utf-8"),
                    (args.teleop_udp_host, int(args.teleop_udp_port)),
                )
            except Exception as exc:
                print(
                    "teleop UDP shutdown heartbeat failed: {}".format(exc),
                    file=sys.stderr,
                    flush=True,
                )
        stop.set()
        slot.close()
        preview_slot.close()
        if worker is not None:
            worker.join(timeout=5.0)
        if display_worker is not None:
            display_worker.join(timeout=3.0)
        gpu_sampler.stop()
        if records_file is not None:
            records_file.close()
        if video is not None:
            video.release()
        if teleop_socket is not None:
            teleop_socket.close()
        capture.stop()
        if display is not None:
            display.close()
        if async_detector is not None:
            async_detector.close()
        if detector_sidecar is not None:
            detector_sidecar.close()
        process_singleton.close()


if __name__ == "__main__":
    raise SystemExit(main())
