#!/usr/bin/env python3
"""Record lossless D455 RGB and color-aligned raw depth with full provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import statistics
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TextIO

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.d455_capture import D455Capture, D455CaptureError, D455Frame  # noqa: E402


SCENARIOS = (
    "G00_STATIC",
    "G01_TRANSLATION_X",
    "G02_TRANSLATION_Y",
    "G03_TRANSLATION_Z",
    "G04_ROLL",
    "G05_PITCH",
    "G06_YAW",
    "G07_OPEN_CLOSE",
    "G08_OCCLUSION",
    "G09_CONTINUOUS",
)


class RecordingInterrupted(D455CaptureError):
    """Raised by the recorder's controlled SIGINT/SIGTERM shutdown path."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        try:
            signal_name = signal.Signals(self.signum).name
        except ValueError:
            signal_name = f"SIGNAL_{self.signum}"
        self.signal_name = signal_name
        super().__init__(f"recording interrupted by {signal_name}")


class RecordingSignalGuard:
    """Temporarily turn SIGINT/SIGTERM into catchable recorder failures.

    Signal handlers can only be installed in Python's main thread.  Unit tests
    and embedders that call ``record`` from another thread still get normal
    exception finalization, but no process-global handlers are changed.
    """

    _SIGNALS = (signal.SIGINT, signal.SIGTERM)

    def __init__(self) -> None:
        self._previous: Dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        raise RecordingInterrupted(signum)

    def __enter__(self) -> "RecordingSignalGuard":
        if threading.current_thread() is threading.main_thread():
            for signum in self._SIGNALS:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_args: Any) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def _failure_reason(exc: BaseException, stage: str) -> Dict[str, Any]:
    """Return stable, machine-readable failure metadata without hiding details."""

    causes: List[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and current not in causes:
        causes.append(current)
        current = current.__cause__ or current.__context__
    if isinstance(exc, RecordingInterrupted):
        code = f"SIGNAL_{exc.signal_name}"
    elif any(isinstance(item, OSError) and item.errno == 28 for item in causes):
        code = "ENOSPC"
    elif stage.startswith("manifest"):
        code = "MANIFEST_WRITE_FAILED"
    elif any(isinstance(item, FileExistsError) for item in causes):
        code = "EXISTING_FILE_CONFLICT"
    elif "queue saturated" in str(exc):
        code = "WRITER_QUEUE_OVERFLOW"
    elif "failed to save" in " ".join(str(item) for item in causes):
        code = "PNG_WRITE_FAILED"
    elif "writer failed" in str(exc):
        code = "WRITER_EXCEPTION"
    elif "timeout" in " ".join(str(item).lower() for item in causes):
        code = "FRAMESET_TIMEOUT"
    else:
        code = "RECORDING_FAILED"
    return {
        "code": code,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        os.fsync(stream.fileno())
    return digest.hexdigest()


def save_frame(session_dir: Path, index: int, frame: D455Frame) -> Dict[str, Any]:
    """Losslessly save raw RGB plus raw-unit aligned Z16 depth."""

    stem = f"{index:06d}"
    rgb_relative = Path("rgb") / f"{stem}.png"
    raw_depth_relative = Path("raw_depth") / f"{stem}.png"
    depth_relative = Path("aligned_depth") / f"{stem}.png"
    output_paths = (
        session_dir / rgb_relative,
        session_dir / raw_depth_relative,
        session_dir / depth_relative,
    )
    conflicts = [str(path.relative_to(session_dir)) for path in output_paths if path.exists()]
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite existing frame files: " + ", ".join(conflicts)
        )
    # OpenCV PNG expects BGR channel order; the stored file remains normal RGB.
    if not cv2.imwrite(
        str(session_dir / rgb_relative), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"failed to save RGB frame {index}")
    if not cv2.imwrite(str(session_dir / raw_depth_relative), frame.raw_depth_raw):
        raise OSError(f"failed to save raw depth frame {index}")
    if not cv2.imwrite(str(session_dir / depth_relative), frame.aligned_depth_raw):
        raise OSError(f"failed to save aligned depth frame {index}")
    record = {
        "schema_version": 2,
        "index": index,
        "rgb_path": str(rgb_relative),
        "raw_depth_path": str(raw_depth_relative),
        "aligned_depth_path": str(depth_relative),
        "rgb_sha256": _sha256(session_dir / rgb_relative),
        "raw_depth_sha256": _sha256(session_dir / raw_depth_relative),
        "aligned_depth_sha256": _sha256(session_dir / depth_relative),
        **frame.metadata(),
        # These fields are intentionally present and invalid until their own
        # phases produce them.  Never synthesize processed outputs at record time.
        "roi": {"valid": False, "reason": "NOT_RUN_P3"},
        "hamer": {"valid": False, "reason": "NOT_RUN_DURING_RAW_RECORDING"},
        "klt_features": {"valid": False, "reason": "NOT_RUN_P4"},
        "state": {
            "capture_valid": True,
            "raw_rgb_saved": True,
            "aligned_raw_depth_saved": True,
        },
    }
    return record


def _distribution(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(data)),
        "median_ms": float(np.median(data)),
        "p95_ms": float(np.percentile(data, 95)),
        "p99_ms": float(np.percentile(data, 99)),
        "max_ms": float(np.max(data)),
    }


def _process_rss_bytes() -> int:
    with Path("/proc/self/status").open() as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise OSError("VmRSS is unavailable")


class ResourceMonitor:
    """Sample process RSS, CPU and open file descriptors during recording."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self._interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._rss: List[int] = []
        self._cpu_percent: List[float] = []
        self._fds: List[int] = []
        self._started_wall = time.monotonic()
        self._started_cpu = time.process_time()
        self._thread = threading.Thread(target=self._run, name="d455-resource-monitor")
        self._thread.start()

    def _run(self) -> None:
        previous_wall = time.monotonic()
        previous_cpu = time.process_time()
        while not self._stop_event.is_set():
            try:
                self._rss.append(_process_rss_bytes())
                self._fds.append(len(list(Path("/proc/self/fd").iterdir())))
            except OSError:
                pass
            current_wall = time.monotonic()
            current_cpu = time.process_time()
            elapsed = current_wall - previous_wall
            if elapsed > 0.0:
                self._cpu_percent.append(100.0 * (current_cpu - previous_cpu) / elapsed)
            previous_wall, previous_cpu = current_wall, current_cpu
            self._stop_event.wait(self._interval_s)

    def finish(self) -> Dict[str, Any]:
        self._stop_event.set()
        self._thread.join()
        elapsed = max(time.monotonic() - self._started_wall, 1e-9)
        total_cpu = time.process_time() - self._started_cpu
        return {
            "sample_interval_s": self._interval_s,
            "samples": len(self._rss),
            "elapsed_s": elapsed,
            "rss_start_bytes": self._rss[0] if self._rss else None,
            "rss_end_bytes": self._rss[-1] if self._rss else None,
            "rss_peak_bytes": max(self._rss) if self._rss else None,
            "cpu_average_percent": 100.0 * total_cpu / elapsed,
            "cpu_sample_peak_percent": max(self._cpu_percent) if self._cpu_percent else None,
            "open_fds_start": self._fds[0] if self._fds else None,
            "open_fds_end": self._fds[-1] if self._fds else None,
            "open_fds_peak": max(self._fds) if self._fds else None,
        }


class AsyncFrameWriter:
    """One ordered, bounded writer; saturation is a recording failure, not a drop."""

    _STOP = object()

    def __init__(
        self,
        session_dir: Path,
        metadata_stream: TextIO,
        queue_depth: int,
        save_function: Callable[[Path, int, D455Frame], Dict[str, Any]] = save_frame,
    ) -> None:
        if queue_depth <= 0:
            raise ValueError("writer queue depth must be positive")
        self._session_dir = session_dir
        self._metadata_stream = metadata_stream
        self._save_function = save_function
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_depth)
        self._records: List[Dict[str, Any]] = []
        self._error: Optional[BaseException] = None
        self._finished = False
        self._queue_max_observed = 0
        self._overflows = 0
        self._enqueued = 0
        self._enqueue_to_start_ms: List[float] = []
        self._write_ms: List[float] = []
        self._service_ms: List[float] = []
        self._thread = threading.Thread(
            target=self._run,
            name="d455-lossless-writer",
            daemon=False,
        )
        self._thread.start()

    @property
    def records(self) -> List[Dict[str, Any]]:
        if not self._finished:
            raise RuntimeError("writer records are only stable after finish")
        return list(self._records)

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise D455CaptureError(
                f"asynchronous frame writer failed: "
                f"{type(self._error).__name__}: {self._error}"
            ) from self._error

    def submit(self, index: int, frame: D455Frame) -> None:
        if self._finished:
            raise RuntimeError("cannot submit to a finished writer")
        self._raise_worker_error()
        item = (index, frame, time.monotonic_ns())
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            self._overflows += 1
            raise D455CaptureError(
                f"writer queue saturated at {self._queue.maxsize} frames; "
                "recording is incomplete"
            ) from exc
        self._enqueued += 1
        self._queue_max_observed = max(self._queue_max_observed, self._queue.qsize())

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                index, frame, enqueued_ns = item
                if self._error is not None:
                    continue
                writer_started_ns = time.monotonic_ns()
                record = self._save_function(self._session_dir, index, frame)
                files_completed_ns = time.monotonic_ns()
                enqueue_to_start_ms = (writer_started_ns - enqueued_ns) / 1e6
                write_ms = (files_completed_ns - writer_started_ns) / 1e6
                record["writer"] = {
                    "enqueued_host_monotonic_ns": enqueued_ns,
                    "started_host_monotonic_ns": writer_started_ns,
                    "files_fsynced_host_monotonic_ns": files_completed_ns,
                    "enqueue_to_start_ms": enqueue_to_start_ms,
                    "file_write_hash_fsync_ms": write_ms,
                }
                self._metadata_stream.write(json.dumps(record, sort_keys=True) + "\n")
                self._metadata_stream.flush()
                metadata_completed_ns = time.monotonic_ns()
                self._records.append(record)
                self._enqueue_to_start_ms.append(enqueue_to_start_ms)
                self._write_ms.append(write_ms)
                self._service_ms.append((metadata_completed_ns - writer_started_ns) / 1e6)
            except BaseException as exc:  # Preserve worker failures for the capture thread.
                self._error = exc
            finally:
                self._queue.task_done()

    def finish(self) -> None:
        if self._finished:
            self._raise_worker_error()
            return
        self._queue.put(self._STOP)
        self._queue.join()
        self._thread.join()
        self._metadata_stream.flush()
        os.fsync(self._metadata_stream.fileno())
        self._finished = True
        self._raise_worker_error()

    def statistics(self, captured_frames: int) -> Dict[str, Any]:
        return {
            "queue_capacity_frames": self._queue.maxsize,
            "queue_max_observed_frames": self._queue_max_observed,
            "queue_overflows": self._overflows,
            "captured_frames": captured_frames,
            "enqueued_frames": self._enqueued,
            "written_frames": len(self._records),
            "worker_error": (
                None
                if self._error is None
                else f"{type(self._error).__name__}: {self._error}"
            ),
            "enqueue_to_start_latency": _distribution(self._enqueue_to_start_ms),
            "file_write_hash_fsync_latency": _distribution(self._write_ms),
            "total_service_through_metadata_fsync_latency": _distribution(
                self._service_ms
            ),
        }


def summarize(
    records: List[Dict[str, Any]],
    allow_usb2: bool = False,
    maximum_skew_ms: float = 2.0,
    writer_statistics: Optional[Dict[str, Any]] = None,
    requested_frames: Optional[int] = None,
    requested_fps: Optional[float] = None,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty recording")
    depth_numbers = [record["depth_frame_number"] for record in records]
    color_numbers = [record["color_frame_number"] for record in records]
    depth_times = [record["depth_timestamp_ms"] for record in records]
    color_times = [record["color_timestamp_ms"] for record in records]
    host_times = [record["host_monotonic_ns_frameset_received"] for record in records]
    depth_domains = [record["depth_timestamp_domain"] for record in records]
    color_domains = [record["color_timestamp_domain"] for record in records]

    def dropped(numbers: List[int]) -> int:
        return sum(max(0, second - first - 1) for first, second in zip(numbers, numbers[1:]))

    def non_increasing(values: List[float]) -> int:
        return sum(second <= first for first, second in zip(values, values[1:]))

    def timing(values: List[float]) -> Dict[str, Any]:
        deltas = np.diff(np.asarray(values, dtype=np.float64))
        if deltas.size == 0:
            return {"mean_ms": None, "median_ms": None, "p95_ms": None}
        return {
            "mean_ms": float(np.mean(deltas)),
            "median_ms": float(np.median(deltas)),
            "p95_ms": float(np.percentile(deltas, 95)),
        }

    writer_pass = writer_statistics is None or (
        writer_statistics["queue_overflows"] == 0
        and writer_statistics["worker_error"] is None
        and writer_statistics["captured_frames"]
        == writer_statistics["enqueued_frames"]
        == writer_statistics["written_frames"]
        == len(records)
    )
    domain_pass = (
        all(depth == color for depth, color in zip(depth_domains, color_domains))
        and len(set(depth_domains)) == 1
        and len(set(color_domains)) == 1
    )
    completion_pass = (
        len(records) == requested_frames if requested_frames is not None else len(records) >= 2
    )
    period_ms = np.diff(np.asarray(color_times, dtype=np.float64))
    host_period_ms = np.diff(np.asarray(host_times, dtype=np.float64)) / 1e6
    if requested_fps is None:
        frame_rate_pass = len(records) >= 2
        requested_period_ms = None
    else:
        requested_period_ms = 1000.0 / float(requested_fps)
        frame_rate_pass = bool(
            period_ms.size > 0
            and abs(float(np.median(period_ms)) - requested_period_ms)
            <= requested_period_ms * 0.02
            and float(np.percentile(period_ms, 95)) <= requested_period_ms * 1.20
        )
    host_cadence_pass = (
        len(records) >= 2
        if requested_period_ms is None
        else bool(
            host_period_ms.size > 0
            and abs(float(np.median(host_period_ms)) - requested_period_ms)
            <= requested_period_ms * 0.10
            and float(np.percentile(host_period_ms, 95)) <= requested_period_ms * 1.50
        )
    )
    data_integrity_pass = (
        writer_pass
        and domain_pass
        and completion_pass
        and frame_rate_pass
        and host_cadence_pass
        and (
        dropped(depth_numbers) == 0
        and dropped(color_numbers) == 0
        and non_increasing(depth_numbers) == 0
        and non_increasing(color_numbers) == 0
        and non_increasing(depth_times) == 0
        and non_increasing(color_times) == 0
        and non_increasing(host_times) == 0
        and max(record["device_timestamp_skew_ms"] for record in records)
        <= maximum_skew_ms
        )
    )
    usb_superspeed = all(record["usb_superspeed"] for record in records)
    device_ns = np.asarray(color_times, dtype=np.float64) * 1e6
    host_ns = np.asarray(host_times, dtype=np.float64)
    if len(records) >= 2:
        device_centered = device_ns - device_ns[0]
        host_centered = host_ns - host_ns[0]
        denominator = float(device_centered @ device_centered)
        slope = (
            float(device_centered @ host_centered) / denominator
            if denominator > 0.0
            else None
        )
        residual_ms = (
            ((host_centered - slope * device_centered) / 1e6).tolist()
            if slope is not None
            else []
        )
    else:
        slope = None
        residual_ms = []
    result = {
        "captured_frames": len(records),
        "depth_frame_number_drops": dropped(depth_numbers),
        "color_frame_number_drops": dropped(color_numbers),
        "depth_frame_number_non_increasing": non_increasing(depth_numbers),
        "color_frame_number_non_increasing": non_increasing(color_numbers),
        "depth_timestamp_non_increasing": non_increasing(depth_times),
        "color_timestamp_non_increasing": non_increasing(color_times),
        "host_frameset_received_monotonic_non_increasing": non_increasing(host_times),
        "depth_period": timing(depth_times),
        "color_period": timing(color_times),
        "host_arrival_period": timing([value / 1e6 for value in host_times]),
        "max_device_timestamp_skew_ms": max(
            record["device_timestamp_skew_ms"] for record in records
        ),
        "maximum_allowed_skew_ms": maximum_skew_ms,
        "timestamp_domain_pass": domain_pass,
        "timestamp_domain": depth_domains[0] if domain_pass else None,
        "requested_frames": requested_frames,
        "planned_capture_complete": completion_pass,
        "requested_fps": requested_fps,
        "requested_period_ms": requested_period_ms,
        "frame_rate_pass": frame_rate_pass,
        "host_cadence_pass": host_cadence_pass,
        "mean_valid_depth_fraction": statistics.fmean(
            record["valid_depth_fraction"] for record in records
        ),
        "minimum_valid_depth_fraction": min(
            record["valid_depth_fraction"] for record in records
        ),
        "usb_superspeed": usb_superspeed,
        "writer_integrity_pass": writer_pass,
        "data_integrity_pass": data_integrity_pass,
        "deployment_link_pass": usb_superspeed,
        "usb2_degraded_mode_explicitly_allowed": bool(allow_usb2),
        "session_acceptance": (
            "PASS"
            if data_integrity_pass and usb_superspeed
            else "DEGRADED_USB2_ACCEPTED_FOR_DEVELOPMENT"
            if data_integrity_pass and allow_usb2 and not usb_superspeed
            else "NOT_PASS"
        ),
        "quality_pass": data_integrity_pass and usb_superspeed,
        "p2_development_gate": (
            "PASS" if data_integrity_pass and (usb_superspeed or allow_usb2) else "NOT_PASS"
        ),
        "development_dataset_eligible": bool(
            data_integrity_pass and (usb_superspeed or allow_usb2)
        ),
        "phase_progression_allowed": bool(
            data_integrity_pass and (usb_superspeed or allow_usb2)
        ),
        "evidence_scope": (
            "FORMAL_USB3_ONLY" if usb_superspeed else "USB2_DEVELOPMENT_ONLY"
        ),
        "usb3_session_candidate_status": (
            "USB3_SESSION_CANDIDATE_PASS"
            if data_integrity_pass and usb_superspeed
            else "NOT_APPLICABLE_USB2"
        ),
        "p2_formal_usb3_gate": (
            "PENDING_FULL_USB3_CERTIFICATION"
            if usb_superspeed
            else "DEFERRED_USB3_UNAVAILABLE"
        ),
        "formal_P2_pass": False,
        "capture_candidate_eligible": data_integrity_pass and usb_superspeed,
        "formal_dataset_eligible": False,
        "formal_dataset_eligibility_reason": "REQUIRES_OFFLINE_VERIFICATION",
        "device_to_host_time_mapping": {
            "device_clock": "color_timestamp_ms with its recorded timestamp_domain",
            "host_clock": "host_monotonic_ns_frameset_received (arrival, not exposure time)",
            "device_anchor_timestamp_ms": color_times[0],
            "host_anchor_monotonic_ns": host_times[0],
            "host_ns_per_device_ns": slope,
            "fit_residual": _distribution(residual_ms),
            "semantics": (
                "diagnostic affine fit to host arrival; includes transport/wait latency; "
                "not ROS time and not an exposure-time guarantee"
            ),
        },
    }
    if writer_statistics is not None:
        result["writer"] = writer_statistics
    return result


def record(
    args: argparse.Namespace,
    *,
    capture_factory: Callable[..., Any] = D455Capture,
    writer_factory: Callable[..., AsyncFrameWriter] = AsyncFrameWriter,
    manifest_writer: Callable[[Path, Any], None] = _write_json,
    signal_guard_factory: Callable[[], Any] = RecordingSignalGuard,
) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    session_name = args.session_name or (
        f"{args.scenario}_{time.strftime('%Y%m%dT%H%M%S', time.localtime())}"
    )
    session_dir = output_root / session_name
    if session_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing session: {session_dir}")
    session_dir.mkdir(parents=True)
    (session_dir / "rgb").mkdir()
    (session_dir / "raw_depth").mkdir()
    (session_dir / "aligned_depth").mkdir()
    records: List[Dict[str, Any]] = []
    captured_frames = 0
    writer_statistics: Optional[Dict[str, Any]] = None
    resource_statistics: Optional[Dict[str, Any]] = None
    failure_stage = "session_initialization"
    incomplete = {
        "schema_version": 2,
        "status": "INCOMPLETE",
        "failure_reason": {
            "code": "PROCESS_DID_NOT_COMPLETE",
            "stage": "recording",
            "exception_type": None,
            "message": (
                "recording has not completed; this marker also covers uncatchable "
                "process termination"
            ),
        },
        "scenario": args.scenario,
        "session_name": session_name,
        "created_host_wall_time_ns": time.time_ns(),
        "capture_plan": {
            "requested_frames": args.frames,
            "maximum_duration_s": args.duration_s,
            "requested_fps": args.fps,
            "warmup_frames": args.warmup,
            "stable_frames": args.stable_frames,
            "maximum_skew_ms": args.maximum_skew_ms,
            "usb2_degraded_mode_explicitly_allowed": bool(args.allow_usb2),
        },
    }
    metadata_path = session_dir / "frames.jsonl"
    try:
        with signal_guard_factory():
            failure_stage = "manifest_initial_write"
            manifest_writer(session_dir / "session.json", incomplete)
            failure_stage = "capture_start_or_frame_wait"
            with capture_factory(
                width=args.width,
                height=args.height,
                fps=args.fps,
                serial=args.serial,
                timeout_ms=args.timeout_ms,
                require_superspeed=not args.allow_usb2,
            ) as capture, metadata_path.open("x") as metadata_stream:
                for _ in range(args.warmup):
                    capture.wait_for_frame()
                capture.wait_for_stable_frames(
                    consecutive=args.stable_frames,
                    maximum_skew_ms=args.maximum_skew_ms,
                )
                writer = writer_factory(
                    session_dir,
                    metadata_stream,
                    queue_depth=args.writer_queue_depth,
                )
                resource_monitor = ResourceMonitor()
                started_ns = time.monotonic_ns()
                capture_error: Optional[BaseException] = None
                capture_failure_stage = failure_stage
                try:
                    while captured_frames < args.frames:
                        if (
                            args.duration_s is not None
                            and (time.monotonic_ns() - started_ns) / 1e9 >= args.duration_s
                        ):
                            break
                        failure_stage = "frame_wait"
                        frame = capture.wait_for_frame()
                        failure_stage = "writer_submit"
                        writer.submit(captured_frames, frame)
                        captured_frames += 1
                except BaseException as exc:
                    capture_error = exc
                    capture_failure_stage = failure_stage
                try:
                    failure_stage = "writer_finish"
                    writer.finish()
                except BaseException as exc:
                    if capture_error is None:
                        capture_error = exc
                        capture_failure_stage = failure_stage
                resource_statistics = resource_monitor.finish()
                records = writer.records
                writer_statistics = writer.statistics(captured_frames)
                if capture_error is not None:
                    failure_stage = capture_failure_stage
                    raise capture_error
                if captured_frames != args.frames:
                    raise D455CaptureError(
                        f"capture plan incomplete: requested {args.frames}, captured "
                        f"{captured_frames} before maximum duration"
                    )
                failure_stage = "data_integrity_summary"
                summary = summarize(
                    records,
                    allow_usb2=args.allow_usb2,
                    maximum_skew_ms=args.maximum_skew_ms,
                    writer_statistics=writer_statistics,
                    requested_frames=args.frames,
                    requested_fps=args.fps,
                )
                if not summary["data_integrity_pass"]:
                    raise D455CaptureError("captured session failed data-integrity gates")
                result = {
                    **{
                        key: value
                        for key, value in incomplete.items()
                        if key != "failure_reason"
                    },
                    "status": "CAPTURE_COMPLETE_UNVERIFIED",
                    "completed_host_wall_time_ns": time.time_ns(),
                    "device": capture.device_metadata,
                    "summary": summary,
                    "resources": resource_statistics,
                    "data_contract": {
                        "rgb": "lossless PNG decoded as RGB uint8",
                        "raw_depth": "lossless uint16 PNG in native depth pixel grid",
                        "aligned_depth": "lossless uint16 PNG in native Z16 units",
                        "metric_depth_formula": "depth_m = aligned_depth_raw * depth_scale_m_per_unit",
                        "intrinsics_frame": "color pixel grid after rs.align(color)",
                        "timestamps": (
                            "device color/depth plus bracketed host monotonic arrival and host wall; "
                            "device-to-host fit is diagnostic only"
                        ),
                        "writer": "ordered bounded queue; overflow or worker error fails closed",
                    },
                }
                failure_stage = "manifest_complete_write"
                manifest_writer(session_dir / "session.json", result)
    except BaseException as exc:
        incomplete["error"] = f"{type(exc).__name__}: {exc}"
        incomplete["failure_reason"] = _failure_reason(exc, failure_stage)
        incomplete["captured_frames_before_failure"] = captured_frames
        incomplete["written_frames_before_failure"] = len(records)
        if writer_statistics is not None:
            incomplete["writer"] = writer_statistics
        if resource_statistics is not None:
            incomplete["resources"] = resource_statistics
        try:
            manifest_writer(session_dir / "session.json", incomplete)
        except BaseException as manifest_exc:
            # A custom/injected writer may be the failed component.  Retry once
            # through the normal atomic writer so tests and recoverable failures
            # retain an explicit marker.  A real ENOSPC can make every write
            # impossible; in that case the pre-created INCOMPLETE manifest (when
            # available) remains non-acceptable and stderr still carries detail.
            incomplete["failure_manifest_write_error"] = (
                f"{type(manifest_exc).__name__}: {manifest_exc}"
            )
            try:
                _write_json(session_dir / "session.json", incomplete)
            except BaseException:
                pass
        raise
    return session_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--output-root", default=str(PROJECT_DIR / "recordings"))
    parser.add_argument("--session-name")
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--stable-frames", type=int, default=15)
    parser.add_argument("--maximum-skew-ms", type=float, default=2.0)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--writer-queue-depth", type=int, default=64)
    parser.add_argument(
        "--allow-usb2",
        action="store_true",
        help="explicitly accept a degraded USB2 development session",
    )
    args = parser.parse_args()
    if (
        args.frames <= 0
        or args.warmup < 0
        or args.stable_frames <= 0
        or args.writer_queue_depth <= 0
    ):
        parser.error("frames/stable-frames must be positive and warmup nonnegative")
    if args.duration_s is not None and args.duration_s <= 0.0:
        parser.error("duration-s must be positive")
    try:
        directory = record(args)
    except (D455CaptureError, OSError, ValueError) as exc:
        print(f"recording failed: {exc}", file=sys.stderr)
        return 2
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
