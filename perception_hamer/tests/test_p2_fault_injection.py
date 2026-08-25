#!/usr/bin/env python3
"""Safe P2 fault injection: mocks and temporary directories only."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.record_rgbd_session import (  # noqa: E402
    AsyncFrameWriter,
    _failure_reason,
    _write_json,
    record,
    save_frame,
    summarize,
)
from scripts.verify_rgbd_session import (  # noqa: E402
    SessionVerificationError,
    verify_session,
)
from src.d455_capture import D455Capture, D455CaptureError, D455Frame  # noqa: E402


def _intrinsics():
    return {
        "width": 6,
        "height": 4,
        "fx": 390.0,
        "fy": 391.0,
        "ppx": 3.0,
        "ppy": 2.0,
        "distortion_model": "none",
        "coeffs": [0.0] * 5,
    }


def make_frame(index: int = 0, *, usb: str = "2.1", skew_ms: float = 0.01) -> D455Frame:
    timestamp_ms = 100.0 + index * (1000.0 / 30.0)
    host_ns = 1_000_000_000 + index * 33_333_333
    rgb = np.full((4, 6, 3), index % 255, dtype=np.uint8)
    depth = np.full((4, 6), 1000 + index, dtype=np.uint16)
    return D455Frame(
        rgb=rgb,
        raw_depth_raw=depth.copy(),
        aligned_depth_raw=depth.copy(),
        depth_scale_m_per_unit=0.001,
        raw_depth_intrinsics=_intrinsics(),
        color_intrinsics=_intrinsics(),
        depth_to_color_extrinsics={
            "rotation_row_major": np.eye(3).reshape(-1).tolist(),
            "translation_m": [0.0, 0.0, 0.0],
        },
        aligned_to_color_intrinsics_differences={
            "width": 0.0,
            "height": 0.0,
            "fx": 0.0,
            "fy": 0.0,
            "ppx": 0.0,
            "ppy": 0.0,
        },
        color_frame_number=100 + index,
        depth_frame_number=100 + index,
        raw_color_frame_number=100 + index,
        raw_depth_frame_number=100 + index,
        color_timestamp_ms=timestamp_ms + skew_ms,
        depth_timestamp_ms=timestamp_ms,
        raw_color_timestamp_ms=timestamp_ms + skew_ms,
        raw_depth_timestamp_ms=timestamp_ms,
        color_timestamp_domain="global_time",
        depth_timestamp_domain="global_time",
        raw_color_timestamp_domain="global_time",
        raw_depth_timestamp_domain="global_time",
        host_monotonic_ns_before_wait=host_ns - 100,
        host_monotonic_ns_frameset_received=host_ns,
        host_monotonic_ns_alignment_completed=host_ns + 100,
        host_wall_time_ns_alignment_completed=host_ns + 200,
        device_serial="MOCK_NO_CAMERA",
        firmware_version="MOCK",
        usb_type_descriptor=usb,
    )


def recorder_args(output_root: Path, session_name: str = "fault_session", frames: int = 3):
    return argparse.Namespace(
        output_root=str(output_root),
        session_name=session_name,
        scenario="G00_STATIC",
        frames=frames,
        duration_s=None,
        fps=30,
        warmup=0,
        stable_frames=1,
        maximum_skew_ms=2.0,
        allow_usb2=True,
        width=6,
        height=4,
        serial=None,
        timeout_ms=100,
        writer_queue_depth=4,
    )


class MockCapture:
    def __init__(self, actions, **_kwargs):
        action_list = list(actions)
        self.actions = iter(action_list)
        self.first = next(
            (item for item in action_list if isinstance(item, D455Frame)),
            make_frame(0),
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    @property
    def device_metadata(self):
        metadata = self.first.metadata()
        return {
            key: metadata[key]
            for key in (
                "device_serial",
                "firmware_version",
                "usb_type_descriptor",
                "depth_scale_m_per_unit",
                "raw_depth_intrinsics",
                "color_intrinsics",
                "depth_to_color_extrinsics",
            )
        }

    def wait_for_stable_frames(self, **_kwargs):
        return self.first

    def wait_for_frame(self):
        action = next(self.actions)
        if isinstance(action, BaseException):
            raise action
        return action


def capture_factory(actions):
    return lambda **kwargs: MockCapture(actions, **kwargs)


def read_manifest(root: Path, session_name: str = "fault_session"):
    return json.loads((root / session_name / "session.json").read_text())


def assert_rejected(testcase: unittest.TestCase, session_dir: Path):
    with testcase.assertRaises(SessionVerificationError):
        verify_session(session_dir)


class FrameWaitFaultTest(unittest.TestCase):
    def test_frameset_timeout_is_wrapped_without_camera(self):
        class TimeoutPipeline:
            def wait_for_frames(self, timeout_ms):
                raise RuntimeError(f"Frame didn't arrive within {timeout_ms}")

        capture = D455Capture(width=6, height=4, fps=30, timeout_ms=17)
        capture._profile = object()
        capture._pipeline = TimeoutPipeline()
        capture._align = object()
        with self.assertRaisesRegex(D455CaptureError, "frame wait/alignment failed"):
            capture.wait_for_frame()

    def test_timeout_record_remains_incomplete_with_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root)
            timeout = D455CaptureError("frame wait/alignment failed: Frame timeout")
            with self.assertRaises(D455CaptureError):
                record(args, capture_factory=capture_factory([timeout]))
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertEqual(manifest["failure_reason"]["code"], "FRAMESET_TIMEOUT")
            assert_rejected(self, root / args.session_name)

    def test_mock_device_disconnect_fails_closed_without_unplugging_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root)
            disconnected = D455CaptureError("injected device disconnected")
            with self.assertRaises(D455CaptureError):
                record(args, capture_factory=capture_factory([disconnected]))
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertEqual(
                manifest["failure_reason"]["exception_type"], "D455CaptureError"
            )
            self.assertIn("disconnected", manifest["failure_reason"]["message"])
            assert_rejected(self, root / args.session_name)


class WriterFaultTest(unittest.TestCase):
    def test_writer_exception_is_propagated(self):
        def fail_save(_root, _index, _frame):
            raise RuntimeError("injected writer exception")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "frames.jsonl").open("x") as stream:
                writer = AsyncFrameWriter(root, stream, 2, fail_save)
                writer.submit(0, make_frame())
                with self.assertRaisesRegex(D455CaptureError, "writer failed"):
                    writer.finish()
            self.assertIn("RuntimeError", writer.statistics(1)["worker_error"])

    def test_writer_exception_record_has_rejected_failure_manifest(self):
        def fail_save(_root, _index, _frame):
            raise RuntimeError("injected writer exception")

        def writer_factory(root, stream, queue_depth):
            return AsyncFrameWriter(root, stream, queue_depth, fail_save)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, frames=1)
            with self.assertRaises(D455CaptureError):
                record(
                    args,
                    capture_factory=capture_factory([make_frame(0)]),
                    writer_factory=writer_factory,
                )
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertEqual(manifest["failure_reason"]["code"], "WRITER_EXCEPTION")
            self.assertIsNotNone(manifest["writer"]["worker_error"])
            assert_rejected(self, root / args.session_name)

    def test_mock_enospc_does_not_fill_disk_and_fails_closed(self):
        def enospc_save(_root, _index, _frame):
            raise OSError(errno.ENOSPC, "injected no space left on device")

        def writer_factory(root, stream, queue_depth):
            return AsyncFrameWriter(root, stream, queue_depth, enospc_save)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, frames=1)
            with self.assertRaises(D455CaptureError) as raised:
                record(
                    args,
                    capture_factory=capture_factory([make_frame(0)]),
                    writer_factory=writer_factory,
                )
            self.assertEqual(_failure_reason(raised.exception, "writer_finish")["code"], "ENOSPC")
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertEqual(manifest["failure_reason"]["code"], "ENOSPC")
            self.assertLess(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()), 100_000)
            assert_rejected(self, root / args.session_name)

    def test_queue_overflow_is_deterministic_and_fails_closed(self):
        started = threading.Event()
        release = threading.Event()

        def blocked_save(_root, index, _frame):
            started.set()
            release.wait(timeout=1.0)
            return {"index": index}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "frames.jsonl").open("x") as stream:
                writer = AsyncFrameWriter(root, stream, 1, blocked_save)
                writer.submit(0, make_frame(0))
                self.assertTrue(started.wait(timeout=1.0))
                writer.submit(1, make_frame(1))
                with self.assertRaisesRegex(D455CaptureError, "queue saturated"):
                    writer.submit(2, make_frame(2))
                release.set()
                writer.finish()
            self.assertEqual(writer.statistics(2)["queue_overflows"], 1)

    def test_queue_overflow_record_has_explicit_failure_reason(self):
        release = threading.Event()

        def blocked_save(_root, index, _frame):
            release.wait(timeout=1.0)
            return {"index": index}

        def writer_factory(root, stream, queue_depth):
            del queue_depth
            return AsyncFrameWriter(root, stream, 1, blocked_save)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, frames=20)
            timer = threading.Timer(0.2, release.set)
            timer.start()
            try:
                with self.assertRaisesRegex(D455CaptureError, "queue saturated"):
                    record(
                        args,
                        capture_factory=capture_factory(
                            [make_frame(index) for index in range(20)]
                        ),
                        writer_factory=writer_factory,
                    )
            finally:
                release.set()
                timer.cancel()
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertEqual(
                manifest["failure_reason"]["code"], "WRITER_QUEUE_OVERFLOW"
            )
            self.assertGreater(manifest["writer"]["queue_overflows"], 0)
            assert_rejected(self, root / args.session_name)

    def test_png_imwrite_false_is_a_worker_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rgb", "raw_depth", "aligned_depth"):
                (root / name).mkdir()
            with mock.patch("scripts.record_rgbd_session.cv2.imwrite", return_value=False):
                with self.assertRaisesRegex(OSError, "failed to save RGB"):
                    save_frame(root, 0, make_frame())

    def test_png_failure_record_has_explicit_failure_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, frames=1)
            with mock.patch("scripts.record_rgbd_session.cv2.imwrite", return_value=False):
                with self.assertRaises(D455CaptureError):
                    record(
                        args,
                        capture_factory=capture_factory([make_frame(0)]),
                    )
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertEqual(manifest["failure_reason"]["code"], "PNG_WRITE_FAILED")
            assert_rejected(self, root / args.session_name)


class ManifestAndConflictFaultTest(unittest.TestCase):
    def test_initial_manifest_failure_gets_best_effort_failure_marker(self):
        def injected_writer(_path, _value):
            raise OSError("injected initial manifest failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, frames=1)
            with self.assertRaisesRegex(OSError, "manifest failure"):
                record(
                    args,
                    capture_factory=capture_factory([make_frame(0)]),
                    manifest_writer=injected_writer,
                )
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertEqual(manifest["failure_reason"]["code"], "MANIFEST_WRITE_FAILED")
            self.assertIn("failure_manifest_write_error", manifest)
            assert_rejected(self, root / args.session_name)

    def test_final_manifest_failure_restores_incomplete_marker(self):
        calls = 0

        def injected_writer(path, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected final manifest failure")
            _write_json(path, value)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, frames=3)
            frames = [make_frame(index) for index in range(3)]
            with self.assertRaisesRegex(OSError, "manifest failure"):
                record(
                    args,
                    capture_factory=capture_factory(frames),
                    manifest_writer=injected_writer,
                )
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertEqual(manifest["failure_reason"]["code"], "MANIFEST_WRITE_FAILED")
            assert_rejected(self, root / args.session_name)

    def test_frame_file_conflict_never_overwrites_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rgb", "raw_depth", "aligned_depth"):
                (root / name).mkdir()
            target = root / "rgb" / "000000.png"
            target.write_bytes(b"sentinel")
            with self.assertRaises(FileExistsError):
                save_frame(root, 0, make_frame())
            self.assertEqual(target.read_bytes(), b"sentinel")
            self.assertFalse((root / "raw_depth" / "000000.png").exists())

    def test_existing_session_directory_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "fault_session"
            session.mkdir()
            sentinel = session / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                record(recorder_args(root), capture_factory=capture_factory([]))
            self.assertEqual(sentinel.read_text(), "keep")


class TemporalAndPathGateTest(unittest.TestCase):
    def _metadata_pair(self):
        return [make_frame(0).metadata(), make_frame(1).metadata()]

    def test_frame_number_rollback_fails_integrity(self):
        records = self._metadata_pair()
        records[1]["depth_frame_number"] = records[0]["depth_frame_number"] - 1
        records[1]["color_frame_number"] = records[0]["color_frame_number"] - 1
        result = summarize(records, allow_usb2=True, requested_frames=2, requested_fps=30)
        self.assertFalse(result["data_integrity_pass"])
        self.assertGreater(result["depth_frame_number_non_increasing"], 0)

    def test_device_timestamp_rollback_fails_integrity(self):
        records = self._metadata_pair()
        records[1]["depth_timestamp_ms"] = records[0]["depth_timestamp_ms"] - 1.0
        records[1]["color_timestamp_ms"] = records[0]["color_timestamp_ms"] - 1.0
        result = summarize(records, allow_usb2=True, requested_frames=2, requested_fps=30)
        self.assertFalse(result["data_integrity_pass"])
        self.assertGreater(result["color_timestamp_non_increasing"], 0)

    def test_rgb_depth_desynchronization_fails_integrity(self):
        records = [make_frame(0, skew_ms=3.0).metadata(), make_frame(1, skew_ms=3.0).metadata()]
        result = summarize(
            records,
            allow_usb2=True,
            maximum_skew_ms=2.0,
            requested_frames=2,
            requested_fps=30,
        )
        self.assertFalse(result["data_integrity_pass"])
        self.assertGreater(result["max_device_timestamp_skew_ms"], 2.0)

    def test_duplicate_paths_are_rejected_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rgb", "raw_depth", "aligned_depth"):
                (root / name).mkdir()
            records = [save_frame(root, index, make_frame(index)) for index in range(2)]
            writer_stats = {
                "queue_capacity_frames": 2,
                "queue_max_observed_frames": 1,
                "queue_overflows": 0,
                "captured_frames": 2,
                "enqueued_frames": 2,
                "written_frames": 2,
                "worker_error": None,
                "enqueue_to_start_latency": {},
                "file_write_hash_fsync_latency": {},
                "total_service_through_metadata_fsync_latency": {},
            }
            summary = summarize(
                records,
                allow_usb2=True,
                writer_statistics=writer_stats,
                requested_frames=2,
                requested_fps=30,
            )
            records[1]["rgb_path"] = records[0]["rgb_path"]
            records[1]["rgb_sha256"] = records[0]["rgb_sha256"]
            (root / "frames.jsonl").write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
            )
            _write_json(
                root / "session.json",
                {
                    "schema_version": 2,
                    "status": "CAPTURE_COMPLETE_UNVERIFIED",
                    "scenario": "G00_STATIC",
                    "session_name": "duplicate",
                    "created_host_wall_time_ns": 1,
                    "completed_host_wall_time_ns": 2,
                    "device": {
                        key: records[0][key]
                        for key in (
                            "device_serial",
                            "firmware_version",
                            "usb_type_descriptor",
                            "depth_scale_m_per_unit",
                            "raw_depth_intrinsics",
                            "color_intrinsics",
                            "depth_to_color_extrinsics",
                        )
                    },
                    "capture_plan": {
                        "requested_frames": 2,
                        "requested_fps": 30,
                        "maximum_skew_ms": 2.0,
                        "usb2_degraded_mode_explicitly_allowed": True,
                    },
                    "summary": summary,
                    "data_contract": {},
                },
            )
            assert_rejected(self, root)

    def test_usb2_development_allowance_cannot_set_formal_gate(self):
        records = self._metadata_pair()
        result = summarize(records, allow_usb2=True, requested_frames=2, requested_fps=30)
        self.assertTrue(result["data_integrity_pass"])
        self.assertFalse(result["deployment_link_pass"])
        self.assertFalse(result["quality_pass"])
        self.assertFalse(result["capture_candidate_eligible"])
        self.assertFalse(result["formal_dataset_eligible"])
        self.assertEqual(result["session_acceptance"], "DEGRADED_USB2_ACCEPTED_FOR_DEVELOPMENT")
        self.assertEqual(result["p2_development_gate"], "PASS")
        self.assertTrue(result["development_dataset_eligible"])
        self.assertTrue(result["phase_progression_allowed"])
        self.assertEqual(result["evidence_scope"], "USB2_DEVELOPMENT_ONLY")
        self.assertEqual(
            result["p2_formal_usb3_gate"], "DEFERRED_USB3_UNAVAILABLE"
        )
        self.assertFalse(result["formal_P2_pass"])

    def test_single_usb3_session_cannot_bypass_full_formal_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, session_name="usb3_candidate", frames=2)
            args.allow_usb2 = False
            session_dir = record(
                args,
                capture_factory=capture_factory(
                    [make_frame(0, usb="3.2"), make_frame(1, usb="3.2")]
                ),
            )
            clean_manifest = json.loads((session_dir / "session.json").read_text())
            self.assertNotIn("failure_reason", clean_manifest)
            summary = clean_manifest["summary"]
            self.assertTrue(summary["capture_candidate_eligible"])
            self.assertFalse(summary["formal_dataset_eligible"])
            self.assertFalse(summary["formal_P2_pass"])
            self.assertNotEqual(summary["p2_formal_usb3_gate"], "PASS")

            verification = verify_session(session_dir)
            self.assertEqual(verification["verification"], "PASS")
            self.assertTrue(verification["usb3_session_candidate_verified"])
            self.assertFalse(verification["formal_dataset_eligible"])
            self.assertFalse(verification["formal_P2_pass"])
            self.assertNotEqual(verification["p2_formal_usb3_gate"], "PASS")

    def test_legacy_schema_v2_summary_is_compatible_but_gates_are_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, session_name="legacy", frames=2)
            session_dir = record(
                args,
                capture_factory=capture_factory([make_frame(0), make_frame(1)]),
            )
            manifest_path = session_dir / "session.json"
            manifest = json.loads(manifest_path.read_text())
            for field in (
                "p2_development_gate",
                "development_dataset_eligible",
                "phase_progression_allowed",
                "evidence_scope",
                "usb3_session_candidate_status",
                "p2_formal_usb3_gate",
                "formal_P2_pass",
            ):
                manifest["summary"].pop(field)
            _write_json(manifest_path, manifest)

            verification = verify_session(session_dir)
            self.assertEqual(verification["verification"], "PASS")
            self.assertEqual(verification["p2_development_gate"], "PASS")
            self.assertEqual(
                verification["p2_formal_usb3_gate"], "DEFERRED_USB3_UNAVAILABLE"
            )
            self.assertFalse(verification["formal_P2_pass"])
            self.assertFalse(verification["formal_dataset_eligible"])

    def test_forged_recorded_formal_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = recorder_args(root, session_name="forged", frames=2)
            session_dir = record(
                args,
                capture_factory=capture_factory([make_frame(0), make_frame(1)]),
            )
            manifest_path = session_dir / "session.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["summary"]["formal_P2_pass"] = True
            manifest["summary"]["formal_dataset_eligible"] = True
            _write_json(manifest_path, manifest)
            assert_rejected(self, session_dir)


class ProcessTerminationTest(unittest.TestCase):
    HARNESS = PROJECT_DIR / "tests" / "p2_fault_signal_child.py"

    def _start_child(self, root: Path, mode: str):
        ready = root / "ready"
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.HARNESS),
                "--output-root",
                str(root),
                "--ready",
                str(ready),
                "--mode",
                mode,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not ready.exists() and process.poll() is None:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "mock child did not reach its signal point")
        return process

    def test_real_sigint_and_sigterm_leave_rejected_incomplete_session(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signal.Signals(signum).name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    process = self._start_child(root, "signal")
                    process.send_signal(signum)
                    stdout, stderr = process.communicate(timeout=5.0)
                    self.assertEqual(process.returncode, 2, stdout + stderr)
                    manifest = read_manifest(root)
                    self.assertEqual(manifest["status"], "INCOMPLETE")
                    self.assertEqual(
                        manifest["failure_reason"]["code"],
                        f"SIGNAL_{signal.Signals(signum).name}",
                    )
                    assert_rejected(self, root / "fault_session")

    def test_uncatchable_abnormal_exit_keeps_precreated_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = self._start_child(root, "abrupt")
            stdout, stderr = process.communicate(timeout=5.0)
            self.assertEqual(process.returncode, 23, stdout + stderr)
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "INCOMPLETE")
            self.assertNotIn("summary", manifest)
            self.assertEqual(
                manifest["failure_reason"]["code"], "PROCESS_DID_NOT_COMPLETE"
            )
            assert_rejected(self, root / "fault_session")


if __name__ == "__main__":
    unittest.main()
