#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math
import sys
import tempfile
import threading
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.d455_capture import (  # noqa: E402
    D455Capture,
    D455CaptureError,
    D455Frame,
    extrinsics_to_dict,
    intrinsics_to_dict,
)
from scripts.record_rgbd_session import (  # noqa: E402
    AsyncFrameWriter,
    _write_json,
    save_frame,
    summarize,
)
from scripts.verify_rgbd_session import (  # noqa: E402
    SessionVerificationError,
    verify_session,
)


class _Intrinsics:
    width = 6
    height = 4
    fx = 390.0
    fy = 391.0
    ppx = 320.0
    ppy = 240.0
    model = "distortion.inverse_brown_conrady"
    coeffs = [0.0] * 5


def make_frame() -> D455Frame:
    y, x = np.indices((4, 6))
    rgb = np.stack((x * 11 + 1, y * 17 + 2, x * 3 + y * 5 + 3), axis=2).astype(
        np.uint8
    )
    raw_depth = (np.arange(24, dtype=np.uint16).reshape(4, 6) + 100).copy()
    aligned_depth = (np.flipud(raw_depth) + 1000).copy()
    return D455Frame(
        rgb=rgb,
        raw_depth_raw=raw_depth,
        aligned_depth_raw=aligned_depth,
        depth_scale_m_per_unit=0.001,
        raw_depth_intrinsics=intrinsics_to_dict(_Intrinsics()),
        color_intrinsics=intrinsics_to_dict(_Intrinsics()),
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
        color_frame_number=12,
        depth_frame_number=10,
        raw_color_frame_number=12,
        raw_depth_frame_number=10,
        color_timestamp_ms=101.01,
        depth_timestamp_ms=101.00,
        raw_color_timestamp_ms=101.01,
        raw_depth_timestamp_ms=101.00,
        color_timestamp_domain="global_time",
        depth_timestamp_domain="global_time",
        raw_color_timestamp_domain="global_time",
        raw_depth_timestamp_domain="global_time",
        host_monotonic_ns_before_wait=100,
        host_monotonic_ns_frameset_received=150,
        host_monotonic_ns_alignment_completed=200,
        host_wall_time_ns_alignment_completed=300,
        device_serial="serial",
        firmware_version="firmware",
        usb_type_descriptor="2.1",
    )


class D455FrameContractTest(unittest.TestCase):
    def test_metric_depth_and_metadata(self):
        frame = make_frame()
        np.testing.assert_allclose(
            frame.aligned_depth_m, frame.aligned_depth_raw.astype(np.float32) * 0.001
        )
        metadata = frame.metadata()
        self.assertFalse(metadata["usb_superspeed"])
        self.assertEqual(metadata["depth_alignment_target"], "color")
        self.assertAlmostEqual(metadata["device_timestamp_skew_ms"], 0.01)
        self.assertAlmostEqual(metadata["valid_depth_fraction"], 1.0)

    def test_shape_dtype_and_time_fail_closed(self):
        base = make_frame()
        values = dict(base.__dict__)
        values["aligned_depth_raw"] = np.zeros((3, 6), dtype=np.uint16)
        with self.assertRaises(D455CaptureError):
            D455Frame(**values)
        values = dict(base.__dict__)
        values["rgb"] = np.zeros((4, 6, 3), dtype=np.float32)
        with self.assertRaises(D455CaptureError):
            D455Frame(**values)
        values = dict(base.__dict__)
        values["depth_timestamp_ms"] = math.nan
        with self.assertRaises(D455CaptureError):
            D455Frame(**values)
        values = dict(base.__dict__)
        values["host_monotonic_ns_alignment_completed"] = 99
        with self.assertRaises(D455CaptureError):
            D455Frame(**values)

    def test_intrinsics_validation(self):
        intrinsics = intrinsics_to_dict(_Intrinsics())
        self.assertEqual(intrinsics["width"], 6)
        self.assertEqual(intrinsics["coeffs"], [0.0] * 5)
        class BadIntrinsics(_Intrinsics):
            pass

        bad = BadIntrinsics()
        bad.fx = math.inf
        with self.assertRaises(D455CaptureError):
            intrinsics_to_dict(bad)

    def test_capture_configuration_validation(self):
        with self.assertRaises(ValueError):
            D455Capture(fps=0)
        with self.assertRaises(ValueError):
            D455Capture(device_models=("D415",))

    def test_intrinsics_dimensions_match_arrays(self):
        frame = make_frame()
        values = dict(frame.__dict__)
        values["raw_depth_intrinsics"] = {**frame.raw_depth_intrinsics, "width": 7}
        with self.assertRaises(D455CaptureError):
            D455Frame(**values)

    def test_extrinsics_column_major_conversion(self):
        angle = math.radians(17.0)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        class Extrinsics:
            pass

        source = Extrinsics()
        source.rotation = rotation.reshape(-1, order="F").tolist()
        source.translation = [0.01, -0.02, 0.03]
        converted = extrinsics_to_dict(source)
        np.testing.assert_allclose(
            np.asarray(converted["rotation_row_major"]).reshape(3, 3), rotation
        )
        point = np.asarray([0.2, -0.1, 0.7])
        np.testing.assert_allclose(
            rotation @ point + np.asarray(source.translation),
            np.asarray(converted["rotation_row_major"]).reshape(3, 3) @ point
            + np.asarray(converted["translation_m"]),
        )


class RecorderContractTest(unittest.TestCase):
    def test_lossless_three_stream_save_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rgb", "raw_depth", "aligned_depth"):
                (root / name).mkdir()
            frame = make_frame()
            record = save_frame(root, 0, frame)
            for path_key, hash_key in (
                ("rgb_path", "rgb_sha256"),
                ("raw_depth_path", "raw_depth_sha256"),
                ("aligned_depth_path", "aligned_depth_sha256"),
            ):
                content = (root / record[path_key]).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), record[hash_key])
            raw = __import__("cv2").imread(
                str(root / record["raw_depth_path"]),
                __import__("cv2").IMREAD_UNCHANGED,
            )
            aligned = __import__("cv2").imread(
                str(root / record["aligned_depth_path"]),
                __import__("cv2").IMREAD_UNCHANGED,
            )
            rgb_bgr = __import__("cv2").imread(
                str(root / record["rgb_path"]),
                __import__("cv2").IMREAD_UNCHANGED,
            )
            np.testing.assert_array_equal(raw, frame.raw_depth_raw)
            np.testing.assert_array_equal(aligned, frame.aligned_depth_raw)
            np.testing.assert_array_equal(
                __import__("cv2").cvtColor(rgb_bgr, __import__("cv2").COLOR_BGR2RGB),
                frame.rgb,
            )
            self.assertFalse(np.array_equal(raw, aligned))
            self.assertEqual(record["roi"]["reason"], "NOT_RUN_P3")
            self.assertEqual(record["klt_features"]["reason"], "NOT_RUN_P4")

    def test_summary_detects_frame_drop(self):
        first = make_frame().metadata()
        second = dict(first)
        second.update(
            {
                "depth_frame_number": first["depth_frame_number"] + 2,
                "color_frame_number": first["color_frame_number"] + 1,
                "depth_timestamp_ms": first["depth_timestamp_ms"] + 33.0,
                "color_timestamp_ms": first["color_timestamp_ms"] + 33.0,
                "host_monotonic_ns_frameset_received": (
                    first["host_monotonic_ns_frameset_received"] + 33_000_000
                ),
                "host_monotonic_ns_alignment_completed": (
                    first["host_monotonic_ns_alignment_completed"] + 33_000_000
                ),
            }
        )
        result = summarize([first, second], allow_usb2=True)
        self.assertEqual(result["depth_frame_number_drops"], 1)
        self.assertEqual(result["color_frame_number_drops"], 0)
        self.assertFalse(result["quality_pass"])

    def test_usb2_integrity_is_not_deployment_pass(self):
        first = make_frame().metadata()
        second = dict(first)
        second.update(
            {
                "depth_frame_number": first["depth_frame_number"] + 1,
                "color_frame_number": first["color_frame_number"] + 1,
                "depth_timestamp_ms": first["depth_timestamp_ms"] + 33.0,
                "color_timestamp_ms": first["color_timestamp_ms"] + 33.0,
                "host_monotonic_ns_frameset_received": (
                    first["host_monotonic_ns_frameset_received"] + 33_000_000
                ),
                "host_monotonic_ns_alignment_completed": (
                    first["host_monotonic_ns_alignment_completed"] + 33_000_000
                ),
            }
        )
        result = summarize([first, second], allow_usb2=True)
        self.assertTrue(result["data_integrity_pass"])
        self.assertFalse(result["deployment_link_pass"])
        self.assertFalse(result["quality_pass"])
        self.assertEqual(
            result["session_acceptance"], "DEGRADED_USB2_ACCEPTED_FOR_DEVELOPMENT"
        )

    def test_duplicate_frame_and_timestamp_fail_integrity(self):
        first = make_frame().metadata()
        duplicate = dict(first)
        result = summarize([first, duplicate], allow_usb2=True)
        self.assertGreater(result["depth_frame_number_non_increasing"], 0)
        self.assertGreater(result["depth_timestamp_non_increasing"], 0)
        self.assertFalse(result["data_integrity_pass"])
        self.assertEqual(result["session_acceptance"], "NOT_PASS")

    def test_one_frame_or_changed_clock_domain_does_not_pass(self):
        first = make_frame().metadata()
        self.assertFalse(summarize([first], allow_usb2=True)["data_integrity_pass"])
        second = dict(first)
        second.update(
            {
                "depth_frame_number": first["depth_frame_number"] + 1,
                "color_frame_number": first["color_frame_number"] + 1,
                "depth_timestamp_ms": first["depth_timestamp_ms"] + 33.333,
                "color_timestamp_ms": first["color_timestamp_ms"] + 33.333,
                "host_monotonic_ns_frameset_received": (
                    first["host_monotonic_ns_frameset_received"] + 33_333_000
                ),
                "host_monotonic_ns_alignment_completed": (
                    first["host_monotonic_ns_alignment_completed"] + 33_333_000
                ),
                "color_timestamp_domain": "hardware_clock",
            }
        )
        result = summarize([first, second], allow_usb2=True, requested_fps=30)
        self.assertFalse(result["timestamp_domain_pass"])
        self.assertFalse(result["data_integrity_pass"])

    def test_planned_count_and_frame_rate_are_gates(self):
        first = make_frame().metadata()
        second = dict(first)
        second.update(
            {
                "depth_frame_number": first["depth_frame_number"] + 1,
                "color_frame_number": first["color_frame_number"] + 1,
                "depth_timestamp_ms": first["depth_timestamp_ms"] + 60.0,
                "color_timestamp_ms": first["color_timestamp_ms"] + 60.0,
                "host_monotonic_ns_frameset_received": (
                    first["host_monotonic_ns_frameset_received"] + 60_000_000
                ),
                "host_monotonic_ns_alignment_completed": (
                    first["host_monotonic_ns_alignment_completed"] + 60_000_000
                ),
            }
        )
        result = summarize(
            [first, second], allow_usb2=True, requested_frames=3, requested_fps=30
        )
        self.assertFalse(result["planned_capture_complete"])
        self.assertFalse(result["frame_rate_pass"])
        self.assertFalse(result["data_integrity_pass"])

    def test_host_delivery_burst_does_not_pass(self):
        records = []
        base = make_frame().metadata()
        for index in range(10):
            item = dict(base)
            item.update(
                {
                    "depth_frame_number": base["depth_frame_number"] + index,
                    "color_frame_number": base["color_frame_number"] + index,
                    "depth_timestamp_ms": base["depth_timestamp_ms"] + 33.333 * index,
                    "color_timestamp_ms": base["color_timestamp_ms"] + 33.333 * index,
                    "host_monotonic_ns_frameset_received": (
                        base["host_monotonic_ns_frameset_received"] + 1_000_000 * index
                    ),
                    "host_monotonic_ns_alignment_completed": (
                        base["host_monotonic_ns_alignment_completed"]
                        + 1_000_000 * index
                    ),
                }
            )
            records.append(item)
        result = summarize(records, requested_frames=10, requested_fps=30)
        self.assertFalse(result["host_cadence_pass"])
        self.assertFalse(result["data_integrity_pass"])

    def test_async_writer_is_ordered(self):
        records = []

        def fake_save(_root, index, _frame):
            return {"index": index}

        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "frames.jsonl").open("w") as stream:
                writer = AsyncFrameWriter(Path(directory), stream, 4, fake_save)
                for index in range(3):
                    writer.submit(index, make_frame())
                writer.finish()
                records = writer.records
        self.assertEqual([item["index"] for item in records], [0, 1, 2])
        self.assertEqual(writer.statistics(3)["written_frames"], 3)

    def test_async_writer_queue_overflow_fails_closed(self):
        started = threading.Event()
        release = threading.Event()

        def blocked_save(_root, index, _frame):
            started.set()
            release.wait(timeout=2.0)
            return {"index": index}

        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "frames.jsonl").open("w") as stream:
                writer = AsyncFrameWriter(Path(directory), stream, 1, blocked_save)
                writer.submit(0, make_frame())
                self.assertTrue(started.wait(timeout=1.0))
                writer.submit(1, make_frame())
                with self.assertRaises(D455CaptureError):
                    writer.submit(2, make_frame())
                release.set()
                writer.finish()
        self.assertEqual(writer.statistics(2)["queue_overflows"], 1)

    def test_round_trip_verifier_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rgb", "raw_depth", "aligned_depth"):
                (root / name).mkdir()
            first = save_frame(root, 0, make_frame())
            second_frame = make_frame()
            values = dict(second_frame.__dict__)
            values.update(
                {
                    "depth_frame_number": second_frame.depth_frame_number + 1,
                    "color_frame_number": second_frame.color_frame_number + 1,
                    "raw_depth_frame_number": second_frame.raw_depth_frame_number + 1,
                    "raw_color_frame_number": second_frame.raw_color_frame_number + 1,
                    "depth_timestamp_ms": second_frame.depth_timestamp_ms + 33.333,
                    "color_timestamp_ms": second_frame.color_timestamp_ms + 33.333,
                    "raw_depth_timestamp_ms": (
                        second_frame.raw_depth_timestamp_ms + 33.333
                    ),
                    "raw_color_timestamp_ms": (
                        second_frame.raw_color_timestamp_ms + 33.333
                    ),
                    "host_monotonic_ns_before_wait": 33_333_100,
                    "host_monotonic_ns_frameset_received": 33_333_150,
                    "host_monotonic_ns_alignment_completed": 33_333_200,
                }
            )
            second = save_frame(root, 1, D455Frame(**values))
            records = [first, second]
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
            (root / "frames.jsonl").write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
            )
            _write_json(
                root / "session.json",
                {
                    "schema_version": 2,
                    "status": "CAPTURE_COMPLETE_UNVERIFIED",
                    "scenario": "G00_STATIC",
                    "session_name": "test",
                    "created_host_wall_time_ns": 1,
                    "completed_host_wall_time_ns": 2,
                    "device": {
                        key: first[key]
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
            self.assertEqual(verify_session(root)["verification"], "PASS")
            (root / first["rgb_path"]).write_bytes(b"tampered")
            with self.assertRaises(SessionVerificationError):
                verify_session(root)

    def test_verifier_rejects_forged_usb_flag_and_reused_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rgb", "raw_depth", "aligned_depth"):
                (root / name).mkdir()
            first = save_frame(root, 0, make_frame())
            second_frame = make_frame()
            values = dict(second_frame.__dict__)
            values.update(
                {
                    "depth_frame_number": second_frame.depth_frame_number + 1,
                    "color_frame_number": second_frame.color_frame_number + 1,
                    "raw_depth_frame_number": second_frame.raw_depth_frame_number + 1,
                    "raw_color_frame_number": second_frame.raw_color_frame_number + 1,
                    "depth_timestamp_ms": second_frame.depth_timestamp_ms + 33.333,
                    "color_timestamp_ms": second_frame.color_timestamp_ms + 33.333,
                    "raw_depth_timestamp_ms": (
                        second_frame.raw_depth_timestamp_ms + 33.333
                    ),
                    "raw_color_timestamp_ms": (
                        second_frame.raw_color_timestamp_ms + 33.333
                    ),
                    "host_monotonic_ns_before_wait": 33_333_100,
                    "host_monotonic_ns_frameset_received": 33_333_150,
                    "host_monotonic_ns_alignment_completed": 33_333_200,
                }
            )
            second = save_frame(root, 1, D455Frame(**values))
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

            def write_session(records):
                summary = summarize(
                    records,
                    allow_usb2=True,
                    writer_statistics=writer_stats,
                    requested_frames=2,
                    requested_fps=30,
                )
                (root / "frames.jsonl").write_text(
                    "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
                )
                _write_json(
                    root / "session.json",
                    {
                        "schema_version": 2,
                        "status": "CAPTURE_COMPLETE_UNVERIFIED",
                        "scenario": "G00_STATIC",
                        "session_name": "test",
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

            forged = [dict(first), dict(second)]
            for record in forged:
                record["usb_superspeed"] = True
            write_session(forged)
            with self.assertRaises(SessionVerificationError):
                verify_session(root)

            reused = [dict(first), dict(second)]
            for key in ("rgb_path", "raw_depth_path", "aligned_depth_path"):
                reused[1][key] = reused[0][key]
            for key in ("rgb_sha256", "raw_depth_sha256", "aligned_depth_sha256"):
                reused[1][key] = reused[0][key]
            write_session(reused)
            with self.assertRaises(SessionVerificationError):
                verify_session(root)


if __name__ == "__main__":
    unittest.main()
