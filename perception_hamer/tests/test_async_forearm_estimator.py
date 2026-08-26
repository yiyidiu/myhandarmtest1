#!/usr/bin/env python3

from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_hamer.src.forearm_fusion import (  # noqa: E402
    ForearmFusionConfig,
    ForearmObservation,
    LatestOnlyForearmEstimator,
)


def _observation(confidence=0.9):
    return ForearmObservation(
        valid=True,
        axis=np.asarray([1.0, 0.0, 0.0]),
        center_m=np.asarray([0.0, 0.0, 0.7]),
        confidence=float(confidence),
        reason="ok",
        status="tracking",
        age_s=0.0,
        span_m=0.12,
        axis_ratio=3.0,
        centerline_rms_m=0.002,
        point_count=500,
        cross_section_count=6,
        wrist_pixel=np.asarray([400.0, 240.0]),
        proximal_pixel=np.asarray([260.0, 240.0]),
        processing_ms=2.0,
    )


class _FakeEstimator:
    def __init__(self, block_first=False):
        self.reset_count = 0
        self.update_count = 0
        self.update_times = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.block_first = bool(block_first)

    def reset(self):
        self.reset_count += 1

    def update(self, *_args, **_kwargs):
        self.update_count += 1
        self.update_times.append(time.monotonic())
        if self.block_first and self.update_count == 1:
            self.first_started.set()
            if not self.release_first.wait(2.0):
                raise RuntimeError("test did not release first forearm job")
        return _observation()


def _submit(worker, sequence, identity, source_monotonic):
    return worker.submit(
        np.full((12, 16), 700, dtype=np.uint16),
        0.001,
        {
            "width": 16,
            "height": 12,
            "fx": 20.0,
            "fy": 20.0,
            "ppx": 8.0,
            "ppy": 6.0,
        },
        [0.1, 0.0, 0.7],
        {
            "valid": True,
            "confidence": 0.95,
            "wrist_pixel": [10.0, 6.0],
            "palm_mcp_pixels": [[11.0, 5.0]] * 4,
        },
        0.01,
        identity,
        sequence,
        source_monotonic,
    )


def _wait_for_completed(worker, count, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if worker.statistics["completed"] >= count:
            return
        time.sleep(0.005)
    raise AssertionError("async forearm worker did not complete in time")


class LatestOnlyForearmEstimatorTest(unittest.TestCase):
    def test_fresh_result_is_age_attenuated_and_identity_bound(self):
        fake = _FakeEstimator()
        worker = LatestOnlyForearmEstimator(
            ForearmFusionConfig(minimum_confidence=0.42), estimator=fake
        )
        try:
            identity = (3, 7, True, 2)
            source = time.monotonic() - 0.01
            _submit(worker, 11, identity, source)
            _wait_for_completed(worker, 1)
            result, diagnostics = worker.latest(
                identity, maximum_source_age_s=0.50
            )
            self.assertIsNotNone(result)
            self.assertTrue(diagnostics["usable"])
            self.assertEqual(diagnostics["source_capture_sequence"], 11)
            self.assertGreater(result.age_s, 0.0)
            self.assertLess(result.confidence, 0.9)
            self.assertEqual(fake.reset_count, 1)

            rejected, mismatch = worker.latest(
                (3, 8, True, 2), maximum_source_age_s=0.50
            )
            self.assertIsNone(rejected)
            self.assertEqual(mismatch["reason"], "forearm_identity_mismatch")
        finally:
            worker.close()

    def test_stale_result_is_never_fused(self):
        fake = _FakeEstimator()
        worker = LatestOnlyForearmEstimator(estimator=fake)
        try:
            identity = (1, 1, False, 0)
            source = time.monotonic() - 0.30
            _submit(worker, 4, identity, source)
            _wait_for_completed(worker, 1)
            result, diagnostics = worker.latest(
                identity, maximum_source_age_s=0.20
            )
            self.assertIsNone(result)
            self.assertFalse(diagnostics["usable"])
            self.assertEqual(diagnostics["reason"], "forearm_source_stale")
        finally:
            worker.close()

    def test_capacity_one_overwrites_pending_job_and_resets_on_context_change(self):
        fake = _FakeEstimator(block_first=True)
        worker = LatestOnlyForearmEstimator(estimator=fake)
        try:
            now = time.monotonic()
            first_identity = (1, 2, True, 0)
            next_identity = (1, 2, True, 1)
            _submit(worker, 1, first_identity, now)
            self.assertTrue(fake.first_started.wait(1.0))
            _submit(worker, 2, next_identity, time.monotonic())
            _submit(worker, 3, next_identity, time.monotonic())
            fake.release_first.set()
            _wait_for_completed(worker, 2)

            result, diagnostics = worker.latest(
                next_identity, maximum_source_age_s=0.50
            )
            self.assertIsNotNone(result)
            self.assertEqual(diagnostics["source_capture_sequence"], 3)
            self.assertGreaterEqual(
                worker.statistics["overwritten_before_estimation"], 1
            )
            self.assertEqual(fake.update_count, 2)
            self.assertEqual(fake.reset_count, 2)
        finally:
            fake.release_first.set()
            worker.close()

    def test_optional_rate_limit_uses_latest_job_without_blocking_submitter(self):
        fake = _FakeEstimator()
        worker = LatestOnlyForearmEstimator(
            estimator=fake, maximum_rate_hz=10.0
        )
        try:
            identity = (2, 3, True, 0)
            _submit(worker, 1, identity, time.monotonic())
            _wait_for_completed(worker, 1)
            started = time.monotonic()
            _submit(worker, 2, identity, time.monotonic())
            _submit(worker, 3, identity, time.monotonic())
            self.assertLess(time.monotonic() - started, 0.02)
            _wait_for_completed(worker, 2)
            self.assertGreaterEqual(
                fake.update_times[1] - fake.update_times[0], 0.08
            )
            result, diagnostics = worker.latest(
                identity, maximum_source_age_s=0.50
            )
            self.assertIsNotNone(result)
            self.assertEqual(diagnostics["source_capture_sequence"], 3)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
