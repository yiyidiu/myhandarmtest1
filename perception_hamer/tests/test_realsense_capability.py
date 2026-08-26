#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.realsense_capability import (  # noqa: E402
    match_supported_device_model,
    normalize_device_models,
    rank_rgbd_candidates,
)


class CapabilityRankingTest(unittest.TestCase):
    def test_supported_models_are_explicit(self):
        self.assertEqual(
            match_supported_device_model("Intel RealSense D435I"), "D435I"
        )
        self.assertEqual(
            match_supported_device_model("Intel RealSense D455", ("D455",)),
            "D455",
        )
        self.assertIsNone(match_supported_device_model("Intel RealSense D415"))
        self.assertIsNone(
            match_supported_device_model("Intel RealSense D435I", ("D455",))
        )

    def test_model_policy_rejects_unreviewed_devices(self):
        with self.assertRaises(ValueError):
            normalize_device_models(("D415",))
        with self.assertRaises(ValueError):
            normalize_device_models(())

    def test_rank_uses_only_enumerated_matched_profiles(self):
        capability = {
            "video_profiles": [
                {
                    "sensor": "RGB",
                    "stream": "stream.color",
                    "format": "format.rgb8",
                    "width": 640,
                    "height": 480,
                    "fps": fps,
                    "stream_index": 0,
                }
                for fps in (30, 15)
            ]
            + [
                {
                    "sensor": "Depth",
                    "stream": "stream.depth",
                    "format": "format.z16",
                    "width": 640,
                    "height": 480,
                    "fps": fps,
                    "stream_index": 0,
                }
                for fps in (30, 15)
            ]
        }
        ranked = rank_rgbd_candidates(capability, "live_algorithm")
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["color"]["fps"], 30)
        self.assertEqual(ranked[0]["depth"]["fps"], 30)

    def test_invalid_purpose_rejected(self):
        with self.assertRaises(ValueError):
            rank_rgbd_candidates({"video_profiles": []}, "unknown")


if __name__ == "__main__":
    unittest.main()
