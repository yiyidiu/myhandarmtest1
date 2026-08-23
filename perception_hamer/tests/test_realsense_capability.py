#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.realsense_capability import rank_rgbd_candidates  # noqa: E402


class CapabilityRankingTest(unittest.TestCase):
    def test_rank_uses_only_enumerated_matched_profiles(self):
        capability = {
            "video_profiles": [
                {
                    "sensor": "RGB",
                    "stream": "stream.color",
                    "format": "format.rgb8",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "stream_index": 0,
                },
                {
                    "sensor": "RGB",
                    "stream": "stream.color",
                    "format": "format.rgb8",
                    "width": 640,
                    "height": 480,
                    "fps": 15,
                    "stream_index": 0,
                },
                {
                    "sensor": "Depth",
                    "stream": "stream.depth",
                    "format": "format.z16",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "stream_index": 0,
                },
                {
                    "sensor": "Depth",
                    "stream": "stream.depth",
                    "format": "format.z16",
                    "width": 640,
                    "height": 480,
                    "fps": 15,
                    "stream_index": 0,
                },
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
