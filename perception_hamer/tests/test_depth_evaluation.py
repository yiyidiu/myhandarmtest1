#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.depth_evaluation import (  # noqa: E402
    DepthEvaluationError,
    depth_bias_metrics,
    rgb_depth_edge_alignment_metrics,
)


class DepthEvaluationTest(unittest.TestCase):
    def test_known_bias_and_outlier_metrics(self):
        depth = np.full((10, 10), 1010, dtype=np.uint16)
        depth[0, 0] = 0
        result = depth_bias_metrics(depth, 0.001, 1.0, [0, 0, 10, 10])
        self.assertAlmostEqual(result["bias_median_mm"], 10.0)
        self.assertAlmostEqual(result["mad_about_median_mm"], 0.0)
        self.assertEqual(result["valid_samples"], 99)

    def test_invalid_roi_or_depth_rejected(self):
        with self.assertRaises(DepthEvaluationError):
            depth_bias_metrics(np.zeros((5, 5), np.uint16), 0.001, 1.0, [0, 0, 5, 5])
        with self.assertRaises(DepthEvaluationError):
            depth_bias_metrics(np.ones((5, 5), np.uint16), 0.001, 1.0, [0, 0, 6, 5])

    def test_aligned_step_edges(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, 50:] = 255
        depth = np.full((100, 100), 1000, dtype=np.uint16)
        depth[:, 50:] = 1200
        result = rgb_depth_edge_alignment_metrics(rgb, depth, [5, 5, 95, 95])
        self.assertLess(result["depth_to_nearest_rgb_edge_p50_px"], 2.0)


if __name__ == "__main__":
    unittest.main()
