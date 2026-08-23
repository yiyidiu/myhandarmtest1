#!/usr/bin/env python3

import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from perception_hamer.scripts.evaluate_rgbd_relative_orientation import (
    evaluate_root, geodesic_deg, so3_log_degrees,
)
from perception_hamer.scripts.replay_rgbd_relative_tracker import replay_session
from perception_hamer.src.rgbd_rigid_tracker import RGBDRigidTrackerConfig


def rz(degrees):
    value = math.radians(degrees)
    return np.asarray([[math.cos(value),-math.sin(value),0],
                       [math.sin(value), math.cos(value),0], [0,0,1]], float)


class P5ScriptTest(unittest.TestCase):
    def test_so3_metric_and_log(self):
        rotation = rz(12.0)
        self.assertAlmostEqual(geodesic_deg(np.eye(3), rotation), 12.0)
        np.testing.assert_allclose(so3_log_degrees(rotation), [0,0,12], atol=1e-9)

    def test_offline_replay_uses_only_rgbd_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"; session.mkdir()
            (session/"rgb").mkdir(); (session/"aligned_depth").mkdir()
            intrinsics = {"width":160,"height":120,"fx":120.0,"fy":120.0,
                "ppx":80.0,"ppy":60.0,"distortion_model":"distortion.none",
                "coeffs":[0.0]*5}
            summary = {"scenario":"P5_STATIC","device":{"color_intrinsics":intrinsics,
                "depth_scale_m_per_unit":.001},"config":RGBDRigidTrackerConfig().__dict__}
            (session/"summary.json").write_text(json.dumps(summary))
            records=[]
            for index in range(4):
                rgb=np.zeros((120,160,3),np.uint8)
                for y in range(25,100,14):
                    for x in range(25,140,14): cv2.circle(rgb,(x+index,y),2,(80+x%150,120,200),-1)
                cv2.imwrite(str(session/f"rgb/{index:06d}.png"),rgb)
                cv2.imwrite(str(session/f"aligned_depth/{index:06d}.png"),np.full((120,160),800,np.uint16))
                records.append({"index":index,"rgb_path":f"rgb/{index:06d}.png",
                    "aligned_depth_path":f"aligned_depth/{index:06d}.png",
                    "timestamp_s":1+index/30,"timestamp_domain":"global_time",
                    "color_frame_number":10+index,"palm_roi":{"bbox":[10,10,150,110]},
                    "hamer_context":{"global_orient":[999],"gesture_changing":False}})
            (session/"frames.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
            result=replay_session(session,session/"replay.jsonl")
            self.assertFalse(result["hamer_loaded"])
            self.assertFalse(result["hamer_orientation_used"])
            self.assertEqual(result["frames"],4)

    def test_evaluator_minimum_criteria(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for scenario in ("P5_STATIC","P5_TRANSLATION","P5_ROTATION","P5_GESTURE"):
                session=root/(scenario+"_test"); session.mkdir()
                summary={"scenario":scenario,"kabsch_processing_ms":{"p95":2.0},
                         "hamer_orientation_used":False}
                (session/"summary.json").write_text(json.dumps(summary))
                records=[]
                for index in range(40):
                    angle = index*0.05 if scenario != "P5_ROTATION" else index*0.5
                    pair={"valid":index>0,"rotation_increment_deg":0.05,
                          "inlier_ratio":.95,"kabsch_rms_m":.0005,"spatial_span_m":.08}
                    result={"state":"TRACKING" if index else "INITIALIZING",
                            "pairwise":pair,"accumulated_rotation":rz(angle).reshape(-1).tolist(),
                            "reinitialization_count":1}
                    records.append({"index":index,"timestamp_s":index/30,
                        "palm_roi":{"bbox":[20+index*.01,20,120+index*.01,100]},"result":result})
                (session/"frames.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
            report=evaluate_root(root)
            self.assertTrue(report["minimum_gazebo_criteria_pass"])
            self.assertFalse(report["hamer_orientation_used"])
            self.assertTrue((root/"p5_rgbd_relative_orientation_metrics.csv").is_file())

    def test_missing_rotation_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            # No usable sessions: evaluator still produces an explicit NOT RUN
            # decision instead of crashing or treating absence as pass.
            report=evaluate_root(root)
            self.assertFalse(report["minimum_gazebo_criteria_pass"])
            self.assertIn("P5_ROTATION", report["not_run_scenarios"])


if __name__ == "__main__": unittest.main()
