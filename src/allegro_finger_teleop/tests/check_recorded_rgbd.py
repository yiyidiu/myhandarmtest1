"""Re-run the live RGB-D packet path on twelve actual archived D455 left-hand frames."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from perception_hamer.src.hamer_crop_inference import HamerCropInference
from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition, estimate_hamer_root_wrist_frame
from perception_hamer.src.teleop_pose_packet import build_live_teleop_packet


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("frameset", type=Path)
    parser.add_argument("assets", type=Path)
    args = parser.parse_args()
    info = json.loads((args.frameset/"manifest.json").read_text())
    calibration = info["calibration"]
    intrinsics = calibration["aligned_color_intrinsics"]
    runner = HamerCropInference(str(args.assets/"hamer_ckpts/checkpoints/hamer.ckpt"),
                                data_root=str(args.assets), perspective_correction=True, freeze_betas=False)
    runner.load()
    faces = runner.mano_faces()
    neutral = build_mano_wrist_definition(*runner.neutral_mano_geometry(False), faces, False)
    accepted, rejected = [], {}
    for frm in info["frames"]:
        directory = args.frameset/frm["folder"]
        record = json.loads((directory/"recorded_frame.json").read_text())
        rgb = cv2.cvtColor(cv2.imread(str(directory/"rgb.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(directory/"depth_aligned_u16.png"), cv2.IMREAD_UNCHANGED)
        frame = SimpleNamespace(aligned_depth_raw=depth, color_intrinsics=intrinsics,
                                depth_scale_m_per_unit=calibration["depth_scale_m_per_unit"])
        result = runner.infer(rgb, frm["bbox_xyxy"], frm["is_right"], record["timestamp"],
                              camera_intrinsics=intrinsics)
        raw = estimate_hamer_root_wrist_frame(result, neutral).as_dict()
        try:
            packet = build_live_teleop_packet(result, dict(control_wrist_frame=raw, control_wrist_frame_raw=raw),
                    frame, SimpleNamespace(confidence=record["roi"]["confidence"]), "archive",
                    record["index"], 1, 1, required_reference_kind="MANO_WRIST_RING_16",
                    wrist_depth_mode="wrist-center-offset", mano_faces=faces, include_hand_landmarks=True)
        except ValueError as exc:
            rejected[str(frm["capture_sequence"])] = str(exc)
            continue
        xyz = np.asarray(packet["hand_landmarks"]["points"])
        restored = xyz @ np.asarray(raw["rotation"]).T + raw["origin"]
        np.testing.assert_allclose(restored, result.pred_keypoints_3d_source_camera_axes, atol=1e-7)
        assert packet["stamp"] == record["timestamp"]
        assert packet["position_source"] == "D455_WRIST_RING_CENTER_FROM_SAME_PIXEL_SURFACE_OFFSET"
        accepted.append(frm["capture_sequence"])
    print(json.dumps(dict(total=len(info["frames"]), accepted=accepted, rejected=rejected), indent=2))
    if len(accepted) < 8:
        raise AssertionError("too few usable archived frames")


if __name__ == "__main__":
    main()
