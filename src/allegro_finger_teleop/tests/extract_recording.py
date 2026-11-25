"""Recompute the 4290–4450 replay using original RGB, aligned depth and the live QB0 geometry."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from perception_hamer.src.causal_wrist_so3_filter import CausalWristSO3Filter, CausalWristSO3FilterConfig
from perception_hamer.src.hamer_crop_inference import HamerCropInference
from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition, estimate_hamer_root_wrist_frame
from perception_hamer.src.teleop_pose_packet import build_live_teleop_packet
from perception_hamer.scripts.run_d455_hamer_crop import MediaPipeDetectionSidecar
from perception_hamer.src.active_hand_selector import AutomaticActiveHandSelector
from perception_hamer.src.teleop_control_gate import TeleopControlGate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgbd", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.rgbd / "quickstart/tools"))
    from recorded_rgbd import RecordedRGBD
    runner = HamerCropInference(str(args.assets/"hamer_ckpts/checkpoints/hamer.ckpt"), data_root=str(args.assets),
                                perspective_correction=True, freeze_betas=False)
    runner.load()
    faces = runner.mano_faces()
    neutral = build_mano_wrist_definition(*runner.neutral_mano_geometry(False), faces, False)
    filt = CausalWristSO3Filter(CausalWristSO3FilterConfig(time_constant_s=.08, minimum_gain=.08, maximum_gain=1.,
                                                           motion_gain_start_deg=1.5, motion_gain_full_deg=8.))
    reader = RecordedRGBD(args.rgbd/"rgbd_20260920.db3")
    detector = MediaPipeDetectionSidecar(reader.intr["width"], reader.intr["height"], .50,
                                         fixed_left_hand=True, finger_shapes=True)
    selector = AutomaticActiveHandSelector(fixed_is_right=False)
    gate = TeleopControlGate(require_hand_landmarks=True)
    calibration_index = None
    data = {k: [] for k in ("points", "stamp", "sequence", "record_index", "confidence",
                            "wrist_position_m", "palm_rotation_row_major", "pose_confidence",
                            "measured_wrist_rotation_row_major", "image_landmarks_px", "surface_depths_m", "shape_signatures", "wrist_motion_support")}
    bad = {}
    try:
        with (args.rgbd/"recorded_hamer_mano_full.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                index = row["index"]
                if index < 4290:
                    continue
                if index > 4450:
                    break
                if not row["valid"] or row["roi"]["is_right"]:
                    bad[index] = "invalid source identity"
                    continue
                frame_number, stamp = row["frame_number"], row["timestamp"]
                try:
                    rgb, depth, _ = reader.read(frame_number, stamp)
                except RuntimeError:
                    rgb, depth, _ = reader.read(frame_number, stamp)
                rgbd = SimpleNamespace(aligned_depth_raw=depth, color_intrinsics=reader.intr,
                                       depth_scale_m_per_unit=reader.scale)
                pred = runner.infer(rgb, row["bbox"], False, stamp, camera_intrinsics=reader.intr)
                witness = selector.select(detector.detect(rgb), stamp)
                xy = (witness["hand_landmarks_pixels"] if witness.get("valid") and witness.get("fingers_inside_image")
                      else np.full((21, 2), np.nan))
                raw = estimate_hamer_root_wrist_frame(pred, neutral).as_dict()
                confidence = float(row["roi"]["confidence"])
                if not raw["valid"]:
                    bad[index] = "invalid MANO wrist"
                    continue
                observed = filt.update(stamp, raw["rotation"], confidence*pred.quality["bbox_visible_fraction"])
                if not observed.valid:
                    bad[index] = "invalid wrist filter"
                    continue
                control = dict(raw, rotation=observed.rotation.tolist(), filter_confidence=observed.confidence)
                try:
                    packet = build_live_teleop_packet(pred, dict(control_wrist_frame_raw=raw, control_wrist_frame=control,
                            finger_image_landmarks_px=np.asarray(xy).tolist(),
                            finger_shape_signature=witness.get("finger_shape_signature") if np.isfinite(xy).all() else None),
                            rgbd, SimpleNamespace(confidence=confidence), "offline", index, 1, 1,
                            required_reference_kind="MANO_WRIST_RING_16", wrist_depth_mode="wrist-center-offset",
                            mano_faces=faces, include_hand_landmarks=True)
                    hand = packet["hand_landmarks"]
                except (KeyError, ValueError) as exc:
                    bad[index] = str(exc)
                    continue
                values = (hand["points"], stamp, row["capture_sequence"], index, hand["confidence"],
                          packet["wrist_position_m"], packet["palm_rotation_row_major"], packet["confidence"],
                          packet["measured_wrist_rotation_row_major"], xy,
                          np.asarray(hand["surface_depths_m"],dtype=float) if hand["surface_depths_m"] is not None else np.full(21,np.nan),
                          hand.get("shape_signatures", np.full((2,5,4), np.nan)),
                          np.asarray(packet["wrist_motion_support"], dtype=float) if packet["wrist_motion_support"] is not None
                          else np.full((9,3), np.nan))
                if calibration_index is None:
                    if not gate.confirmation_pending:
                        gate.request_confirmation(index-1, now=stamp)
                    hand["image_landmarks_px"] = np.asarray(xy).tolist()
                    if gate.confirm_packet(packet, lambda: True, now=stamp) is not None:
                        calibration_index = len(data["stamp"])
                for target, value in zip(data.values(), values):
                    target.append(value)
    finally:
        detector.close()
        reader.close()
    if calibration_index is None:
        raise RuntimeError("no visible stable open-hand reference in the RGB-D segment")
    np.savez_compressed(args.output, calibration_index=calibration_index, gesture_protocol="open_hand_wrist_sweep", **data)
    print(json.dumps({"accepted":len(data["stamp"]), "rejected":bad, "moved_m":np.ptp(data["wrist_position_m"], axis=0).tolist()}))


if __name__ == "__main__":
    main()
