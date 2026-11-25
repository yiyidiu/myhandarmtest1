"""Recreate a short camera packet replay from the September RGB-D recording; no physical camera."""
import argparse
import gzip
import subprocess
import types
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from perception_hamer.scripts.run_d455_hamer_crop import MediaPipeDetectionSidecar
from perception_hamer.src.active_hand_selector import AutomaticActiveHandSelector
from perception_hamer.src.causal_wrist_so3_filter import CausalWristSO3Filter, CausalWristSO3FilterConfig
from perception_hamer.src.hamer_crop_inference import HamerCropInference
from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition, estimate_hamer_root_wrist_frame, estimate_mano_wrist_frame
from perception_hamer.src.teleop_pose_packet import build_live_teleop_packet, build_invalid_teleop_packet
from perception_hamer.src.teleop_control_gate import TeleopControlGate

def upstream_module(name):
    path = "perception_hamer/src/" + name + ".py"
    source = subprocess.check_output(["git","show","77d6acc:"+path],cwd=Path(__file__).resolve().parents[3])
    name = "perception_hamer.src.upstream_0921_" + name
    module = types.ModuleType(name)
    module.__package__ = "perception_hamer.src"
    sys.modules[name] = module
    exec(compile(source,path,"exec"),module.__dict__)
    return module

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path);parser.add_argument("assets",type=Path)
    parser.add_argument("output",type=Path);parser.add_argument("--qb0",action="store_true")
    parser.add_argument("--compare-0921",type=Path)
    a=parser.parse_args()
    sys.path.insert(0,str(a.data/"quickstart/tools"))
    from recorded_rgbd import RecordedRGBD
    runner=HamerCropInference(str(a.assets/"hamer_ckpts/checkpoints/hamer.ckpt"),data_root=str(a.assets),
                              perspective_correction=a.qb0, freeze_betas=False)
    runner.load()
    faces=runner.mano_faces()
    definition=build_mano_wrist_definition(*runner.neutral_mano_geometry(False),faces,False)
    reader=RecordedRGBD(a.data/"rgbd_20260920.db3")
    sidecar=MediaPipeDetectionSidecar(reader.intr["width"],reader.intr["height"],.50,fixed_left_hand=True,finger_shapes=True)
    selector=AutomaticActiveHandSelector(fixed_is_right=False)
    gate=TeleopControlGate(retain_reference_until_operator_action=True, confirmation_delay_s=2.)
    filt=CausalWristSO3Filter(CausalWristSO3FilterConfig(time_constant_s=.08,minimum_gain=.08,maximum_gain=1.,
                              large_angle_mode="reject",motion_gain_start_deg=1.5,motion_gain_full_deg=8.))
    session="archive-"+("qb0" if a.qb0 else "0921")
    if a.compare_0921:
        if a.qb0: raise ValueError("The archive comparison uses the 9.21 observation.")
        upstream_wrist = upstream_module("mano_wrist_reference")
        upstream_filter = upstream_module("causal_wrist_so3_filter")
        upstream_pose = upstream_module("teleop_pose_packet")
        upstream_gate = upstream_module("teleop_control_gate").TeleopControlGate(
            retain_reference_until_operator_action=True, confirmation_delay_s=2.)
        upstream_def = upstream_wrist.build_mano_wrist_definition(*runner.neutral_mano_geometry(False),faces,False)
        upstream_filt = upstream_filter.CausalWristSO3Filter(filt.config)
    paired = []
    count=enabled=0
    try:
        with (a.data/"recorded_hamer_mano_full.jsonl").open() as source, gzip.open(a.output,"wt",encoding="utf-8") as output:
            for line in source:
                row=json.loads(line); i=row["index"]
                if i<4290: continue
                if i>4450: break
                if not row["valid"] or row["roi"]["is_right"]: continue
                stamp=row["timestamp"]; seq=row["capture_sequence"]
                try:
                    rgb,depth,_=reader.read(row["frame_number"],stamp)
                except RuntimeError:
                    rgb,depth,_=reader.read(row["frame_number"],stamp)
                frame=SimpleNamespace(aligned_depth_raw=depth,color_intrinsics=reader.intr,depth_scale_m_per_unit=reader.scale)
                start=time.monotonic();pred=runner.infer(rgb,row["bbox"],False,stamp,**({"camera_intrinsics":reader.intr} if a.qb0 else {}))
                inference=time.monotonic()-start
                witness=selector.select(sidecar.detect(rgb),stamp)
                root=estimate_hamer_root_wrist_frame(pred,definition).as_dict()
                raw=root if a.qb0 else estimate_mano_wrist_frame(pred.pred_vertices_source_camera_axes,definition).as_dict()
                confidence=float(row["roi"]["confidence"])
                try:
                    observed=filt.update(stamp,raw["rotation"],confidence*pred.quality["bbox_visible_fraction"])
                    if not observed.valid: raise ValueError("ORIENTATION_REJECTED")
                    orientation_method = ("HAMER_GLOBAL_ORIENT_TIMES_NEUTRAL_WRIST_RING_16" if a.qb0
                                          else "HAMER_MANO_WRIST_RING_16_IRLS_KABSCH") + "_QUALITY_ADAPTIVE_CAUSAL_SO3"
                    estimates=dict(control_wrist_frame_raw=raw,control_wrist_frame=dict(raw,rotation=observed.rotation.tolist(),filter_confidence=observed.confidence, orientation_source=orientation_method),
                                   finger_wrist_frame_raw=root)
                    if witness.get("valid") and witness.get("fingers_inside_image"):
                        estimates.update(finger_image_landmarks_px=witness["hand_landmarks_pixels"],finger_shape_signature=witness.get("finger_shape_signature"))
                    else:
                        estimates["teleop_fingers_visible"]=False
                    packet=build_live_teleop_packet(pred,estimates,frame,SimpleNamespace(confidence=confidence),session,i,1,1,
                        required_reference_kind="MANO_WRIST_RING_16",wrist_depth_mode="wrist-center-offset" if a.qb0 else "palm-skin",
                        mano_faces=faces,include_hand_landmarks=True)
                except (ValueError,KeyError) as exc:
                    packet=build_invalid_teleop_packet(session_id=session,sequence=i,stamp=stamp,invalid_reason=str(exc),
                                                      hand_is_right=False,presence_generation=1,active_hand_generation=1)
                elapsed=time.monotonic()-start
                if a.compare_0921:
                    original_raw = upstream_wrist.estimate_mano_wrist_frame(pred.pred_vertices_source_camera_axes,upstream_def).as_dict()
                    original_filtered = upstream_filt.update(stamp,original_raw["rotation"],confidence*pred.quality["bbox_visible_fraction"])
                    try:
                        if not original_filtered.valid: raise ValueError("ORIENTATION_REJECTED")
                        oc=dict(original_raw,rotation=original_filtered.rotation.tolist(),filter_confidence=original_filtered.confidence,orientation_source=orientation_method)
                        op=upstream_pose.build_live_teleop_packet(pred,dict(control_wrist_frame_raw=original_raw,control_wrist_frame=oc),
                            frame,SimpleNamespace(confidence=confidence),session,i,1,1,required_reference_kind="MANO_WRIST_RING_16",
                            wrist_depth_mode="palm-skin",mano_faces=faces)
                    except (ValueError, KeyError) as exc:
                        op=upstream_pose.build_invalid_teleop_packet(session,i,stamp,str(exc),1,1,False)
                    assert packet["valid"] == op["valid"], i
                    if op["valid"]:
                        paired.append([float(np.max(np.abs(np.asarray(packet[k])-np.asarray(op[k])))) for k in
                                      ("wrist_position_m","palm_rotation_row_major","confidence")])
                    if count>=8 and not upstream_gate.enabled:
                        if not upstream_gate.confirmation_pending: upstream_gate.request_confirmation(i-1,now=stamp)
                        upstream_gate.confirm_packet(op,lambda:True,now=stamp,source_monotonic_s=stamp)
                    orig=upstream_gate.decorate(op)
                packet["capture_monotonic_s"]=stamp
                packet["timing"]=dict(contract_version=1,capture_sequence=seq,dropped_capture_frames=0,inference_executed=True,
                                      inference_call_s=inference,model_inference_s=min(inference,pred.inference_time_s),
                                      postprocess_s=elapsed-inference,capture_to_publish_s=elapsed)
                if count>=8 and not gate.enabled:
                    if not gate.confirmation_pending: gate.request_confirmation(i-1,now=stamp)
                    gate.confirm_packet(packet,lambda:True,now=stamp,source_monotonic_s=stamp)
                decorated=gate.decorate(packet)
                if a.compare_0921:
                    for k in ("control_enabled","control_reference_epoch","control_reference_token","valid"):
                        assert decorated[k] == orig[k], (i,k)
                output.write(json.dumps(decorated,separators=(",",":"),allow_nan=False)+"\n")
                count+=1
                enabled+=int(decorated["valid"])
    finally:
        sidecar.close();reader.close()
    if a.compare_0921:
        maxima=np.max(paired,axis=0).tolist()
        assert max(maxima)<1e-10
        a.compare_0921.write_text(json.dumps(dict(source="77d6acc",frames=count,compared=len(paired),enabled=enabled,
            max_position_difference_m=maxima[0],max_rotation_matrix_entry_difference=maxima[1],max_confidence_difference=maxima[2],
            calibration_tokens_and_validity_identical=True),indent=2))
    print(session,count,enabled,flush=True)
    if enabled<60: raise RuntimeError("insufficient enabled frames")

if __name__=="__main__":
    main()
