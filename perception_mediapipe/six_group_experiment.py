"""Replay six raw-pose methods on existing exact-frame RGB-D recordings.

Run `mp` with mediapipe_env; `hamer` with hamer_rtx2060. No ROS is started.
The model inputs, outputs, status and elapsed times are saved for inspection.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception_mediapipe.six_group_geometry import (PALM, five_point_frame,
    region_definition, hamer_frames, mp_frames)


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


def source_rows(source, label):
    # Repeated inference on exactly one RGB capture must not count twice.
    rows = [json.loads(line) for line in open(source/label/'mediapipe.jsonl')]
    unique = {}; duplicates = 0
    for row in sorted(rows, key=lambda r: r['t']):
        key = (row['frame_number'], row['t'])
        if key in unique:
            duplicates += 1
        else:
            unique[key] = row
    return list(unique.values()), duplicates


def run_mp(args):
    from perception_mediapipe.pose import MediaPipePose
    from perception_hamer.src.hand_detection_gate import hand_landmark_visibility
    model = MediaPipePose()
    for label in args.sessions:
        rows, duplicates = source_rows(args.source, label)
        out = args.output/label; out.mkdir(parents=True, exist_ok=True)
        meta = json.load(open(args.source/label/'metadata.json'))
        meta.update(input_rows=len(rows)+duplicates, unique_frames=len(rows),
                    duplicate_rows_removed=duplicates, filter_enabled=False,
                    palm_auxiliary_enabled=False, temporal_hold=False,
                    handedness_policy='fixed_left', applied_is_right=False,
                    expected_physical_hand='left', sampling='all available cached unique captures in annotated intervals')
        for f in ['annotations.json', 'time_alignment.json', 'imu_signals.npz']:
            (out/f).write_bytes((args.source/label/f).read_bytes())
        json.dump(meta, open(out/'metadata.json', 'w'), indent=2)
        with open(out/'mp.jsonl', 'w') as stream:
            for i, row in enumerate(rows):
                if args.limit and i >= args.limit: break
                result = {k: row.get(k) for k in ['t', 'seq', 'frame_number', 'file', 'source_found', 'rgb_depth_skew_ms', 'betas']}
                result['source_row'] = i
                result['detection'] = None
                result['groups'] = {g:dict(valid=False, position_valid=False, reason='SOURCE_FRAME_MISSING') for g in 'EF'}
                if row.get('source_found') and (args.source/label/'frames'/row['file']).exists():
                    data = np.load(args.source/label/'frames'/row['file'])
                    rgb = data['rgb']; depth = data['depth']
                    begin = time.perf_counter()
                    detected = model.hands.process(np.ascontiguousarray(rgb))
                    result['detector_ms'] = 1000*(time.perf_counter()-begin)
                    candidates = []
                    for n, (image, handed) in enumerate(zip(detected.multi_hand_landmarks or [], detected.multi_handedness or [])):
                        xy = np.array([[p.x*rgb.shape[1], p.y*rgb.shape[0]] for p in image.landmark])
                        if not np.all(np.isfinite(xy)): continue
                        score = handed.classification[0]
                        low, high = xy.min(0), xy.max(0); margin = (high-low)*.18
                        bbox = np.r_[low-margin, high+margin]
                        if np.prod(bbox[2:]-bbox[:2])/(rgb.shape[0]*rgb.shape[1]) < .006: continue
                        world = (np.array([[p.x,p.y,p.z] for p in detected.multi_hand_world_landmarks[n].landmark])
                                 if detected.multi_hand_world_landmarks and n < len(detected.multi_hand_world_landmarks) else np.full((21,3), np.nan))
                        candidates.append(dict(xy=xy, world=world, bbox=bbox,
                            is_right=False, detector_reported_is_right=score.label.lower()=='left',
                            handedness_policy='fixed_left', confidence=float(score.score),
                            visibility=hand_landmark_visibility(xy,rgb.shape[1],rgb.shape[0],palm_margin_px=8)))
                    d = max(candidates,key=lambda x:x['confidence']) if candidates else None
                    result['detection'] = d
                    result['groups'] = mp_frames(d, depth, meta['depth_scale'], meta['intrinsics'])
                    result['handedness_mismatch'] = bool(d is not None and d['detector_reported_is_right'])
                    result['applied_hand_is_right'] = None if d is None else d['is_right']
                    for g in 'EF':
                        result['groups'][g]['processing_ms'] = result['detector_ms']+result['groups'][g]['geometry_ms']
                stream.write(json.dumps(result,default=plain)+'\n')
                if i % 200 == 0:
                    stream.flush(); print('MP',label,i,'/',len(rows),flush=True)
    model.close()


def run_hamer(args):
    from perception_hamer.src.hamer_crop_inference import HamerCropInference
    from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition
    from perception_hamer.src.teleop_pose_packet import metric_wrist_ring_from_arrays
    runner = HamerCropInference(str(args.assets/'hamer_ckpts/checkpoints/hamer.ckpt'),
        data_root=str(args.assets),device='cuda:0',precision='fp16',freeze_betas=True)
    runner.load(); runner.warmup()
    definitions = {}; regions = {}; template = {}
    weights = runner._model.mano.lbs_weights.detach().cpu().numpy()[:,0]
    for side in [False,True]:
        v,j = runner.neutral_mano_geometry(side)
        definitions[side] = build_mano_wrist_definition(v,j,runner.mano_faces(),side)
        regions[side] = region_definition(v,j,weights)
        template[str(side)] = dict(regions=regions[side], wrist_ring=definitions[side].wrist_loop,
            neutral_common_frame=five_point_frame(j[PALM]), neutral_vertices=v, neutral_joints=j,
            faces=runner.mano_faces(),
            region_rule='12 disjoint neutral-template vertices each, root skin weight > .45; both skin surfaces; no outcome tuning')
    json.dump(template,open(args.output/'geometry_definition.json','w'),default=plain,indent=2)
    for label in args.sessions:
        out = args.output/label; meta = json.load(open(out/'metadata.json'))
        rows = [json.loads(line) for line in open(out/'mp.jsonl')]
        first_betas = next(np.asarray(r['betas']) for r in rows if r.get('betas') is not None)
        for side in [False,True]:runner.set_frozen_betas(side,first_betas)
        arrays = {k:[] for k in ['vertices','joints','global_orient','hand_pose','betas','frame_index']}
        with open(out/'hamer.jsonl','w') as stream:
            for i,row in enumerate(rows):
                if args.limit and i >= args.limit: break
                d = row['detection']
                reason = 'SOURCE_FRAME_MISSING' if not row.get('source_found') else 'NO_HAND_DETECTION'
                result = dict(t=row['t'],frame_number=row['frame_number'],groups={g:dict(valid=False,position_valid=False,reason=reason) for g in 'ABCD'})
                if d is not None:
                    if not d['visibility']['palm_inside_image']:
                        for g in 'ABCD':result['groups'][g]['reason']='PALM_OUTSIDE_IMAGE'
                    else:
                        data = np.load(args.source/label/'frames'/row['file']);side=d['is_right']
                        try:
                            begin=time.perf_counter()
                            pred=runner.infer(data['rgb'],d['bbox'],side,row['t'])
                            inference_ms=1000*(time.perf_counter()-begin)
                            result['groups']=hamer_frames(pred,definitions[side],regions[side])
                            begin=time.perf_counter();pos_valid=False;pos_reason='';diag={}
                            try:
                                pos,conf,diag,_=metric_wrist_ring_from_arrays(
                                    pred.pred_vertices_mano_right_canonical,definitions[side].wrist_loop,
                                    pred.hamer_crop_projection_translation,pred.hamer_nominal_crop_focal_length,
                                    pred.quality['affine_original_to_crop'],data['depth'],meta['depth_scale'],meta['intrinsics'],
                                    reference_depth_m=None,palm_anchor_pixels=np.asarray(d['xy'])[PALM])
                                pos_valid=bool(conf>0)
                                if not pos_valid:pos_reason='METRIC_DEPTH_INVALID'
                            except ValueError as exc:pos_reason=str(exc)
                            position_ms=1000*(time.perf_counter()-begin)
                            ring=pred.pred_vertices_source_camera_axes[definitions[side].wrist_loop].mean(0)
                            for g in 'ABCD':
                                r=result['groups'][g];r.update(position_valid=pos_valid,position_reason=pos_reason,
                                    inference_ms=inference_ms,core_ms=pred.inference_time_s*1000,position_ms=position_ms,
                                    processing_ms=row['detector_ms']+inference_ms+r['geometry_ms']+position_ms)
                                if pos_valid:r['position']=(pos if g=='B' else pos+pred.pred_keypoints_3d_source_camera_axes[0]-ring).tolist()
                            result['position_diagnostics']=diag
                            for key,val in [('vertices',pred.pred_vertices_source_camera_axes),('joints',pred.pred_keypoints_3d_source_camera_axes),
                                ('global_orient',pred.global_orient),('hand_pose',pred.hand_pose),('betas',pred.betas),('frame_index',i)]:arrays[key].append(val)
                        except (ValueError,RuntimeError) as exc:
                            for g in 'ABCD':result['groups'][g]=dict(valid=False,position_valid=False,reason='HAMER_INFERENCE_FAILED:'+str(exc))
                stream.write(json.dumps(result,default=plain)+'\n')
                if i % 200 == 0:
                    stream.flush();print('HaMeR ABCD',label,i,'/',len(rows),flush=True)
        np.savez_compressed(out/'hamer_raw_geometry.npz',**{k:np.asarray(v)for k,v in arrays.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['mp','hamer'])
    p.add_argument('--source',type=Path,default=ROOT/'.runtime/comparison')
    p.add_argument('--output',type=Path,default=ROOT/'.runtime/six_group_fixed_left_20260917')
    p.add_argument('--assets',type=Path,default=Path('/home/dongtian/myhandarmtest1-v1.0.1-ubuntu2004/perception_hamer/_DATA'))
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--sessions',nargs='+',default=['earlier','later'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{8,9,10,11})
    (run_mp if args.stage=='mp' else run_hamer)(args)

if __name__=='__main__':main()
