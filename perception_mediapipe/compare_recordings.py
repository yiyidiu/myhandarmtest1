"""Read-only exact-frame comparison. Run mp first, then hamer, then summarize.

Independent of ROS: no robot commands, no IMU fitting, no auxiliary correction.
"""
import argparse, datetime, gzip, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = [
    ('earlier', 'hand_motion_observation_20260911T153019_6Wdm', 'analysis_motion_intent'),
    ('later', 'hand_motion_observation_20260911T202058_Ime9', 'analysis_feedback'),
]


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def run_mp(args):
    from perception_mediapipe.pose import MediaPipePose
    from perception_mediapipe.recorded_rgbd import RecordedRGBD
    model=MediaPipePose()
    for label,name,folder in SOURCES:
        source=args.source/name;analysis=source/folder
        ann=json.load(open(analysis/'annotations.json'))
        signals=json.load(gzip.open(analysis/'perception_signals.json.gz','rt'))
        rows=[];last=-1e9
        for r in signals:
            if any(a['start']<=r['t']<=a['end']for a in ann) and r['t']-last>=args.period:
                rows.append(r);last=r['t']
        out=args.output/label;out.mkdir(parents=True,exist_ok=True)
        cache=out/'frames';cache.mkdir(exist_ok=True)
        for f in ['annotations.json','time_alignment.json','imu_signals.npz']:
            (out/f).write_bytes((analysis/f).read_bytes())
        origin=json.load(open(analysis/'time_origin.json'))['video_wall_origin']
        reader=RecordedRGBD(source/'rgbd.db3')
        json.dump(dict(source=str(source),source_commit='a0cae6c',intrinsics=reader.intr,
            depth_scale=reader.scale,video_origin=origin,requested_frames=len(rows),sampling_period_s=args.period,
            reader='indexed CDR plus original librealsense rs.align'),open(out/'metadata.json','w'),indent=2)
        try:
            with (out/'mediapipe.jsonl').open('w') as stream:
                for i,row in enumerate(rows):
                    entry=dict(t=row['t'],seq=row['seq'],frame_number=row['frame_number'],
                        recorded_raw_q=row['raw_q'],recorded_filter_q=row['filter_q'],
                        recorded_position=row['position'],recorded_valid=row['published_valid'],
                        recorded_timing=row.get('timing'),recorded_roi=row.get('roi'),betas=row.get('betas'),source_found=False)
                    try:
                        try:rgb,depth,diag=reader.read(row['frame_number'],origin+row['t'])
                        except RuntimeError:
                            if i!=0:raise
                            rgb,depth,diag=reader.read(row['frame_number'],origin+row['t'])
                        d,seconds=model.detect(rgb)
                        entry.update(diag,source_found=True,mp_ms=seconds*1000,
                            poses=model.poses(d,depth,reader.scale,reader.intr),detection=d)
                        entry['file']=f'f{row["frame_number"]:07d}.npz'
                        np.savez_compressed(cache/entry['file'],rgb=rgb,depth=depth)
                    except ValueError as exc:entry['missing_reason']=str(exc)
                    stream.write(json.dumps(entry,default=plain)+'\n')
                    if i%200==0:stream.flush();print(label,i,'/',len(rows),'video',round(row['t'],2),flush=True)
        finally:reader.close()
    model.close()


def run_hamer(args):
    from perception_hamer.src.hamer_crop_inference import HamerCropInference
    from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition, estimate_mano_wrist_frame
    from perception_hamer.src.teleop_pose_packet import metric_wrist_ring_from_arrays
    asset = args.assets
    runner = HamerCropInference(str(asset/'hamer_ckpts/checkpoints/hamer.ckpt'),
        data_root=str(asset), device='cuda:0', precision='fp16', freeze_betas=True,
        source_frame='camera_color_optical_frame', timestamp_clock_domain='recorded')
    runner.load(); runner.warmup(); definitions = {}
    for side in (True,False):
        v,j = runner.neutral_mano_geometry(side)
        definitions[side] = build_mano_wrist_definition(v,j,runner.mano_faces(),side)
    for label,_,_ in SOURCES:
        out=args.output/label; meta=json.load(open(out/'metadata.json'))
        with (out/'hamer.jsonl').open('w') as stream:
            for index,line in enumerate(open(out/'mediapipe.jsonl')):
                if args.limit and index >= args.limit: break
                row=json.loads(line); d=row.get('detection'); result=dict(t=row['t'],valid=False,reason='NO_HAND')
                if d is not None and d['visibility']['palm_inside_image']:
                    data=np.load(out/'frames'/row['file']);side=d['is_right'];definition=definitions[side]
                    # Frozen shape from this same recording, never fitted to IMU.
                    if row.get('betas') is not None: runner.set_frozen_betas(side,np.array(row['betas']))
                    try:
                        began=time.perf_counter()
                        pred=runner.infer(data['rgb'],d['bbox'],side,row['t'])
                        inference_ms=(time.perf_counter()-began)*1000
                        began=time.perf_counter()
                        frame=estimate_mano_wrist_frame(pred.pred_vertices_source_camera_axes,definition)
                        wrist, confidence, diag, _=metric_wrist_ring_from_arrays(
                            pred.pred_vertices_mano_right_canonical,definition.wrist_loop,
                            pred.hamer_crop_projection_translation,pred.hamer_nominal_crop_focal_length,
                            pred.quality['affine_original_to_crop'],data['depth'],meta['depth_scale'],meta['intrinsics'],
                            palm_anchor_pixels=[d['xy'][i] for i in [0,5,9,13,17]])
                        result=dict(t=row['t'],valid=bool(frame.valid and confidence>0),reason=frame.failure_reason,
                                    rotation=frame.rotation.reshape(-1).tolist(),position=wrist.tolist(),
                                    confidence=confidence, inference_ms=inference_ms,
                                    core_ms=1000*pred.inference_time_s,geometry_ms=(time.perf_counter()-began)*1000)
                    except (ValueError,RuntimeError) as exc: result['reason']=str(exc)
                stream.write(json.dumps(result,default=plain)+'\n')
                if index%200==0:stream.flush();print(label,index,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['mp','hamer'])
    p.add_argument('--source',type=Path,default=Path('/home/dongtian/myhandarmtest1-teleop-repair/.runtime'))
    p.add_argument('--output',type=Path,default=ROOT/'.runtime/comparison')
    p.add_argument('--assets',type=Path,default=Path('/home/dongtian/myhandarmtest1-v1.0.1-ubuntu2004/perception_hamer/_DATA'))
    p.add_argument('--limit',type=int,default=0,help='small inference smoke check only')
    p.add_argument('--period',type=float,default=.25,help='minimum source-time spacing for accuracy evaluation')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    (run_mp if a.stage=='mp' else run_hamer)(a)

if __name__=='__main__':main()
