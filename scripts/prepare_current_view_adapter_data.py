#!/usr/bin/env python3
"""Prepare one newly recorded session with fixed chronological partitions.

No motion annotations are fabricated. IMU is a training/evaluation reference,
not an inference input. One source calibration frame is shared as the zero;
future validation/test frames never enter the training set.
"""
import argparse, csv, json, sys, subprocess
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame
from perception_mediapipe.recorded_rgbd import RecordedRGBD
from scripts.replay_hamer_fixed_crop import source_rgb


def run(args):
    args.out.mkdir(parents=True,exist_ok=True);analysis=args.out/'source_analysis';analysis.mkdir(exist_ok=True)
    with (args.source/'screen_frame_timeline.csv').open()as f:clocks=list(csv.DictReader(f))
    origin=float(np.median([int(x['estimated_frame_wall_ns'])/1e9-float(x['video_time_s'])for x in clocks]))
    (analysis/'time_origin.json').write_text(json.dumps(dict(video_wall_origin=origin)))
    rows=[];ct=None;failures=0
    for path in sorted(args.source.glob('perception/*/frames.jsonl*')):
        process=subprocess.Popen(['zstd','-q','-dc',str(path)],stdout=subprocess.PIPE,text=True)if path.suffix=='.zst'else None
        with (process.stdout if process else path.open())as f:
            for line in f:
                d=json.loads(line)
                if d.get('mano_joints') is None or d.get('betas_user') is None or d.get('bbox') is None:
                    failures+=1;continue
                palm=build_hamer_five_point_palm_frame(d['mano_joints'],False)
                if not palm.valid:failures+=1;continue
                h=d.get('hand_pose_6d')or{}
                r=dict(t=d['timestamp']-origin,stamp=d['timestamp'],seq=d['capture_sequence'],frame=d['frame_number'],
                    bbox=d['bbox'],betas=d['betas_user'],raw_palm=palm.rotation.tolist(),
                    source=args.source.name,analysis=str(analysis.resolve()))
                fingers=((d.get('published_pose')or{}).get('finger_observation')or{}).get('flexion')
                r['recorded_finger_flexion']=float(np.mean(fingers[1:]))if fingers and len(fingers)==5 else None
                rows.append(r)
                if ct is None and h.get('calibrated') and (d.get('teleop_udp')or{}).get('control_enabled'):
                    ct=r['t']
        if process and process.wait()!=0:raise RuntimeError('Could not read source pose archive')
    if ct is None:raise RuntimeError('No source-frame calibration event found')
    if args.task_only:
        # Locate a closing-hand passage from archived model measurements,
        # without claiming that this proves successful physical grasping.
        times=np.array([x['t']for x in rows]);flex=np.array([x['recorded_finger_flexion']or 0 for x in rows])
        previous=np.searchsorted(times,times-1).clip(0,len(times)-1)
        delta=flex-flex[previous];delta[times<ct+30]=-np.inf
        center=float(times[int(np.argmax(delta))]);dense=[center-15,center+15]
        selected=[];last=-1e9
        for r in rows:
            if r['t']<ct:continue
            if dense[0]<=r['t']<=dense[1] or r['t']-last>=.5:
                last=r['t'];selected.append(dict(r,split='task',case='grasp_recording',role='measurement'))
        folder=args.out/'source_png';folder.mkdir(exist_ok=True);temp=None;reader=None
        try:
            database=args.source/'rgbd.db3'
            if not database.exists():
                temp=args.out/'temporary_source_rgbd.db3'
                with temp.open('xb')as f:subprocess.run(['zstd','-q','-dc',str(args.source/'rgbd.db3.zst')],stdout=f,check=True)
                database=temp
            reader=RecordedRGBD(database)
            for i,r in enumerate(selected):
                f=folder/f'{r["seq"]}.png';r['file']=str(f.resolve())
                if not f.exists():
                    image=source_rgb(reader,r)
                    if not cv2.imwrite(str(f),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):raise RuntimeError('RGB write failed')
                if i%100==0:print('Grasp source RGB',i,'/',len(selected),flush=True)
            intr=reader.intr
        finally:
            if reader is not None:reader.close()
            if temp is not None:temp.unlink(missing_ok=True)
        protocol=dict(source=str(args.source.resolve()),imu_available=False,
            source_calibration_time_s=ct,n=len(selected),dense_window_s=dense,intrinsics=intr,
            selection='Full recording sampled about 2 Hz plus native-cadence 30s window around greatest recorded flexion increase',
            scope='Independent grasp-scene stress replay without orientation truth; not an angle-accuracy or grasp-success benchmark')
        (args.out/'source_manifest.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
        (args.out/'task_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
        print(json.dumps(protocol,ensure_ascii=False,indent=2),flush=True);return
    with (args.source/'imu_time_estimated.csv').open()as f:imurows=list(csv.DictReader(f))
    it=np.array([int(x['estimated_sample_wall_time_ns'])/1e9-origin for x in imurows])
    iq=R.from_euler('xyz',[[float(x[k])for k in ('roll_deg','pitch_deg','yaw_deg')]for x in imurows],degrees=True)
    gyro=np.deg2rad([[float(x[k])for k in ('gyro_x_dps','gyro_y_dps','gyro_z_dps')]for x in imurows])
    iv=np.array([x['time_valid']=='1'for x in imurows])
    np.savez_compressed(analysis/'imu_signals.npz',t=it,q=iq.as_quat(),gyro_rad_s=gyro,time_valid=iv)
    ts=np.array([x['t']for x in rows]);rr=R.from_matrix([x['raw_palm']for x in rows])
    dt=np.diff(ts);good=(dt>0)&(dt<.2);vt=(ts[:-1]+ts[1:])/2
    speed=np.rad2deg((rr[:-1].inv()*rr[1:]).magnitude())/np.maximum(dt,1e-6)
    def fit(lo,hi):
        grid=np.arange(lo,hi,.02)
        vis=gaussian_filter1d(median_filter(np.interp(grid,vt[good],speed[good]),size=3),2)
        vals=[]
        for shift in np.arange(-.5,.5001,.005):
            ref=gaussian_filter1d(np.interp(grid+shift,it,np.rad2deg(np.linalg.norm(gyro,axis=1))),2)
            vals.append([float(shift),float(np.corrcoef(vis,ref)[0,1])])
        best=max(vals,key=lambda x:x[1]);return dict(range_s=[lo,hi],offset_s=best[0],correlation=best[1],curve=vals)
    calibration=fit(ts[0]+1,ct-1)
    end=float(min(ts[-1],it[-1]-.5));duration=end-ct
    train_end=ct+.60*duration;val_end=ct+.80*duration
    if calibration['correlation']<.6:
        calibration=fit(ct+1,train_end-3)
        calibration['reason']='Pre-C matching insufficient; use the training portion only, including stronger rotations'
    (analysis/'alignment_diagnostic.json').write_text(json.dumps(dict(source_start=ts[0],source_C=ct,
        proposed_partition_train_end=train_end,calibration=calibration),indent=2))
    print('Alignment diagnostic',ct,calibration['range_s'],calibration['offset_s'],calibration['correlation'],flush=True)
    if calibration['correlation']<.6 or abs(calibration['offset_s'])>=.49:
        raise RuntimeError('No sufficiently identifiable time alignment; inspect calibration motion before fitting')
    offset=calibration['offset_s']
    (analysis/'time_alignment.json').write_text(json.dumps(dict(used_offset_s=offset,calibration=calibration,
        sign='query IMU at camera source relative time plus offset',scope='One motion-derived offset, not hardware synchronization'),indent=2))
    anchor=min(rows,key=lambda x:abs(x['t']-ct));selected=[]
    bounds={'train':(ct,train_end-3),'validation':(train_end+3,val_end-3),'test':(val_end+3,end)}
    for split,(lo,hi) in bounds.items():
        selected.append(dict(anchor,split=split,case='calibration',role='calibration'))
        last=lo
        for r in rows:
            if r['t']<lo or r['t']>hi or r['t']-last<.5 or r['seq']==anchor['seq']:continue
            last=r['t'];selected.append(dict(r,split=split,case=f'window_{int((r["t"]-lo)//15):02d}',role='measurement'))
    # This is source-frame extraction only. Depth remains preserved in the
    # original file; no synthetic or screen-video image replaces source RGB.
    folder=args.out/'source_png';folder.mkdir(exist_ok=True)
    reader=RecordedRGBD(args.source/'rgbd.db3')
    try:
        for i,row in enumerate(selected):
            p=folder/f'{row["seq"]}.png';row['file']=str(p.resolve())
            if not p.exists():
                rgb=source_rgb(reader,row)
                if not cv2.imwrite(str(p),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise RuntimeError('RGB write failed')
            if i%100==0:print('Exact RGB',i,'/',len(selected),flush=True)
        intrinsics=reader.intr
    finally:reader.close()
    summary=dict(source=str(args.source.resolve()),source_calibration_time_s=ct,end_time_s=end,
        usable_pose_rows=len(rows),unusable_source_rows=failures,time_offset_s=offset,
        alignment_correlation=calibration['correlation'],partition_bounds_s=bounds,
        gap_each_side_s=3,shared_calibration_frame=anchor['seq'],intrinsics=intrinsics,
        counts={s:sum(x['split']==s for x in selected)for s in bounds},
        scope='Temporal holdout inside one recording; not an independent recording or new operator',
        source_annotations_used=False)
    (args.out/'source_manifest.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    (args.out/'data_protocol.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--task-only',action='store_true')
    run(p.parse_args())
