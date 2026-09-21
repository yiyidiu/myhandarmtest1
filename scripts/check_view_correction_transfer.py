#!/usr/bin/env python3
"""Apply the already fitted horizontal gain to two older recorded sessions.
No refitting, no offset optimization, no new model inference or robot output.
Uses recorded raw wrist-frame rotations before filtering/fusion. Metric is
relative-angle magnitude error; it is not full directional orientation error.
"""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    gain=json.load(open(ROOT/'docs/imu_view_correction_20260920/correction_metrics.json'))['calibration']['gain']
    base=Path('/home/dongtian/myhandarmtest1-teleop-repair/.runtime')
    sessions=[('20260911_153019','hand_motion_observation_20260911T153019_6Wdm','analysis_motion_intent'),('20260911_202058','hand_motion_observation_20260911T202058_Ime9','analysis_feedback')]
    summary=dict(gain=gain,gain_refit=False,reference='IMU as provisional ground truth per user instruction',
        intrinsics=dict(fx=606.5111083984375,fy=606.2513427734375,cx=319.0163879394531,cy=248.72071838378906),
        metric='mean absolute difference in relative rotation magnitude, same segment start',
        scope='Older recorded raw wrist-frame outputs; no filters/fusion; same camera profile; no new inference of the latest frontend, new person or camera test.',sessions=[])
    allsegments=[];allsamples=[]
    for label,name,sub in sessions:
        p=base/name/sub;allrows=json.load(gzip.open(p/'perception_signals.json.gz','rt'));rows=[x for x in allrows if x.get('raw_q')and x.get('bbox')]
        t=np.array([x['t']for x in rows]);raw=R.from_quat([x['raw_q']for x in rows]);u=np.array([(x['bbox'][0]+x['bbox'][2])/2 for x in rows]);theta=np.arctan2(u-319.0163879394531,606.5111083984375)
        corrected=R.from_euler('y',gain*theta)*raw;imu=np.load(p/'imu_signals.npz');si=Slerp(imu['t'],R.from_quat(imu['q']));offset=json.load(open(p/'time_alignment.json'))['used_offset_s'];anns=json.load(open(p/'annotations.json'));segments=[]
        for ann in anns:
            kind=ann['kind']if sub=='analysis_motion_intent'else {2:'translation_ud',3:'translation_ud',4:'translation_lr',5:'translation_lr',6:'translation_lr',7:'translation_lr',8:'rotation_fingers_tilt',9:'rotation_inplane',10:'rotation_side_flip',11:'mixed'}.get(ann['id'],'other')
            idx=np.flatnonzero((t>=ann['start']+.2)&(t<=ann['end']-.2)&(t+offset>=imu['t'][0])&(t+offset<=imu['t'][-1]))
            ix=np.clip(np.searchsorted(imu['t'],t[idx]+offset),1,len(imu['t'])-1);idx=idx[imu['time_valid'][ix]&imu['time_valid'][ix-1]]
            if len(idx)<3:continue
            iq=si(t[idx]+offset);reference=np.rad2deg((iq[0].inv()*iq).magnitude())
            mag={n:np.rad2deg((q[idx]*q[idx[0]].inv()).magnitude())for n,q in [('raw',raw),('corrected',corrected)]}
            result=dict(session=label,id=ann['id'],kind=kind,intent=ann['intent'],start=ann['start'],end=ann['end'],n=len(idx),
                raw_magnitude_mae_deg=float(np.mean(abs(mag['raw']-reference))),corrected_magnitude_mae_deg=float(np.mean(abs(mag['corrected']-reference))),
                recorded_right_hand_frames=int(sum(bool((rows[j].get('roi')or{}).get('is_right'))for j in idx)))
            segments.append(result);allsegments.append(result)
            for k,j in enumerate(idx):
                allsamples.append(dict(session=label,segment=ann['id'],kind=kind,t=float(t[j]),seq=rows[j]['seq'],view_x_deg=float(np.rad2deg(theta[j])),
                    imu_rotation_deg=float(reference[k]),raw_rotation_deg=float(mag['raw'][k]),corrected_rotation_deg=float(mag['corrected'][k])))
        cat={}
        for kind in sorted({x['kind']for x in segments}):
            subset=[x for x in segments if x['kind']==kind];cat[kind]=dict(segments=len(subset),frames=sum(x['n']for x in subset),
                **{key:float(np.average([x[key]for x in subset],weights=[x['n']for x in subset]))for key in ['raw_magnitude_mae_deg','corrected_magnitude_mae_deg']})
        summary['sessions'].append(dict(label=label,source=str(p.parent),source_records=len(allrows),valid_raw_pose_and_box=len(rows),
            existing_time_offset_s=offset,frame_axis='recorded raw wrist frame in source camera coordinates',categories=cat))
    (a.out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(a.out/'segments.json').write_text(json.dumps(allsegments,ensure_ascii=False,indent=2))
    for filename,rows in [('segments.csv',allsegments),('frame_pairs.csv',allsamples)]:
        with (a.out/filename).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
