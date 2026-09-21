#!/usr/bin/env python3
"""Offline viewpoint correction with IMU as user-selected provisional truth.

First test the geometric unit correction. A single horizontal gain is then fit
on slow translation 54-83s only. Subsequent translation and all active rotation
are held out. Corrected predictions never read the current frame's IMU value.
"""
import argparse, csv, gzip, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as R, Slerp

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent'
    data=json.load(gzip.open(ROOT/'.runtime/palm_ring_surface_20260920/paired_frames.json.gz','rt'))
    imu=np.load(source/'imu_signals.npz');off=json.load(open(source/'time_alignment.json'))['used_offset_s'];ct=42.78118896484375
    rows=[x for x in data if x.get('palm_valid') and x['t']<350];ts=np.array([x['t']for x in rows]);ix=np.clip(np.searchsorted(imu['t'],ts+off),1,len(imu['t'])-1)
    ok=(ts+off>=imu['t'][0])&(ts+off<=imu['t'][-1])&imu['time_valid'][ix]&imu['time_valid'][ix-1]
    rows=[x for x,keep in zip(rows,ok)if keep];ts=ts[ok];ci=int(np.argmin(abs(ts-ct)))
    raw=R.from_quat([x['palm_q']for x in rows]);u=np.array([(x['bbox'][0]+x['bbox'][2])/2 for x in rows]);v=np.array([(x['bbox'][1]+x['bbox'][3])/2 for x in rows])
    rayx=np.arctan2(u-319.0163879394531,606.5111083984375);rayy=np.arctan2(v-248.72071838378906,606.2513427734375)
    xy=np.c_[(u-319.0163879394531)/606.5111083984375,(v-248.72071838378906)/606.2513427734375];length=np.linalg.norm(xy,axis=1)
    viewvec=np.c_[-xy[:,1],xy[:,0],np.zeros(len(rows))]*(np.arctan(length)/np.maximum(length,1e-12))[:,None]
    mount=R.from_matrix([[0,0,1],[0,1,0],[-1,0,0]])
    def relative(q):return mount*q[ci].inv()*q*mount.inv()
    def horizontal(gain):return R.from_euler('y',gain*rayx)*raw
    si=Slerp(imu['t'],R.from_quat(imu['q']));reference=si(ct+off).inv()*si(ts+off)
    train=(ts>=54)&(ts<83)
    def loss(gain):
        e=(reference[train].inv()*relative(horizontal(gain))[train]).magnitude()
        return float(np.mean(e**2))
    fit=minimize_scalar(loss,bracket=(0.,1.),method='brent')
    if not fit.success:raise RuntimeError('Scalar calibration did not converge')
    gain=float(fit.x)
    native={'raw':raw,'full_geometric':R.from_rotvec(viewvec)*raw,'horizontal_unit':horizontal(1.),'horizontal_calibrated':horizontal(gain)}
    relative_q={name:relative(q)for name,q in native.items()}
    refangle=np.rad2deg(reference.magnitude())
    def mean(values):return float(np.mean(values))
    def comparison(mask):
        result={'n':int(mask.sum()),'imu_peak_deg':float(refangle[mask].max()),'methods':{}}
        for name,q in relative_q.items():
            mag=np.rad2deg(q[mask].magnitude());err=np.rad2deg((reference[mask].inv()*q[mask]).magnitude())
            result['methods'][name]=dict(magnitude_mae_deg=mean(abs(mag-refangle[mask])),full_rotation_mae_deg=mean(err),full_rotation_p95_deg=float(np.percentile(err,95)),peak_deg=float(mag.max()))
        return result
    summary=dict(reference='Synchronized fused IMU treated as provisional ground truth per user instruction',time_offset_s=off,C_time_s=ct,
        axis_alignment='Existing fixed nominal hand-to-IMU mounting; not refit',
        calibration=dict(parameter='horizontal view angle gain',gain=gain,train_range_s=[54,83],train_frames=int(train.sum()),objective='mean squared full relative SO(3) error',uses_live_imu_in_corrected_prediction=False),
        ranges={},segment_categories={},examples=[])
    for name,lo,hi in [('calibration_slow_lr',54,83),('heldout_lr',83,110),('heldout_depth',113,167),('heldout_vertical',176,209),('heldout_pitch',217,260),('heldout_yaw',265,349)]:
        summary['ranges'][name]=comparison((ts>=lo)&(ts<hi))
    annotations=json.load(open(source/'annotations.json'));segments=[];samples=[]
    for ann in annotations:
        idx=np.flatnonzero((ts>=ann['start']+.2)&(ts<=ann['end']-.2));ri=reference[idx]*reference[idx[0]].inv();im=np.rad2deg(ri.magnitude())
        row=dict(id=ann['id'],kind=ann['kind'],time=ann['time_label'],n=len(idx),imu_peak_deg=float(im.max()),methods={})
        for name,q in relative_q.items():
            rel=q[idx]*q[idx[0]].inv();mag=np.rad2deg(rel.magnitude());err=np.rad2deg((ri.inv()*rel).magnitude())
            row['methods'][name]=dict(magnitude_mae_deg=mean(abs(mag-im)),full_rotation_mae_deg=mean(err),peak_deg=float(mag.max()))
        segments.append(row)
    for kind in sorted({x['kind']for x in segments}):
        subset=[x for x in segments if x['kind']==kind]
        summary['segment_categories'][kind]={name:{key:float(np.average([x['methods'][name][key]for x in subset],weights=[x['n']for x in subset]))for key in ['magnitude_mae_deg','full_rotation_mae_deg']}for name in native}
    for t in [62,72,180,187,220,268,295,308,322,332,345]:
        j=int(np.argmin(abs(ts-t)));summary['examples'].append(dict(t=float(ts[j]),imu_deg=float(refangle[j]),**{n:float(np.rad2deg(q[j].magnitude()))for n,q in relative_q.items()}))
    for j,row in enumerate(rows):
        item=dict(t=float(ts[j]),seq=row['seq'],ray_x_deg=float(np.rad2deg(rayx[j])),ray_y_deg=float(np.rad2deg(rayy[j])),imu_angle_deg=float(refangle[j]),calibration_frame=bool(train[j]))
        for name,q in relative_q.items():
            item[name+'_angle_deg']=float(np.rad2deg(q[j].magnitude()))
            item[name+'_full_error_deg']=float(np.rad2deg((reference[j].inv()*q[j]).magnitude()))
        samples.append(item)
    with (args.out/'frame_comparison.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
    (args.out/'metrics.json').write_text(json.dumps(summary,indent=2));(args.out/'segment_metrics.json').write_text(json.dumps(segments,indent=2))
    np.savez_compressed(args.out/'corrected_native_poses.npz',t=ts,seq=[x['seq']for x in rows],**{name:q.as_quat()for name,q in native.items()})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
