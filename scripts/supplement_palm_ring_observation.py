#!/usr/bin/env python3
"""Additional views of the same comparison: history, timing, origin and control.
No new model inference, physical ground truth, or simulated robot execution.
"""
import argparse,gzip,json,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R,Slerp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src/handarm_moveit_demo/src')]
from evaluate_palm_ring_observation import stats,deg,writecsv,jsonout
from handarm_moveit_demo.shared_teleop_core import WholeWristPoseMapper
import yaml


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    rows=json.load(gzip.open(out/'paired_frames.json.gz','rt'));meta=json.load(open(out/'replay_metadata.json'));metrics=json.load(open(out/'metrics.json'))
    ts=np.array([x['t'] for x in rows]);p=R.from_quat([x['palm_q'] for x in rows]);b=R.from_quat([x['ring_q'] for x in rows]);ct=metrics['c_video_s'];ci=np.argmin(abs(ts-ct));pc=p[ci];bc=b[ci]
    iz=np.load(args.source/'analysis_intent/imu_signals.npz');it=iz['t'];iq=R.from_quat(iz['q']);si=Slerp(it,iq);offset=metrics['imu_offset_s'];ic=si(ct+offset)
    idx=np.clip(np.searchsorted(it,ts+offset),1,len(it)-1);valid=iz['time_valid'][idx]&iz['time_valid'][idx-1]
    global_q=[]
    for line in next(args.source.glob('perception/*/frames.jsonl')).open():
        d=json.loads(line)
        if d['timestamp']-meta['video_wall_origin']>=350:break
        flip=np.diag([-1.,1.,1.]);global_q.append(R.from_matrix(flip@np.array(d['global_orient']).reshape(3,3)@flip).as_quat())
    root=R.from_quat(global_q)
    root_delta=root*root[ci].inv();summary={}
    summary['model_internal_change_from_C']={n:stats(deg(root_delta.inv()*(v*v[ci].inv()))) for n,v in [('palm',p),('ring',b)]}
    # Estimate ONE shared camera-to-IMU-C basis on the already declared two
    # calibration motions using the model root, never fit either tested frame.
    # This is only an exploratory calibration sensitivity analysis.
    cal_ranges=[(217,229),(265,273)];source_vectors=[];reference_vectors=[]
    for lo,hi in cal_ranges:
        grid=np.arange(lo,hi,.05);rr=Slerp(ts,root)(grid);ii=si(grid+offset)
        wv=(rr[1:]*rr[:-1].inv()).as_rotvec()/.05
        wi=((ic.inv()*ii[1:])*(ic.inv()*ii[:-1]).inv()).as_rotvec()/.05
        wv=gaussian_filter1d(wv,2,axis=0);wi=gaussian_filter1d(wi,2,axis=0)
        good=(np.linalg.norm(wi,axis=1)>.15)
        source_vectors.extend(wv[good]);reference_vectors.extend(wi[good])
    fit,rssd=R.align_vectors(np.asarray(reference_vectors),np.asarray(source_vectors))
    nominal=R.from_matrix([[0,0,1],[0,1,0],[-1,0,0]])*pc.inv()
    summary['shared_calibration']=dict(ranges=cal_ranges,rotation=fit.as_matrix().tolist(),
        difference_from_nominal_deg=float(deg(fit*nominal.inv())),fit_vector_samples=len(source_vectors),
        interpretation='Shared root-motion-derived camera/IMU basis; descriptive calibration sensitivity only, not independent accuracy')
    anns=json.load(open(out/'annotations_confirmed.json'));cal_results=[]
    for a in anns:
        inds=np.flatnonzero((ts>=a['start']+.2)&(ts<=a['end']-.2)&valid)
        if len(inds)<3:continue
        iscal=any(a['start']<hi and a['end']>lo for lo,hi in cal_ranges)
        i=ic.inv()*si(ts[inds]+offset);ir=i*i[0].inv()
        result=dict(segment=a['id'],kind=a['kind'],calibration_motion=iscal)
        for n,v in [('palm',p),('ring',b)]:
            vr=fit*(v[inds]*v[inds[0]].inv())*fit.inv()
            result[n]=stats(deg(ir.inv()*vr))
        cal_results.append(result)
    summary['shared_calibration']['segments']=cal_results
    for n in ['palm','ring']:
        vals=[r[n] for r in cal_results if not r['calibration_motion']]
        summary['shared_calibration'][n+'_heldout_frame_weighted_mean_deg']=sum(x['mean']*x['n'] for x in vals)/sum(x['n'] for x in vals)
    # Native origin effect and task mapper counterfactual on the same source.
    cfg=yaml.safe_load((ROOT/'src/handarm_moveit_demo/config/shared_teleop.yaml').read_text())
    prof=cfg['mapping_profiles']['whole_wrist_pose'];K=np.array(prof['wrist_ring_to_palm_basis']);maps=[]
    for n in ['palm','ring']:
        mapper=WholeWristPoseMapper(prof['translation_matrix'],prof['translation_gain'],[2,2,2],np.pi,np.deg2rad(88),orientation_gain=prof['orientation_gain'])
        if n=='ring':mapper.rotation_matrix=mapper.rotation_matrix@K.T
        v=p if n=='palm' else b;zero=pc if n=='palm' else bc
        maps.append(R.from_matrix([mapper._mapped_rotation((zero.inv()*x).as_matrix()) for x in v]))
    summary['same_controller_counterfactual']=dict(rotation_difference_deg=stats(deg(maps[0].inv()*maps[1])),orientation_gain=prof['orientation_gain'],
         scope='raw estimates through same mapper; no live filtering, no QP, no new simulated or real arm trial')
    depthgood=[x for x in rows if x['skin_valid']];dt=np.diff([x['t'] for x in depthgood]);z=np.array([x['palm_position'][2] for x in depthgood]);dz=np.diff(z)
    summary['surface_depth_steps_mm']=stats(abs(dz[dt<.2])*1000)
    summary['skin_patches']=dict(sample_count=stats([x['skin_diagnostics']['sample_count'] for x in depthgood]),
         depth_mad_mm=stats([x['skin_diagnostics']['depth_mad_m']*1000 for x in depthgood]))
    summary['source_interval_ms']=stats(np.diff(ts)*1000)
    summary['source_over_100ms_gaps']=int(sum(np.diff(ts)>.1))
    summary['source_over_200ms_gaps']=int(sum(np.diff(ts)>.2))
    summary['source_effective_hz']=(len(ts)-1)/(ts[-1]-ts[0])
    summary['original_capture_drops']=sum(x['timing'].get('dropped_capture_frames',0) for x in rows)
    # Same fixed zero for the starts/ends of annotated return motions.
    returns=[]
    for a in anns:
        if not a['return_phase']:continue
        j=np.flatnonzero((ts>=a['end']-.6)&(ts<=a['end']-.2)&valid)
        if not len(j):continue
        returns.append(dict(segment=a['id'],time=a['time_label'],
            palm_return_from_C_deg=float(np.median(deg(pc.inv()*p[j]))),
            ring_return_from_C_deg=float(np.median(deg(bc.inv()*b[j]))),
            imu_return_from_C_deg=float(np.median(deg(ic.inv()*si(ts[j]+offset))))))
    writecsv(out/'回程与初始零位.csv',returns);summary['returns']=returns
    # Earlier fixed-left, unfiltered B/C inference caches. Their saved depth
    # estimator differs: use ONLY orientation, explicitly not new skin depth.
    history=[]
    for label in ['s1','s2']:
        directory=ROOT/'.runtime/gesture_pose_20260917/results'/label
        source=[json.loads(x) for x in (directory/'hamer.jsonl').open()]
        good=[x for x in source if x['groups']['B']['valid'] and x['groups']['C']['valid']]
        t=np.array([x['t'] for x in good]);br=R.from_matrix([np.array(x['groups']['B']['rotation']).reshape(3,3) for x in good]);pr=R.from_matrix([np.array(x['groups']['C']['rotation']).reshape(3,3) for x in good])
        h=np.load(directory/'imu_signals.npz');hs=Slerp(h['t'],R.from_quat(h['q']));ofs=json.load(open(directory/'time_alignment.json'))['used_offset_s']
        an=json.load(open(directory/'annotations.json'));segments=[]
        for a in an:
            j=np.flatnonzero((t>=a['start']+.2)&(t<=a['end']-.2));ix=np.clip(np.searchsorted(h['t'],t[j]+ofs),1,len(h['t'])-1)
            j=j[h['time_valid'][ix]&h['time_valid'][ix-1]]
            if len(j)<3:continue
            pi=pr[j]*pr[j[0]].inv();bi=br[j]*br[j[0]].inv();ii=hs(t[j]+ofs);im=deg(ii[0].inv()*ii)
            segments.append(dict(id=a['id'],kind=a['kind'],gesture=a.get('gesture'),start=a['start'],end=a['end'],n=len(j),
                between_methods=stats(deg(pi.inv()*bi)),palm_magnitude_error=stats(abs(deg(pi)-im)),ring_magnitude_error=stats(abs(deg(bi)-im))))
        history.append(dict(label=label,source=json.load(open(directory/'metadata.json'))['source'],processed=len(source),paired_pose=len(good),segments=segments,
            scope='fixed-left raw orientation only; older depth not pooled with current skin experiment'))
    summary['history']=history
    jsonout(out/'supplement.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['history','returns','shared_calibration']},indent=2))
    print('calibration shift',summary['shared_calibration']['difference_from_nominal_deg'],'heldout',[summary['shared_calibration'][n+'_heldout_frame_weighted_mean_deg'] for n in ['palm','ring']])
    for h in history:
        print(h['label'],'paired',h['paired_pose'],'segments',len(h['segments']))
        for name in ['between_methods','palm_magnitude_error','ring_magnitude_error']:
            print(name,sum(x[name]['mean']*x['n'] for x in h['segments'])/sum(x['n'] for x in h['segments']),max(x[name]['max'] for x in h['segments']))


if __name__=='__main__':main()
