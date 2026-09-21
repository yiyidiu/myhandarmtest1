#!/usr/bin/env python3
"""Multimetric descriptive analysis of paired observation replay.

IMU angles are a comparison reference, not independent ground truth. Segment
resets and one-C-only results are both retained. No p values from video frames.
"""
import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def deg(x): return np.rad2deg(x.magnitude())
def stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x): return dict(n=0)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                rmse=float(np.sqrt(np.mean(x*x))),p95=float(np.percentile(x,95)),max=float(x.max()))
def writecsv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
def jsonout(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    origin=args.source/'analysis_intent';out=args.out
    rows=json.load(gzip.open(out/'paired_frames.json.gz','rt'))
    annotations=json.loads((origin/'annotations.json').read_text())
    annotations[39].update(start=255,end=257,time_label='4:15–4:17',correction='User explicitly confirmed 4:15–4:17 on 2026-09-20')
    jsonout(out/'annotations_confirmed.json',annotations)
    iz=np.load(origin/'imu_signals.npz');it=iz['t'];valid=iz['time_valid'];si=Slerp(it,R.from_quat(iz['q']));sg=Slerp(it,R.from_quat(iz['gyro_q']))
    offset=json.loads((origin/'time_alignment.json').read_text())['used_offset_s']
    robot=json.load(gzip.open(origin/'robot_signals.json.gz','rt'));ct=robot['reference_hand']['source_t']
    good=[x for x in rows if x['palm_valid'] and x['ring_valid']]
    ts=np.array([x['t'] for x in good]);idx=np.searchsorted(it,ts+offset);idx=np.clip(idx,1,len(it)-1)
    keep=valid[idx]&valid[idx-1]&(ts+offset>=it[0])&(ts+offset<=it[-1])
    good=[x for x,k in zip(good,keep) if k];ts=ts[keep]
    p=R.from_quat([x['palm_q'] for x in good]);b=R.from_quat([x['ring_q'] for x in good]);i=si(ts+offset);g=sg(ts+offset)
    ci=int(np.argmin(abs(ts-ct)));pc=p[ci];bc=b[ci];ic=si(ct+offset)
    a=R.from_matrix([[0,0,1],[0,1,0],[-1,0,0]])
    # Both visual methods expressed in the same C palm basis. No independent
    # orientation fitting or per-method alignment to the IMU.
    m=a*pc.inv();vp=m*p*pc.inv()*m.inv();vb=m*b*bc.inv()*m.inv();vi=ic.inv()*i
    per_segment=[];samples=[];vectors={};allsteps=[]
    for ann in annotations:
        lo,hi=ann['start']+.2,ann['end']-.2
        inds=np.flatnonzero((ts>=lo)&(ts<=hi));raw=[x for x in rows if lo<=x['t']<=hi]
        ms=dict(ann,total_frames=len(raw),paired_imu_frames=len(inds),
                palm_pose_valid=sum(x['palm_valid'] for x in raw),ring_pose_valid=sum(x['ring_valid'] for x in raw),
                skin_valid=sum(x['skin_valid'] for x in raw),
                original_published_valid=sum(x['original_published_valid'] for x in raw))
        if len(inds)<3: per_segment.append(ms);continue
        t=ts[inds];rps=vp[inds]*vp[inds[0]].inv();rbs=vb[inds]*vb[inds[0]].inv();ris=vi[inds]*vi[inds[0]].inv()
        mags=deg(ris);p_mag=deg(rps);b_mag=deg(rbs)
        pv=np.rad2deg(rps.as_rotvec());bv=np.rad2deg(rbs.as_rotvec());iv=np.rad2deg(ris.as_rotvec())
        main_axis={'yaw':0,'pitch':1}.get(ann['kind']);other=[k for k in range(3) if k!=main_axis]
        ms['imu_peak_deg']=float(mags.max());ms['imu_axis_peak_deg']=np.max(abs(iv),axis=0).tolist()
        ms['imu_gyro_disagreement']=stats(deg((i[inds[0]].inv()*i[inds]).inv()*(g[inds[0]].inv()*g[inds])))
        ms['between_methods']=stats(deg(rps.inv()*rbs))
        for name,rs,mag,vec in [('palm',rps,p_mag,pv),('ring',rbs,b_mag,bv)]:
            ms[name]=dict(angle_magnitude_difference=stats(abs(mag-mags)),
                          full_orientation_nominal_imu_difference=stats(deg(ris.inv()*rs)),
                          peak_deg=float(mag.max()),axis_peak_deg=np.max(abs(vec),axis=0).tolist(),
                          axis_mean_absolute_difference_deg=np.mean(abs(vec-iv),axis=0).tolist(),
                          unintended_axis_peak_deg=float(np.linalg.norm(vec[:,other],axis=1).max()),
                          unintended_axis_imu_residual_peak_deg=float(np.linalg.norm((vec-iv)[:,other],axis=1).max()),
                          end_nominal_imu_difference_deg=float(deg(ris[-1].inv()*rs[-1])))
            if main_axis is not None:
                x=iv[:,main_axis];y=vec[:,main_axis]
                ms[name]['main_axis_gain_origin']=float(np.dot(x,y)/np.dot(x,x)) if np.dot(x,x)>1 else None
                ms[name]['main_axis_corr']=float(np.corrcoef(x,y)[0,1]) if np.std(x)>.1 and np.std(y)>.1 else None
                ms[name]['main_axis_peak_deg']=float(np.max(abs(y)))
        ms['timing_sensitivity_angle_mae']={}
        for shift in [offset-.04,offset,offset+.04]:
            ir=si(t+shift);mag=deg(ir[0].inv()*ir)
            ms['timing_sensitivity_angle_mae'][str(round(shift,3))]=[float(np.mean(abs(p_mag-mag))),float(np.mean(abs(b_mag-mag)))]
        for k,j in enumerate(inds):
            row=good[j];sample=dict(segment=ann['id'],kind=ann['kind'],speed=ann['speed'],return_phase=ann['return_phase'],
                t=ts[j],palm_mag=p_mag[k],ring_mag=b_mag[k],imu_mag=mags[k],
                palm_magnitude_error=abs(p_mag[k]-mags[k]),ring_magnitude_error=abs(b_mag[k]-mags[k]),
                palm_full_difference=deg(ris[k].inv()*rps[k]),ring_full_difference=deg(ris[k].inv()*rbs[k]),
                between_methods=deg(rps[k].inv()*rbs[k]),
                palm_width_px=row.get('palm_width_px'),detector_palm_width_px=row.get('detector_palm_width_px'),
                skin_valid=row['skin_valid'])
            for prefix,vec in [('palm',pv),('ring',bv),('imu',iv)]:
                for axis,col in [('yaw',0),('pitch',1),('roll',2)]:sample[prefix+'_'+axis]=vec[k,col]
            samples.append(sample)
        per_segment.append(ms)
    for k in range(1,len(ts)):
        dt=ts[k]-ts[k-1]
        if not .005<dt<.20:continue
        ps=vp[k]*vp[k-1].inv();bs=vb[k]*vb[k-1].inv();is_=vi[k]*vi[k-1].inv()
        allsteps.append(dict(t=ts[k],dt=dt,palm_step=deg(ps),ring_step=deg(bs),imu_step=deg(is_),
                             palm_step_residual=deg(is_.inv()*ps),ring_step_residual=deg(is_.inv()*bs),
                             imu_speed=deg(is_)/dt))
    summary=dict(rows=len(rows),paired_imu_frames=len(good),c_video_s=ct,c_source_nearest_s=float(ts[ci]),imu_offset_s=offset,
                 palm_valid=sum(x['palm_valid'] for x in rows),ring_valid=sum(x['ring_valid'] for x in rows),
                 skin_valid=sum(x['skin_valid'] for x in rows),
                 depth_failure_counts=dict(Counter(x.get('palm_depth_reason') or x.get('ring_depth_reason') or x.get('depth_reason') or x.get('geometry_reason') for x in rows if not x['skin_valid'])),
                 pose_failure_counts=dict(Counter(x.get('geometry_reason') or x.get('palm_reason') or x.get('ring_reason') for x in rows if not(x['palm_valid'] and x['ring_valid']))),
                 geometric_compute_ms={n:stats([x[n+'_compute_ms'] for x in rows if n+'_compute_ms'in x]) for n in ['palm','ring']},
                 capture_to_publish_ms=stats([x['timing']['capture_to_publish_s']*1000 for x in rows if x.get('timing')]),
                 projection_reconstruction_px=stats([x['projection_reconstruction_max_px'] for x in rows if 'projection_reconstruction_max_px'in x]),
                 skin_position_Z_pair_max_difference_m=max(abs(x['palm_position'][2]-x['ring_position'][2]) for x in rows if x['skin_valid']),
                 native_origins_xy_difference_mm=stats([1000*np.linalg.norm(np.array(x['palm_position'])[:2]-np.array(x['ring_position'])[:2]) for x in rows if x['skin_valid']]),
                 categories={},narrow_yaw={},steps={},one_C_only={})
    for key in ['all']+sorted(set(x['kind'] for x in samples)):
        ss=[x for x in samples if key=='all' or x['kind']==key]
        summary['categories'][key]={k:stats([x[k] for x in ss]) for k in ['palm_magnitude_error','ring_magnitude_error','palm_full_difference','ring_full_difference','between_methods']}
    yaw=[x for x in samples if x['kind']=='yaw']
    for name,lo,hi in [('under15px',0,15),('15to25px',15,25),('over25px',25,np.inf)]:
        ss=[x for x in yaw if x['palm_width_px'] is not None and lo<=x['palm_width_px']<hi]
        summary['narrow_yaw'][name]={k:stats([x[k] for x in ss]) for k in ['palm_magnitude_error','ring_magnitude_error','palm_full_difference','ring_full_difference','between_methods']}
        summary['narrow_yaw'][name]['skin_valid_fraction']=float(np.mean([x['skin_valid'] for x in ss])) if ss else None
    for name in ['palm','ring']:
        step=np.array([x[name+'_step'] for x in allsteps]);quiet=[x for x in allsteps if x['imu_speed']<5]
        summary['steps'][name]=dict(all_step=stats(step),over10deg=int(sum(step>10)),over20deg=int(sum(step>20)),
             quiet_imu_step=stats([x[name+'_step'] for x in quiet]),quiet_imu_step_residual=stats([x[name+'_step_residual'] for x in quiet]))
        v=vp if name=='palm' else vb
        after=ts>=54
        summary['one_C_only'][name]=dict(full_nominal_difference=stats(deg(vi[after].inv()*v[after])),
            angle_magnitude_difference=stats(abs(deg(vi[after])-deg(v[after]))))
    summary['one_C_only']['between_methods']=stats(deg(vp[ts>=54].inv()*vb[ts>=54]))
    summary['one_C_only']['imu_vs_gyro']=stats(deg((ic.inv()*i[ts>=54]).inv()*(sg(ct+offset).inv()*g[ts>=54])))
    # Paired segment averages describe repeat motions. No frame-wise significance.
    diff=np.array([x['ring']['angle_magnitude_difference']['mean']-x['palm']['angle_magnitude_difference']['mean'] for x in per_segment if 'palm'in x])
    summary['paired_segment_mae_difference_ring_minus_palm']=dict(mean=float(diff.mean()),median=float(np.median(diff)),min=float(diff.min()),max=float(diff.max()),ring_better_count=int(sum(diff<0)),palm_better_count=int(sum(diff>0)))
    jsonout(out/'metrics.json',summary);jsonout(out/'segment_metrics.json',per_segment)
    writecsv(out/'aligned_samples.csv',samples);writecsv(out/'steps.csv',allsteps)
    flat=[]
    for x in per_segment:
        f={k:x[k] for k in ['id','time_label','kind','speed','return_phase','intent','action','complaint','total_frames','paired_imu_frames','skin_valid']}
        for n in ['palm','ring']:
            if n not in x:continue
            for k in ['main_axis_gain_origin','main_axis_corr','main_axis_peak_deg','peak_deg','unintended_axis_peak_deg','unintended_axis_imu_residual_peak_deg']:
                f[n+'_'+k]=x[n].get(k)
            for k in ['angle_magnitude_difference','full_orientation_nominal_imu_difference']:
                for stat in ['mean','rmse','p95','max']:f[n+'_'+k+'_'+stat]=x[n][k][stat]
            for axis,j in [('yaw',0),('pitch',1),('roll',2)]:f[n+'_'+axis+'_peak_deg']=x[n]['axis_peak_deg'][j];f['imu_'+axis+'_peak_deg']=x['imu_axis_peak_deg'][j]
        flat.append(f)
    names={'id':'段号','time_label':'视频时间','kind':'动作类型','speed':'速度',
           'return_phase':'回程','intent':'用户意图','action':'实际动作描述','complaint':'用户指出的问题',
           'total_frames':'处理记录数','paired_imu_frames':'同源姿态及IMU有效配对帧数','skin_valid':'皮肤深度有效帧数'}
    measures={'main_axis_gain_origin':'主轴幅值拟合比例','main_axis_corr':'主轴相关系数',
              'main_axis_peak_deg':'主轴峰值_度','peak_deg':'总转角峰值_度',
              'unintended_axis_peak_deg':'非目标轴合成峰值_度',
              'unintended_axis_imu_residual_peak_deg':'非目标轴与IMU差峰值_度'}
    for method,label in [('palm','掌骨'),('ring','腕口')]:
        for key,title in measures.items():names[method+'_'+key]=label+'_'+title
        for key,title in [('angle_magnitude_difference','转角大小差'),('full_orientation_nominal_imu_difference','完整姿态差_名义IMU轴')]:
            for stat,title2 in [('mean','平均绝对值'),('rmse','RMSE'),('p95','95分位'),('max','最大值')]:
                names[method+'_'+key+'_'+stat]=label+'_'+title+'_'+title2+'_度'
    for method,label in [('palm','掌骨'),('ring','腕口'),('imu','IMU名义轴')]:
        for axis,title in [('yaw','偏航'),('pitch','俯仰'),('roll','横滚')]:names[method+'_'+axis+'_peak_deg']=label+'_'+title+'分量峰值_度'
    kinds={'translation_lr':'左右平移','translation_depth':'靠近远离相机','translation_image_ud':'画面上下平移','pitch':'俯仰','yaw':'偏航'}
    for row in flat:
        row['kind']=kinds[row['kind']];row['speed']={'slow':'慢速','medium':'中速','fast':'快速','unspecified':'未指定'}[row['speed']]
        row['return_phase']='是' if row['return_phase'] else '否'
    writecsv(out/'55段多指标对照.csv',[{names[k]:v for k,v in row.items()} for row in flat])
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
