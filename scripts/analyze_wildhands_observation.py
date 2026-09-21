#!/usr/bin/env python3
"""Paired published WildHands / recorded HaMeR evaluation, no robot execution."""
import argparse,csv,gzip,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_palm_ring_observation import deg,stats,writecsv,jsonout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'.runtime/palm_ring_surface_20260920';source=ROOT/'.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent'
    wild=[json.loads(x) for x in (args.run/'predictions.jsonl').open()];hamer=json.load(gzip.open(data/'paired_frames.json.gz','rt'));h={x['seq']:x for x in hamer};anns=json.load(open(data/'annotations_confirmed.json'))
    imu=np.load(source/'imu_signals.npz');it=imu['t'];si=Slerp(it,R.from_quat(imu['q']));offset=json.load(open(source/'time_alignment.json'))['used_offset_s'];ct=json.load(open(data/'metrics.json'))['c_video_s']
    paired=[x for x in wild if x['valid'] and x['seq']in h and h[x['seq']]['palm_valid'] and h[x['seq']]['ring_valid']]
    ts=np.array([x['t'] for x in paired]);ix=np.clip(np.searchsorted(it,ts+offset),1,len(it)-1);valid=imu['time_valid'][ix]&imu['time_valid'][ix-1]&(ts+offset>=it[0])&(ts+offset<=it[-1]);paired=[x for x,k in zip(paired,valid) if k];ts=ts[valid]
    ci=np.argmin(abs(ts-ct));i=si(ts+offset);ic=si(ct+offset)
    pr=R.from_quat([h[x['seq']]['palm_q'] for x in paired]);m=R.from_matrix([[0,0,1],[0,1,0],[-1,0,0]])*pr[ci].inv()
    rots={}
    for estimator in ['hamer','wildhands']:
        for frame in ['ring','palm']:
            q=R.from_quat([h[x['seq']][frame+'_q'] if estimator=='hamer' else x[frame+'_q'] for x in paired])
            rots[estimator+'_'+frame]=m*q*q[ci].inv()*m.inv()
    ri=ic.inv()*i;segment=[];sample=[]
    for a in anns:
        mask=np.flatnonzero((ts>=a['start']+.2)&(ts<=a['end']-.2));item=dict(a,n=len(mask))
        if len(mask)<3:segment.append(item);continue
        ir=ri[mask]*ri[mask[0]].inv();im=deg(ir);iv=np.rad2deg(ir.as_rotvec());item['imu_peak_deg']=float(im.max())
        cache={}
        for n,q in rots.items():
            rel=q[mask]*q[mask[0]].inv();mag=deg(rel);vec=np.rad2deg(rel.as_rotvec());err=deg(ir.inv()*rel)
            item[n]=dict(magnitude_difference=stats(abs(mag-im)),full_nominal_difference=stats(err),peak_deg=float(mag.max()),axis_peak_deg=np.max(abs(vec),axis=0).tolist())
            axis={'yaw':0,'pitch':1}.get(a['kind'])
            if axis is not None:
                xx=iv[:,axis];yy=vec[:,axis];item[n]['main_axis_gain']=float(np.dot(xx,yy)/np.dot(xx,xx)) if np.dot(xx,xx)>1 else None
                item[n]['main_axis_corr']=float(np.corrcoef(xx,yy)[0,1]) if np.std(xx)>.1 and np.std(yy)>.1 else None
            cache[n]=(mag,err,vec)
        for k,j in enumerate(mask):
            x=paired[j];row=dict(segment=a['id'],kind=a['kind'],speed=a['speed'],return_phase=a['return_phase'],t=ts[j],seq=x['seq'],imu_magnitude=im[k],
               source_ray_x_deg=float(np.degrees(np.arctan2((x['bbox'][0]+x['bbox'][2])/2-319.0163879394531,606.5111083984375))),
               source_ray_y_deg=float(np.degrees(np.arctan2((x['bbox'][1]+x['bbox'][3])/2-248.72071838378906,606.2513427734375))))
            for n,(mag,err,vec) in cache.items():
                row[n+'_magnitude']=mag[k];row[n+'_magnitude_error']=abs(mag[k]-im[k]);row[n+'_full_difference']=err[k]
                for axis,z in [('yaw',0),('pitch',1),('roll',2)]:row[n+'_'+axis]=vec[k,z]
            for axis,z in [('yaw',0),('pitch',1),('roll',2)]:row['imu_'+axis]=iv[k,z]
            sample.append(row)
        segment.append(item)
    summary=dict(total_source_rows=len(hamer),wildhands_processed=len(wild),wildhands_geometry_valid=sum(x['valid']for x in wild),paired_imu_rows=len(paired),paired_main_samples=len(sample),
        C_video_s=ct,common_time_offset_s=offset,shape_frozen_rows=sum(x.get('shape_frozen',False) for x in wild),
        failure_reasons=[x.get('failure') for x in wild if not x['valid']],categories={},temporal={},one_C_examples=[],
        latency=dict(wildhands_network_ms=stats([x['network_ms'] for x in wild if x['valid']]),wildhands_preprocess_ms=stats([x['preprocess_ms'] for x in wild if x['valid']]),wildhands_postprocess_ms=stats([x['postprocess_ms'] for x in wild if x['valid']]),
            recorded_hamer_network_ms=stats([h[x['seq']]['timing']['model_inference_s']*1000 for x in paired]),scope='different runs and precision; not a controlled end-to-end latency experiment'))
    for kind in ['all']+sorted(set(x['kind'] for x in sample)):
        ss=[x for x in sample if kind=='all' or x['kind']==kind]
        summary['categories'][kind]={n:{k:stats([x[n+'_'+k] for x in ss])for k in ['magnitude_error','full_difference']} for n in rots}
    dt=np.diff(ts);ok=(dt>.005)&(dt<.2);ir_step=deg(ri[1:]*ri[:-1].inv());quiet=ok&(ir_step/dt<5)&(ts[1:]>=54)
    for n,q in rots.items():
        step=deg(q[1:]*q[:-1].inv());summary['temporal'][n]=dict(all_step=stats(step[ok]),quiet_imu_step=stats(step[quiet]),over10_count=int(sum(step[ok]>10)),over20_count=int(sum(step[ok]>20)))
    for t in [62,72,220,238,268,295,308,322,332,345]:
        j=np.argmin(abs(ts-t));summary['one_C_examples'].append(dict(t=float(ts[j]),seq=paired[j]['seq'],imu_angle=float(deg(ri[j])),**{n:float(deg(q[j]))for n,q in rots.items()}))
    # Position-related bias is descriptive and uses a single C, not segment resets.
    masks={'horizontal_translation':(ts>=54)&(ts<=110),'vertical_image_translation':(ts>=176)&(ts<=209)}
    summary['position_dependence']={}
    for name,mask in masks.items():
        ray=np.array([np.arctan2((x['bbox'][0]+x['bbox'][2])/2-319.0163879394531,606.5111083984375) if name=='horizontal_translation' else np.arctan2((x['bbox'][1]+x['bbox'][3])/2-248.72071838378906,606.2513427734375) for x in paired]);ray=np.rad2deg(ray[mask]);axis=0 if name=='horizontal_translation' else 1
        z={}
        for n,q in rots.items():
            error=np.rad2deg(q[mask].as_rotvec())[:,axis]-np.rad2deg(ri[mask].as_rotvec())[:,axis]
            slope,bias=np.polyfit(ray,error,1);z[n]=dict(slope_deg_per_ray_degree=float(slope),correlation=float(np.corrcoef(ray,error)[0,1]),signed_component_difference_rms_deg=float(np.sqrt(np.mean(error**2))))
        summary['position_dependence'][name]=z
    jsonout(out/'metrics.json',summary);jsonout(out/'segment_metrics.json',segment);writecsv(out/'paired_samples.csv',sample)
    flat=[]
    for x in segment:
        row={'段号':x['id'],'时间':x['time_label'],'类型':x['kind'],'动作':x['action'],'配对帧数':x['n'],'IMU转角峰值_度':x.get('imu_peak_deg')}
        for n,title in [('hamer_ring','HaMeR腕口'),('wildhands_ring','WildHands腕口'),('hamer_palm','HaMeR掌骨'),('wildhands_palm','WildHands掌骨')]:
            if n not in x:continue
            for key,label in [('magnitude_difference','转角大小差'),('full_nominal_difference','完整姿态差_名义IMU轴')]:
                for stat,st in [('mean','均值'),('p95','P95'),('max','最大')]:row[title+'_'+label+'_'+st+'_度']=x[n][key][stat]
            row[title+'_转角峰值_度']=x[n]['peak_deg'];row[title+'_主轴比例']=x[n].get('main_axis_gain')
        flat.append(row)
    writecsv(out/'55段模型对照.csv',flat)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
