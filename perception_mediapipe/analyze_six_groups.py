"""Paired descriptive analysis of six unfiltered methods; no IMU fitting."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.runtime/six_group_fixed_left_20260917'
REPORT = ROOT/'docs/six_group_fixed_left_20260917'
GROUPS = list('ABCDEF')


def clean(x):
    if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v)for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def dump(path,x):
    path.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf-8')


def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig')as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        w.writerows({k:clean(v)for k,v in r.items()}for r in rows)


def category(a):
    kind=a.get('kind','');s=a['intent']
    if kind=='mixed':return '组合动作'
    if kind=='static' or '静止' in s or ('保持' in s and '标定' in s):return '静止'
    if kind.startswith('translation') or '平移' in s:return '平移'
    if kind=='rotation_fingers_tilt' or '俯仰' in s:return '俯仰'
    if kind=='rotation_side_flip' or '横滚' in s:return '横滚'
    if kind=='rotation_inplane' or '偏航' in s:return '偏航'
    if '丢失' in s:return '丢失片段'
    return '组合动作'


def speed(a):
    s=a['intent']
    if '快速' in s:return '快速'
    if '中速' in s:return '中速'
    if '慢速' in s or '低速' in s:return '慢速'
    return '未标速度'


def load(label,output_root=OUT):
    p=output_root/label
    mp=[json.loads(x)for x in open(p/'mp.jsonl')]
    h=[json.loads(x)for x in open(p/'hamer.jsonl')]
    assert len(mp)==len(h)
    t=np.array([x['t']for x in mp]);assert np.all(np.diff(t)>0)
    assert np.array_equal(t,np.array([x['t']for x in h]))
    imu=np.load(p/'imu_signals.npz');offset=json.load(open(p/'time_alignment.json'))['used_offset_s']
    si=Slerp(imu['t'],R.from_quat(imu['q']));sg=Slerp(imu['t'],R.from_quat(imu['gyro_q']))
    q=si(t+offset);gyro=sg(t+offset)
    imu_good=np.interp(t+offset,imu['t'],imu['time_valid'].astype(float))>.99
    d=dict(label=label,t=t,mp=mp,ha=h,imu=q,gyro=gyro,si=si,sg=sg,offset=offset,imu_good=imu_good,
           ann=json.load(open(p/'annotations.json')),meta=json.load(open(p/'metadata.json')))
    for g in GROUPS:
        rows=[x['groups'][g]for x in (h if g in 'ABCD' else mp)]
        v=np.array([r['valid']for r in rows]);pv=np.array([r.get('position_valid',False)for r in rows])
        rotations=np.full((len(t),3,3),np.nan);pos=np.full((len(t),3),np.nan)
        if v.any():rotations[v]=np.array([r['rotation']for r in rows if r['valid']]).reshape(-1,3,3)
        if pv.any():pos[pv]=np.array([r['position']for r in rows if r.get('position_valid')])
        d[g]=dict(rows=rows,valid=v,position_valid=pv,mat=rotations,pos=pos)
    return d


def longest_failure(t,valid):
    longest=0;duration=0.;start=None;count=0
    for i,v in enumerate(valid):
        if v or (i>0 and t[i]-t[i-1]>.25):
            if start is not None:
                longest=max(longest,count);duration=max(duration,t[i-1]-t[start])
            start=None;count=0
        if not v:
            if start is None:start=i
            count+=1
    if start is not None:longest=max(longest,count);duration=max(duration,t[-1]-t[start])
    return longest,duration


def angle(rot):return np.rad2deg((rot[0].inv()*rot).magnitude())


def errors(values):
    x=np.asarray(values)
    return dict(mae_deg=float(np.mean(abs(x))),rmse_deg=float(np.sqrt(np.mean(x*x))),
        p95_abs_deg=float(np.percentile(abs(x),95)),max_abs_deg=float(np.max(abs(x))),bias_deg=float(np.mean(x)))


def concentration(rot):
    v=(rot[0].inv()*rot).as_rotvec()
    eig=np.linalg.eigvalsh(v.T@v)
    return float(np.sqrt(max(0.,sum(eig[:2]))/max(sum(eig),1e-12)))


def corr(a,b):
    if np.std(a)<1e-8 or np.std(b)<1e-8:return np.nan
    return float(np.corrcoef(a,b)[0,1])


def main():
    REPORT.mkdir(exist_ok=True,parents=True)
    datasets=[load(s)for s in ['earlier','later']]
    metrics=[];own=[];coverage=[];failure_rows=[];frame_errors=[];traces={};excluded=[];pooled={g:[]for g in GROUPS}
    fixed_zero=[];fixed_zero_frames=[];fixed_traces={}
    for d in datasets:
        t=d['t'];n=len(t)
        valid_all=d['imu_good']&np.logical_and.reduce([d[g]['valid']for g in GROUPS])
        fixed_ids=np.flatnonzero(valid_all);fixed_truth=angle(d['imu'][fixed_ids])
        fixed_gyro=angle(d['gyro'][fixed_ids])
        fixed_traces[d['label']]=dict(t=t[fixed_ids],imu=fixed_truth,gyro=fixed_gyro,models={})
        for g in GROUPS:
            fixed_q=R.from_matrix(d[g]['mat'][fixed_ids]);fixed_angle=angle(fixed_q)
            fixed_traces[d['label']]['models'][g]=fixed_angle
            fixed_zero.append(dict(session=d['label'],group=g,n=len(fixed_ids),reference_time=float(t[fixed_ids[0]]),
                gyro_mae_deg=float(np.mean(abs(fixed_angle-fixed_gyro))),
                reference_disagreement_mae_deg=float(np.mean(abs(fixed_truth-fixed_gyro))),**errors(fixed_angle-fixed_truth)))
            for k,i in enumerate(fixed_ids):
                fixed_zero_frames.append(dict(session=d['label'],group=g,t=t[i],reference_time=float(t[fixed_ids[0]]),
                    imu_angle_deg=fixed_truth[k],model_angle_deg=fixed_angle[k],error_deg=fixed_angle[k]-fixed_truth[k]))
        for g in GROUPS:
            v=d[g]['valid'];pv=d[g]['position_valid'];c=Counter(r.get('reason','UNSPECIFIED')for r in d[g]['rows']if not r['valid'])
            longest,seconds=longest_failure(t,v)
            timings=np.array([r['processing_ms']for r in d[g]['rows']if 'processing_ms'in r])
            geom=np.array([r['geometry_ms']for r in d[g]['rows']if 'geometry_ms'in r])
            coverage.append(dict(session=d['label'],group=g,frames=n,valid_orientation=int(v.sum()),valid_position=int(pv.sum()),
                valid_pose=int((v&pv).sum()),orientation_rate_pct=100*v.mean(),pose_rate_pct=100*(v&pv).mean(),
                reasons=dict(c),handedness_mismatch=sum(r.get('handedness_mismatch',False)for r in d['mp']),
                applied_right_hand_frames=sum(bool(r.get('detection',{}).get('is_right'))for r in d['mp']if r.get('detection')),
                handedness_policy=d['meta'].get('handedness_policy','automatic'),
                longest_observed_failure_frames=longest,longest_observed_failure_duration_s=seconds,
                processing_median_ms=float(np.median(timings)),processing_p95_ms=float(np.percentile(timings,95)),
                geometry_median_ms=float(np.median(geom))))
            for i,r in enumerate(d[g]['rows']):
                if not r['valid'] or not r.get('position_valid') or d['mp'][i].get('handedness_mismatch'):
                    failure_rows.append(dict(session=d['label'],group=g,frame=d['mp'][i]['frame_number'],t=t[i],
                        orientation_valid=r['valid'],position_valid=r.get('position_valid',False),reason=r.get('reason',''),
                        position_reason=r.get('position_reason',''),handedness_mismatch=d['mp'][i].get('handedness_mismatch',False)))
        for a in d['ann']:
            eligible=(t>=a['start'])&(t<=a['end'])
            common=eligible&d['imu_good']
            for g in GROUPS:common&=d[g]['valid']
            ids=np.flatnonzero(common)
            if len(ids)<3:
                excluded.append(dict(session=d['label'],id=a['id'],reason='FEWER_THAN_THREE_COMMON_VALID_FRAMES',n=len(ids)))
                continue
            iq=d['imu'][ids];truth=angle(iq);gyro_truth=angle(d['gyro'][ids]);it=t[ids]
            pi,pj=np.triu_indices(len(ids),1);pair_truth=np.rad2deg((iq[pi].inv()*iq[pj]).magnitude())
            cat=category(a);intended={'俯仰':0,'横滚':1,'偏航':2}.get(cat)
            trace=dict(session=d['label'],id=a['id'],intent=a['intent'],category=cat,t=it,imu=truth,gyro=gyro_truth,models={})
            for g in GROUPS:
                q=R.from_matrix(d[g]['mat'][ids]);turn=angle(q);err=turn-truth;pooled[g].extend(err)
                rv=np.rad2deg((q[0].inv()*q).as_rotvec())
                pair=np.rad2deg((q[pi].inv()*q[pj]).magnitude())
                spread=np.rad2deg((q.mean().inv()*q).magnitude())
                delta=np.rad2deg((q[:-1].inv()*q[1:]).magnitude());steps=np.diff(it);cont=steps<.25
                posids=ids[d[g]['position_valid'][ids]];pos=d[g]['pos'][posids]
                radial=np.linalg.norm(pos-pos.mean(0),axis=1)*1000 if len(pos) else np.array([np.nan])
                pos_extent=np.ptp(pos,axis=0)*1000 if len(pos) else np.full(3,np.nan)
                gain=float(np.dot(turn,truth)/np.dot(truth,truth)) if truth.max()>5 else np.nan
                m=dict(session=d['label'],id=a['id'],intent=a['intent'],category=cat,speed=speed(a),group=g,
                    start=a['start'],end=a['end'],n=len(ids),eligible=int(eligible.sum()),reference_time=float(it[0]),
                    **errors(err),pair_mae_deg=float(np.mean(abs(pair-pair_truth))),pair_rmse_deg=float(np.sqrt(np.mean((pair-pair_truth)**2))),
                    imu_peak_deg=float(truth.max()),model_peak_deg=float(turn.max()),amplitude_gain=gain,
                    peak_ratio=float(turn.max()/truth.max())if truth.max()>5 else np.nan,
                    correlation=corr(turn,truth),end_angle_error_deg=float(err[-1]),
                    orientation_spread_rms_deg=float(np.sqrt(np.mean(spread**2))),orientation_spread_p95_deg=float(np.percentile(spread,95)),
                    adjacent_step_median_deg=float(np.median(delta[cont]))if cont.any()else np.nan,
                    adjacent_step_p95_deg=float(np.percentile(delta[cont],95))if cont.any()else np.nan,
                    jumps_over_10deg=int(np.sum((delta>10)&cont)),position_spread_rms_mm=float(np.sqrt(np.mean(radial**2))),
                    position_n=len(pos),position_span_x_mm=pos_extent[0],position_span_y_mm=pos_extent[1],position_span_z_mm=pos_extent[2],
                    axis_x_rms_deg=float(np.sqrt(np.mean(rv[:,0]**2))),axis_y_rms_deg=float(np.sqrt(np.mean(rv[:,1]**2))),axis_z_rms_deg=float(np.sqrt(np.mean(rv[:,2]**2))),
                    imu_nonprincipal_ratio=concentration(iq),model_nonprincipal_ratio=concentration(q),
                    gyro_reference_mae_deg=float(np.mean(abs(turn-gyro_truth))),imu_gyro_disagreement_mae_deg=float(np.mean(abs(truth-gyro_truth))))
                correct_hand=np.array([not d['mp'][i].get('handedness_mismatch',False)for i in ids])
                m['handedness_mismatch_common_frames']=int((~correct_hand).sum())
                m['mae_without_handedness_mismatch_deg']=float(np.mean(abs(err[correct_hand])))if correct_hand.any()else np.nan
                if intended is not None:
                    off=np.delete(rv,intended,axis=1)
                    m.update(off_intent_axis_rms_deg=float(np.sqrt(np.mean(np.sum(off**2,axis=1)))),
                             intended_axis_rms_deg=float(np.sqrt(np.mean(rv[:,intended]**2))),
                             off_intent_energy_ratio=float(np.sqrt(np.sum(off**2)/max(np.sum(rv**2),1e-12))))
                for shift,name in [(-.04,'minus40ms'),(.04,'plus40ms')]:
                    shifted=d['si'](it+d['offset']+shift)
                    m['mae_'+name+'_deg']=float(np.mean(abs(turn-angle(shifted))))
                if intended is not None and truth.max()>15 and len(ids)>=12:
                    lags=np.arange(-.25,.2501,.01);correlations=[]
                    for lag in lags:
                        shifted=d['si'](it+d['offset']-lag)
                        correlations.append(corr(turn,angle(shifted)))
                    best=int(np.nanargmax(correlations))
                    m.update(apparent_lag_ms=float(lags[best]*1000),lag_best_correlation=correlations[best],
                             lag_at_search_boundary=bool(best in [0,len(lags)-1]))
                metrics.append(m)
                trace['models'][g]=dict(angle=turn,rotvec=rv,position_mm=d[g]['pos'][ids]*1000,position_valid=d[g]['position_valid'][ids])
                for k,index in enumerate(ids):frame_errors.append(dict(session=d['label'],segment=a['id'],group=g,t=t[index],
                    imu_angle_deg=truth[k],model_angle_deg=turn[k],angle_error_deg=err[k],
                    axis_x_deg=rv[k,0],axis_y_deg=rv[k,1],axis_z_deg=rv[k,2],
                    position_valid=bool(d[g]['position_valid'][index]),x_m=d[g]['pos'][index,0],y_m=d[g]['pos'][index,1],z_m=d[g]['pos'][index,2]))
                # Own-valid sensitivity keeps the same common zero instant.
                ownids=np.flatnonzero(eligible&d['imu_good']&d[g]['valid']&(t>=it[0]))
                oq=R.from_matrix(d[g]['mat'][ownids]);oi=d['imu'][ownids]
                ot=np.rad2deg((q[0].inv()*oq).magnitude());truth_own=np.rad2deg((iq[0].inv()*oi).magnitude())
                own.append(dict(session=d['label'],id=a['id'],group=g,n=len(ownids),**errors(ot-truth_own)))
            traces[d['label']+'_'+str(a['id'])]=trace
    summary={}
    for g in GROUPS:
        rows=[r for r in metrics if r['group']==g];stat=[r for r in rows if r['category']=='静止'];cov=[r for r in coverage if r['group']==g]
        summary[g]=dict(n_segments=len(rows),n_common_comparisons=sum(r['n']for r in rows),
            macro_mae_deg=float(np.mean([r['mae_deg']for r in rows])),macro_rmse_deg=float(np.mean([r['rmse_deg']for r in rows])),
            macro_pair_rmse_deg=float(np.mean([r['pair_rmse_deg']for r in rows])),pooled=errors(pooled[g]),
            own_valid_macro_mae_deg=float(np.mean([r['mae_deg']for r in own if r['group']==g])),
            macro_mae_minus40ms_deg=float(np.mean([r['mae_minus40ms_deg']for r in rows])),macro_mae_plus40ms_deg=float(np.mean([r['mae_plus40ms_deg']for r in rows])),
            valid_orientation=sum(r['valid_orientation']for r in cov),valid_pose=sum(r['valid_pose']for r in cov),frames=sum(r['frames']for r in cov),
            static_rotation_rms_median_deg=float(np.median([r['orientation_spread_rms_deg']for r in stat])),
            static_position_rms_median_mm=float(np.median([r['position_spread_rms_mm']for r in stat])))
    stratified=[]
    for field in ['session','category','speed']:
        for value in sorted(set(r[field]for r in metrics)):
            for g in GROUPS:
                rows=[r for r in metrics if r['group']==g and r[field]==value]
                stratified.append(dict(stratum=field,value=value,group=g,n_segments=len(rows),
                    mae_deg=float(np.mean([r['mae_deg']for r in rows])),rmse_deg=float(np.mean([r['rmse_deg']for r in rows])),
                    pair_rmse_deg=float(np.mean([r['pair_rmse_deg']for r in rows]))))
    paired=[]
    for label in ['earlier','later','all']:
        for other in 'ACD':
            b=[r for r in metrics if r['group']=='B' and (label=='all'or r['session']==label)]
            lookup={(r['session'],r['id']):r for r in metrics if r['group']==other}
            delta=np.array([r['mae_deg']-lookup[r['session'],r['id']]['mae_deg']for r in b])
            paired.append(dict(session=label,comparison='B-'+other,n=len(delta),mean_delta_mae_deg=delta.mean(),median_delta_mae_deg=np.median(delta),
                q25=np.percentile(delta,25),q75=np.percentile(delta,75),B_lower_error_count=int((delta<0).sum()),
                B_lower_by_more_than_0_1deg=int((delta<-.1).sum()),B_higher_by_more_than_0_1deg=int((delta>.1).sum())))
    manifest=dict(models=summary,coverage=coverage,paired=paired,excluded_segments=excluded,
        dataset=[d['meta']for d in datasets],n_common_frames_unique=sum(int(np.logical_and.reduce([d[g]['valid']for g in GROUPS]).sum())for d in datasets),
        reference_validity=[dict(session=d['label'],frames=len(d['t']),imu_valid=int(d['imu_good'].sum()),
            six_methods_and_imu_valid=int((d['imu_good']&np.logical_and.reduce([d[g]['valid']for g in GROUPS])).sum()))for d in datasets],
        scope='Unfiltered angle-magnitude agreement with synchronized IMU; no independent orientation extrinsics, position truth or end-to-end robot latency.')
    dump(REPORT/'summary.json',manifest);dump(REPORT/'segment_metrics.json',metrics);dump(OUT/'traces.json',traces)
    dump(OUT/'fixed_zero_traces.json',fixed_traces)
    csvwrite(REPORT/'逐动作六组指标.csv',metrics);csvwrite(REPORT/'各组自身有效帧指标.csv',own)
    csvwrite(REPORT/'失败和左右手误判明细.csv',failure_rows);csvwrite(REPORT/'逐帧误差与三轴变化.csv',frame_errors)
    csvwrite(REPORT/'按录像动作速度汇总.csv',stratified);csvwrite(REPORT/'腕环配对增益.csv',paired)
    csvwrite(REPORT/'整段录像固定零位指标.csv',fixed_zero);csvwrite(REPORT/'整段录像固定零位逐帧.csv',fixed_zero_frames)
    csvwrite(REPORT/'有效率与耗时.csv',[{k:v for k,v in r.items()if k!='reasons'}for r in coverage])
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
