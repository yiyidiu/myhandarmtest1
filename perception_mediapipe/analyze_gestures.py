"""Gesture-dependent pose agreement and within-MANO deformation audit.

No output filtering, reference-based pose correction, inferred depth filling,
error-based frame rejection, or per-gesture clock-offset optimization is used.
Relative angle magnitude agreement is NOT full 3-D orientation error.
"""
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perception_mediapipe.analyze_six_groups import load,angle,errors,corr,concentration,csvwrite,dump,clean,longest_failure
from perception_mediapipe.fingers import _digit_shape
OUT=ROOT/'.runtime/gesture_pose_20260917';RESULT=OUT/'results';REPORT=ROOT/'docs/gesture_pose_comparison_20260917'
GROUPS=list('ABCDEF')


def stats(x,prefix):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return {prefix+k:np.nan for k in ['mean','rms','p95','max']}
    return {prefix+'mean':float(x.mean()),prefix+'rms':float(np.sqrt(np.mean(x*x))),prefix+'p95':float(np.percentile(x,95)),prefix+'max':float(x.max())}


def relative(q,zero):return np.rad2deg((zero.inv()*q).magnitude())


def base_flexion(points):
    """Wrist-MCP ray versus proximal phalanx angle; not clinical MCP truth."""
    p=np.asarray(points);a=p[[5,9,13,17]]-p[0];b=p[[6,10,14,18]]-p[[5,9,13,17]]
    cosine=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    return np.rad2deg(np.arccos(np.clip(cosine,-1,1)))


def add_features(d):
    n=len(d['t']);p=RESULT/d['label'];raw=np.load(p/'hamer_raw_geometry.npz')
    fi=raw['frame_index'].astype(int);joints=raw['joints'];vertices=raw['vertices']
    template=json.load(open(RESULT/'geometry_definition.json'))['False'];ring=template['wrist_ring'];regions=template['regions']
    for name in ['hamer_bend','mp3d_bend','mp2d_bend']:
        d[name]=np.full((n,5),np.nan)
        d[name+'_base']=np.full((n,4),np.nan)
    for idx,j in zip(fi,joints):
        try:d['hamer_bend'][idx]=_digit_shape(j)[0];d['hamer_bend_base'][idx]=base_flexion(j)
        except ValueError:pass
    for i,row in enumerate(d['mp']):
        det=row.get('detection')
        if not det:continue
        for name,key in [('mp3d_bend','world'),('mp2d_bend','xy')]:
            try:d[name][i]=_digit_shape(np.asarray(det[key]))[0];d[name+'_base'][i]=base_flexion(det[key])
            except ValueError:pass
    # Express predicted shape in A's anatomical root frame. This removes the
    # estimated global rotation exactly; it does not create physical truth.
    local=np.einsum('nij,njk->nik',vertices-joints[:,0,None,:],d['A']['mat'][fi])
    d['ring_center_local']=np.full((n,3),np.nan);d['ring_center_local'][fi]=local[:,ring].mean(1)
    d['region_center_local']=np.full((n,4,3),np.nan)
    d['region_center_local'][fi]=np.stack([local[:,ids].mean(1)for ids in regions.values()],axis=1)
    d['palm_joint_local']=np.full((n,5,3),np.nan)
    d['palm_joint_local'][fi]=np.einsum('nij,njk->nik',joints[:,[0,5,9,13,17]]-joints[:,0,None,:],d['A']['mat'][fi])
    for g in 'ABCD':
        v=d['A']['valid']&d[g]['valid'];q=np.full((n,3,3),np.nan)
        q[v]=np.swapaxes(d['A']['mat'][v],1,2)@d[g]['mat'][v];d[g]['root_removed']=q
    raw.close()


def assess_clock(d):
    """Diagnostic only: estimate apparent alignment on held-out open controls.

    Estimate before applying the one common offset selected on open controls.
    Estimation is not hardware synchronization and can contain model errors.
    """
    rows=[];t=d['t'];imu_raw=np.load(RESULT/d['label']/'imu_signals.npz')
    gt=imu_raw['t'];speed=np.linalg.norm(imu_raw['gyro_rad_s'],axis=1)
    for a in d['ann']:
        if a['phase'] not in ['rotation_calibration','rotation_validation']:continue
        for g in ['A','E']:
            # Finite differences over three capture intervals (~0.1 s) are a
            # diagnostic angular speed only; reported poses remain unfiltered.
            i=np.flatnonzero((t>=a['start'])&(t<a['end']))
            i=i[i+3<len(t)];j=i+3
            ok=(t[j]<a['end'])&(t[j]-t[i]<.15)&d[g]['valid'][i]&d[g]['valid'][j]&d['imu_good'][i]&d['imu_good'][j]
            i,j=i[ok],j[ok]
            if len(i)<10:continue
            vs=(R.from_matrix(d[g]['mat'][i]).inv()*R.from_matrix(d[g]['mat'][j])).magnitude()/(t[j]-t[i]);mid=(t[i]+t[j])/2
            offsets=np.arange(-.30,.3001,.01);cs=[corr(vs,np.interp(mid+off,gt,speed))for off in offsets]
            k=int(np.nanargmax(cs))if np.any(np.isfinite(cs))else 0
            rows.append(dict(session=d['label'],phase=a['phase'],id=a['id'],start=a['start'],end=a['end'],group=g,n=len(i),
                nominal_speed_correlation=cs[30],best_offset_ms=offsets[k]*1000,best_correlation=cs[k],search_boundary=k in [0,len(cs)-1],
                used_offset_s=0.,diagnostic_only=True))
    return rows


def apply_reference_time(d,offset):
    d['offset']=float(offset);d['imu']=d['si'](d['t']+offset);d['gyro']=d['sg'](d['t']+offset)
    raw=np.load(RESULT/d['label']/'imu_signals.npz')
    d['imu_good']=np.interp(d['t']+offset,raw['t'],raw['time_valid'].astype(float))>.99
    raw.close()


def motion_metrics(d,ids,g,zero,with_lag=False):
    if len(ids)<3:return {}
    q=R.from_matrix(d[g]['mat'][ids]);q0=R.from_matrix(d[g]['mat'][zero]);iq=d['imu'][ids];i0=d['imu'][zero]
    va=relative(q,q0);ia=relative(iq,i0);ga=relative(d['gyro'][ids],d['gyro'][zero]);err=va-ia
    localv=angle(q);locali=angle(iq)
    rv=np.rad2deg((q0.inv()*q).as_rotvec());step=np.rad2deg((q[:-1].inv()*q[1:]).magnitude())
    istep=np.rad2deg((iq[:-1].inv()*iq[1:]).magnitude());cont=(np.diff(d['t'][ids])<.05)&(np.diff(ids)==1)
    spread=np.rad2deg((q.mean().inv()*q).magnitude());ispread=np.rad2deg((iq.mean().inv()*iq).magnitude())
    posids=ids[d[g]['position_valid'][ids]];p=d[g]['pos'][posids]
    # Root magnitude is frame invariant, unlike uncalibrated IMU/model axes.
    m=dict(n=len(ids),zero_t=d['t'][zero],**errors(err),local_phase_mae_deg=float(np.mean(abs(localv-locali))),
        imu_excursion_deg=float(locali.max()),model_excursion_deg=float(localv.max()),
        correlation=corr(va,ia),amplitude_gain=float(np.dot(localv,locali)/np.dot(locali,locali))if locali.max()>5 else np.nan,
        end_error_deg=float(err[-1]),imu_gyro_mae_deg=float(np.mean(abs(ia-ga))),
        gyro_reference_mae_deg=float(np.mean(abs(va-ga))),gyro_reference_rmse_deg=float(np.sqrt(np.mean((va-ga)**2))),
        model_nonprincipal_ratio=concentration(q),imu_nonprincipal_ratio=concentration(iq),
        orientation_rms_deg=float(np.sqrt(np.mean(spread**2))),imu_orientation_rms_deg=float(np.sqrt(np.mean(ispread**2))),
        adjacent_pairs=int(cont.sum()),jumps_gt10_deg=int(((step>10)&cont).sum()),reference_steps_gt10_deg=int(((istep>10)&cont).sum()),
        adjacent_step_median_deg=float(np.median(step[cont]))if cont.any()else np.nan,
        adjacent_step_p95_deg=float(np.percentile(step[cont],95))if cont.any()else np.nan,
        adjacent_step_error_mae_deg=float(np.mean(abs(step[cont]-istep[cont])))if cont.any()else np.nan,
        axis_x_rms_deg=float(np.sqrt(np.mean(rv[:,0]**2))),axis_y_rms_deg=float(np.sqrt(np.mean(rv[:,1]**2))),axis_z_rms_deg=float(np.sqrt(np.mean(rv[:,2]**2))),
        position_n=len(p),position_scatter_rms_mm=float(np.sqrt(np.mean(np.sum((p-p.mean(0))**2,axis=1)))*1000)if len(p)else np.nan)
    for j,c in enumerate('xyz'):m['position_span_'+c+'_mm']=float(np.ptp(p[:,j])*1000)if len(p)else np.nan
    for shift,name in [(-.10,'minus100ms'),(-.04,'minus40ms'),(.04,'plus40ms'),(.10,'plus100ms'),(-d['offset'],'nominal_host_time')]:
        ia2=relative(d['si'](d['t'][ids]+d['offset']+shift),d['si']([d['t'][zero]+d['offset']+shift])[0])
        m['mae_'+name+'_deg']=float(np.mean(abs(va-ia2)))
    if with_lag and locali.max()>15 and len(ids)>20:
        shifts=np.arange(-.30,.3001,.01);cc=[]
        for lag in shifts:
            ref=relative(d['si'](d['t'][ids]+d['offset']-lag),d['si']([d['t'][zero]+d['offset']-lag])[0]);cc.append(corr(va,ref))
        if np.any(np.isfinite(cc)):
            k=int(np.nanargmax(cc));m.update(apparent_lag_ms=shifts[k]*1000,lag_correlation=cc[k],lag_search_boundary=k in [0,len(cc)-1])
    return m


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    allphase=[];own=[];cycles=[];coverage=[];failures=[];coupling=[];shape=[];paired=[];quiet=[];clock=[];frame_rows=[];traces={};fixed=[]
    datasets=[]
    for label in ['s1','s2']:
        d=load(label,RESULT);apply_reference_time(d,0.);clock_rows=assess_clock(d)
        # Four independent estimates from two open-hand control intervals,
        # using A and E equally. One median offset is shared by A-F. No gesture
        # angle errors or method rankings enter this clock diagnostic.
        estimates=[r['best_offset_ms']for r in clock_rows if r['phase']=='rotation_calibration']
        offset=round(float(np.median(estimates))/1000,6);apply_reference_time(d,offset)
        for row in clock_rows:row['used_offset_s']=offset
        alignment=dict(used_offset_s=offset,method='median of four A/E angular-speed cross-correlation offsets on two predeclared open-hand rotation controls',
            calibration_offsets_ms=estimates,validation='separate rotation_validation intervals, never used to choose offset',
            absolute_sensor_latency_known=False,warning='Exploratory coarse time alignment, not a device-level synchronized reference. No per-method or per-gesture optimization.')
        dump(RESULT/label/'time_alignment.json',alignment)
        add_features(d);datasets.append(d);clock.extend(clock_rows);t=d['t'];n=len(t)
        common=d['imu_good']&np.logical_and.reduce([d[g]['valid']for g in GROUPS])
        abcd_model=np.logical_and.reduce([d[g]['valid']for g in 'ABCD']);abcd=d['imu_good']&abcd_model
        # Availability uses every requested source frame, irrespective of IMU.
        for state in ['all','open','half','closed','closing','opening','ambiguous']:
            sel=np.ones(n,bool)if state=='all'else np.zeros(n,bool)
            if state!='all':
                for a in d['ann']:
                    if a['gesture']==state:sel|=(t>=a['start'])&(t<a['end'])
            ids=np.flatnonzero(sel)
            for g in GROUPS:
                v=d[g]['valid'][ids];pv=d[g]['position_valid'][ids];rs=[d[g]['rows'][i]for i in ids];cnt=Counter(r.get('reason','')for r in rs if not r['valid'])
                pcnt=Counter(r.get('position_reason',r.get('reason',''))for r in rs if not r.get('position_valid'))
                longest,dur=longest_failure(t[ids],v)
                coverage.append(dict(session=label,gesture=state,group=g,n=len(ids),orientation_valid=int(v.sum()),position_valid=int(pv.sum()),pose_valid=int((v&pv).sum()),
                    orientation_rate_pct=100*v.mean()if len(v)else np.nan,pose_rate_pct=100*(v&pv).mean()if len(v)else np.nan,
                    invalid_reasons=dict(cnt),position_invalid_reasons=dict(pcnt),longest_loss_frames=longest,longest_loss_s=dur))
        for g in GROUPS:
            for i,r in enumerate(d[g]['rows']):
                if r['valid']and r.get('position_valid'):continue
                a=next((a for a in d['ann']if a['start']<=t[i]<a['end']),{})
                failures.append(dict(session=label,t=t[i],frame=d['mp'][i]['frame_number'],group=g,gesture=a.get('gesture'),cycle_id=a.get('cycle_id'),
                    orientation_valid=r['valid'],position_valid=r.get('position_valid',False),reason=r.get('reason',''),position_reason=r.get('position_reason','')))
        # A single zero per gesture cycle, shared by all six methods.
        zeromap={}
        for cy in sorted({a['cycle_id']for a in d['ann']if a['cycle_id']}):
            aa=[a for a in d['ann']if a['cycle_id']==cy];lo=min(a['start']for a in aa);hi=max(a['end']for a in aa)
            ids=np.flatnonzero(common&(t>=lo)&(t<hi))
            if not len(ids):continue
            zero=ids[0];zeromap[cy]=zero;key=f'{label}_cycle{cy}'
            traces[key]=dict(session=label,cycle_id=cy,start=lo,end=hi,t=t[ids],imu=relative(d['imu'][ids],d['imu'][zero]),
                gyro=relative(d['gyro'][ids],d['gyro'][zero]),phases=aa,models={},finger_t=t,
                hamer_bend=np.nanmean(d['hamer_bend'][:,1:],axis=1),mp3d_bend=np.nanmean(d['mp3d_bend'][:,1:],axis=1),mp2d_bend=np.nanmean(d['mp2d_bend'][:,1:],axis=1))
            for g in GROUPS:
                m=motion_metrics(d,ids,g,zero,with_lag=True);cycles.append(dict(session=label,cycle_id=cy,group=g,start=lo,end=hi,**m))
                traces[key]['models'][g]=relative(R.from_matrix(d[g]['mat'][ids]),R.from_matrix(d[g]['mat'][zero]))
                if g in 'ABCD':
                    shapeids=np.flatnonzero(abcd_model&(t>=t[zero])&(t<hi));traces[key]['root_t']=t[shapeids]
                    traces[key].setdefault('root_extra',{})[g]=relative(R.from_matrix(d[g]['root_removed'][shapeids]),R.from_matrix(d[g]['root_removed'][zero]))
            # Additional joint A-D comparison avoids conditioning on F depth.
            aid=np.flatnonzero(abcd&(t>=lo)&(t<hi));az=aid[0]
            am={g:motion_metrics(d,aid,g,az)for g in 'ABCD'}
            for other in 'ACD':paired.append(dict(session=label,cycle_id=cy,comparison='B-'+other,n=len(aid),delta_mae_deg=am['B']['mae_deg']-am[other]['mae_deg']))
        fixedids=np.flatnonzero(common)
        for g in GROUPS:fixed.append(dict(session=label,group=g,**motion_metrics(d,fixedids,g,fixedids[0])))
        for a in d['ann']:
            eligible=(t>=a['start'])&(t<a['end']);ids=np.flatnonzero(eligible&common)
            if len(ids)<3:continue
            zero=zeromap.get(a['cycle_id'],ids[0]);ids=ids[ids>=zero]
            if len(ids)<3:continue
            base={k:a[k]for k in ['id','cycle_id','start','end','gesture','phase','intent']};base.update(session=label,eligible=int(eligible.sum()))
            for g in GROUPS:
                m=motion_metrics(d,ids,g,zero);allphase.append(dict(**base,group=g,**m))
                ownids=np.flatnonzero(eligible&d['imu_good']&d[g]['valid']&(t>=t[zero]))
                own.append(dict(**base,group=g,**motion_metrics(d,ownids,g,zero)))
                q0=R.from_matrix(d[g]['mat'][zero]);rot=R.from_matrix(d[g]['mat'][ids]);va=relative(rot,q0);ia=relative(d['imu'][ids],d['imu'][zero]);rv=np.rad2deg((q0.inv()*rot).as_rotvec())
                for k,i in enumerate(ids):frame_rows.append(dict(session=label,phase_id=a['id'],cycle_id=a['cycle_id'],phase=a['phase'],gesture=a['gesture'],group=g,t=t[i],zero_t=t[zero],
                    imu_angle_deg=ia[k],visual_angle_deg=va[k],error_deg=va[k]-ia[k],x_deg=rv[k,0],y_deg=rv[k,1],z_deg=rv[k,2]))
            # Shape features kept separate: F's 2-D bends are not 3-D bends.
            for name in ['hamer_bend','mp3d_bend','mp2d_bend']:
                sids=np.flatnonzero(eligible&np.all(np.isfinite(d[name]),axis=1));f=d[name][sids]
                shape.append(dict(**base,model=name,n=len(sids),long_finger_mean_bend=float(f[:,1:].mean())if len(f)else np.nan,
                    long_finger_base_angle_deg=float(d[name+'_base'][sids].mean())if len(f)else np.nan,
                    **{digit+'_bend_mean':float(f[:,k].mean())if len(f)else np.nan for k,digit in enumerate(['thumb','index','middle','ring','little'])}))
            # The same A-D prediction, subtract only A global pose. C/D may
            # still deform with articulation even if the root were perfect.
            aids=np.flatnonzero(eligible&abcd_model&(t>=t[zero]));az=zero
            for g in 'ABCD':
                if len(aids)<3:continue
                rr=R.from_matrix(d[g]['root_removed'][aids]);rz=R.from_matrix(d[g]['root_removed'][az]);delta=np.rad2deg((rz.inv()*rr).as_rotvec());da=np.linalg.norm(delta,axis=1)
                ringmotion=np.linalg.norm(d['ring_center_local'][aids]-d['ring_center_local'][az],axis=1)*1000
                regional=np.linalg.norm(d['region_center_local'][aids]-d['region_center_local'][az],axis=2)*1000
                joint=np.linalg.norm(d['palm_joint_local'][aids]-d['palm_joint_local'][az],axis=2)*1000
                coupling.append(dict(**base,group=g,n=len(aids),**stats(da,'extra_axis_deg_'),
                    x_extra_rms_deg=float(np.sqrt(np.mean(delta[:,0]**2))),y_extra_rms_deg=float(np.sqrt(np.mean(delta[:,1]**2))),z_extra_rms_deg=float(np.sqrt(np.mean(delta[:,2]**2))),
                    ring_center_motion_mean_mm=float(ringmotion.mean()),ring_center_motion_max_mm=float(ringmotion.max()),
                    region_center_motion_mean_mm=float(regional.mean()),region_center_motion_max_mm=float(regional.max()),
                    palm_joint_motion_mean_mm=float(joint.mean()),palm_joint_motion_max_mm=float(joint.max()),
                    bend_extra_axis_correlation=corr(d['hamer_bend'][aids,1:].mean(1),da)))
            # Non-overlapping 1 s windows, accepted solely on both IMU signals
            # moving <= 2 deg and at least 20/30 common-valid camera frames.
            # These are quiet-reference observations, not a wrist fixture.
            if a['phase'] in ['rotation_calibration','rotation_validation']:continue
            for lo in np.arange(a['start'],a['end']-.99,1.):
                qids=np.flatnonzero(common&(t>=lo)&(t<lo+1))
                if len(qids)<20:continue
                im=angle(d['imu'][qids]);gy=angle(d['gyro'][qids])
                if max(im.max(),gy.max())>2:continue
                for g in GROUPS:
                    quiet.append(dict(**{k:base[k]for k in ['session','id','cycle_id','gesture','phase']},window_start=lo,window_end=lo+1,
                        imu_peak_deg=float(im.max()),gyro_peak_deg=float(gy.max()),group=g,**motion_metrics(d,qids,g,qids[0])))
        print('analysed',label,n,'six + IMU common',int(common.sum()),flush=True)
    # Pooled errors give frame weights; macro values give every phase equal
    # weight. Neither treats serial frames as independent human participants.
    summary=[]
    for scope in ['all_selected','gesture_cycles','confirmed_gesture_cycles','open_controls']:
      for state in ['all','open','half','closed','closing','opening','ambiguous']:
        for g in GROUPS:
            def in_scope(r):return(scope=='all_selected'or(scope=='gesture_cycles'and r['cycle_id']>0)or(scope=='confirmed_gesture_cycles'and r['cycle_id']>0 and r['cycle_id']!=7)or(scope=='open_controls'and r['cycle_id']==0))
            def selected(r):
                return r['group']==g and(state=='all'or r['gesture']==state)and in_scope(r)
            rows=[r for r in allphase if selected(r)]
            fr=[r for r in frame_rows if selected(r)]
            ownrows=[r for r in own if selected(r)and r.get('n',0)>0]
            if not fr:continue
            nn=nv=npv=0
            for ds in datasets:
                mask=np.zeros(len(ds['t']),bool)
                for a in ds['ann']:
                    if in_scope(a)and(state=='all'or a['gesture']==state):mask|=(ds['t']>=a['start'])&(ds['t']<a['end'])
                nn+=int(mask.sum());nv+=int((mask&ds[g]['valid']).sum());npv+=int((mask&ds[g]['valid']&ds[g]['position_valid']).sum())
            e=np.array([r['error_deg']for r in fr]);s=dict(scope=scope,gesture=state,group=g,n_phases=len(rows),common_frames=len(fr),**errors(e),
                macro_mae_deg=float(np.mean([r['mae_deg']for r in rows])),own_valid_macro_mae_deg=float(np.mean([r['mae_deg']for r in ownrows])),
                eligible_frames=nn,orientation_valid=nv,pose_valid=npv)
            s['availability_scope']=scope;s['orientation_rate_pct']=100*s['orientation_valid']/nn;s['pose_rate_pct']=100*s['pose_valid']/nn
            for name in ['minus100ms','minus40ms','plus40ms','plus100ms','nominal_host_time']:s['macro_mae_'+name+'_deg']=float(np.mean([r['mae_'+name+'_deg']for r in rows]))
            summary.append(s)
    features=[]
    for d in datasets:
        for i,ti in enumerate(d['t']):
            a=next((a for a in d['ann']if a['start']<=ti<a['end']),{})
            r=dict(session=d['label'],t=ti,frame=d['mp'][i]['frame_number'],gesture=a.get('gesture'),cycle_id=a.get('cycle_id'),phase=a.get('phase'),
                imu_time_valid=bool(d['imu_good'][i]),raw_reported_right=bool((d['mp'][i].get('detection')or{}).get('detector_reported_is_right',False)))
            for name in ['hamer_bend','mp3d_bend','mp2d_bend']:
                for k,digit in enumerate(['thumb','index','middle','ring','little']):r[name+'_'+digit]=d[name][i,k]
                for k,digit in enumerate(['index','middle','ring','little']):r[name+'_base_'+digit+'_deg']=d[name+'_base'][i,k]
            for g in GROUPS:r[g+'_orientation_valid']=bool(d[g]['valid'][i]);r[g+'_position_valid']=bool(d[g]['position_valid'][i])
            for key,val in d['B']['rows'][i].get('registration',{}).items():
                if isinstance(val,(int,float)):r['B_'+key]=val
            features.append(r)
    manifest=dict(scope=__doc__,sessions=[dict(label=d['label'],metadata=d['meta'],imu_valid=int(d['imu_good'].sum()),common_valid=int((d['imu_good']&np.logical_and.reduce([d[g]['valid']for g in GROUPS])).sum()))for d in datasets],
        summary=summary,cycles=cycles,clock_diagnostic=clock,quiet_window_count=len(quiet)//6,
        exclusions='Source frames kept in availability. IMU-invalid and method-invalid frames excluded only from numerical error; all-six and own-valid results both exported. No outcome-based outlier removal.',
        reference='Common host timestamps and one declared offset per recording; no device-level synchronization or fixed IMU-to-palm extrinsics. No independent position or finger-angle truth.')
    dump(REPORT/'summary.json',manifest);dump(OUT/'gesture_traces.json',traces)
    for name,rows in [('按手势汇总指标',summary),('逐阶段六组指标',allphase),('各组自身有效帧指标',own),('完整手势周期指标',cycles),('有效率及失败分类',coverage),
        ('逐帧失败明细',failures),('去除HaMeR整体旋转后的建轴偏转',coupling),('手指弯曲及模型形变',shape),('腕环相对其他建轴的配对增益',paired),
        ('参考近静止窗口稳定性',quiet),('时钟对齐诊断',clock),('逐帧角度误差与三轴变化',frame_rows),('逐帧手指状态及配准质量',features),('单零位全录像指标',fixed)]:
        csvwrite(REPORT/(name+'.csv'),rows)
    print(json.dumps(clean(summary[:6]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
