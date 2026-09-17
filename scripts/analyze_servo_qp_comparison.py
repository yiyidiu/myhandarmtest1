#!/usr/bin/env python3
"""Descriptive controller comparison on the original replay timeline.

Errors are NEVER evaluated after shifting the actual trajectory in time.
Stalls include legitimate limit stops; they are not automatically called bugs.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

CASES = [
    ('teleop_feedback_20260908T182545_bZ7k', '9月8日：搬运、左右移动与返回'),
    ('hand_motion_observation_20260912T145617_iJYZ', '9月12日：平移与多轴转动'),
    ('teleop_feedback_20260913T205659_IkL0', '9月13日：近期连续跟随'),
]


def angle_degrees(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(a*b, axis=-1)), 0., 1.)))


def intervals(mask, t, minimum=.3):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(float(t[a]), float(t[b-1] + .02)) for a, b in zip(starts, ends)
            if t[b-1] + .02 - t[a] >= minimum]


def recorded_input_validity(source, t):
    with (source/'screen_frame_timeline.csv').open() as stream:
        first=next(csv.DictReader(stream))
    origin=int(first['estimated_frame_wall_ns'])/1e9-float(first['video_time_s'])
    stamps,values=[],[]
    with (source/'camera_hand_pose.csv').open() as stream:
        for row in csv.DictReader(stream):
            stamps.append(int(row['host_receive_wall_ns'])/1e9-origin)
            values.append(row['valid']=='1' and row['control_enabled']=='1')
    indices=np.clip(np.searchsorted(stamps,t,side='right')-1,0,len(stamps)-1)
    return np.array(values,dtype=bool)[indices]


def translation_lag(target, actual, valid):
    """Descriptive phase matching in dynamic 10 s windows, not transport latency."""
    desired = np.gradient(uniform_filter1d(target, 5, axis=0), .02, axis=0)
    measured = np.gradient(uniform_filter1d(actual, 5, axis=0), .02, axis=0)
    lags=[]
    for start in range(0, len(desired)-500, 500):
        if np.mean(valid[start:start+500]) < .95:
            continue
        a=desired[start:start+450]
        a=a-a.mean(axis=0)
        if np.sqrt(np.mean(a*a)) < .01:
            continue
        scores=[]
        for lag in range(51):
            b=measured[start+lag:start+lag+450]
            b=b-b.mean(axis=0)
            scores.append(float(np.sum(a*b)/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b))))
        best=int(np.argmax(scores))
        if scores[best] >= .65 and best < 50:
            lags.append(best*20.)
    return (float(np.median(lags)) if lags else None),len(lags)


def paired_audit(a,b):
    start=max(a['t'][0],b['t'][0]);end=min(a['t'][-1],b['t'][-1])
    t=np.arange(np.ceil(start/.02)*.02,end,.02)
    arrays=[]
    for s in [a,b]:
        v={k:np.column_stack([np.interp(t,s['t'],s[k][:,i]) for i in range(3)])
           for k in ['target','actual','relative_target']}
        v['valid']=np.interp(t,s['t'],s['valid'].astype(float))>.999
        arrays.append(v)
    valid=arrays[0]['valid'] & arrays[1]['valid']
    difference=np.linalg.norm(arrays[0]['relative_target']-arrays[1]['relative_target'],axis=1)*1000
    windows=[]
    for beginning in np.arange(np.ceil(start/10)*10,end-10,10):
        sl=(t>=beginning)&(t<beginning+10)
        if sl.sum()<499 or np.mean(valid[sl])<.95:continue
        measurements=[]
        for v in arrays:
            desired=np.gradient(uniform_filter1d(v['target'][sl],5,axis=0),.02,axis=0)
            measured=np.gradient(uniform_filter1d(v['actual'][sl],5,axis=0),.02,axis=0)
            n=len(desired)-50;first=desired[:n]-desired[:n].mean(axis=0)
            if np.sqrt(np.mean(first*first))<.01:break
            scores=[]
            for lag in range(51):
                last=measured[lag:lag+n];last=last-last.mean(axis=0)
                scores.append(float(np.sum(first*last)/max(1e-12,np.linalg.norm(first)*np.linalg.norm(last))))
            best=int(np.argmax(scores))
            if scores[best]<.65 or best==50:break
            measurements.append((best*20.,scores[best]))
        if len(measurements)==2:
            windows.append(dict(video_start=beginning,qp_lag_ms=measurements[0][0],
                                servo_lag_ms=measurements[1][0],correlations=[x[1] for x in measurements]))
    return dict(common_scored_end_s=end,
                relative_target_difference_rms_mm=float(np.sqrt(np.mean(difference[valid]**2))),
                relative_target_difference_p95_mm=float(np.percentile(difference[valid],95)),
                matched_windows=windows,
                qp_paired_phase_lag_ms=float(np.median([x['qp_lag_ms'] for x in windows])) if windows else None,
                servo_paired_phase_lag_ms=float(np.median([x['servo_lag_ms'] for x in windows])) if windows else None)


def summarize(path, scored_end=None, scored_start=None):
    with gzip.open(path, 'rt') as stream:
        run = json.load(stream)
    samples = [r for r in run['observations'] if r['t'] >= run['first_c_video_s'] and
               r.get('trend', {}).get('current_position') is not None and
               r['trend'].get('requested_target_position') is not None and
               r['trend'].get('current_quaternion_xyzw') is not None]
    if len(samples) < 5:
        raise RuntimeError('Too few actual robot observations: ' + str(path))
    t_raw = np.array([r['t'] for r in samples])
    beginning=max(t_raw[0],scored_start) if scored_start is not None else t_raw[0]
    ending=min(t_raw[-1],scored_end) if scored_end is not None else t_raw[-1]
    t = np.arange(np.ceil(beginning/.02)*.02, ending, .02)
    index = np.clip(np.searchsorted(t_raw, t, side='right')-1, 0, len(samples)-1)
    p = np.array([r['trend']['current_position'] for r in samples])
    actual = np.column_stack([np.interp(t, t_raw, p[:, k]) for k in range(3)])
    target = np.array([samples[i]['trend']['requested_target_position'] for i in index])
    aq = np.array([samples[i]['trend']['current_quaternion_xyzw'] for i in index])
    tq = np.array([samples[i]['trend']['requested_target_quaternion_xyzw'] for i in index])
    # Use the operator's recorded input validity, not solver validity: a
    # controller that refuses a valid command must not disappear from scores.
    valid = recorded_input_validity(Path(run['source']),t)
    position_error = np.linalg.norm(target-actual, axis=1)*1000
    rotation_error = angle_degrees(tq, aq)
    zq=np.array([samples[i]['trend']['robot_zero_quaternion_xyzw'] for i in index])
    normal=(Rotation.from_quat(zq).inv()*Rotation.from_quat(aq)).apply([0.,0.,1.])
    side=np.degrees(np.arctan2(-normal[:,1],normal[:,2]))
    side_defined=np.sum(normal[:,1:]**2,axis=1)>1e-6
    lag_ms,lag_windows=translation_lag(target,actual,valid)
    # Smooth only the numerical velocity estimate, never the error curves.
    velocity = np.linalg.norm(np.gradient(uniform_filter1d(actual, 5, axis=0), .02, axis=0), axis=1)
    angular_velocity = np.r_[0., angle_degrees(aq[1:], aq[:-1])/.02]
    angular_velocity = uniform_filter1d(angular_velocity, 5)
    pending = (position_error > 15.) | (rotation_error > 3.)
    stalled = intervals(valid & pending & (velocity < .005) & (angular_velocity < 2.), t)
    hold = np.array([samples[i].get('egm', {}).get('hard_safety_hold', False) for i in index])
    faults = run.get('faults', [])
    planned_end = run.get('planned_end_video_s', run['last_input_video_s'])
    metrics = dict(source=Path(run['source']).name, solver=run['solver'],
                   complete=not faults and run['last_input_video_s'] >= planned_end-.1,
                   first_c_video_s=run['first_c_video_s'], last_input_video_s=run['last_input_video_s'],
                   scored_end_video_s=float(t[-1]),
                   planned_end_video_s=planned_end, valid_control_s=float(valid.sum()*.02),
                   position_rmse_mm=float(np.sqrt(np.mean(position_error[valid]**2))),
                   position_p95_mm=float(np.percentile(position_error[valid],95)),
                   orientation_rmse_deg=float(np.sqrt(np.mean(rotation_error[valid]**2))),
                   orientation_p95_deg=float(np.percentile(rotation_error[valid],95)),
                   stall_total_s=float(sum(b-a for a,b in stalled)),
                   stall_count=len(stalled), longest_stall_s=max([b-a for a,b in stalled] or [0.]),
                   safety_hold_s=float(np.sum(hold & valid)*.02),
                   fault_count=len(faults), first_fault=faults[0] if faults else None,
                   replay_lateness_p95_ms=run['replay_lateness_p95_ms'],
                   simulation_to_wall_ratio=run['simulation_to_wall_ratio'],
                   translation_phase_lag_ms=lag_ms,matched_dynamic_windows=lag_windows,
                   maximum_absolute_side_angle_deg=float(np.max(np.abs(side[valid & side_defined]))),
                   initial_q=run['initial_q'], stall_intervals_video_s=stalled)
    series = dict(t=t, target=target, actual=actual, position_error=position_error,
                  rotation_error=rotation_error, valid=valid, velocity=velocity,
                  relative_target=target-np.array([samples[i]['trend']['robot_zero_position'] for i in index]))
    return metrics, series


def main(args):
    root = args.directory
    out = root/'results'
    out.mkdir(exist_ok=True)
    metrics, curves = [], {}
    for name, label in CASES:
        common_start,common_end = None,None
        available = [root/(name+'_'+s+'.json.gz') for s in ['qp','servo']]
        if all(p.exists() for p in available):
            limits = []
            for p in available:
                with gzip.open(p,'rt') as stream:
                    run = json.load(stream)
                usable=[r for r in run['observations'] if r['t']>=run['first_c_video_s'] and
                        r.get('trend',{}).get('current_position') is not None and
                        r['trend'].get('requested_target_position') is not None and
                        r['trend'].get('current_quaternion_xyzw') is not None]
                limits.append((usable[0]['t'],usable[-1]['t']))
            common_start=max(x[0] for x in limits)
            common_end=min(x[1] for x in limits)
        for solver in ['qp', 'servo']:
            path = root/(name+'_'+solver+'.json.gz')
            if not path.exists():
                continue
            row, _ = summarize(path, common_end, common_start)
            _, series = summarize(path)
            metrics.append(row); curves[(name,solver)] = series
            with (out/(name+'_'+solver+'_series.csv')).open('w') as stream:
                writer=csv.writer(stream)
                writer.writerow(['video_s','position_error_mm','orientation_error_deg','valid',
                                 'requested_x_m','requested_y_m','requested_z_m','actual_x_m','actual_y_m','actual_z_m'])
                for i in range(len(series['t'])):
                    writer.writerow([series['t'][i],series['position_error'][i],series['rotation_error'][i],
                                     int(series['valid'][i]),*series['target'][i],*series['actual'][i]])
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    audits={name:paired_audit(curves[(name,'qp')],curves[(name,'servo')])
            for name,_ in CASES if (name,'qp') in curves and (name,'servo') in curves}
    (out/'paired_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
    fields=['source','solver','complete','scored_end_video_s','valid_control_s','position_rmse_mm','position_p95_mm',
            'orientation_rmse_deg','orientation_p95_deg','stall_total_s','stall_count','longest_stall_s',
            'safety_hold_s','fault_count','translation_phase_lag_ms','matched_dynamic_windows',
            'maximum_absolute_side_angle_deg','replay_lateness_p95_ms','simulation_to_wall_ratio']
    with (out/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(metrics)
    for row in metrics:
        print(json.dumps({k:row[k] for k in fields},ensure_ascii=False))
    if not args.plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import FuncFormatter
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    fig, axes=plt.subplots(3,2,figsize=(13,10),sharex='row')
    colors={'qp':'#2B6F9C','servo':'#C97833'}
    for row,(name,label) in enumerate(CASES):
        for solver in ['qp','servo']:
            if (name,solver) not in curves:
                continue
            s=curves[(name,solver)]
            for col,key in enumerate(['position_error','rotation_error']):
                axes[row,col].plot(s['t'],np.where(s['valid'],s[key],np.nan),
                                   color=colors[solver],lw=1.2,label='QP' if solver=='qp' else 'MoveIt Servo')
        for col in range(2):
            ax=axes[row,col]
            for m in metrics:
                if m['source']==name and m['first_fault']:
                    fault_t=m['first_fault']['t']
                    ax.axvline(fault_t,color='#9B3039',ls=':',lw=1)
                    ax.text(fault_t+1,.97,'Servo 自碰撞锁停',color='#9B3039',fontsize=9,
                            transform=ax.get_xaxis_transform(),ha='left',va='top')
            ax.set_title(label + (' · 位置' if col==0 else ' · 姿态'),loc='left',fontsize=11)
            ax.set_ylabel('位置误差（mm）' if col==0 else '姿态误差（°）')
            ax.set_xlabel('原录像时间（分:秒）')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{int(x)//60}:{int(x)%60:02d}'))
            ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18)
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.95),frameon=False)
    fig.suptitle('同一段手部输入：目标位姿与机械臂实际位姿的误差',fontsize=16,y=.995)
    fig.text(.5,.02,'原时间轴计算；未移动曲线抵消延迟。空白表示输入无效或该回放已中止。\n'
             '相同机器人、初始姿态、映射与执行链；无环境坐标输入，不重算视觉，不验收抓取。',
             ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.91),h_pad=2)
    fig.savefig(out/'tracking_errors.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--plot',action='store_true')
    main(parser.parse_args())
