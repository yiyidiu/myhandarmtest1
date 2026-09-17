"""Compare old/new requested finger targets using this session's raw landmarks.

This isolates mapping with ideal joint following, not a measured robot replay.
Invalid finger observations keep the last command, just as the ROS adapter does.
"""
import argparse,copy,csv,json,sys,time
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/handarm_moveit_demo/src')]
from perception_mediapipe.fingers import observe_fingers
from handarm_moveit_demo.finger_retargeting import ThreeFingerRetargeter
from handarm_moveit_demo.hamer_input_contract import HamerPacketContract


def replay(recording,output):
    rows=[json.loads(x)for x in open(recording/'perception/frames.jsonl')]
    cfg=yaml.safe_load(open(ROOT/'src/handarm_moveit_demo/config/shared_teleop.yaml'))['finger_retargeting']
    hand=yaml.safe_load(open(ROOT/'src/handarm_sim_demo/config/hand_commands.yaml'));names=hand['joint_names']
    reports={};all_results={};compute_ms=[]
    for mode in ['before','after']:
        ctrl=ThreeFingerRetargeter(hand['commands']['OPEN']['positions'],hand['commands']['CLOSE']['positions'],
            [hand['joint_limits'][n][0]for n in names],[hand['joint_limits'][n][1]for n in names],cfg['source_mixing_matrix'],cfg)
        current=np.array(hand['commands']['OPEN']['positions']);out=[];contract=HamerPacketContract('camera_color_optical_frame',.25,True,True)
        for row in rows:
            p=copy.deepcopy(row['packet'])
            if not p['control_enabled']:ctrl.block_active_reference();continue
            if not p['valid']:continue
            old=p['finger_observation']
            if mode=='after':
                start=time.perf_counter()
                f=observe_fingers(row['landmarks_world_m'],row['landmarks'],row['raw_pose']['confidence'])
                compute_ms.append((time.perf_counter()-start)*1000)
                # Keep the existing fingertip-clipping rule, too.
                if not old['valid']:f.update(valid=False,confidence=0.,invalid_reason=old['invalid_reason'])
                p['finger_observation']=f
            else:f=old
            checked=contract.validate(p)
            # A bad finger estimate must not invalidate a good arm pose.
            assert checked['observation_valid']
            if f['valid']:
                r=ctrl.update(p['stamp'],p['control_reference_token'],f['flexion'],f['confidence'],current)
                if r.command_target is not None:current=r.command_target
            out.append(dict(t=row['elapsed_s'],mean_command_deg=float(np.rad2deg(current[1:].mean())),
                finger_valid=f['valid'],finger_reason=f['invalid_reason'],arm_pose_valid=p['valid']))
        intervals=[]
        for lo,hi in [(177.3,192.8),(200.5,212.1),(226.8,234.8)]:
            a=[x['mean_command_deg']for x in out if lo<=x['t']<=hi]
            intervals.append(dict(start=lo,end=hi,min_deg=min(a),max_deg=max(a)))
        fists=[]
        for lo,hi in [(131.7,133.4),(215,216.6)]:
            a=[x['mean_command_deg']for x in out if lo<=x['t']<=hi]
            fists.append(dict(start=lo,end=hi,peak_command_deg=max(a)))
        reports[mode]=dict(open_translation_intervals=intervals,fist_intervals=fists,
            finger_hold_frames=sum(not x['finger_valid']for x in out),arm_pose_valid_frames=len(out))
        all_results[mode]=out
    output.mkdir(parents=True,exist_ok=True)
    reports['scope']=__doc__;reports['compute_ms_median']=float(np.median(compute_ms));reports['recording']=str(recording)
    json.dump(reports,open(output/'summary.json','w'),ensure_ascii=False,indent=2)
    json.dump(all_results,open(output/'target_traces.json','w'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(12,4.4))
    for mode,label,color in [('before','修正前：含腕点与指根夹角','#c8791d'),('after','修正后：手指内部弯曲＋图像一致性','#087c9d')]:
        a=all_results[mode];ax.plot([x['t']for x in a],[x['mean_command_deg']for x in a],label=label,color=color,lw=1.4)
    for lo,hi in [(177.3,192.8),(200.5,212.1),(226.8,234.8)]:ax.axvspan(lo,hi,color='#a9d8bb',alpha=.25)
    ax.set_xlabel('相机开始处理画面后的时间（s），不是录屏时间');ax.set_ylabel('三指主关节平均目标角度（°）')
    ax.set_title('同一手部记录：绿色区间二维指节持续伸直；两次测试握拳的响应保留')
    ax.axhline(5,color='#444',ls=':',lw=.8);ax.legend();ax.grid(alpha=.18);fig.tight_layout()
    fig.savefig(output/'开合修正对比.png',dpi=150);plt.close(fig)
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();replay(a.recording,a.output)
