"""Engineering comparison figures, without an unmeasured absolute-pose claim."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.runtime/comparison'
REPORT=OUT/'report'
MODELS=['hamer','rgbd','world']
NAMES={'hamer':'HaMeR（无辅助）','rgbd':'MediaPipe + RGB-D','world':'MediaPipe 3D'}
COLORS={'hamer':'#8b6bb1','rgbd':'#d78418','world':'#087ca7','imu':'#262b30'}
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,'font.size':10,
    'savefig.facecolor':'white','figure.facecolor':'white'})


def group(a):
    k=a.get('kind','')
    if k.startswith('translation'):return '平移'
    known={'static':'静止','rotation_fingers_tilt':'主动俯仰','rotation_side_flip':'主动横滚','rotation_inplane':'主动偏航','mixed':'其他／组合'}
    if k in known:return known[k]
    s=a['intent']
    if '静止' in s or ('保持' in s and '标定' in s):return '静止'
    if '平移' in s:return '平移'
    if any(x in s for x in ['俯仰','上下翻转','四指']):return '主动俯仰'
    if any(x in s for x in ['横滚','左右翻转','小拇指']):return '主动横滚'
    if any(x in s for x in ['偏航','顺时针','逆时针']):return '主动偏航'
    if '平移' in s:return '平移'
    return '其他／组合'


def load(label):
    d=OUT/label;meta=json.load(open(d/'metadata.json'))
    mp=[json.loads(x) for x in open(d/'mediapipe.jsonl')]
    ha=[json.loads(x) for x in open(d/'hamer.jsonl')]
    assert len(mp)==len(ha)==meta['requested_frames']
    t=np.array([x['t']for x in mp]);assert np.max(np.abs(t-[x['t']for x in ha]))<1e-6
    n=len(t); poses={'hamer':ha}
    for key in ['rgbd','world']:
        poses[key]=[x.get('poses',{}).get(key,{'valid':False}) for x in mp]
    imu=np.load(d/'imu_signals.npz');offset=json.load(open(d/'time_alignment.json'))['used_offset_s']
    si=Slerp(imu['t'],R.from_quat(imu['q']));qi=si(t+offset)
    imu_good=np.interp(t+offset,imu['t'],imu['time_valid'].astype(float))>.99
    data=dict(label=label,t=t,mp=mp,ha=ha,poses=poses,imu=qi,imu_good=imu_good,
              ann=json.load(open(d/'annotations.json')),meta=meta,imu_interp=si,offset=offset)
    for key in MODELS:
        v=np.array([bool(x.get('valid')) for x in poses[key]])
        q=np.full((n,4),np.nan);pos=np.full((n,3),np.nan)
        if v.any():
            q[v]=R.from_matrix(np.array([x['rotation']for x in poses[key]if x.get('valid')]).reshape(-1,3,3)).as_quat()
            pos[v]=[x['position']for x in poses[key]if x.get('valid')]
        data[key]=dict(valid=v,q=q,pos=pos)
    return data


def savefig(fig,name):
    fig.savefig(REPORT/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(REPORT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    REPORT.mkdir(exist_ok=True);datasets=[load(x)for x in ['earlier','later']];stats=[];traces={};coverage=[]
    for d in datasets:
        for key in MODELS:
            coverage.append(dict(session=d['label'],model=key,valid=int(d[key]['valid'].sum()),total=len(d['t'])))
        for a in d['ann']:
            trim=min(.2,(a['end']-a['start'])*.1)
            eligible=(d['t']>=a['start']+trim)&(d['t']<=a['end']-trim)
            valid=eligible&d['imu_good']
            for key in MODELS:valid &= d[key]['valid']
            ids=np.flatnonzero(valid)
            if len(ids)<3:continue
            # Same frames and same first reference for all models and IMU.
            truth_q=d['imu'][ids];truth=np.rad2deg((truth_q[0].inv()*truth_q).magnitude())
            pi,pj=np.triu_indices(len(ids),1)
            pair_truth=np.rad2deg((truth_q[pi].inv()*truth_q[pj]).magnitude())
            trace=dict(t=d['t'][ids],imu=truth,intent=a['intent'])
            for key in MODELS:
                q=R.from_quat(d[key]['q'][ids]);turn=np.rad2deg((q[0].inv()*q).magnitude())
                pair=np.rad2deg((q[pi].inv()*q[pj]).magnitude());err=pair-pair_truth
                centre=q.mean();jitter=np.rad2deg((centre.inv()*q).magnitude())
                pos=d[key]['pos'][ids];posrad=np.linalg.norm(pos-np.mean(pos,axis=0),axis=1)*1000
                stats.append(dict(session=d['label'],id=a['id'],intent=a['intent'],category=group(a),
                    model=key,start=a['start'],end=a['end'],n=len(ids),eligible=int(eligible.sum()),
                    pair_angle_rmse_deg=float(np.sqrt(np.mean(err**2))),
                    onset_angle_rmse_deg=float(np.sqrt(np.mean((turn-truth)**2))),
                    imu_peak_deg=float(truth.max()),model_peak_deg=float(turn.max()),
                    amplitude_error_deg=float(turn.max()-truth.max()),
                    orientation_spread_rms_deg=float(np.sqrt(np.mean(jitter**2))),
                    position_spread_rms_mm=float(np.sqrt(np.mean(posrad**2)))))
                for dt in [-.04,.04]:
                    shifted=d['imu_interp'](d['t'][ids]+d['offset']+dt)
                    ref=np.rad2deg((shifted[pi].inv()*shifted[pj]).magnitude())
                    stats[-1]['rmse_time_minus40ms_deg' if dt<0 else 'rmse_time_plus40ms_deg']=float(np.sqrt(np.mean((pair-ref)**2)))
                trace[key]=turn
            traces[(d['label'],a['id'])]=trace
    with (REPORT/'逐动作对照.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    json.dump(stats,open(REPORT/'segment_metrics.json','w'),ensure_ascii=False,indent=2)
    summary={}
    for key in MODELS:
        s=[x for x in stats if x['model']==key];cov=[x for x in coverage if x['model']==key]
        summary[key]=dict(label=NAMES[key],n_segments=len(s),
            valid_frames=sum(x['valid']for x in cov),frames=sum(x['total']for x in cov),
            category_macro_rmse_deg={cat:float(np.mean([x['pair_angle_rmse_deg']for x in s if x['category']==cat]))
                for cat in sorted(set(x['category']for x in s))},
            macro_rmse_deg=float(np.mean([x['pair_angle_rmse_deg']for x in s])),
            macro_rmse_time_minus40ms_deg=float(np.mean([x['rmse_time_minus40ms_deg']for x in s])),
            macro_rmse_time_plus40ms_deg=float(np.mean([x['rmse_time_plus40ms_deg']for x in s])))
        for cat in ['静止']:
            rows=[x for x in s if x['category']==cat]
            summary[key]['static_position_spread_rms_mm']=float(np.median([x['position_spread_rms_mm']for x in rows])) if rows else None
            summary[key]['static_rotation_spread_rms_deg']=float(np.median([x['orientation_spread_rms_deg']for x in rows])) if rows else None
    json.dump(dict(models=summary,coverage=coverage,matched_segments=len(stats)//3,
        scope='Total rotation-distance agreement with IMU, not per-axis/absolute-pose accuracy. No model fitted to these labels.',
        selected_source_frames=sum(len(d['t'])for d in datasets)),open(REPORT/'summary.json','w'),ensure_ascii=False,indent=2)

    picks=[('earlier',2),('earlier',3),('later',4),('later',7),('later',2),('later',8),('later',9),('later',10)]
    # Later annotation IDs are read verbatim in the titles, avoiding inferred labels.
    fig,axs=plt.subplots(4,2,figsize=(13,15))
    for ax,pick in zip(axs.flat,picks):
        if pick not in traces:ax.set_visible(False);continue
        tr=traces[pick];t=tr['t'];x=t-t[0]
        for key in MODELS+['imu']:ax.plot(x,tr[key],label=NAMES.get(key,'IMU 参考'),color=COLORS[key],lw=2 if key=='imu' else 1.6)
        ax.set_title(f"{'录像一' if pick[0]=='earlier' else '录像二'} {int(t[0])//60}:{int(t[0])%60:02d}  {tr['intent']}",fontsize=11)
        ax.set_xlabel('相对本片段首个共同有效帧的时间（s）');ax.set_ylabel('相对首帧的总转角（°）');ax.grid(alpha=.2)
    fig.suptitle('同一画面、同一时间：真实转动是否被保留，多余转动是否减少',fontsize=16,y=.99)
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.015))
    fig.tight_layout(rect=(0,.055,1,.97));savefig(fig,'01_动作曲线')

    cats=[c for c in ['平移','主动俯仰','主动横滚','主动偏航','静止','其他／组合'] if any(x['category']==c for x in stats)]
    fig,axs=plt.subplots(1,2,figsize=(13,5.4));x=np.arange(len(cats));width=.25
    for i,key in enumerate(MODELS):
        vals=[summary[key]['category_macro_rmse_deg'].get(c,np.nan)for c in cats]
        bars=axs[0].bar(x+(i-1)*width,vals,width,label=NAMES[key],color=COLORS[key])
        for b in bars:axs[0].text(b.get_x()+b.get_width()/2,b.get_height()+.15,f'{b.get_height():.1f}',ha='center',fontsize=8)
    axs[0].set_xticks(x);axs[0].set_xticklabels(cats,rotation=15);axs[0].set_ylabel('跨时刻转角差 RMSE（°），越低越接近 IMU')
    axs[0].set_title('按动作分组，先逐片段计算再平均')
    for i,key in enumerate(MODELS):
        vals=[100*next(c['valid']/c['total']for c in coverage if c['model']==key and c['session']==label)for label in ['earlier','later']]
        b=axs[1].bar(np.arange(2)+(i-1)*width,vals,width,color=COLORS[key],label=NAMES[key])
        for item,v in zip(b,vals):axs[1].text(item.get_x()+item.get_width()/2,v+.6,f'{v:.1f}%',ha='center',fontsize=8)
    axs[1].set_xticks([0,1]);axs[1].set_xticklabels(['录像一','录像二']);axs[1].set_ylim(0,110);axs[1].set_ylabel('产生完整有效位姿的帧数 / 选取的全部帧（%）');axs[1].set_title('同时报告失败帧，不只挑识别成功的画面')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,0),fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1));savefig(fig,'02_误差与有效率')

    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,title,unit in [(axs[0],'static_position_spread_rms_mm','静止片段的位置散布','mm'),(axs[1],'static_rotation_spread_rms_deg','静止片段的姿态散布','°')]:
        vals=[summary[k][metric]for k in MODELS]
        bars=ax.bar([NAMES[k]for k in MODELS],vals,color=[COLORS[k]for k in MODELS])
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v,f'{v:.2f}',ha='center',va='bottom')
        ax.set_title(title+'（各片段中位数）');ax.set_ylabel(unit+'，包含人的真实微动');ax.tick_params(axis='x',labelsize=9)
    fig.suptitle('位置没有外部真值：这些是稳定性指标，不是绝对定位精度',fontsize=13)
    fig.tight_layout();savefig(fig,'03_静止稳定性')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
