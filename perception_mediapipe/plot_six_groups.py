"""Python figures from measured six-group results; no synthetic observations.

Figure contract: compare paired angle error before efficiency, expose movement
attenuation and failures, and show the fixed mesh regions. Experimental units
are annotated episodes within two recordings, not independent participants.
PDF/SVG are editable; PNG is the preview. Font floor is 7 pt.
"""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'docs/six_group_fixed_left_20260917'
OUT=ROOT/'.runtime/six_group_fixed_left_20260917'
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','Arial','DejaVu Sans'],'axes.unicode_minus':False,
    'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
GROUPS=list('ABCDEF')
COLORS=dict(zip(GROUPS,['#424954','#20639B','#7295BF','#865A9B','#2A8B7C','#C18038']))
STYLES=dict(zip(GROUPS,[':','-','--','-.','-','-']))
NAMES={'A':'A 原始朝向','B':'B 腕环','C':'C 五点','D':'D 四区','E':'E MP三维','F':'F MP二维+深度'}


def save(fig,name):
    fig.text(.985,.992,'固定左手 · 无时间滤波',ha='right',va='top',fontsize=7,color='#555555')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(REPORT/(name+'.alignment.json')),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(REPORT/(name+'.pdf'),facecolor='white')
    fig.savefig(REPORT/(name+'.svg'),facecolor='white')
    svg=REPORT/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')
    fig.savefig(REPORT/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)


def layout(rows=1,cols=2,height=3.5):
    fig,axs=plt.subplots(rows,cols,figsize=(7.2,height),squeeze=False)
    fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.90,wspace=.37,hspace=.65)
    for ax in axs.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    return fig,axs


def legend(fig):
    handles=[Line2D([0],[0],color=COLORS[g],ls=STYLES[g],label=NAMES[g],lw=1.5)for g in GROUPS]
    handles.append(Line2D([0],[0],color='black',label='IMU参考',lw=1.8))
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.53,.015))


def gap_plot(ax,x,y,**kwargs):
    x=np.asarray(x);y=np.asarray(y);breaks=np.flatnonzero(np.diff(x)>.20)+1
    ax.plot(np.insert(x,breaks,np.nan),np.insert(y,breaks,np.nan),**kwargs)


def main():
    s=json.load(open(REPORT/'summary.json'));rows=json.load(open(REPORT/'segment_metrics.json'))
    timing=json.load(open(REPORT/'processing_timing.json'));tr=json.load(open(OUT/'traces.json'))
    fig,axs=layout();a,b=axs[0]
    a.bar(GROUPS,[s['models'][g]['macro_mae_deg']for g in GROUPS],color=[COLORS[g]for g in GROUPS],alpha=.8)
    for sess,marker in [('earlier','o'),('later','s')]:
        vals=[np.mean([r['mae_deg']for r in rows if r['group']==g and r['session']==sess])for g in GROUPS]
        a.plot(GROUPS,vals,marker=marker,color='#30343B',lw=.6,ms=3,label='录像一'if sess=='earlier'else'录像二')
    a.set_ylabel('相对转角 MAE（°）');a.set_title('a  六组整体及逐录像误差');a.legend(frameon=False,loc='upper left')
    lookup={(r['session'],r['id'],r['group']):r for r in rows}
    for k,g in enumerate('ACD'):
        delta=[r['mae_deg']-lookup[r['session'],r['id'],g]['mae_deg']for r in rows if r['group']=='B']
        # Horizontal displacement separates measured dots, never changes data.
        offset=np.arange(len(delta))/max(len(delta)-1,1)*.20-.10
        b.scatter(k+offset,delta,s=7,color=COLORS[g],alpha=.7)
        b.scatter([k],[np.mean(delta)],marker='D',s=24,c='black',zorder=3)
    b.axhline(0,color='gray',lw=.8);b.set_xticks(range(3),['B − A','B − C','B − D'])
    b.set_ylabel('逐片段 MAE 差（°），负值有利于腕环');b.set_title('b  腕环增益的成对比较')
    fig.text(.5,.04,'左：柱为79片段等权平均；点线为两段录像。右：每点一片段，黑菱形为平均差。',ha='center',fontsize=7)
    save(fig,'01_整体误差与腕环增益')

    cats=['平移','俯仰','横滚','偏航','静止','组合动作','丢失片段']
    fig,axs=layout(1,1,3.4);ax=axs[0,0]
    vals=np.array([[np.mean([r['mae_deg']for r in rows if r['group']==g and r['category']==c])for c in cats]for g in GROUPS])
    im=ax.imshow(vals,cmap='Blues',vmin=0,vmax=max(12,float(vals.max())),aspect='auto')
    ax.set_xticks(range(len(cats)),cats);ax.set_yticks(range(6),[NAMES[g]for g in GROUPS]);ax.grid(False)
    for i in range(6):
        for j in range(len(cats)):ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',color='white'if vals[i,j]>8 else 'black',fontsize=8)
    ax.set_title('动作类别会改变方法的表现：逐片段 MAE 平均值（°）')
    fig.subplots_adjust(left=.23,bottom=.20);fig.text(.56,.04,'颜色只编码误差；动作片段数量不等。丢失片段指原录像的人工标注。',ha='center',fontsize=7)
    save(fig,'02_按动作类别的误差')

    picks=['later_4','later_8','later_9','later_10','earlier_1','later_13']
    titles=['录像二：向左平移','录像二：主动俯仰','录像二：主动横滚','录像二：偏航','录像一：静止','录像二：原标注裁剪丢失']
    fig,axs=layout(3,2,7.3)
    for ax,key,title in zip(axs.flat,picks,titles):
        t=tr[key];x=np.array(t['t'])-t['t'][0]
        for g in GROUPS:gap_plot(ax,x,t['models'][g]['angle'],color=COLORS[g],ls=STYLES[g],lw=1.1,alpha=.9)
        gap_plot(ax,x,t['imu'],c='black',lw=1.5);ax.set_title(title);ax.set_xlabel('相对片段首帧时间（s）');ax.set_ylabel('相对转角（°）')
    fig.subplots_adjust(bottom=.20,top=.94,hspace=.7);legend(fig)
    fig.text(.5,.105,'相邻共同有效观测间隔超过0.20 s时断线；没有补造中间观测。',ha='center',fontsize=7)
    save(fig,'03_动作跟随与误转实例')

    fig,axs=layout(1,3,3.6)
    for ax,key,title,target in zip(axs.flat,['later_8','later_9','later_10'],['俯仰','横滚','偏航'],[0,1,2]):
        vals=[]
        for g in GROUPS:
            r=next(r for r in rows if r['session']=='later'and r['id']==int(key.split('_')[1])and r['group']==g)
            vals.append(r['off_intent_axis_rms_deg'])
        ax.bar(GROUPS,vals,color=[COLORS[g]for g in GROUPS]);ax.set_title(title);ax.set_ylabel('非目标轴活动 RMS（°）')
    fig.subplots_adjust(left=.09,wspace=.55,bottom=.25)
    fig.text(.5,.06,'按解剖轴分解的旋转向量；包含人手真实伴随转动。未标定IMU外参，不能称为串轴真值误差。',ha='center',fontsize=7)
    save(fig,'04_非目标轴活动')

    fig,axs=layout(1,3,3.6)
    for ax,metric,title,unit in [(axs[0,0],'static_rotation_rms_median_deg','静止姿态散布','°'),
                               (axs[0,1],'static_position_rms_median_mm','静止位置散布','mm')]:
        ax.bar(GROUPS,[s['models'][g][metric]for g in GROUPS],color=[COLORS[g]for g in GROUPS]);ax.set_ylabel(unit);ax.set_title(title)
    ax=axs[0,2];x=np.arange(6)
    ax.bar(x-.16,[100*s['models'][g]['valid_orientation']/s['models'][g]['frames']for g in GROUPS],width=.30,color='#4682AA',label='姿态')
    ax.bar(x+.16,[100*s['models'][g]['valid_pose']/s['models'][g]['frames']for g in GROUPS],width=.30,color='#BDC9D1',label='完整位姿')
    ax.set_xticks(x,GROUPS);ax.set_ylim(98.8,100.1);ax.set_ylabel('有效率（%，截断纵轴）');ax.set_title('缓存采样时刻有效率');ax.legend(frameon=False,loc='lower left')
    fig.subplots_adjust(wspace=.60,bottom=.25)
    fig.text(.5,.06,'散布为3个静止片段RMS的中位数，包含真实微动；位置无外部真值。有效率分母为4,053。',ha='center',fontsize=7)
    save(fig,'05_稳定性与有效率')

    fig,axs=layout(1,1,3.1);ax=axs[0,0]
    for i,g in enumerate(GROUPS):
        m=timing['models'][g];ax.scatter([i]*3,m['repeat_median_ms'],color=COLORS[g],s=22)
        ax.plot([i,i],[m['median_ms'],m['p95_ms']],color=COLORS[g],lw=2)
        ax.text(i,m['p95_ms']+1.7,f"{m['median_ms']:.1f}",ha='center',fontsize=8)
    ax.set_xticks(range(6),GROUPS);ax.set_ylim(0,62);ax.set_ylabel('处理耗时（ms）');ax.set_title('每条路线重新识别同一帧：96帧 × 3轮')
    fig.text(.5,.04,'点为各轮中位数；线从总体中位数至P95；数字为总体中位数。排除相机、显示和机械臂延迟。',ha='center',fontsize=7)
    save(fig,'06_实际处理耗时')

    geom=json.load(open(OUT/'geometry_definition.json'))['False']
    v=np.array(geom['neutral_vertices']);j=np.array(geom['neutral_joints']);basis=np.array(geom['neutral_common_frame'])
    xy=((v-j[0])@basis)[:,:2]*1000
    fig,axs=layout(1,1,4.8);ax=axs[0,0]
    faces=np.array(geom['faces']);edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(xy[edges],colors='#BBC1C6',linewidths=.25,alpha=.7))
    labels={'near':'近腕','distal':'远端掌部','thumb':'拇指侧掌部','little':'小指侧掌部'}
    for k,c in zip(geom['regions'],['#20639B','#7295BF','#865A9B','#C18038']):
        ids=geom['regions'][k];ax.scatter(xy[ids,0],xy[ids,1],s=18,color=c,label=labels[k]+'（12点）')
    palm=((j[[0,5,9,13,17]]-j[0])@basis)[:,:2]*1000
    ax.scatter(palm[:,0],palm[:,1],marker='x',s=22,color='black',label='腕点与四指根')
    ax.autoscale();ax.set_aspect('equal');ax.set_xlabel('横向：小指侧 → 食指侧（mm）');ax.set_ylabel('纵向：腕部 → 指根（mm）')
    ax.set_title('D组在中性模板上固定区域');ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1.03,1.),fontsize=7)
    fig.subplots_adjust(bottom=.16,right=.72);save(fig,'07_固定网格区域')

    fixed=json.load(open(OUT/'fixed_zero_traces.json'));fig,axs=layout(2,1,6.5)
    for ax,label,title in zip(axs.flat,['earlier','later'],['录像一：零位固定在190.01 s','录像二：零位固定在65.30 s']):
        d=fixed[label]
        for g in GROUPS:gap_plot(ax,d['t'],d['models'][g],color=COLORS[g],ls=STYLES[g],lw=.9)
        gap_plot(ax,d['t'],d['imu'],color='black',lw=1.5)
        gap_plot(ax,d['t'],d['gyro'],color='#B24854',ls='--',lw=1.2)
        ax.set_ylabel('相对固定零位转角（°）');ax.set_xlabel('录像时间（s）');ax.set_title(title)
    fig.subplots_adjust(bottom=.27,top=.94,hspace=.5)
    handles=[Line2D([0],[0],color=COLORS[g],ls=STYLES[g],label=NAMES[g],lw=1.3)for g in GROUPS]
    handles += [Line2D([0],[0],c='black',label='IMU融合角'),Line2D([0],[0],c='#B24854',ls='--',label='陀螺积分')]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.53,.015))
    fig.text(.5,.13,'长时间共同零位下，两种IMU参考也存在差异；不能把全部偏差归因于视觉模型。',ha='center',fontsize=7)
    save(fig,'08_整段录像固定零位')


if __name__=='__main__':main()
