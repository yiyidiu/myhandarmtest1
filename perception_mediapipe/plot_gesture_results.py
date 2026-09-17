"""Python scientific figures: gesture examples, paired errors, and decomposition.

Image plate: all 8 reviewed cycles, midpoint frames from each declared phase;
only crop/resize, no contrast edits. Quantitative figures retain all cycles and
all accepted quiet-reference windows. No error-based selection or smoothing.
"""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,'/home/dongtian/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
from perception_mediapipe.analyze_six_groups import load,angle
from scipy.spatial.transform import Rotation as R
OUT=ROOT/'.runtime/gesture_pose_20260917';RESULT=OUT/'results';REPORT=ROOT/'docs/gesture_pose_comparison_20260917';FIG=REPORT/'figures'
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9'];CC=dict(zip('ABCDEF',COLORS))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})


def read(name):return list(csv.DictReader(open(REPORT/(name+'.csv'),encoding='utf-8-sig')))
def num(r,k):return float(r[k])if r.get(k)not in [None,'','None']else np.nan

def trace_line(ax,x,y,**kwargs):
    xx=[];yy=[]
    for i,(t,v)in enumerate(zip(x,y)):
        if i and t-x[i-1]>.05:xx.append(np.nan);yy.append(np.nan)
        xx.append(t);yy.append(v)
    return ax.plot(xx,yy,**kwargs)

def save(fig,name):
    fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(FIG/(name+'.alignment.json')),strict=True)
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.png',dpi=300)
    p=FIG/(name+'.svg');p.write_text('\n'.join(x.rstrip()for x in p.read_text().splitlines())+'\n')
    plt.close(fig)


def examples():
    provenance=[];ann=json.load(open(OUT/'gesture_annotations.json'));cycles=[]
    for s,aa in ann.items():
        for cy in sorted({a['cycle_id']for a in aa if a['cycle_id']}):cycles.append((s,cy,[a for a in aa if a['cycle_id']==cy]))
    for page in range(2):
        fig,axes=plt.subplots(4,3,figsize=(7.2,8.2),layout='constrained')
        for row,(s,cy,aa)in enumerate(cycles[page*4:page*4+4]):
            rows=[json.loads(x)for x in open(RESULT/s/'mp.jsonl')];meta=json.load(open(RESULT/s/'metadata.json'))
            for col,phase in enumerate(['open_before','flexed_hold','open_after']):
                a=next(x for x in aa if x['phase']==phase);mid=(a['start']+a['end'])/2
                candidates=[r for r in rows if a['start']<=r['t']<a['end']];r=min(candidates,key=lambda z:abs(z['t']-mid))
                raw=np.load(Path(meta['frame_cache_directory'])/r['file'])['rgb'];h,w=raw.shape[:2];det=r.get('detection')
                box=[0,0,w,h]if det is None else list(np.round(det['bbox']).astype(int))
                x0,y0,x1,y1=box;x0=max(0,x0-12);y0=max(0,y0-12);x1=min(w,x1+12);y1=min(h,y1+12)
                im=Image.fromarray(raw[y0:y1,x0:x1]);im.thumbnail((280,220));canvas=Image.new('RGB',(300,230),'white');canvas.paste(im,((300-im.width)//2,(230-im.height)//2))
                ax=axes[row,col];ax.imshow(canvas);ax.set_aspect('auto');ax.set_axis_off()
                state={'open_before':'Open before','flexed_hold':({'half':'Partial flexion','closed':'Four-finger grasp','ambiguous':'Flexion uncertain / tilt'}.get(a['gesture'],'')),'open_after':'Open after'}[phase]
                ax.set_title(f'Cycle {cy} | {state}\n{s}, {r["t"]:.2f} s',pad=5)
                provenance.append(dict(cycle_id=cy,session=s,phase=phase,t=r['t'],frame=r['frame_number'],cache=str(Path(meta['frame_cache_directory'])/r['file']),crop_xyxy=[int(v)for v in [x0,y0,x1,y1]],brightness_contrast='unchanged',selection='phase midpoint, irrespective of model success'))
        save(fig,f'01_gesture_examples_{page+1}')
    json.dump(provenance,open(FIG/'image_provenance.json','w'),indent=2)


def error_map():
    rows=read('完整手势周期指标');z=np.array([[num(next(r for r in rows if int(r['cycle_id'])==cy and r['group']==g),'mae_deg')for g in 'ABCDEF']for cy in range(1,9)])
    fig,ax=plt.subplots(figsize=(7.2,4.6),layout='constrained');im=ax.imshow(z,cmap='cividis',aspect='auto',vmin=0,vmax=z.max())
    for i in range(8):
        for j in range(6):ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white'if z[i,j]<z.max()*.45 else'black')
    ax.set_xticks(range(6),['A\nGlobal','B\nWrist ring','C\nFive points','D\nFour regions','E\nMP 3D','F\nMP 2D + depth'])
    ax.set_yticks(range(8),[f'Cycle {i}'+('*'if i==1 else('†'if i==7 else''))for i in range(1,9)]);ax.set_title('Whole-cycle relative-angle discrepancy (MAE)',pad=12)
    fig.supxlabel('* IMU references disagree; † Finger flexion uncertain',fontsize=7)
    fig.colorbar(im,ax=ax,label='MAE (degrees)',shrink=.8);save(fig,'02_cycle_angle_error')


def decomposition():
    traces=json.load(open(OUT/'gesture_traces.json'));tr=traces['s1_cycle1'];tt=np.array(tr['t']);fingt=np.array(tr['finger_t']);mask=(fingt>=187)&(fingt<201)
    d=load('s1',RESULT);a=next(a for a in d['ann']if a['cycle_id']==1 and a['phase']=='closing');ok=(d['t']>=a['start'])&(d['t']<a['end'])&d['imu_good']&np.logical_and.reduce([d[g]['valid']for g in 'ABCDEF']);ids=np.flatnonzero(ok)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
    ax=axes[0,0];x=d['t'][ids]-d['t'][ids[0]];trace_line(ax,x,angle(d['imu'][ids]),color='black',lw=1.7,label='IMU')
    for g in 'ABCDEF':trace_line(ax,x,angle(R.from_matrix(d[g]['mat'][ids])),color=CC[g],lw=1,label=g,ls='--'if g in 'BCD'else'-')
    ax.set(xlabel='Time since first valid closing frame (s)',ylabel='Relative angle (degrees)',title='a  Quiet-reference finger closing')
    ax.legend(ncol=4,loc='upper left',bbox_to_anchor=(0,1.02),fontsize=6.5);ax.margins(y=.28);ax.set_ylim(bottom=0)
    ax=axes[0,1]
    for g in 'BCD':trace_line(ax,np.array(tr['root_t'])-187,tr['root_extra'][g],lw=1.2,color=CC[g],label=g)
    ax.axvspan(190.5-187,192-187,color='.9',zorder=-2);ax.set(xlabel='Time since cycle start (s)',ylabel='Extra axis turn (degrees)',title='b  Same model, global rotation removed');ax.legend(loc='upper right')
    ax=axes[1,0]
    for key,name,c in [('hamer_bend','HaMeR',CC['A']),('mp3d_bend','MP 3D',CC['E']),('mp2d_bend','MP 2D projection',CC['F'])]:ax.plot(fingt[mask]-187,np.array(tr[key],float)[mask],lw=1,color=c,label=name)
    ax.set(xlabel='Time since cycle start (s)',ylabel='Mean long-finger bend / pi',title='c  Visible flexion and predicted hand shape');ax.legend(loc='upper right',fontsize=6.5);ax.margins(y=.3);ax.set_ylim(bottom=0)
    ax=axes[1,1];rows=read('去除HaMeR整体旋转后的建轴偏转')
    for g in 'BCD':
        y=[max(num(r,'extra_axis_deg_max')for r in rows if r['group']==g and int(r['cycle_id'])==cy)for cy in range(1,9)]
        ax.plot(range(1,9),y,'o-',ms=3,lw=1,color=CC[g],label=g)
    ax.set(xlabel='Gesture cycle',ylabel='Peak extra axis turn (degrees)',title='d  Internal deformation across all cycles');ax.set_xticks(range(1,9));ax.legend(loc='upper center',ncol=3)
    save(fig,'03_pose_and_shape_decomposition')


def stability():
    rows=read('参考近静止窗口稳定性');fig,axes=plt.subplots(1,2,figsize=(7.2,3.25),layout='constrained')
    for gidx,g in enumerate('ABCDEF'):
        rr=[r for r in rows if r['group']==g];y=np.array([num(r,'orientation_rms_deg')for r in rr]);x=np.full(len(y),gidx)+np.linspace(-.16,.16,len(y))
        axes[0].scatter(x,y,s=7,color=CC[g],alpha=.6);axes[0].plot([gidx-.2,gidx+.2],[np.median(y)]*2,color='black',lw=1.4)
    axes[0].set_xticks(range(6),list('ABCDEF'));axes[0].set(ylabel='Orientation RMS around window mean (deg)',title='a  Every 1 s quiet-reference window')
    cov=read('有效率及失败分类');states=['open','half','closed','closing','opening','ambiguous'];colors=['#777777','#0072B2','#E69F00','#009E73','#D55E00','#CC79A7']
    for j,state in enumerate(states):
        y=[]
        for g in 'ABCDEF':
            rr=[r for r in cov if r['group']==g and r['gesture']==state];y.append(100*sum(int(r['pose_valid'])for r in rr)/sum(int(r['n'])for r in rr))
        axes[1].plot(range(6),y,'o-',label=state.capitalize(),lw=1,ms=3,color=colors[j])
    axes[1].set_xticks(range(6),list('ABCDEF'));axes[1].set(ylabel='Valid position + orientation (%)',title='b  All source frames count for availability')
    axes[1].legend(loc='center left',fontsize=6.5,bbox_to_anchor=(0,.36));axes[1].set_ylim(top=100.3)
    save(fig,'04_stability_and_availability')


if __name__=='__main__':
    FIG.mkdir(parents=True,exist_ok=True);examples();error_map();decomposition();stability()
