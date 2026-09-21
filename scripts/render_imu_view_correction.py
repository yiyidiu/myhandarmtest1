#!/usr/bin/env python3
"""Raw source-time curves for IMU-referenced correction and WildHands diagnosis.
One session, held-out motion intervals, no pose smoothing or significance test.
Vector PDF/SVG and 300 dpi PNG previews, 7.2 inch figure width.
"""
import argparse,csv,json,shutil,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    p=argparse.ArgumentParser();p.add_argument('--correction',type=Path,required=True);p.add_argument('--wild',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--figure-tools',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.figure_tools));from audit_panel_alignment import require_matplotlib_panel_alignment
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font,'DejaVu Sans'],'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    data=list(csv.DictReader((a.correction/'frame_comparison.csv').open()));ts=np.array([float(x['t'])for x in data])
    wild=json.load(open(a.wild/'predictions.json'));wt=np.array([x['t']for x in wild]);ci=next(j for j,x in enumerate(wild)if x['seq']==298)
    wa={}
    for name in ['official','direct_original']:
        q=R.from_quat([x['results'][name]['q']for x in wild]);wa[name]=np.rad2deg((q*q[ci].inv()).magnitude())
    for filename,lo,hi,title in [('translation_correction',54,110,'左右平移：前半段标定，后半段未参与拟合'),('yaw_comparison',265,273,'同一偏航动作：校正原 HaMeR 与替换成 WildHands')]:
        fig,ax=plt.subplots(figsize=(7.2,3.9));fig.subplots_adjust(left=.11,right=.98,bottom=.18,top=.69 if filename=='yaw_comparison' else .76)
        mask=(ts>=lo)&(ts<=hi);t=ts[mask]
        for field,label,color in [('imu_angle_deg','IMU 基准','#444444'),('raw_angle_deg','原 HaMeR','#BC5C38'),('horizontal_calibrated_angle_deg','HaMeR 水平校正','#277DA8')]:
            val=np.array([float(x[field])for x in data])[mask];val[np.r_[False,np.diff(t)>.18]]=np.nan
            ax.plot(t,val,label=label,color=color,lw=1.3)
        if filename=='yaw_comparison':
            for name,label,color,ls in [('official','WildHands 原裁剪','#558C72','--'),('direct_original','WildHands 原图直裁','#7977A8',':')]:
                mask=(wt>=lo)&(wt<=hi);ax.plot(wt[mask],wa[name][mask],label=label,color=color,lw=1.3,ls=ls)
        else:
            ax.axvline(83,color='#9199A1',ls=':',lw=1)
        ax.set(xlabel='原录像时间（秒）',ylabel='相对同一 C 零位的总转角（度）',xlim=(lo,hi),ylim=(0,105 if filename=='yaw_comparison' else 35))
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2 if filename=='yaw_comparison' else 3)
        fig.text(.11,.96,title,va='top',fontsize=10)
        fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(a.out/(filename+'.alignment.json')),strict=True)
        fig.savefig(a.out/(filename+'.pdf'));fig.savefig(a.out/(filename+'.svg'));fig.savefig(a.out/(filename+'.png'),dpi=300);plt.close(fig)
        f=a.out/(filename+'.svg');f.write_text('\n'.join(x.rstrip()for x in f.read_text().splitlines())+'\n')
    for name in ['metrics.json','segment_metrics.json','frame_comparison.csv','corrected_native_poses.npz']:
        shutil.copyfile(a.correction/name,a.out/('correction_'+name))
    for name in ['metrics.json','predictions.json','crop_geometry_check.json']:
        shutil.copyfile(a.wild/name,a.out/('wildhands_'+name))
    shutil.copytree(a.wild/'inputs',a.out/'wildhands_inputs',dirs_exist_ok=True)
    print('Source-time correction and failure-diagnosis figures exported.')


if __name__=='__main__':main()
