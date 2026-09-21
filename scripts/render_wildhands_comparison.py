#!/usr/bin/env python3
"""Source-grounded plot and three replay excerpts; no pose smoothing or AI images.
Figure contract: test preservation of real rotation under a model replacement.
One operator, matched frames; descriptive curves, no significance claims.
"""
import argparse,csv,gzip,json,subprocess,sys
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perception_mediapipe.recorded_rgbd import RecordedRGBD


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--figure-tools',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.figure_tools));from audit_panel_alignment import require_matplotlib_panel_alignment
    wild=[json.loads(x) for x in (args.run/'predictions.jsonl').open()];hamer=json.load(gzip.open(ROOT/'.runtime/palm_ring_surface_20260920/paired_frames.json.gz','rt'));h={x['seq']:x for x in hamer};wild=[x for x in wild if x['valid']]
    ts=np.array([x['t'] for x in wild]);ct=json.load(open(ROOT/'.runtime/palm_ring_surface_20260920/metrics.json'))['c_video_s'];ci=np.argmin(abs(ts-ct));hc=R.from_quat(h[wild[ci]['seq']]['ring_q']);wc=R.from_quat(wild[ci]['ring_q'])
    imu=np.load(ROOT/'.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent/imu_signals.npz');si=Slerp(imu['t'],R.from_quat(imu['q']));ic=si(ct+.025)
    ah=np.rad2deg((R.from_quat([h[x['seq']]['ring_q'] for x in wild])*hc.inv()).magnitude());aw=np.rad2deg((R.from_quat([x['ring_q'] for x in wild])*wc.inv()).magnitude())
    ai=np.full(len(ts),np.nan);ix=np.clip(np.searchsorted(imu['t'],ts+.025),1,len(imu['t'])-1);ok=(ts+.025>=imu['t'][0])&(ts+.025<=imu['t'][-1])&imu['time_valid'][ix]&imu['time_valid'][ix-1];ai[ok]=np.rad2deg((ic.inv()*si(ts[ok]+.025)).magnitude())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    for name,lo,hi in [('yaw_amplitude',265,349),('translation_amplitude',54,110)]:
        fig,ax=plt.subplots(figsize=(7.2,3.5));fig.subplots_adjust(left=.10,right=.98,bottom=.19,top=.79);mask=(ts>=lo)&(ts<=hi)
        for y,label,c,style in [(ai,'IMU reference','#555555','-'),(ah,'HaMeR','#277DA8','-'),(aw,'WildHands','#B66645','--')]:ax.plot(ts[mask],y[mask],label=label,color=c,ls=style,lw=1.2)
        ax.set(xlabel='Source video time (s)',ylabel='Total rotation from the same C (deg)',xlim=(lo,hi),ylim=(0,None));ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=3)
        fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(out/(name+'.alignment.json')),strict=True)
        fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.svg'));fig.savefig(out/(name+'.png'),dpi=300);plt.close(fig)
        svg=out/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    with (out/'single_C_curve_data.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['video_time_s','hamer_angle_deg','wildhands_angle_deg','imu_reference_angle_deg','imu_time_valid'])
        writer.writerows(zip(ts,ah,aw,ai,ok))
    source=ROOT/'.runtime/hand_imu_observation_20260920T162851_bphL';reader=RecordedRGBD(source/'rgbd.db3')
    raw_path=args.run.parent/'comparison_mp4v.mp4';video=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*'mp4v'),8,(1280,610))
    windows=[(54,64,'Translation'),(217,227,'Pitch'),(290,300,'Yaw')];saved=set();manifest=[]
    for lo,hi,title in windows:
        for requested in np.arange(lo,hi,1/8):
            j=int(np.argmin(abs(ts-requested)));row=wild[j];old=h[row['seq']]
            for attempt in range(2):
                try:rgb,_,_=reader.read(row['frame_number'],row['stamp']);break
                except RuntimeError:
                    if attempt:raise
            canvas=np.full((610,1280,3),245,np.uint8)
            for col,label,points,angle in [(0,'HaMeR',np.array(old['palm_pixels']),ah[j]),(1,'WildHands',np.array(row['pixels'])[[0,5,9,13,17]],aw[j])]:
                bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);box=np.rint(row['bbox']).astype(int);cv2.rectangle(bgr,tuple(box[:2]),tuple(box[2:]),(50,210,60),1)
                for pt in points:
                    x,y=np.rint(pt).astype(int);cv2.circle(bgr,(x,y),3,(0,160,245),-1)
                canvas[:480,col*640:(col+1)*640]=bgr
                cv2.putText(canvas,f'{label} | {title} | source {row["t"]:.3f}s',(col*640+12,507),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1,cv2.LINE_AA)
                cv2.putText(canvas,f'Wrist-ring rotation from C: {angle:.1f} deg',(col*640+12,540),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1,cv2.LINE_AA)
            cv2.putText(canvas,f'IMU reference from C: {ai[j]:.1f} deg | same source RGB and recorded ROI',(12,575),cv2.FONT_HERSHEY_SIMPLEX,.6,(25,25,25),1,cv2.LINE_AA)
            cv2.putText(canvas,'Dots are projected model joints, not ground truth. Three excerpts; 8 fps display.',(12,600),cv2.FONT_HERSHEY_SIMPLEX,.55,(60,60,60),1,cv2.LINE_AA)
            video.write(canvas)
            if any(abs(row['t']-q)<.07 for q in [62,220,295]) and title not in saved:
                cv2.imwrite(str(out/(title.lower()+'_source_example.png')),canvas);saved.add(title)
            manifest.append(dict(display_clip=title,source_t=row['t'],seq=row['seq'],source_frame=row['frame_number']))
    video.release();reader.close()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(raw_path),'-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'hamer_wildhands_replay.mp4')],check=True)
    (out/'replay_manifest.json').write_text(json.dumps(manifest,indent=2));print('comparison figures and 30-second source replay ready')


if __name__=='__main__':main()
