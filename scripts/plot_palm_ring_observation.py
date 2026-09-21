#!/usr/bin/env python3
"""Evidence plots, source-derived diagnostic frames and a source-timed replay.

Contract: show paired-method similarity separately from arm tracking shortfall.
Single-panel quantitative figures, Python, editable PDF/SVG + PNG; descriptive
one-person repeated motions, no frame-level significance or invented error bars.
"""
import argparse,gzip,json,sys,subprocess
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--figure-tools',type=Path,required=True);args=ap.parse_args()
    sys.path.insert(0,str(args.figure_tools));from audit_panel_alignment import require_matplotlib_panel_alignment
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    def save(fig,name):
        fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(out/(name+'.alignment.json')),strict=True)
        fig.savefig(out/(name+'.pdf'))
        fig.savefig(out/(name+'.svg'))
        fig.savefig(out/(name+'.png'),dpi=300)
        plt.close(fig)
    segments=json.load(open(args.data/'segment_metrics.json'))
    fig,ax=plt.subplots(figsize=(7.2,3.5));fig.subplots_adjust(left=.10,right=.98,bottom=.19,top=.78)
    ids=[x['id'] for x in segments]
    ax.plot(ids,[x['palm']['angle_magnitude_difference']['mean'] for x in segments],color='#277DA8',label='Five-point palm',marker='o',ms=3,lw=1)
    ax.plot(ids,[x['ring']['angle_magnitude_difference']['mean'] for x in segments],color='#B66645',label='Wrist ring',marker='x',ms=4,lw=1,ls='--')
    ax.set(xlabel='Annotated motion segment (1–55)',ylabel='Angle-magnitude MAE vs IMU (deg)',xlim=(0,56),ylim=(0,None));ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=2)
    save(fig,'paired_segment_error')
    robot=json.load(gzip.open(args.source/'analysis_intent/robot_signals.json.gz','rt'))['rows']['trend'];tr=[x for x in robot if 292<=x['t']<=299];zero=R.from_quat(tr[0]['robot_zero_quaternion_xyzw'])
    fig,ax=plt.subplots(figsize=(7.2,3.5));fig.subplots_adjust(left=.10,right=.98,bottom=.19,top=.78)
    for field,label,col in [('target_quaternion_xyzw','Robot target','#277DA8'),('current_quaternion_xyzw','Robot measured','#B66645')]:
        rot=(zero.inv()*R.from_quat([x[field] for x in tr])).as_matrix();side=np.rad2deg(np.arctan2(-rot[:,1,2],rot[:,2,2]))
        ax.plot([x['t'] for x in tr],abs(side),label=label,color=col,lw=1.8)
    ax.axvline(295,color='0.7',lw=.7,ls=':');ax.set(xlabel='Video time (s); 295 s = 4:55',ylabel='Right-side turn magnitude (deg)',xlim=(292,299),ylim=(0,95));ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=2)
    save(fig,'yaw_target_actual')
    rows=json.load(gzip.open(args.data/'paired_frames.json.gz','rt'));frames=[x for x in rows if x.get('source_image')];meta=json.load(open(args.data/'replay_metadata.json'));intr=meta['intrinsics']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    selected=[];sheet=Image.new('RGB',(1200,760),'white')
    for k,target in enumerate([267.5,281,295,307.5,322,332]):
        row=min(frames,key=lambda r:abs(r['t']-target));im=Image.open(args.data/row['source_image']);box=row['bbox'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        # Same 240x300 source crop and isotropic scaling in every panel.
        crop=(int(cx-120),int(cy-150),int(cx+120),int(cy+150));im=im.crop(crop);im.thumbnail((360,300))
        x=(k%3)*400;y=(k//3)*380;sheet.paste(im,(x+(400-im.width)//2,y))
        draw=ImageDraw.Draw(sheet);draw.text((x+15,y+307),f'{row["t"]:.2f} s   palm width {row["palm_width_px"]:.1f} px',font=small,fill='black')
        draw.text((x+15,y+336),f'Skin depth {row["palm_position"][2]:.3f} m',font=small,fill='black')
        selected.append(dict(t=row['t'],frame=row['frame'],seq=row['seq'],source_image=row['source_image'],crop=list(crop),palm_width_px=row['palm_width_px']))
    sheet.save(out/'narrow_yaw_source.png');(out/'narrow_yaw_source_manifest.json').write_text(json.dumps(selected,indent=2))
    # Fixed-rate video selects nearest available exact-source frame and prints
    # its original time. It does not interpolate poses or smooth estimates.
    raw_video=args.data/'yaw_native_axes_mp4v.mp4'
    video=cv2.VideoWriter(str(raw_video),cv2.VideoWriter_fourcc(*'mp4v'),8,(1280,600))
    ft=np.array([x['t'] for x in frames]);cache={}
    for display_t in np.arange(265,349,1/8):
        row=frames[int(np.argmin(abs(ft-display_t)))];path=row['source_image']
        if path not in cache:cache={path:cv2.imread(str(args.data/path))}
        source=cache[path];canvas=np.full((600,1280,3),245,np.uint8)
        for col,name,title in [(0,'palm','FIVE-POINT PALM: native axes'),(1,'ring','WRIST RING: native axes')]:
            image=source.copy();pos=row.get(name+'_position')
            if pos is not None:
                mat=R.from_quat(row[name+'_q']).as_matrix();points=np.vstack([pos,np.array(pos)+mat.T*.08]);pixels=np.column_stack([intr['fx']*points[:,0]/points[:,2]+intr['ppx'],intr['fy']*points[:,1]/points[:,2]+intr['ppy']]).round().astype(int)
                for j,color in enumerate([(20,20,235),(30,160,30),(220,110,20)]):
                    cv2.arrowedLine(image,tuple(pixels[0]),tuple(pixels[j+1]),color,2,cv2.LINE_AA,tipLength=.15);cv2.putText(image,'XYZ'[j],tuple(pixels[j+1]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
            canvas[:480,col*640:(col+1)*640]=image
            cv2.putText(canvas,title,(col*640+12,510),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),1,cv2.LINE_AA)
            cv2.putText(canvas,f'Source {row["t"]:.3f}s | width {row["palm_width_px"]:.1f}px',(col*640+12,544),cv2.FONT_HERSHEY_SIMPLEX,.60,(30,30,30),1,cv2.LINE_AA)
        cv2.putText(canvas,'Native axes differ by definition; compare C-relative motion, not arrow equality.',(12,582),cv2.FONT_HERSHEY_SIMPLEX,.6,(40,40,40),1,cv2.LINE_AA)
        video.write(canvas)
    video.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(raw_video),'-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'yaw_palm_ring_replay.mp4')],check=True)
    print('plots, source frame sheet and 84-second yaw replay written')


if __name__=='__main__':main()
