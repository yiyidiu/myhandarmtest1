"""Inspect original RGB-D gestures, then prepare exact-frame six-group input.

Survey features only locate candidate gestures. Gesture labels require visual
inspection; no pose-error outcome is used to choose the gesture windows.
"""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'.runtime/gesture_pose_20260917'
SOURCES={
    's1':Path('/home/dongtian/teleop_recordings/20260914_dac3518/teleop_feedback_20260915T225301_14943'),
    's2':Path('/home/dongtian/teleop_recordings/20260914_dac3518/teleop_feedback_20260915T225953_19'),
}


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def source_info(path):
    session=json.load(open(path/'session.json'));origin=session['created_host_wall_ns']/1e9
    with open(path/'camera_hand_pose.csv')as f:
        rows=list(csv.DictReader(f));enabled=[r for r in rows if r.get('control_enabled')=='1']
    first_enabled=float(enabled[0]['source_camera_time_s'])-origin
    summary=json.load(open(next(path.glob('perception/*/summary.json'))))
    return dict(source=str(path),origin=origin,first_enabled=first_enabled,
                betas=summary['betas_calibration']['betas_user'])


def survey():
    from PIL import Image,ImageDraw
    from perception_mediapipe.pose import MediaPipePose
    from perception_mediapipe.fingers import _digit_shape
    from perception_mediapipe.recorded_rgbd import RecordedRGBD
    model=MediaPipePose()
    for label,path in SOURCES.items():
        directory=OUT/'survey'/label;directory.mkdir(parents=True,exist_ok=True)
        meta=source_info(path);reader=RecordedRGBD(path/'rgbd.db3');last=-np.inf;rows=[]
        json.dump(meta,open(directory/'source.json','w'),indent=2)
        for c in reader.meta['color']:
            t=c['time']-meta['origin']
            if t<meta['first_enabled']-4 or t-last<.48:continue
            last=t
            try:
                try:rgb,depth,diag=reader.read(c['frame'],c['time'])
                except RuntimeError:rgb,depth,diag=reader.read(c['frame'],c['time'])
                d,dt=model.detect(rgb)
                row=dict(t=t,frame_number=c['frame'],detected=d is not None)
                if d is not None:
                    b2,e2,l2=_digit_shape(d['xy']);b3,e3,l3=_digit_shape(d['world'])
                    row.update(bend2d=b2,bend3d=b3,extension2d=e2,bbox=d['bbox'],
                        xy=d['xy'],world=d['world'],palm_inside=d['visibility']['palm_inside_image'],
                        is_right=d['is_right'],raw_is_right=d['detector_reported_is_right'])
                    x0,y0,x1,y1=np.asarray(d['bbox']).astype(int);x0=max(0,x0-15);y0=max(0,y0-15);x1=min(640,x1+15);y1=min(480,y1+15)
                    crop=rgb[y0:y1,x0:x1]
                else:crop=rgb
                filename=f'{len(rows):04d}.jpg'
                im=Image.fromarray(crop);im.thumbnail((250,240));canvas=Image.new('RGB',(260,265),'white')
                canvas.paste(im,((260-im.width)//2,5));draw=ImageDraw.Draw(canvas)
                draw.text((6,245),f'{label} {t:.2f}s f{c["frame"]}',fill='black')
                canvas.save(directory/filename,quality=90)
                row['preview']=filename;rows.append(row)
                if len(rows)%50==0:print(label,'survey',len(rows),round(t,1),flush=True)
            except ValueError as exc:rows.append(dict(t=t,frame_number=c['frame'],error=str(exc)))
        reader.close();json.dump(rows,open(directory/'survey.json','w'),default=plain)
        # Complete contact sheets at 2 s spacing; dense previews remain available.
        selected=rows[::4]
        for page,start in enumerate(range(0,len(selected),30)):
            canvas=Image.new('RGB',(6*260,5*265),'white')
            for j,r in enumerate(selected[start:start+30]):
                if r.get('preview'):canvas.paste(Image.open(directory/r['preview']),((j%6)*260,(j//6)*265))
            canvas.save(directory/f'contact_{page:02d}.png')
        print(label,'survey done',len(rows),flush=True)
    model.close()


def annotations():
    """Image-reviewed intervals, fixed before six-method outcomes are computed.

    'closed' means the four long fingers visibly flexed into a grasp, without
    requiring the thumb to cover the knuckles. 'half' is a claw/partial grasp.
    Boundaries are approximate to the 0.5 s survey, not precise onset truth.
    """
    cycles={
        's1':[
            (1,187.,201.,'正面半握',[190.5,192.,196.2,198.2],'half'),
            (2,214.,230.,'侧向半握',[218.4,220.5,226.5,228.3],'half'),
            (3,299.,321.,'横向握拳',[304.6,306.4,316.5,319.2],'closed')],
        's2':[
            (4,107.,150.,'横向握拳及保持',[111.2,114.2,140.8,147.8],'closed'),
            (5,154.,173.,'正面半握',[159.3,160.8,167.3,169.8],'half'),
            (6,184.,205.,'侧向半握',[187.7,190.1,198.4,200.3],'half'),
            (7,242.5,250.5,'快速俯仰与手势变化（屈指不确定）',[245.1,246.2,247.2,248.],'ambiguous'),
            (8,276.,296.5,'侧向握拳',[280.,282.5,288.5,290.3],'closed')],
    }
    controls={
        's1':[(148,154,'open_hold'),(174,181,'open_hold'),(201,211,'rotation_calibration'),
              (230,240,'rotation_calibration'),(270,280,'rotation_validation'),
              (336,344,'rotation_validation'),(350,361,'rotation_validation')],
        's2':[(75,84,'rotation_calibration'),(100,107,'open_hold'),(173,184,'open_hold'),
              (205,213,'rotation_validation'),(261,270,'open_hold'),(270,276,'rotation_calibration')],
    }
    result={}
    for label in SOURCES:
        a=[];n=0
        for cycle,lo,hi,title,edges,gesture in cycles[label]:
            borders=[lo]+edges+[hi]
            for phase,(start,end)in enumerate(zip(borders[:-1],borders[1:])):
                n+=1;state=['open','closing',gesture,'opening','open'][phase]
                if gesture=='ambiguous'and phase in [1,2,3]:state='ambiguous'
                a.append(dict(id=n,cycle_id=cycle,start=start,end=end,gesture=state,
                    phase=['open_before','closing','flexed_hold','opening','open_after'][phase],
                    intent=title+' '+state,kind='gesture',speed='unspecified',
                    boundary_uncertainty_s=.5,label_source='assistant visual review of original RGB; not independently user-adjudicated'))
                if gesture=='ambiguous':a[-1]['review_note']='Dense RGB review after shape-feature inspection cannot distinguish base-finger flexion from wrist tilt. Retained in all-cycle results; excluded only from confirmed-flexion claims.'
        for lo,hi,kind in controls[label]:
            n+=1;a.append(dict(id=n,cycle_id=0,start=lo,end=hi,gesture='open',phase=kind,
                              intent='张手对照 '+kind,kind=kind,speed='unspecified',boundary_uncertainty_s=.5))
        result[label]=sorted(a,key=lambda x:x['start'])
    return result


def prepare():
    from scipy.spatial.transform import Rotation as R
    from perception_mediapipe.recorded_rgbd import RecordedRGBD
    all_ann=annotations();json.dump(all_ann,open(OUT/'gesture_annotations.json','w'),ensure_ascii=False,indent=2)
    for label,path in SOURCES.items():
        directory=OUT/'input'/label;cache=directory/'frames';cache.mkdir(parents=True,exist_ok=True)
        info=source_info(path);ann=all_ann[label];reader=RecordedRGBD(path/'rgbd.db3')
        origin=info['origin'];requested=[c for c in reader.meta['color']if any(a['start']<=c['time']-origin<a['end']for a in ann)]
        meta=dict(source=str(path),source_commit='6f43bd3',origin=origin,video_origin=origin,
            time_axis='seconds since recording session creation, not screen MP4 PTS',
            intrinsics=reader.intr,depth_scale=reader.scale,requested_frames=len(requested),sampling_period_s=0.,
            raw_camera_frames=len(reader.meta['color']),betas=info['betas'],handedness_policy='fixed_left',
            frame_cache_directory=str(cache.resolve()),selection='all source camera frames in every image-reviewed gesture cycle and declared open-hand controls')
        json.dump(meta,open(directory/'metadata.json','w'),indent=2)
        json.dump(ann,open(directory/'annotations.json','w'),ensure_ascii=False,indent=2)
        with open(path/'imu_time_estimated.csv')as f:imu=list(csv.DictReader(f))
        it=np.array([int(r['estimated_sample_wall_time_ns'])/1e9-origin for r in imu])
        iq=R.from_euler('xyz',[[float(r[k])for k in ['roll_deg','pitch_deg','yaw_deg']]for r in imu],degrees=True)
        gyro=np.deg2rad([[float(r[k])for k in ['gyro_x_dps','gyro_y_dps','gyro_z_dps']]for r in imu])
        steps=R.from_rotvec((gyro[:-1]+gyro[1:])*.5*np.diff(it)[:,None]).as_quat()
        q=np.zeros((len(it),4));q[0,3]=1
        for i,step in enumerate(steps):
            x,y,z,w=q[i];X,Y,Z,W=step
            q[i+1]=[w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z]
            q[i+1]/=np.linalg.norm(q[i+1])
        np.savez_compressed(directory/'imu_signals.npz',t=it,q=iq.as_quat(),gyro_q=q,gyro_rad_s=gyro,
                            time_valid=np.array([r['time_valid']=='1'for r in imu]))
        json.dump(dict(used_offset_s=0.,method='nominal common host time; to be checked using open-hand rotation controls only',
                       absolute_sensor_latency_known=False),open(directory/'time_alignment.json','w'),indent=2)
        with open(directory/'mediapipe.jsonl','w')as stream:
            for i,c in enumerate(requested):
                row=dict(t=c['time']-origin,seq=c['frame'],frame_number=c['frame'],betas=info['betas'],
                         file=f'f{c["frame"]:07d}.npz',source_found=False)
                try:
                    try:rgb,depth,diag=reader.read(c['frame'],c['time'])
                    except RuntimeError:rgb,depth,diag=reader.read(c['frame'],c['time'])
                    row.update(source_found=True,depth_pair_available=True,**diag)
                except ValueError as exc:
                    row['source_reason']=str(exc)
                    if str(exc)=='PAIRED_DEPTH_MISSING':
                        tid=reader.topics[reader.prefix['color']+'/image/data']
                        blob=reader.db.execute('select data from messages where timestamp=? and topic_id=?',(c['stamp'],tid)).fetchone()[0]
                        im=reader.types.deserialize_cdr(blob,'sensor_msgs/msg/Image')
                        rgb=im.data.reshape(im.height,im.width,3).copy();depth=np.zeros(rgb.shape[:2],np.uint16)
                        row.update(source_found=True,depth_pair_available=False)
                if row['source_found']:np.savez_compressed(cache/row['file'],rgb=rgb,depth=depth)
                stream.write(json.dumps(row)+'\n')
                if i%300==0:stream.flush();print(label,'prepare',i,'/',len(requested),flush=True)
        reader.close();print(label,'prepared',len(requested),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['survey','prepare'])
    args=p.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    (survey if args.stage=='survey'else prepare)()
