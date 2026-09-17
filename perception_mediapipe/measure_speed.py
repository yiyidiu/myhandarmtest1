"""Measured, unthrottled RGB-D -> raw-pose throughput, all stages included.

Images are preloaded. All modes use fresh MediaPipe detection, the same CPU
affinity and the same small local IPC boundary. No disk, GUI, ROS or camera wait
is counted. HaMeR includes its extra GPU network; not just cached ROI timing.
"""
import json,os,sys,time,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perception_mediapipe.compare_recordings import plain
OUT=ROOT/'.runtime/comparison'


def preload():
    rows=json.load(open(OUT/'speed_input.json'))
    for row in rows:
        f=np.load(row['file']);row['rgb']=f['rgb'];row['depth']=f['depth']
    return rows


def server():
    from perception_mediapipe.pose import MediaPipePose
    model=MediaPipePose();rows=preload()
    print('READY',flush=True)
    for line in sys.stdin:
        req=json.loads(line);r=rows[req['i']];d,dt=model.detect(r['rgb']);reply=dict(d=d)
        if req['mode']!='hamer':
            reply['pose']=model.poses(d,r['depth'],r['scale'],r['intrinsics'],modes=(req['mode'],))[req['mode']]
        print(json.dumps(reply,default=plain),flush=True)
    model.close()


def main():
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{8,9,10,11})
    if '--server' in sys.argv:return server()
    from perception_hamer.src.hamer_crop_inference import HamerCropInference
    from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition,estimate_mano_wrist_frame
    from perception_hamer.src.teleop_pose_packet import metric_wrist_ring_from_arrays
    selected=[]
    for label in ['earlier','later']:
        p=OUT/label;meta=json.load(open(p/'metadata.json'))
        mp=[json.loads(x)for x in open(p/'mediapipe.jsonl')];ha=[json.loads(x)for x in open(p/'hamer.jsonl')]
        ok=[i for i,(m,h)in enumerate(zip(mp,ha))if h['valid'] and all(m.get('poses',{}).get(k,{}).get('valid',False)for k in ['rgbd','world'])]
        for i in np.linspace(0,len(ok)-1,64).astype(int):
            r=mp[ok[i]];selected.append(dict(file=str(p/'frames'/r['file']),intrinsics=meta['intrinsics'],scale=meta['depth_scale'],t=r['t'],betas=r['betas']))
    json.dump(selected,open(OUT/'speed_input.json','w'))
    rows=preload();asset=Path('/home/dongtian/myhandarmtest1-v1.0.1-ubuntu2004/perception_hamer/_DATA')
    runner=HamerCropInference(str(asset/'hamer_ckpts/checkpoints/hamer.ckpt'),data_root=str(asset),device='cuda:0',precision='fp16',freeze_betas=True)
    runner.load();runner.warmup();definitions={}
    for side in [False,True]:
        v,j=runner.neutral_mano_geometry(side);definitions[side]=build_mano_wrist_definition(v,j,runner.mano_faces(),side)
    child=subprocess.Popen(['/home/dongtian/anaconda3/envs/mediapipe_env/bin/python','-u',__file__,'--server'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(OUT/'speed_server.log','w'),text=True)
    assert child.stdout.readline().strip()=='READY'
    results={}
    try:
        for mode in ['world','rgbd','hamer']:
            timings=[];valid=[]
            # Five warmup operations, then every selected input exactly once.
            for i in list(range(5))+list(range(len(rows))):
                r=rows[i];began=time.perf_counter()
                child.stdin.write(json.dumps(dict(i=i,mode=mode))+'\n');child.stdin.flush()
                answer=json.loads(child.stdout.readline());d=answer['d'];good=d is not None
                if mode=='hamer' and good:
                    side=d['is_right'];definition=definitions[side]
                    if r['betas'] is not None:runner.set_frozen_betas(side,np.array(r['betas']))
                    pred=runner.infer(r['rgb'],d['bbox'],side,r['t'])
                    frame=estimate_mano_wrist_frame(pred.pred_vertices_source_camera_axes,definition)
                    pos,conf,_,_=metric_wrist_ring_from_arrays(pred.pred_vertices_mano_right_canonical,definition.wrist_loop,pred.hamer_crop_projection_translation,pred.hamer_nominal_crop_focal_length,pred.quality['affine_original_to_crop'],r['depth'],r['scale'],r['intrinsics'],palm_anchor_pixels=[d['xy'][j] for j in [0,5,9,13,17]])
                    good=bool(frame.valid and conf>0)
                elif mode!='hamer':good=answer['pose']['valid']
                timings.append((time.perf_counter()-began)*1000);valid.append(good)
            a=np.array(timings[5:]);results[mode]=dict(n=len(a),valid=sum(valid[5:]),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),mean_ms=float(a.mean()),measured_processing_hz=float(1000/a.mean()),samples_ms=a.tolist())
            print(mode,{k:v for k,v in results[mode].items()if k!='samples_ms'},flush=True)
    finally:
        child.stdin.close();child.wait(timeout=10)
    json.dump(dict(scope=__doc__,cpu_affinity=[8,9,10,11],camera_fps=30,models=results),open(OUT/'speed.json','w'),indent=2)

if __name__=='__main__':main()
