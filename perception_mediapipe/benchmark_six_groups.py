"""Preloaded RGB-D to unfiltered pose timing, including fresh detection.

Six routes share CPU affinity and one local IPC request/reply. Three randomized
route-order repetitions use identical inputs, without pose-success screening.
The default has 96 frames; gesture stratification uses eight per state/session.
This measures processing latency, not camera/display/robot end-to-end latency.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perception_mediapipe.six_group_experiment import plain
from perception_mediapipe.six_group_geometry import mp_frames,hamer_frames,region_definition,PALM
OUT=ROOT/'.runtime/six_group_fixed_left_20260917'
REPORT=ROOT/'docs/six_group_fixed_left_20260917'


def preload():
    rows=json.load(open(OUT/'timing_inputs.json'))
    for r in rows:
        f=np.load(r['file']);r['rgb']=f['rgb'];r['depth']=f['depth']
    return rows


def server():
    from perception_mediapipe.pose import MediaPipePose
    model=MediaPipePose();rows=preload();print('READY',flush=True)
    for line in sys.stdin:
        req=json.loads(line);r=rows[req['i']];d,_=model.detect(r['rgb'])
        reply=dict(detection=d)
        if req['group'] in 'EF':reply['pose']=mp_frames(d,r['depth'],r['scale'],r['intrinsics'],groups=req['group'])[req['group']]
        print(json.dumps(reply,default=plain),flush=True)
    model.close()


def main():
    global OUT,REPORT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--report',type=Path,default=REPORT)
    parser.add_argument('--sessions',nargs='+',default=['earlier','later'])
    parser.add_argument('--gesture-stratified',action='store_true')
    args=parser.parse_args();OUT=args.output.resolve();REPORT=args.report.resolve()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{8,9,10,11})
    if args.server:return server()
    REPORT.mkdir(parents=True,exist_ok=True)
    selected=[]
    for label in args.sessions:
        p=OUT/label;meta=json.load(open(p/'metadata.json'));rows=[json.loads(x)for x in open(p/'mp.jsonl')]
        candidates=[r for r in rows if r['source_found']]
        betas=next(r['betas']for r in rows if r.get('betas')is not None)
        if args.gesture_stratified:
            ann=json.load(open(p/'annotations.json'));strata={}
            for r in candidates:
                a=next((a for a in ann if a['start']<=r['t']<a['end']),None)
                if a:strata.setdefault(a['gesture'],[]).append(r)
            chosen=[(state,rs[i])for state,rs in strata.items()for i in np.linspace(0,len(rs)-1,min(8,len(rs))).astype(int)]
        else:chosen=[('unspecified',candidates[i])for i in np.linspace(0,len(candidates)-1,48).astype(int)]
        cache=Path(meta.get('frame_cache_directory',str(ROOT/'.runtime/comparison'/label/'frames')))
        for state,r in chosen:
            selected.append(dict(file=str(cache/r['file']),gesture=state,
                intrinsics=meta['intrinsics'],scale=meta['depth_scale'],t=r['t'],betas=betas,session=label))
    json.dump(selected,open(OUT/'timing_inputs.json','w'))
    from perception_hamer.src.hamer_crop_inference import HamerCropInference
    from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition
    from perception_hamer.src.teleop_pose_packet import metric_wrist_ring_from_arrays
    asset=Path('/home/dongtian/myhandarmtest1-v1.0.1-ubuntu2004/perception_hamer/_DATA')
    runner=HamerCropInference(str(asset/'hamer_ckpts/checkpoints/hamer.ckpt'),data_root=str(asset),device='cuda:0',precision='fp16',freeze_betas=True)
    runner.load();runner.warmup();defs={};regions={}
    weights=runner._model.mano.lbs_weights.detach().cpu().numpy()[:,0]
    for side in [False,True]:
        v,j=runner.neutral_mano_geometry(side);defs[side]=build_mano_wrist_definition(v,j,runner.mano_faces(),side)
        regions[side]=region_definition(v,j,weights)
    rows=preload()
    child=subprocess.Popen(['/home/dongtian/anaconda3/envs/mediapipe_env/bin/python','-u',__file__,'--server','--output',str(OUT)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(OUT/'timing_server.log','w'),text=True)
    assert child.stdout.readline().strip()=='READY'
    samples=[];rng=np.random.default_rng(17);orders=[]
    try:
        for repeat in range(3):
            order=rng.permutation(list('ABCDEF')).tolist();orders.append(order)
            for group in order:
                for index in list(range(4))+list(range(len(rows))):
                    r=rows[index];runner.set_frozen_betas(False,np.asarray(r['betas']));runner.set_frozen_betas(True,np.asarray(r['betas']))
                    start=time.perf_counter()
                    child.stdin.write(json.dumps(dict(i=index,group=group))+'\n');child.stdin.flush()
                    result=json.loads(child.stdout.readline());d=result['detection'];valid=False
                    if group in 'EF':valid=result['pose']['valid'] and result['pose']['position_valid']
                    elif d is not None and d['visibility']['palm_inside_image']:
                        side=d['is_right'];pred=runner.infer(r['rgb'],d['bbox'],side,r['t'])
                        pose=hamer_frames(pred,defs[side],regions[side],groups=group)[group]
                        try:
                            _,conf,_,_=metric_wrist_ring_from_arrays(pred.pred_vertices_mano_right_canonical,defs[side].wrist_loop,
                                pred.hamer_crop_projection_translation,pred.hamer_nominal_crop_focal_length,pred.quality['affine_original_to_crop'],
                                r['depth'],r['scale'],r['intrinsics'],reference_depth_m=None,palm_anchor_pixels=np.asarray(d['xy'])[PALM])
                            valid=pose['valid'] and conf>0
                        except ValueError:valid=False
                    elapsed=1000*(time.perf_counter()-start)
                    samples.append(dict(repeat=repeat,group=group,index=index,ms=elapsed,valid=bool(valid),gesture=r['gesture'],session=r['session'],t=r['t']))
                # Drop the four warmups by their loop positions, not frame IDs.
                del samples[-(len(rows)+4):-(len(rows))]
                a=[x['ms']for x in samples if x['group']==group and x['repeat']==repeat]
                print('timing',repeat,group,round(float(np.median(a)),2),'ms',flush=True)
    finally:
        child.stdin.close();child.wait(timeout=15)
    summary={};stratified=[]
    for g in 'ABCDEF':
        r=[x for x in samples if x['group']==g];a=np.array([x['ms']for x in r])
        summary[g]=dict(n=len(a),valid=sum(x['valid']for x in r),median_ms=float(np.median(a)),mean_ms=float(a.mean()),
            p95_ms=float(np.percentile(a,95)),repeat_median_ms=[float(np.median([x['ms']for x in r if x['repeat']==i]))for i in range(3)])
        va=[x['ms']for x in r if x['valid']]
        summary[g].update(valid_median_ms=float(np.median(va))if va else None,valid_p95_ms=float(np.percentile(va,95))if va else None)
        for state in sorted({x['gesture']for x in r}):
            sr=[x for x in r if x['gesture']==state];vs=[x['ms']for x in sr if x['valid']]
            stratified.append(dict(group=g,gesture=state,n=len(sr),valid=len(vs),attempt_median_ms=float(np.median([x['ms']for x in sr])),
                valid_median_ms=float(np.median(vs))if vs else None,valid_p95_ms=float(np.percentile(vs,95))if vs else None))
    json.dump(dict(scope='Preloaded RGB-D to one pose, fresh detector, one IPC, no disk/camera/display/robot latency; successful and failed routes separated.',
        cpu_affinity=[8,9,10,11],warmups_per_route=4,input_frames=len(rows),repetitions=3,
        route_orders=orders,models=summary,by_gesture=stratified,samples=samples),open(REPORT/'processing_timing.json','w'),indent=2)

if __name__=='__main__':main()
