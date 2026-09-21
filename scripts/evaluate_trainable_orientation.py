#!/usr/bin/env python3
"""Fixed-checkpoint transfer and native-cadence checks; no tuning or control."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from experiment_hamer_camera_adapter import (
    ROOT, LATEST, M, MOUNT, CameraAdapter, camera_encoding, load_pack, model, predict, score,
)
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame


def stats(x):
    x = np.asarray(x)
    return dict(n=len(x), mean_deg=float(x.mean()), p95_deg=float(np.percentile(x,95)), max_deg=float(x.max())) if len(x) else dict(n=0)


def native_score(q, pack):
    t = np.array([x['t'] for x in pack['meta']]); imu = pack['imu']; valid = pack['valid']
    measurement = np.array([x.get('role') != 'calibration' for x in pack['meta']])
    idx = np.flatnonzero(valid & measurement); zero = int(np.flatnonzero(valid)[0])
    ir = imu[zero].T @ imu[idx]; pr = MOUNT @ q[zero].T @ q[idx] @ MOUNT.T
    result = dict(single_C=stats(np.rad2deg(R.from_matrix(np.swapaxes(ir,1,2) @ pr).magnitude())), horizons={})
    for h in [1,5,10]:
        j = np.array([int(np.argmin(abs(t-t[i]-h))) for i in idx])
        keep = (j>idx) & valid[j] & (abs(t[j]-t[idx]-h)<=.05)
        a, b = idx[keep], j[keep]
        ir = np.swapaxes(imu[a],1,2) @ imu[b]; pr = MOUNT @ np.swapaxes(q[a],1,2) @ q[b] @ MOUNT.T
        err = np.rad2deg(R.from_matrix(np.swapaxes(ir,1,2) @ pr).magnitude())
        result['horizons'][str(h)] = stats(err)
        if h == 1:
            v = R.from_matrix(pr).as_rotvec(); rv = R.from_matrix(ir).as_rotvec(); ra = np.linalg.norm(rv,axis=1)
            low = ra < np.deg2rad(3); large = ra > np.deg2rad(10)
            ratio = np.sum(v[large]*rv[large],axis=1)/ra[large]**2
            cross = np.rad2deg(np.linalg.norm(v[large]-ratio[:,None]*rv[large],axis=1))
            result['motion'] = dict(low_reference_pairs=int(low.sum()), low_reference_mean_deg=float(np.rad2deg(ra[low]).mean()),
                low_visual_mean_deg=float(np.rad2deg(np.linalg.norm(v[low],axis=1)).mean()), large_reference_pairs=int(large.sum()),
                projected_ratio_mean=float(ratio.mean()), projected_ratio_median=float(np.median(ratio)),
                projected_abs_ratio_error_mean=float(abs(ratio-1).mean()), opposite_projection_pairs=int((ratio<0).sum()),
                cross_axis_mean_deg=float(cross.mean()))
    result['balanced_horizon_mae_deg'] = float(np.mean([v['mean_deg'] for v in result['horizons'].values()]))
    dt = np.diff(t)
    a = np.flatnonzero(valid[:-1] & valid[1:] & measurement[:-1] & measurement[1:] & (dt>0) & (dt<.2)); b = a+1
    ir = np.swapaxes(imu[a],1,2) @ imu[b]; pr = MOUNT @ np.swapaxes(q[a],1,2) @ q[b] @ MOUNT.T
    errors = np.rad2deg(R.from_matrix(np.swapaxes(ir,1,2) @ pr).magnitude())
    result['adjacent_reference_error'] = dict(**stats(errors), above5deg=int((errors>5).sum()), above10deg=int((errors>10).sum()))
    result['adjacent_visual_step'] = stats(np.rad2deg(R.from_matrix(pr).magnitude()))
    a = int(np.argmin(abs(t-439.21906876564026))); b = int(np.argmin(abs(t-440.28619956970215)))
    ir = imu[a].T @ imu[b]; pr = MOUNT @ q[a].T @ q[b] @ MOUNT.T
    result['previous_grip_failure_pair'] = dict(times=[float(t[a]),float(t[b])],
        reference_angle_deg=float(np.rad2deg(R.from_matrix(ir).magnitude())),
        error_deg=float(np.rad2deg(R.from_matrix(ir.T @ pr).magnitude())))
    return result


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    runner = model(); head = runner._model.mano_head.float().eval()
    for p in runner._model.parameters(): p.requires_grad_(False)
    models = {'baseline':(head,None)}
    for name,camera in [('constant',False),('camera',True)]:
        legacy = CameraAdapter(camera).cuda()
        legacy.load_state_dict(torch.load(args.frozen / ('constant_control.pt' if not camera else 'camera_conditioned.pt')))
        models['frozen_'+name] = (head,legacy)
        state = torch.load(args.fit / ('constant_control.pt' if not camera else 'camera_conditioned.pt'),map_location='cpu')
        adapted_head = copy.deepcopy(head); adapted_head.load_state_dict(state['head']); adapted_head.eval()
        adapter = CameraAdapter(camera).cuda(); adapter.load_state_dict(state['adapter']); adapter.eval()
        for p in adapted_head.parameters():p.requires_grad_(False)
        models['trained_'+name] = (adapted_head,adapter)
        del state
    runner._model.backbone.cpu(); runner._model.mano.cpu(); torch.cuda.empty_cache()
    (args.out / 'protocol.json').write_text(json.dumps(dict(checkpoints=str(args.fit),frozen_comparison=str(args.frozen),
        scope='All checkpoints already chosen on validation, no fitting or selection on these replay results.',
        native_horizons='Closest future source sample at 1/5/10 seconds within 50 ms; no fixed index offsets at native cadence.',
        native_adjacent='Consecutive saved valid source inputs with 0<dt<0.2 s. Longer gaps are counted separately.',
        task='No IMU and no controller replay; visual-step statistics are not tracking accuracy or grasp success.'),indent=2))
    pack = load_pack(args.native, 'native')
    t = np.array([x['t'] for x in pack['meta']]); dt=np.diff(t)[1:]
    summary = dict(frames=len(t),measurement_frames=len(t)-1, imu_valid_measurements=int(pack['valid'][1:].sum()),
        processed_dt_median_s=float(np.median(dt)), gaps_over_0p2s=int((dt>=.2).sum()),methods={})
    predictions = {}
    for name,(h,ad) in models.items():
        predictions[name],_=predict(h,ad,pack)
        summary['methods'][name]=native_score(predictions[name],pack)
    np.savez_compressed(args.out/'native_predictions.npz',t=t,imu=pack['imu'],imu_valid=pack['valid'],**predictions)
    (args.out/'native_metrics.json').write_text(json.dumps(summary,indent=2))
    print('Native cadence complete',len(t),flush=True);del pack,predictions
    pack=load_pack(ROOT/'.runtime/hamer_camera_adapter_20260921','train');pack['meta'][0]['role']='calibration'
    times=np.array([x['t']for x in pack['meta']]);poses={};metrics={};segments=[]
    for name,(h,ad) in models.items():
        poses[name],_=predict(h,ad,pack);metrics[name]=score(poses[name],pack)
    for ann in json.loads((LATEST/'analysis_intent/annotations.json').read_text()):
        ix=np.flatnonzero((times>=ann['start'])&(times<ann['end'])&pack['valid']); row=dict(annotation=ann,frames=len(ix),methods={})
        if len(ix)>=2:
            a,b=ix[0],ix[1:];ir=pack['imu'][a].T@pack['imu'][b]
            for name,q in poses.items():
                pr=MOUNT@q[a].T@q[b]@MOUNT.T
                row['methods'][name]=stats(np.rad2deg(R.from_matrix(np.swapaxes(ir,1,2)@pr).magnitude()))
        segments.append(row)
    np.savez_compressed(args.out/'transfer_predictions.npz',t=times,imu=pack['imu'],imu_valid=pack['valid'],**poses)
    (args.out/'transfer_metrics.json').write_text(json.dumps(metrics,indent=2))
    (args.out/'transfer_segments.json').write_text(json.dumps(segments,ensure_ascii=False,indent=2))
    print('Different-recording replay complete',len(times),flush=True);del pack,poses
    data=ROOT/'.runtime/hamer_camera_adapter_grasp_20260921';z=np.load(data/'task_features.npz')
    meta=json.loads((data/'task_metadata.json').read_text())
    pack=dict(meta=meta,data={k:torch.from_numpy(z[k].astype(np.float32)if k!='features'else z[k])for k in z.files})
    t=np.array([x['t']for x in meta]);dt=np.diff(t);native=(dt>.005)&(dt<.2);poses={};metrics={}
    for name,(h,ad) in models.items():
        q,_=predict(h,ad,pack);poses[name]=q
        steps=np.rad2deg(R.from_matrix(q[1:]@np.swapaxes(q[:-1],1,2)).magnitude())[native]
        metrics[name]=dict(**stats(steps),above10deg=int((steps>10).sum()))
    np.savez_compressed(args.out/'task_predictions.npz',t=t,**poses)
    (args.out/'task_metrics.json').write_text(json.dumps(dict(frames=len(t),native_pairs=int(native.sum()),imu_available=False,methods=metrics),indent=2))
    print('Existing grasp replay complete',len(t),flush=True);del pack,poses,z
    # Known-pose synthetic test. The original head stays in runner so every
    # candidate receives the same original features and articulation factor.
    runner._model.backbone.cuda();runner._model.mano.cuda();captured=[]
    hook=runner._model.backbone.register_forward_hook(lambda _m,_i,x:captured.append(x.detach().float()))
    source=ROOT/'docs/viewpoint_pose_cross_20260921';scene=np.load(source/'frozen_scene.npz')
    cases=json.loads((source/'results.json').read_text());rows=[];refs={};reflection=torch.tensor(M,dtype=torch.float32,device='cuda')
    for row in cases:
        rgb=cv2.cvtColor(cv2.imread(str(source/row['image'])),cv2.COLOR_BGR2RGB)
        rgb=cv2.copyMakeBorder(rgb,160,160,160,160,cv2.BORDER_CONSTANT,value=(214,219,224))
        captured.clear();runner.set_frozen_betas(False,scene['betas']);pred=runner.infer(rgb,row['input_box'],False,0.)
        palm=build_hamer_five_point_palm_frame(pred.pred_keypoints_3d_source_camera_axes,False)
        if not palm.valid:raise ValueError(palm.failure_reason)
        local=torch.tensor(pred.global_orient.T@M@palm.rotation,dtype=torch.float32,device='cuda')
        enc=torch.tensor(camera_encoding(pred.quality['affine_original_to_crop'],160),device='cuda')[None]
        item=dict(view_deg=row['value_deg'],true_yaw_deg=row['true_yaw_deg'],methods={})
        for name,(h,ad) in models.items():
            with torch.no_grad():
                x=captured[0]if ad is None else ad(captured[0],enc)[0]
                g=h(x)[0]['global_orient'][0,0];q=(reflection@g@local).cpu().numpy()
            if not rows:refs[name]=q
            item['methods'][name]=dict(rotation=q.tolist(),full_error_deg=float(np.rad2deg(R.from_matrix(np.array(row['true_rotation']).T@q@refs[name].T).magnitude())))
        rows.append(item)
    hook.remove()
    for row in rows:
        center=next(x for x in rows if x['view_deg']==0 and x['true_yaw_deg']==row['true_yaw_deg'])
        for name,item in row['methods'].items():
            item['same_pose_position_difference_deg']=float(np.rad2deg(R.from_matrix(np.array(item['rotation'])@np.array(center['methods'][name]['rotation']).T).magnitude()))
    edges=[x for x in rows if x['view_deg']!=0]
    summary={n:{k:float(np.mean([x['methods'][n][k]for x in edges]))for k in ['full_error_deg','same_pose_position_difference_deg']}for n in models}
    (args.out/'synthetic_predictions.json').write_text(json.dumps(rows,indent=2))
    (args.out/'synthetic_metrics.json').write_text(json.dumps(summary,indent=2))
    print('Known-pose synthetic test complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    run(p.parse_args())
