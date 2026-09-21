#!/usr/bin/env python3
"""Early camera conditioning probe with matched constant-input control.

The same learned encoding is added before all ViT blocks and before the pose
decoder, following the two insertion points in the author's HaMeR/KPE code.
Pretrained weights and original articulation stay frozen. This is a small-data
adapter experiment, not a reproduction of the author's full training recipe.
"""
import argparse,json,time
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_hamer_camera_adapter import (
    model,CameraAdapter,load_pack,score,horizontal_baseline,M,MOUNT,camera_encoding)
from perception_hamer.src.hamer_crop_inference import prepare_hamer_crop
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame


def patch_tokens(runner,metadata,path):
    if path.exists():return torch.from_numpy(np.load(path))
    arrays=[];cfg=runner._cfg
    for i,row in enumerate(metadata):
        p=Path(row['file']);rgb=np.load(p)['rgb']if p.suffix=='.npz'else cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
        crop,_,_=prepare_hamer_crop(rgb,row['bbox'],False,image_size=int(cfg.MODEL.IMAGE_SIZE),
            bbox_shape=tuple(cfg.MODEL.get('BBOX_SHAPE',[192,256])),rescale_factor=runner.rescale_factor,
            image_mean=tuple(cfg.MODEL.IMAGE_MEAN),image_std=tuple(cfg.MODEL.IMAGE_STD))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            token,shape=runner._model.backbone.patch_embed(torch.from_numpy(crop[None,:,:,32:-32]).cuda())
        if shape!=(16,12):raise ValueError('Unexpected patch grid')
        arrays.append(token[0].cpu().numpy().astype(np.float16))
        if i%150==0:print('patch inputs',path.stem,i,'/',len(metadata),flush=True)
    result=np.stack(arrays);np.save(path,result);return torch.from_numpy(result)


def load_inputs(runner,data,split,out):
    pack=load_pack(data,split);pack['data'].pop('features')
    pack['patches']=patch_tokens(runner,pack['meta'],out/f'{split}_patches.npy')
    return pack


def forward(runner,adapter,patches,encoding,local,training=False):
    backbone=runner._model.backbone
    bias=adapter.net(encoding if adapter.use_camera else torch.zeros_like(encoding))if adapter is not None else None
    with torch.autocast('cuda',dtype=torch.float16):
        # Use the original positional embeddings and original 32 blocks.
        x=patches+backbone.pos_embed[:,1:]+backbone.pos_embed[:,:1]
        if bias is not None:x=x+bias[:,None,:].to(x.dtype)
        for block in backbone.blocks:
            x=checkpoint(block,x,use_reentrant=False)if training else block(x)
        x=backbone.last_norm(x)
    context=x.permute(0,2,1).reshape(len(patches),1280,16,12).float()
    if bias is not None:context=context+bias[:,:,None,None]
    G=runner._model.mano_head(context)[0]['global_orient'][:,0]
    palm=torch.tensor(M,dtype=torch.float32,device='cuda')@G@local
    return palm,bias


def predict(runner,adapter,pack,batch=16):
    result=[]
    with torch.no_grad():
        for first in range(0,len(pack['meta']),batch):
            ix=slice(first,first+batch)
            p,_=forward(runner,adapter,pack['patches'][ix].cuda(),pack['data']['encoding'][ix].cuda(),pack['data']['local_factor'][ix].cuda())
            result.append(p.cpu().numpy())
    return np.concatenate(result)


def synthetic(runner,adapters,out):
    source=ROOT/'docs/viewpoint_pose_cross_20260921';cases=json.loads((source/'results.json').read_text())
    scene=np.load(source/'frozen_scene.npz');rows=[];refs={}
    for row in cases:
        rgb=cv2.cvtColor(cv2.imread(str(source/row['image'])),cv2.COLOR_BGR2RGB)
        rgb=cv2.copyMakeBorder(rgb,160,160,160,160,cv2.BORDER_CONSTANT,value=(214,219,224))
        runner.set_frozen_betas(False,scene['betas']);pred=runner.infer(rgb,row['input_box'],False,0.)
        palm=build_hamer_five_point_palm_frame(pred.pred_keypoints_3d_source_camera_axes,False)
        if not palm.valid:raise ValueError(palm.failure_reason)
        local=torch.tensor(pred.global_orient.T@M@palm.rotation,dtype=torch.float32,device='cuda')[None]
        crop,affine,_=prepare_hamer_crop(rgb,row['input_box'],False)
        enc=torch.tensor(camera_encoding(affine,160),device='cuda')[None]
        item=dict(view_deg=row['value_deg'],true_yaw_deg=row['true_yaw_deg'],methods={})
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            patches,_=runner._model.backbone.patch_embed(torch.from_numpy(crop[None,:,:,32:-32]).cuda())
        for name,adapter in [('baseline',None)]+list(adapters.items()):
            with torch.no_grad():p,_=forward(runner,adapter,patches,enc,local)
            q=p[0].cpu().numpy()
            if not rows:refs[name]=q
            error=np.rad2deg(R.from_matrix(np.array(row['true_rotation']).T@q@refs[name].T).magnitude())
            item['methods'][name]=dict(rotation=q.tolist(),full_relative_error_deg=float(error))
        rows.append(item)
    for row in rows:
        center=next(x for x in rows if x['true_yaw_deg']==row['true_yaw_deg']and x['view_deg']==0)
        for name,item in row['methods'].items():
            item['same_pose_position_error_deg']=float(np.rad2deg(R.from_matrix(np.array(item['rotation'])@np.array(center['methods'][name]['rotation']).T).magnitude()))
    (out/'synthetic_predictions.json').write_text(json.dumps(rows,indent=2))
    edges=[x for x in rows if x['view_deg']!=0]
    return {name:{k:float(np.mean([r['methods'][name][k]for r in edges]))for k in ['full_relative_error_deg','same_pose_position_error_deg']}for name in refs}


def task_test(runner,adapters,data,out):
    z=np.load(data/'task_features.npz');meta=json.loads((data/'task_metadata.json').read_text())
    pack=dict(meta=meta,data={k:torch.from_numpy(z[k].astype(np.float32))for k in ['encoding','local_factor']})
    pack['patches']=patch_tokens(runner,meta,out/'task_patches.npy')
    poses={name:predict(runner,ad,pack)for name,ad in [('baseline',None)]+list(adapters.items())}
    t=np.array([x['t']for x in meta]);dt=np.diff(t);keep=(dt>.005)&(dt<.2);metrics={}
    br=poses['baseline']@poses['baseline'][0].T
    for name,q in poses.items():
        steps=np.degrees(R.from_matrix(q[1:]@np.swapaxes(q[:-1],1,2)).magnitude())[keep]
        diff=np.degrees(R.from_matrix(np.swapaxes(br,1,2)@(q@q[0].T)).magnitude())
        metrics[name]=dict(step_mean_deg=float(steps.mean()),step_p95_deg=float(np.percentile(steps,95)),step_max_deg=float(steps.max()),
            steps_above_10deg=int((steps>10).sum()),difference_from_baseline_mean_deg=float(diff.mean()),difference_from_baseline_p95_deg=float(np.percentile(diff,95)))
    np.savez_compressed(out/'task_predictions.npz',t=t,**poses)
    return dict(source=meta[0]['source'],ground_truth_available=False,methods=metrics,
        scope='Prediction-change diagnostics only; not error, grasp success, or closed-loop robot performance')


def run(args):
    args.out.mkdir(parents=True,exist_ok=True);runner=model()
    for p in runner._model.parameters():p.requires_grad_(False)
    # Match the previous experiment: frozen FP16 image backbone, FP32 decoder.
    runner._model.mano_head.float()
    runner._model.eval()
    tr=load_inputs(runner,args.data,'train',args.out);va=load_inputs(runner,args.data,'validation',args.out)
    t=np.array([x['t']for x in tr['meta']]);usable=np.flatnonzero(tr['valid'][:-2]&tr['valid'][2:]&(t[2:]-t[:-2]<1.3))
    target=torch.from_numpy((MOUNT.T@np.swapaxes(tr['imu'][:-2],1,2)@tr['imu'][2:]@MOUNT).astype(np.float32))
    bv=predict(runner,None,va)
    protocol=dict(source_data=str(args.data),epochs=args.epochs,pairs_per_batch=args.pairs_per_batch,learning_rate=args.lr,seed=21,
        insertion='Same camera encoding before all 32 image blocks and before pose decoder',
        frozen='All pretrained weights and original per-frame articulation/shape',
        controls='Identical zero-input adapter, original HaMeR, previous late-only results',
        selection='validation full_increment_mae_deg only',supervision='IMU relative rotation about one second apart, fixed nominal mounting',
        boundary='Exploratory reuse of previously inspected development recording; no new blind dataset; not full paper training recipe')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary=dict(protocol=protocol,baseline_validation=score(bv,va),variants={})
    fitted={}
    for name,camera in [('constant_control',False),('camera_conditioned',True)]:
        torch.manual_seed(21);np.random.seed(21)
        adapter=CameraAdapter(camera).cuda();opt=torch.optim.Adam(adapter.parameters(),lr=args.lr)
        scaler=torch.cuda.amp.GradScaler(init_scale=128.)
        best=float('inf');best_epoch=0;history=[]
        for epoch in range(args.epochs+1):
            started=time.perf_counter();losses=[]
            if epoch:
                order=np.random.permutation(usable)
                for first in range(0,len(order),args.pairs_per_batch):
                    ix=order[first:first+args.pairs_per_batch];both=np.r_[ix,ix+2]
                    p,bias=forward(runner,adapter,tr['patches'][both].cuda(),tr['data']['encoding'][both].cuda(),tr['data']['local_factor'][both].cuda(),training=True)
                    rel=p[:len(ix)].transpose(1,2)@p[len(ix):]
                    loss=(rel-target[ix].cuda()).square().mean()+1e-4*bias.square().mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; stop this run')
                    opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.step(opt);scaler.update()
                    losses.append(float(loss.detach()))
                    if first==0 or first%(args.pairs_per_batch*20)==0:
                        print(name,'epoch',epoch,'pairs',first,'/',len(order),'loss',round(losses[-1],6),
                              'gpu_MiB',round(torch.cuda.max_memory_allocated()/2**20),flush=True)
            pv=predict(runner,adapter,va);metric=score(pv,va);value=metric['full_increment_mae_deg']
            history.append(dict(epoch=epoch,loss=float(np.mean(losses))if losses else None,validation=metric,seconds=time.perf_counter()-started))
            if value<best:
                best=value;best_epoch=epoch;torch.save(adapter.state_dict(),args.out/f'{name}.pt')
            (args.out/f'{name}_history.json').write_text(json.dumps(history,indent=2))
            print(name,'epoch',epoch,'validation full 3D increment MAE',round(value,4),'best',best_epoch,'seconds',round(time.perf_counter()-started,1),flush=True)
        adapter.load_state_dict(torch.load(args.out/f'{name}.pt'));fitted[name]=adapter
        summary['variants'][name]=dict(best_epoch=best_epoch,validation=history[best_epoch]['validation'])
    # No test scores or task scores participate in checkpoint selection.
    test=load_inputs(runner,args.data,'test',args.out)
    predictions={'baseline':predict(runner,None,test)}
    summary['baseline_test']=score(predictions['baseline'],test)
    predictions['horizontal_baseline']=horizontal_baseline(predictions['baseline'],test['meta'])
    summary['horizontal_baseline_test']=score(predictions['horizontal_baseline'],test)
    for name,ad in fitted.items():
        predictions[name]=predict(runner,ad,test);summary['variants'][name]['test']=score(predictions[name],test)
    np.savez_compressed(args.out/'test_predictions.npz',**predictions,t=[x['t']for x in test['meta']],imu=test['imu'],imu_valid=test['valid'])
    summary['synthetic_test']=synthetic(runner,fitted,args.out)
    if args.task_data:summary['task_test']=task_test(runner,fitted,args.task_data,args.out)
    (args.out/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--task-data',type=Path);p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--pairs-per-batch',type=int,default=8);p.add_argument('--lr',type=float,default=.001)
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2);run(args)
