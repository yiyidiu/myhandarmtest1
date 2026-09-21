#!/usr/bin/env python3
"""Matched IMU supervision with/without known-camera-rotation augmentation.

Offline single-image orientation experiment. Original RGB features and hand
articulation are frozen; no IMU, timestamps or action labels enter inference.
"""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from experiment_hamer_camera_adapter import (
    ROOT, K, M, MOUNT, CameraAdapter, camera_encoding, load_pack, model, predict, reference, score,
)
from experiment_camera_geometry import (
    KINV, affine, bounds, corners, coverage, frame_outputs, project, read_rgb,
    relative_errors, grouped, stats,
)
from experiment_trainable_orientation import enable_orientation_training


def collect_views(runner,metadata,valid):
    captured=[]
    hook=runner._model.backbone.register_forward_hook(
        lambda _m,_i,o:captured.append(o.detach().cpu().numpy()[0].astype(np.float16)))
    features=[];encodings=[];locals_=[];source=[];rotations=[];records=[]
    rejected=dict(virtual_box_outside=0,incomplete_crop=0)
    transforms=[(axis,sign,R.from_euler(axis,sign*12,degrees=True).as_matrix())
        for axis in ['x','y']for sign in [-1,1]]
    for i,row in enumerate(metadata):
        if not valid[i]:continue
        rgb=read_rgb(row);_,quality=affine(rgb.shape,row['bbox'],runner)
        for axis,sign,S in transforms:
            h=K@S@KINV;bb=bounds(project(corners(row['bbox']),h))
            if bb[0]<0 or bb[1]<0 or bb[2]>=rgb.shape[1] or bb[3]>=rgb.shape[0]:
                rejected['virtual_box_outside']+=1
                continue
            av,_=affine(rgb.shape,bb,runner)
            H=av@h
            # Geometric membership only: no invented pixels, original or virtual
            # frame-edge truncation, error-based selection, or IMU pose selection.
            if min(coverage(rgb.shape,H),coverage(rgb.shape,av))<1:
                rejected['incomplete_crop']+=1
                continue
            captured.clear()
            out=frame_outputs(runner,rgb,H,np.eye(3),row['betas'],quality['crop_bbox_size_px'])
            if len(captured)!=1:raise RuntimeError('Expected one original image feature extraction')
            features.append(captured[0]);encodings.append(camera_encoding(av[:2]))
            # Source-camera global = M G M; L=G^T M P=M (M G M)^T P.
            locals_.append(M@np.asarray(out['global_rotation']).T@np.asarray(out['palm']))
            source.append(i);rotations.append(S)
            records.append(dict(source_index=i,seq=row['seq'],t=row['t'],axis=axis,degrees=sign*12))
        if i%50==0:print('Geometry inputs',i,'/',len(metadata),'accepted',len(features),flush=True)
    hook.remove()
    if not features:raise RuntimeError('No fully supported camera transforms')
    print('Geometry inputs complete',len(features),'from',len(set(source)),'source images',flush=True)
    return dict(features=np.asarray(features),encoding=np.asarray(encodings,dtype=np.float32),
        local_factor=np.asarray(locals_,dtype=np.float32),source_index=np.asarray(source),
        source_to_virtual=np.asarray(rotations,dtype=np.float32)),records,rejected


def cache(args):
    metadata=json.loads((args.data/'train_metadata.json').read_text())
    _,valid=reference(metadata)
    data,records,rejected=collect_views(model(),metadata,valid)
    np.savez_compressed(args.out/'augmentation_features.npz',**data)
    (args.out/'augmentation_metadata.json').write_text(json.dumps(records,indent=2))
    (args.out/'augmentation_exclusions.json').write_text(json.dumps(rejected,indent=2))


def load_aug(args):
    z=np.load(args.out/'augmentation_features.npz')
    return {k:torch.from_numpy(z[k].astype(np.float32)if k not in ['features','source_index']else z[k])for k in z.files}


def original_head():
    # Features are already frozen on disk. Loading the full ViT and copying it
    # back to host RAM is unnecessary and caused an earlyoom termination.
    from hamer.configs import get_config
    from hamer.models.heads.mano_head import build_mano_head
    data=ROOT/'perception_hamer/_DATA'
    cfg=get_config(str(data/'hamer_ckpts/model_config.yaml'),update_cachedir=False)
    cfg.defrost();cfg.MANO.MEAN_PARAMS=str(data/'data/mano_mean_params.npz');cfg.freeze()
    checkpoint=torch.load(str(data/'hamer_ckpts/checkpoints/hamer.ckpt'),map_location='cpu',mmap=True)
    state={k[len('mano_head.'):]:v for k,v in checkpoint['state_dict'].items()if k.startswith('mano_head.')}
    head=build_mano_head(cfg);head.load_state_dict(state,strict=True,assign=True)
    # Match existing cached-feature experiments: inference quantizes Linear
    # weights to fp16, then the differentiable decoder is promoted to fp32.
    for layer in head.modules():
        if isinstance(layer,(torch.nn.Linear,torch.nn.Conv2d)):layer.half()
    head=head.float().cuda().eval()
    del checkpoint,state;gc.collect()
    for p in head.parameters():p.requires_grad_(False)
    return head


def fit(args):
    tr=load_pack(args.data,'train');va=load_pack(args.data,'validation');aug=load_aug(args)
    valid=np.flatnonzero(tr['valid']);n=len(valid)
    head=original_head()
    initial={k:v.detach().cpu().clone()for k,v in head.state_dict().items()}
    base_train,_=predict(head,None,tr);base_val,_=predict(head,None,va)
    zero=int(valid[0])
    target=torch.from_numpy((base_train[zero]@MOUNT.T@tr['imu'][zero].T@tr['imu']@MOUNT).astype(np.float32))
    aug_target=aug['source_to_virtual']@target[aug['source_index']]
    protocol=dict(question='Does known-camera-rotation augmentation improve position robustness beyond identical single-zero IMU supervision?',
        input='Original image features and source-camera crop-ray encoding. No online IMU, timestamp, action label or intended motion.',
        training_source=str(args.data),target='One original C anchor plus measured relative IMU rotation, fixed existing nominal mounting. Augmented target is S times that camera-frame pose.',
        changed_between_methods='Only the extra training views: identical source image repeated versus image, camera rays and orientation consistently transformed by known camera rotation.',
        augmentation='Camera X/Y rotations of +/-12 degrees; crops must fit original and virtual 640x480 images. No new surfaces synthesized.',
        augmented_inputs=len(aug['features']),augmented_source_images=len(torch.unique(aug['source_index'])),
        samples_per_epoch=2*n,original_samples=n,extra_samples=n,epochs=args.epochs,seed=21,
        trainable='Original decoder transformer and first six global-orientation readout rows plus camera adapter. Original articulation, shape and backbone frozen.',
        adapter_lr=.001,head_lr=.00001,loss='Camera-frame palm rotation matrix MSE plus 1e-4 adapter bias squared mean.',
        selection='Lowest validation single-C full rotation mean error, including epoch zero. Same metric and budget for both methods.',
        scope='Development data already inspected. Not a blind new test; no production deployment.')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary=dict(protocol=protocol,baseline_validation=score(base_val,va),methods={})
    reflection=torch.tensor(M,dtype=torch.float32,device='cuda')
    for name,augment in [('imu_only',False),('imu_geometry',True)]:
        head.load_state_dict(initial);params,handles=enable_orientation_training(head)
        torch.manual_seed(21);np.random.seed(21)
        adapter=CameraAdapter(True).cuda();adapter.train()
        opt=torch.optim.Adam([dict(params=adapter.parameters(),lr=.001),dict(params=params,lr=.00001)])
        history=[];best=float('inf');best_epoch=None
        for epoch in range(args.epochs+1):
            started=time.perf_counter();losses=[]
            if epoch:
                extra=np.random.choice(len(aug['features']),n,replace=len(aug['features'])<n)
                source=aug['source_index'][extra]
                extra_data=aug if augment else tr['data'];extra_ix=extra if augment else source
                # Both methods see exactly the same source-image visits and order.
                batch_data={k:torch.cat([tr['data'][k][valid],extra_data[k][extra_ix]])for k in ['features','encoding','local_factor']}
                targets=torch.cat([target[valid],aug_target[extra]if augment else target[source]])
                order=np.random.permutation(2*n)
                for first in range(0,len(order),8):
                    ix=order[first:first+8];x=batch_data['features'][ix].cuda().float();enc=batch_data['encoding'][ix].cuda()
                    x,bias=adapter(x,enc);g=head(x)[0]['global_orient'][:,0]
                    p=reflection@g@batch_data['local_factor'][ix].cuda()
                    loss=(p-targets[ix].cuda()).square().mean()+1e-4*bias.square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
                del batch_data,targets
            pred,_=predict(head,adapter,va);metric=score(pred,va);value=metric['single_zero_full_mae_deg']
            history.append(dict(epoch=epoch,loss=float(np.mean(losses))if losses else None,validation=metric,seconds=time.perf_counter()-started))
            if value<best:
                best=value;best_epoch=epoch
                torch.save(dict(head=head.state_dict(),adapter=adapter.state_dict(),epoch=epoch,protocol=protocol),args.out/f'{name}.pt')
            (args.out/f'{name}_history.json').write_text(json.dumps(history,indent=2))
            print(name,'epoch',epoch,'validation same C',round(value,3),'best',best_epoch,'seconds',round(time.perf_counter()-started,1),flush=True)
        for handle in handles:handle.remove()
        if not torch.equal(head.decpose.weight[6:].detach().cpu(),initial['decpose.weight'][6:]):
            raise RuntimeError('Finger-pose readout weights changed')
        summary['methods'][name]=dict(selected_epoch=best_epoch,validation=history[best_epoch]['validation'])
        del opt,params
    (args.out/'fit_metrics.json').write_text(json.dumps(summary,indent=2))


def current_pack(args):
    packs=[load_pack(args.data,s)for s in ['train','validation','test']]
    meta=sum([p['meta']for p in packs],[]);keep={}
    for i,row in enumerate(meta):keep.setdefault(row['seq'],i)
    indices=sorted(keep.values(),key=lambda i:meta[i]['t'])
    return dict(meta=[meta[i]for i in indices],data={k:torch.cat([p['data'][k]for p in packs])[indices]for k in packs[0]['data']},
        imu=np.concatenate([p['imu']for p in packs])[indices],valid=np.concatenate([p['valid']for p in packs])[indices])


def evaluate(args):
    runner=model();head=runner._model.mano_head.float().eval()
    for p in runner._model.parameters():p.requires_grad_(False)
    test=load_pack(args.data,'test')
    membership=test['valid']&np.array([r.get('role')!='calibration'for r in test['meta']])
    view_data,view_rows,rejections=collect_views(runner,test['meta'],membership)
    virtual_pack=dict(data={k:torch.from_numpy(v)for k,v in view_data.items()},meta=view_rows)
    (args.out/'test_geometry_membership.json').write_text(json.dumps(dict(rows=view_rows,exclusions=rejections),indent=2))
    del runner;gc.collect();torch.cuda.empty_cache()
    models={'baseline':(head,None)}
    for name in ['imu_only','imu_geometry']:
        checkpoint=torch.load(args.out/f'{name}.pt',map_location='cpu')
        adapted=copy.deepcopy(head);adapted.load_state_dict(checkpoint['head']);adapted.eval()
        adapter=CameraAdapter(True).cuda();adapter.load_state_dict(checkpoint['adapter']);adapter.eval()
        models[name]=(adapted,adapter)
    consistency={};control_predictions={}
    for name,(h,adapter)in models.items():
        original,_=predict(h,adapter,test);virtual,_=predict(h,adapter,virtual_pack)
        restored=np.swapaxes(view_data['source_to_virtual'],1,2)@virtual
        diff=np.rad2deg(R.from_matrix(np.swapaxes(original[view_data['source_index']],1,2)@restored).magnitude())
        consistency[name]=dict(overall=stats(diff),by_axis={axis:stats(diff[np.array([r['axis']==axis for r in view_rows])])for axis in ['x','y']})
        control_predictions[name]=virtual
        print('Known-camera consistency',name,consistency[name]['overall'],flush=True)
    (args.out/'test_geometry_consistency.json').write_text(json.dumps(consistency,indent=2))
    np.savez_compressed(args.out/'test_geometry_predictions.npz',**control_predictions,
        source_index=view_data['source_index'],source_to_virtual=view_data['source_to_virtual'])
    del test,virtual_pack,view_data,control_predictions
    summary={}
    for source in ['current','annotated']:
        pack=current_pack(args)if source=='current'else load_pack(ROOT/'.runtime/hamer_camera_adapter_20260921','train')
        if source=='annotated':pack['meta'][0]['role']='calibration'
        times=np.array([x['t']for x in pack['meta']]);valid=pack['valid'];imu=pack['imu']
        rays=np.array([np.degrees(np.arctan(((np.array(r['bbox'][:2])+r['bbox'][2:])/2-K[:2,2])/np.array([K[0,0],K[1,1]])))for r in pack['meta']])
        positions=rays[:,0];predictions={};out={}
        for name,(h,adapter)in models.items():
            pred,_=predict(h,adapter,pack);predictions[name]=pred;item=dict(overall=score(pred,pack))
            zero=int(np.flatnonzero(valid)[0]);indices=np.flatnonzero(valid);indices=indices[indices!=zero]
            error=relative_errors(pred,imu,zero,indices);item['position_groups']=grouped(error,positions[indices])
            item['sides']={s:grouped({k:v[mask]for k,v in error.items()},positions[indices][mask])for s,mask in [('negative',positions[indices]<0),('positive',positions[indices]>=0)]}
            if source=='current':
                item['temporal_partitions']={}
                for split in ['train','validation','test']:
                    selected=np.array([i for i,r in enumerate(pack['meta'])if r['split']==split or r.get('role')=='calibration'])
                    part=dict(meta=[pack['meta'][i]for i in selected],imu=imu[selected],valid=valid[selected])
                    item['temporal_partitions'][split]=score(pred[selected],part)
            else:
                ix=np.flatnonzero(valid&(times>=54)&(times<110));zero=ix[0];ix=ix[1:]
                item['horizontal_block']=grouped(relative_errors(pred,imu,zero,ix),positions[ix])
                pieces={};segments=[]
                anns=json.loads((ROOT/'.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent/annotations.json').read_text())
                for ann in anns:
                    ix=np.flatnonzero(valid&(times>=ann['start'])&(times<ann['end']));s=dict(id=ann['id'],kind=ann['kind'],frames=len(ix))
                    if len(ix)>1:
                        e=relative_errors(pred,imu,ix[0],ix[1:]);pieces.setdefault(ann['kind'],[]).append(e)
                        s.update(full_error=stats(e['full_error']),angle_error=stats(e['angle_error']))
                    segments.append(s)
                item['segments']=segments;item['actions']={}
                for kind,parts in pieces.items():
                    e={k:np.concatenate([p[k]for p in parts])for k in parts[0]};large=e['reference_angle']>10
                    ratio=np.sum(e['predicted_vector'][large]*e['reference_vector'][large],axis=1)/e['reference_angle'][large]**2
                    item['actions'][kind]=dict(full_error=stats(e['full_error']),angle_error=stats(e['angle_error']),
                        large_reference_frames=int(large.sum()),amplitude_ratio_mean=float((e['predicted_angle'][large]/e['reference_angle'][large]).mean())if large.any()else None,
                        projected_ratio_mean=float(ratio.mean())if len(ratio)else None)
            out[name]=item
            print(source,name,'same C',round(item['overall']['single_zero_full_mae_deg'],3),'edge',round(item['position_groups']['15-90']['full_error']['mean_deg'],3),flush=True)
        np.savez_compressed(args.out/f'{source}_predictions.npz',**predictions,t=times,imu=imu,valid=valid,view_angles_deg=rays)
        summary[source]=out
        del pack,predictions
    (args.out/'evaluation_metrics.json').write_text(json.dumps(summary,indent=2))


def native(args):
    from evaluate_trainable_orientation import native_score
    pack=load_pack(ROOT/'.runtime/hamer_trainable_orientation_20260921/native_data','native')
    head=original_head();initial={k:v.detach().cpu().clone()for k,v in head.state_dict().items()}
    result={};predictions={}
    for name in ['baseline','imu_only','imu_geometry']:
        adapter=None
        if name=='baseline':head.load_state_dict(initial)
        else:
            checkpoint=torch.load(args.out/f'{name}.pt',map_location='cpu')
            head.load_state_dict(checkpoint['head']);adapter=CameraAdapter(True).cuda()
            adapter.load_state_dict(checkpoint['adapter']);adapter.eval();del checkpoint
        pred,_=predict(head,adapter,pack);predictions[name]=pred;result[name]=native_score(pred,pack)
        print(name,json.dumps(result[name]),flush=True)
    np.savez_compressed(args.out/'native_predictions.npz',**predictions,t=[r['t']for r in pack['meta']],imu=pack['imu'],valid=pack['valid'])
    (args.out/'native_metrics.json').write_text(json.dumps(result,indent=2))


def residual_features(poses,rays,conditioned):
    # A camera-frame rotation correction that is identically zero on the
    # optical axis. No pose/direction locking, region switch or time input.
    state=np.c_[np.ones(len(poses)),poses.reshape(-1,9)]if conditioned else np.ones((len(poses),1))
    return (rays[:,:,None]*state[:,None,:]).reshape(len(poses),-1)


def bbox_rays(metadata,transforms=None):
    rays=[]
    for i,row in enumerate(metadata):
        bb=np.asarray(row['bbox'])
        if transforms is not None:bb=bounds(project(corners(bb),K@transforms[i]@KINV))
        rays.append(np.arctan(((bb[:2]+bb[2:])/2-K[:2,2])/np.array([K[0,0],K[1,1]])))
    return np.asarray(rays)


def residual(args):
    from evaluate_trainable_orientation import native_score
    from scipy.optimize import least_squares
    trmeta=json.loads((args.data/'train_metadata.json').read_text());imu,valid=reference(trmeta)
    current=np.load(args.out/'current_predictions.npz');times=current['t']
    indices=np.array([int(np.argmin(abs(times-r['t'])))for r in trmeta]);raw=current['baseline'][indices]
    target=raw[0]@MOUNT.T@imu[0].T@imu@MOUNT
    aug=load_aug(args);augmeta=json.loads((args.out/'augmentation_metadata.json').read_text());head=original_head()
    ar,_=predict(head,None,dict(data=aug,meta=augmeta));sources=aug['source_index'].numpy();S=aug['source_to_virtual'].numpy()
    at=S@target[sources];rays=bbox_rays(trmeta);aray=bbox_rays([trmeta[i]for i in sources],S)
    all_raw=np.concatenate([raw[valid],ar]);all_target=np.concatenate([target[valid],at]);all_rays=np.concatenate([rays[valid],aray])
    y=R.from_matrix(all_target@np.swapaxes(all_raw,1,2)).as_rotvec()
    weights=np.r_[np.full(valid.sum(),.5/valid.sum()),np.full(len(ar),.5/len(ar))]
    protocol=dict(question='Can an independent spatial correction preserve the original central-view pose path while reducing edge drift?',
        inputs='Two camera ray angles and optional original palm orientation matrix. Original HaMeR frozen; no online IMU, action label or time.',
        equation='P_corrected=Exp(B(ray_x,ray_y,original_pose)) P_raw; correction vanishes exactly at ray=(0,0).',
        model='Weighted linear ridge fit of camera-frame rotation residual. Constant control has 6 coefficients; pose-conditioned correction has 60.',
        loss='Original training inputs and known-camera-rotation training views each have total weight 0.5. Feature RMS normalization without mean subtraction or intercept.',
        ridge=.01,selection='One prespecified regularization, no sweep, no fitting or selection on validation, test, or the other recording.',
        calibration_follow_up='Also evaluate the same six-coefficient model fitted directly to rotations relative to its corrected C pose. This is a declared development follow-up, not a new blind test.',
        reference='Existing provisional IMU and fixed nominal mounting, same C anchor as the preceding matched experiment.',
        scope='Development follow-up after observed central-view regression; not a blind confirmatory test or production deployment.')
    (args.out/'residual_protocol.json').write_text(json.dumps(protocol,indent=2))
    coefficients={};fit_info={}
    for name,cond in [('ray_constant',False),('ray_pose',True)]:
        x=residual_features(all_raw,all_rays,cond);scale=np.sqrt((weights[:,None]*x*x).sum(0)).clip(1e-6);xs=x/scale
        gram=xs.T@(weights[:,None]*xs);coef=np.linalg.solve(gram+.01*np.eye(len(scale)),xs.T@(weights[:,None]*y))/scale[:,None]
        coefficients[name]=coef;fit_info[name]=dict(coefficients=coef.tolist(),conditioned=cond,
            training_residual_rmse_deg=float(np.rad2deg(np.sqrt((weights[:,None]*(x@coef-y)**2).sum()))))
    # The spatial model generally also corrects the original C image. Fit
    # measured relative rotations directly so that correction is included in
    # the reference, instead of forcing its absolute pose to remain uncorrected.
    reference_relative=MOUNT.T@imu[0].T@imu@MOUNT
    desired=np.concatenate([reference_relative[valid],reference_relative[sources]])
    restore=np.concatenate([np.repeat(np.eye(3)[None],valid.sum(),axis=0),np.swapaxes(S,1,2)])
    scale=np.sqrt((weights[:,None]*all_rays**2).sum(0))
    def objective(flat):
        b=flat.reshape(2,3);zero=R.from_rotvec(rays[0]@b).as_matrix()@raw[0]
        corrected=R.from_rotvec(all_rays@b).as_matrix()@all_raw
        relative=zero.T@restore@corrected
        e=R.from_matrix(np.swapaxes(desired,1,2)@relative).as_rotvec()
        return np.r_[(e*np.sqrt(weights[:,None])).ravel(),(.1*scale[:,None]*b).ravel()]
    fitted=least_squares(objective,np.zeros(6),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    coefficients['ray_relative']=fitted.x.reshape(2,3)
    fit_info['ray_relative']=dict(coefficients=coefficients['ray_relative'].tolist(),conditioned=False,
        objective='Same six coefficients fitted to full rotations relative to the corrected C pose, with the same original/augmented weights and regularization; no temporal fitting inputs at inference.',
        training_objective=float(np.sum(fitted.fun**2)),optimization_termination=str(fitted.message))
    (args.out/'residual_coefficients.json').write_text(json.dumps(fit_info,indent=2))
    del aug,head;gc.collect();torch.cuda.empty_cache()
    def correct(name,p,r):return R.from_rotvec(residual_features(p,r,name=='ray_pose')@coefficients[name]).as_matrix()@p
    summary={};anns=json.loads((ROOT/'.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent/annotations.json').read_text())
    for source in ['current','annotated','native']:
        z=np.load(args.out/f'{source}_predictions.npz');t=z['t'];ii=z['imu'];vv=z['valid'];p=z['baseline']
        if source=='native':
            meta=json.loads((ROOT/'.runtime/hamer_trainable_orientation_20260921/native_data/native_metadata.json').read_text());rr=bbox_rays(meta)
        else:rr=np.radians(z['view_angles_deg'])
        poses={'baseline':p,**{name:correct(name,p,rr)for name in coefficients}};out={}
        for name,pp in poses.items():
            if source=='native':out[name]=native_score(pp,dict(meta=meta,imu=ii,valid=vv));continue
            ix=np.flatnonzero(vv);origin=ix[0];ix=ix[1:];e=relative_errors(pp,ii,origin,ix)
            item=dict(single_C=dict(full_error=stats(e['full_error']),angle_error=stats(e['angle_error'])),position_groups=grouped(e,np.degrees(rr[ix,0])))
            if source=='current':
                item['temporal_partitions']={}
                for part,lo,hi in [('train',74.9,296.701),('validation',302.701,371.629),('test',377.628,450)]:
                    k=np.flatnonzero(vv&(t>=lo)&(t<hi)&(np.arange(len(t))!=origin));e=relative_errors(pp,ii,origin,k)
                    item['temporal_partitions'][part]={f:stats(e[f])for f in ['full_error','angle_error']}
            else:
                k=np.flatnonzero(vv&(t>=54)&(t<110));a=k[0];k=k[1:];e=relative_errors(pp,ii,a,k)
                item['horizontal_block']=grouped(e,np.degrees(rr[k,0]));item['horizontal_sides']={side:grouped({f:v[mask]for f,v in e.items()},np.degrees(rr[k,0][mask]))for side,mask in [('negative',rr[k,0]<0),('positive',rr[k,0]>=0)]}
                groups={}
                for ann in anns:
                    k=np.flatnonzero(vv&(t>=ann['start'])&(t<ann['end']))
                    if len(k)>1:groups.setdefault(ann['kind'],[]).append(relative_errors(pp,ii,k[0],k[1:]))
                item['actions']={}
                for kind,parts in groups.items():
                    e={f:np.concatenate([part[f]for part in parts])for f in parts[0]};large=e['reference_angle']>10
                    item['actions'][kind]=dict(full_error=stats(e['full_error']),angle_error=stats(e['angle_error']),
                        amplitude_ratio_mean=float((e['predicted_angle'][large]/e['reference_angle'][large]).mean())if large.any()else None)
            out[name]=item
        np.savez_compressed(args.out/f'residual_{source}_predictions.npz',**poses,t=t,imu=ii,valid=vv,rays=rr)
        summary[source]=out;print('Spatial residual',source,json.dumps({n:v.get('single_C')for n,v in out.items()}),flush=True)
    (args.out/'residual_metrics.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['cache','fit','evaluate','native','residual'])
    parser.add_argument('--data',type=Path,default=ROOT/'.runtime/hamer_camera_adapter_current_20260921')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    {'cache':cache,'fit':fit,'evaluate':evaluate,'native':native,'residual':residual}[args.mode](args)
