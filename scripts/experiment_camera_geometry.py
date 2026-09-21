#!/usr/bin/env python3
"""No-training, RGB-only tests of off-axis image geometry.

Mechanism: rotate a virtual camera about its optical center using the exact
pinhole homography on a fixed real RGB image, and undo its known rotation on
predicted poses. Compare against a pixel-translation control. This does not
synthesize the newly visible surfaces caused by physically translating a hand.
Replay: ordinary crop versus calibrated perspective crop and paired output
rotation, on actual source images. No IMU fitting, filtering or robot commands.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from experiment_hamer_camera_adapter import ROOT, K, M, MOUNT, model, reference
from perception_hamer.src.hamer_crop_inference import prepare_hamer_crop, validate_rotation_matrices
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame

KINV = np.linalg.inv(K)


def corners(bb):
    x1,y1,x2,y2=bb
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float64)


def project(points,H):
    v=np.c_[points,np.ones(len(points))]@H.T
    return v[:,:2]/v[:,2:]


def bounds(points):
    return np.r_[points.min(0),points.max(0)]


def look_at(pixel):
    # Camera-from-virtual rotation; same zero-roll convention as PCL authors.
    x,y=(np.asarray(pixel)-K[:2,2])/np.array([K[0,0],K[1,1]])
    nx=np.sqrt(1+x*x);n=np.sqrt(1+x*x+y*y)
    return np.array([[1/nx,-x*y/(nx*n),x/n],[0,nx/n,y/n],[-x/nx,-y/(nx*n),1/n]])


def affine(shape,bb,runner):
    _,a,quality=prepare_hamer_crop(np.zeros(shape,dtype=np.uint8),bb,False,
        image_size=int(runner._cfg.MODEL.IMAGE_SIZE),bbox_shape=tuple(runner._cfg.MODEL.get('BBOX_SHAPE',[192,256])),
        rescale_factor=runner.rescale_factor,image_mean=tuple(runner._cfg.MODEL.IMAGE_MEAN),image_std=tuple(runner._cfg.MODEL.IMAGE_STD))
    return np.vstack([a,[0,0,1]]),quality


def coverage(shape,H):
    yy,xx=np.mgrid[0:256,32:224];p=np.c_[xx.ravel(),yy.ravel()]
    source=project(p,np.linalg.inv(H))
    ok=(source[:,0]>=0)&(source[:,0]<=shape[1]-1)&(source[:,1]>=0)&(source[:,1]<=shape[0]-1)
    return float(ok.mean())


def frame_outputs(runner,rgb,H,restore,betas,base_size):
    # Common anti-aliasing rule for both crop types; exactly one resampling.
    factor=base_size/256/2
    if factor>1.1:rgb=cv2.GaussianBlur(rgb,(0,0),sigmaX=(factor-1)/2,sigmaY=(factor-1)/2)
    patch=cv2.warpPerspective(rgb,H,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(np.float32)
    mean=255*np.array(runner._cfg.MODEL.IMAGE_MEAN,dtype=np.float32)
    std=255*np.array(runner._cfg.MODEL.IMAGE_STD,dtype=np.float32)
    image=np.ascontiguousarray(((patch-mean)/std).transpose(2,0,1))
    runner.set_frozen_betas(False,np.asarray(betas))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        out=runner._forward_with_frozen_betas({'img':torch.from_numpy(image[None]).cuda()},False)
    pts=out['pred_keypoints_3d'][0].detach().float().cpu().numpy()@M
    palm=build_hamer_five_point_palm_frame(pts,False)
    if not palm.valid:raise ValueError(palm.failure_reason)
    g=validate_rotation_matrices(out['pred_mano_params']['global_orient'][0,0].detach().float().cpu().numpy(),(3,3),'global')
    return dict(palm=(restore@palm.rotation).tolist(),global_rotation=(restore@M@g@M).tolist(),
        source_coverage=coverage(rgb.shape,H))


def geometry_transforms(row,shape,runner):
    bb=np.asarray(row['bbox']);a,quality=affine(shape,bb,runner)
    q=look_at((bb[:2]+bb[2:])/2);h=K@q.T@KINV
    polygon=project(corners(bb),h);av,_=affine(shape,bounds(polygon),runner)
    return a,av@h,q,quality['crop_bbox_size_px']


def read_rgb(row):
    p=Path(row['file'])
    return np.load(p)['rgb']if p.suffix=='.npz'else cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)


def current_rows():
    rows=json.loads((ROOT/'.runtime/hamer_camera_adapter_current_20260921/source_manifest.json').read_text())
    return sorted({x['seq']:x for x in rows}.values(),key=lambda x:x['t'])


def angle(a,b):
    return float(np.rad2deg(R.from_matrix(np.asarray(a).T@np.asarray(b)).magnitude()))


def mechanism(args,runner):
    rows=current_rows();candidates=[];shape=(480,640,3)
    # Membership uses geometry and time, never the model's errors or IMU.
    for row in rows:
        bb=np.array(row['bbox']);c=(bb[:2]+bb[2:])/2
        ray=np.degrees(np.arctan((c-K[:2,2])/np.array([K[0,0],K[1,1]])))
        if np.max(abs(ray))>8:continue
        a,quality=affine(shape,bb,runner)
        if coverage(shape,a)<1:continue
        variants=[]
        for beta in [-20,-10,0,10,20]:
            s=R.from_euler('y',beta-ray[0],degrees=True).as_matrix();h=K@s@KINV
            boxv=bounds(project(corners(bb),h));av,_=affine(shape,boxv,runner)
            translation=np.eye(3);translation[0,2]=K[0,2]+K[0,0]*np.tan(np.deg2rad(beta))-c[0]
            boxt=bb+np.array([translation[0,2],0,translation[0,2],0]);at,_=affine(shape,boxt,runner)
            if min(coverage(shape,av@h),coverage(shape,at@translation))<1:break
            variants.append((beta,s,av@h,at@translation))
        if len(variants)==5:candidates.append((row,quality['crop_bbox_size_px'],variants))
    selected=[]
    for target in np.linspace(rows[0]['t']+10,rows[-1]['t']-10,12):
        unused=[x for x in candidates if all(abs(x[0]['t']-y[0]['t'])>5 for y in selected)]
        if unused:selected.append(min(unused,key=lambda x:abs(x[0]['t']-target)))
    if len(selected)<4:raise ValueError('Too few fully supported real-image controls')
    protocol=dict(scope='Mechanistic real-image camera-rotation test, not physical hand translation or a pose-accuracy benchmark.',
        controls='Same original RGB, same frozen hand shape/model, fixed left, one image resampling, no missing source pixels. 2D pixel translation as control.',
        geometry='Known virtual camera yaw about optical center, H=K S K^-1; hand world pose unchanged, predicted poses restored by S^T.',
        boundary='Virtual full-frame clipping is deliberately absent; crops sample only pixels available in the saved original image. This isolates geometry from frame clipping/black fill. No newly visible surface is synthesized.',
        selection='12 time-spread geometrically eligible images, center ray within +/-8 degrees, all tested crops fully supported by source pixels; no error/IMU selection.',
        rows=[x[0]for x in selected])
    (args.out/'mechanism_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    results=[];identities=[]
    for i,(row,size,variants)in enumerate(selected):
        rgb=read_rgb(row);a,_=affine(rgb.shape,row['bbox'],runner)
        runner.set_frozen_betas(False,np.asarray(row['betas']))
        original=runner.infer(rgb,row['bbox'],False,row['stamp'])
        fp=build_hamer_five_point_palm_frame(original.pred_keypoints_3d_source_camera_axes,False)
        if not fp.valid:raise ValueError(fp.failure_reason)
        identity=frame_outputs(runner,rgb,a,np.eye(3),row['betas'],size)
        identities.append(angle(fp.rotation,identity['palm']))
        subset=[]
        for beta,s,H,Ht in variants:
            subset.append(dict(seq=row['seq'],t=row['t'],beta_deg=beta,
                camera_rotation=frame_outputs(runner,rgb,H,s.T,row['betas'],size),
                pixel_translation=frame_outputs(runner,rgb,Ht,np.eye(3),row['betas'],size)))
        zero=next(x for x in subset if x['beta_deg']==0)
        for item in subset:
            for method in ['camera_rotation','pixel_translation']:
                item[method]['palm_difference_deg']=angle(zero[method]['palm'],item[method]['palm'])
                item[method]['global_difference_deg']=angle(zero[method]['global_rotation'],item[method]['global_rotation'])
        results.extend(subset);print('Mechanism real image',i+1,'/',len(selected),flush=True)
    summary=dict(images=len(selected),identity_vs_existing_crop_deg=dict(mean=float(np.mean(identities)),max=float(max(identities))),views={})
    for beta in [-20,-10,0,10,20]:
        sub=[x for x in results if x['beta_deg']==beta];summary['views'][str(beta)]={}
        for method in ['camera_rotation','pixel_translation']:
            v=[x[method]['palm_difference_deg']for x in sub];g=[x[method]['global_difference_deg']for x in sub]
            summary['views'][str(beta)][method]=dict(mean_deg=float(np.mean(v)),p95_deg=float(np.percentile(v,95)),max_deg=float(max(v)),native_global_mean_deg=float(np.mean(g)))
    (args.out/'mechanism_predictions.json').write_text(json.dumps(results,indent=2))
    (args.out/'mechanism_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


def replay(args,runner):
    sources={'current':current_rows(),'annotated':json.loads((ROOT/'.runtime/hamer_camera_adapter_20260921/train_metadata.json').read_text())}
    for name,rows in sources.items():
        results=[];failures=[]
        for index,row in enumerate(rows):
            rgb=read_rgb(row);a,h,q,size=geometry_transforms(row,rgb.shape,runner)
            bb=np.array(row['bbox']);c=(bb[:2]+bb[2:])/2
            item=dict(seq=row['seq'],t=row['t'],bbox=row['bbox'],view_angles_deg=np.degrees(np.arctan((c-K[:2,2])/np.array([K[0,0],K[1,1]]))).tolist(),methods={})
            for method,H,restore in [('baseline',a,np.eye(3)),('perspective',h,q)]:
                try:item['methods'][method]=frame_outputs(runner,rgb,H,restore,row['betas'],size)
                except (ValueError,RuntimeError)as exc:
                    failures.append(dict(source=name,seq=row['seq'],method=method,error=str(exc)))
                    if isinstance(exc,torch.cuda.OutOfMemoryError):raise
            results.append(item)
            if index%100==0:print('Geometric replay',name,index,'/',len(rows),flush=True)
        iq,valid=reference(rows)
        (args.out/f'{name}_predictions.json').write_text(json.dumps(results,indent=2))
        (args.out/f'{name}_failures.json').write_text(json.dumps(failures,indent=2))
        np.savez_compressed(args.out/f'{name}_reference.npz',t=[x['t']for x in rows],imu=iq,valid=valid)
        print('Geometric replay complete',name,len(rows),'failures',len(failures),flush=True)


def stats(x):
    x=np.asarray(x)
    return dict(n=len(x),mean_deg=float(x.mean()),p95_deg=float(np.percentile(x,95)),max_deg=float(x.max())) if len(x) else dict(n=0)


def relative_errors(pred,imu,origin,indices):
    ir=imu[origin].T@imu[indices]
    pr=MOUNT@pred[origin].T@pred[indices]@MOUNT.T
    iv=np.rad2deg(R.from_matrix(ir).as_rotvec());pv=np.rad2deg(R.from_matrix(pr).as_rotvec())
    ia=np.linalg.norm(iv,axis=-1);pa=np.linalg.norm(pv,axis=-1)
    error=np.rad2deg(R.from_matrix(np.swapaxes(ir,1,2)@pr).magnitude())
    return dict(full_error=error,angle_error=abs(pa-ia),signed_angle_error=pa-ia,
        reference_angle=ia,predicted_angle=pa,reference_vector=iv,predicted_vector=pv)


def grouped(e,position):
    result={}
    for lo,hi in [(0,5),(5,10),(10,15),(15,90)]:
        ix=(abs(position)>=lo)&(abs(position)<hi)
        result[f'{lo}-{hi}']=dict(full_error=stats(e['full_error'][ix]),angle_error=stats(e['angle_error'][ix]))
    c=abs(position)<5;edge=abs(position)>=15
    result['edge_minus_center_deg']={k:float(e[k][edge].mean()-e[k][c].mean())if edge.any()and c.any()else None for k in ['full_error','angle_error']}
    return result


def analyze(args):
    results={};frame_rows=[]
    anns=json.loads((ROOT/'.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent/annotations.json').read_text())
    for name in ['current','annotated']:
        rows=json.loads((args.out/f'{name}_predictions.json').read_text());ref=np.load(args.out/f'{name}_reference.npz')
        t=ref['t'];imu=ref['imu'];valid=ref['valid'].copy()
        valid &= np.array([all(m in r['methods']for m in ['baseline','perspective'])for r in rows])
        # Matrix arrays retain every input row. A missing prediction is not an identity estimate.
        predictions={m:np.array([r['methods'].get(m,{}).get('palm',np.full((3,3),np.nan))for r in rows])for m in ['baseline','perspective']}
        x=np.array([r['view_angles_deg'][0]for r in rows]);y=np.array([r['view_angles_deg'][1]for r in rows])
        item=dict(input_frames=len(t),paired_valid_frames=int(valid.sum()),t_range_s=[float(t[0]),float(t[-1])],methods={})
        for m,pred in predictions.items():
            zero=int(np.flatnonzero(valid)[0]);ix=np.flatnonzero(valid);ix=ix[ix!=zero]
            err=relative_errors(pred,imu,zero,ix)
            cov=np.array([r['methods'].get(m,{}).get('source_coverage',0)for r in rows])
            method=dict(single_zero=dict(origin_s=float(t[zero]),full_error=stats(err['full_error']),angle_error=stats(err['angle_error'])),
                source_coverage=dict(min=float(cov.min()),mean=float(cov.mean()),fully_supported_frames=int((cov==1).sum())),
                single_zero_horizontal_position_groups=grouped(err,x[ix]))
            if name=='annotated':
                origin=int(np.flatnonzero(valid&(t>=54)&(t<110))[0]);idx=np.flatnonzero(valid&(t>=54)&(t<110))
                idx=idx[idx!=origin];e=relative_errors(pred,imu,origin,idx)
                method['horizontal_block']=dict(origin_s=float(t[origin]),groups=grouped(e,x[idx]),sides={})
                for side,mask in [('negative',x[idx]<0),('positive',x[idx]>=0)]:
                    method['horizontal_block']['sides'][side]=grouped({k:v[mask]for k,v in e.items()},x[idx][mask])
                for j,k in enumerate(idx):
                    frame_rows.append(dict(source=name,method=m,t_s=float(t[k]),origin_s=float(t[origin]),view_x_deg=float(x[k]),view_y_deg=float(y[k]),
                        source_coverage=float(cov[k]),**{key:float(e[key][j])for key in ['full_error','angle_error','signed_angle_error','reference_angle','predicted_angle']}))
                segments=[];aggregates={}
                for ann in anns:
                    ix=np.flatnonzero(valid&(t>=ann['start'])&(t<ann['end']))
                    segment=dict(annotation_id=ann['id'],kind=ann['kind'],start=ann['start'],end=ann['end'],frames=len(ix))
                    if len(ix)>1:
                        a,b=ix[0],ix[1:];e=relative_errors(pred,imu,a,b)
                        segment.update(origin_s=float(t[a]),full_error=stats(e['full_error']),angle_error=stats(e['angle_error']),
                            reference_peak_deg=float(e['reference_angle'].max()),predicted_peak_deg=float(e['predicted_angle'].max()))
                        aggregates.setdefault(ann['kind'],[]).append(e)
                    segments.append(segment)
                method['segments']=segments;method['by_action']={}
                for kind,pieces in aggregates.items():
                    e={k:np.concatenate([p[k]for p in pieces])for k in pieces[0]};large=e['reference_angle']>10
                    ratio=np.sum(e['predicted_vector'][large]*e['reference_vector'][large],axis=1)/e['reference_angle'][large]**2
                    method['by_action'][kind]=dict(segments=len(pieces),full_error=stats(e['full_error']),angle_error=stats(e['angle_error']),
                        large_reference_frames=int(large.sum()),projected_ratio_mean=float(ratio.mean())if len(ratio)else None,
                        amplitude_ratio_mean=float((e['predicted_angle'][large]/e['reference_angle'][large]).mean())if large.any()else None)
            item['methods'][m]=method
        results[name]=item
    (args.out/'replay_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    with (args.out/'horizontal_block.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(frame_rows[0]));w.writeheader();w.writerows(frame_rows)
    for source,v in results.items():
        print(source)
        for name,m in v['methods'].items():
            print(name,'single zero',m['single_zero'])
            if 'horizontal_block'in m:
                print('horizontal groups',m['horizontal_block']['groups'])
                print('actions',m['by_action'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mechanism','replay','analyze']);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='analyze':
        analyze(args)
        raise SystemExit(0)
    cv2.setNumThreads(1);torch.set_num_threads(2);runner=model()
    {'mechanism':mechanism,'replay':replay}[args.mode](args,runner)
