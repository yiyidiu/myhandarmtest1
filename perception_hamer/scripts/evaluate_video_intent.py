#!/usr/bin/env python3
"""Offline test of actual causal video features added to recorded pose/depth.

The official pretrained online TSM encoder is frozen. Its gesture classifier
is removed; independent heads are fitted to this operator's annotations.
Nothing in this script publishes robot commands or opens a live camera.
"""
import argparse
import datetime
import gzip
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate_continuous_intent import ROOT, SOURCE, command_angle_binding, past_features

OUTPUT = ROOT / '.runtime/video_intent_20260913'
COMPONENTS = ['translate', 'pitch', 'roll', 'yaw']


def square_view(rgb, box=None):
    """Keep full-frame position in one stream, hand/forearm detail in another."""
    h, w = rgb.shape[:2]
    if box is None:
        side = max(h, w)
        left, top = (w-side)//2, (h-side)//2
    else:
        x1,y1,x2,y2 = box
        side = int(np.ceil(max(x2-x1, y2-y1)*2.2))
        side = max(64, min(side, 2*max(h,w)))
        left = int((x1+x2-side)/2)
        top = int((y1+y2-side)/2 + .20*(y2-y1))
    canvas = np.zeros((side,side,3), dtype=np.uint8)
    x1,y1,x2,y2 = max(0,left),max(0,top),min(w,left+side),min(h,top+side)
    if x2>x1 and y2>y1:
        canvas[y1-top:y2-top,x1-left:x2-left] = rgb[y1:y2,x1:x2]
    return cv2.resize(canvas,(224,224),interpolation=cv2.INTER_AREA)


def encoder():
    sys.path.insert(0,str(OUTPUT/'temporal-shift-module/online_demo'))
    from mobilenet_v2_tsm import MobileNetV2
    model = MobileNetV2(n_class=27)
    model.load_state_dict(torch.load(OUTPUT/'mobilenetv2_jester_online.pth.tar',map_location='cpu',weights_only=True))
    model.classifier = torch.nn.Identity()
    return model.eval().cuda()


def zero_buffers():
    return [torch.zeros((2,c,h,h),device='cuda')
            for c,h in [(3,56),(4,28),(4,28),(8,14),(8,14),(8,14),(12,14),(12,14),(20,7),(20,7)]]


def stationary_video_features(model,x):
    """Exact steady output for an indefinitely repeated identical image.

    At every temporal-shift block the previous activation then equals the
    current activation. This equals repeating that image through the official
    streaming model until its finite history has been replaced.
    """
    from mobilenet_v2_tsm import InvertedResidualWithShift
    for layer in model.features:
        x=x+layer.conv(x) if isinstance(layer,InvertedResidualWithShift) else layer(x)
    return x.mean(3).mean(2)


def extract(split, limit, static_only=False, full_recording=False):
    import pyrealsense2 as rs
    torch.set_num_threads(2)
    definition = json.loads((SOURCE/'intent_data_definition.json').read_text())
    dataset = np.load(SOURCE/'intent_feature_dataset.npz')
    _,session,folder = next(s for s in definition['sources'] if s[0]==split)
    source = ROOT/'.runtime'/session/folder
    rows = json.load(gzip.open(source/'perception_signals.json.gz','rt'))
    rows = [r for r in rows if r.get('frame_number') and r.get('roi',{}).get('bbox')]
    times = np.asarray([r['t'] for r in rows])
    grid = dataset[split+'_t']
    selected = np.zeros(len(grid),dtype=bool)
    for a in definition['annotations'][split]:
        selected |= (grid>=a['start']-1.3)&(grid<=a['end']+.1)
    if full_recording:
        first_c=next(r['t'] for r in rows if r.get('control_enabled'))
        selected=grid>=first_c-1.3
    indices = np.flatnonzero(selected & (grid>=times[0]))
    if limit:
        indices = indices[:limit]
    pose_index = np.maximum(0,np.searchsorted(times,grid,side='right')-1)
    origin = json.loads((source/'time_origin.json').read_text())['video_wall_origin']
    model = encoder()
    buffers = zero_buffers()
    mean = torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    features = np.full((len(grid),2,1280),np.nan,dtype=np.float16)
    stationary=np.full_like(features,np.nan) if full_recording else None
    source_times = np.full(len(grid),np.nan)
    durations = []
    pipe = rs.pipeline(); config = rs.config()
    config.enable_device_from_file(str(source.parent/'rgbd.db3'),False)
    profile = pipe.start(config); playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    first = pipe.wait_for_frames(10000)
    base = first.get_timestamp()/1000-playback.get_position()/1e9
    playback.pause()
    last_grid = None; last_source = None; rgb = None
    started = time.monotonic()
    print('Extracting',split,len(indices),'causal samples',flush=True)
    try:
        with torch.inference_mode():
            for number,i in enumerate(indices):
                row = rows[pose_index[i]]
                target = origin+row['t']
                if last_grid is None or grid[i]-last_grid>.2:
                    buffers = zero_buffers()
                if last_source != row['frame_number']:
                    if last_grid is None or grid[i]-last_grid>.4:
                        playback.seek(datetime.timedelta(seconds=max(0,target-base-1.)))
                    playback.resume(); found=False
                    for attempt in range(2):
                        for _ in range(150):
                            frame=pipe.wait_for_frames(5000); color=frame.get_color_frame()
                            if not color: continue
                            if color.get_timestamp()/1000>target+.002: break
                            if color.get_frame_number()==row['frame_number'] and abs(color.get_timestamp()/1000-target)<.002:
                                rgb=np.asanyarray(color.get_data()).copy(); found=True; break
                        if found: break
                        playback.pause();playback.seek(datetime.timedelta(seconds=max(0,target-base-1.)));playback.resume()
                    playback.pause()
                    if not found:
                        raise RuntimeError(('exact RGB frame missing',split,row['seq'],row['frame_number'],row['t']))
                    last_source=row['frame_number']
                views=np.stack([square_view(rgb),square_view(rgb,row['roi']['bbox'])])
                x=torch.from_numpy(views.transpose(0,3,1,2).copy()).cuda().float()/255.
                x=(x-mean)/std
                torch.cuda.synchronize();stamp=time.perf_counter()
                if static_only:
                    encoded=stationary_video_features(model,x)
                else:
                    result=model(x,*buffers);buffers=result[1:];encoded=result[0]
                torch.cuda.synchronize();durations.append(time.perf_counter()-stamp)
                features[i]=encoded.cpu().numpy().astype(np.float16)
                if full_recording:
                    stationary[i]=stationary_video_features(model,x).cpu().numpy().astype(np.float16)
                source_times[i]=row['t'];last_grid=grid[i]
                if number==0:
                    cv2.imwrite(str(OUTPUT/(split+'_input_views.jpg')),cv2.cvtColor(np.hstack(views),cv2.COLOR_RGB2BGR))
                if number%200==0:
                    print(split,number,'/',len(indices),'video_s',round(float(grid[i]),2),'elapsed_s',round(time.monotonic()-started,1),flush=True)
    finally:
        pipe.stop()
    suffix='_probe' if limit else ''
    if static_only:suffix+='_stationary'
    if full_recording:suffix+='_full'
    np.savez_compressed(OUTPUT/(split+'_video_features'+suffix+'.npz'),t=grid,image_features=features,
                        source_t=source_times,model_s=durations,
                        **({'stationary_features':stationary} if stationary is not None else {}))
    print('Finished',split,'encoder p50/p95 ms',np.quantile(durations,[.5,.95])*1000,flush=True)


def fit(adapt=False, stationary_negatives=False, motion_contrast=False):
    definition=json.loads((SOURCE/'intent_data_definition.json').read_text())
    data=np.load(SOURCE/'intent_feature_dataset.npz')
    classes=definition['classes']
    binding=np.load(ROOT/'.runtime/continuous_intent_20260912/continuous_signal_binding.npz')
    angles={s:command_angle_binding(data,s,definition) for s in ['train','validation','test']}
    base={};video={};masks={};labels={};stationary_video={}
    adjusted={k:data[k].copy() for k in data.files}
    for s in angles:
        a=angles[s]
        speed=np.vstack([np.zeros(3),np.diff(a['angles'],axis=0)*15])
        speed[np.r_[False,np.diff(a['epoch'])!=0]]=0.
        adjusted[s+'_x'][:,3:6]=speed
        base[s]=np.column_stack([past_features(adjusted,s,SOURCE,True),a['position_delta'],a['angles']])
        cached=np.load(OUTPUT/(s+'_video_features.npz'))
        video[s]=cached['image_features'].astype(np.float32).reshape(len(base[s]),-1)
        if stationary_negatives:
            stationary_video[s]=np.load(OUTPUT/(s+'_video_features_stationary.npz'))['image_features'].astype(np.float32).reshape(len(base[s]),-1)
            if motion_contrast:
                # Ask what the recent images add compared with holding the
                # current image constant. Remove the common posture response
                # before compression; no angle or axis is subtracted here.
                video[s]-=stationary_video[s]
                stationary_video[s]=np.where(np.isfinite(stationary_video[s]),0.,np.nan)
        masks[s]=(data[s+'_y']>=0)&(data[s+'_y']<5)&binding[s+'_valid']&binding[s+'_enabled']&np.isfinite(video[s]).all(1)
        labels[s]=np.column_stack([data[s+'_y']==classes.index(c) for c in COMPONENTS]).astype(int)
    # The same earlier session trains both heads. No latest-feedback adaptation
    # in this experiment; the latest recording remains a development comparison.
    train=masks['train']
    fit_selections=[('train',train)]
    feedback_training_groups=[2,4,6,11,12,14]
    if adapt:
        fit_selections.append(('test',masks['test']&np.isin(data['test_group'],feedback_training_groups)))
    if stationary_negatives:
        # Paired negatives are made only from fitting clips. Other recordings
        # remain comparison data. Keep the same posture/position, but all
        # observations constant after the temporal history has settled.
        originals=list(fit_selections)
        for s,mask in originals:
            name=s+'_stationary';base[name]=np.zeros_like(base[s]);base[name][:,30]=1.
            base[name][:,-6:]=base[s][:,-6:];video[name]=stationary_video[s]
            labels[name]=np.zeros_like(labels[s])
            # Work on a mutable copy only for these explicit synthetic rows.
            if not isinstance(data,dict):data={k:data[k] for k in data.files}
            # Give synthetic stationary examples their own weight bucket so
            # the small set of genuine, noisy stationary observations keeps
            # its original relative importance among the real action types.
            data[name+'_y']=np.full(len(mask),6,dtype=int);data[name+'_group']=data[s+'_group']
            # 5 Hz samples suffice for identical-image states and avoid
            # giving repeated neighbouring images extra weight.
            chosen=mask&np.isfinite(video[name]).all(1)&(np.arange(len(mask))%3==0)
            fit_selections.append((name,chosen))
    fit_labels=np.concatenate([labels[s][mask] for s,mask in fit_selections])
    fit_classes=np.concatenate([data[s+'_y'][mask] for s,mask in fit_selections])
    fit_groups=np.concatenate([data[s+'_group'][mask]+1000*j for j,(s,mask) in enumerate(fit_selections)])
    w=np.zeros(len(fit_classes))
    for label in np.unique(fit_classes):
        groups=np.unique(fit_groups[fit_classes==label])
        for group in groups:
            ids=fit_groups==group;w[ids]=1./(len(groups)*ids.sum())
    w*=len(w)/w.sum()
    join=lambda features:np.concatenate([features[s][mask] for s,mask in fit_selections])
    scaler=StandardScaler().fit(join(video))
    pca=PCA(n_components=64,random_state=37,svd_solver='randomized')
    pca.fit(scaler.transform(join(video)))
    embedded={s:pca.transform(scaler.transform(np.nan_to_num(video[s]))) for s in video}
    adapters=dict(video_mean=scaler.mean_,video_scale=scaler.scale_,pca_mean=pca.mean_,pca_components=pca.components_)
    report=dict(scope=__doc__,live_control_changed=False,
                source_recordings=definition['sources'],components=COMPONENTS,
                data_role='Earlier recording for fitting; second for comparison; latest repeatedly inspected development evaluation, not an untouched final test.',
                predictor='Four independent logistic heads, C=0.1, equal action type and clip weights; no mutually exclusive class.',
                video_encoder='Official online MobileNetV2 TSM, frozen Jester weights; full-frame and expanded hand/forearm views; no future frames.',
                pca_training_only=True,models={})
    report['current_feedback_training_groups']=feedback_training_groups if adapt else []
    report['adapted_to_current_feedback']=adapt
    report['stationary_negative_pairs']=stationary_negatives
    report['motion_contrast']=motion_contrast
    if adapt:
        report['data_role']='Earlier recording plus specified latest-feedback clips fit both models; second complete recording and other latest repeats are comparison data. All are development data.'
    suffix='_adapted' if adapt else ''
    if stationary_negatives:suffix+='_stationary'
    if motion_contrast:suffix+='_contrast'
    for name,features in [('pose_depth_statistics',base),('pose_depth_with_video',{s:np.column_stack([base[s],embedded[s]]) for s in base}),
                          ('video_only',embedded)]:
        normalize=StandardScaler().fit(join(features))
        normalized={s:normalize.transform(x) for s,x in features.items()}
        models=[]
        for axis in range(4):
            model=LogisticRegression(C=.1,max_iter=1500,solver='lbfgs')
            model.fit(join(normalized),fit_labels[:,axis],sample_weight=w)
            models.append(model)
        scores={s:np.column_stack([m.predict_proba(x)[:,1] for m in models]) for s,x in normalized.items()}
        adapters.update({name+'_mean':normalize.mean_,name+'_scale':normalize.scale_,
                         name+'_coef':np.vstack([m.coef_[0] for m in models]),
                         name+'_intercept':np.asarray([m.intercept_[0] for m in models])})
        records={}
        for s in definition['annotations']:
            records[s]=[]
            for annotation in definition['annotations'][s]:
                ids=(data[s+'_group']==annotation['id'])&binding[s+'_valid']&binding[s+'_enabled']&np.isfinite(video[s]).all(1)
                if not ids.any():continue
                a=dict(annotation,n=int(ids.sum()),mean_score=np.mean(scores[s][ids],axis=0).tolist(),
                       active_fraction_at_half=(scores[s][ids]>=.5).mean(0).tolist())
                a['used_for_fitting']=(s=='train' and annotation['label']!='combined') or (adapt and s=='test' and annotation['id'] in feedback_training_groups)
                if name=='pose_depth_with_video':
                    a['mean_logit_contributions']={component:dict(
                        pose_depth=float(np.mean(normalized[s][ids,:base[s].shape[1]]@models[j].coef_[0,:base[s].shape[1]])),
                        video=float(np.mean(normalized[s][ids,base[s].shape[1]:]@models[j].coef_[0,base[s].shape[1]:])),
                        intercept=float(models[j].intercept_[0])) for j,component in enumerate(COMPONENTS)}
                if annotation['label'] in COMPONENTS:
                    axis=COMPONENTS.index(annotation['label'])
                    a['intended_component_active_fraction']=a['active_fraction_at_half'][axis]
                records[s].append(a)
            np.savez_compressed(OUTPUT/(s+'_'+name+suffix+'_predictions.npz'),t=data[s+'_t'],score=scores[s],
                                valid=masks[s],image_valid=np.isfinite(video[s]).all(1),**angles[s])
        report['models'][name]=records
        print(name,flush=True)
        for r in records['test']:
            print(r['start'],r['label'],'active',np.round(r['active_fraction_at_half'],2).tolist(),flush=True)
    (OUTPUT/('video_intent_results'+suffix+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(OUTPUT/('video_intent_adapters'+suffix+'.npz'),**adapters)
    np.savez_compressed(OUTPUT/('video_pca_features'+suffix+'.npz'),**{s:embedded[s] for s in embedded},
                        **{s+'_valid':np.isfinite(video[s]).all(1) for s in video})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['extract','fit'])
    parser.add_argument('--split',choices=['train','validation','test'],default='test')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--adapt-current-feedback',action='store_true')
    parser.add_argument('--stationary-negatives',action='store_true')
    parser.add_argument('--static-only',action='store_true')
    parser.add_argument('--motion-contrast',action='store_true')
    parser.add_argument('--full-recording',action='store_true')
    args=parser.parse_args()
    if args.static_only and args.full_recording:
        parser.error('--full-recording already saves both dynamic and stationary features; omit --static-only')
    if args.motion_contrast and not args.stationary_negatives:
        parser.error('--motion-contrast requires --stationary-negatives for the stored constant-image comparison')
    OUTPUT.mkdir(exist_ok=True)
    if args.mode=='extract':extract(args.split,args.limit,args.static_only,args.full_recording)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            fit(args.adapt_current_feedback,args.stationary_negatives,args.motion_contrast)
