#!/usr/bin/env python3
"""One observed-event 2x2 intervention: same RGB / same ROI, frozen shape.
Does not send robot commands or infer general causation from one event.
"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/hamer_runtime')]
from perception_hamer.src.hamer_crop_inference import HamerCropInference
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame
from perception_mediapipe.recorded_rgbd import RecordedRGBD


def run(args):
    cv2.setNumThreads(1);torch.set_num_threads(2)
    origin=json.load(open(args.source/'analysis_intent/time_origin.json'))['video_wall_origin'];wanted=[]
    for line in next(args.source.glob('perception/*/frames.jsonl')).open():
        d=json.loads(line);t=d['timestamp']-origin
        if any(abs(t-q)<.001 for q in [args.first,args.second]):wanted.append(d)
        if len(wanted)==2:break
    if len(wanted)!=2:raise ValueError('Requested two exact recorded times were not found')
    reader=RecordedRGBD(args.source/'rgbd.db3');images=[]
    for d in wanted:
        for attempt in range(2):
            try:rgb,_,_=reader.read(d['frame_number'],d['timestamp']);images.append(rgb);break
            except RuntimeError:
                if attempt:raise
    reader.close()
    runner=HamerCropInference(str(ROOT/'perception_hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt'),data_root=str(ROOT/'perception_hamer/_DATA'),precision='fp16',freeze_betas=True)
    runner.load();runner.warmup();betas=np.asarray(wanted[0]['betas_user'])
    if not np.allclose(betas,wanted[1]['betas_user']):raise ValueError('Event does not share frozen shape')
    runner.set_frozen_betas(False,betas);rotations={};cases=[]
    for image_id in [0,1]:
        for box_id in [0,1]:
            pred=runner.infer(images[image_id],wanted[box_id]['bbox'],False,wanted[image_id]['timestamp'])
            matrix=build_hamer_five_point_palm_frame(pred.pred_keypoints_3d_source_camera_axes,False).rotation
            rotations[(image_id,box_id)]=R.from_matrix(matrix);cases.append(dict(image=image_id,bbox=box_id,rotation=matrix.tolist()))
    def angle(a,b):return float(np.rad2deg((a.inv()*b).magnitude()))
    original=[R.from_matrix(build_hamer_five_point_palm_frame(d['mano_joints'],False).rotation) for d in wanted]
    result=dict(times=[d['timestamp']-origin for d in wanted],boxes=[d['bbox'] for d in wanted],cases=cases,
        recorded_original_change_deg=angle(*original),replayed_original_change_deg=angle(rotations[(0,0)],rotations[(1,1)]),
        original_replay_discrepancy_deg=[angle(original[k],rotations[(k,k)]) for k in [0,1]],
        same_image_changed_box_deg=[angle(rotations[(k,0)],rotations[(k,1)]) for k in [0,1]],
        same_box_changed_image_deg=[angle(rotations[(0,k)],rotations[(1,k)]) for k in [0,1]],
        scope='One-event 2x2 image/box intervention; no physical ground-truth claim or global causal attribution')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--first',type=float,default=298.30239057540894);p.add_argument('--second',type=float,default=298.3690528869629)
    run(p.parse_args())
