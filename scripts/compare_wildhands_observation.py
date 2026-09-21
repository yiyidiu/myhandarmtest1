#!/usr/bin/env python3
"""Source-matched WildHands replay. Offline only; never sends robot commands.

Uses the authors' demo model/dataset, real RGB intrinsics, physical left-hand
head and saved detection boxes. Native preprocessing differs from HaMeR, so
this is a model-system comparison, not an isolated KPE ablation.
"""
import argparse,json,os,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame
from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition,estimate_mano_wrist_frame
from perception_mediapipe.recorded_rgbd import RecordedRGBD

OPENPOSE_MAP=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def to_device(batch,device):
    return [{k:(v.unsqueeze(0).to(device) if isinstance(v,torch.Tensor) else
                 torch.as_tensor(v).unsqueeze(0).to(device)) for k,v in group.items()}
            for group in batch]


def run(args):
    vendor=ROOT/'third_party/wildhands_demo';sys.path.insert(0,str(vendor))
    os.environ['WILDHANDS_MANO_DIR']=str(args.mano.resolve())
    from wildhands.configs.parser import construct_args
    from wildhands.models.wrapper import WildHandsWrapper
    from wildhands.datasets.dataset import WildHandsDataset
    from wildhands.common.data_utils import generate_patch_image_clean
    cv2.setNumThreads(1);torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'preview').mkdir(exist_ok=True)
    cfg=construct_args();device=torch.device('cuda')
    model=WildHandsWrapper(cfg)
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    unused=['head_o','arti_head','grasp_classifier']
    state={k:v for k,v in checkpoint['state_dict'].items() if not any(s in k for s in unused)}
    model.load_state_dict(state,strict=True);del checkpoint,state
    model=model.to(device).eval()
    mano=model.model.mano_l.mano
    # Build a neutral definition from this exact left MANO geometry. The
    # model adds its mean articulation; cancel it only for the neutral template.
    with torch.inference_mode():
        neutral=mano(betas=torch.zeros(1,10,device=device),global_orient=torch.zeros(1,3,device=device),hand_pose=-mano.pose_mean[None,3:])
    neutral_vertices=neutral.vertices[0].cpu().numpy();neutral_joints=neutral.joints[0].cpu().numpy()[OPENPOSE_MAP]
    definition=build_mano_wrist_definition(neutral_vertices,neutral_joints,mano.faces_tensor.cpu().numpy(),False)
    reader=RecordedRGBD(args.source/'rgbd.db3');intr=reader.intr
    K=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1]],np.float32)
    origin=json.load(open(args.source/'analysis_intent/time_origin.json'))['video_wall_origin']
    wanted=[]
    for line in next(args.source.glob('perception/*/frames.jsonl')).open():
        d=json.loads(line);t=d['timestamp']-origin
        if t>=args.end:break
        if d.get('mano_joints') is not None and d.get('bbox') is not None:wanted.append(d)
    # The first 30 input frames reproduce the existing shape-freeze policy.
    if args.times:
        selected={int(np.argmin([abs(d['timestamp']-origin-t) for d in wanted])) for t in args.times}
        selected.update(range(min(30,len(wanted))))
        wanted=[d for i,d in enumerate(wanted) if i in selected]
    metadata=dict(source=str(args.source.resolve()),video_wall_origin=origin,checkpoint=str(args.checkpoint.resolve()),
        vendor_commit='f99dfea0d1fce970aed2d31d1018eda280e05f47',intrinsics=intr,
        precision='float32',physical_hand='left',mano_output_order='native 16 joints plus tips remapped to OpenPose 21',
        pose_filter=False,palm_auxiliary=False,hand_bbox='same recorded HaMeR source-frame ROI',
        crop_scale=args.crop_scale,input_resolution=cfg.img_res,
        official_square_preprocessing=args.official_square,
        geometry_shape_policy='median of first 30 predicted beta vectors; raw native geometry also retained',
        depth_policy='orientation-only replay; model camera translation is never used as measured depth',
        scope='WildHands published model versus recorded HaMeR; not KPE-only ablation',
        baseline_commit=(args.source/'commit.txt').read_text().strip() if (args.source/'commit.txt').exists() else None)
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    beta_samples=[];frozen=None;vertices_all=[];joints_all=[];sequences=[];errors=0
    with (args.out/'predictions.jsonl').open('w') as f:
        for index,d in enumerate(wanted):
            row=dict(t=d['timestamp']-origin,seq=d['capture_sequence'],frame_number=d['frame_number'],stamp=d['timestamp'],bbox=d['bbox'],valid=False)
            try:
                for attempt in range(2):
                    try:rgb,depth,diag=reader.read(d['frame_number'],d['timestamp']);break
                    except RuntimeError:
                        if attempt:raise
                begin=time.perf_counter()
                input_rgb=rgb;input_box=np.array([d['bbox']],np.float32);input_K=K;input_focal=float((intr['fx']+intr['fy'])/2)
                if args.official_square:
                    side=max(rgb.shape[:2])
                    input_rgb,affine,_=generate_patch_image_clean(rgb,[rgb.shape[1]/2,rgb.shape[0]/2,side,side],1.,0.,
                        [args.official_square,args.official_square],cv2.INTER_CUBIC)
                    input_rgb=input_rgb.clip(0,255).astype(np.uint8)
                    ends=np.column_stack([input_box.reshape(2,2),np.ones(2)])
                    input_box=(ends@affine.T).reshape(1,4)
                    input_K=np.vstack([affine,[0,0,1]])@K
                    input_focal*=args.official_square/side
                ds=WildHandsDataset(cfg,input_rgb,input_box,np.array([0]),
                    focal_length=input_focal,rescale_factor=args.crop_scale,intrx=input_K)
                inputs=to_device(ds[0],device)
                torch.cuda.synchronize();ready=time.perf_counter()
                if index==0:
                    with torch.inference_mode():
                        for _ in range(5):model(inputs)
                    torch.cuda.synchronize();begin=ready=time.perf_counter()
                with torch.inference_mode():prediction=model(inputs)
                torch.cuda.synchronize();inferred=time.perf_counter()
                beta=prediction['pred.beta.l'][0].cpu().numpy()
                if frozen is None:
                    beta_samples.append(beta)
                    if len(beta_samples)==30:frozen=np.median(beta_samples,axis=0)
                native_joints=prediction['pred.joints3d.l'][0].cpu().numpy()[OPENPOSE_MAP]
                native_vertices=prediction['pred.vertices.l'][0].cpu().numpy()
                vertices=native_vertices;joints=native_joints
                if frozen is not None:
                    with torch.inference_mode():
                        frozen_output=model.model.mano_l(rotmat=prediction['pred.pose.l'].to(device),shape=torch.tensor(frozen[None],dtype=torch.float32,device=device),
                            cam=prediction['pred.cam_t.wp.l'].to(device),K=inputs[1]['intrinsics'])
                    vertices=frozen_output['vertices.l'][0].cpu().numpy();joints=frozen_output['joints3d.l'][0].cpu().numpy()[OPENPOSE_MAP]
                palm=build_hamer_five_point_palm_frame(joints,False);ring=estimate_mano_wrist_frame(vertices,definition)
                raw_palm=build_hamer_five_point_palm_frame(native_joints,False)
                cam=prediction['pred.cam_t.l'][0].cpu().numpy();projected=joints+cam
                xy=np.column_stack([intr['fx']*projected[:,0]/projected[:,2]+intr['ppx'],intr['fy']*projected[:,1]/projected[:,2]+intr['ppy']])
                row.update(valid=bool(palm.valid and ring.valid),palm_q=palm.quaternion_xyzw,ring_q=ring.quaternion_xyzw,
                    native_palm_q=raw_palm.quaternion_xyzw,global_rotation=prediction['pred.pose.l'][0,0].cpu().numpy(),
                    joints=joints,pixels=xy,model_camera_translation=cam,beta=beta,frozen_beta=frozen,
                    shape_frozen=frozen is not None,ring_quality=ring.quality,
                    center_angle=inputs[0]['l_center_angle'][0].cpu().numpy(),corners_angle=inputs[0]['l_corner_angle'][0].cpu().numpy(),
                    native_input_bbox=inputs[0]['l_bbox'][0].cpu().numpy(),intrinsics_network=inputs[1]['intrinsics'][0].cpu().numpy(),
                    preprocess_ms=1000*(ready-begin),network_ms=1000*(inferred-ready),postprocess_ms=1000*(time.perf_counter()-inferred),
                    source_pair=diag)
                vertices_all.append(vertices);joints_all.append(joints);sequences.append(d['capture_sequence'])
                if args.times or index%250==0:
                    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    for j in [0,5,9,13,17]:
                        u,v=np.rint(xy[j]).astype(int);cv2.circle(image,(u,v),3,(0,170,255),-1);cv2.putText(image,str(j),(u+4,v-4),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,170,255),1)
                    cv2.putText(image,f'WildHands {row["t"]:.2f}s | projected model points',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                    cv2.imwrite(str(args.out/'preview'/f'{d["capture_sequence"]}.jpg'),image)
                if index%100==0:print('frames',index+1,'time',round(row['t'],2),'network_ms',round(row['network_ms'],2),flush=True)
            except (ValueError,KeyError) as exc:
                row['failure']=str(exc);errors+=1;print('frame failure',row['seq'],str(exc),flush=True)
            f.write(json.dumps(row,default=plain)+'\n');f.flush()
    reader.close()
    np.savez_compressed(args.out/'geometry.npz',seq=sequences,vertices=np.array(vertices_all,dtype=np.float32),joints=np.array(joints_all,dtype=np.float32),faces=mano.faces_tensor.cpu().numpy())
    print('completed',len(wanted),'failures',errors,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--mano',type=Path,required=True)
    ap.add_argument('--times',type=lambda x:[float(v) for v in x.split(',')]);ap.add_argument('--end',type=float,default=350.)
    ap.add_argument('--crop-scale',type=float,default=1.75)
    ap.add_argument('--official-square',type=int,default=840,
        help='authors demo pads/resizes full image to this square before native preprocessing; 0 only for adapter sensitivity diagnosis')
    run(ap.parse_args())
