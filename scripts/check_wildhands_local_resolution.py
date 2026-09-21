#!/usr/bin/env python3
"""One-variable WildHands local-image intervention on exact cached source PNGs.

Global input, crop geometry (including official integer rounding), KPE, real K,
checkpoint, handedness and frozen shape stay fixed. Only local image sampling
changes from downsample-then-enlarge to one warp from original RGB.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import cv2, numpy as np, torch
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/wildhands_demo')]
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'inputs').mkdir(exist_ok=True)
    lab=ROOT/'.runtime/wildhands_evaluation_20260920'
    os.environ['WILDHANDS_MANO_DIR']=str(lab/'assets/mano')
    from wildhands.configs.parser import construct_args
    from wildhands.models.wrapper import WildHandsWrapper
    from wildhands.datasets.dataset import WildHandsDataset
    from wildhands.common import data_utils as du
    cv2.setNumThreads(1);torch.set_num_threads(2)
    source=ROOT/'.runtime/fixed_crop_replay_20260920/full'
    records=[json.loads(s)for s in (source/'predictions.jsonl').open()]
    records=[x for x in records if (x['seq']==298 or 265<=x['t']<=273) and (source/'source_png'/f'{x["seq"]}.png').exists()]
    wanted={x['seq']for x in records};old={}
    proc=subprocess.Popen(['zstd','-q','-dc',str(lab/'full/predictions.jsonl.zst')],stdout=subprocess.PIPE,text=True)
    for line in proc.stdout:
        x=json.loads(line)
        if x['seq']in wanted:old[x['seq']]=x
    if proc.wait():raise RuntimeError('Cannot read original prediction archive')
    cfg=construct_args();model=WildHandsWrapper(cfg)
    ck=torch.load(lab/'assets/wildhands.ckpt',map_location='cpu')
    state={k:v for k,v in ck['state_dict'].items()if not any(q in k for q in ['head_o','arti_head','grasp_classifier'])}
    model.load_state_dict(state,strict=True);del ck,state
    model=model.cuda().eval();device=torch.device('cuda')
    meta=json.load(open(lab/'full/metadata.json'));intr=meta['intrinsics']
    K=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1]],np.float32)
    joint_order=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
    def h(mat):return np.vstack([mat,[0,0,1]])
    rows=[]
    for index,row in enumerate(records):
        rgb=cv2.cvtColor(cv2.imread(str(source/'source_png'/f'{row["seq"]}.png')),cv2.COLOR_BGR2RGB)
        square,A,_=du.generate_patch_image_clean(rgb,[320,240,640,640],1.,0.,[840,840],cv2.INTER_CUBIC)
        square=square.clip(0,255).astype(np.uint8)
        box=np.c_[np.array(row['bbox']).reshape(2,2),np.ones(2)]@A.T
        ds=WildHandsDataset(cfg,square,box.reshape(1,4),np.array([0]),focal_length=(intr['fx']+intr['fy'])/2*840/640,rescale_factor=1.75,intrx=h(A)@K)
        groups=ds[0];inputs=[{k:(v.unsqueeze(0).to(device)if isinstance(v,torch.Tensor)else torch.as_tensor(v).unsqueeze(0).to(device))for k,v in group.items()}for group in groups]
        # Repeat the author's exact keypoint/bbox rounding before deriving the
        # local affine; KPE and boxes remain the original tensors in both arms.
        aug={'rot':0,'sc':1.,'flip':0,'pn':np.ones(3)}
        ends=du.j2d_processing(du.pad_jts2d(box.copy()),[420,420],4.2,aug,224)
        ends=(((ends[:,:2]+1)/2)*224).flatten().clip(0,223)
        b=[ends[0],ends[1],ends[2]-ends[0],ends[3]-ends[1]]
        x0,y0,x1,y1=int(b[0]),int(b[1]),int(b[0]+b[2]),int(b[1]+b[3])
        mx,my,size=(x0+x1)//2,(y0+y1)//2,max(x1-x0,y1-y0)
        G=du.gen_trans_from_patch_cv(420,420,840,840,224,224,1.,0.)
        L=du.gen_trans_from_patch_cv(mx,my,size*1.75,size*1.75,224,224,1.,0.)
        direct=cv2.warpAffine(rgb,(h(L)@h(G)@h(A))[:2],(224,224),flags=cv2.INTER_CUBIC).astype(np.float32).clip(0,255)/255
        direct=ds.normalize_img(torch.from_numpy(direct.transpose(2,0,1)).float()).unsqueeze(0).cuda()
        rr=dict(t=row['t'],seq=row['seq'],bbox=row['bbox'],intrinsics_network=inputs[1]['intrinsics'][0].cpu().tolist(),
                local_source_affine=(h(L)@h(G)@h(A)).tolist(),results={})
        for name in ['official','direct_original']:
            inp=[dict(inputs[0]),dict(inputs[1])]
            if name=='direct_original':inp[0]['l_img']=direct
            if index==0 and name=='official':
                with torch.inference_mode():
                    for _ in range(3):model(inp)
            with torch.inference_mode():
                pred=model(inp)
                frozen=model.model.mano_l(rotmat=pred['pred.pose.l'].to(device),shape=torch.tensor([old[row['seq']]['frozen_beta']],dtype=torch.float32,device=device),cam=pred['pred.cam_t.wp.l'].to(device),K=inp[1]['intrinsics'])
            joints=frozen['joints3d.l'][0].cpu().numpy()[joint_order]
            frame=build_hamer_five_point_palm_frame(joints,False)
            rr['results'][name]=dict(q=frame.quaternion_xyzw.tolist(),root=pred['pred.pose.l'][0,0].cpu().tolist())
            if name=='official':rr['official_vs_saved_rotation_deg']=float(np.rad2deg((R.from_quat(frame.quaternion_xyzw).inv()*R.from_quat(old[row['seq']]['palm_q'])).magnitude()))
            if index==0 or abs(row['t']-268)<.08:
                im=inp[0]['l_img'][0].cpu().numpy().transpose(1,2,0)
                im=np.clip((im*[.229,.224,.225]+[.485,.456,.406])*255,0,255).astype(np.uint8)
                cv2.imwrite(str(a.out/'inputs'/f'{row["seq"]}_{name}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        rows.append(rr)
        if index%10==0:print(index+1,len(records),row['t'],flush=True)
    (a.out/'predictions.json').write_text(json.dumps(rows,indent=2))
    src=ROOT/'.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent'
    imu=np.load(src/'imu_signals.npz');si=Slerp(imu['t'],R.from_quat(imu['q']));off=json.load(open(src/'time_alignment.json'))['used_offset_s'];ts=np.array([x['t']for x in rows]);ci=next(j for j,x in enumerate(rows)if x['seq']==298)
    ir=si(ts[ci]+off).inv()*si(ts+off);im=np.rad2deg(ir.magnitude());ix=np.clip(np.searchsorted(imu['t'],ts+off),1,len(imu['t'])-1);ok=imu['time_valid'][ix]&imu['time_valid'][ix-1]&(ts>=265)
    summary=dict(n=len(rows),yaw_imu_valid_n=int(ok.sum()),reference='IMU treated as provisional ground truth by user instruction',offset_s=off,C_time=ts[ci],
        original_replay_max_difference_deg=max(x['official_vs_saved_rotation_deg']for x in rows),methods={},examples=[])
    angles={}
    for name in ['official','direct_original']:
        r=R.from_quat([x['results'][name]['q']for x in rows]);mag=np.rad2deg((r*r[ci].inv()).magnitude());angles[name]=mag
        summary['methods'][name]=dict(mean_magnitude_error_deg=float(np.mean(abs(mag[ok]-im[ok]))),peak_rotation_deg=float(mag[ok].max()),imu_peak_deg=float(im[ok].max()))
    for t in [268,269,270,272]:
        j=int(np.argmin(abs(ts-t)));summary['examples'].append(dict(t=float(ts[j]),imu_deg=float(im[j]),**{n:float(v[j])for n,v in angles.items()}))
    (a.out/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
