#!/usr/bin/env python3
"""Known rigid hand motions through the learned correction, without ROS.

Use a recorded hand's 3-D geometry and real camera intrinsics. Transform it
by known simultaneous rotations/translations and project exact landmarks.
This tests ideal, physically specified input; it is not a human trial.
"""
from pathlib import Path
import json
import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from evaluate_continuous_intent import ROOT, SOURCE, image_motion_samples, motion_statistics
from train_continuous_intent import ContinuousIntent


OUTPUT = ROOT / '.runtime/continuous_intent_20260912'


def simulate_features(geometry, zero_position, k, requested, movement, rise_s=1., hold_s=1., fall_s=1., current_pose=False, visual_motion=False, translation_delay_s=0.):
    frame = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    times = np.arange(0., 2.+rise_s+hold_s+fall_s+translation_delay_s, 1./15.)
    u = np.clip((times-1.)/rise_s, 0., 1.)
    rise = u*u*u*(10.-15.*u+6.*u*u)
    u = np.clip((times-1.-rise_s-hold_s)/fall_s, 0., 1.)
    profile = rise-u*u*u*(10.-15.*u+6.*u*u)
    u = np.clip((times-1.-translation_delay_s)/rise_s, 0., 1.)
    translation_profile = u*u*u*(10.-15.*u+6.*u*u)
    u = np.clip((times-1.-translation_delay_s-rise_s-hold_s)/fall_s, 0., 1.)
    translation_profile -= u*u*u*(10.-15.*u+6.*u*u)
    angles = profile[:,None]*np.deg2rad(requested)
    xyz = []
    rotation = []
    for a, h in zip(angles, translation_profile):
        pitch, roll, yaw = a
        mapped = (R.from_rotvec([yaw,0,0])*R.from_rotvec([0,-pitch,0])*R.from_rotvec([0,0,-roll])).as_matrix()
        camera = frame.T@mapped@frame
        rotation.append(camera)
        xyz.append(geometry@camera.T+zero_position+h*np.array(movement))
    xyz = np.array(xyz)
    uv = np.stack((k['fx']*xyz[:,:,0]/xyz[:,:,2]+k['ppx'], k['fy']*xyz[:,:,1]/xyz[:,:,2]+k['ppy']),axis=2)
    p = zero_position+translation_profile[:,None]*np.array(movement)
    q = R.from_matrix(rotation)
    measured = np.column_stack([np.vstack([np.zeros(3),np.diff(p,axis=0)*15]),
                                np.vstack([np.zeros(3),(q[1:]*q[:-1].inv()).as_rotvec()*15])])
    if current_pose:
        measured[:,3:]=np.vstack([np.zeros(3),np.diff(angles,axis=0)*15])
    sampled = np.arange(0,len(times),3)
    cues = []
    for i in sampled:
        z = xyz[i,:,2]
        palm = np.median(z[[5,9,13,17]])
        long = uv[i,[5,9,13,17]].mean(0)-uv[i,0]
        cues.append([np.median(z[[6,7,8,10,11,12,14,15,16,18,19,20]])-palm,
                     palm-z[0],z[17]-z[5],np.arctan2(long[1],long[0]),
                     np.log(max(10.,np.linalg.norm(uv[i,17]-uv[i,5]))),np.log(max(10.,np.linalg.norm(long)))])
    cues = np.array(cues)
    cues[:,3] = np.unwrap(cues[:,3])
    speed = np.vstack([np.zeros(6),np.diff(cues,axis=0)/.2])
    speed = np.clip(speed,[-.5,-.5,-.5,-5,-3,-3],[.5,.5,.5,5,3,3])
    observed = speed[np.arange(len(times))//3]
    flow=None
    if visual_motion:
        flow=image_motion_samples([dict(t=times[i],xy=uv[i]) for i in sampled])[np.arange(len(times))//3]
    features=motion_statistics(measured,observed,np.ones(len(times)),flow)
    if current_pose:
        features=np.column_stack([features,p-zero_position,angles])
    return times, angles, features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='feedback_trajectory')
    args=parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(OUTPUT / (args.model+'.pt'), map_location='cpu', weights_only=False)
    model = ContinuousIntent(len(checkpoint['mean']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    manifest = json.loads((SOURCE / 'c_rgbd/manifest.json').read_text())
    reference = next(r for r in manifest['rows'] if r['seq'] == 1476)
    records = SOURCE.parent / 'perception/DEV_HAMER_6DOF_OBSERVATION_20260912T145638/frames.jsonl'
    with records.open() as stream:
        row = next(r for r in map(json.loads, stream) if r.get('capture_sequence') == 1476)
    geometry = np.array(row['mano_joints'])
    geometry -= geometry[0]
    zero_position = np.array(reference['position'])
    k = manifest['intrinsics']
    cases = [([.5,0,0],[0,0,0]),([0,.5,0],[0,0,0]),([0,0,.5],[0,0,0]),
             ([5,0,0],[0,0,0]),([0,5,0],[0,0,0]),([0,0,5],[0,0,0]),
             ([10,15,0],[0,0,0]),([10,15,20],[.025,-.02,.02])]
    results = []
    cases = [(requested, movement, 1., 0.) for requested, movement in cases]
    cases.append(([0,0,30],[.03,0,0],3.,2.))
    for requested, movement, hold, translation_delay in cases:
        times, angles, features = simulate_features(geometry, zero_position, k, requested, movement,
                                                   hold_s=hold,translation_delay_s=translation_delay,
                                                   current_pose=checkpoint.get('current_pose',False),
                                                   visual_motion=checkpoint.get('visual_motion',False))
        features=np.clip((np.array(features)-checkpoint['mean'])/checkpoint['std'],-8.,8.)
        lag=checkpoint.get('decision_lag_frames',0)
        windows=np.stack([features[np.clip(np.arange(i-checkpoint['window']+1,i+1)+lag,0,len(features)-1)] for i in range(len(times))])
        with torch.no_grad():score=model(torch.tensor(windows.transpose(0,2,1),dtype=torch.float32)).numpy()
        if checkpoint.get('absolute_residual',False):
            from train_intent_trajectory import direct_pose_correction
            corrected=direct_pose_correction(torch.tensor(angles),torch.tensor(score)).numpy()
        else:
            keep=np.minimum(1.,score/np.maximum(checkpoint['thresholds'],1e-9))
            corrected=np.cumsum(np.vstack([np.zeros(3),np.diff(angles,axis=0)])*keep,axis=0)
        record=dict(requested_degrees=requested,translation_m=movement,
                    translation_delay_s=translation_delay,
                    decision_lag_s=lag/15.,
                    reached_degrees=np.rad2deg(np.max(corrected,axis=0)).tolist(),
                    final_degrees=np.rad2deg(corrected[-1]).tolist(),
                    main_curve_rms_error_degrees=np.rad2deg(np.sqrt(np.mean((corrected-angles)**2,axis=0))).tolist())
        if translation_delay:
            ids=(times>=3.)&(times<=4.)
            record['held_yaw_during_translation_min_deg']=float(np.rad2deg(corrected[ids,2].min()))
            record['held_yaw_during_translation_max_deg']=float(np.rad2deg(corrected[ids,2].max()))
        results.append(record)
        print(record,flush=True)
    (OUTPUT/(args.model+'_known_combination_results.json')).write_text(json.dumps(dict(scope=__doc__,results=results),indent=2))


if __name__ == '__main__':
    main()
