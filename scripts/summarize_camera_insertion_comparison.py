#!/usr/bin/env python3
"""Describe paired orientation results without treating frames as replicates."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

A=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])


def details(directory):
    z=np.load(directory/'test_predictions.npz');t=z['t'];iq=z['imu'];valid=z['imu_valid']
    pair=np.flatnonzero(valid[:-2]&valid[2:]&(t[2:]-t[:-2]<1.3))
    ref=np.swapaxes(iq[pair],1,2)@iq[pair+2];rv=R.from_matrix(ref).as_rotvec();ra=np.linalg.norm(rv,axis=1)
    small=ra<np.deg2rad(3);large=ra>np.deg2rad(10);result={}
    for name in ['baseline','constant_control','camera_conditioned']:
        q=z[name];delta=A@np.swapaxes(q[pair],1,2)@q[pair+2]@A.T
        v=R.from_matrix(delta).as_rotvec();error=np.rad2deg(R.from_matrix(np.swapaxes(ref,1,2)@delta).magnitude())
        ratio=np.sum(v[large]*rv[large],axis=1)/ra[large]**2
        cross=np.rad2deg(np.linalg.norm(v[large]-ratio[:,None]*rv[large],axis=1))
        result[name]=dict(interval_pairs=len(pair),
            low_reference_rotation=dict(n=int(small.sum()),reference_mean_deg=float(np.rad2deg(ra[small]).mean()),
                visual_mean_deg=float(np.rad2deg(np.linalg.norm(v[small],axis=1)).mean()),
                full_error_mean_deg=float(error[small].mean())),
            large_reference_rotation=dict(n=int(large.sum()),projected_following_ratio_mean=float(ratio.mean()),
                projected_following_ratio_median=float(np.median(ratio)),
                projected_ratio_abs_error_mean=float(np.mean(abs(ratio-1))),
                projected_ratio_p05=float(np.percentile(ratio,5)),projected_ratio_p95=float(np.percentile(ratio,95)),
                opposite_projected_direction_count=int((ratio<0).sum()),cross_axis_rotvec_mean_deg=float(cross.mean()),
                full_error_mean_deg=float(error[large].mean())))
    return result


def main(args):
    result=dict(scope='Descriptive diagnostics on reused development data, not statistical significance or independent ground truth',
        bins='IMU rotation increment <3 degrees or >10 degrees over the same roughly one-second intervals. No runtime action classification.',
        stages={})
    for name,p in [('late',args.late),('early_and_late',args.early)]:
        m=json.loads((p/'metrics.json').read_text())
        result['stages'][name]=dict(source=str(p.resolve()),epochs=m['protocol']['epochs'],baseline=m['baseline_test'],
            variants=m['variants'],motion_preservation=details(p),synthetic=m['synthetic_test'])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for stage,d in result['stages'].items():
        print(stage,'epochs',d['epochs'])
        for n,x in d['variants'].items():print(n,'epoch',x['best_epoch'],'test increment',x['test']['full_increment_mae_deg'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--early',type=Path,required=True);p.add_argument('--late',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
