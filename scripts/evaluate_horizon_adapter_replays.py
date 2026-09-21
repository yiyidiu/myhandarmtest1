#!/usr/bin/env python3
"""Frozen-weight transfer and task checks for the 1/5/10-second ablation.

No training or checkpoint selection. The Sep 20 IMU session has already been
inspected in earlier research; it is a different-recording development check,
not a newly blinded benchmark. The grasp replay has no orientation reference.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from experiment_hamer_camera_adapter import (
    ROOT, MOUNT, LATEST, CameraAdapter, load_pack, model, predict, score,
)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        frozen_weights=[str(args.fit / arm / f'{variant}.pt')
            for arm in ['short', 'multi'] for variant in ['constant_control', 'camera_conditioned']],
        transfer_source=str(LATEST), transfer_selection='All 582 existing sampled valid model inputs, 42.78–349.88 s; no outcome-based selection.',
        transfer_role='Different recording from all four Sep 21 fits, but previously inspected development data. Camera setup differs.',
        task_source='teleop_feedback_20260920T145923_kmxD',
        task_selection='Previously frozen 1180 inputs and native-cadence grasp window; no IMU truth.',
        tuning='None: original alignment, axes, source boxes, shape and selected checkpoints retained.',
        motion_labels='Used only for post-prediction segment scores on Sep 20; never supplied to the model.')
    (args.out / 'replay_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    runner = model(); head = runner._model.mano_head.float().eval()
    for param in runner._model.parameters(): param.requires_grad_(False)
    runner._model.backbone.cpu(); runner._model.mano.cpu(); torch.cuda.empty_cache()
    adapters = {}
    for arm in ['short', 'multi']:
        for variant, camera in [('constant_control', False), ('camera_conditioned', True)]:
            adapter = CameraAdapter(camera).cuda()
            adapter.load_state_dict(torch.load(args.fit / arm / f'{variant}.pt'))
            adapters[f'{arm}_{variant}'] = adapter
    source = load_pack(ROOT / '.runtime/hamer_camera_adapter_20260921', 'train')
    source['meta'][0]['role'] = 'calibration'
    annotations = json.loads((LATEST / 'analysis_intent/annotations.json').read_text())
    times = np.array([x['t'] for x in source['meta']]); imu = source['imu']
    poses = {}; metrics = {}; segments = []
    for name, adapter in [('baseline', None)] + list(adapters.items()):
        poses[name], _ = predict(head, adapter, source)
        metrics[name] = score(poses[name], source)
    for ann in annotations:
        idx = np.flatnonzero((times >= ann['start']) & (times < ann['end']) & source['valid'])
        row = dict(annotation=ann, frames=len(idx), methods={})
        if len(idx) >= 2:
            i, ii = idx[0], idx[1:]
            ir = imu[i].T @ imu[ii]
            for name, q in poses.items():
                pr = MOUNT @ q[i].T @ q[ii] @ MOUNT.T
                e = np.rad2deg(R.from_matrix(np.swapaxes(ir, 1, 2) @ pr).magnitude())
                row['methods'][name] = dict(full_relative_mean_deg=float(e.mean()),
                    full_relative_p95_deg=float(np.percentile(e, 95)),
                    imu_peak_deg=float(np.rad2deg(R.from_matrix(ir).magnitude()).max()),
                    visual_peak_deg=float(np.rad2deg(R.from_matrix(pr).magnitude()).max()))
        else:
            row['exclusion_reason'] = 'Fewer than two available sampled inputs; no relative rotation can be measured.'
        segments.append(row)
    np.savez_compressed(args.out / 'transfer_predictions.npz', t=times, imu=imu, imu_valid=source['valid'], **poses)
    (args.out / 'transfer_metrics.json').write_text(json.dumps(metrics, indent=2))
    (args.out / 'transfer_segments.json').write_text(json.dumps(segments, ensure_ascii=False, indent=2))
    print('Different-recording evaluation complete', len(times), flush=True)
    del source
    task_data = ROOT / '.runtime/hamer_camera_adapter_grasp_20260921'
    z = np.load(task_data / 'task_features.npz')
    meta = json.loads((task_data / 'task_metadata.json').read_text())
    pack = dict(meta=meta, data={k:torch.from_numpy(z[k].astype(np.float32) if k != 'features' else z[k]) for k in z.files})
    poses = {}
    for name, adapter in [('baseline', None)] + list(adapters.items()):
        poses[name], _ = predict(head, adapter, pack)
    t = np.array([x['t'] for x in meta]); dt = np.diff(t); native = (dt > .005) & (dt < .2)
    baseline_relative = poses['baseline'] @ poses['baseline'][0].T
    task = dict(frames=len(t), native_interval_pairs=int(native.sum()), ground_truth_available=False,
        scope='Output-change stress check; neither tracking accuracy nor detection availability nor physical grasp success.', methods={})
    for name, q in poses.items():
        step = np.rad2deg(R.from_matrix(q[1:] @ np.swapaxes(q[:-1], 1, 2)).magnitude())[native]
        relative = q @ q[0].T
        change = np.rad2deg(R.from_matrix(np.swapaxes(baseline_relative, 1, 2) @ relative).magnitude())
        task['methods'][name] = dict(native_step_mean_deg=float(step.mean()),
            native_step_p95_deg=float(np.percentile(step, 95)), native_step_max_deg=float(step.max()),
            steps_above_10deg=int((step > 10).sum()),
            relative_output_difference_from_baseline_mean_deg=float(change.mean()),
            relative_output_difference_from_baseline_p95_deg=float(np.percentile(change, 95)))
    np.savez_compressed(args.out / 'task_predictions.npz', t=t, **poses)
    (args.out / 'task_metrics.json').write_text(json.dumps(task, indent=2))
    print('Existing grasp stress replay complete', len(t), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    cv2.setNumThreads(1); torch.set_num_threads(2)
    main(parser.parse_args())
