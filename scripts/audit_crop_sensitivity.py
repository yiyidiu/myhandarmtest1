#!/usr/bin/env python3
"""Measure crop sensitivity on identical source images, without training.

Separate actual recropping with matched geometry from the earlier artificial
geometry-only input swap. Neither measures the population's natural detector
jitter distribution, or apportions every error in a real moving-hand video.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from experiment_hamer_camera_adapter import M, MOUNT, CameraAdapter, model, camera_encoding, reference
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame


def box(center, wh):
    return np.r_[center - wh / 2, center + wh / 2]


def angle(a, b):
    return float(np.rad2deg(R.from_matrix(a.T @ b).magnitude()))


def score_case_pair(cases, imu):
    lookup = {(x['image_seq'], x['center_from'], x['size_from']):x for x in cases}
    ref = imu[0].T @ imu[1]
    comparisons = [('natural', (11833,0,0), (11865,1,1)),
        ('both_open_box', (11833,0,0), (11865,0,0)),
        ('both_closed_box', (11833,1,1), (11865,1,1)),
        ('both_open_center_closed_size', (11833,0,1), (11865,0,1)),
        ('both_closed_center_open_size', (11833,1,0), (11865,1,0))]
    results = {}
    for name, key_a, key_b in comparisons:
        methods = {}
        for method in ['baseline', 'short', 'multi']:
            p = np.asarray(lookup[key_a]['poses'][method]); q = np.asarray(lookup[key_b]['poses'][method])
            methods[method] = angle(ref, MOUNT @ p.T @ q @ MOUNT.T)
        results[name] = methods
    return results


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.data / 'test_metadata.json').read_text())
    rows = [x for x in meta if x.get('role') != 'calibration']
    protocol = dict(source=str(args.data), images=len(rows),
        scope='Same-image crop invariance: the physical pose and viewing direction do not change within any image.',
        perturbations=['box dimensions x0.9', 'box dimensions x1.1', 'center x-5px', 'center x+5px', 'center y-5px', 'center y+5px'],
        limitations='Artificial perturbation sizes, not a measured distribution of natural detector noise. Previously inspected temporal development frames. No training or threshold selection.',
        case='The previously identified open/closed images at 439.219 and 440.286 seconds; cross their box centers and dimensions.',
        methods='Original frozen model and the frozen short/multi camera adapters from the previous experiment.',
        hand='Fixed left; original per-image frozen betas; no output filter.')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    runner = model(); head = runner._model.mano_head.float().eval()
    for p in runner._model.parameters(): p.requires_grad_(False)
    adapters = {}
    for arm in ['short', 'multi']:
        ad = CameraAdapter(True).cuda()
        ad.load_state_dict(torch.load(args.fit / arm / 'camera_conditioned.pt'))
        adapters[arm] = ad
    captured = []
    hook = runner._model.backbone.register_forward_hook(lambda _m, _i, x: captured.append(x.detach().float()))
    reflection = torch.tensor(M, dtype=torch.float32, device='cuda')

    def outputs(feature, encoding, local):
        out = {}
        with torch.no_grad():
            for name, ad in [('baseline', None)] + list(adapters.items()):
                x = feature if ad is None else ad(feature, encoding)[0]
                g = head(x)[0]['global_orient'][0, 0]
                out[name] = (reflection @ g @ local).cpu().numpy()
        return out

    def infer(rgb, bbox, row):
        captured.clear(); runner.set_frozen_betas(False, np.array(row['betas']))
        pred = runner.infer(rgb, bbox, False, row['stamp'])
        palm = build_hamer_five_point_palm_frame(pred.pred_keypoints_3d_source_camera_axes, False)
        if not palm.valid: raise ValueError(palm.failure_reason)
        local = torch.tensor(pred.global_orient.T @ M @ palm.rotation, dtype=torch.float32, device='cuda')
        enc = torch.tensor(camera_encoding(pred.quality['affine_original_to_crop']), device='cuda')[None]
        feature = captured[0]
        return dict(feature=feature, encoding=enc, local=local,
            poses=outputs(feature, enc, local), global_native=pred.global_orient,
            crop_size_px=pred.quality['crop_bbox_size_px'])

    case_rows = [next(x for x in rows if x['seq'] == seq) for seq in [11833, 11865]]
    boxes = np.array([x['bbox'] for x in case_rows]); centers = (boxes[:, :2] + boxes[:, 2:]) / 2; sizes = boxes[:, 2:] - boxes[:, :2]
    cases = []
    for source, row in enumerate(case_rows):
        rgb = cv2.cvtColor(cv2.imread(row['file']), cv2.COLOR_BGR2RGB)
        ref = infer(rgb, row['bbox'], row)
        for center_source in [0, 1]:
            for size_source in [0, 1]:
                bbox = box(centers[center_source], sizes[size_source]); current = infer(rgb, bbox, row)
                encoding_only = outputs(ref['feature'], current['encoding'], ref['local'])
                image_only = outputs(current['feature'], ref['encoding'], current['local'])
                cases.append(dict(image_seq=row['seq'], t=row['t'], center_from=center_source, size_from=size_source,
                    bbox=bbox.tolist(), crop_size_px=current['crop_size_px'],
                    poses={n:p.tolist() for n,p in current['poses'].items()},
                    native_global_change_deg=angle(ref['global_native'], current['global_native']),
                    consistent_recrop_change_deg={n:angle(ref['poses'][n], p) for n, p in current['poses'].items()},
                    encoding_only_change_deg={n:angle(ref['poses'][n], p) for n, p in encoding_only.items()},
                    image_only_change_deg={n:angle(ref['poses'][n], p) for n, p in image_only.items()}))
    (args.out / 'case_center_size.json').write_text(json.dumps(cases, indent=2))
    imu, valid = reference(case_rows)
    if not valid.all(): raise ValueError('The two case images require valid IMU references')
    pair_errors = score_case_pair(cases, imu)
    (args.out / 'consistent_pair_errors.json').write_text(json.dumps(pair_errors, indent=2))
    if args.case_only:
        hook.remove()
        return
    results = []; failures = []
    for index, row in enumerate(rows):
        rgb = cv2.cvtColor(cv2.imread(row['file']), cv2.COLOR_BGR2RGB)
        bb = np.asarray(row['bbox']); c = (bb[:2] + bb[2:]) / 2; wh = bb[2:] - bb[:2]
        ref = infer(rgb, bb, row)
        perturbations = [('scale_0.9', c, wh * .9), ('scale_1.1', c, wh * 1.1),
            ('center_x_minus5', c + [-5, 0], wh), ('center_x_plus5', c + [5, 0], wh),
            ('center_y_minus5', c + [0, -5], wh), ('center_y_plus5', c + [0, 5], wh)]
        for name, cc, ss in perturbations:
            bbox = box(cc, ss)
            try:
                current = infer(rgb, bbox, row)
            except (ValueError, RuntimeError) as exc:
                failures.append(dict(seq=row['seq'], perturbation=name, error=str(exc)))
                if isinstance(exc, torch.cuda.OutOfMemoryError): raise
                continue
            results.append(dict(seq=row['seq'], t=row['t'], perturbation=name, bbox=bbox.tolist(),
                native_global_change_deg=angle(ref['global_native'], current['global_native']),
                change_deg={n:angle(ref['poses'][n], p) for n, p in current['poses'].items()}))
        if index % 20 == 0: print('Same-image recrop', index, '/', len(rows), flush=True)
    hook.remove()
    summary = dict(images=len(rows), comparisons=len(results), failures=failures, perturbations={})
    for perturbation in sorted({x['perturbation'] for x in results}) + ['all']:
        subset = [x for x in results if perturbation == 'all' or x['perturbation'] == perturbation]
        result = {}
        for method in ['baseline', 'short', 'multi', 'native_global']:
            v = np.array([x['native_global_change_deg'] if method == 'native_global' else x['change_deg'][method] for x in subset])
            result[method] = dict(n=len(v), mean_deg=float(v.mean()), median_deg=float(np.median(v)),
                p95_deg=float(np.percentile(v, 95)), max_deg=float(v.max()), comparisons_above5deg=int((v > 5).sum()))
        summary['perturbations'][perturbation] = result
    (args.out / 'recrop_predictions.json').write_text(json.dumps(results, indent=2))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case-only', action='store_true')
    cv2.setNumThreads(1); torch.set_num_threads(2)
    run(p.parse_args())
