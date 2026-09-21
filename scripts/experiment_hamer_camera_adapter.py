#!/usr/bin/env python3
"""Bounded offline camera-conditioned decoder adaptation, not deployed control.

Freeze HaMeR and its per-image articulation. Learn only an additive camera
encoding before its existing pose decoder. A constant-input adapter is the
matched no-camera control. Membership comes from a frozen source manifest;
the original default uses Sep 20 training and two Sep 11 evaluation sessions.
Prepared manifests can instead specify current-view temporal partitions.
None of the runtime inputs contain IMU,
time, motion labels, or desired motion. This is not the full published KPE model.
"""
import argparse
import csv
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/hamer_runtime')]
from perception_hamer.src.hamer_crop_inference import HamerCropInference
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame
from scripts.replay_hamer_fixed_crop import source_rgb
from perception_mediapipe.recorded_rgbd import RecordedRGBD

K = np.array([[606.5111083984375, 0, 319.0163879394531],
              [0, 606.2513427734375, 248.72071838378906], [0, 0, 1.]])
M = np.diag([-1., 1., 1.])
MOUNT = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
LATEST = ROOT / '.runtime/hand_imu_observation_20260920T162851_bphL'
EARLIER = Path('/home/dongtian/myhandarmtest1-teleop-repair/.runtime')


def horizontal_baseline(pred, metadata):
    gain = json.loads((ROOT / 'docs/imu_view_correction_20260920/correction_metrics.json').read_text())['calibration']['gain']
    u = np.array([(x['bbox'][0]+x['bbox'][2])/2 for x in metadata])
    rays = np.arctan2(u - K[0,2], K[0,0])
    return R.from_euler('y', gain * rays).as_matrix() @ pred


def model():
    m = HamerCropInference(str(ROOT / 'perception_hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt'),
        data_root=str(ROOT / 'perception_hamer/_DATA'), precision='fp16', freeze_betas=True)
    m.load(); m.warmup()
    return m


def select_sources(out):
    """Freeze image membership before fitting; source annotations only label scores."""
    path = out / 'source_manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    origin = json.loads((LATEST / 'analysis_intent/time_origin.json').read_text())['video_wall_origin']
    anns = json.loads((LATEST / 'analysis_intent/annotations.json').read_text())
    log = next(LATEST.glob('perception/*/frames.jsonl.zst'))
    proc = subprocess.Popen(['zstd', '-q', '-dc', str(log)], stdout=subprocess.PIPE, text=True)
    selected = []; last = -1e9
    for line in proc.stdout:
        d = json.loads(line); t = d['timestamp'] - origin
        if t < 42.78118 or t >= 350 or d.get('betas_user') is None or not d.get('bbox'):
            continue
        if selected and t - last < .5:
            continue
        last = t
        ann = next((x for x in anns if x['start'] <= t <= x['end']), None)
        selected.append(dict(split='train', source='20260920', t=t, seq=d['capture_sequence'],
            stamp=d['timestamp'], frame=d['frame_number'], bbox=d['bbox'], betas=d['betas_user'],
            case=ann['kind'] if ann else 'between_actions'))
    proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError('Could not read complete source pose archive')
    cache = ROOT / '.runtime/fixed_crop_replay_20260920/full/source_png'
    own = out / 'source_png'; own.mkdir(exist_ok=True)
    missing = []
    for row in selected:
        p = cache / f'{row["seq"]}.png'
        if not p.exists(): p = own / f'{row["seq"]}.png'
        row['file'] = str(p)
        if not p.exists(): missing.append(row)
    if missing:
        temp = out / 'temporary_source_rgbd.db3'
        if temp.exists():
            raise RuntimeError('Temporary RGB-D file already exists; inspect before reuse')
        print('Extracting missing exact RGB frames from preserved RGB-D', len(missing), flush=True)
        reader = None
        try:
            with temp.open('xb') as f:
                subprocess.run(['zstd', '-q', '-dc', str(LATEST / 'rgbd.db3.zst')], stdout=f, check=True)
            reader = RecordedRGBD(temp)
            for row in missing:
                rgb = source_rgb(reader, row)
                if not cv2.imwrite(row['file'], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError('Failed to save source RGB frame')
        finally:
            if reader is not None: reader.close()
            temp.unlink(missing_ok=True)
    for split, name, sub in [('validation', 'hand_motion_observation_20260911T153019_6Wdm', 'analysis_motion_intent'),
                              ('test', 'hand_motion_observation_20260911T202058_Ime9', 'analysis_feedback')]:
        folder = EARLIER / name / sub / 'selected_rgbd'
        manifest = json.loads((folder / 'manifest.json').read_text())
        rows = manifest if isinstance(manifest, list) else manifest['rows']
        for x in rows:
            selected.append(dict(split=split, source=name, analysis=str(folder.parent),
                t=x['t'], seq=x['seq'], stamp=x.get('actual_t', x['t']),
                bbox=x['bbox'], betas=x['betas'], case=x['case'], file=str(folder / x['file'])))
    path.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    return selected


def camera_encoding(affine, image_margin=0):
    # Inverse crop transform includes fixed-left reflection. Preserve the
    # actual 192x256 image corners, not the discarded 32-pixel side margins.
    q = np.array([[128., 128.], [32., 0.], [223., 0.], [223., 255.], [32., 255.]])
    original = cv2.transform(q[None].astype(np.float64), cv2.invertAffineTransform(np.asarray(affine)))[0] - image_margin
    angles = np.arctan((original - K[:2, 2]) / np.array([K[0, 0], K[1, 1]]))
    phase = angles[:, :, None] * 2. ** np.arange(4)[None, None, :]
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).reshape(-1).astype(np.float32)


def cache_features(args):
    rows = select_sources(args.out)
    runner = model(); captured = []
    def hook(_module, _inputs, output): captured.append(output.detach().cpu().numpy()[0].astype(np.float16))
    handle = runner._model.backbone.register_forward_hook(hook)
    packs = {}; failures = []
    for i, row in enumerate(rows):
        p = Path(row['file'])
        rgb = np.load(p)['rgb'] if p.suffix == '.npz' else cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        captured.clear()
        try:
            runner.set_frozen_betas(False, np.array(row['betas']))
            pred = runner.infer(rgb, row['bbox'], False, row['stamp'])
            palm = build_hamer_five_point_palm_frame(pred.pred_keypoints_3d_source_camera_axes, False)
            if not palm.valid: raise ValueError(palm.failure_reason)
            if len(captured) != 1: raise RuntimeError('Backbone feature capture was not one forward pass')
        except (ValueError, RuntimeError) as exc:
            failures.append(dict(row=row, reason=str(exc)))
            if isinstance(exc, torch.cuda.OutOfMemoryError): raise
            continue
        pack = packs.setdefault(row['split'], dict(features=[], encoding=[], local_factor=[], raw_palm=[], metadata=[]))
        # Keeping the original articulation: changing native global G rotates
        # the complete raw hand by M G_new G_old^T M. No reflection is a rotation.
        pack['features'].append(captured[0])
        pack['encoding'].append(camera_encoding(pred.quality['affine_original_to_crop']))
        pack['local_factor'].append(pred.global_orient.T @ M @ palm.rotation)
        pack['raw_palm'].append(palm.rotation)
        pack['metadata'].append(row)
        if i % 80 == 0: print('features', i, '/', len(rows), row['split'], flush=True)
    handle.remove()
    for split, p in packs.items():
        np.savez_compressed(args.out / f'{split}_features.npz', **{k:np.asarray(v)for k,v in p.items()if k!='metadata'})
        (args.out / f'{split}_metadata.json').write_text(json.dumps(p['metadata'], ensure_ascii=False, indent=2))
    (args.out / 'input_failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    print('Cached membership', {k:len(v['metadata'])for k,v in packs.items()}, 'failures', len(failures), flush=True)


class CameraAdapter(torch.nn.Module):
    def __init__(self, use_camera):
        super().__init__(); self.use_camera = use_camera
        self.net = torch.nn.Sequential(torch.nn.Linear(80, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1280))
        torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, encoding):
        e = encoding if self.use_camera else torch.zeros_like(encoding)
        bias = self.net(e)
        return x + bias[:, :, None, None], bias


def reference(metadata):
    source = Path(metadata[0]['analysis']) if metadata[0].get('analysis') else LATEST / 'analysis_intent'
    imu = np.load(source / 'imu_signals.npz')
    offset = json.loads((source / 'time_alignment.json').read_text())['used_offset_s']
    t = np.array([x['t']for x in metadata]) + offset
    ix = np.clip(np.searchsorted(imu['t'], t), 1, len(imu['t']) - 1)
    valid = (t >= imu['t'][0]) & (t <= imu['t'][-1]) & imu['time_valid'][ix] & imu['time_valid'][ix-1]
    q = Slerp(imu['t'], R.from_quat(imu['q']))(np.clip(t, imu['t'][0], imu['t'][-1])).as_matrix()
    return q, valid


def load_pack(out, split):
    z = np.load(out / f'{split}_features.npz')
    meta = json.loads((out / f'{split}_metadata.json').read_text())
    imu, valid = reference(meta)
    return dict(data={k:torch.from_numpy(z[k].astype(np.float32)if k!='features'else z[k])for k in z.files},
        meta=meta, imu=imu, valid=valid)


def predict(head, adapter, pack, batch_size=8):
    parts = []; clock = time.perf_counter(); reflection = torch.tensor(M, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for first in range(0, len(pack['meta']), batch_size):
            ix = slice(first, first + batch_size)
            x = pack['data']['features'][ix].cuda().float(); enc = pack['data']['encoding'][ix].cuda()
            if adapter is not None: x, _ = adapter(x, enc)
            G = head(x)[0]['global_orient'][:, 0]
            P = reflection @ G @ pack['data']['local_factor'][ix].cuda()
            parts.append(P.cpu().numpy())
    return np.concatenate(parts), time.perf_counter() - clock


def horizon_pairs(pack, horizon):
    """Offline relative-rotation targets; no temporal input at inference."""
    times = np.array([x['t'] for x in pack['meta']]); valid = pack['valid']
    if horizon == 1:
        i = np.flatnonzero(valid[:-2] & valid[2:] & (times[2:] - times[:-2] < 1.3))
        return np.column_stack([i, i + 2])
    pairs = []
    for i in np.flatnonzero(valid):
        j = int(np.argmin(abs(times - times[i] - horizon)))
        if j > i and valid[j] and abs(times[j] - times[i] - horizon) <= .3:
            pairs.append((i, j))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def horizon_score(pred, pack):
    results = {}
    for horizon in [1, 5, 10]:
        pairs = horizon_pairs(pack, horizon)
        if not len(pairs):
            results[str(horizon)] = dict(pairs=0, mean_deg=None, p95_deg=None)
            continue
        a, b = pairs.T
        pr = MOUNT @ np.swapaxes(pred[a], 1, 2) @ pred[b] @ MOUNT.T
        ir = np.swapaxes(pack['imu'][a], 1, 2) @ pack['imu'][b]
        error = np.rad2deg(R.from_matrix(np.swapaxes(ir, 1, 2) @ pr).magnitude())
        results[str(horizon)] = dict(pairs=len(pairs), mean_deg=float(error.mean()),
            p95_deg=float(np.percentile(error, 95)))
    return results


def score(pred, pack):
    categories = {}; errors = []
    # The old manifests contain preselected contiguous cases. Labels are used
    # only after prediction, never to choose a runtime equation or coefficient.
    for case in sorted({x['case']for x in pack['meta']}):
        ix = np.array([i for i,x in enumerate(pack['meta'])if x['case']==case and pack['valid'][i]])
        if len(ix) < 3: continue
        ii = pack['imu'][ix]; pp = pred[ix]
        ai = R.from_matrix(ii[0].T @ ii).magnitude()
        ap = R.from_matrix(pp @ pp[0].T).magnitude()
        e = np.rad2deg(abs(ai - ap)); errors.extend(e.tolist())
        categories[case] = dict(n=len(ix), magnitude_mae_deg=float(e.mean()),
            imu_peak_deg=float(np.rad2deg(ai.max())), predicted_peak_deg=float(np.rad2deg(ap.max())))
    times=np.array([x['t']for x in pack['meta']]);valid=pack['valid']
    paired=np.flatnonzero(valid[:-2]&valid[2:]&(times[2:]-times[:-2]<1.3))
    if len(paired):
        pdelta=np.swapaxes(pred[paired],1,2)@pred[paired+2]
        idelta=np.swapaxes(pack['imu'][paired],1,2)@pack['imu'][paired+2]
        inc=np.rad2deg(R.from_matrix(np.swapaxes(idelta,1,2)@MOUNT@pdelta@MOUNT.T).magnitude())
    else:inc=np.array([np.nan])
    zero=int(np.flatnonzero(valid)[0])
    idx=np.array([i for i,x in enumerate(pack['meta'])if valid[i] and x.get('role')!='calibration'])
    ir=pack['imu'][zero].T@pack['imu'][idx]
    pr=MOUNT@pred[zero].T@pred[idx]@MOUNT.T
    full=np.rad2deg(R.from_matrix(np.swapaxes(ir,1,2)@pr).magnitude())
    horizons = horizon_score(pred, pack)
    balanced = [v['mean_deg'] for v in horizons.values() if v['pairs']]
    return dict(magnitude_mae_deg=float(np.mean(errors)), magnitude_p95_deg=float(np.percentile(errors,95)),
        full_increment_mae_deg=float(np.mean(inc)),full_increment_p95_deg=float(np.percentile(inc,95)),
        full_increment_pairs=len(paired),single_zero_time_s=float(times[zero]),
        single_zero_full_mae_deg=float(full.mean()),single_zero_full_p95_deg=float(np.percentile(full,95)),
        horizon_errors=horizons,balanced_horizon_mae_deg=float(np.mean(balanced)),
        categories=categories,scope='Magnitude plus full 3-D comparisons using fixed nominal IMU mounting; mounting is not independently calibrated')


def synthetic_test(runner, adapters, out):
    """Held-out known rotations; no checkpoint selection or parameter update."""
    source = ROOT / 'docs/viewpoint_pose_cross_20260921'
    cases = json.loads((source / 'results.json').read_text())
    scene = np.load(source / 'frozen_scene.npz')
    runner._model.backbone.cuda(); runner._model.mano.cuda()
    captured = []
    hook = runner._model.backbone.register_forward_hook(lambda _m, _i, x: captured.append(x.detach().float()))
    reflection = torch.tensor(M, dtype=torch.float32, device='cuda')
    results = []; references = {}
    for row in cases:
        rgb = cv2.cvtColor(cv2.imread(str(source / row['image'])), cv2.COLOR_BGR2RGB)
        rgb = cv2.copyMakeBorder(rgb,160,160,160,160,cv2.BORDER_CONSTANT,value=(214,219,224))
        runner.set_frozen_betas(False, scene['betas']); captured.clear()
        pred = runner.infer(rgb, row['input_box'], False, 0.)
        palm = build_hamer_five_point_palm_frame(pred.pred_keypoints_3d_source_camera_axes, False)
        if not palm.valid: raise ValueError(palm.failure_reason)
        local = torch.tensor(pred.global_orient.T @ M @ palm.rotation, dtype=torch.float32, device='cuda')
        encoding = torch.tensor(camera_encoding(pred.quality['affine_original_to_crop'],160), device='cuda')[None]
        result = dict(kind=row['kind'], view_deg=row['value_deg'], true_yaw_deg=row['true_yaw_deg'], methods={})
        with torch.no_grad():
            for name, adapter in [('baseline', None), ('horizontal_baseline', None)] + list(adapters.items()):
                x = captured[0]
                if adapter is not None: x, _ = adapter(x, encoding)
                G = runner._model.mano_head(x)[0]['global_orient'][0,0]
                current = (reflection @ G @ local).cpu().numpy()
                if name == 'horizontal_baseline':
                    current = horizontal_baseline(current[None], [dict(bbox=row['source_box'])])[0]
                if not results: references[name] = current
                error = R.from_matrix(np.asarray(row['true_rotation']).T @ current @ references[name].T).magnitude()
                result['methods'][name] = dict(rotation=current.tolist(), full_relative_error_deg=float(np.rad2deg(error)))
        results.append(result)
    hook.remove()
    for row in results:
        center = next(x for x in results if x['true_yaw_deg']==row['true_yaw_deg'] and x['view_deg']==0)
        for name, item in row['methods'].items():
            err = R.from_matrix(np.array(item['rotation']) @ np.array(center['methods'][name]['rotation']).T).magnitude()
            item['same_pose_position_error_deg'] = float(np.rad2deg(err))
    (out / 'synthetic_test_predictions.json').write_text(json.dumps(results, indent=2))
    edges = [x for x in results if x['view_deg']!=0]
    return {name:{key:float(np.mean([r['methods'][name][key]for r in edges]))
        for key in ['full_relative_error_deg','same_pose_position_error_deg']}
        for name in references}


def train(args):
    # Test data are not loaded until both checkpoints are selected.
    data = args.data or args.out
    tr = load_pack(data, 'train'); va = load_pack(data, 'validation')
    runner = model(); head = runner._model.mano_head.float().eval()
    for p in runner._model.parameters(): p.requires_grad_(False)
    runner._model.backbone.cpu(); runner._model.mano.cpu(); torch.cuda.empty_cache()
    base_train, _ = predict(head, None, tr); base_val, _ = predict(head, None, va)
    target = base_train[0] @ MOUNT.T @ tr['imu'][0].T @ tr['imu'] @ MOUNT
    target = torch.from_numpy(target.astype(np.float32))
    usable = np.flatnonzero(tr['valid'])
    pair_target = None; pair_groups = None; pairs = None; epoch_pairs = None
    if args.loss in ['increments', 'multihorizon']:
        horizons = [1] if args.loss == 'increments' else [1, 5, 10]
        pair_groups = [horizon_pairs(tr, h) for h in horizons]
        epoch_pairs = len(pair_groups[0])
        pairs = np.concatenate(pair_groups)
        usable = np.arange(len(pairs))
        a, b = pairs.T
        pair_target = torch.from_numpy((MOUNT.T @ np.swapaxes(tr['imu'][a],1,2) @ tr['imu'][b] @ MOUNT).astype(np.float32))
    if not tr['valid'][0]: raise ValueError('Original C must have a valid IMU reference')
    data_protocol=json.loads((data/'data_protocol.json').read_text())if (data/'data_protocol.json').exists()else None
    summaries = {'baseline_validation':score(base_val, va),
        'horizontal_baseline_validation':score(horizontal_baseline(base_val,va['meta']),va), 'variants':{},
        'protocol':dict(train=tr['meta'][0]['source'],selection=va['meta'][0]['source'],
            test='fixed source manifest test membership, opened only after checkpoint selection',
            adapter='80-dimensional center/corner camera-angle encoding -> 64 -> 1280 additive decoder context',
            frozen='image backbone, pose decoder, original articulation and shape',
            reference='IMU provisional truth; original fixed nominal mounting axes, no per-motion fitting',
            limitation='Small single-operator prototype; not full published HaMeR+KPE; only global orientation is adapted',
            learning_rate=args.lr, epochs=args.epochs, batch_size=8, seed=21,
            supervision=args.loss, feature_data=str(data),selection_metric=args.selection,
            chronological_data_protocol=data_protocol,
            pairs_per_epoch=epoch_pairs,
            available_pairs_by_horizon=dict(zip(horizons,[len(p) for p in pair_groups])) if pair_groups else None,
            horizon_sampling='One-second: all pairs once per epoch. Multihorizon: equal numbers from 1, 5, 10 seconds, sampled without replacement; same total pairs and updates per epoch.',
            prior_benchmark_exposure=args.development_reuse or (args.loss=='increments' and data_protocol is None),
            supervision_note='Diagnostic on previously inspected development data' if args.development_reuse or (args.loss=='increments' and data_protocol is None) else 'Fixed membership before model fitting')}
    reflection = torch.tensor(M, dtype=torch.float32, device='cuda')
    for name, camera in [('constant_control', False), ('camera_conditioned', True)]:
        torch.manual_seed(21); np.random.seed(21)
        adapter = CameraAdapter(camera).cuda(); opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
        history = []; best = float('inf'); best_epoch = None
        for epoch in range(args.epochs + 1):
            if epoch:
                if args.loss == 'multihorizon':
                    order_parts = []; offset = 0
                    for k, group in enumerate(pair_groups):
                        count = epoch_pairs // 3 + (k < epoch_pairs % 3)
                        order_parts.append(offset + np.random.choice(len(group), count, replace=False))
                        offset += len(group)
                    order = np.random.permutation(np.concatenate(order_parts))
                else:
                    order = np.random.permutation(usable)
                losses = []
                for first in range(0, len(order), 8):
                    ix = order[first:first+8]
                    inputs = np.r_[pairs[ix,0],pairs[ix,1]] if pair_target is not None else ix
                    x = tr['data']['features'][inputs].cuda().float(); enc = tr['data']['encoding'][inputs].cuda()
                    x, bias = adapter(x, enc); G = head(x)[0]['global_orient'][:,0]
                    palm = reflection @ G @ tr['data']['local_factor'][inputs].cuda()
                    if pair_target is not None:
                        rel = palm[:len(ix)].transpose(1,2) @ palm[len(ix):]
                        loss = (rel - pair_target[ix].cuda()).square().mean()
                    else:
                        loss = (palm - target[ix].cuda()).square().mean()
                    loss = loss + 1e-4 * bias.square().mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
            pred, _ = predict(head, adapter, va)
            metric = score(pred, va); value = metric[args.selection]
            history.append(dict(epoch=epoch, train_loss=float(np.mean(losses))if epoch else None, validation=metric))
            if value < best:
                best = value; best_epoch = epoch
                torch.save(adapter.state_dict(), args.out / f'{name}.pt')
            (args.out / f'{name}_history.json').write_text(json.dumps(history, indent=2))
            print(name, 'epoch', epoch, 'validation',args.selection,round(value,4),'best epoch',best_epoch,flush=True)
        summaries['variants'][name] = dict(best_epoch=best_epoch, validation_selection_value=best,
            validation_selection_metric=args.selection,validation=history[best_epoch]['validation'])
    # Evaluate the reserved subset after selection; protocol records prior reuse.
    test = load_pack(data, 'test'); baseline, elapsed = predict(head, None, test)
    summaries['baseline_test'] = score(baseline, test)
    horizontal = horizontal_baseline(baseline,test['meta'])
    summaries['horizontal_baseline_test'] = score(horizontal,test)
    predictions = dict(baseline=baseline, horizontal_baseline=horizontal); fitted = {}
    validation_predictions = dict(baseline=base_val)
    summaries['baseline_train'] = score(base_train, tr)
    for name, camera in [('constant_control',False),('camera_conditioned',True)]:
        adapter = CameraAdapter(camera).cuda(); adapter.load_state_dict(torch.load(args.out / f'{name}.pt'))
        fitted[name] = adapter
        pred, elapsed = predict(head, adapter, test)
        predictions[name] = pred; summaries['variants'][name]['test'] = score(pred, test)
        summaries['variants'][name]['decoder_batch_evaluation_seconds'] = elapsed
        validation_predictions[name], _ = predict(head, adapter, va)
        train_pred, _ = predict(head, adapter, tr)
        summaries['variants'][name]['train'] = score(train_pred, tr)
    np.savez_compressed(args.out / 'test_predictions.npz', **predictions,
        imu=test['imu'], imu_valid=test['valid'], t=[x['t']for x in test['meta']])
    np.savez_compressed(args.out / 'validation_predictions.npz', **validation_predictions,
        imu=va['imu'], imu_valid=va['valid'], t=[x['t']for x in va['meta']])
    summaries['synthetic_test'] = synthetic_test(runner, fitted, args.out)
    (args.out / 'metrics.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


def task_replay(args):
    if args.checkpoint_dir is None:raise ValueError('Task replay requires frozen checkpoints')
    data=args.data or args.out;z=np.load(data/'task_features.npz')
    metadata=json.loads((data/'task_metadata.json').read_text())
    pack=dict(meta=metadata,data={k:torch.from_numpy(z[k].astype(np.float32)if k!='features'else z[k])for k in z.files})
    runner=model();head=runner._model.mano_head.float().eval()
    for p in runner._model.parameters():p.requires_grad_(False)
    runner._model.backbone.cpu();runner._model.mano.cpu();torch.cuda.empty_cache()
    poses={};poses['baseline'],_=predict(head,None,pack)
    poses['horizontal_baseline']=horizontal_baseline(poses['baseline'],metadata)
    for name,camera in [('constant_control',False),('camera_conditioned',True)]:
        adapter=CameraAdapter(camera).cuda();adapter.load_state_dict(torch.load(args.checkpoint_dir/f'{name}.pt'))
        poses[name],_=predict(head,adapter,pack)
    ts=np.array([x['t']for x in metadata]);dt=np.diff(ts);native=(dt>.005)&(dt<.2)
    baseline_relative=poses['baseline']@poses['baseline'][0].T
    summary=dict(source=metadata[0]['source'],frames=len(ts),native_interval_pairs=int(native.sum()),
        checkpoints=str(args.checkpoint_dir),ground_truth_available=False,methods={},
        scope='Output stability stress replay. Smaller step sizes are not proof of accuracy; detector and controller were not rerun.')
    for name,q in poses.items():
        step=np.rad2deg(R.from_matrix(q[1:]@np.swapaxes(q[:-1],1,2)).magnitude())[native]
        relative=q@q[0].T
        change=np.rad2deg(R.from_matrix(np.swapaxes(baseline_relative,1,2)@relative).magnitude())
        summary['methods'][name]=dict(native_step_mean_deg=float(step.mean()),native_step_p95_deg=float(np.percentile(step,95)),
            native_step_max_deg=float(step.max()),steps_above_10deg=int((step>10).sum()),
            relative_output_difference_from_baseline_mean_deg=float(change.mean()),
            relative_output_difference_from_baseline_p95_deg=float(np.percentile(change,95)))
    np.savez_compressed(args.out/'task_predictions.npz',t=ts,**poses)
    (args.out/'task_metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['cache','train','task'])
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--data',type=Path); ap.add_argument('--loss',choices=['absolute','increments','multihorizon'],default='absolute')
    ap.add_argument('--selection',choices=['magnitude_mae_deg','full_increment_mae_deg','balanced_horizon_mae_deg'],default='magnitude_mae_deg')
    ap.add_argument('--checkpoint-dir',type=Path)
    ap.add_argument('--development-reuse',action='store_true')
    ap.add_argument('--lr', type=float, default=.001); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1); torch.set_num_threads(2)
    {'cache':cache_features,'train':train,'task':task_replay}[a.mode](a)
