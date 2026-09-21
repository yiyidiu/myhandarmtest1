#!/usr/bin/env python3
"""Train HaMeR's orientation decoder on the fixed current-view experiment.

The only new trainable path is the original transformer decoder and its six
global-orientation readout rows. Original image features, hand articulation,
shape, supervision, sampling budget and validation rule are retained. No
runtime IMU, temporal information or motion labels enter the prediction.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment_hamer_camera_adapter import (
    CameraAdapter, M, MOUNT, horizon_pairs, load_pack, model, predict, score,
)


def enable_orientation_training(head):
    if head.joint_rep_dim != 6 or head.cfg.MODEL.MANO_HEAD.get('IEF_ITERS', 1) != 1:
        raise ValueError('This experiment assumes one iteration and six parameters per joint')
    for p in head.parameters(): p.requires_grad_(False)
    for p in head.transformer.parameters(): p.requires_grad_(True)
    # Keep the original shared readout architecture. The other 90 rows receive
    # exactly zero gradient and Adam has no weight decay. Their predicted
    # articulation is not used: the original per-image local factor is retained.
    head.decpose.weight.requires_grad_(True); head.decpose.bias.requires_grad_(True)
    handles = []
    for p in [head.decpose.weight, head.decpose.bias]:
        mask = torch.zeros_like(p); mask[:6] = 1
        handles.append(p.register_hook(lambda grad, mask=mask: grad * mask))
    return [p for p in head.parameters() if p.requires_grad], handles


def load_candidate(head, path, camera):
    state = torch.load(path, map_location='cpu')
    head.load_state_dict(state['head']); head.eval()
    adapter = CameraAdapter(camera).cuda()
    adapter.load_state_dict(state['adapter']); adapter.eval()
    return adapter


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    tr = load_pack(args.data, 'train'); va = load_pack(args.data, 'validation')
    groups = [horizon_pairs(tr, h) for h in [1, 5, 10]]
    pairs = np.concatenate(groups); a, b = pairs.T; count = len(groups[0])
    target = torch.from_numpy((MOUNT.T @ np.swapaxes(tr['imu'][a], 1, 2) @ tr['imu'][b] @ MOUNT).astype(np.float32))
    protocol = dict(
        hypothesis='Does allowing the original orientation decoder to adapt improve complete pose accuracy beyond the frozen decoder adapter?',
        data=str(args.data), prior_comparison=str(args.frozen_reference),
        changed='Original transformer decoder and six global-orientation output rows are trainable.',
        held_fixed='Original backbone features, original per-image articulation/shape, camera encoding, 1/5/10 s relative loss, regularizer, temporal partitions, reference alignment and epoch selection.',
        methods=['constant_control', 'camera_conditioned'],
        loss='Relative rotation matrix MSE plus 1e-4 times adapter bias squared mean.',
        pairs_per_epoch=count, horizon_pairs={str(h):len(g) for h,g in zip([1,5,10],groups)},
        pair_batch=8, epochs=args.epochs, seed=21,
        adapter_learning_rate=args.lr, pretrained_head_learning_rate=args.head_lr,
        head_lr_scope='Prespecified 1e-5 for both pretrained heads, no sweep or test-based tuning; adapter learning rate remains 1e-3.',
        selection='Validation equal-weight full rotation error at 1/5/10 s, including the unchanged epoch zero.',
        reference='IMU provisional truth with existing fixed nominal mounting and +25ms source-time alignment.',
        test_scope='Previously inspected temporal development test, not a newly blinded dataset.',
        runtime_input='Single-frame image features and optional source-camera crop rays only.',
        deployment='Offline only. No live model or robot configuration is modified.')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    runner = model(); head = runner._model.mano_head.float().eval()
    for p in runner._model.parameters(): p.requires_grad_(False)
    runner._model.backbone.cpu(); runner._model.mano.cpu(); torch.cuda.empty_cache()
    initial = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    base_train, _ = predict(head, None, tr); base_val, _ = predict(head, None, va)
    summaries = dict(protocol=protocol, baseline_train=score(base_train,tr), baseline_validation=score(base_val,va), variants={})
    original = json.loads((args.frozen_reference / 'metrics.json').read_text())
    delta = abs(summaries['baseline_validation']['balanced_horizon_mae_deg'] - original['baseline_validation']['balanced_horizon_mae_deg'])
    if delta > .001: raise ValueError(f'Unexpected original-model baseline discrepancy: {delta} degrees')
    reflection = torch.tensor(M, dtype=torch.float32, device='cuda')
    for name, camera in [('constant_control',False), ('camera_conditioned',True)]:
        head.load_state_dict(initial); head.eval()
        params, handles = enable_orientation_training(head)
        torch.manual_seed(21); np.random.seed(21)
        adapter = CameraAdapter(camera).cuda()
        opt = torch.optim.Adam([dict(params=adapter.parameters(), lr=args.lr), dict(params=params, lr=args.head_lr)])
        print(name, 'pretrained trainable parameter tensors', sum(p.numel() for p in params), 'includes masked readout rows', flush=True)
        history = []; best = float('inf'); best_epoch = None
        for epoch in range(args.epochs + 1):
            started = time.perf_counter(); losses = []
            if epoch:
                parts = []; offset = 0
                for k, group in enumerate(groups):
                    n = count // 3 + (k < count % 3)
                    parts.append(offset + np.random.choice(len(group), n, replace=False)); offset += len(group)
                order = np.random.permutation(np.concatenate(parts))
                for first in range(0, len(order), 8):
                    ix = order[first:first+8]; indices = np.r_[pairs[ix,0],pairs[ix,1]]
                    x = tr['data']['features'][indices].cuda().float(); enc = tr['data']['encoding'][indices].cuda()
                    x, bias = adapter(x,enc); g = head(x)[0]['global_orient'][:,0]
                    palm = reflection @ g @ tr['data']['local_factor'][indices].cuda()
                    rel = palm[:len(ix)].transpose(1,2) @ palm[len(ix):]
                    loss = (rel-target[ix].cuda()).square().mean() + 1e-4*bias.square().mean()
                    if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss in {name}, epoch {epoch}')
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
            pred, _ = predict(head,adapter,va); metric = score(pred,va)
            value = metric['balanced_horizon_mae_deg']
            history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)) if losses else None,
                validation=metric, elapsed_s=time.perf_counter()-started))
            if value < best:
                best = value; best_epoch = epoch
                torch.save(dict(adapter=adapter.state_dict(), head=head.state_dict(), epoch=epoch,
                    use_camera=camera, protocol=protocol), args.out / f'{name}.pt')
            (args.out / f'{name}_history.json').write_text(json.dumps(history,indent=2))
            print(name,'epoch',epoch,'validation',round(value,4),'best',best_epoch,'seconds',round(time.perf_counter()-started,2),flush=True)
        for h in handles: h.remove()
        # This is a targeted implementation check: the 90 articulated-hand
        # output rows must not have become a second changed training variable.
        if not torch.equal(head.decpose.weight[6:].detach().cpu(),initial['decpose.weight'][6:]):
            raise RuntimeError('Articulated-hand readout weights were changed')
        summaries['variants'][name] = dict(best_epoch=best_epoch, validation_selection_value=best,
            validation_selection_metric='balanced_horizon_mae_deg', validation=history[best_epoch]['validation'])
        del opt, params
    test = load_pack(args.data, 'test')
    head.load_state_dict(initial)
    baseline, _ = predict(head,None,test); summaries['baseline_test'] = score(baseline,test)
    predictions = dict(baseline=baseline); validation_predictions = dict(baseline=base_val)
    for name, camera in [('constant_control',False), ('camera_conditioned',True)]:
        ad = load_candidate(head,args.out / f'{name}.pt',camera)
        pred, _ = predict(head,ad,test); predictions[name] = pred
        summaries['variants'][name]['test'] = score(pred,test)
        validation_predictions[name], _ = predict(head,ad,va)
        training_pred, _ = predict(head,ad,tr)
        summaries['variants'][name]['train'] = score(training_pred,tr)
    for split, pack, p in [('test',test,predictions),('validation',va,validation_predictions)]:
        np.savez_compressed(args.out / f'{split}_predictions.npz', **p, imu=pack['imu'], imu_valid=pack['valid'], t=[x['t'] for x in pack['meta']])
    (args.out / 'metrics.json').write_text(json.dumps(summaries,indent=2))
    print('Selected-checkpoint evaluation complete',flush=True)
    for name,v in summaries['variants'].items():
        print(name,'test',v['test']['balanced_horizon_mae_deg'],'same C',v['test']['single_zero_full_mae_deg'],flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--frozen-reference', type=Path, required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--head-lr',type=float,default=.00001)
    cv2.setNumThreads(1); torch.set_num_threads(2)
    main(p.parse_args())
