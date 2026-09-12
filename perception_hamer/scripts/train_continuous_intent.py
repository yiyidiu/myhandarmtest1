#!/usr/bin/env python3
"""Single small causal network, three independent continuous angular outputs.

Training and validation retain their recording split. Existing combined
motion clips are held out. The loss measures angular increments that would
be removed/retained, not a mutually exclusive action classification score.
No motion is sent to ROS.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import numpy as np
import torch
from torch import nn

from evaluate_continuous_intent import AXES, ROOT, SOURCE, retention_threshold, segment_metrics

sys.path.insert(0, str(SOURCE))
from tcn_vendor import TemporalConvNet


OUTPUT = ROOT / '.runtime/continuous_intent_20260912'
WINDOW = 16


class ContinuousIntent(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = TemporalConvNet(input_size, [24, 24], kernel_size=3, dropout=.15)
        self.head = nn.Linear(24, 3)

    def forward(self, x):
        return torch.sigmoid(self.head(self.encoder(x)[:, :, -1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-features', action='store_true',
                        help='compare temporal motion statistics with the earlier full pose/image features')
    args = parser.parse_args()
    name = 'motion_tcn' if args.motion_features else 'continuous_tcn'
    torch.set_num_threads(2)
    torch.manual_seed(19)
    np.random.seed(19)
    loaded = np.load(SOURCE / 'observed_intent_feature_dataset.npz')
    data = {key: loaded[key] for key in loaded.files}
    binding = np.load(OUTPUT / 'continuous_signal_binding.npz')
    if args.motion_features:
        for split in ['train', 'validation', 'test']:
            data[split + '_x'] = binding[split + '_features']
    specification = json.loads((SOURCE / 'intent_data_definition.json').read_text())
    classes = specification['classes']
    known = (data['train_y'] >= 0) & (data['train_y'] < classes.index('combined')) & binding['train_valid']
    mean = np.mean(data['train_x'][known], axis=0)
    std = np.maximum(.03, np.std(data['train_x'][known], axis=0))
    sets = {}
    for split in ['train', 'validation', 'test']:
        x = np.clip((data[split + '_x'] - mean) / std, -8., 8.)
        windows = np.stack([x[np.maximum(0, np.arange(i - WINDOW + 1, i + 1))] for i in range(len(x))])
        labels = data[split + '_y']
        target = np.column_stack([labels == classes.index(name) for name in AXES]).astype(np.float32)
        valid = (labels >= 0) & (labels < classes.index('combined')) & binding[split + '_valid']
        # An active-axis loss is proportional to the motion that a wrong
        # prediction would remove. Balance clips rather than adjacent frames.
        amount = np.abs(binding[split + '_delta'])
        weights = np.zeros_like(amount)
        for group in np.unique(data[split + '_group'][valid]):
            mask = valid & (data[split + '_group'] == group)
            label = classes[int(labels[mask][0])]
            for axis in range(3):
                denominator = amount[mask, axis].sum()
                if denominator > 1e-7:
                    weights[mask, axis] = amount[mask, axis] / denominator
                    if label == AXES[axis]:
                        weights[mask, axis] *= 9.  # deleting intentional motion costs more
        weights *= max(1, valid.sum()) / max(weights.sum(), 1e-9)
        sets[split] = dict(x=torch.tensor(windows.transpose(0, 2, 1), dtype=torch.float32),
                           target=torch.tensor(target), weights=torch.tensor(weights, dtype=torch.float32),
                           valid=valid, labels=labels)
    model = ContinuousIntent(len(mean))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.002)
    best_loss, best_state, best_epoch = float('inf'), None, None
    ids = torch.tensor(np.flatnonzero(sets['train']['valid']))
    history, train_history = [], []
    for epoch in range(80):
        model.train()
        for selected in ids[torch.randperm(len(ids))].split(128):
            optimizer.zero_grad()
            prediction = model(sets['train']['x'][selected])
            loss = ((prediction - sets['train']['target'][selected])**2 * sets['train']['weights'][selected]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_prediction = model(sets['train']['x'])
            train_history.append(float(((train_prediction - sets['train']['target'])**2 * sets['train']['weights']).sum() /
                                       sets['train']['weights'].sum()))
            prediction = model(sets['validation']['x'])
            validation_loss = float(((prediction - sets['validation']['target'])**2 * sets['validation']['weights']).sum() /
                                    sets['validation']['weights'].sum())
        history.append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score = {split: model(s['x']).numpy() for split, s in sets.items()}
    validation_labels = sets['validation']['labels'].copy()
    validation_labels[~binding['validation_valid']] = -1
    thresholds = [retention_threshold(score['validation'][:, axis], np.abs(binding['validation_delta'][:, axis]),
                                     validation_labels, data['validation_group'], classes.index(name))
                  for axis, name in enumerate(AXES)]
    summary = dict(scope=__doc__, deployed=False, mutually_exclusive=False, axes=AXES,
                   selected_epoch=best_epoch, validation_loss=best_loss, validation_history=history,
                   train_history=train_history, input_features='past_motion_statistics' if args.motion_features else 'pose_and_observed_landmarks',
                   thresholds=thresholds, main_motion_loss_cost=9., splits={},
                   limitations=['Intent-axis annotations do not supply exact desired angles',
                                'Previously used development recordings, not a fresh final test',
                                'Three-axis rotation plus translation not independently annotated'])
    for split in sets:
        keep = np.minimum(1., score[split] / np.maximum(np.array(thresholds), 1e-9))
        keep[~binding[split + '_valid']] = 1.
        records = []
        for annotation in specification['annotations'][split]:
            mask = (data[split + '_group'] == annotation['id']) & binding[split + '_valid']
            if not mask.any():
                continue
            axes = [2] if split == 'train' and annotation['label'] == 'combined' else None
            records.append(dict(annotation, **segment_metrics(binding[split + '_delta'], keep, mask, annotation['label'], axes)))
        summary['splits'][split] = records
        np.savez_compressed(OUTPUT / (split + '_' + name + '_predictions.npz'),
                            t=data[split + '_t'], score=score[split], keep=keep,
                            delta=binding[split + '_delta'], quaternion=binding[split + '_quaternion'],
                            valid=binding[split + '_valid'], enabled=binding[split + '_enabled'])
    # Causality is necessary for live feasibility; it does not prove intent accuracy.
    x = torch.randn(1, len(mean), 32)
    changed = x.clone()
    changed[:, :, 16:] = torch.randn_like(changed[:, :, 16:]) * 20.
    with torch.no_grad():
        summary['future_change_max_past_encoder_change'] = float((model.encoder(x)[:, :, :16] - model.encoder(changed)[:, :, :16]).abs().max())
    elapsed = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            model(sets['test']['x'][:1])
        elapsed.append((time.perf_counter() - start) * 1000)
    summary['cpu_inference_ms_p95'] = float(np.percentile(elapsed, 95))
    torch.save(dict(state_dict=best_state, mean=mean, std=std, thresholds=thresholds,
                    window=WINDOW, axes=AXES, deployed=False), OUTPUT / (name + '.pt'))
    (OUTPUT / (name + '_results.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Selected epoch', best_epoch, 'thresholds', thresholds, flush=True)
    for row in summary['splits']['test']:
        print(row['start'], row['label'], row.get('desired_component_retained'), row.get('other_rotation_suppressed'), flush=True)
    print('Held-out yaw+translation', [(r['id'], r.get('desired_component_retained')) for r in summary['splits']['train'] if r['label'] == 'combined'], flush=True)


if __name__ == '__main__':
    main()
