#!/usr/bin/env python3
"""Offline, independent-axis intent baseline on existing operator recordings.

Outputs continuous retention for all three angular components. It does not
publish robot commands. Operator annotations indicate intended components;
recorded angular motion is a proxy for amplitude, not IMU ground truth.
"""
from pathlib import Path
import argparse
import gzip
import json

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '.runtime/hand_motion_observation_20260912T145617_iJYZ/analysis_feedback'
AXES = ['pitch', 'roll', 'yaw']  # camera X, Y, Z, before the existing mapping


def image_motion_samples(rows):
    """Observed global flow and fingertip motion relative to the wrist."""
    uv=np.asarray([r['xy'] for r in rows],dtype=float)
    times=np.asarray([r['t'] for r in rows],dtype=float)
    centre=uv[:,[5,9,13,17]].mean(1)
    scale=np.maximum(10.,np.linalg.norm(centre-uv[:,0],axis=1))
    global_points=np.concatenate([uv[:,0],centre],axis=1)
    relative=(uv[:,[8,12,16,20]]-uv[:,:1])/scale[:,None,None]
    derivative=np.zeros((len(rows),12),dtype=float)
    dt=np.diff(times)
    derivative[1:,:4]=np.diff(global_points,axis=0)/dt[:,None]/((scale[1:]+scale[:-1])/2)[:,None]
    derivative[1:,4:]=np.diff(relative.reshape(len(rows),-1),axis=0)/dt[:,None]
    derivative[np.r_[False,dt>.30]]=0.
    return np.clip(np.nan_to_num(derivative),-10.,10.)


def motion_statistics(measured, observed, fresh, image_flow=None):
    features=[]
    for i in range(len(measured)):
        recent=measured[max(0,i-8):i+1]
        energy=np.mean(np.abs(recent),0);signed=np.mean(recent,0);net=np.abs(signed)
        image_recent=observed[max(0,i-8):i+1]
        row=np.r_[energy,net,net/np.maximum(energy,1e-5),
                  np.mean(np.abs(image_recent),0),np.abs(np.mean(image_recent,0)),fresh[i]]
        if image_flow is not None:
            flow=image_flow[max(0,i-8):i+1]
            flow_mean=np.mean(flow,0)
            cross=np.mean(recent[:,3:,None]*image_recent[:,None,:4],axis=0)
            norm=np.sqrt(np.mean(recent[:,3:]**2,axis=0)[:,None]*np.mean(image_recent[:,:4]**2,axis=0)[None,:])
            row=np.r_[row,signed,np.mean(image_recent,0),flow_mean,
                      np.abs(flow_mean)/np.maximum(np.mean(np.abs(flow),0),1e-5),
                      (cross/np.maximum(norm,1e-8)).ravel()]
        features.append(row)
    return np.asarray(features)


def past_features(data, split, source, visual_motion=False):
    """Reuse the previous temporal baseline's features; no future samples."""
    rows = [r for r in json.loads((source / (split + '_observed_finger_cues.json')).read_text())['rows'] if r['valid']]
    times = np.array([r['t'] for r in rows])
    cues = []
    for row in rows:
        xyz, uv = np.array(row['xyz']), np.array(row['xy'])
        z = xyz[:, 2]
        palm = np.nanmedian(z[[5, 9, 13, 17]])
        long_axis = uv[[5, 9, 13, 17]].mean(0) - uv[0]
        cues.append([
            np.nanmedian(z[[6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]]) - palm,
            palm - z[0], z[17] - z[5], np.arctan2(long_axis[1], long_axis[0]),
            np.log(max(10., np.linalg.norm(uv[17] - uv[5]))),
            np.log(max(10., np.linalg.norm(long_axis))),
        ])
    cues = np.array(cues)
    cues[:, 3] = np.unwrap(cues[:, 3])
    speed = np.vstack([np.zeros(6), np.diff(cues, axis=0) / np.diff(times)[:, None]])
    speed = np.clip(np.nan_to_num(speed), [-.5, -.5, -.5, -5, -3, -3], [.5, .5, .5, 5, 3, 3])
    grid = data[split + '_t']
    index = np.maximum(0, np.searchsorted(times, grid, side='right') - 1)
    fresh = (grid >= times[0]) & (grid - times[index] < .30)
    observed = speed[index].copy()
    observed[~fresh] = 0.
    flow=image_motion_samples(rows)[index] if visual_motion else None
    if flow is not None:
        flow[~fresh]=0.
    return motion_statistics(data[split+'_x'][:,:6],observed,fresh,flow)


def published_motion(data, split, specification):
    _, session, folder = next(s for s in specification['sources'] if s[0] == split)
    path = ROOT / '.runtime' / session / folder / 'perception_signals.json.gz'
    with gzip.open(path, 'rt') as stream:
        rows = [r for r in json.load(stream) if r.get('published_q')]
    times = np.array([r['t'] for r in rows])
    grid = data[split + '_t']
    index = np.maximum(0, np.searchsorted(times, grid, side='right') - 1)
    q = np.array([rows[i]['published_q'] for i in index])
    rotation = Rotation.from_quat(q)
    delta = np.vstack([np.zeros(3), (rotation[1:] * rotation[:-1].inv()).as_rotvec()])
    enabled = np.array([rows[i]['control_enabled'] for i in index])
    valid = (grid >= times[0]) & (grid - times[index] < .18)
    valid &= np.array([rows[i]['published_valid'] and min(rows[i]['confidence'][3:]) > 0 for i in index])
    return dict(delta=delta, quaternion=q, enabled=enabled, valid=valid)


def command_angle_binding(data, split, specification):
    """Decode the SAME ordered angle channels consumed by RelativePoseMapper.

    Returned order is pitch, roll, yaw before their unchanged gains. This
    does not identify a camera angular-velocity component with an Euler angle.
    """
    import yaml
    mapping = yaml.safe_load((ROOT / 'src/handarm_moveit_demo/config/shared_teleop.yaml').read_text())['mapping']
    axes = np.asarray(mapping['camera_rotation_to_tool_zero'], dtype=float)
    _, session, folder = next(s for s in specification['sources'] if s[0] == split)
    with gzip.open(ROOT / '.runtime' / session / folder / 'perception_signals.json.gz', 'rt') as stream:
        rows = [r for r in json.load(stream) if r.get('published_q')]
    grid = data[split + '_t']
    index = np.maximum(0, np.searchsorted([r['t'] for r in rows], grid, side='right') - 1)
    decoded, epochs, references, positions = [], [], [], []
    # Before C there is no robot pose command. Use the first actual C frame
    # only as the offline label coordinate frame for pre-C static examples.
    # It is never fed to the causal motion-feature encoder. An arbitrary
    # initial camera pose followed by a C reset would create a fictitious
    # large angular step inside an otherwise stationary training clip.
    first_reference = next((r for r in rows if r['control_enabled']), rows[0])
    reference = Rotation.from_quat(first_reference['published_q']).as_matrix()
    reference_position = np.asarray(first_reference['position'], dtype=float)
    token, epoch = None, 0
    values = []
    # Bind C at its original input sample before resampling onto the grid.
    for row in rows:
        if row['control_enabled'] and row['reference_token'] != token:
            token = row['reference_token']
            reference = Rotation.from_quat(row['published_q']).as_matrix()
            reference_position = np.asarray(row['position'], dtype=float)
            epoch += 1
        current = Rotation.from_quat(row['published_q']).as_matrix()
        mapped = axes @ current @ reference.T @ axes.T
        tilt = np.arcsin(np.clip(mapped[0, 2], -1., 1.))
        if abs(np.cos(tilt)) > 1e-8:
            side = np.arctan2(-mapped[1, 2], mapped[2, 2])
            roll = np.arctan2(-mapped[0, 1], mapped[0, 0])
        else:
            side = np.arctan2(mapped[2, 1], mapped[1, 1])
            roll = 0.
        values.append(([-tilt, -roll, side], epoch, Rotation.from_matrix(reference).as_quat(),
                       np.asarray(row['position'],dtype=float)-reference_position))
    for i in index:
        a, e, q, p = values[i]
        decoded.append(a)
        epochs.append(e)
        references.append(q)
        positions.append(p)
    decoded, epochs = np.array(decoded), np.array(epochs)
    for e in np.unique(epochs):
        mask = epochs == e
        for axis in [1, 2]:
            decoded[mask, axis] = np.unwrap(decoded[mask, axis])
    return dict(angles=decoded, epoch=epochs, reference=np.array(references),
                position_delta=np.asarray(positions), camera_axes=axes)


def retention_threshold(score, amount, labels, groups, desired_label):
    """Largest threshold retaining >=99% angular travel in EACH valid clip.

    This is an engineering comparison tolerance, not a claim about perception.
    It is selected on the earlier validation recording only.
    """
    masks = [(labels == desired_label) & (groups == g) for g in np.unique(groups[labels == desired_label])]
    masks = [m for m in masks if amount[m].sum() > 1e-9]
    if not masks:
        return 1e-9
    low, high = 1e-9, 1.
    for _ in range(50):
        threshold = (low + high) / 2
        keep = np.minimum(1., score / threshold)
        if all(np.sum(amount[m] * keep[m]) / amount[m].sum() >= .99 for m in masks):
            low = threshold
        else:
            high = threshold
    return low


def segment_metrics(delta, keep, mask, label, intended_axes=None):
    before, after = delta[mask], delta[mask] * keep[mask]
    angular_before = np.linalg.norm(before, axis=1).sum()
    result = dict(n=int(mask.sum()), angular_travel_before_deg=float(np.rad2deg(angular_before)))
    if label in AXES:
        intended_axes = [AXES.index(label)]
    if intended_axes:
        result['desired_component_retained'] = {
            AXES[axis]: float(np.abs(after[:, axis]).sum() / max(np.abs(before[:, axis]).sum(), 1e-9))
            for axis in intended_axes}
    if label in AXES or label in ['translate', 'still']:
        other = [a for a in range(3) if label not in AXES or a != AXES.index(label)]
        unwanted = np.linalg.norm(before[:, other], axis=1).sum()
        result['other_rotation_suppressed'] = float(1. - np.linalg.norm(after[:, other], axis=1).sum() / max(unwanted, 1e-9))
    result['minimum_retention_by_axis'] = np.min(keep[mask], axis=0).tolist() if mask.any() else []
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.runtime/continuous_intent_20260912')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE / 'intent_feature_dataset.npz')
    specification = json.loads((SOURCE / 'intent_data_definition.json').read_text())
    classes = specification['classes']
    features = {s: past_features(data, s, SOURCE) for s in ['train', 'validation', 'test']}
    motion = {s: published_motion(data, s, specification) for s in features}
    # Keep the existing simultaneous yaw+translation clips OUT of training.
    train_mask = (data['train_y'] >= 0) & (data['train_y'] != classes.index('combined'))
    train_mask &= motion['train']['valid']
    models, thresholds = [], []
    scores = {s: [] for s in features}
    for axis, name in enumerate(AXES):
        target = (data['train_y'] == classes.index(name)).astype(int)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced', max_iter=1500, random_state=19))
        model.fit(features['train'][train_mask], target[train_mask])
        models.append(model)
        for split in features:
            scores[split].append(model.predict_proba(features[split])[:, 1])
        validation_labels = data['validation_y'].copy()
        validation_labels[~motion['validation']['valid']] = -1
        threshold = retention_threshold(scores['validation'][-1], np.abs(motion['validation']['delta'][:, axis]),
                                        validation_labels, data['validation_group'], classes.index(name))
        thresholds.append(threshold)
    summary = dict(scope=__doc__, axes=AXES, model='three independent logistic heads, same past-frame features',
                   mutually_exclusive=False, thresholds=thresholds, threshold_selection='earlier validation only; 99% per-clip angular-travel retention',
                   evaluation_recording_previously_used_for_development=True, deployed=False, splits={})
    for split in features:
        score = np.column_stack(scores[split])
        keep = np.minimum(1., score / np.maximum(np.array(thresholds), 1e-9))
        keep[~motion[split]['valid']] = 1.
        records = []
        for annotation in specification['annotations'][split]:
            mask = (data[split + '_group'] == annotation['id']) & motion[split]['valid']
            if not mask.any():
                continue
            known_axes = [2] if split == 'train' and annotation['label'] == 'combined' else None
            result = dict(annotation, **segment_metrics(motion[split]['delta'], keep, mask, annotation['label'], known_axes))
            if annotation['label'] == 'combined':
                result['evaluation_note'] = 'Held-out explicit yaw+translation' if known_axes else 'Component intent not specified; not counted as full multi-axis ground truth'
            records.append(result)
        summary['splits'][split] = records
        np.savez_compressed(args.output / (split + '_independent_predictions.npz'),
                            t=data[split + '_t'], score=score, keep=keep, quaternion=motion[split]['quaternion'],
                            delta=motion[split]['delta'], valid=motion[split]['valid'], enabled=motion[split]['enabled'])
    (args.output / 'independent_axis_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in summary['splits']['test']:
        print(row['start'], row['label'], row.get('desired_component_retained'), row.get('other_rotation_suppressed'), flush=True)
    print('held-out yaw+translation:', [(r['id'], r.get('desired_component_retained')) for r in summary['splits']['train'] if r['label'] == 'combined'])
    print('thresholds:', thresholds)
    # Reuse this exact signal binding for the temporal comparison; original
    # recordings and user annotations are never rewritten by this script.
    cache = {}
    for split in features:
        cache.update({split + '_features': features[split],
                      split + '_delta': motion[split]['delta'],
                      split + '_quaternion': motion[split]['quaternion'],
                      split + '_valid': motion[split]['valid'],
                      split + '_enabled': motion[split]['enabled']})
    np.savez_compressed(args.output / 'continuous_signal_binding.npz', **cache)


if __name__ == '__main__':
    main()
