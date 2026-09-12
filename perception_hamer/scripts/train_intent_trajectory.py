#!/usr/bin/env python3
"""Learn independent component corrections using angle trajectories, not jitter travel.

This is an offline experiment. Main-axis trajectories are noisy measurement
proxies constrained by operator action labels, not exact desired-pose truth.
The held-out combined clips are not used for fitting or checkpoint selection.
"""
from pathlib import Path
import argparse
import copy
import gzip
import json

import numpy as np
import torch

from evaluate_continuous_intent import AXES, ROOT, SOURCE, command_angle_binding, past_features
from train_continuous_intent import ContinuousIntent, WINDOW


OUTPUT = ROOT / '.runtime/continuous_intent_20260912'


def direct_pose_correction(angles, score):
    """Offline absolute-pose residual; no integration and no moving C reference.

    Its magnitude is bounded by the current C-relative input magnitude. Thus
    an exactly zero pose is zero for every temporal state. This structural
    property does not prove correct intention inference or stationary noise
    rejection; those still need comparison against the recorded trajectories.
    """
    residual = 2. * score - 1.
    residual = residual / torch.clamp(torch.linalg.vector_norm(residual, dim=-1, keepdim=True), min=1.)
    return angles - torch.linalg.vector_norm(angles, dim=-1, keepdim=True) * residual


def trajectory_metrics(angles, keep, ids, axis=None, output_angles=None):
    source = angles[ids] - angles[ids[0]]
    delta = np.vstack([np.zeros(3), np.diff(angles[ids], axis=0)])
    corrected = (np.cumsum(delta * keep[ids], axis=0) if output_angles is None
                 else output_angles[ids] - output_angles[ids[0]])
    out = dict(before_peak_deg=np.rad2deg(np.max(np.abs(source), axis=0)).tolist(),
               after_peak_deg=np.rad2deg(np.max(np.abs(corrected), axis=0)).tolist(),
               before_final_deg=np.rad2deg(source[-1]).tolist(),
               after_final_deg=np.rad2deg(corrected[-1]).tolist(),
               component_curve_error_rms_deg=np.rad2deg(np.sqrt(np.mean((corrected-source)**2, axis=0))).tolist())
    if axis is not None:
        peak = max(np.max(np.abs(source[:, axis])), 1e-9)
        out['main_peak_retained'] = float(np.max(np.abs(corrected[:, axis])) / peak)
        out['main_curve_error_rms_deg'] = out['component_curve_error_rms_deg'][axis]
        out['main_final_error_deg'] = float(np.rad2deg(corrected[-1, axis]-source[-1, axis]))
    other = [i for i in range(3) if i != axis]
    out['other_peak_before_deg'] = float(np.rad2deg(np.linalg.norm(source[:, other], axis=1).max()))
    out['other_peak_after_deg'] = float(np.rad2deg(np.linalg.norm(corrected[:, other], axis=1).max()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapt-current-feedback', action='store_true',
                        help='learn selected newly annotated failures, reserving separate repeated actions')
    parser.add_argument('--clean-motion-constraint', action='store_true',
                        help='require preservation of known rigid single/combined motions from separate geometry')
    parser.add_argument('--decision-lag-frames', type=int, default=0, choices=[0, 1],
                        help='one frame means real additional decision delay, not a free future observation')
    parser.add_argument('--current-pose', action='store_true',
                        help='include the requested C-relative pose and express rates in the actual command channels')
    parser.add_argument('--visual-motion', action='store_true',
                        help='retain signed global/relative image motion and cross-modal agreement')
    parser.add_argument('--absolute-residual', action='store_true',
                        help='predict a bounded correction to the fixed-C pose, without integrating retained increments')
    args = parser.parse_args()
    if args.absolute_residual and not args.current_pose:
        parser.error('--absolute-residual requires --current-pose')
    run_name = 'feedback_trajectory' if args.adapt_current_feedback else 'trajectory_intent'
    if args.clean_motion_constraint:
        run_name += '_clean'
    if args.decision_lag_frames:
        run_name += '_lag1'
    if args.current_pose:
        run_name += '_pose'
    if args.visual_motion:
        run_name += '_visual'
    if args.absolute_residual:
        run_name += '_absolute'
    torch.set_num_threads(2)
    torch.manual_seed(19)
    np.random.seed(19)
    data = np.load(SOURCE / 'intent_feature_dataset.npz')
    binding = np.load(OUTPUT / 'continuous_signal_binding.npz')
    spec = json.loads((SOURCE / 'intent_data_definition.json').read_text())
    sets = {}
    angle_data = {s:command_angle_binding(data,s,spec) for s in ['train','validation','test']}
    features = {s:binding[s+'_features'] for s in angle_data}
    if args.current_pose:
        adjusted={key:data[key].copy() for key in data.files}
        for split in angle_data:
            a=angle_data[split]
            speed=np.vstack([np.zeros(3),np.diff(a['angles'],axis=0)*15])
            speed[np.r_[False,np.diff(a['epoch'])!=0]]=0.
            adjusted[split+'_x'][:,3:6]=speed
            features[split]=np.column_stack([past_features(adjusted,split,SOURCE,args.visual_motion),a['position_delta'],a['angles']])
    elif args.visual_motion:
        features={s:past_features(data,s,SOURCE,True) for s in angle_data}
    known = (data['train_y'] >= 0) & (data['train_y'] < 5) & binding['train_valid']
    feedback_training_groups = [2, 4, 6, 11, 12, 14]
    feedback_validation_groups = [3, 5, 7, 8, 9, 13]
    if args.current_pose:
        known &= binding['train_enabled']
    fitting_features = [features['train'][known]]
    if args.adapt_current_feedback:
        current = np.isin(data['test_group'], feedback_training_groups) & binding['test_valid']
        fitting_features.append(features['test'][current])
    mean = np.concatenate(fitting_features).mean(0)
    std = np.maximum(.03, np.concatenate(fitting_features).std(0))
    for split in ['train', 'validation', 'test']:
        x = np.clip((features[split]-mean)/std, -8., 8.)
        windows = np.stack([x[np.clip(np.arange(i-WINDOW+1,i+1)+args.decision_lag_frames,0,len(x)-1)] for i in range(len(x))])
        tensor = torch.tensor(windows.transpose(0, 2, 1), dtype=torch.float32)
        episodes = []
        for a in spec['annotations'][split]:
            if a['label'] == 'combined':
                continue
            ids = np.flatnonzero((data[split + '_group'] == a['id']) & binding[split + '_valid'])
            if args.current_pose:
                ids=ids[binding[split+'_enabled'][ids]]
            if args.decision_lag_frames:
                ids=ids[ids+args.decision_lag_frames<len(x)]
                ids=ids[binding[split+'_valid'][ids+args.decision_lag_frames] &
                        (angle_data[split]['epoch'][ids]==angle_data[split]['epoch'][ids+args.decision_lag_frames])]
            if len(ids) < 3:
                continue
            curve = angle_data[split]['angles'][ids] - angle_data[split]['angles'][ids[0]]
            target = np.zeros_like(curve)
            axis_weights = np.ones(3)
            if a['label'] in AXES:
                axis = AXES.index(a['label'])
                target[:, axis] = curve[:, axis]
                axis_weights[axis] = 9.
            scale = np.maximum(np.max(np.abs(curve), axis=0), np.deg2rad(1.))
            delta = np.vstack([np.zeros(3), np.diff(curve, axis=0)])
            episodes.append(dict(annotation=a, split=split, ids=ids, x=tensor[ids],
                                 angles=torch.tensor(angle_data[split]['angles'][ids], dtype=torch.float32),
                                 absolute_target=torch.tensor(target + angle_data[split]['angles'][ids[0]], dtype=torch.float32),
                                 delta=torch.tensor(delta, dtype=torch.float32),
                                 target=torch.tensor(target, dtype=torch.float32),
                                 scale=torch.tensor(scale, dtype=torch.float32),
                                 axis_weights=torch.tensor(axis_weights, dtype=torch.float32)))
        counts = {label: sum(e['annotation']['label'] == label for e in episodes)
                  for label in ['still', 'translate', 'pitch', 'roll', 'yaw']}
        for e in episodes:
            e['weight'] = 1. / counts[e['annotation']['label']]
        sets[split] = dict(x=tensor, episodes=episodes)

    training_episodes = list(sets['train']['episodes'])
    validation_episodes = list(sets['validation']['episodes'])
    if args.adapt_current_feedback:
        training_episodes += [e for e in sets['test']['episodes'] if e['annotation']['id'] in feedback_training_groups]
        validation_episodes += [e for e in sets['test']['episodes'] if e['annotation']['id'] in feedback_validation_groups]
    for episodes in [training_episodes, validation_episodes]:
        counts = {label: sum(e['annotation']['label'] == label for e in episodes)
                  for label in ['still', 'translate', 'pitch', 'roll', 'yaw']}
        for e in episodes:
            e['weight'] = 1. / counts[e['annotation']['label']]

    clean_specification = []
    if args.clean_motion_constraint:
        from check_intent_combinations import simulate_features
        old = ROOT / '.runtime/hand_motion_observation_20260911T153019_6Wdm'
        with gzip.open(old / 'analysis_motion_intent/perception_signals.json.gz', 'rt') as stream:
            reference = next(r for r in json.load(stream) if r.get('control_enabled') and r.get('published_q'))
        with (old / 'perception/DEV_HAMER_6DOF_OBSERVATION_20260911T153039/frames.jsonl').open() as stream:
            row = next(r for r in map(json.loads, stream) if r.get('capture_sequence') == reference['seq'])
        geometry = np.array(row['mano_joints'])
        geometry -= geometry[0]
        intrinsics = json.loads((SOURCE / 'c_rgbd/manifest.json').read_text())['intrinsics']
        random = np.random.default_rng(47021)
        for role, count, episodes in [('training', 32, training_episodes), ('validation', 8, validation_episodes)]:
            for number in range(count):
                active = random.choice(3, size=1+number%3, replace=False)
                amplitude = np.zeros(3)
                amplitude[active] = np.exp(random.uniform(np.log(.3),np.log(30.),len(active)))*random.choice([-1.,1.],len(active))
                movement = random.uniform(-.06,.06,3) if number%2 else np.zeros(3)
                rise, hold, fall = random.uniform(.6,1.5), random.uniform(.5,1.2), random.uniform(.6,1.5)
                times, curve, features = simulate_features(geometry, np.array(reference['position']), intrinsics,
                                                           amplitude, movement, rise, hold, fall,current_pose=args.current_pose,
                                                           visual_motion=args.visual_motion)
                x = np.clip((features-mean)/std,-8.,8.)
                windows = np.stack([x[np.clip(np.arange(i-WINDOW+1,i+1)+args.decision_lag_frames,0,len(x)-1)] for i in range(len(x))])
                scale = np.maximum(np.max(np.abs(curve),axis=0),np.deg2rad(.1))
                episodes.append(dict(annotation=dict(label='clean',id=number),split='clean_'+role,
                                     x=torch.tensor(windows.transpose(0,2,1),dtype=torch.float32),
                                     angles=torch.tensor(curve,dtype=torch.float32),
                                     absolute_target=torch.tensor(curve,dtype=torch.float32),
                                     delta=torch.tensor(np.vstack([np.zeros(3),np.diff(curve,axis=0)]),dtype=torch.float32),
                                     target=torch.tensor(curve,dtype=torch.float32),
                                     scale=torch.tensor(scale,dtype=torch.float32),
                                     axis_weights=torch.tensor((np.abs(amplitude)>0).astype(float),dtype=torch.float32),
                                     weight=1./count, clean_angles=curve, active_axes=active))
                clean_specification.append(dict(role=role,amplitude_deg=amplitude.tolist(),translation_m=movement.tolist(),
                                                rise_s=float(rise),hold_s=float(hold),fall_s=float(fall)))

    def episode_loss(model, e):
        prediction = (direct_pose_correction(e['angles'], model(e['x'])) if args.absolute_residual
                      else torch.cumsum(model(e['x']) * e['delta'], dim=0))
        target = e['absolute_target'] if args.absolute_residual else e['target']
        error = ((prediction-target) / e['scale'])**2
        return (error.mean(0)*e['axis_weights']).sum()/e['axis_weights'].sum()

    model = ContinuousIntent(len(mean))
    if args.absolute_residual:
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.002)
    best, selected, state = float('inf'), None, None
    history = []
    for epoch in range(80):
        model.train()
        order = np.random.permutation(len(training_episodes))
        for chunk in np.array_split(order, max(1, len(order)//8)):
            optimizer.zero_grad()
            value = sum(episode_loss(model, training_episodes[i])*training_episodes[i]['weight'] for i in chunk)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = sum(float(episode_loss(model, e))*e['weight'] for e in validation_episodes) / (6. if args.clean_motion_constraint else 5.)
        history.append(validation_loss)
        if validation_loss < best:
            best, selected, state = validation_loss, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        score = {s: model(v['x']).numpy() for s, v in sets.items()}
        for e in validation_episodes:
            if e['annotation']['label']=='clean':
                e['score']=model(e['x']).numpy()
    thresholds = []
    # Calibrate to macro angle trajectories. Unlike cumulative absolute
    # increments, this criterion does not demand preservation of every wiggle.
    for axis, name in enumerate(AXES):
        if args.absolute_residual:
            thresholds.append(1.)  # unused; direct correction has no intent gate
            continue
        episodes = [e for e in validation_episodes if e['annotation']['label'] == name]
        threshold = 1e-6
        for proposal in np.linspace(.001, 1., 250):
            good = True
            for e in episodes:
                split = e['split']
                keep = np.ones_like(score[split])
                keep[:, axis] = np.minimum(1., score[split][:, axis] / proposal)
                result = trajectory_metrics(angle_data[split]['angles'], keep, e['ids'], axis)
                peak = result['before_peak_deg'][axis]
                good &= (result['main_peak_retained'] >= .99 and
                         result['main_curve_error_rms_deg'] <= max(.1, .02*peak))
            for e in validation_episodes:
                if e['annotation']['label']!='clean' or axis not in e['active_axes']:
                    continue
                keep=np.ones_like(e['score']);keep[:,axis]=np.minimum(1.,e['score'][:,axis]/proposal)
                result=trajectory_metrics(e['clean_angles'],keep,np.arange(len(keep)),axis)
                peak=result['before_peak_deg'][axis]
                good &= (result['main_peak_retained']>=.99 and result['main_curve_error_rms_deg']<=max(.01,.02*peak))
            if good:
                threshold = float(proposal)
        thresholds.append(threshold)
    report = dict(scope=__doc__, deployed=False, selected_epoch=selected, validation_loss=best,
                  validation_history=history, thresholds=thresholds, axes=AXES,
                  angle_basis='Exact ordered RelativePoseMapper input channels; pitch, roll, yaw before gains',
                  selection='Equal intent-type contribution; preserve main angle trajectory, suppress other channels',
                  current_feedback_adaptation=args.adapt_current_feedback,
                  current_training_groups=feedback_training_groups if args.adapt_current_feedback else [],
                  current_validation_groups=feedback_validation_groups if args.adapt_current_feedback else [],
                  current_C_and_return_excluded_from_training_and_selection=True,
                  clean_motion_constraint=args.clean_motion_constraint,clean_motion_examples=clean_specification,
                  clean_motion_geometry_recording='hand_motion_observation_20260911T153019_6Wdm' if args.clean_motion_constraint else None,
                  decision_lag_frames=args.decision_lag_frames,
                  current_pose=args.current_pose,
                  visual_motion=args.visual_motion,
                  absolute_residual=args.absolute_residual,
                  nominal_additional_decision_delay_s=args.decision_lag_frames/15.,
                  curve_error_scope='Source-associated angle-path error; excludes the explicitly reported decision delay',
                  splits={})
    for split in sets:
        keep = np.minimum(1., score[split] / np.maximum(thresholds, 1e-9))
        keep[~binding[split + '_valid']] = 1.
        output_angles = None
        if args.absolute_residual:
            with torch.no_grad():
                output_angles = direct_pose_correction(torch.tensor(angle_data[split]['angles']),
                                                       torch.tensor(score[split])).numpy()
        records = []
        for a in spec['annotations'][split]:
            ids = np.flatnonzero((data[split + '_group'] == a['id']) & binding[split + '_valid'])
            if len(ids) < 3:
                continue
            axis = AXES.index(a['label']) if a['label'] in AXES else 2 if split == 'train' and a['label'] == 'combined' else None
            role = 'training' if split == 'train' and a['label'] != 'combined' else 'validation' if split == 'validation' else 'held_out'
            if args.adapt_current_feedback and split == 'test':
                role = 'training' if a['id'] in feedback_training_groups else 'validation' if a['id'] in feedback_validation_groups else 'held_out'
            records.append(dict(a, role=role, **trajectory_metrics(angle_data[split]['angles'], keep, ids, axis, output_angles)))
            if output_angles is not None:
                records[-1]['output_start_C_angles_deg'] = np.rad2deg(output_angles[ids[0]]).tolist()
                records[-1]['output_end_C_angles_deg'] = np.rad2deg(output_angles[ids[-1]]).tolist()
        report['splits'][split] = records
        np.savez_compressed(OUTPUT / (split + '_' + run_name + '_predictions.npz'),
                            t=data[split + '_t'], score=score[split], keep=keep,
                            decision_frame_t=data[split+'_t']+args.decision_lag_frames/15.,
                            valid=binding[split + '_valid'], enabled=binding[split + '_enabled'],
                            **({'output_angles':output_angles} if output_angles is not None else {}),
                            **angle_data[split])
    torch.save(dict(state_dict=state, mean=mean, std=std, thresholds=thresholds, window=WINDOW,
                    axes=AXES, decision_lag_frames=args.decision_lag_frames, current_pose=args.current_pose,
                    visual_motion=args.visual_motion, absolute_residual=args.absolute_residual, deployed=False), OUTPUT / (run_name + '.pt'))
    (OUTPUT / (run_name + '_results.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Selected', selected, 'thresholds', thresholds)
    for r in report['splits']['test']:
        print(r['start'], r['label'], 'main', r.get('main_peak_retained'), 'other', r['other_peak_before_deg'], '->', r['other_peak_after_deg'], flush=True)
    print('Held out', [(r['id'],r['main_peak_retained']) for r in report['splits']['train'] if r['label']=='combined'])


if __name__ == '__main__':
    main()
