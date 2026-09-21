#!/usr/bin/env python3
"""Exact-source replay with one literal fixed image crop versus recorded crops.

Only the crop policy changes. No robot output, pose filtering, model training,
depth correction, time-offset fitting, or per-frame hand reclassification.
The fixed crop retains the whole 640x480 RGB in the actual 192x256 backbone
input (with black vertical padding). Its lower hand resolution is reported.
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
from perception_hamer.src.hamer_crop_inference import HamerCropInference, HamerInferenceError, prepare_hamer_crop
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame
from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition, estimate_mano_wrist_frame
from perception_hamer.src.realtime_hamer_pipeline import normalized_crop_points_to_original
from perception_mediapipe.recorded_rgbd import RecordedRGBD

WINDOWS = [(54, 110, 'horizontal'), (176, 209, 'vertical'), (265, 273, 'yaw_control')]
FIXED_BOX = [160., 26.666666666666657, 480., 453.33333333333337]
METHODS = ['dynamic', 'fixed']


def read_records(source):
    origin = json.load(open(source / 'analysis_intent/time_origin.json'))['video_wall_origin']
    logs = list(source.glob('perception/*/frames.jsonl'))
    process = None
    if logs:
        stream = logs[0].open()
    else:
        log = next(source.glob('perception/*/frames.jsonl.zst'))
        process = subprocess.Popen(['zstd', '-q', '-dc', str(log)], stdout=subprocess.PIPE, text=True)
        stream = process.stdout
    records = []
    try:
        for line in stream:
            d = json.loads(line)
            t = d['timestamp'] - origin
            if not (42.6 <= t <= 43 or any(lo <= t <= hi for lo, hi, _ in WINDOWS)):
                continue
            if d.get('betas_user') is None:
                raise ValueError('Selected source frame has no recorded frozen betas')
            records.append(dict(t=t, stamp=d['timestamp'], frame=d['frame_number'],
                                seq=d['capture_sequence'], bbox=d['bbox'], betas=d['betas_user']))
    finally:
        stream.close()
        if process and process.wait() != 0:
            raise RuntimeError('Compressed original inference log could not be read')
    return records


def source_rgb(reader, row):
    # Reuse the indexed recording's original color metadata. Depth is irrelevant
    # to this pose-only intervention; no compressed screen-video input is used.
    meta = reader.colors[row['frame']]
    if abs(meta['time'] - row['stamp']) > .001:
        raise ValueError('EXACT_SOURCE_RGB_TIMESTAMP_MISMATCH')
    tid = reader.topics[reader.prefix['color'] + '/image/data']
    blob = reader.db.execute('select data from messages where timestamp=? and topic_id=?',
                             (meta['stamp'], tid)).fetchone()[0]
    im = reader.types.deserialize_cdr(blob, 'sensor_msgs/msg/Image')
    return im.data.reshape(im.height, im.width, 3).copy()


def angle(rotation):
    return np.rad2deg(rotation.magnitude())


def stats(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(x.max())) if len(x) else dict(n=0)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(rows, source, out):
    all_rows = rows
    rows = [x for x in rows if all(x[n].get('valid', False) for n in METHODS)]
    imu = np.load(source / 'analysis_intent/imu_signals.npz')
    offset = json.load(open(source / 'analysis_intent/time_alignment.json'))['used_offset_s']
    ts = np.array([x['t'] for x in rows])
    ix = np.clip(np.searchsorted(imu['t'], ts + offset), 1, len(imu['t']) - 1)
    ok = ((ts + offset >= imu['t'][0]) & (ts + offset <= imu['t'][-1]) &
          imu['time_valid'][ix] & imu['time_valid'][ix - 1])
    rots = {n: R.from_quat([x[n]['palm_q'] for x in rows]) for n in METHODS}
    si = Slerp(imu['t'], R.from_quat(imu['q']))
    anns = json.load(open(source / 'analysis_intent/annotations.json'))
    segments, samples, blocks = [], [], []
    for ann in anns:
        if not any(lo <= ann['start'] and ann['end'] <= hi for lo, hi, _ in WINDOWS):
            continue
        mask = np.flatnonzero(ok & (ts >= ann['start'] + .2) & (ts <= ann['end'] - .2))
        if len(mask) < 3:
            continue
        ref = si(ts[mask] + offset)
        im = angle(ref[0].inv() * ref)
        result = dict(id=ann['id'], time=ann['time_label'], kind=ann['kind'], action=ann['action'],
                      n=len(mask), zero_t=float(ts[mask[0]]), imu_peak_deg=float(im.max()))
        mags = {}
        for n, rot in rots.items():
            mag = angle(rot[mask] * rot[mask[0]].inv())
            mags[n] = mag
            result[n + '_peak_deg'] = float(mag.max())
            result[n + '_magnitude_difference'] = stats(abs(mag - im))
        for k, j in enumerate(mask):
            sample = dict(segment=ann['id'], kind=ann['kind'], t=ts[j], seq=rows[j]['seq'], imu_deg=im[k])
            for n in METHODS:
                sample[n + '_deg'] = mags[n][k]
                sample[n + '_abs_difference_deg'] = abs(mags[n][k] - im[k])
            samples.append(sample)
        segments.append(result)
    # One zero for each complete movement block: no reset at image edges/returns.
    curves = []
    for lo, hi, name in WINDOWS:
        mask = np.flatnonzero(ok & (ts >= lo) & (ts <= hi))
        ref = si(ts[mask] + offset)
        im = angle(ref[0].inv() * ref)
        mags = {n: angle(q[mask] * q[mask[0]].inv()) for n, q in rots.items()}
        block = dict(name=name, range=[lo, hi], zero_t=float(ts[mask[0]]), n=len(mask),
                     imu_peak_deg=float(im.max()))
        for n in METHODS:
            block[n + '_peak_deg'] = float(mags[n].max())
            block[n + '_magnitude_difference'] = stats(abs(mags[n] - im))
        blocks.append(block)
        for k, j in enumerate(mask):
            b = rows[j]['bbox']
            sample = dict(block=name, t=ts[j], seq=rows[j]['seq'],
                source_box_center_x=(b[0] + b[2]) / 2, source_box_center_y=(b[1] + b[3]) / 2,
                imu_deg=im[k])
            for n in METHODS:
                sample[n + '_deg'] = mags[n][k]
                sample[n + '_abs_difference_deg'] = abs(mags[n][k] - im[k])
            curves.append(sample)
    categories = {}
    for kind in sorted(set(x['kind'] for x in samples)):
        categories[kind] = {n: stats([x[n + '_abs_difference_deg'] for x in samples if x['kind'] == kind]) for n in METHODS}
    summary = dict(source_frames=len(all_rows), both_geometry_valid=len(rows), imu_time_valid=int(ok.sum()), segment_samples=len(samples),
        valid_outputs={n: sum(x[n].get('valid', False) for x in all_rows) for n in METHODS},
        failures=[dict(t=x['t'], method=n, reason=x[n]['failure']) for x in all_rows for n in METHODS if not x[n].get('valid', False)],
        imu_common_offset_s=offset, categories=categories, blocks=blocks,
        replay_original_difference_deg=stats([x['dynamic']['recorded_rotation_difference_deg'] for x in rows]),
        input_hand_height_px={n: stats([x[n]['input_hand_height_px'] for x in all_rows]) for n in METHODS},
        input_bbox_fully_inside={n: sum(x[n]['source_box_inside_actual_input'] for x in all_rows) for n in METHODS},
        scope='Same RGB, model, frozen shape, fixed left, FP16, raw poses. IMU is an imperfect reference; magnitude difference is not full 3-D orientation error. Fixed crop changes image occupancy and hand resolution.')
    (out / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / 'segment_metrics.json').write_text(json.dumps(segments, ensure_ascii=False, indent=2))
    write_csv(out / 'paired_samples.csv', samples)
    write_csv(out / 'block_curves.csv', curves)
    flat = [{k: v for k, v in x.items() if not isinstance(v, dict)} | {
        n + '_mean_abs_difference_deg': x[n + '_magnitude_difference']['mean'] for n in METHODS
    } for x in segments]
    write_csv(out / 'segment_comparison.csv', flat)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'source_png').mkdir(exist_ok=True)
    (args.out / 'inputs').mkdir(exist_ok=True)
    rows = ([json.loads(x) for x in (args.out / 'predictions.jsonl').open()]
            if args.size_control else read_records(args.source))
    if args.pilot:
        times = [54, 58, 62, 68, 72, 75, 83, 103, 107, 110, 176, 180, 187, 209, 265, 268, 273]
        ts = np.array([x['t'] for x in rows])
        rows = [rows[k] for k in sorted(set(int(np.argmin(abs(ts - t))) for t in times))]
    old = {x['seq']: x for x in json.load(gzip.open(ROOT / '.runtime/palm_ring_surface_20260920/paired_frames.json.gz', 'rt'))}
    model = HamerCropInference(str(ROOT / 'perception_hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt'),
        data_root=str(ROOT / 'perception_hamer/_DATA'), precision='fp16', freeze_betas=True)
    model.load()
    model.warmup()
    nv, nj = model.neutral_mano_geometry(False)
    definition = build_mano_wrist_definition(nv, nj, model.mano_faces(), False)
    reader = RecordedRGBD(args.rgbd)
    last_saved = -1e10
    started = time.monotonic()
    outrows = []
    prediction_name = 'size_control_predictions.jsonl' if args.size_control else 'predictions.jsonl'
    with (args.out / prediction_name).open('w') as f:
        for j, row in enumerate(rows):
            rgb = source_rgb(reader, row)
            if rgb.shape != (480, 640, 3):
                raise ValueError('Fixed crop is specified for this 640x480 recording only')
            model.set_frozen_betas(False, np.asarray(row['betas']))
            row = dict(row)
            save_image = not args.size_control and (args.pilot or row['t'] - last_saved >= .115)
            if save_image:
                cv2.imwrite(str(args.out / 'source_png' / f'{row["seq"]}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                last_saved = row['t']
            b = np.array(row['bbox'])
            center = (b[:2] + b[2:]) / 2
            half_size = (np.array(FIXED_BOX[2:]) - FIXED_BOX[:2]) / 2
            large_center_box = np.r_[center - half_size, center + half_size].tolist()
            cases = [('large_centered', large_center_box)] if args.size_control else [('dynamic', row['bbox']), ('fixed', FIXED_BOX)]
            for name, box in cases:
                crop, affine, _ = prepare_hamer_crop(rgb, box, False)
                b = row['bbox']
                corners = np.array([[b[0], b[1], 1], [b[2], b[3], 1]]) @ affine.T
                lower, upper = corners.min(axis=0), corners.max(axis=0)
                row[name] = dict(valid=False, box=list(box), affine=affine.tolist(),
                    input_hand_height_px=float(upper[1] - lower[1]),
                    source_box_inside_actual_input=bool(lower[0] >= 32 and upper[0] <= 224 and lower[1] >= 0 and upper[1] <= 256))
                try:
                    pred = model.infer(rgb, box, False, row['stamp'])
                    palm = build_hamer_five_point_palm_frame(pred.pred_keypoints_3d_source_camera_axes, False)
                    ring = estimate_mano_wrist_frame(pred.pred_vertices_source_camera_axes, definition)
                    if not palm.valid or not ring.valid:
                        raise ValueError('Invalid output pose: ' + palm.failure_reason + ' ' + ring.failure_reason)
                except (HamerInferenceError, ValueError) as exc:
                    row[name]['failure'] = str(exc)
                    continue
                row[name].update(valid=True, palm_q=R.from_matrix(palm.rotation).as_quat().tolist(),
                    ring_q=R.from_matrix(ring.rotation).as_quat().tolist(),
                    pixels=normalized_crop_points_to_original(pred.pred_keypoints_2d_crop_normalized, affine, 256).tolist(),
                    model_ms=pred.inference_time_s * 1000)
                if name == 'dynamic':
                    row[name]['recorded_rotation_difference_deg'] = float(angle(R.from_quat(old[row['seq']]['palm_q']).inv() * R.from_matrix(palm.rotation)))
                if args.pilot:
                    image = np.clip((crop.transpose(1, 2, 0) * np.array([.229, .224, .225]) + np.array([.485, .456, .406])) * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(args.out / 'inputs' / f'{row["seq"]}_{name}.png'), cv2.cvtColor(image[:, 32:-32], cv2.COLOR_RGB2BGR))
            f.write(json.dumps(row) + '\n')
            f.flush()
            outrows.append(row)
            if j % 100 == 0 or args.pilot:
                print(f'{j+1}/{len(rows)} source={row["t"]:.3f}s elapsed={time.monotonic()-started:.1f}s', flush=True)
    reader.close()
    metadata = dict(source=str(args.source.resolve()), source_rgbd=str(args.rgbd.resolve()),
        fixed_bbox_argument=FIXED_BOX, fixed_actual_backbone_source_rectangle=[0., -186.66666666666669, 640., 666.6666666666667],
        input_size=[192, 256], windows=WINDOWS, n=len(rows), pilot=args.pilot,
        intervention='Actual input affine is constant; complete source RGB is retained. Same original checkpoint, FP16, left hand, each recorded frozen shape. No filters or robot output.',
        caveat='Fixed crop reduces hand pixel height and moves hand away from center; this experiment does not isolate all physical perspective effects.')
    metadata['size_control'] = args.size_control
    metadata_name = 'size_control_metadata.json' if args.size_control else 'metadata.json'
    (args.out / metadata_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not args.pilot:
        analyze(outrows, args.source, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--rgbd', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--size-control', action='store_true', help='Append a same-size, hand-centered control to an existing completed run')
    args = parser.parse_args()
    if args.size_control:
        METHODS.append('large_centered')
    run(args)
