#!/usr/bin/env python3
"""Source-grounded figures for the controlled synthetic viewpoint intervention.

Claim: this frozen synthetic hand produces false inferred rotation under 3-D
translation; matched 2-D shifts do not. It does not identify the dominant cause
in human recordings. One frozen mesh, fixed positions, no population inference.
Python figures: vector PDF/SVG and 300 dpi preview, 7.2 inches wide.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figure-tools', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.figure_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    rows = json.load(open(args.run / 'results.json'))
    by_key = {(x['kind'], x['value_deg']): x for x in rows}
    center = by_key['center', 0]
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [font, 'DejaVu Sans'],
        'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

    def save(fig, name):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(args.out / (name + '.alignment.json')), strict=True)
        fig.savefig(args.out / (name + '.pdf'))
        fig.savefig(args.out / (name + '.svg'))
        fig.savefig(args.out / (name + '.png'), dpi=300)
        plt.close(fig)
        f = args.out / (name + '.svg')
        f.write_text('\n'.join(x.rstrip() for x in f.read_text().splitlines()) + '\n')

    angles = [-20., -10., 0., 10., 20.]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    fig.subplots_adjust(left=.12, right=.97, bottom=.2, top=.76)
    for kind, label, color, marker in [
        ('perspective_translation', '三维平移：视角及投影改变', '#BC5C38', 'o'),
        ('image_translation', '二维移图：手的投影不变', '#277DA8', 's')]:
        vals = [by_key[kind, a]['predicted_angle_deg'] if a else 0 for a in angles]
        ax.plot(angles, vals, color=color, marker=marker, markersize=5, lw=1.5, label=label)
    ax.set(xlabel='三维组观察方向／二维组对应图片位置（度）', ylabel='HaMeR 输出的多余转角（度）',
           xticks=angles, xlim=(-22, 22), ylim=(-1, 26))
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=1)
    fig.text(.12, .97, '受控合成实验：手的真实转角始终为 0°', va='top', fontsize=10)
    save(fig, 'viewpoint_intervention')

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.2))
    fig.subplots_adjust(left=.09, right=.97, top=.89, bottom=.08, wspace=.3, hspace=.4)
    for r, (kind, title, color) in enumerate([
        ('image_translation', '二维移图', '#277DA8'),
        ('perspective_translation', '三维平移', '#BC5C38')]):
        for c, a in enumerate([-20., 0., 20.]):
            row = by_key[kind, a] if a else center
            ax = axes[r, c]
            ax.imshow(Image.open(args.run / row['input_image']), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{title} {a:+.0f}°', color=color, fontsize=9, pad=10)
            ax.set_xlabel(f'输出转角 {row["predicted_angle_deg"]:.3f}°', color=color, fontsize=9)
            ax.annotate(chr(97 + r * 3 + c), xy=(0, 1), xycoords='axes fraction',
                        xytext=(-12, 10), textcoords='offset points', weight='bold', fontsize=10)
    fig.text(.09, .985, '网络实际输入：所有平移组手部高度均为 128 像素', va='top', fontsize=10)
    fig.text(.09, .035, '相同手形与朝向、相同背景；中央图为两组共用基准。模型输入均按左手镜像。', fontsize=8)
    save(fig, 'actual_inputs')

    # User's practical criterion: preserve actual incidental rotation and inspect
    # the excess over synchronized IMU, rather than force translation to 0 deg.
    source = ROOT / '.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent'
    imu = np.load(source / 'imu_signals.npz')
    offset = json.load(open(source / 'time_alignment.json'))['used_offset_s']
    si = Slerp(imu['t'], R.from_quat(imu['q']))
    with gzip.open(ROOT / '.runtime/palm_ring_surface_20260920/paired_frames.json.gz', 'rt') as f:
        real = [x for x in json.load(f) if 54 <= x['t'] <= 110 and x['palm_valid']]
    source_n = len(real)
    ts = np.array([x['t'] for x in real])
    ix = np.clip(np.searchsorted(imu['t'], ts + offset), 1, len(imu['t']) - 1)
    ok = imu['time_valid'][ix] & imu['time_valid'][ix - 1]
    real = [x for x, keep in zip(real, ok) if keep]
    ts = ts[ok]
    rotation = R.from_quat([x['palm_q'] for x in real])
    visual = np.rad2deg((rotation * rotation[0].inv()).magnitude())
    ref = np.rad2deg((si(ts[0] + offset).inv() * si(ts + offset)).magnitude())
    u = np.array([(x['bbox'][0] + x['bbox'][2]) / 2 for x in real])
    ray = np.rad2deg(np.arctan2(u - 319.0163879394531, 606.5111083984375))
    excess = visual - ref
    real_rows = [dict(t=float(ts[j]), seq=x['seq'], source_hand_center_u=float(u[j]),
        viewing_ray_deg=float(ray[j]), visual_total_angle_deg=float(visual[j]),
        imu_total_angle_deg=float(ref[j]), signed_excess_deg=float(excess[j]),
        magnitude_absolute_difference_deg=float(abs(excess[j]))) for j, x in enumerate(real)]
    bins = []
    for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 90)]:
        mask = (abs(ray) >= lo) & (abs(ray) < hi)
        bins.append(dict(abs_viewing_ray_range_deg=[lo, hi], n=int(mask.sum()),
            mean_absolute_magnitude_difference_deg=float(np.mean(abs(excess[mask]))),
            mean_signed_excess_deg=float(np.mean(excess[mask])),
            mean_visual_angle_deg=float(visual[mask].mean()), mean_imu_angle_deg=float(ref[mask].mean()),
            p95_absolute_magnitude_difference_deg=float(np.percentile(abs(excess[mask]), 95))))
    selected = [real_rows[int(np.argmin(abs(ts - t)))] for t in [54, 58, 62, 68, 72, 75, 83]]
    real_metrics = dict(source_n=source_n, paired_imu_n=len(real), imu_offset_s=offset,
        zero_time_s=float(ts[0]), observed_ray_range_deg=[float(ray.min()), float(ray.max())],
        abs_position_abs_difference_correlation=float(np.corrcoef(abs(ray), abs(excess))[0, 1]),
        bins=bins, examples=selected,
        scope='Observed original HaMeR only, all valid-time pairs, common zero; raw poses; IMU is an imperfect reference; bins are descriptive, not causal thresholds; no requirement that real hand rotation be zero.')
    (args.out / '真人回放位置相关余差.json').write_text(json.dumps(real_metrics, ensure_ascii=False, indent=2))
    with (args.out / '真人回放逐帧余差.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(real_rows[0])); w.writeheader(); w.writerows(real_rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    fig.subplots_adjust(left=.12, right=.97, bottom=.2, top=.77)
    gap = np.r_[False, np.diff(ts) > .18]
    for y, label, color in [(visual, 'HaMeR 总转角', '#BC5C38'), (ref, '同步 IMU 总转角', '#4B5563')]:
        display = y.copy(); display[gap] = np.nan
        ax.plot(ts, display, label=label, color=color, lw=1.2)
    ax.set(xlabel='原录像时间（秒）', ylabel='相对同一段起点的总转角（度）', xlim=(54, 110), ylim=(0, 33))
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2)
    fig.text(.12, .96, '真人回放：保留附带真实转动，检查额外误转', fontsize=10, va='top')
    save(fig, 'real_translation_reference')

    for name in ['results.json', 'metadata.json', 'frozen_scene.npz']:
        shutil.copyfile(args.run / name, args.out / name)
    for directory in ['images', 'inputs']:
        shutil.copytree(args.run / directory, args.out / directory, dirs_exist_ok=True)
    table = []
    for x in rows:
        table.append(dict(condition=x['kind'], requested_value_deg=x['value_deg'],
            actual_view_direction_deg=x['actual_view_direction_deg'],
            true_rotation_deg=20. if x['kind'].startswith('rotation') else 0.,
            predicted_total_rotation_deg=x['predicted_angle_deg'], full_relative_error_deg=x['full_relative_error_deg'],
            rotation_x_deg=x['predicted_relative_rotvec_deg'][0],
            rotation_y_deg=x['predicted_relative_rotvec_deg'][1],
            rotation_z_deg=x['predicted_relative_rotvec_deg'][2],
            input_hand_height_px=x['input_hand_height_px'],
            input_max_absolute_difference=x['input_max_absolute_difference']))
    with (args.out / '全部条件结果.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print('Controlled inputs, prescribed-rotation checks, and all intervention data exported.')


if __name__ == '__main__':
    main()
