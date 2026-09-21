#!/usr/bin/env python3
"""Observed fixed-crop replay: raw curves and a source-timed input comparison.

Question: does locking the input crop remove translation-associated rotation?
Raw paired time series, one recording/operator, no smoothing or inferential test.
Python/matplotlib figures: editable PDF/SVG and PNG preview, 7.2 inches wide.
Video displays original RGB and actual model inputs; images are not enhanced.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception_hamer.src.hamer_crop_inference import prepare_hamer_crop

LABELS = {'imu': '同步 IMU 参照', 'dynamic': '原动态框', 'fixed': '位置、尺寸都固定', 'large_centered': '同尺寸大框随手居中'}
COLORS = {'imu': '#4B5563', 'dynamic': '#277DA8', 'fixed': '#BC5C38', 'large_centered': '#558C72'}
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figure-tools', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.figure_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    font_name = FontProperties(fname=FONT).get_name()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [font_name, 'DejaVu Sans'], 'font.size': 9, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'axes.unicode_minus': False})
    curves = list(csv.DictReader((args.run / 'block_curves.csv').open()))
    for block, title in [('horizontal', '左右平移'), ('vertical', '上下平移'), ('yaw_control', '真实偏航对照')]:
        selected = [x for x in curves if x['block'] == block]
        ts = np.array([float(x['t']) for x in selected])
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        fig.subplots_adjust(left=.11, right=.98, bottom=.19, top=.75)
        for name in LABELS:
            vals = np.array([float(x[name + '_deg']) for x in selected])
            # A missing output/time pair is a visible gap, never a filled pose.
            vals[np.r_[False, np.diff(ts) > .18]] = np.nan
            ax.plot(ts, vals, label=LABELS[name], color=COLORS[name], lw=1.15,
                    ls='--' if name in ['fixed', 'large_centered'] else '-')
        ax.set(xlabel='原视频时间（秒）', ylabel='相对本组起点的转角（度）',
               xlim=(ts[0], ts[-1]), ylim=(0, 185 if block != 'yaw_control' else 100))
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2, fontsize=9)
        fig.text(.11, .96, title + '：各组使用同一时刻设零，全段不重置', fontsize=10, va='top')
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(args.out / (block + '.alignment.json')), strict=True)
        fig.savefig(args.out / (block + '.png'), dpi=300)
        fig.savefig(args.out / (block + '.pdf'))
        fig.savefig(args.out / (block + '.svg'))
        plt.close(fig)
        svg = args.out / (block + '.svg')
        svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines()) + '\n')

    rows = [json.loads(x) for x in (args.run / 'size_control_predictions.jsonl').open()]
    by_seq = {x['seq']: x for x in rows}
    saved = [by_seq[int(f.stem)] for f in (args.run / 'source_png').glob('*.png') if int(f.stem) in by_seq]
    saved.sort(key=lambda x: x['t'])
    times = np.array([x['t'] for x in saved])
    metrics = json.load(open(args.run / 'metrics.json'))
    refs = {x['name']: min(rows, key=lambda d: abs(d['t'] - x['zero_t'])) for x in metrics['blocks']}
    imu = np.load(ROOT / '.runtime/hand_imu_observation_20260920T162851_bphL/analysis_intent/imu_signals.npz')
    si = Slerp(imu['t'], R.from_quat(imu['q']))
    offset = metrics['imu_common_offset_s']
    f24 = ImageFont.truetype(FONT, 24)
    f22 = ImageFont.truetype(FONT, 22)
    f19 = ImageFont.truetype(FONT, 19)
    raw_path = args.run / 'comparison_intermediate.mp4'
    video = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*'mp4v'), 8, (1280, 688))
    clips = [(54, 64, 'horizontal', '平移到一侧边缘'), (69, 77, 'horizontal', '平移到另一侧边缘'),
             (176, 182, 'vertical', '上下平移'), (265, 271, 'yaw_control', '真实偏航')]
    manifest = []
    stills = set()
    for lo, hi, block, title in clips:
        for requested in np.arange(lo, hi, 1 / 8):
            row = saved[int(np.argmin(abs(times - requested)))]
            rgb = cv2.cvtColor(cv2.imread(str(args.run / 'source_png' / f'{row["seq"]}.png')), cv2.COLOR_BGR2RGB)
            canvas = Image.new('RGB', (1280, 688), '#F7F8FA')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 10), f'{title} | 原视频 {row["t"]:.3f} 秒', font=f24, fill='#17212B')
            draw.text((16, 50), '原始相机画面', font=f22, fill='#17212B')
            draw.text((660, 50), '原动态框：实际输入', font=f22, fill=COLORS['dynamic'])
            draw.text((976, 50), '固定大框：实际输入', font=f22, fill=COLORS['fixed'])
            canvas.paste(Image.fromarray(rgb), (8, 88))
            draw.rectangle((8, 88, 647, 567), outline=COLORS['fixed'], width=3)
            b = row['bbox']
            draw.rectangle((8 + b[0], 88 + b[1], 8 + b[2], 88 + b[3]), outline=COLORS['dynamic'], width=2)
            for n, left in [('dynamic', 660), ('fixed', 976)]:
                crop, _, _ = prepare_hamer_crop(rgb, row[n]['box'], False)
                image = np.clip((crop.transpose(1, 2, 0) * np.array([.229, .224, .225]) + np.array([.485, .456, .406])) * 255, 0, 255).astype(np.uint8)[:, 32:-32]
                canvas.paste(Image.fromarray(image).resize((288, 384), Image.Resampling.NEAREST), (left, 88))
                draw.text((left, 484), f'手框高 {row[n]["input_hand_height_px"]:.1f} 像素', font=f19, fill='#17212B')
                if row[n]['valid']:
                    a = float(np.rad2deg((R.from_quat(row[n]['palm_q']) * R.from_quat(refs[block][n]['palm_q']).inv()).magnitude()))
                    label = f'相对起点转角：{a:.1f}°'
                else:
                    label = '无有效旋转输出'
                draw.text((left, 517), label, font=f19, fill=COLORS[n])
            t = row['t'] + offset
            i = np.clip(np.searchsorted(imu['t'], t), 1, len(imu['t']) - 1)
            if imu['time_valid'][i] and imu['time_valid'][i - 1]:
                a = float(np.rad2deg((si(refs[block]['t'] + offset).inv() * si(t)).magnitude()))
                text = f'同步 IMU 相对起点转角：{a:.1f}°（参照，并非独立真值）'
            else:
                text = '该帧没有有效的同步 IMU 参照'
            draw.text((16, 583), text, font=f22, fill='#17212B')
            draw.text((16, 622), '两组使用同一原图；模型输入都按原工程做左手镜像。固定框的位置、尺寸全程不变。', font=f19, fill='#4B5563')
            draw.text((16, 655), '每秒展示 8 帧；只选取已计算的原始帧，不插值姿态。三种完整对照见曲线及逐帧数据。', font=f19, fill='#4B5563')
            video.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
            for target in [62., 75., 180., 268.]:
                if abs(row['t'] - target) < .12 and target not in stills:
                    canvas.save(args.out / f'input_example_{int(target)}s.png')
                    stills.add(target)
            manifest.append(dict(block=block, source_t=row['t'], seq=row['seq'], source_frame=row['frame']))
    video.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(raw_path), '-c:v', 'libx264', '-crf', '21', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(args.out / '固定裁剪框回放.mp4')], check=True)
    (args.out / 'replay_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Source-timed input video and three raw comparison curves written.')


if __name__ == '__main__':
    main()
