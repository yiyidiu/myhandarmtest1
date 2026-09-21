#!/usr/bin/env python3
"""Controlled projection intervention; synthetic evidence, not human ground truth.

One recorded MANO mesh is frozen. Relative rotation is prescribed by the scene,
not estimated from IMU. Only translation varies in the perspective arm. A 2-D
integer-shift arm and prescribed-rotation controls expose nuisance effects.
Vertex colors are baked once; unlit rendering prevents position-dependent
lighting/shadows. No detector, filtering, model retraining, or robot commands.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import numpy as np
import pyrender
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/hamer_runtime')]
from perception_hamer.src.hamer_crop_inference import HamerCropInference, prepare_hamer_crop
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame
from perception_hamer.src.realtime_hamer_pipeline import normalized_crop_points_to_original

K = np.array([[606.5111083984375, 0, 319.0163879394531],
              [0, 606.2513427734375, 248.72071838378906], [0, 0, 1.]])
BACKGROUND = (214, 219, 224)


def recorded_seed(source, seq=635):
    logs = list(source.glob('perception/*/frames.jsonl'))
    if logs:
        stream = logs[0].open()
        proc = None
    else:
        path = next(source.glob('perception/*/frames.jsonl.zst'))
        proc = subprocess.Popen(['zstd', '-q', '-dc', str(path)], stdout=subprocess.PIPE, text=True)
        stream = proc.stdout
    result = None
    for line in stream:
        d = json.loads(line)
        if d['capture_sequence'] == seq:
            result = d
            break
    stream.close()
    if proc:
        proc.terminate()
        proc.wait()
    if result is None:
        raise ValueError('Exact seed source record not found')
    return result


def project(points):
    image = points @ K.T
    return image[:, :2] / image[:, 2:]


def crop_rgb(rgb, box):
    crop, affine, diag = prepare_hamer_crop(rgb, box, False)
    img = np.clip((crop.transpose(1, 2, 0) * [.229, .224, .225] + [.485, .456, .406]) * 255, 0, 255).astype(np.uint8)
    return img[:, 32:-32], crop[:, :, 32:-32], affine, diag


def run(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    crossed_yaw = getattr(args, 'crossed_yaw', False)
    if crossed_yaw and not args.controlled_crop:
        raise ValueError('The crossed-pose probe requires the fixed-size controlled crop')
    gain = None
    if crossed_yaw:
        gain = json.loads((ROOT / 'docs/imu_view_correction_20260920/correction_metrics.json').read_text())['calibration']['gain']
    for name in ['images', 'inputs']:
        (args.out / name).mkdir(exist_ok=True)
    seed = recorded_seed(args.source)
    model = HamerCropInference(str(ROOT / 'perception_hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt'),
        data_root=str(ROOT / 'perception_hamer/_DATA'), precision='fp16', freeze_betas=True)
    model.load()
    model.warmup()
    model.set_frozen_betas(False, np.array(seed['betas_user']))
    verts = np.array(seed['mano_vertices'], dtype=float)
    joints = np.array(seed['mano_joints'], dtype=float)
    # Physical-left output includes a reflection; reverse winding accordingly.
    faces = model.mano_faces()[:, [0, 2, 1]]
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    verts -= center
    joints -= center
    depth = K[1, 1] * np.ptp(verts[:, 1]) / 160.
    base_translation = np.array([0., 0., depth])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Prebake diffuse shading. Colors remain attached to vertices throughout.
    direction = np.array([-.3, -.4, -1.]); direction /= np.linalg.norm(direction)
    brightness = .52 + .48 * np.clip(mesh.vertex_normals @ direction, 0, 1)
    colors = np.column_stack([np.array([202, 158, 127])[None] * brightness[:, None], np.full(len(verts), 255)]).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    obj = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene = pyrender.Scene(bg_color=np.r_[np.array(BACKGROUND) / 255, 1.], ambient_light=[0, 0, 0])
    node = scene.add(obj)
    # OpenCV camera x right/y down/z forward -> OpenGL y up/z backward.
    cv_to_gl = np.diag([1., -1., -1., 1.])
    scene.add(pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=.01, zfar=10.))
    renderer = pyrender.OffscreenRenderer(640, 480)

    def render(rot, translation):
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = translation
        scene.set_pose(node, cv_to_gl @ pose)
        rgb, z = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
        points = verts @ rot.T + translation
        pixels = project(points)
        lo, hi = pixels.min(axis=0), pixels.max(axis=0)
        box = np.r_[lo, hi]
        if min(lo) < 1 or hi[0] > 639 or hi[1] > 479:
            raise ValueError('Rendered hand clipped by source image boundary')
        if not (z > 0).any():
            raise ValueError('Renderer returned no visible hand')
        return rgb, box

    base, base_box = render(np.eye(3), base_translation)
    cases = [('center', 0., np.eye(3), base_translation.copy(), None)]
    if crossed_yaw:
        # One hypothesis: does the frozen horizontal correction generalize
        # across known global orientations at the same shape and distance?
        for yaw in [-80., -60., -30., 0., 30., 60., 80.]:
            for view in [-20., 0., 20.]:
                if yaw == 0. and view == 0.:
                    continue
                trans = base_translation.copy()
                dx = int(round(K[0, 0] * np.tan(np.deg2rad(view))))
                trans[0] = depth * dx / K[0, 0]
                cases.append((f'cross_yaw_{yaw:+05.1f}', view,
                    R.from_euler('y', yaw, degrees=True).as_matrix(), trans, None))
    else:
        # These minimal controls are evaluated before interpreting the translation.
        for axis in ['y', 'x']:
            for degrees in [-20., 20.]:
                cases.append(('rotation_' + axis, degrees, R.from_euler(axis, degrees, degrees=True).as_matrix(), base_translation.copy(), None))
        for degrees in [-20., -10., 10., 20.]:
            trans = base_translation.copy()
            dx = int(round(K[0, 0] * np.tan(np.deg2rad(degrees))))
            trans[0] = depth * dx / K[0, 0]
            cases.append(('perspective_translation', degrees, np.eye(3), trans, None))
            cases.append(('image_translation', degrees, np.eye(3), base_translation.copy(), dx))
    results = []
    for kind, value, rot, trans, image_shift in cases:
        key = f'{kind}_{value:+05.1f}'
        if image_shift is None:
            rgb, silhouette_box = render(rot, trans)
            # Move a fixed-size local crop only by the known projection of the
            # 3-D center. Do not derive its center/size from changing silhouette.
            shift = project(trans[None])[0] - project(base_translation[None])[0]
            box = base_box + np.r_[shift, shift] if args.controlled_crop else silhouette_box
        else:
            rgb = cv2.warpAffine(base, np.array([[1., 0., image_shift], [0., 1., 0.]]), (640, 480),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND)
            box = base_box + [image_shift, 0, image_shift, 0]
            silhouette_box = box.copy()
        margin = args.background_margin
        input_rgb = cv2.copyMakeBorder(rgb, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=BACKGROUND) if margin else rgb
        input_box = box + margin
        display, input_tensor, affine, diag = crop_rgb(input_rgb, input_box)
        cv2.imwrite(str(args.out / 'images' / (key + '.png')), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.out / 'inputs' / (key + '.png')), cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        pred = model.infer(input_rgb, input_box, False, float(len(results)))
        palm = build_hamer_five_point_palm_frame(pred.pred_keypoints_3d_source_camera_axes, False)
        if not palm.valid:
            raise ValueError(palm.failure_reason)
        if not results:
            base_prediction = R.from_matrix(palm.rotation)
            base_tensor = input_tensor.copy()
        if crossed_yaw:
            ray = float(np.arctan2((box[0] + box[2]) / 2 - K[0, 2], K[0, 0]))
            corrected = R.from_euler('y', gain * ray) * R.from_matrix(palm.rotation)
            if not results:
                corrected_reference = corrected
        predicted_relative = R.from_matrix(palm.rotation) * base_prediction.inv()
        error = R.from_matrix(rot).inv() * predicted_relative
        px_height = float((silhouette_box[3] - silhouette_box[1]) * 256 / diag['crop_bbox_size_px'])
        row = dict(kind=kind, value_deg=value, true_rotation=rot.tolist(), translation_m=trans.tolist(),
            actual_view_direction_deg=float(np.degrees(np.arctan2(trans[0], trans[2]))),
            image_shift_px=image_shift, source_box=box.tolist(), silhouette_box=silhouette_box.tolist(),
            input_box=input_box.tolist(), input_hand_height_px=px_height,
            input_mean_absolute_difference=float(np.mean(abs(input_tensor - base_tensor))),
            input_max_absolute_difference=float(np.max(abs(input_tensor - base_tensor))),
            predicted_rotation=palm.rotation.tolist(),
            predicted_relative_rotation=predicted_relative.as_matrix().tolist(),
            predicted_relative_rotvec_deg=np.rad2deg(predicted_relative.as_rotvec()).tolist(),
            predicted_angle_deg=float(np.rad2deg(predicted_relative.magnitude())),
            full_relative_error_deg=float(np.rad2deg(error.magnitude())),
            projected_joints=(normalized_crop_points_to_original(pred.pred_keypoints_2d_crop_normalized, pred.quality['affine_original_to_crop'], 256) - margin).tolist(),
            image='images/' + key + '.png', input_image='inputs/' + key + '.png')
        if crossed_yaw:
            corrected_relative = corrected * corrected_reference.inv()
            row.update(true_yaw_deg=0. if kind == 'center' else float(kind.split('_')[-1]),
                status='valid', fixed_horizontal_gain=gain,
                crop_center_view_deg=float(np.rad2deg(ray)),
                corrected_rotation=corrected.as_matrix().tolist(),
                corrected_relative_rotvec_deg=np.rad2deg(corrected_relative.as_rotvec()).tolist(),
                corrected_full_relative_error_deg=float(np.rad2deg((R.from_matrix(rot).inv() * corrected_relative).magnitude())))
        results.append(row)
        print(kind, value, 'true rotation', np.rad2deg(R.from_matrix(rot).magnitude()),
            'predicted', row['predicted_angle_deg'], 'full error', row['full_relative_error_deg'],
            'hand pixels', px_height, flush=True)
        (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    renderer.delete()
    np.savez_compressed(args.out / 'frozen_scene.npz', vertices=verts, joints=joints,
        faces=faces, colors=colors, camera_K=K, betas=np.array(seed['betas_user']))
    metadata = dict(seed_source=str(args.source.resolve()), seed_capture_sequence=635,
        seed_timestamp=seed['timestamp'], camera_K=K.tolist(), physical_hand='left',
        depth_m=float(depth), precision='fp16', renderer='pyrender EGL; FLAT baked vertex colors',
        background=BACKGROUND, background_margin_px=margin, controlled_crop=args.controlled_crop,
        roi=('Fixed pixel size, center follows known 3-D center projection only' if args.controlled_crop else 'Exact mesh bounding box') + '; rescale=2; no detector',
        filtered=False, primary_hypothesis='Perspective appearance change under pure 3-D translation changes inferred orientation despite fixed true rotation',
        boundary='Synthetic mechanism probe; not an independent measurement of real hand pose and not a proof of the dominant cause in recorded human data')
    if crossed_yaw:
        metadata.update(primary_hypothesis='A frozen position-only correction reduces viewpoint-induced orientation error across different true orientations',
            prescribed_camera_y_rotations_deg=[-80, -60, -30, 0, 30, 60, 80],
            prescribed_horizontal_views_deg=[-20, 0, 20], fixed_horizontal_gain=gain,
            gain_refit=False, model_receives_true_orientation=False,
            crop_source='Known scene-center projection; detector removed as a nuisance variable, not an end-to-end deployment test',
            boundary='Synthetic position-by-orientation probe at one shape and depth; no evidence about continuous-motion latency or general human performance')
        for row in results:
            center_row = next(x for x in results if x['true_yaw_deg'] == row['true_yaw_deg'] and x['value_deg'] == 0.)
            for label, field in [('raw', 'predicted_rotation'), ('corrected', 'corrected_rotation')]:
                delta = R.from_matrix(row[field]) * R.from_matrix(center_row[field]).inv()
                row[label + '_same_pose_position_error_deg'] = float(np.rad2deg(delta.magnitude()))
        (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--background-margin', type=int, default=0)
    p.add_argument('--controlled-crop', action='store_true')
    p.add_argument('--crossed-yaw', action='store_true', help='Cross three image positions with seven prescribed side-turn angles; keep the fitted gain fixed')
    run(p.parse_args())
