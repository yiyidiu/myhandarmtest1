#!/usr/bin/env python3
"""Paired raw HaMeR palm/ring replay; exact recorded RGB-D, no pose filtering.

This script extracts measurements only. It never treats IMU or model output as
ground truth, and does not replay commands to a robot.
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception_hamer.src.hamer_palm_frame import build_hamer_five_point_palm_frame
from perception_hamer.src.mano_wrist_reference import build_mano_wrist_definition, estimate_mano_wrist_frame
from perception_hamer.src.realtime_hamer_pipeline import normalized_crop_points_to_original, project_hamer_vertices_to_original
from perception_hamer.src.teleop_pose_packet import metric_palm_skin_from_arrays
from perception_mediapipe.recorded_rgbd import RecordedRGBD


def crop_camera(joints_source, normalized_points, affine):
    # Recorded 2-D projections and 3-D joints determine the missing crop-only
    # translation. Never use this translation as measured physical depth.
    x = np.array(joints_source, dtype=float)
    x[:, 0] *= -1
    uv = np.array(normalized_points) * 256 / 5000
    m = np.zeros((42, 3)); m[::2, 0] = 1; m[1::2, 1] = 1
    m[::2, 2] = -uv[:, 0]; m[1::2, 2] = -uv[:, 1]
    b = np.empty(42)
    b[::2] = uv[:, 0]*x[:, 2]-x[:, 0]
    b[1::2] = uv[:, 1]*x[:, 2]-x[:, 1]
    translation = np.linalg.lstsq(m, b, rcond=None)[0]
    projected, _ = project_hamer_vertices_to_original(x, translation, [5000, 5000], affine)
    recorded = normalized_crop_points_to_original(normalized_points, affine, 256)
    residual = float(np.max(np.linalg.norm(projected-recorded, axis=1)))
    if residual > .25:
        raise ValueError('CROP_CAMERA_RECONSTRUCTION_RESIDUAL_OVER_0_25_PIXEL')
    return translation, residual


def palm_width(p):
    direction = p[1:].mean(axis=0)-p[0]
    norm = np.linalg.norm(direction)
    return float(abs(np.cross(direction, p[-1]-p[1]))/norm) if norm > 1e-6 else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--geometry', type=Path, required=True)
    ap.add_argument('--end', type=float, default=350)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'yaw_source_frames').mkdir(exist_ok=True)
    origin = json.loads((args.source/'analysis_intent/time_origin.json').read_text())['video_wall_origin']
    geometry = json.loads(args.geometry.read_text())['False']
    definition = build_mano_wrist_definition(geometry['neutral_vertices'], geometry['neutral_joints'], geometry['faces'], False)
    neutral_palm = build_hamer_five_point_palm_frame(geometry['neutral_joints'], False).rotation
    basis = definition.frame.T@neutral_palm
    reader = RecordedRGBD(args.source/'rgbd.db3')
    rows = []; last_saved = -np.inf
    log = next(args.source.glob('perception/*/frames.jsonl'))
    for line in log.open():
        d = json.loads(line); t = float(d['timestamp'])-origin
        if t >= args.end: break
        presence = d.get('hand_presence') or {}
        row = dict(t=t, frame=d['frame_number'], seq=d['capture_sequence'],
                   inference_valid=d.get('valid'), original_reason=d.get('failure_reason'),
                   timing=d.get('timing'), bbox=d.get('bbox'), crop_quality=d.get('crop_quality'),
                   presence_reason=presence.get('reason'), detector_age_s=presence.get('detection_age_s'),
                   palm_valid=False, ring_valid=False, skin_valid=False,
                   original_published_valid=(d.get('published_pose') or {}).get('valid', False))
        if d.get('mano_joints') is None or d.get('mano_vertices') is None:
            row['geometry_reason']='NO_HAMER_GEOMETRY'; rows.append(row); continue
        if (d.get('published_pose') or {}).get('hand_is_right') is True:
            row['geometry_reason']='RIGHT_HAND_IN_LEFT_HAND_EXPERIMENT'; rows.append(row); continue
        start = time.perf_counter()
        palm = build_hamer_five_point_palm_frame(d['mano_joints'], False)
        row['palm_compute_ms']=(time.perf_counter()-start)*1000
        start = time.perf_counter()
        ring = estimate_mano_wrist_frame(d['mano_vertices'], definition)
        row['ring_compute_ms']=(time.perf_counter()-start)*1000
        row.update(palm_valid=palm.valid, ring_valid=ring.valid,
                   palm_reason=palm.failure_reason, ring_reason=ring.failure_reason,
                   ring_quality=ring.quality,
                   betas_frozen=(d.get('hamer_quality') or {}).get('betas_frozen'))
        if palm.valid: row['palm_q']=R.from_matrix(palm.rotation).as_quat().tolist()
        if ring.valid:
            row['ring_q']=R.from_matrix(ring.rotation).as_quat().tolist()
            row['ring_common_q']=R.from_matrix(ring.rotation@basis).as_quat().tolist()
        try:
            affine=d['hamer_quality']['affine_original_to_crop']
            pixels=normalized_crop_points_to_original(d['mano_joints_2d_crop_normalized'],affine,256)
            row['palm_pixels']=pixels[[0,5,9,13,17]].tolist()
            row['palm_width_px']=palm_width(pixels[[0,5,9,13,17]])
            detector=presence.get('confirmed_detection') or {}
            if detector.get('wrist_pixel') is not None and detector.get('palm_mcp_pixels') is not None:
                row['detector_palm_width_px']=palm_width(np.vstack([detector['wrist_pixel'],detector['palm_mcp_pixels']]))
            translation,residual=crop_camera(d['mano_joints'],d['mano_joints_2d_crop_normalized'],affine)
            row['projection_reconstruction_max_px']=residual
            v=np.array(d['mano_vertices']); v[:,0]*=-1
            ring_pixel=project_hamer_vertices_to_original(v[definition.wrist_loop].mean(axis=0)[None],translation,[5000,5000],affine)[0][0]
            row['ring_origin_pixel']=ring_pixel.tolist()
            # The software synchronizer drops its first injected pair at startup.
            # Retry the same original pair, never a neighboring source image.
            for attempt in range(2):
                try:
                    rgb,depth,diag=reader.read(d['frame_number'],d['timestamp']); break
                except RuntimeError:
                    if attempt: raise
            row.update(diag)
            if t>=265 and t-last_saved>=.12:
                cv2.imwrite(str(args.out/'yaw_source_frames'/f'{d["capture_sequence"]}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,93])
                row['source_image']=f'yaw_source_frames/{d["capture_sequence"]}.jpg';last_saved=t
            for name,origin_pixel in [('palm',pixels[0]),('ring',ring_pixel)]:
                try:
                    pos,conf,dg=metric_palm_skin_from_arrays(pixels[[0,5,9,13,17]],origin_pixel,depth,reader.scale,reader.intr)
                    row[name+'_position']=pos.tolist(); row[name+'_depth_confidence']=conf
                    if name=='palm':row['skin_diagnostics']=dg
                except ValueError as e: row[name+'_depth_reason']=str(e)
            row['skin_valid']='palm_position' in row and 'ring_position' in row
        except (ValueError,RuntimeError,KeyError) as e:row['depth_reason']=str(e)
        rows.append(row)
        if len(rows)%500==0:print('frames',len(rows),'video_s',round(t,2),flush=True)
    metadata=dict(source=str(args.source.resolve()),video_wall_origin=origin,
                  intrinsics=reader.intr,depth_scale=reader.scale,ring_to_palm_basis=basis.tolist(),
                  total_recorded_rgb=len(reader.meta['color']),total_rows=len(rows),end_excluded_s=args.end,
                  depth_policy='same measured palm skin patch; no model offset; no temporal fill',
                  orientation_policy='raw same inference; fixed left; no filters; no IMU correction')
    reader.close()
    with gzip.open(args.out/'paired_frames.json.gz','wt') as f:json.dump(rows,f)
    (args.out/'replay_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('done',len(rows),flush=True)


if __name__=='__main__':main()
