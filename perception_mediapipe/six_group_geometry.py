"""Stateless A--F geometry. No temporal filtering, IMU correction or holding."""
import time
import numpy as np

PALM = [0, 5, 9, 13, 17]


def axes(width, longitudinal):
    x = np.asarray(width, float).copy()
    y = np.asarray(longitudinal, float).copy()
    if not np.all(np.isfinite([x, y])):
        raise ValueError('NONFINITE_GEOMETRY')
    if np.linalg.norm(y) < .015:
        raise ValueError('DEGENERATE_PALM_LENGTH')
    y /= np.linalg.norm(y)
    x -= y * np.dot(x, y)
    if np.linalg.norm(x) < .01:
        raise ValueError('DEGENERATE_PALM_WIDTH')
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y)))


def five_point_frame(points):
    p = np.asarray(points, float)
    return axes(p[1]-p[4], p[1:].mean(axis=0)-p[0])


def measured_points(xy, depth, scale, intr):
    """Same-frame 5x5 measured depth only; no model z or old depth accepted."""
    points = []
    h, w = depth.shape
    for u, v in np.asarray(xy, float):
        ix, iy = int(round(u)), int(round(v))
        if not (2 <= ix < w-2 and 2 <= iy < h-2):
            raise ValueError('LANDMARK_OUTSIDE_DEPTH_IMAGE')
        values = depth[iy-2:iy+3, ix-2:ix+3].astype(float).ravel()*scale
        good = values[(values > .15) & (values < 2.5)]
        if len(good) < 8:
            raise ValueError('INSUFFICIENT_LANDMARK_DEPTH')
        z = float(np.median(good))
        if np.median(np.abs(good-z)) > .012:
            raise ValueError('MIXED_DEPTH_AT_LANDMARK')
        points.append([(u-intr['ppx'])*z/intr['fx'], (v-intr['ppy'])*z/intr['fy'], z])
    return np.asarray(points)


def region_definition(vertices, joints, root_weights):
    """Freeze four disjoint 12-vertex regions on the neutral MANO template.

    Selection uses anatomical template geometry, never recordings or IMU error.
    Both skin surfaces are allowed; this defines a regional volume centroid.
    Root skinning weight excludes articulated fingers and most of the thumb.
    """
    frame = five_point_frame(joints[PALM])
    local = (vertices-joints[0]) @ frame
    length = np.linalg.norm(joints[PALM[1:]].mean(0)-joints[0])
    width = np.linalg.norm((joints[5]-joints[17]) @ frame[:, 0])
    uv = local[:, :2] / [width, length]
    pool = np.flatnonzero((uv[:, 1] >= .05) & (uv[:, 1] <= .88)
                         & (np.abs(uv[:, 0]) <= .60) & (root_weights > .45))
    targets = {'near': [0., .18], 'distal': [0., .72],
               'thumb': [.38, .45], 'little': [-.38, .45]}
    regions = {}
    used = set()
    for name, target in targets.items():
        order = pool[np.argsort(np.sum((uv[pool]-target)**2, axis=1), kind='stable')]
        ids = [int(i) for i in order if int(i) not in used][:12]
        if len(ids) != 12:
            raise ValueError('INSUFFICIENT_NEUTRAL_PALM_VERTICES')
        regions[name] = ids
        used.update(ids)
    return regions


def regional_frame(vertices, regions):
    c = {k: vertices[v].mean(axis=0) for k, v in regions.items()}
    return axes(c['thumb']-c['little'], c['distal']-c['near'])


def hamer_frames(pred, definition, regions, groups='ABCD'):
    from perception_hamer.src.mano_wrist_reference import estimate_mano_wrist_frame
    common = five_point_frame(definition.joints[PALM])
    mirror = np.eye(3) if pred.is_right else np.diag([-1., 1., 1.])
    results = {}
    for group in groups:
        start = time.perf_counter()
        result = dict(valid=False, reason='')
        try:
            if group == 'A':
                # Two-sided reflection keeps det=+1, then anatomical basis.
                rot = mirror @ pred.global_orient @ mirror @ common
            elif group == 'B':
                fit = estimate_mano_wrist_frame(pred.pred_vertices_source_camera_axes, definition)
                if not fit.valid:
                    raise ValueError('REGISTRATION_FAILED:'+fit.failure_reason)
                rot = fit.rotation @ definition.frame.T @ common
                result['registration'] = fit.quality
            elif group == 'C':
                # With frozen betas, these five MANO joints are the wrist and
                # four direct root children. In exact rotational arithmetic
                # this is the same root orientation as A, up to a fixed basis.
                # Treat C as a consistency control, not an independent estimate.
                rot = five_point_frame(pred.pred_keypoints_3d_source_camera_axes[PALM])
            else:
                rot = regional_frame(pred.pred_vertices_source_camera_axes, regions)
                neutral = regional_frame(definition.vertices, regions)
                rot = rot @ neutral.T @ common
            # Native network matrices have small floating-point orthogonality
            # error. Match the inference adapter's numerical tolerance, then
            # project all methods to SO(3); this is not temporal smoothing.
            if not np.allclose(rot.T @ rot, np.eye(3), atol=5e-3) or abs(np.linalg.det(rot)-1.) > .005:
                raise ValueError('INVALID_ROTATION_MATRIX')
            u,_,vt = np.linalg.svd(rot)
            numerical = u @ vt
            result['numerical_orthogonalization_frobenius'] = float(np.linalg.norm(rot-numerical))
            rot = numerical
            result.update(valid=True, rotation=rot.reshape(-1).tolist())
        except (ValueError, np.linalg.LinAlgError) as exc:
            result['reason'] = str(exc)
        result['geometry_ms'] = 1000*(time.perf_counter()-start)
        results[group] = result
    return results


def mp_frames(detection, depth, scale, intr, groups='EF'):
    results = {}
    for group in groups:
        start = time.perf_counter()
        result = dict(valid=False, position_valid=False, reason='')
        try:
            if detection is None:
                raise ValueError('NO_HAND_DETECTION')
            if not detection['visibility']['palm_inside_image']:
                raise ValueError('PALM_OUTSIDE_IMAGE')
            xy = np.asarray(detection['xy'])
            if group == 'E':
                p = np.asarray(detection['world'])[PALM]
            else:
                # Deliberately no access to detection['world'] in group F.
                p = measured_points(xy[PALM], depth, scale, intr)
                lengths = np.linalg.norm(p[1:]-p[0], axis=1)
                if np.any(lengths < .025) or np.any(lengths > .18):
                    raise ValueError('INCONSISTENT_PALM_DEPTH')
            rot = five_point_frame(p)
            result.update(valid=True, rotation=rot.reshape(-1).tolist())
            try:
                wrist = p[0] if group == 'F' else measured_points(xy[[0]], depth, scale, intr)[0]
                result.update(position_valid=True, position=wrist.tolist())
            except ValueError as exc:
                result['position_reason'] = str(exc)
        except (ValueError, np.linalg.LinAlgError) as exc:
            result['reason'] = str(exc)
        result['geometry_ms'] = 1000*(time.perf_counter()-start)
        results[group] = result
    return results
