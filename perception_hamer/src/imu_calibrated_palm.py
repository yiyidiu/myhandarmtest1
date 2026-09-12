"""RGB-D pose correction calibrated offline against hand-mounted IMU motion.

No IMU stream or motion-intent label is used at runtime. Calibration and
subsequent observations share one C reference; this module never moves it.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def measure_palm_surface(depth, scale, intrinsics, points, previous_normal=None):
    """Estimate a palm frame from interior depth and 2-D anatomical seeds."""
    points = np.asarray(points, dtype=float)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        return {"valid": False, "reason": "INVALID_PALM_SEEDS"}
    if any(abs(float(x)) > 1e-8 for x in intrinsics.get("coeffs", [])):
        return {"valid": False, "reason": "DISTORTED_PROFILE_NOT_CALIBRATED"}
    hull = cv2.convexHull(points.astype(np.float32))[:, 0]
    centre = hull.mean(0)
    hull = centre + .70 * (hull-centre)
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    yy, xx = np.nonzero(mask)
    z = depth[yy, xx].astype(float) * scale
    keep = (z > .15) & (z < 2.)
    xx, yy, z = xx[keep], yy[keep], z[keep]
    if len(z) < 30:
        return {"valid": False, "reason": "TOO_FEW_SURFACE_PIXELS"}
    median = np.median(z)
    keep = np.abs(z-median) < .025
    xx, yy, z = xx[keep], yy[keep], z[keep]
    if len(z) < 30:
        return {"valid": False, "reason": "INSUFFICIENT_COHERENT_SURFACE"}
    ux = (xx-intrinsics['ppx']) / intrinsics['fx']
    vy = (yy-intrinsics['ppy']) / intrinsics['fy']
    mean = np.array([ux.mean(), vy.mean()])
    spread = np.array([np.std(ux), np.std(vy)])
    if min(spread) < 1e-4:
        return {"valid": False, "reason": "INSUFFICIENT_SURFACE_EXTENT"}
    design = np.column_stack([(ux-mean[0])/spread[0],
                              (vy-mean[1])/spread[1], np.ones(len(z))])
    weights = np.ones(len(z))
    for _ in range(5):
        sw = np.sqrt(weights)
        beta = np.linalg.lstsq(design*sw[:, None], sw/z, rcond=None)[0]
        inverse = design @ beta
        if np.any(inverse <= 0):
            return {"valid": False, "reason": "INVALID_SURFACE_PLANE"}
        residual = 1/inverse-z
        sigma = max(.001, 1.4826*np.median(np.abs(residual-np.median(residual))))
        weights = np.minimum(1., 1.5*sigma/np.maximum(np.abs(residual), 1e-9))
    a, b = beta[:2]/spread
    normal = np.array([a, b, beta[2]-a*mean[0]-b*mean[1]])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    if previous_normal is not None and normal @ previous_normal < 0:
        normal = -normal
    rays = np.column_stack([(points[:, 0]-intrinsics['ppx'])/intrinsics['fx'],
                             (points[:, 1]-intrinsics['ppy'])/intrinsics['fy'],
                             np.ones(len(points))])
    denominator = rays @ normal
    if np.min(np.abs(denominator)) < .03:
        return {"valid": False, "reason": "PALM_RAYS_NEAR_SURFACE_TANGENT"}
    xyz = rays * (np.sign(np.mean(denominator))/denominator)[:, None]
    longitudinal = xyz[1:].mean(0)-xyz[0]
    longitudinal -= normal*(longitudinal @ normal)
    if np.linalg.norm(longitudinal) < 1e-6:
        return {"valid": False, "reason": "PALM_LONGITUDINAL_AXIS_UNAVAILABLE"}
    longitudinal /= np.linalg.norm(longitudinal)
    width = np.cross(longitudinal, normal)
    width /= np.linalg.norm(width)
    frame = np.column_stack([normal, width, np.cross(normal, width)])
    return {"valid": True, "normal": normal.tolist(),
            "rotation_row_major": frame.reshape(-1).tolist(),
            "pixels": len(z), "depth_residual_rms_mm": float(np.sqrt(np.mean(residual**2))*1000),
            "scope": "DEPTH_SURFACE_AND_2D_PALM_SEEDS_NOT_GROUND_TRUTH"}


class ImuCalibratedPalmCorrector:
    def __init__(self, calibration_file, enabled=False):
        self.enabled = bool(enabled)
        if not self.enabled:
            return
        self.calibration = json.loads(Path(calibration_file).read_text())
        parameters = np.asarray(self.calibration['parameters'], dtype=float)
        if parameters.shape != (4,) or not np.all(np.isfinite(parameters)) or np.any(parameters < 0):
            raise ValueError('invalid RGB-D pose calibration parameters')
        self.view_weight, self.disagreement_weight, self.time_constant, self.large_y_response = parameters
        self._previous_normal = None
        self._reference_key = None
        self._frame_offset = None
        self._last_raw_bias = np.zeros(3)
        self._filtered_bias = np.zeros(3)
        self._last_rotation = None
        self._last_stamp = None

    def measure(self, depth, scale, intrinsics, detection, detection_age):
        if not self.enabled:
            return {"valid": False, "reason": "CORRECTION_DISABLED"}
        if ((intrinsics.get('width'), intrinsics.get('height')) != tuple(self.calibration['image_size']) or
                detection_age > .18 or not detection or not detection.get('wrist_pixel') or
                not detection.get('palm_mcp_pixels')):
            return {"valid": False, "reason": "NO_CURRENT_CALIBRATED_PALM_OBSERVATION"}
        try:
            result = measure_palm_surface(depth, scale, intrinsics,
                [detection['wrist_pixel']]+detection['palm_mcp_pixels'], self._previous_normal)
        except (ValueError, IndexError, np.linalg.LinAlgError, cv2.error) as exc:
            # An unavailable auxiliary depth plane does not invalidate the
            # hand pose, position, or finger channels.
            return {"valid": False, "reason": "SURFACE_FIT_UNAVAILABLE:"+type(exc).__name__}
        if result['valid']:
            self._previous_normal = np.asarray(result['normal'])
        return result

    def apply(self, packet, reference):
        output = dict(packet)
        if not self.enabled:
            return output
        if reference is None:
            self._reference_key = None
            self._last_rotation = None
            self._last_stamp = None
            return output
        if packet.get('valid') is not True or packet.get('hand_is_right') is not self.calibration['hand_is_right']:
            return output
        key = (reference['sequence'], reference['stamp'])
        reference_rotation = np.asarray(reference['palm_rotation_row_major']).reshape(3, 3)
        if key != self._reference_key:
            self._reference_key = key
            surface = reference.get('palm_surface_observation') or {}
            self._frame_offset = (np.asarray(surface['rotation_row_major']).reshape(3, 3).T @ reference_rotation
                                  if surface.get('valid') else None)
            self._last_raw_bias = np.zeros(3)
            self._filtered_bias = np.zeros(3)
            self._last_rotation = reference_rotation.copy()
            self._last_stamp = float(reference['stamp'])
        if packet.get('orientation_held', False) or not packet.get('orientation_channel_valid', True):
            output['palm_rotation_row_major'] = self._last_rotation.reshape(-1).tolist()
            output['imu_calibrated_pose'] = {'held': True, 'runtime_imu_used': False}
            return output
        raw = np.asarray(packet['palm_rotation_row_major']).reshape(3, 3)
        phi = Rotation.from_matrix(raw @ reference_rotation.T).as_rotvec()
        y = float(phi[1])
        phi[1] += self.large_y_response*np.sign(y)*max(0., abs(y)-np.deg2rad(5.))
        base = Rotation.from_rotvec(phi).as_matrix() @ reference_rotation
        surface = packet.get('palm_surface_observation') or {}
        if self._frame_offset is not None:
            if surface.get('valid'):
                geometric = np.asarray(surface['rotation_row_major']).reshape(3, 3) @ self._frame_offset
                self._last_raw_bias = Rotation.from_matrix(geometric @ raw.T).as_rotvec()
            else:
                geometric = Rotation.from_rotvec(self._last_raw_bias).as_matrix() @ raw
            bias = Rotation.from_matrix(geometric @ base.T).as_rotvec()
            p = packet['wrist_position_m']
            p0 = reference['wrist_position_m']
            du = max(0., np.arctan2(p[0], p[2])-np.arctan2(p0[0], p0[2]))
            weight = -np.expm1(-self.view_weight*du**2-self.disagreement_weight*np.dot(bias, bias))
            requested = weight*bias
        else:
            weight = 0.
            requested = np.zeros(3)
        stamp = float(packet['stamp'])
        dt = max(0., stamp-self._last_stamp)
        alpha = -np.expm1(-dt/self.time_constant) if self.time_constant > 1e-5 else 1.
        self._filtered_bias += alpha*(requested-self._filtered_bias)
        corrected = Rotation.from_rotvec(self._filtered_bias).as_matrix() @ base
        self._last_stamp = max(stamp, self._last_stamp)
        self._last_rotation = corrected.copy()
        output['palm_rotation_row_major'] = corrected.reshape(-1).tolist()
        output['orientation_source'] = str(packet.get('orientation_source', 'MANO_WRIST')) + '+OFFLINE_IMU_CALIBRATED_RGBD'
        output['imu_calibrated_pose'] = {
            'runtime_imu_used': False, 'held': False, 'surface_current': bool(surface.get('valid')),
            'surface_reference_available': self._frame_offset is not None,
            'geometry_weight': float(weight),
            'correction_deg': float(np.rad2deg(Rotation.from_matrix(corrected @ raw.T).magnitude())),
        }
        return output
