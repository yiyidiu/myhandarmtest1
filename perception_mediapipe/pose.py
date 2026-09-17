"""Two explicitly different pose constructions from the same MediaPipe Hands.

rgbd: image landmarks deprojected with measured, aligned depth.
world: predicted metric world landmarks (hand-centred); RGB-D locates the wrist.
Neither mode blends a HaMeR estimate or the former palm-plane auxiliary.
"""
import time
import numpy as np
import mediapipe as mp

from perception_mediapipe.fingers import observe_fingers
from perception_hamer.src.hand_detection_gate import hand_landmark_visibility

PALM = [0, 5, 9, 13, 17]


def palm_frame(points):
    p = np.asarray(points, dtype=float)
    y = p[1:].mean(axis=0) - p[0]  # wrist -> metacarpal bases
    x = p[1] - p[4]               # little -> index
    if np.linalg.norm(y) < .015 or np.linalg.norm(x) < .015:
        raise ValueError("DEGENERATE_PALM")
    y /= np.linalg.norm(y)
    x -= y * np.dot(x, y)
    if np.linalg.norm(x) < .01:
        raise ValueError("DEGENERATE_PALM_WIDTH")
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def depth_points(xy, depth, scale, intr):
    """Median 5x5 depth, no borrowing values from another frame or model."""
    h, w = depth.shape
    points = []
    for u, v in xy:
        ix, iy = int(round(u)), int(round(v))
        if not (2 <= ix < w-2 and 2 <= iy < h-2):
            raise ValueError("LANDMARK_OUTSIDE_DEPTH_IMAGE")
        patch = depth[iy-2:iy+3, ix-2:ix+3].astype(float).ravel()*scale
        good = patch[(patch > .15) & (patch < 2.5)]
        if len(good) < 8:
            raise ValueError("INSUFFICIENT_LANDMARK_DEPTH")
        z = float(np.median(good))
        if np.median(np.abs(good-z)) > .012:
            raise ValueError("MIXED_DEPTH_AT_LANDMARK")
        points.append([(u-intr['ppx'])*z/intr['fx'],
                       (v-intr['ppy'])*z/intr['fy'], z])
    return np.asarray(points)


class MediaPipePose:
    def __init__(self, fixed_left_hand=True):
        self.fixed_left_hand = bool(fixed_left_hand)
        # Match the existing detector: a fresh palm detection on every image.
        # Tracking-only acceleration is deliberately not another AB variable.
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2, model_complexity=1,
            min_detection_confidence=.5, min_tracking_confidence=.5)

    def close(self):
        self.hands.close()

    def detect(self, rgb, preferred_right=None):
        begin = time.perf_counter()
        result = self.hands.process(np.ascontiguousarray(rgb))
        elapsed = time.perf_counter()-begin
        candidates = []
        for image, world, handed in zip(result.multi_hand_landmarks or [],
                                       result.multi_hand_world_landmarks or [],
                                       result.multi_handedness or []):
            label = handed.classification[0]
            right = label.label.lower() == 'left'  # unmirrored camera, selfie labels
            if not self.fixed_left_hand and preferred_right is not None and right != preferred_right:
                continue
            xy = np.array([[v.x*rgb.shape[1], v.y*rgb.shape[0]] for v in image.landmark])
            xyz = np.array([[v.x, v.y, v.z] for v in world.landmark])
            if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(xyz)):
                continue
            visibility = hand_landmark_visibility(xy, rgb.shape[1], rgb.shape[0], palm_margin_px=8)
            low, high = xy.min(axis=0), xy.max(axis=0)
            margin = (high-low)*.18
            bbox = np.r_[low-margin, high+margin]
            if np.prod(bbox[2:]-bbox[:2])/(rgb.shape[0]*rgb.shape[1]) < .006:
                continue
            candidates.append(dict(xy=xy, world=xyz,
                is_right=False if self.fixed_left_hand else right,
                detector_reported_is_right=right,
                handedness_policy='fixed_left' if self.fixed_left_hand else 'automatic',
                confidence=float(label.score), bbox=bbox, visibility=visibility))
        candidate = max(candidates, key=lambda x:x['confidence']) if candidates else None
        return candidate, elapsed

    def poses(self, detection, depth, scale, intr, modes=('rgbd','world')):
        result = {}
        if detection is None:
            return {mode:dict(valid=False, reason='NO_HAND') for mode in modes}
        d = detection
        for mode in modes:
            begin = time.perf_counter()
            try:
                if not d['visibility']['palm_inside_image']:
                    raise ValueError('PALM_OUTSIDE_IMAGE')
                # Always anchor position to the measured wrist, never world[0]
                # (whose origin moves with the model's hand-centred coordinates).
                wrist = depth_points(d['xy'][[0]], depth, scale, intr)[0]
                points = (depth_points(d['xy'][PALM], depth, scale, intr)
                          if mode == 'rgbd' else d['world'][PALM])
                if mode == 'rgbd':
                    lengths = np.linalg.norm(points[1:]-points[0], axis=1)
                    if np.any(lengths > .18) or np.any(lengths < .025):
                        raise ValueError('INCONSISTENT_PALM_DEPTH')
                rotation = palm_frame(points)
                fingers = observe_fingers(d['world'], d['xy'], d['confidence'])
                if not d['visibility']['fingers_inside_image']:
                    fingers.update(valid=False, confidence=0., invalid_reason='FINGERTIPS_OUTSIDE_IMAGE')
                result[mode] = dict(valid=True, position=wrist.tolist(),
                    rotation=rotation.reshape(-1).tolist(), confidence=d['confidence'],
                    fingers=fingers, is_right=d['is_right'], reason='')
            except (ValueError, np.linalg.LinAlgError) as exc:
                result[mode] = dict(valid=False, reason=str(exc))
            result[mode]['geometry_ms'] = (time.perf_counter()-begin)*1000
        return result
