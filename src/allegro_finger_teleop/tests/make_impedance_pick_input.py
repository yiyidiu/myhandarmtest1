"""Generate the repeatable synthetic D455-format grasp input used by the Gazebo check."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np

from test_retarget import TeleopControlGate, camera_packet, human_hand
from test_hand_isolation import image_observation
from test_wrist_motion import PALM_TO_CAMERA, POSITION
from perception_hamer.src.finger_observation import finger_shape_signature, normalized_thumb_tip_gaps


def trajectory():
    yield from [("open", 0., 0.)] * 45
    yield from (("close", bend, 0.) for bend in np.linspace(0, 1, 65))
    yield from [("clamp", 1., 0.)] * 110
    yield from (("lift", 1., height) for height in np.linspace(0, .055, 50))
    yield from [("hold", 1., .055)] * 165
    yield from (("lower", 1., height) for height in np.linspace(.055, .022, 48))
    yield from [("lower", 1., .022)] * 24
    yield from (("release", bend, .022) for bend in np.linspace(1, 0, 35))
    yield from [("release", 0., .022)] * 55


def write_input(output):
    neutral = human_hand()
    pinched = human_hand([.67, 0, 0, 0], pinch=0)
    rotation = PALM_TO_CAMERA.ravel().tolist()
    start = 447000.
    gate = TeleopControlGate(require_hand_landmarks=True, retain_reference_until_operator_action=True)
    gate.request_confirmation(0, now=start)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp:
        path = Path(temp.name)
    try:
        with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
            for i, (stage, bend, height) in enumerate(trajectory()):
                t = i * .044
                points = neutral + (pinched - neutral) * bend
                position = POSITION + [0., 0., height]
                packet = camera_packet(i + 1, start + t, points)
                packet.pop("wrist_pose_baseline", None)
                hand = packet["hand_landmarks"]
                hand.update({k: v.tolist() for k, v in image_observation(points, PALM_TO_CAMERA, position).items()})
                hand["shape_signatures"] = [finger_shape_signature(points).tolist()] * 2
                hand["independent_tip_gaps"] = normalized_thumb_tip_gaps(points).tolist()
                palm = points[[0, 5, 9, 13, 17]]
                support = np.vstack((palm, .55*palm[0] + .45*palm[1:])) @ PALM_TO_CAMERA.T + position
                packet.update(stamp=1795590000. + t, session_id="allegro-synthetic-index-pick",
                              position_source="DIRECT_CAMERA_PALM_SKIN_DEPTH_NO_MODEL_OFFSET",
                              wrist_position_m=position.tolist(), palm_rotation_row_major=rotation,
                              measured_wrist_rotation_row_major=rotation, wrist_motion_support=support.tolist())
                gate.confirm_packet(packet, lambda: True, now=start+t, source_monotonic_s=start+t)
                row = dict(time=round(t, 4), stage=stage, packet=gate.decorate(packet))
                archive.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
        path.replace(output)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination .jsonl.gz file")
    write_input(parser.parse_args().output)
