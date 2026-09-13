#!/usr/bin/env python3
"""Gazebo-only continuous replay using existing operator labels as an oracle.

This diagnostic does not infer intent and is not a live teleoperation mode.
Only orientation changes; original positions, validity, confidence, timing and
C references are retained. Fingers are open in BOTH arms of the comparison.
Unlabelled intervals pass through, so this is a sparse-label intervention,
not an oracle for the entire recording or an independently measured truth.
"""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.runtime/hand_motion_observation_20260912T145617_iJYZ'
OUT = ROOT / '.runtime/known_intent_replay_20260913'
AXES = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
FROM_MATRIX = getattr(R, 'from_matrix', None) or R.from_dcm


def quaternion(q):
    return [q.x, q.y, q.z, q.w]


def read_inputs():
    import rosbag
    from handarm_moveit_demo.msg import HamerHandPose
    origin = json.loads((SOURCE / 'analysis_feedback/time_origin.json').read_text())['video_wall_origin']
    labels = json.loads((SOURCE / 'analysis_feedback/intent_data_definition.json').read_text())['annotations']['test']
    records = []
    with rosbag.Bag(str(SOURCE / 'control_chain.bag')) as bag:
        started = False
        for _, old, stamp in bag.read_messages(topics=['/shared_teleop/hamer_pose']):
            video_t = stamp.to_sec() - origin
            started = started or bool(old.valid and old.control_enabled)
            if not started or video_t > 320.5:
                continue
            msg = HamerHandPose()
            for field in msg.__slots__:
                setattr(msg, field, copy.deepcopy(getattr(old, field)))
            capture_t = msg.source_timestamp - origin
            label = next((a for a in labels if a['start'] <= capture_t <= a['end']), None)
            records.append(dict(t=video_t, capture_t=capture_t, msg=msg, annotation=label,
                                source_age=stamp.to_sec() - msg.source_timestamp))
    return records, labels


def prepare(records, release_s=.25, result_name='input_targets.json'):
    """Hold unsupported components; reconnect to absolute C angles smoothly.

    The baseline hard switch is saved for comparison. At reactivation only
    the held correction is faded out with a finite cubic transition. This
    avoids an instantaneous switch without integrating deleted increments.
    It cannot determine a correct absolute pose from a coarse action label.
    """
    axes = FROM_MATRIX(AXES)
    previous = None
    epoch = None
    output = np.zeros(3)
    hard_output = np.zeros(3)
    previous_raw = np.zeros(3)
    previous_mask = np.ones(3, dtype=bool)
    releasing = np.zeros(3, dtype=bool)
    correction = np.zeros(3)
    release_start = np.zeros(3)
    metrics = []
    for record in records:
        msg = record['msg']
        record['new_q'] = quaternion(msg.wrist_pose.orientation)
        if not (msg.valid and msg.control_enabled and min(msg.confidence[3:]) > 0):
            continue
        ref = R.from_quat(quaternion(msg.control_reference_wrist_pose.orientation))
        rotation = R.from_quat(quaternion(msg.wrist_pose.orientation))
        xyz = (axes * rotation * ref.inv() * axes.inv()).as_euler('XYZ')
        raw = np.array([-xyz[1], -xyz[2], xyz[0]])  # pitch, roll, yaw
        annotation = record['annotation']
        label = annotation['label'] if annotation else 'unlabelled'
        mask = (np.ones(3, dtype=bool) if label in ('unlabelled', 'combined')
                else np.array([label == x for x in ('pitch', 'roll', 'yaw')]))
        # Optional offline comparison with the previously saved four scores;
        # the Gazebo entry uses the operator annotations above.
        if 'support_mask' in record:
            mask = np.asarray(record['support_mask'], dtype=bool)
        if previous is None or epoch != msg.control_reference_token:
            output = raw.copy()
            hard_output = raw.copy()
            releasing[:] = False
        else:
            hard_output = np.where(mask, raw, hard_output)
            for axis in range(3):
                if not mask[axis]:
                    releasing[axis] = False
                    continue
                if not previous_mask[axis]:
                    correction[axis] = previous_raw[axis] - output[axis]
                    release_start[axis] = previous
                    releasing[axis] = True
                weight = 0.
                if releasing[axis]:
                    u = np.clip((record['capture_t'] - release_start[axis]) / release_s, 0., 1.)
                    weight = 1. - 3. * u * u + 2. * u * u * u
                    releasing[axis] = u < 1.
                output[axis] = raw[axis] - weight * correction[axis]
        mapped = R.from_euler('XYZ', [output[2], -output[0], -output[1]])
        record['new_q'] = (axes.inv() * mapped * axes * ref).as_quat().tolist()
        metrics.append(dict(t=record['t'], capture_t=record['capture_t'],
                            label=label, id=annotation['id'] if annotation else 0,
                            raw=raw.tolist(), hard=hard_output.tolist(), corrected=output.tolist()))
        previous = record['capture_t']
        previous_raw = raw.copy()
        previous_mask = mask.copy()
        epoch = msg.control_reference_token
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / result_name).write_text(json.dumps(dict(
        scope=__doc__, release_s=release_s, rows=metrics), ensure_ascii=False))
    return metrics


def replay(records, variant):
    import rospy
    from std_msgs.msg import String, Bool
    from sensor_msgs.msg import JointState
    from handarm_moveit_demo.msg import HamerHandPose
    rospy.init_node('known_intent_replay', anonymous=True)
    rospy.wait_for_message('/handarm_sim_demo/scene_ready', Bool, timeout=90)
    latest = {}
    observation = []
    start = [None]
    first_t = records[0]['t']
    fault = [None]
    def on_trend(message):
        if start[0] is None:
            return
        d = json.loads(message.data)
        observation.append(dict(t=first_t + time.monotonic() - start[0], trend=d,
                                **copy.deepcopy(latest)))
    def on_joint(message):
        indices = {name: i for i, name in enumerate(message.name)}
        keys = ['joint_' + str(i) for i in range(1, 7)]
        if all(k in indices for k in keys):
            latest['joint_position'] = [message.position[indices[k]] for k in keys]
            if len(message.velocity) == len(message.name):
                latest['joint_velocity'] = [message.velocity[indices[k]] for k in keys]
    def on_guard(message):
        latest['guard'] = message.data
        if start[0] is not None and message.data.startswith(('SELF_COLLISION:', 'INVALID_JOINT_STATE:')):
            fault[0] = message.data
    rospy.Subscriber('/shared_teleop/trend_diagnostics', String, on_trend, queue_size=100)
    rospy.Subscriber('/shared_teleop/geometric_filter_diagnostics', String,
                     lambda m: latest.update(geometry=json.loads(m.data)), queue_size=1)
    rospy.Subscriber('/egm_position_reference/diagnostics', String,
                     lambda m: latest.update(egm=json.loads(m.data)), queue_size=1)
    rospy.Subscriber('/joint_states', JointState, on_joint, queue_size=1)
    rospy.Subscriber('/full_robot_self_collision_guard/diagnostic', String, on_guard, queue_size=1)
    pub = rospy.Publisher('/shared_teleop/hamer_pose', HamerHandPose, queue_size=1)
    def publish(record, enabled=None):
        msg = copy.deepcopy(record['msg'])
        msg.header.stamp = rospy.Time.now()
        msg.source_timestamp = time.time() - record['source_age']
        if variant == 'oracle_continuous':
            q = record['new_q']
            msg.wrist_pose.orientation.x, msg.wrist_pose.orientation.y, msg.wrist_pose.orientation.z, msg.wrist_pose.orientation.w = q
        if enabled is not None:
            msg.control_enabled = enabled
            if not enabled:
                msg.control_reference_token = ''
                msg.control_reference_epoch = 0
        msg.finger_flexion = [.1] * 5
        msg.finger_tracking_present = True
        msg.finger_tracking_valid = True
        msg.finger_tracking_confidence = .9
        pub.publish(msg)
    warm = time.monotonic()
    while time.monotonic() - warm < 5:
        publish(records[0], False)
        time.sleep(.065)
    lateness = []
    start[0] = time.monotonic()
    last_print = -1
    for record in records:
        deadline = start[0] + record['t'] - first_t
        while time.monotonic() < deadline and not rospy.is_shutdown():
            time.sleep(min(.003, max(0., deadline - time.monotonic())))
        if rospy.is_shutdown() or fault[0]:
            break
        lateness.append(time.monotonic() - deadline)
        publish(record)
        if int(record['t'] / 20) != last_print:
            last_print = int(record['t'] / 20)
            print(variant, round(record['t'], 2), latest.get('geometry', {}).get('reason'), flush=True)
    publish(record, False)
    time.sleep(.2)
    result = dict(scope=__doc__, variant=variant, source=str(SOURCE), rows=observation,
                  scheduling_lateness_p95_ms=float(1000 * np.percentile(lateness, 95)),
                  fault=fault[0], final_input_time=record['t'])
    with gzip.open(OUT / (variant + '_actual.json.gz'), 'wt') as stream:
        json.dump(result, stream, ensure_ascii=False)
    print('saved', variant, 'time', record['t'], 'fault', fault[0], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=['prepare', 'baseline', 'oracle_continuous'])
    args = parser.parse_args()
    inputs, annotations = read_inputs()
    prepare(inputs)
    if args.variant != 'prepare':
        replay(inputs, args.variant)
