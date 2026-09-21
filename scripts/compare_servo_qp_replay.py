#!/usr/bin/env python3
"""Replay recorded hand pose/fingers and C events for the Servo/QP comparison.

No image re-inference, edited intent, object knowledge, or grasp acceptance.
The simulator and complete control chain must already be running.
"""
import argparse
import copy
import csv
import gzip
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
from handarm_moveit_demo.msg import HamerHandPose


def read_inputs(source, end):
    with (source / 'screen_frame_timeline.csv').open() as stream:
        row = next(csv.DictReader(stream))
    origin = int(row['estimated_frame_wall_ns']) / 1e9 - float(row['video_time_s'])
    clock = []
    if (source / 'host_ros_clock.csv').exists():
        with (source / 'host_ros_clock.csv').open() as stream:
            for row in csv.DictReader(stream):
                stamp = int(row['ros_now_ns'])
                if stamp > 0:
                    clock.append((stamp / 1e9, int(row['host_wall_ns']) / 1e9))
    clock = np.array(clock)
    if len(clock):
        _, index = np.unique(clock[:, 0], return_index=True)
        clock = clock[index]
    records = []
    first_c = None
    with rosbag.Bag(str(source / 'control_chain.bag')) as bag:
        for _, old, stamp in bag.read_messages(topics=['/shared_teleop/hamer_pose']):
            receipt = stamp.to_sec()
            if receipt < 1e9:
                if not len(clock):
                    raise RuntimeError('ROS-time recording lacks a host clock bridge')
                receipt = float(np.interp(receipt, clock[:, 0], clock[:, 1]))
            video_t = receipt - origin
            if end is not None and video_t > end:
                break
            msg = HamerHandPose()
            for field in msg.__slots__:
                if hasattr(old, field):
                    setattr(msg, field, copy.deepcopy(getattr(old, field)))
            if first_c is None and msg.valid and msg.control_enabled:
                first_c = video_t
            age = receipt - msg.source_timestamp if msg.source_timestamp > 0 else 0.0
            records.append(dict(t=video_t, age=age, receipt_wall=receipt, msg=msg))
    if first_c is None:
        raise RuntimeError('No valid recorded C event')
    records = [r for r in records if r['t'] >= first_c - 1.0]
    return records, first_c


def run(args):
    records, first_c = read_inputs(args.source, args.end)
    if args.take is not None:
        records = [r for r in records if r['t'] <= first_c + args.take]
    rospy.init_node('servo_qp_recorded_input_comparison', anonymous=True)
    rospy.wait_for_service('/get_planning_scene', timeout=90)
    request = GetPlanningSceneRequest()
    request.components.components = 16 | 32
    scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)(request).scene
    if scene.world.collision_objects or scene.world.octomap.octomap.data:
        raise RuntimeError('Controller received forbidden environment geometry')
    solver = rospy.get_param('/shared_teleop/control/joint_velocity_solver')
    profiles = dict(servo=rospy.get_param('/servo_server', {}),
                    reference=rospy.get_param('/egm_position_reference_adapter'),
                    control=rospy.get_param('/shared_teleop'))
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    latest, observations, joints, faults, guards = {}, [], [], [], []
    begun = [None]
    capture_clock_shift = [None]
    first_t = records[0]['t']
    names = ['joint_' + str(i) for i in range(1, 7)]

    def video_time():
        return first_t + time.monotonic() - begun[0]

    def on_json(message, key):
        latest[key] = json.loads(message.data)

    def on_trend(message):
        data = json.loads(message.data)
        latest['trend'] = data
        if begun[0] is not None:
            observations.append(dict(t=video_time(), sim=rospy.Time.now().to_sec(),
                                     **copy.deepcopy(latest)))

    def on_joint(message):
        values = dict(zip(message.name, message.position))
        if not all(n in values for n in names):
            return
        latest['q'] = [values[n] for n in names]
        if begun[0] is not None:
            velocity = dict(zip(message.name, message.velocity))
            joints.append(dict(t=video_time(), sim=message.header.stamp.to_sec(),
                               q=latest['q'], qd=[velocity.get(n, 0.) for n in names]))

    def on_guard(message):
        latest['guard'] = message.data
        if begun[0] is not None:
            guards.append(dict(t=video_time(), reason=message.data))
            if message.data.startswith(('SELF_COLLISION:', 'INVALID_JOINT_STATE:')):
                if not faults or faults[-1]['reason'] != message.data:
                    faults.append(dict(t=video_time(), reason=message.data))

    rospy.Subscriber('/shared_teleop/trend_diagnostics', String, on_trend, queue_size=100)
    for key, topic in [('geometry', '/shared_teleop/geometric_filter_diagnostics'),
                       ('egm', '/egm_position_reference/diagnostics'),
                       ('output', '/shared_teleop/output_diagnostics')]:
        rospy.Subscriber(topic, String, on_json, callback_args=key, queue_size=1)
    rospy.Subscriber('/joint_states', JointState, on_joint, queue_size=10)
    rospy.Subscriber('/full_robot_self_collision_guard/diagnostic', String, on_guard, queue_size=1)
    publisher = rospy.Publisher('/shared_teleop/hamer_pose', HamerHandPose, queue_size=1)

    def publish(record, disable=False):
        msg = copy.deepcopy(record['msg'])
        msg.header.stamp = rospy.Time.now()
        if capture_clock_shift[0] is not None and msg.source_timestamp > 0:
            # Preserve acquisition intervals AND equality of repeated frames.
            # Replacing each stamp by now-age turns scheduling jitter into
            # artificial new camera samples in the upstream temporal filter.
            msg.source_timestamp += capture_clock_shift[0]
        else:
            msg.source_timestamp = time.time() - record['age']
        if disable:
            msg.control_gate_present = True
            msg.control_enabled = False
            msg.control_reference_epoch = 0
            msg.control_reference_token = ''
        publisher.publish(msg)

    zero = next(r for r in records if r['msg'].valid and r['msg'].control_enabled)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline and not rospy.is_shutdown():
        publish(zero, True)
        if publisher.get_num_connections() and latest.get('guard') == 'SAFE' and 'q' in latest:
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Robot feedback/controller not ready')
    warm = time.monotonic()
    initial_target = np.array([0., 0., 0., 0., np.pi/2, 0.])
    while time.monotonic() - warm < 15.0:
        publish(zero, True)
        time.sleep(.05)
        if (time.monotonic() - warm >= 3.0 and
                np.max(np.abs(np.array(latest['q']) - initial_target)) <= .01):
            break
    initial_q = latest['q']
    if np.max(np.abs(np.array(initial_q) - initial_target)) > .01:
        raise RuntimeError('Replay must start from the same approved initial joints: ' + str(initial_q))
    print('START', solver, args.source.name, 'video', round(first_c, 3),
          'through', round(records[-1]['t'], 3), 'initial_q', initial_q, flush=True)
    begun[0] = time.monotonic()
    capture_clock_shift[0] = time.time() - records[0]['receipt_wall']
    sim_start = rospy.Time.now().to_sec()
    lateness, last_print = [], -1
    try:
        for record in records:
            due = begun[0] + record['t'] - first_t
            while time.monotonic() < due and not rospy.is_shutdown():
                time.sleep(min(.002, max(0., due - time.monotonic())))
            if rospy.is_shutdown() or faults:
                break
            lateness.append(time.monotonic() - due)
            publish(record)
            if int(record['t'] / 30) != last_print:
                last_print = int(record['t'] / 30)
                print(solver, round(record['t'], 2), latest.get('geometry', {}).get('reason'),
                      latest.get('guard'), flush=True)
    finally:
        try:
            publish(zero, True)
        except rospy.ROSException:
            pass
    elapsed = time.monotonic() - begun[0]
    result = dict(scope=__doc__, source=str(args.source.resolve()), solver=solver,
                  source_commit=source_commit,
                  capture_timestamp_policy='single global shift; original intervals and duplicate stamps preserved',
                  profiles=profiles, planned_end_video_s=records[-1]['t'],
                  first_c_video_s=first_c, last_input_video_s=record['t'],
                  input_rows=len(lateness), initial_q=initial_q,
                  replay_lateness_p95_ms=float(np.percentile(lateness, 95) * 1000),
                  simulation_to_wall_ratio=(rospy.Time.now().to_sec()-sim_start)/elapsed,
                  observations=observations, joint_samples=joints, faults=faults,
                  guard_samples=guards,
                  source_age_summary_s=[float(np.percentile(
                      [r['age'] for r in records if r['msg'].valid], p)) for p in [0,50,95,100]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'wt') as stream:
        json.dump(result, stream)
    print('DONE', solver, 'samples', len(observations), 'faults', faults[:1],
          'RTF', result['simulation_to_wall_ratio'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--end', type=float)
    parser.add_argument('--take', type=float, help='Probe seconds after the first recorded C')
    run(parser.parse_args())
