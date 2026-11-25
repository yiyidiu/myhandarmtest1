"""Measure finger-only arm leakage and bilateral yaw in an isolated wrist-mode Gazebo run."""
import json
import socket
import time

import numpy as np
import rospy
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from test_retarget import TeleopControlGate, human_hand, matrices, Rotation
from test_fingertip_graph import folded_fist, opposition
from test_wrist_motion import PALM_TO_CAMERA, POSITION, observed_packet
from wrist_arm import rotation_vector

NS = '/fr3_allegro/'


def check():
    rospy.init_node('check_fr3_wrist_motion',anonymous=True)
    if rospy.get_param('/allegro_fingers/source') != 'external' or rospy.get_param('/allegro_fingers/control_point') != 'wrist':
        raise ValueError('requires a dedicated wrist-mode external-input instance')
    port = rospy.get_param('/allegro_fingers/port')
    arm = rospy.get_param(NS+'arm_controller/joints')
    hand = rospy.get_param(NS+'hand_controller/joints')
    states, targets, commands = [], [], []
    rospy.Subscriber(NS+'joint_states',JointState,lambda m:states.append(dict(zip(m.name,m.position))),queue_size=12)
    rospy.Subscriber(NS+'palm_target',PoseStamped,lambda m:targets.append(m.pose),queue_size=12)
    rospy.Subscriber(NS+'arm_controller/command',JointTrajectory,lambda m:commands.append(m.points[-1].positions),queue_size=12)
    link = rospy.ServiceProxy('/gazebo/get_link_state',GetLinkState)
    deadline = time.monotonic()+90
    while not states or not rospy.get_param(NS+'fingers_ready',False):
        if time.monotonic()>deadline: raise TimeoutError('simulation did not become ready')
        time.sleep(.1)
    gate = TeleopControlGate(require_hand_landmarks=True,resume_fixed_hand_after_loss=True)
    session = 'wrist-isolation-'+str(time.monotonic_ns())
    sender = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    seq = 0
    neutral = human_hand()
    def send(hand, axis=None, deg=0., offset=None, bias=0., confidence=True):
        nonlocal seq
        rot = PALM_TO_CAMERA if axis is None else PALM_TO_CAMERA @ matrices(Rotation.from_rotvec(np.deg2rad(deg)*axis))
        now = time.monotonic()
        seq += 1
        packet = observed_packet(seq,now,hand,rot,POSITION+(np.zeros(3) if offset is None else offset),
                                 np.array([.014,-.012,.018])*bias,[0.,15.*bias,0.],session)
        if not confidence: packet['wrist_motion_support'] = None
        gate.confirm_packet(packet,lambda:True,now=now)
        sender.sendto(json.dumps(gate.decorate(packet),allow_nan=False).encode(),('127.0.0.1',port))
        time.sleep(.045)

    def coordinates(pose):
        t,r = pose.position,pose.orientation
        return np.array([t.x,t.y,t.z]),matrices(Rotation.from_quat([r.x,r.y,r.z,r.w]))

    def physical(name):
        reply = link('fr3_allegro::'+name,'fr3_allegro::fr3_link0')
        if not reply.success:raise RuntimeError(reply.status_message)
        return coordinates(reply.link_state.pose)

    def joints(names,entry):
        return np.array([entry[n] for n in names])

    gate.request_confirmation(0)
    for _ in range(26):send(neutral)
    if not gate.enabled or not targets or not commands:raise RuntimeError('calibration/arm target missing')
    p0,r0 = physical('algr_lh_palm')
    e0 = physical('fr3_link4')[0]
    q0 = joints(arm,states[-1])
    saved_state,saved_cmd,saved_targets = len(states),len(commands),len(targets)
    for shape in (human_hand([1,0,0,0]),opposition((2,)),folded_fist(),neutral):
        for i in range(28): send(shape,bias=np.sin(.22*(i+1)))
    measured_arm = np.array([joints(arm,s) for s in states[saved_state:]])
    measured_hand = np.array([joints(hand,s) for s in states[saved_state:]])
    def angle(a,b):return np.degrees(np.linalg.norm(rotation_vector(a@b.T)))
    target_samples = [coordinates(x) for x in targets[saved_targets:]]
    anchor = coordinates(targets[saved_targets-1])
    fingers = dict(injected_false_rotation_deg=15.,injected_false_translation_mm=float(np.linalg.norm([14,12,18])),
                   max_palm_target_shift_mm=max(1000*np.linalg.norm(p-anchor[0]) for p,_ in target_samples),
                   max_palm_target_turn_deg=max(angle(r,anchor[1]) for _,r in target_samples),
                   arm_command_range_deg=float(np.max(np.degrees(np.ptp(commands[saved_cmd:],axis=0)))),
                   actual_arm_joint_range_deg=float(np.max(np.degrees(np.ptp(measured_arm,axis=0)))),
                   actual_finger_joint_range_deg=float(np.max(np.degrees(np.ptp(measured_hand,axis=0)))))
    arcs = {}
    for index, axis_name in ((2,'thumb_edge_tilt'),(0,'image_plane_yaw')):
        axis = np.eye(3)[index]
        for sign in (1,-1):
            for fraction in np.linspace(0.,1.,18):send(neutral,axis,sign*25*fraction)
            for _ in range(30):send(neutral,axis,sign*25.)
            p,r = physical('algr_lh_palm')
            arcs[axis_name+str(sign)] = dict(measured_turn_deg=angle(r,r0),wrist_drift_mm=1000*np.linalg.norm(p-p0),
                        elbow_travel_mm=1000*np.linalg.norm(physical('fr3_link4')[0]-e0),
                        max_joint_change_deg=np.degrees(np.max(np.abs(joints(arm,states[-1])-q0))))
            for fraction in np.linspace(1.,0.,18):send(neutral,axis,sign*25*fraction)
            for _ in range(16):send(neutral,axis,0.)
    for fraction in np.linspace(0.,1.,50):
        send(human_hand([0,fraction,0,0]),np.eye(3)[0],15*fraction,[0.,-.04*fraction,0.])
    for _ in range(32):send(human_hand([0,1,0,0]),np.eye(3)[0],15.,[0.,-.04,0.])
    p,r = physical('algr_lh_palm')
    mixed = dict(wrist_delta_base_mm=(1000*(p-p0)).tolist(), wrist_turn_deg=angle(r,r0),
                 measured_middle_joint_excursion_deg=np.max(np.abs(np.degrees(joints(hand,states[-1])[4:8]-joints(hand,states[saved_state])[4:8]))))
    sender.close()
    print(json.dumps(dict(finger_only=fingers,arcs=arcs,simultaneous=mixed),indent=2),flush=True)
    if (max(fingers['max_palm_target_shift_mm'],fingers['max_palm_target_turn_deg'],fingers['arm_command_range_deg'])>.005
            or fingers['actual_arm_joint_range_deg']>.45 or fingers['actual_finger_joint_range_deg']<35):
        raise AssertionError('finger-only motion moved the arm or failed to articulate the fingers')
    if any(abs(v['measured_turn_deg']-25)>2.5 or v['wrist_drift_mm']>4. for v in arcs.values()):
        raise AssertionError('true wrist yaw failed')
    if (np.linalg.norm(1000*(p-p0)-[40.,0.,0.])>5. or abs(mixed['wrist_turn_deg']-15)>2.5
            or mixed['measured_middle_joint_excursion_deg']<35):
        raise AssertionError('simultaneous wrist and finger motion failed')


if __name__=='__main__':
    check()
