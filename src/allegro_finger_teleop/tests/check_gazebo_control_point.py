"""Measure both physical control points for the same wrist inputs under each pivot setting."""
import json
import socket
import time
from xml.etree import ElementTree

import numpy as np
import rospy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetLinkState
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from test_retarget import OPEN, Retargeter, TeleopControlGate, camera_packet, human_hand, load_robot, matrices, Rotation
from test_wrist_hand import PALM_TO_CAMERA
from test_hand_isolation import image_observation
from wrist_arm import rotation_vector

NS = "/fr3_allegro/"


def main():
    rospy.init_node("check_allegro_pivot", anonymous=True)
    pivot = rospy.get_param("/allegro_fingers/control_point")
    if rospy.get_param("/allegro_fingers/source") != "external" or not rospy.get_param("/allegro_fingers/follow_wrist"):
        raise ValueError("expected a dedicated external-input arm and hand simulation")
    state, gestures, contacts = {}, [], []
    rospy.Subscriber(NS+"joint_states", JointState, lambda m:state.update(zip(m.name,m.position)),queue_size=1)
    rospy.Subscriber(NS+"hand_gesture", String, lambda m:gestures.append(m.data),queue_size=20)
    rospy.Subscriber(NS+"thumb_contacts", ContactsState, lambda m:contacts.append((time.monotonic(), m.states)),queue_size=40)
    deadline=time.monotonic()+60
    while not state or not rospy.get_param(NS+"fingers_ready",False):
        if time.monotonic()>deadline:
            raise TimeoutError("simulation not ready")
        time.sleep(.1)
    link = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
    model = load_robot()
    description = ElementTree.fromstring(rospy.get_param("/robot_description"))
    hand = human_hand()
    basis = Retargeter(model,OPEN,hand).basis
    hand_names = rospy.get_param(NS+"hand_controller/joints")
    origin = np.array([0.,0.,.62])
    session = "control-point-"+str(time.monotonic_ns())
    gate = TeleopControlGate(require_hand_landmarks=True, resume_fixed_hand_after_loss=True)
    seq = 0
    port = rospy.get_param("/allegro_fingers/port")
    wire = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    def sample():
        result=link("fr3_allegro::algr_lh_palm","fr3_allegro::fr3_link0")
        if not result.success:
            raise RuntimeError(result.status_message)
        p,o=result.link_state.pose.position,result.link_state.pose.orientation
        r=matrices(Rotation.from_quat([o.x,o.y,o.z,o.w]))
        wrist=np.array([p.x,p.y,p.z])
        return wrist,wrist+r@model.tcp,r,np.array([state[n] for n in hand_names])

    def send(shift, tilt, shape, count, estimated=None):
        nonlocal seq
        r=PALM_TO_CAMERA @ basis @ matrices(Rotation.from_rotvec([0.,np.deg2rad(tilt),0.])) @ basis.T
        pos=origin+shift
        image = {name:value.tolist() for name,value in image_observation(shape,r,pos).items()}
        for _ in range(count):
            now=time.monotonic()
            seq+=1
            msg=camera_packet(seq,now,shape if estimated is None else estimated)
            msg.update(session_id=session,wrist_position_m=pos.tolist(),palm_rotation_row_major=r.ravel().tolist(),
                       measured_wrist_rotation_row_major=r.ravel().tolist())
            msg["hand_landmarks"].update(image)
            gate.confirm_packet(msg,lambda:True,now=now)
            wire.sendto(json.dumps(gate.decorate(msg)).encode(),("127.0.0.1",port))
            time.sleep(.06)

    zero=np.zeros(3)
    gate.request_confirmation(0)
    send(zero,0,hand,22)
    if not gate.enabled or not gestures:
        raise RuntimeError("calibration or hand gesture telemetry missing")
    start_wrist,start_palm,start_rot,start_q=sample()
    report=dict(control_point=pivot, mount_to_palm_mm=1000*np.linalg.norm(model.tcp))
    def displacement():
        w,p,r,q=sample()
        return dict(wrist_delta_mm=(1000*(w-start_wrist)).tolist(),palm_delta_mm=(1000*(p-start_palm)).tolist(),
                    wrist_drift_mm=1000*np.linalg.norm(w-start_wrist),palm_drift_mm=1000*np.linalg.norm(p-start_palm),
                    rotation_deg=np.rad2deg(np.linalg.norm(rotation_vector(start_rot.T@r))),
                    max_measured_finger_change_deg=np.rad2deg(np.abs(q-start_q)).max())
    try:
        send([0.,-.04,0.],0,hand,35)
        report['pure_translation']=displacement()
        send(zero,0,hand,25)
        # The model invents some flexion; the independently projected visible hand stays open.
        send(zero,25,hand,44,human_hand([.20,.15,.10,.18]))
        report['pure_rotation']=displacement()
        report['pure_rotation']['hand_gesture']=gestures[-1]
        send(zero,0,hand,25)
        report['isolated_contacts']={}
        for f,site in ((1,3),(0,1),(1,2),(2,1)):
            label=("index","middle","ring")[f]+"_"+("mcp","pip","dip","tip")[site]
            bend=np.eye(4)[f]*(1. if site==3 else .74)
            target=human_hand(bend,pinch=f,pinch_site=site)
            send([0.,-.016,0.],15,target,42)
            w,p,r,q=sample()
            last=time.monotonic()-.5
            names={n.split("::")[-2] for t,states in contacts if t>=last for s in states
                   for n in (s.collision1_name,s.collision2_name) if "algr_lh_th_" not in n}
            # PIP and DIP are shared boundaries. Locate the real contact in
            # world coordinates so touching either adjacent link is assessed.
            code = ("if","mf","rf")[f]
            distal_link = "algr_lh_{}_{}".format(code, ("px","md","ds","ds")[site])
            target_pose = link("fr3_allegro::"+distal_link, "world").link_state.pose
            xyz,quat = target_pose.position,target_pose.orientation
            center = np.array([xyz.x,xyz.y,xyz.z])
            if site == 3:
                center += matrices(Rotation.from_quat([quat.x,quat.y,quat.z,quat.w])) @ model.pad_offsets[f]
            points = [[point.x,point.y,point.z] for t,entries in contacts if t>=last for contact in entries
                      if any("algr_lh_"+code+"_" in n for n in (contact.collision1_name,contact.collision2_name))
                      for point in contact.contact_positions]
            gap = np.median(np.linalg.norm(np.asarray(points)-center,axis=1)) if points else None
            if site == 3:
                allowed_links = {distal_link}
                radius = model.pad_radii[f]
            else:
                joint = description.find("./joint[@name='algr_lh_{}_{}']".format(code,("mcp","pip","dip")[site]))
                allowed_links = {distal_link, joint.find("parent").get("link")}
                box = description.find("./link[@name='{}']/collision/geometry/box".format(distal_link))
                radius = .5*np.linalg.norm(np.fromstring(box.get("size"),sep=" ")[:2])
            near_site = bool(allowed_links & names) and gap is not None and gap < radius+.001
            report['isolated_contacts'][label]=dict(contact_at_requested_site=bool(near_site),classification=gestures[-1],contact_links=sorted(names),
                       contact_to_target_center_mm=float(1000*gap) if gap is not None else None,
                       rotation_deg=np.rad2deg(np.linalg.norm(rotation_vector(start_rot.T@r))),
                       other_long_finger_excursion_deg=np.rad2deg(np.max(np.abs(np.delete((q-start_q)[:12].reshape(3,4),f,0)))))
            send(zero,0,hand,24)
        send(zero,-12,human_hand([1,1,1,1]),42)
        report['fist']=dict(classification=gestures[-1],
                   long_finger_mean_curl_deg=np.rad2deg((sample()[3]-start_q)[:12].reshape(3,4)[:,1:].mean(1)).tolist())
        send(zero,0,hand,35)
    finally:
        wire.close()
    print(json.dumps(report,indent=2),flush=True)
    trans,turn=report['pure_translation'],report['pure_rotation']
    if (any(np.linalg.norm(np.asarray(trans[k])-[40,0,0])>4 for k in ('wrist_delta_mm','palm_delta_mm'))
            or abs(turn['rotation_deg']-25)>1 or turn[pivot+'_drift_mm']>3
            or turn['max_measured_finger_change_deg']>2.5 or turn['hand_gesture']!='free'
            or report['fist']['classification']!='power_grasp' or min(report['fist']['long_finger_mean_curl_deg'])<25
            or any(v['classification']!='thumb_'+k for k,v in report['isolated_contacts'].items())
            or not all(v['contact_at_requested_site'] for v in report['isolated_contacts'].values())):
        raise AssertionError("measured pivot or gesture acceptance failed")

if __name__=='__main__':
    main()
