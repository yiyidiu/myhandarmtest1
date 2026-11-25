"""Exercise single/multiple opposition and fist closure via the live camera contract in wrist mode."""
import json
import socket
import time
from collections import Counter, deque
from xml.etree import ElementTree

import numpy as np
import rospy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetLinkState
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from test_fingertip_graph import folded_fist, opposition, retained_ring_pose
from test_retarget import TeleopControlGate, camera_packet, human_hand, load_robot, matrices, Rotation
from allegro_v4_limits import hardware_limits
from test_hand_isolation import image_observation, PALM_TO_CAMERA
from perception_hamer.src.finger_observation import finger_shape_signature

NS = "/fr3_allegro/"
NAMES = ("index", "middle", "ring")
CODES = ("if", "mf", "rf", "th")


def check(hold_only=False, extend_pinch=False):
    rospy.init_node("allegro_fingertip_graph_check", anonymous=True)
    if (rospy.get_param("/allegro_fingers/source") != "external"
            or rospy.get_param("/allegro_fingers/control_point") != "wrist"):
        raise ValueError("This test requires a dedicated wrist-mode external-input Gazebo instance")
    description = ElementTree.fromstring(rospy.get_param("/robot_description"))
    for name, span in hardware_limits():
        lim = description.find("./joint[@name='{}']/limit".format(name))
        np.testing.assert_allclose([float(lim.get(x)) for x in ("lower", "upper")], span, atol=1e-10, rtol=0)

    model = load_robot()
    state, gestures, hits = {}, [], deque(maxlen=160)
    commands, touched = {}, deque(maxlen=160)
    joint_history = deque(maxlen=450)
    def observe_state(msg):
        state.update(zip(msg.name, msg.position))
        if all(n in state for n in model.names):
            joint_history.append((time.monotonic(), np.array([state[n] for n in model.names])))
    rospy.Subscriber(NS+"joint_states", JointState, observe_state, queue_size=4)
    rospy.Subscriber(NS+"hand_gesture", String, lambda m: gestures.append(m.data), queue_size=12)
    rospy.Subscriber(NS+"hand_controller/command", JointTrajectory,
                     lambda m: commands.update(zip(m.joint_names, m.points[-1].positions)), queue_size=4)

    def observe(msg):
        bodies = [name for s in msg.states for name in (s.collision1_name, s.collision2_name)]
        pads = {i for i, code in enumerate(CODES[:3])
                if any("algr_lh_"+code+"_ds_tip_collision_" in name for name in bodies)}
        hits.append((time.monotonic(), pads))
        touched.append((time.monotonic(), {name.split("::")[-2] for name in bodies if "_th_" not in name}))
    rospy.Subscriber(NS+"thumb_contacts", ContactsState, observe, queue_size=80)
    get_link = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
    deadline = time.monotonic()+90
    while not state or not rospy.get_param(NS+"fingers_ready", False):
        if time.monotonic() >= deadline:
            raise TimeoutError("controllers did not become ready")
        time.sleep(.1)

    def pads():
        pts = []
        for digit, offset in zip(CODES, model.pad_offsets):
            rsp = get_link("fr3_allegro::algr_lh_"+digit+"_ds", "fr3_allegro::algr_lh_palm")
            if not rsp.success:
                raise RuntimeError(rsp.status_message)
            p, o = rsp.link_state.pose.position, rsp.link_state.pose.orientation
            pts.append(matrices(Rotation.from_quat([o.x, o.y, o.z, o.w])) @ offset + [p.x, p.y, p.z])
        return np.array(pts)

    hand = human_hand()
    gate = TeleopControlGate(require_hand_landmarks=True, resume_fixed_hand_after_loss=True)
    gate.request_confirmation(0)
    sequence = 0
    session = "finger-graph-" + str(time.monotonic_ns())
    source = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = rospy.get_param("/allegro_fingers/port")
    pos = np.array([0., 0., .65])

    def send(shape, count, turn=0):
        nonlocal sequence
        r = matrices(Rotation.from_euler("y", turn, degrees=True)) @ PALM_TO_CAMERA
        obs = {k:v.tolist() for k, v in image_observation(shape, r, pos).items()}
        obs["shape_signatures"] = [finger_shape_signature(shape).tolist(), finger_shape_signature(shape @ r.T+pos).tolist()]
        for _ in range(count):
            now = time.monotonic()
            sequence += 1
            pkt = camera_packet(sequence, now, shape)
            pkt.update(session_id=session, wrist_position_m=pos.tolist(),
                       position_source=rospy.get_param("/hamer_input_adapter/required_position_source", pkt["position_source"]),
                       palm_rotation_row_major=r.ravel().tolist(), measured_wrist_rotation_row_major=r.ravel().tolist())
            pkt["hand_landmarks"].update(obs)
            gate.confirm_packet(pkt, lambda: True, now=now)
            source.sendto(json.dumps(gate.decorate(pkt)).encode(), ("127.0.0.1", port))
            time.sleep(.053)

    def angles():
        return np.array([state[name] for name in model.names])

    groups = ((0,), (1,), (2,), (0,1), (1,2), (0,2), (0,1,2))
    def capture(group, q0):
        tips = pads()
        recent = [s for t,s in list(hits) if t > time.monotonic()-.58]
        joint_change = np.degrees(angles()-q0).reshape(4,4)
        cmd = np.array([commands[n] for n in model.names])
        return dict(
            recognized=gestures[-1], thumb_to_pad_centers_mm=(1000*np.linalg.norm(tips[:3]-tips[3],axis=1)).tolist(),
            adjacent_long_tip_spacing_mm=(1000*np.linalg.norm(np.diff(tips[:3],axis=0),axis=1)).tolist(),
            actual_joint_deg=np.degrees(angles()).round(2).tolist(), commanded_joint_deg=np.degrees(cmd).round(2).tolist(),
            touched_links=sorted(set().union(*(s for t,s in list(touched) if t>time.monotonic()-.58))),
            simultaneous_pad_contact_samples=sum(s == set(group) for s in recent),
            individual_pad_contact_samples=[sum(i in s for s in recent) for i in range(3)],
            contact_samples=len(recent), other_digits_max_joint_change_deg=max(
                [float(np.max(np.abs(joint_change[i]))) for i in range(3) if i not in group] or [0.]))

    def name(group):
        return "+".join(NAMES[i] for i in group)

    def test_retained_ring():
        send(retained_ring_pose(), 45)
        records = {}
        for label, old, new in (("index_flex", (0,0), (1,0)), ("index_extend_middle_flex", (1,0), (0,1)),
                                ("both_flex", (0,1), (1,1)), ("both_extend", (1,1), (0,0))):
            start, first = time.monotonic(), len(gestures)
            for fraction in np.linspace(0., 1., 16):
                pose = retained_ring_pose(*(np.asarray(old) + fraction*(np.asarray(new)-old)))
                send(pose, 1)
            send(pose, 14)
            records[label] = hold_measurement(start, first)
        switches = (("add_middle", (1,2), retained_ring_pose(.25,.50,True)),
                    ("remove_middle", (2,), retained_ring_pose(.25,.50)))
        for label, group, pose in switches if extend_pinch else ():
            start, first = time.monotonic(), len(gestures)
            send(pose, 40)
            rec = records[label] = hold_measurement(start, first)
            settled = capture(group, q0)
            rec["final_contacts"] = settled["simultaneous_pad_contact_samples"]
            rec["final_thumb_to_pad_mm"] = settled["thumb_to_pad_centers_mm"]
            rec["touched_links"] = settled["touched_links"]
            rec["final_gesture"] = gestures[-1]
        send(hand, 30)
        tips = pads()
        records["release"] = dict(gesture=gestures[-1], ring_to_thumb_mm=float(1000*np.linalg.norm(tips[2]-tips[3])))
        return records

    def hold_measurement(start, first):
        samples = [q for t,q in list(joint_history) if t >= start]
        seen = [s for t,s in list(hits) if t >= start]
        gaps = []
        for q in samples:
            p = model.kinematics(q)[0][:, 4]
            gaps.append(1000*(np.linalg.norm(p[2]-p[3])-model.pad_radii[[2,3]].sum()))
        return dict(recognized_frames=dict(Counter(gestures[first:])),
                    ring_contact_samples=sum(2 in s for s in seen), sampled_contacts=len(seen),
                    peak_ring_surface_gap_mm=max(0.,float(max(gaps))),
                    actual_joint_excursion_deg=np.degrees(np.ptp(samples,axis=0)).reshape(4,4).max(axis=1).tolist())

    records = {}
    try:
        send(hand, 22)
        q0 = angles()
        if hold_only:
            records = test_retained_ring()
            print(json.dumps(records, indent=2), flush=True)
            independent = ("index_flex", "index_extend_middle_flex", "both_flex", "both_extend")
            for label in independent:
                rec = records[label]
                if (set(rec["recognized_frames"]) != {"thumb_ring_tip"}
                        or max(rec["actual_joint_excursion_deg"][2:]) > 4.
                        or rec["ring_contact_samples"] < .98*rec["sampled_contacts"]):
                    raise AssertionError("bystander motion affected the stationary thumb-ring pair during " + label)
            changes = ("add_middle", "remove_middle") if extend_pinch else ()
            for label in independent + changes:
                rec = records[label]
                if (not rec["sampled_contacts"] or rec["ring_contact_samples"] < .90*rec["sampled_contacts"]
                        or rec["peak_ring_surface_gap_mm"] > 2.0
                        or any("ring" not in g for g in rec["recognized_frames"])):
                    raise AssertionError("lost retained ring while " + label)
            if (records["index_flex"]["actual_joint_excursion_deg"][0] < 35
                    or records["index_extend_middle_flex"]["actual_joint_excursion_deg"][1] < 35
                    or records["release"]["gesture"] != "free" or records["release"]["ring_to_thumb_mm"] < 60):
                raise AssertionError("free digits or intentional release failed")
            if extend_pinch and any(records[name]["final_contacts"] < 3 for name in changes):
                raise AssertionError("the additional digit did not enter or leave the measured contact set")
            return
        for group in groups:
            send(opposition(group), 48, 12)
            records[name(group)] = capture(group, q0)
            send(hand, 28)
        switches = records["change_without_opening"] = {}
        for group in ((0,), (0,1), (0,1,2), (1,2), (2,), (1,)):
            send(opposition(group), 48, 12)
            switches[name(group)] = capture(group, q0)
        send(hand, 30)
        for label, pose in (("folded",folded_fist()), ("thumb_open",folded_fist(False))):
            send(pose, 45)
            q = np.degrees(angles()-q0).reshape(4,4)
            records[label] = dict(recognized=gestures[-1], thumb_change_deg=q[3].tolist(),
                                  three_long_flexion_deg=q[:3,1:].tolist())
            send(hand, 29)
        records["reopen_max_error_deg"] = float(np.max(np.abs(np.degrees(angles()-q0))))
    finally:
        source.close()
    print(json.dumps(records, indent=2), flush=True)
    samples = [(name(group), records[name(group)]) for group in groups] + list(switches.items())
    for label, rec in samples:
        if (rec["recognized"] != "thumb_"+label.replace("+","_")+"_tip"
                or rec["simultaneous_pad_contact_samples"] < 3
                or rec["other_digits_max_joint_change_deg"] > 4.0):
            raise AssertionError(label+": "+str(rec))
    if (records["folded"]["recognized"] != "power_grasp" or max(records["folded"]["thumb_change_deg"][2:]) < 30
            or np.min(records["folded"]["three_long_flexion_deg"]) < 40
            or np.max(np.abs(records["thumb_open"]["thumb_change_deg"])) > 4
            or records["reopen_max_error_deg"] > 4):
        raise AssertionError("fist or release failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold-only", action="store_true", help="keep ring pinched while moving the other digits")
    parser.add_argument("--extend-held-pinch", action="store_true", help="also add and remove the middle finger while retaining ring contact")
    args = parser.parse_args()
    check(args.hold_only or args.extend_held_pinch, args.extend_held_pinch)
