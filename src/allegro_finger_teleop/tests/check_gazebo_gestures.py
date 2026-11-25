#!/usr/bin/env python3
"""Inject synthetic human landmarks through the loopback transport of a dedicated external-input simulation."""
import json
import socket
import time

import numpy as np
import rospy
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs.point_cloud2 import read_points
from trajectory_msgs.msg import JointTrajectory

from test_retarget import OPEN, TeleopControlGate, camera_packet, human_hand, load_robot, matrices, Rotation
from test_wrist_hand import long_hand, thumb_sweep
from retarget import DIGITS


def main():
    rospy.init_node("allegro_gesture_check", anonymous=True)
    if rospy.get_param("/allegro_fingers/source") != "external":
        raise ValueError("requires a dedicated fingers.launch source:=external instance")
    model = load_robot()
    names = rospy.get_param("/fr3_allegro/hand_controller/joints")
    arm = rospy.get_param("/fr3_allegro/arm_controller/joints")
    ready = rospy.get_param("/fr3_allegro/ready_arm")
    states, commands, clouds, contacts = [], [], [], []
    rospy.Subscriber("/fr3_allegro/joint_states", JointState,
                     lambda m: states.append(dict(zip(m.name, m.position))), queue_size=10)
    rospy.Subscriber("/fr3_allegro/hand_controller/command", JointTrajectory,
                     lambda m: commands.append(m.points[-1].positions), queue_size=20)
    rospy.Subscriber("/fr3_allegro/human_wrist/landmarks", PointCloud2, clouds.append, queue_size=10)
    def record_contacts(msg):
        if msg.states:
            contacts.append((time.monotonic(), [(s.collision1_name, s.collision2_name) for s in msg.states]))
    rospy.Subscriber("/fr3_allegro/thumb_contacts", ContactsState, record_contacts, queue_size=20)
    get_link = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
    seq = 0
    session = "gesture-" + str(time.monotonic_ns())
    gate = TeleopControlGate(require_hand_landmarks=True)
    cutoff = time.monotonic()+50
    while not (rospy.get_param("/fr3_allegro/fingers_ready", False) and states):
        if time.monotonic() > cutoff:
            raise TimeoutError("simulation did not initialize")
        time.sleep(.05)

    def measured():
        return np.array([states[-1][n] for n in names])

    def gazebo_pads():
        # Independent of the solver's joint FK: use Gazebo's distal-link poses relative to the palm.
        pads = []
        for digit, offset in zip(DIGITS, model.pad_offsets):
            result = get_link("fr3_allegro::algr_lh_"+digit+"_ds", "fr3_allegro::algr_lh_palm")
            if not result.success:
                raise ValueError(result.status_message)
            pose = result.link_state.pose
            o, p = pose.orientation, pose.position
            pads.append(matrices(Rotation.from_quat([o.x,o.y,o.z,o.w])) @ offset + [p.x,p.y,p.z])
        return np.array(pads)

    def send(hand, count, valid=True):
        nonlocal seq
        for _ in range(count):
            t = time.monotonic()
            seq += 1
            p = camera_packet(seq, t, hand)
            p["session_id"] = session
            p["valid"] = valid
            gate.confirm_packet(p, lambda: True, now=t)
            wire.sendto(json.dumps(gate.decorate(p)).encode(), ("127.0.0.1", 5031))
            time.sleep(.065)

    opened = human_hand()
    report = dict(single_finger_max_excursion_deg={}, passive_finger_max_excursion_deg={}, pinch_pad_centres_mm={},
                  sustained_pad_contact_messages_last_half_second={}, contacted_finger_links={},
                  deep_curl_actual_deg={}, thumb_sweep_joint_delta_deg={})
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as wire:
        send(opened, 4)
        if commands:
            raise AssertionError("received commands before C")
        gate.request_confirmation(seq)
        send(opened, 14)
        if not clouds:
            raise AssertionError("no stamped landmarks after C")
        last = list(read_points(clouds[-1], ("x", "y", "z", "landmark_id", "capture_sequence")))
        np.testing.assert_allclose(np.array(last)[:, :3], opened, atol=1e-6)
        np.testing.assert_equal(np.array(last)[:, 3], np.arange(21))
        if set(np.array(last)[:, 4]) != {seq+40}:
            raise AssertionError("capture sequence was not preserved")
        for i, label in enumerate(("index", "middle", "ring", "thumb")):
            open_q = measured()
            send(human_hand(np.eye(4)[i]), 22)
            excursions = np.rad2deg(np.max(np.abs((measured()-open_q).reshape(4,4)), axis=1))
            report["single_finger_max_excursion_deg"][label] = excursions[i]
            report["passive_finger_max_excursion_deg"][label] = np.delete(excursions,i).max()
            send(opened, 22)
        for i, label in enumerate(("index","middle","ring")):
            initial = measured()
            send(long_hand(i,[1.40,1.53,1.36]), 30)
            report["deep_curl_actual_deg"][label] = np.rad2deg((measured()-initial).reshape(4,4)[i,1:]).tolist()
            send(opened, 26)
        for sweep in (.4,-.18):
            initial = measured()
            send(thumb_sweep(sweep), 24)
            report["thumb_sweep_joint_delta_deg"][str(sweep)] = np.rad2deg((measured()-initial)[12:]).tolist()
            send(opened, 22)
        poses = [(i, label, np.eye(4)[i]) for i,label in enumerate(("index","middle","ring"))]
        poses += [(1,"middle_partly_bent",[0,.63,0,0]), (1,"middle_among_close_digits",[.8,.78,.62,.3])]
        for i, label, bend in poses:
            send(human_hand(bend, pinch=i), 40)
            pads = gazebo_pads()
            report["pinch_pad_centres_mm"][label] = 1000*np.linalg.norm(pads[i]-pads[3])
            cutoff = time.monotonic()-.5
            suffix = "algr_lh_"+DIGITS[i]+"_ds_tip_collision_"
            recent = [pairs for stamp, pairs in contacts if stamp >= cutoff]
            report["sustained_pad_contact_messages_last_half_second"][label] = sum(
                any(suffix in a or suffix in b for a,b in pairs) for pairs in recent)
            report["contacted_finger_links"][label] = sorted({c.split("::")[-2] for pairs in recent
               for pair in pairs for c in pair if "_th_" not in c})
            send(opened, 26)
        send(human_hand([1,1,1,1]), 27)
        count = len(commands)
        last_q = measured()
        send(opened, 7, valid=False)
        time.sleep(.45)
        report["commands_while_invalid_or_silent"] = len(commands)-count
        report["hold_measured_drift_deg"] = np.rad2deg(np.max(np.abs(measured()-last_q)))
        send(opened, 28)
    report["max_arm_error_deg"] = np.rad2deg(np.max(np.abs(np.array([[s[n] for n in arm] for s in states])-ready)))
    report["deep_curl_human_target_deg"] = np.rad2deg([1.40,1.53,1.36]).tolist()
    report["deep_curl_max_absolute_error_deg"] = max(np.max(np.abs(np.asarray(v)-report["deep_curl_human_target_deg"]))
                                                   for v in report["deep_curl_actual_deg"].values())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (min(report["single_finger_max_excursion_deg"].values()) < 10
            or max(report["passive_finger_max_excursion_deg"].values()) > 3
            or min(report["sustained_pad_contact_messages_last_half_second"].values()) < 2
            or report["deep_curl_max_absolute_error_deg"] > 6
            or max(abs(x) for a in report["thumb_sweep_joint_delta_deg"].values() for x in a[1:]) > 3
            or report["commands_while_invalid_or_silent"] or report["hold_measured_drift_deg"] > 2
            or report["max_arm_error_deg"] > 1):
        raise AssertionError("measured gesture acceptance failed")


if __name__ == "__main__":
    main()
