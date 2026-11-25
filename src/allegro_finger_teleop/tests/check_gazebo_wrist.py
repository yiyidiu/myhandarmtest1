"""Exercise the calibrated wrist/arm route and independently read the Gazebo palm rigid body."""
import json
import socket
import time

import numpy as np
import rospy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from test_retarget import OPEN, Retargeter, TeleopControlGate, camera_packet, human_hand, load_robot, matrices, Rotation
from test_wrist_hand import BACK_TO_CAMERA, HPOS, HROT, PALM_TO_CAMERA
from wrist_arm import rotation_vector
from perception_hamer.src.teleop_pose_packet import build_invalid_teleop_packet


def main():
    rospy.init_node("check_wrist_palm", anonymous=True)
    if rospy.get_param("/allegro_fingers/source") != "external" or not rospy.get_param("/allegro_fingers/follow_wrist"):
        raise ValueError("requires fingers.launch source:=external follow_wrist:=true")
    deadline = time.monotonic()+50
    while not rospy.get_param("/fr3_allegro/fingers_ready", False):
        if time.monotonic() > deadline:
            raise TimeoutError("simulation not ready")
        time.sleep(.1)
    get_link = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
    hand = load_robot()
    basis = Retargeter(hand, OPEN, human_hand()).basis
    commands, hand_commands, contacts, targets = [], [], [], []
    joint_positions = {}
    rospy.Subscriber("/fr3_allegro/arm_controller/command", JointTrajectory, commands.append, queue_size=20)
    rospy.Subscriber("/fr3_allegro/hand_controller/command", JointTrajectory, hand_commands.append, queue_size=20)
    rospy.Subscriber("/fr3_allegro/palm_target", PoseStamped, targets.append, queue_size=5)
    rospy.Subscriber("/fr3_allegro/joint_states", JointState,
                     lambda msg: joint_positions.update(zip(msg.name, msg.position)), queue_size=1)
    def contact(msg):
        if any("algr_lh_mf_ds_tip" in (s.collision1_name+s.collision2_name) for s in msg.states):
            contacts.append(time.monotonic())
    rospy.Subscriber("/fr3_allegro/thumb_contacts", ContactsState, contact, queue_size=20)

    def palm():
        result = get_link("fr3_allegro::algr_lh_palm", "fr3_allegro::fr3_link0")
        if not result.success:
            raise RuntimeError(result.status_message)
        p, o = result.link_state.pose.position, result.link_state.pose.orientation
        r = matrices(Rotation.from_quat([o.x, o.y, o.z, o.w]))
        return np.array([p.x,p.y,p.z])+r@hand.tcp, r

    session = "wrist-" + str(time.monotonic_ns())
    gate = TeleopControlGate(require_hand_landmarks=True, resume_fixed_hand_after_loss=True)
    seq = 0
    zero = np.zeros(3)
    def send(dp, turn, count, hand_shape=None, valid=True, held=False, camera_rotation=None, presence=3):
        nonlocal seq
        # Each test specifies independent optical translation and robot-anatomical local rotation.
        r = camera_rotation
        if r is None:
            r = HROT @ basis @ matrices(Rotation.from_rotvec(turn)) @ basis.T
        for _ in range(count):
            t = time.monotonic()
            seq += 1
            p = camera_packet(seq,t,human_hand() if hand_shape is None else hand_shape)
            p.update(session_id=session,wrist_position_m=(HPOS+dp).tolist(),palm_rotation_row_major=r.ravel().tolist(),
                     valid=valid, presence_generation=presence)
            gate.observe_identity(presence, p["active_hand_generation"], False)
            if held:
                p.update(orientation_held=True, orientation_channel_valid=False, confidence=[1.]*3+[0.]*3)
                p.pop("hand_landmarks")
            gate.confirm_packet(p,lambda:True,now=t)
            wire.sendto(json.dumps(gate.decorate(p)).encode(),("127.0.0.1",5031))
            time.sleep(.060)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as wire:
        send(zero,zero,6)
        if commands:
            raise AssertionError("arm moved before C")
        # Read world-space heights of two actual, palm-attached joint origins. The outermost
        # Allegro digit is its ring finger. The human depth comparison uses thumb and little.
        def thumb_minus_ring_height():
            zs = []
            for name in ("algr_lh_th_mp", "algr_lh_rf_bs"):
                response = get_link("fr3_allegro::"+name, "fr3_allegro::fr3_link0")
                if not response.success:
                    raise RuntimeError(response.status_message)
                zs.append(response.link_state.pose.position.z)
            return zs[0]-zs[1]

        optical_cases = {}
        thumb_minus_little = human_hand()[1] - human_hand()[17]
        for view, neutral in (("palm", PALM_TO_CAMERA), ("back", BACK_TO_CAMERA)):
            gate.request_confirmation(seq)
            send(zero, zero, 22, camera_rotation=neutral)
            start_p, start_r = palm()
            start_height = thumb_minus_ring_height()
            for yaw in np.deg2rad([-25, 25]):
                observed = matrices(Rotation.from_rotvec([0., yaw, 0.])) @ neutral
                dz_mm = 1000*((observed-neutral) @ thumb_minus_little)[2]
                send(zero, zero, 30, camera_rotation=observed)
                p, rot = palm()
                optical_cases[view+("_near" if dz_mm<0 else "_far")] = dict(
                    human_thumb_relative_to_little_depth_change_mm=dz_mm,
                    palm_normal_base_y=rot[1, 0], palm_normal_base_z=rot[2, 0],
                    palm_normal_y_change=rot[1, 0]-start_r[1, 0],
                    expected_normal_y=np.sign(dz_mm)*np.sin(abs(yaw)),
                    measured_thumb_relative_to_ring_height_change_mm=1000*(thumb_minus_ring_height()-start_height),
                    measured_palm_position_change_mm=1000*np.linalg.norm(p-start_p))
                send(zero, zero, 24, camera_rotation=neutral)
        gate.request_confirmation(seq)
        send(zero,zero,16)
        tcp0, rot0 = palm()
        cases = {}
        def verify(label, dp, rv, n=21, shape=None):
            send(np.array(dp),np.array(rv),n,shape)
            tip, rot = palm()
            dp_base = np.array([-dp[1], -dp[0], dp[2]])
            goal_r = rot0 @ matrices(Rotation.from_rotvec(rv))
            cases[label] = dict(expected_delta_base_mm=(1000*dp_base).tolist(),
               actual_delta_base_mm=(1000*(tip-tcp0)).tolist(),
               actual_rotation_local_deg=np.rad2deg(rotation_vector(rot0.T@rot)).tolist(),
               position_error_mm=float(1000*np.linalg.norm(tip-tcp0-dp_base)),
               rotation_error_deg=float(np.rad2deg(np.linalg.norm(rotation_vector(rot@goal_r.T)))))
        for i,axis in enumerate("xyz"):
            d=np.eye(3)[i]
            for sign,label in ((1,"plus"),(-1,"minus")):
                verify("optical_"+axis+"_"+label,sign*.040*d,zero)
                verify("palm_rotation_"+axis+"_"+label,zero,sign*np.deg2rad(17)*d)
        dp, rv = [.025,-.034,.018], np.deg2rad([9,-13,7])
        verify("combined_middle_pinch",dp,rv,43,human_hand([0,1,0,0],pinch=1))
        combined_contact = sum(stamp>time.monotonic()-.50 for stamp in contacts)
        before, before_rot = palm()
        count = len(commands)
        send([.09,.03,-.05],zero,6,valid=False)
        time.sleep(.40)
        invalid_commands = len(commands)-count
        gate.request_confirmation(seq)
        return_dp = np.array([-.06,.06,.03])
        send(return_dp,zero,19)
        after, after_rot = palm()
        send([-.06,.035,.03],[.4,-.2,.3],28,held=True)
        translated, held_rot = palm()
        # Lower and tilt the palm while bending the middle finger. Reproduce real
        # detector heartbeats without a hand identity, followed by a new presence interval.
        low_dp = return_dp + [-.022,.045,-.044]
        send(low_dp, [.15,.30,-.16], 75, human_hand([0,1,0,0]))
        low, _ = palm()
        goal = targets[-1].pose.position
        low_target = np.array([goal.x,goal.y,goal.z])
        low_pip = joint_positions["algr_lh_mf_pip"]
        start_counts = np.array([len(commands), len(hand_commands)])
        old_epoch = gate.reference_epoch
        for _ in range(20):
            now = time.monotonic()
            seq += 1
            missing = build_invalid_teleop_packet(session,seq,time.time(),"no_real_hand:no_hand_detected",4,4)
            missing.update(capture_monotonic_s=now, timing=dict(capture_sequence=seq+40,
                           inference_executed=False, capture_to_publish_s=.01))
            wire.sendto(json.dumps(gate.decorate(missing)).encode(), ("127.0.0.1",5031))
            time.sleep(.06)
        counts_after_gap = np.array([len(commands), len(hand_commands)])
        paused, _ = palm()
        send(return_dp, zero, 75, presence=5)
        returned, _ = palm()
        goal = targets[-1].pose.position
        return_target = np.array([goal.x,goal.y,goal.z])
        desired_step = (return_dp-low_dp)[[1,0,2]] * [-1,-1,1]
        recovery = dict(commands_during_loss=(counts_after_gap-start_counts).tolist(),
                        commands_after_return=(np.array([len(commands),len(hand_commands)])-counts_after_gap).tolist(),
                        same_C_epoch=gate.enabled and gate.reference_epoch == old_epoch,
                        drift_while_paused_mm=1000*np.linalg.norm(paused-low),
                        before_loss_target_error_mm=1000*np.linalg.norm(low-low_target),
                        after_return_target_error_mm=1000*np.linalg.norm(returned-return_target),
                        distance_from_previously_reached_C_pose_mm=1000*np.linalg.norm(returned-after),
                        requested_base_step_mm=(1000*desired_step).tolist(),
                        published_base_step_mm=(1000*(return_target-low_target)).tolist(),
                        measured_base_step_mm=(1000*(returned-low)).tolist(),
                        return_error_mm=1000*np.linalg.norm(returned-low-desired_step),
                        middle_pip_opened_deg=np.rad2deg(low_pip-joint_positions["algr_lh_mf_pip"]))
        position_max = max(c["position_error_mm"] for c in cases.values())
        rotation_max = max(c["rotation_error_deg"] for c in cases.values())
        report=dict(cases=cases,optical_thumb_side=optical_cases,reacquisition=recovery,
                    max_position_error_mm=position_max,max_rotation_error_deg=rotation_max,
                    combined_middle_contact_messages_last_half_second=combined_contact,
                    commands_while_invalid_or_silent=invalid_commands,
                    held_rotation_delta_base_mm=(1000*(translated-after)).tolist(),
                    held_rotation_translation_error_mm=1000*np.linalg.norm(translated-after-[.025,0,0]),
                    held_rotation_angle_change_deg=np.rad2deg(np.linalg.norm(rotation_vector(held_rot@after_rot.T))),
                    re_C_position_change_mm=1000*np.linalg.norm(after-before),
                    re_C_angle_change_deg=np.rad2deg(np.linalg.norm(rotation_vector(after_rot@before_rot.T))))
        print(json.dumps(report,indent=2))
        if (any(abs(v["palm_normal_base_y"]-v["expected_normal_y"]) > .045
                or v["measured_palm_position_change_mm"] > 5 for v in optical_cases.values())
                or position_max>4 or rotation_max>1 or combined_contact<2 or invalid_commands
                or any(recovery["commands_during_loss"]) or min(recovery["commands_after_return"])<30
                or not recovery["same_C_epoch"] or recovery["return_error_mm"]>5
                or recovery["distance_from_previously_reached_C_pose_mm"]>5
                or recovery["drift_while_paused_mm"]>4 or recovery["middle_pip_opened_deg"]<20
                or report["re_C_position_change_mm"]>3 or report["re_C_angle_change_deg"]>1
                or report["held_rotation_translation_error_mm"]>4 or report["held_rotation_angle_change_deg"]>1):
            raise AssertionError("measured wrist/palm tracking acceptance failed")


if __name__ == "__main__":
    main()
