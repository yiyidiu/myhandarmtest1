#!/usr/bin/env python3
"""Measure both controllers and the achieved palm during 'handarm fr3 teleop --replay'."""
import json
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory
from test_retarget import Rotation, matrices
from test_wrist_hand import arm_model
from wrist_arm import CAMERA_TRANSLATION_TO_BASE, rotation_vector


NS = "/fr3_allegro/"


def main():
    rospy.init_node("check_allegro_replay", anonymous=True)
    commands, samples, targets = [], [], []
    rospy.Subscriber(NS+"hand_controller/command", JointTrajectory,
                     lambda m: commands.append(np.array(m.points[-1].positions)), queue_size=50)
    rospy.Subscriber(NS+"joint_states", JointState,
                     lambda m: samples.append((m.header.stamp.to_sec(), dict(zip(m.name, m.position))))
                     if commands else None, queue_size=100)
    rospy.Subscriber(NS+"palm_target", PoseStamped, targets.append, queue_size=30)
    remaining = time.monotonic() + 90
    while time.monotonic() < remaining and not rospy.get_param(NS+"replay_complete", False):
        time.sleep(.1)
    if not rospy.get_param(NS+"replay_complete", False):
        raise TimeoutError("Gazebo replay did not finish")
    time.sleep(1)
    hand = rospy.get_param(NS+"hand_controller/joints")
    arm = rospy.get_param(NS+"arm_controller/joints")
    ready = rospy.get_param(NS+"ready_arm")
    sampled_at = np.array([stamp for stamp, _ in samples])
    hand_q = np.array([[s[n] for n in hand] for _, s in samples])
    arm_q = np.array([[s[n] for n in arm] for _, s in samples])
    finger_ranges = np.rad2deg(np.ptp(hand_q, axis=0).reshape(4, 4).max(axis=1))
    arm_error = np.rad2deg(np.abs(arm_q-ready).max())
    end_error = np.rad2deg(np.abs(hand_q[-1]-commands[-1]).max())
    with np.load(rospy.get_param("/allegro_fingers/recording"), allow_pickle=False) as clip:
        isolated = "gesture_protocol" in clip and str(clip["gesture_protocol"]) == "open_hand_wrist_sweep"
        ci = int(clip["calibration_index"]) if "calibration_index" in clip else 0
        expected = len(clip["stamp"])-ci
        moving_arm = rospy.get_param("/allegro_fingers/follow_wrist") and "wrist_position_m" in clip
        if moving_arm:
            human_translation = (clip["wrist_position_m"][ci:]-clip["wrist_position_m"][ci]) @ CAMERA_TRANSLATION_TO_BASE.T
    model = arm_model()
    endpoint = [model.fk(q)[:2] for q in arm_q]
    travel = 1000*np.ptp(np.array([p for p,_ in endpoint]),axis=0)
    result = dict(control_point=rospy.get_param("/allegro_fingers/control_point"),
                  accepted_frames=rospy.get_param(NS+"replay_accepted_frames"),
                  received_palm_targets=len(targets), measured_palm_xyz_range_mm=travel.tolist(),
                  received_commands=len(commands), measured_joint_state_samples=len(samples),
                  max_joint_excursion_deg_by_finger=dict(zip(("index","middle","ring","thumb"), finger_ranges.tolist())),
                  max_arm_excursion_from_ready_deg=arm_error, final_hand_tracking_error_deg=end_error)
    if targets:
        pose = targets[-1].pose
        p,o = pose.position, pose.orientation
        expected_r = matrices(Rotation.from_quat([o.x,o.y,o.z,o.w]))
        p1,r1 = endpoint[-1]
        result.update(final_palm_to_last_observed_target_mm=1000*np.linalg.norm(p1-[p.x,p.y,p.z]),
                      final_palm_to_last_observed_target_deg=np.rad2deg(np.linalg.norm(rotation_vector(r1@expected_r.T))))
    if moving_arm and len(targets) == expected:
        target_times = np.array([t.header.stamp.to_sec() for t in targets])
        nearest = np.abs(target_times[:, None]-sampled_at).argmin(axis=1)
        palms = np.array([endpoint[i][0] for i in nearest])
        wrists = np.array([endpoint[i][0]-endpoint[i][1] @ model.wrist_to_palm for i in nearest])
        for name, point in (("wrist", wrists), ("palm", palms)):
            error = point - point[0] - human_translation
            result[name+"_to_human_translation_rms_mm"] = 1000*np.sqrt(np.mean(np.sum(error**2, axis=1)))
        result["max_state_to_target_clock_difference_ms"] = 1000*max(abs(target_times-sampled_at[nearest]))
    # The bundled annotated wrist turn shows a held open hand; its desired finger excursion is zero.
    command_ranges = np.rad2deg(np.ptp(np.asarray(commands),axis=0)).max()
    result["max_commanded_finger_excursion_deg"] = command_ranges
    print(json.dumps(result, ensure_ascii=False, indent=2))
    finger_failed = (command_ranges > 3 or max(finger_ranges) > 3) if isolated else np.min(finger_ranges) < 2
    if (result["accepted_frames"] != expected or len(commands) != expected or finger_failed
            or len(targets) != (expected if moving_arm else 0)
            or (moving_arm and (max(travel)<15 or result["final_palm_to_last_observed_target_mm"]>20
                                or result["final_palm_to_last_observed_target_deg"]>10))
            or (not moving_arm and arm_error>1) or end_error > 3):
        raise ValueError("measured replay did not meet acceptance criteria")


if __name__ == "__main__":
    main()
