"""Exercise the actual UDP/ROS/QP/Gazebo path at both signed side-turn limits."""
import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from test_fingertip_graph import folded_fist
from test_retarget import camera_packet, human_hand, matrices, Rotation, TeleopControlGate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rospy.init_node("baseline_gazebo_turn_check", anonymous=True)
    if rospy.get_param("/allegro_fingers/source") != "external":
        raise RuntimeError("an isolated external-input simulation is required")
    listener = tf.TransformListener()
    targets, diagnostics, joint_feedback = [], {}, {}
    rospy.Subscriber("/shared_teleop/pose_target", PoseStamped, lambda m: targets.append(m.pose))
    rospy.Subscriber("/shared_teleop/trend_diagnostics", String, lambda m: diagnostics.update(json.loads(m.data)))
    rospy.Subscriber("/fr3_allegro/joint_states", JointState, lambda m: joint_feedback.update(zip(m.name, m.position)))
    end = time.monotonic()+80
    while not rospy.get_param("/fr3_allegro/fingers_ready",False):
        if time.monotonic()>end:
            raise RuntimeError("FR3 readiness timed out")
        time.sleep(.1)
    listener.waitForTransform("base_link","tool0",rospy.Time(0),rospy.Duration(5))
    source = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    port = rospy.get_param("/allegro_fingers/port")
    gate = TeleopControlGate(retain_reference_until_operator_action=True, confirmation_delay_s=2.0)
    B = np.array(rospy.get_param("/shared_teleop/mapping_profiles/whole_wrist_pose/wrist_ring_to_palm_basis"))
    neutral = np.array([0.,0.,.65])
    hand = human_hand()
    seq = 0
    def send(angle=0., delta=(0.,0.,0.), fingers=True, shape=None, valid=True):
        nonlocal seq
        seq += 1
        now = time.monotonic()
        pkt = camera_packet(seq,now,hand if shape is None else shape)
        if shape is not None:
            pkt["finger_observation"]["flexion"]=[.8]*5
        if not valid:
            pkt.update(valid=False,invalid_reason="TEST_WRIST_OUTSIDE_VIEW",confidence=[0.]*6)
        observed = B @ matrices(Rotation.from_euler("z",angle,degrees=True)) @ B.T
        pkt.update(stamp=time.time(),session_id=session,wrist_position_m=(neutral+delta).tolist(),
                   palm_rotation_row_major=observed.ravel().tolist(),position_source="DIRECT_CAMERA_PALM_SKIN_DEPTH_NO_MODEL_OFFSET")
        pkt.pop("wrist_pose_baseline")
        if not fingers:
            pkt.pop("hand_landmarks")
        gate.confirm_packet(pkt,lambda:True,now=now,source_monotonic_s=now)
        source.sendto(json.dumps(gate.decorate(pkt)).encode(),("127.0.0.1",port))
        time.sleep(.035)
    def actual():
        p,q=listener.lookupTransform("base_link","tool0",rospy.Time(0))
        return np.array(p),matrices(Rotation.from_quat(q))
    def target():
        a=targets[-1]; p,q=a.position,a.orientation
        return np.array([p.x,p.y,p.z]),matrices(Rotation.from_quat([q.x,q.y,q.z,q.w]))
    def side(matrix):
        return float(np.degrees(np.arctan2(-matrix[1,2],matrix[2,2])))
    session = "fr3-baseline-test-"+str(time.monotonic_ns())
    for _ in range(20): send()
    gate.request_confirmation(seq)
    for _ in range(90): send()
    if not targets:
        raise AssertionError("no C-relative ROS targets: "+repr(diagnostics)[:600])
    p0,r0=actual();pt0,rt0=target()
    results={}
    def sample():
        p,r=actual();pt,rt=target()
        return dict(target_side_deg=side(rt0.T@rt),actual_side_deg=side(r0.T@r),
                    actual_travel_mm=((p-p0)*1000).round(3).tolist(),target_travel_mm=((pt-pt0)*1000).round(3).tolist(),
                    actual_joints_deg=[round(float(np.degrees(joint_feedback['fr3_joint'+str(i)])),2) for i in range(1,8)])
    stationary = []
    for finger_pose in (folded_fist(),hand):
        for _ in range(26):
            send(shape=finger_pose)
            pp,rr=actual()
            stationary.append((np.linalg.norm(pp-p0)*1000.,np.degrees(np.arccos(np.clip((np.trace(r0.T@rr)-1.)/2.,-1.,1.)))))
    results["wrist_fixed_fingers_only_max_mm_deg"] = np.max(stationary,axis=0).tolist()
    assert results["wrist_fixed_fingers_only_max_mm_deg"][0]<2.0 and results["wrist_fixed_fingers_only_max_mm_deg"][1]<.6
    for sign in (1,-1):
        for a in np.linspace(0,40,31): send(sign*a)
        for _ in range(38): send(sign*40)
        results[str(sign)+"_40"]=sample()
        for a in np.linspace(40,102,65): send(sign*a)
        for _ in range(47): send(sign*102)
        results[str(sign)+"_102"]=sample()
        # Keep the wrist turned and move along all three optical axes with no hand-shape packet.
        d=np.array([.019,-.026,.020])
        for ratio in np.linspace(0,1,29): send(sign*102,d*ratio,False)
        for _ in range(50): send(sign*102,d,False)
        results[str(sign)+"_translate"]=sample()
        print(sign, results[str(sign)+"_translate"], flush=True)
        if sign==1:
            before,_=actual()
            for _ in range(25): send(sign*102,d,False,valid=False)
            after,_=actual()
            results["lost_wrist_hold_mm"]=1000.*float(np.linalg.norm(after-before))
            assert results["lost_wrist_hold_mm"]<2.
            # Reacquire at the old C, keeping the hand at the same turned pose.
            for _ in range(27): send(sign*102,d,False)
            assert gate.enabled and seq>0
        for ratio in np.linspace(1,0,65): send(sign*102*ratio,d*ratio)
        for _ in range(50): send()
        results[str(sign)+"_return"]=sample()
    for sign in (1, -1):
        for angle in np.linspace(0., 102., 85):
            send(sign * angle)
        # Start the translation on the very next input packet. Keep the two-second
        # settling segment only at the end to read the final measured position.
        for ratio in np.linspace(.025, 1., 40):
            send(sign * 102., d * ratio, False)
        for _ in range(55):
            send(sign * 102., d, False)
        measured = sample()
        results[str(sign) + "_immediate_translate"] = measured
        np.testing.assert_allclose(measured["actual_travel_mm"], [41.6, -38, 24], atol=5.)
        assert abs(measured["actual_side_deg"] - sign * 88) < 4.
        for ratio in np.linspace(1., 0., 65):
            send(sign * 102. * ratio, d * ratio)
        for _ in range(50):
            send()
    args.output.write_text(json.dumps(results,indent=2))
    for s in ("1","-1"):
        sign=int(s)
        for key,expected in (("_40",52),("_102",88)):
            a=results[s+key]
            assert abs(a["target_side_deg"]-sign*expected)<.6, a
            assert abs(a["actual_side_deg"]-sign*expected)<4.0, a
        a=results[s+"_translate"]
        np.testing.assert_allclose(a["target_travel_mm"],[41.6,-38,24],atol=.9)
        np.testing.assert_allclose(a["actual_travel_mm"],[41.6,-38,24],atol=5.)
        assert abs(a["actual_side_deg"]-sign*88)<4
        np.testing.assert_allclose(results[s+"_return"]["actual_travel_mm"],0,atol=5)

if __name__ == "__main__":
    main()
