"""Measure retention of a presented 13 mm object under gravity while index and middle fingers close."""

import argparse, json, time, socket
from pathlib import Path
import numpy as np
import rospy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetLinkState
from test_retarget import TeleopControlGate, camera_packet, human_hand, load_robot, matrices, Rotation, HUMAN
from test_fingertip_graph import retained_ring_pose, aim_finger
from test_wrist_motion import PALM_TO_CAMERA, POSITION
from test_hand_isolation import image_observation
from perception_hamer.src.finger_observation import finger_shape_signature
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
    GetModelState,
    SetModelState,
    GetLinkProperties,
    SetLinkProperties,
)
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("at least one trial per shape is required")
    NS = "/fr3_allegro/"
    rospy.init_node("fr3_edge_audit", anonymous=True)
    assert (
        rospy.get_param("/allegro_fingers/source") == "external"
        and rospy.get_param("/allegro_fingers/control_point") == "wrist"
    )
    hm = load_robot()
    port = rospy.get_param("/allegro_fingers/port")

    def coord(pose):
        p, o = pose.position, pose.orientation
        return np.array([p.x, p.y, p.z]), matrices(Rotation.from_quat([o.x, o.y, o.z, o.w]))

    link = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
    deadline = time.monotonic() + 75.0
    while not rospy.get_param(NS + "fingers_ready", False):
        if time.monotonic() > deadline:
            raise RuntimeError("Gazebo did not reach its ready posture")
        time.sleep(0.1)
    gate = TeleopControlGate(require_hand_landmarks=True, resume_fixed_hand_after_loss=True)
    gate.request_confirmation(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    neutral = human_hand()
    sess = "audit-" + str(time.monotonic_ns())

    def send(points=neutral):
        nonlocal seq
        now = time.monotonic()
        seq += 1
        rt, pos = PALM_TO_CAMERA, POSITION
        obs = {k: v.tolist() for k, v in image_observation(points, rt, pos).items()}
        obs["shape_signatures"] = [
            finger_shape_signature(points).tolist(),
            finger_shape_signature(points @ rt.T + pos).tolist(),
        ]
        sites = points[[0, 5, 9, 13, 17]]
        support = (np.vstack([sites, 0.55 * sites[0] + 0.45 * sites[1:]]) @ rt.T + pos).tolist()
        pkt = camera_packet(seq, now, points)
        pkt.update(
            session_id=sess,
            wrist_position_m=pos.tolist(),
            palm_rotation_row_major=rt.ravel().tolist(),
            measured_wrist_rotation_row_major=rt.ravel().tolist(),
            wrist_motion_support=support,
        )
        pkt["hand_landmarks"].update(obs)
        gate.confirm_packet(pkt, lambda: True, now=now)
        sender.sendto(json.dumps(gate.decorate(pkt), allow_nan=False).encode(), ("127.0.0.1", port))
        time.sleep(0.04)

    key = "pinch_probe_" + uuid.uuid4().hex[:8]
    spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    delete = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    get_object = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    get_props = rospy.ServiceProxy("/gazebo/get_link_properties", GetLinkProperties)
    set_props = rospy.ServiceProxy("/gazebo/set_link_properties", SetLinkProperties)

    def gravity(enabled):
        name = key + "::link"
        current = get_props(name)
        assert current.success, current.status_message
        reply = set_props(
            name,
            current.com,
            enabled,
            current.mass,
            current.ixx,
            current.ixy,
            current.ixz,
            current.iyy,
            current.iyz,
            current.izz,
        )
        assert reply.success, reply.status_message

    contact_log = []

    def object_contacts(msg):
        contact_log.append(
            (time.monotonic(), any(key in s.collision1_name + " " + s.collision2_name for s in msg.states))
        )

    rospy.Subscriber(NS + "thumb_contacts", ContactsState, object_contacts, queue_size=50)
    both_pads = []

    def pairs_on_object(msg):
        names = " ".join(s.collision1_name + " " + s.collision2_name for s in msg.states)
        both_pads.append(
            (
                time.monotonic(),
                all("algr_lh_" + finger + "_ds_tip_collision_" in names for finger in ("th", "rf")),
            )
        )

    rospy.Subscriber("/" + key + "/contacts", ContactsState, pairs_on_object, queue_size=50)

    def create(shape, xyz):
        geom = (
            "<sphere><radius>.0065</radius></sphere>"
            if shape == "sphere"
            else "<box><size>.013 .013 .013</size></box>"
        )
        xml = f"""<sdf version="1.6"><model name="{key}"><link name="link"><gravity>false</gravity>
        <inertial><mass>.006</mass><inertia><ixx>0.00000017</ixx><iyy>0.00000017</iyy><izz>0.00000017</izz></inertia></inertial>
        <collision name="surface"><geometry>{geom}</geometry><surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface></collision>
        <sensor name="pad_contacts" type="contact"><always_on>true</always_on><update_rate>50</update_rate>
        <contact><collision>surface</collision></contact><plugin name="pad_contact_ros" filename="libgazebo_ros_bumper.so">
        <bumperTopicName>/{key}/contacts</bumperTopicName><frameName>world</frameName></plugin></sensor>
        <visual name="skin"><geometry>{geom}</geometry></visual></link></model></sdf>"""
        response = spawn(
            key, xml, "", Pose(position=Point(*xyz), orientation=Quaternion(0, 0, 0, 1)), "world"
        )
        assert response.success, response.status_message

    def position():
        response = get_object(key, "world")
        assert response.success, response.status_message
        v = response.pose.position
        return np.array([v.x, v.y, v.z])

    def open_hand(n):
        for _ in range(n):
            send()

    def pinch_shape(t):
        h = retained_ring_pose(t, t)
        delta = h[16] - h[4]
        aim_finger(h, HUMAN[2], h[16] + 0.006 * delta / np.linalg.norm(delta))
        return h

    def pinch(t, n):
        h = pinch_shape(t)
        for _ in range(n):
            send(points=h)

    def mid():
        pads = []
        for i, name in [(2, "rf"), (3, "th")]:
            result = link("fr3_allegro::algr_lh_" + name + "_ds", "world")
            assert result.success
            p, r = coord(result.link_state.pose)
            pads.append(p + r @ hm.pad_offsets[i])
        return np.mean(pads, axis=0)

    def fraction(t, history):
        data = [hit for stamp, hit in history if stamp >= t]
        return [int(sum(data)), len(data)]

    out = []
    try:
        open_hand(36)
        for trial in range(args.trials):
            for shape in ["sphere", "cube"]:
                open_hand(38)
                pinch(0.0, 65)
                xyz = mid()
                open_hand(38)
                create(shape, xyz)
                fixture_state = ModelState(
                    model_name=key,
                    pose=Pose(position=Point(*xyz), orientation=Quaternion(0, 0, 0, 1)),
                    reference_frame="world",
                )
                shape_held = pinch_shape(0.0)
                for _ in range(70):
                    placed = set_state(fixture_state)
                    assert placed.success, placed.status_message
                    send(points=shape_held)
                fixture_start = time.monotonic() - 0.6
                fixture = fraction(fixture_start, contact_log)
                presented_pair = fraction(fixture_start, both_pads)
                gravity(True)
                start = time.monotonic()
                sim_start = rospy.get_time()
                ev = []
                for t in np.r_[np.zeros(8), np.linspace(0, 1, 28), np.ones(28)]:
                    send(points=pinch_shape(t))
                    ev.append(position())
                final = ev[-1]
                stats = fraction(start, contact_log)
                held_pair = fraction(start, both_pads)
                seconds = rospy.get_time() - sim_start
                open_hand(45)
                released = position()
                row = dict(
                    shape=shape,
                    trial=trial,
                    initial_xyz_m=xyz.tolist(),
                    fixture_thumb_contacts=fixture,
                    holding_thumb_contacts=stats,
                    fixture_simultaneous_pads=presented_pair,
                    holding_simultaneous_pads=held_pair,
                    observed_sim_seconds=seconds,
                    max_deviation_mm=float(1000 * np.linalg.norm(np.asarray(ev) - xyz, axis=1).max()),
                    end_deviation_mm=float(1000 * np.linalg.norm(final - xyz)),
                    drop_mm=float(1000 * (xyz[2] - final[2])),
                    release_drop_mm=float(1000 * (final[2] - released[2])),
                )
                row["maintained"] = (
                    row["end_deviation_mm"] < 25 and row["release_drop_mm"] > 50 and stats[0] > 0
                )
                out.append(row)
                if args.output:
                    args.output.write_text(json.dumps(out, indent=2))
                rospy.loginfo("%s trial %d, displacement %.1f mm", shape, trial + 1, row["end_deviation_mm"])
                assert delete(key).success
    finally:
        if get_object(key, "world").success:
            delete(key)
        sender.close()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
