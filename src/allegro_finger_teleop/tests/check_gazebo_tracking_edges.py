"""Check tilted-wrist translation and a retained loose pinch on an external-input Gazebo instance."""

import json
import socket
import time

import numpy as np
import rospy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from test_fingertip_graph import aim_finger, retained_ring_pose
from test_hand_isolation import image_observation
from test_retarget import HUMAN, TeleopControlGate, Rotation, camera_packet, human_hand, load_robot, matrices
from test_wrist_hand import arm_model
from test_wrist_motion import PALM_TO_CAMERA, POSITION
from perception_hamer.src.finger_observation import finger_shape_signature


def check():
    rospy.init_node("fr3_tracking_edge_check", anonymous=True)
    if (
        rospy.get_param("/allegro_fingers/source") != "external"
        or rospy.get_param("/allegro_fingers/control_point") != "wrist"
    ):
        raise RuntimeError("Use a dedicated external-input Gazebo instance in wrist mode")
    ns = "/fr3_allegro/"
    model, arm = load_robot(), arm_model()
    port = rospy.get_param("/allegro_fingers/port")
    joints, targets, contacts, gestures = [], [], [], []

    def joint_feedback(msg):
        angles = dict(zip(msg.name, msg.position))
        if all(name in angles for name in model.names):
            joints.append((time.monotonic(), np.array([angles[name] for name in model.names])))

    def target_feedback(msg):
        p, q = msg.pose.position, msg.pose.orientation
        rotation = matrices(Rotation.from_quat([q.x, q.y, q.z, q.w]))
        targets.append(np.array([p.x, p.y, p.z]) - rotation @ arm.wrist_to_palm)

    def contact_feedback(msg):
        touching = any(
            "algr_lh_rf_ds_tip_collision_" in (s.collision1_name + " " + s.collision2_name)
            for s in msg.states
        )
        contacts.append((time.monotonic(), touching))

    rospy.Subscriber(ns + "joint_states", JointState, joint_feedback, queue_size=8)
    rospy.Subscriber(ns + "palm_target", PoseStamped, target_feedback, queue_size=12)
    rospy.Subscriber(ns + "thumb_contacts", ContactsState, contact_feedback, queue_size=48)
    rospy.Subscriber(ns + "hand_gesture", String, lambda m: gestures.append(m.data), queue_size=12)
    link = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)

    def wrist_position():
        response = link("fr3_allegro::algr_lh_palm", "fr3_allegro::fr3_link0")
        if not response.success:
            raise RuntimeError(response.status_message)
        point = response.link_state.pose.position
        return np.array([point.x, point.y, point.z])

    deadline = time.monotonic() + 75
    while not joints or not rospy.get_param(ns + "fingers_ready", False):
        if time.monotonic() > deadline:
            raise RuntimeError("Gazebo did not reach its ready posture")
        time.sleep(0.1)
    gate = TeleopControlGate(require_hand_landmarks=True, resume_fixed_hand_after_loss=True)
    gate.request_confirmation(0)
    neutral, session, sequence = human_hand(), str(time.monotonic_ns()), 0
    results = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:

        def send(shape=neutral, angle=0.0, forward=0.0, sparse=False):
            nonlocal sequence
            sequence += 1
            now = time.monotonic()
            rotation = PALM_TO_CAMERA @ matrices(Rotation.from_euler("z", angle, degrees=True))
            position = POSITION + [0.0, -forward, 0.0]
            sites = shape[[0, 5, 9, 13, 17]]
            support = (
                np.vstack([sites, 0.55 * sites[0] + 0.45 * sites[1:]]) @ rotation.T + position
            ).tolist()
            if sparse:
                support[3:] = [None] * 6
            packet = camera_packet(sequence, now, shape)
            packet.update(
                session_id=session,
                wrist_position_m=position.tolist(),
                palm_rotation_row_major=rotation.ravel().tolist(),
                measured_wrist_rotation_row_major=rotation.ravel().tolist(),
                wrist_motion_support=support,
            )
            image = image_observation(shape, rotation, position)
            packet["hand_landmarks"].update({key: value.tolist() for key, value in image.items()})
            packet["hand_landmarks"]["shape_signatures"] = [
                finger_shape_signature(shape).tolist(),
                finger_shape_signature(shape @ rotation.T + position).tolist(),
            ]
            gate.confirm_packet(packet, lambda: True, now=now)
            sender.sendto(json.dumps(gate.decorate(packet), allow_nan=False).encode(), ("127.0.0.1", port))
            time.sleep(0.04)

        for _ in range(32):
            send()
        assert gate.enabled and targets
        for angle in (75, -75):
            for value in np.r_[np.linspace(0, angle, 44), np.full(32, angle)]:
                send(angle=value)
            start = wrist_position()
            for displacement in np.r_[np.linspace(0.0, 0.04, 32), np.full(24, 0.04)]:
                send(angle=angle, forward=displacement, sparse=True)
            moved = 1000 * (wrist_position() - start)
            first = len(targets) - 1
            for _ in range(50):
                send(angle=angle, forward=0.04)
            step = 1000 * np.linalg.norm(np.diff(targets[first:], axis=0), axis=1).max()
            results[str(angle)] = dict(actual_delta_mm=moved.tolist(), recovery_target_step_mm=step)
            assert np.linalg.norm(moved - [40.0, 0.0, 0.0]) < 4.0 and step < 4.0, results
            for t in np.r_[np.linspace(1.0, 0.0, 44), np.zeros(24)]:
                send(angle=angle * t, forward=0.04 * t)

        def shape_at(bend):
            shape = retained_ring_pose(bend, bend)
            direction = shape[16] - shape[4]
            aim_finger(shape, HUMAN[2], shape[16] + 0.006 * direction / np.linalg.norm(direction))
            return shape

        for _ in range(62):
            send(shape=shape_at(0.0))
        start, first = time.monotonic(), len(gestures)
        for bend in np.r_[np.linspace(0.0, 1.0, 28), np.ones(36)]:
            send(shape=shape_at(bend))
        hits = [touching for t, touching in contacts if t >= start]
        excursion = np.degrees(np.ptp([q for t, q in joints if t >= start], axis=0)).reshape(4, 4).max(axis=1)
        results["held_ring"] = dict(
            contacts=sum(hits), samples=len(hits), joint_excursion_deg=excursion.tolist()
        )
        assert hits and np.mean(hits) >= 0.98 and set(gestures[first:]) == {"thumb_ring_tip"}, results
        assert min(excursion[:2]) > 35.0 and max(excursion[2:]) < 4.0, results
        for _ in range(30):
            send()
        assert gestures[-1] == "free"
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    check()
