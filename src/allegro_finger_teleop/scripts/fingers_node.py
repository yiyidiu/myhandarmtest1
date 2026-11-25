#!/usr/bin/env python3
"""Drive the simulated FR3 palm and the left Allegro from the same calibrated camera frames."""
import json
from pathlib import Path
from queue import Empty, Full, Queue
import time

import numpy as np
import rospy
from dynamic_reconfigure.msg import Config, DoubleParameter
from dynamic_reconfigure.srv import Reconfigure
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState, PointCloud2, PointField
from sensor_msgs.point_cloud2 import create_cloud
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


from after_fr3_ready import pin_gazebo_physics
from arm_reference import SelfClearance
from finger_stream import FingerStream
from object_impedance import ObjectPinch
from retarget import AllegroModel, MIDDLE_WEIGHTS


NS = "/fr3_allegro"
PINCH_PARTNERS = ("index", "middle", "ring")
FIELDS = [PointField(name, offset, kind, 1) for name, offset, kind in (
    ("x", 0, PointField.FLOAT32), ("y", 4, PointField.FLOAT32), ("z", 8, PointField.FLOAT32),
    ("landmark_id", 12, PointField.UINT32), ("capture_sequence", 16, PointField.UINT32))]


def publish_joints(publisher, names, q, dt):
    command = JointTrajectory(joint_names=names)
    # The retargeter has already limited this step to the model speed over dt.
    # Issue it over that same camera interval so consecutive frames do not
    # replace commands which were deliberately stretched to 60 ms.
    command.points = [JointTrajectoryPoint(positions=q.tolist(), time_from_start=rospy.Duration(dt))]
    publisher.publish(command)


def enable_grasp_impedance(names, gains):
    """Switch the effort servos after the measured home check and before enabling input."""
    if set(names) != set(gains):
        raise ValueError("阻抗配置必须覆盖当前 Allegro 的全部关节")
    for joint in names:
        parameters = gains[joint]
        path = NS + "/hand_controller/gains/" + joint + "/set_parameters"
        rospy.wait_for_service(path, timeout=3.)
        request = Config(doubles=[DoubleParameter(key, value) for key, value in parameters.items()])
        reply = rospy.ServiceProxy(path, Reconfigure)(request).config
        actual = {p.name: p.value for p in reply.doubles}
        if any(key not in actual or abs(actual[key] - value) > 1e-8 for key, value in parameters.items()):
            raise RuntimeError("阻抗增益回读不符 " + joint)
    rospy.loginfo("Allegro 16 个关节的抓取阻抗增益已生效")


def main():
    rospy.init_node("allegro_fingers")
    mode = rospy.get_param("~source")
    names = rospy.get_param(NS + "/hand_controller/joints")
    description = rospy.get_param("/robot_description")
    model = AllegroModel(description, names)
    follow_wrist = rospy.get_param("~follow_wrist", True)
    state = [None]
    def observe(msg):
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in names):
            q = np.array([positions[n] for n in names])
            if np.isfinite(q).all():
                state[0] = (msg.header.stamp.to_sec(), q)
    rospy.Subscriber(NS + "/joint_states", JointState, observe, queue_size=1)
    command_topic = ("/full_robot_self_collision_guard/requested_hand_command" if follow_wrist
                     else NS + "/hand_controller/command")
    commands = rospy.Publisher(command_topic, JointTrajectory, queue_size=1)
    cloud = rospy.Publisher(NS + "/human_wrist/landmarks", PointCloud2, queue_size=1)
    gesture = rospy.Publisher(NS + "/hand_gesture", String, queue_size=1)
    packets = Queue(maxsize=1)
    def receive(message):
        try:
            packets.put_nowait(message.data)
        except Full:
            try:
                packets.get_nowait()
            except Empty:
                pass
            packets.put_nowait(message.data)
    def shared_frames():
        while not rospy.is_shutdown():
            try:
                yield json.loads(packets.get(timeout=.2))
            except (Empty, ValueError):
                yield None
    rospy.Subscriber(NS + "/camera_packet", String, receive, queue_size=1)
    frames = shared_frames()
    camera_ready_file = rospy.get_param("~camera_ready_file", "")
    ready_path = Path(camera_ready_file) if mode == "live" and camera_ready_file else None
    pin_after_first_feedback = ready_path is not None
    while not rospy.is_shutdown():
        homing_error = rospy.get_param(NS + "/homing_error", "")
        if homing_error:
            raise RuntimeError("Gazebo 臂手未能回到初始位，C 标定无法开始。控制器报告 " + homing_error)
        observation = state[0]
        if observation is not None and -.02 <= rospy.Time.now().to_sec() - observation[0] <= .4:
            if pin_after_first_feedback:
                # Physics and the early camera now run concurrently even during homing.
                pin_gazebo_physics()
                pin_after_first_feedback = False
            if commands.get_num_connections() and rospy.get_param(NS + "/ready", False):
                break
        time.sleep(.1)
    if rospy.is_shutdown():
        return
    arm_names = rospy.get_param(NS + "/arm_controller/joints")
    housing_guard = SelfClearance(description, arm_names + names)
    object_pinch = None
    if rospy.has_param("~grasp_gains"):
        gains = rospy.get_param("~grasp_gains")
        enable_grasp_impedance(names, gains)
        object_pinch = ObjectPinch(model, [gains[n]["p"] for n in names], rospy.get_param("~pad_preload_n"))
        rospy.Subscriber("/handarm_sim_demo/target_contacts", ContactsState, object_pinch.observe, queue_size=1)
        object_mode = rospy.Publisher(NS + "/object_pinch", String, queue_size=1)
    stream = FingerStream(model, rospy.get_param(NS + "/open_hand"),
                          middle_weights=rospy.get_param("~middle_weights", list(MIDDLE_WEIGHTS)),
                          housing_guard=housing_guard, object_pinch=object_pinch)
    rospy.set_param(NS + "/fingers_ready", True)
    if ready_path is not None:
        ready_path.touch(exist_ok=False)
        rospy.on_shutdown(lambda: ready_path.unlink(missing_ok=True))
    rospy.loginfo("FR3 和 Allegro 已就绪。请在相机提示可以标定后张开左手按 C，保持两秒。")
    accepted = 0
    try:
        for packet in frames:
            if rospy.is_shutdown():
                return
            if packet is None:
                continue
            now = rospy.Time.now().to_sec()
            stamp, measured = state[0]
            if not -.02 <= now-stamp <= .4:
                continue
            result = stream.accept_frame(packet, time.monotonic(), now, measured)
            if result is None or result.hand is None:
                rospy.loginfo_throttle(3., "Allegro 保持：%s", stream.reason)
                continue
            publish_joints(commands, names, result.hand, result.dt)
            if object_pinch is not None:
                object_mode.publish("+".join(PINCH_PARTNERS[i] for i in object_pinch.partners) or "free")
            accepted += 1
            if mode == "replay":
                rospy.set_param(NS + "/replay_accepted_frames", accepted)
            unresolved = stream.shape.edge_on and stream.shape.hidden_partners.any()
            if unresolved:
                rospy.logwarn_throttle(2., "手指根部在画面中重叠。稍微转回手掌，确认捏合手指后可再转腕。")
            gesture.publish("pinch_unresolved" if unresolved and not stream.retargeter.contacts else stream.retargeter.gesture)
            if result.points is not None and cloud.get_num_connections():
                header = Header(frame_id="human_wrist_qb0", stamp=rospy.Time.from_sec(float(packet["stamp"])))
                capture = int(packet["timing"]["capture_sequence"])
                cloud.publish(create_cloud(header, FIELDS, [(*point, i, capture) for i, point in enumerate(result.points)]))
    finally:
        housing_guard.close()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
