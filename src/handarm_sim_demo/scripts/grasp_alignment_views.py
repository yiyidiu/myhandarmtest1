#!/usr/bin/env python3
"""Two external Gazebo views of the hand; never publish robot commands."""

import json
import math
import threading
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SetModelState, SpawnModel
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler, quaternion_matrix


def object_home_states(world_file):
    """Read the known grasp props present in the exact world loaded by Gazebo."""
    models = {m.get("name"): m for m in ET.parse(world_file).findall("./world/model")}
    states = []
    for name in ("target_object", "ground_object_left", "ground_object_right"):
        model = models.get(name)
        if model is None:
            continue
        pose = model.find("pose")
        if pose is None or pose.get("relative_to"):
            raise ValueError("object {} requires a world-relative initial pose".format(name))
        xyzrpy = [float(v) for v in pose.text.split()]
        if len(xyzrpy) != 6 or not all(math.isfinite(v) for v in xyzrpy):
            raise ValueError("invalid initial pose for {}".format(name))
        state = ModelState(model_name=name, reference_frame="world")
        state.pose.position.x, state.pose.position.y, state.pose.position.z = xyzrpy[:3]
        q = quaternion_from_euler(*xyzrpy[3:])
        (state.pose.orientation.x, state.pose.orientation.y,
         state.pose.orientation.z, state.pose.orientation.w) = q
        # ModelState's default Twist is zero in all six components.
        states.append(state)
    if not states:
        raise ValueError("world has no resettable grasp objects")
    return states


def restore_grasp_objects(states):
    """Restore free props, without resetting the robot, clock or controllers."""
    rospy.wait_for_service("/gazebo/get_world_properties", timeout=2.0)
    rospy.wait_for_service("/gazebo/set_model_state", timeout=2.0)
    present = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)()
    missing = {s.model_name for s in states} - set(present.model_names)
    if not present.success or missing:
        raise RuntimeError("grasp models unavailable: {}".format(sorted(missing)))
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    for state in states:
        if rospy.is_shutdown():
            raise RuntimeError("scene observer is shutting down")
        result = set_state(state)
        if not result.success:
            raise RuntimeError("{}: {}".format(state.model_name, result.status_message))


class GraspAlignmentViews:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.images = {}
        self.pouring_state = None
        self.pouring_received_at = 0.0
        self.rate_hz = float(rospy.get_param("~rate_hz", 8.0))
        self.subscribers = []
        self.subscribers.append(rospy.Subscriber(
            "/teleop_pouring/state", String, self.on_pouring_state, queue_size=1))
        self.owned_models = []
        self.closed = False
        self.timer = None
        self.reset_busy = False
        self.last_reset_request = float("-inf")
        self.reset_message = ""
        self.reset_message_until = 0.0
        self.home_states = []
        self.reset_events = rospy.Publisher(
            "/teleop_observers/object_reset_events", String, queue_size=10)
        world_file = rospy.get_param("~object_home_world", "")
        if world_file:
            try:
                self.home_states = object_home_states(world_file)
            except (OSError, ET.ParseError, ValueError) as error:
                rospy.logwarn("物体复位不可用，观察窗仍可使用：%s", error)
        rospy.on_shutdown(self.close)
        rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=45.0)
        spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        # These static, collision-free cameras exist only while the observer
        # is running. Closing the views removes the rendering workload too.
        for key in ("front", "side"):
            name = "teleop_{}_camera".format(key)
            initial_view = ("1 0 0.55 0 0.2984989316 3.1415926536" if key == "front"
                            else "0.35 0.65 0.55 0 0.2984989316 -1.5707963268")
            sdf = """<sdf version="1.6"><model name="{name}">
              <static>true</static><pose>{initial_view}</pose>
              <link name="camera_link"><sensor name="{key}_camera" type="camera">
                <always_on>false</always_on><update_rate>{rate}</update_rate><visualize>false</visualize>
                <camera name="{key}"><horizontal_fov>0.9</horizontal_fov>
                  <image><width>640</width><height>480</height><format>R8G8B8</format></image>
                  <clip><near>0.02</near><far>5</far></clip></camera>
                <plugin name="{key}_ros_camera" filename="libgazebo_ros_camera.so">
                  <robotNamespace>/teleop_observers</robotNamespace><cameraName>{key}</cameraName>
                  <imageTopicName>image_raw</imageTopicName><cameraInfoTopicName>camera_info</cameraInfoTopicName>
                  <frameName>teleop_{key}_optical_frame</frameName><updateRate>{rate}</updateRate>
                </plugin></sensor></link></model></sdf>""".format(
                    name=name, key=key, rate=self.rate_hz, initial_view=initial_view)
            initial_pose = Pose()
            initial_pose.orientation.w = 1.0
            result = spawn(name, sdf, "/teleop_observers", initial_pose, "world")
            if not result.success:
                raise RuntimeError(result.status_message)
            self.owned_models.append(name)
        self.zoom = {"front": 1.0, "side": 1.0}
        self.side_sign = 1.0
        self.names = {"front": "正面观察｜T复位物体｜滚轮缩放｜Esc关闭视图",
                      "side": "侧面观察｜T复位物体｜滚轮缩放｜Esc关闭视图"}
        self.buffer = tf2_ros.Buffer(cache_time=rospy.Duration(2.0))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.camera_pose = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=2)
        for index, key in enumerate(("front", "side")):
            name = self.names[key]
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 480, 360)
            cv2.moveWindow(name, 180 + index * 520, 1040)
            cv2.setMouseCallback(name, self.mouse, key)
            self.subscribers.append(rospy.Subscriber(
                "/teleop_observers/{}/image_raw".format(key), Image,
                self.image, callback_args=key, queue_size=1, buff_size=4 * 1024 * 1024))
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.follow_hand)

    def on_pouring_state(self, message):
        try:
            state = json.loads(message.data)
            values = tuple(float(state[key]) for key in ("bottle_ml", "cup_ml", "spilled_ml"))
            if not all(math.isfinite(value) for value in values):
                return
        except (ValueError, TypeError, KeyError):
            return
        with self.lock:
            self.pouring_state = values
            self.pouring_received_at = time.monotonic()

    def image(self, message, key):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        with self.lock:
            self.images[key] = frame

    def mouse(self, event, _x, _y, flags, key):
        if event == cv2.EVENT_MOUSEWHEEL:
            self.zoom[key] = float(np.clip(
                self.zoom[key] * (1.15 if flags > 0 else 1.0 / 1.15), 1.0, 3.0))

    def request_object_reset(self):
        now = time.monotonic()
        with self.lock:
            if self.reset_busy or self.closed or now - self.last_reset_request < 1.0:
                return
            self.last_reset_request = now
            if not self.home_states:
                self.reset_message = "Object reset unavailable in this scene"
                self.reset_message_until = now + 5.0
                return
            self.reset_busy = True
            self.reset_message = "Restoring objects..."
        # A service wait must never block the view/keyboard loop or perception.
        threading.Thread(target=self._restore_objects, daemon=True).start()

    def _restore_objects(self):
        event = dict(event="object_layout_reset", source="alignment_view_key_t",
                     requested_host_wall_ns=time.time_ns(),
                     requested_host_monotonic_ns=time.monotonic_ns(),
                     requested_ros_time_s=rospy.Time.now().to_sec(),
                     models=[s.model_name for s in self.home_states])
        try:
            restore_grasp_objects(self.home_states)
            event.update(success=True)
            message = "Objects restored - continue experiment"
            rospy.loginfo("%d件物体已恢复初始桌面位置和朝向，速度已清零；可继续实验。",
                          len(self.home_states))
        except (rospy.ROSException, rospy.ServiceException, RuntimeError) as error:
            event.update(success=False, error=str(error))
            message = "Object reset failed - see terminal"
            rospy.logerr("物体复位失败：%s", error)
        finally:
            event["finished_host_wall_ns"] = time.time_ns()
            event["finished_ros_time_s"] = rospy.Time.now().to_sec()
            with self.lock:
                self.reset_busy = False
                self.reset_message = message
                self.reset_message_until = time.monotonic() + 5.0
        if not rospy.is_shutdown():
            self.reset_events.publish(String(data=json.dumps(event, ensure_ascii=False)))

    def follow_hand(self, _event):
        try:
            transform = self.buffer.lookup_transform("world", "tool0", rospy.Time(0)).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        q = transform.rotation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        translation = transform.translation
        center = np.array([translation.x, translation.y, translation.z]) + rotation @ np.array(
            [0.0085, 0.0101, 0.0757])
        # View a side object from the outside of its table, avoiding the base
        # between camera and hand. Hysteresis prevents flips at the midline.
        if center[1] > 0.12:
            self.side_sign = 1.0
        elif center[1] < -0.12:
            self.side_sign = -1.0
        pitch = math.atan2(0.20, 0.65)
        for key, offset, yaw in (
                ("front", [0.65, 0, 0.20], math.pi),
                ("side", [0, self.side_sign * 0.65, 0.20], -self.side_sign * math.pi / 2)):
            state = ModelState(model_name="teleop_{}_camera".format(key), reference_frame="world")
            position = center + np.array(offset)
            state.pose.position.x, state.pose.position.y, state.pose.position.z = position
            quat = quaternion_from_euler(0, pitch, yaw)
            (state.pose.orientation.x, state.pose.orientation.y,
             state.pose.orientation.z, state.pose.orientation.w) = quat
            self.camera_pose.publish(state)

    def run(self):
        try:
            while not rospy.is_shutdown():
                # OpenCV 4.2's GTK backend reports WND_PROP_VISIBLE as 0
                # even for visible windows. AUTOSIZE supports destroyed-
                # window detection without immediately closing both views.
                try:
                    if any(cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 0
                           for name in self.names.values()):
                        break
                except cv2.error:
                    break
                with self.lock:
                    frames = dict(self.images)
                    water = (self.pouring_state if time.monotonic() -
                             self.pouring_received_at < 2.0 else None)
                    message = (self.reset_message if self.reset_busy or
                               time.monotonic() < self.reset_message_until else
                               "T: restore objects | Wheel: zoom | Esc: close")
                for key, name in self.names.items():
                    frame = frames.get(key)
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Waiting for Gazebo view...", (35, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)
                    h, w = frame.shape[:2]
                    scale = self.zoom[key]
                    if scale > 1:
                        cw, ch = int(w / scale), int(h / scale)
                        frame = cv2.resize(frame[(h-ch)//2:(h+ch)//2, (w-cw)//2:(w+cw)//2], (w, h))
                    frame = frame.copy()
                    if water is not None:
                        cv2.rectangle(frame, (0, 0), (w, 30), (25, 25, 25), -1)
                        cv2.putText(frame, "Water: bottle %.0f ml | cup %.0f ml | spilled %.0f ml" % water,
                                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.48, (235, 235, 235), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (0, h-30), (w, h), (25, 25, 25), -1)
                    cv2.putText(frame, message, (8, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (240, 240, 240), 1, cv2.LINE_AA)
                    cv2.imshow(name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in (ord("t"), ord("T")):
                    self.request_object_reset()
                time.sleep(1.0 / self.rate_hz)
        finally:
            self.close()
            cv2.destroyAllWindows()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.timer is not None:
            self.timer.shutdown()
        for subscriber in self.subscribers:
            subscriber.unregister()
        for name in self.owned_models:
            try:
                self.delete_model(name)
            except rospy.ServiceException:
                pass
        self.owned_models.clear()


if __name__ == "__main__":
    rospy.init_node("grasp_alignment_views")
    GraspAlignmentViews().run()
