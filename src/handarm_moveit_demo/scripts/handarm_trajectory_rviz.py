#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RViz trajectory visualizer for D455/AprilTag -> ABB teleoperation.

Displays:
- rainbow LINE_STRIP: mapped hand/palm target trajectory from /target_ee_pose;
- green LINE_STRIP: actual robot tool0 trajectory from TF;
- cyan sphere + text: mapped palm-center start point;
- blue sphere: current hand target;
- green sphere: current actual tool0 position;
- red segment: instantaneous target tracking error.

Important:
Start this node before pressing c in the camera sender.  The first accepted
/target_ee_pose is the robot-space position corresponding to the calibrated
hand palm center and is stored as the trajectory start point.

Reset:
    rosservice call /handarm_trajectory/reset
Then keep the tag visible and press c again.
"""

import colorsys
import math
import threading

import rospy
import tf
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, EmptyResponse
from visualization_msgs.msg import Marker, MarkerArray


def distance3(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def copy_point(p):
    q = Point()
    q.x = float(p.x)
    q.y = float(p.y)
    q.z = float(p.z)
    return q


def rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def rainbow_color(index, count):
    """
    Old points are blue/cyan; newest points gradually become yellow/red.
    Marker.LINE_STRIP accepts one ColorRGBA per point.
    """
    if count <= 1:
        u = 0.0
    else:
        u = float(index) / float(count - 1)
    hue = (0.66 * (1.0 - u))  # blue -> red
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return rgba(r, g, b, 1.0)


class HandArmTrajectoryVisualizer:
    def __init__(self):
        rospy.init_node("handarm_trajectory_visualizer")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_frame = rospy.get_param("~ee_frame", "tool0")
        self.target_topic = rospy.get_param("~target_topic", "/target_ee_pose")
        self.marker_topic = rospy.get_param(
            "~marker_topic", "/handarm_trajectory/markers")

        self.sample_rate = float(rospy.get_param("~sample_rate", 30.0))
        self.min_target_distance = float(
            rospy.get_param("~min_target_distance", 0.002))
        self.min_actual_distance = float(
            rospy.get_param("~min_actual_distance", 0.002))
        self.max_points = int(rospy.get_param("~max_points", 3000))

        self.target_line_width = float(
            rospy.get_param("~target_line_width", 0.008))
        self.actual_line_width = float(
            rospy.get_param("~actual_line_width", 0.006))
        self.start_sphere_scale = float(
            rospy.get_param("~start_sphere_scale", 0.035))
        self.current_sphere_scale = float(
            rospy.get_param("~current_sphere_scale", 0.025))

        self.auto_reset_jump = float(
            rospy.get_param("~auto_reset_jump", 0.0))

        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1, latch=True)

        self.lock = threading.Lock()
        self.armed = False
        self.start_point = None
        self.current_target = None
        self.current_actual = None
        self.target_points = []
        self.actual_points = []

        self.target_sub = rospy.Subscriber(
            self.target_topic,
            PoseStamped,
            self.target_callback,
            queue_size=1,
        )
        self.reset_srv = rospy.Service(
            "~reset", Empty, self.reset_callback)

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.sample_rate, 1.0)),
            self.timer_callback,
        )

        rospy.loginfo("========== Hand/Arm RViz Trajectory Visualizer ==========")
        rospy.loginfo("base_frame: %s", self.base_frame)
        rospy.loginfo("ee_frame: %s", self.ee_frame)
        rospy.loginfo("target_topic: %s", self.target_topic)
        rospy.loginfo("marker_topic: %s", self.marker_topic)
        rospy.loginfo("reset service: %s/reset", rospy.get_name())
        rospy.loginfo(
            "Start this node before pressing c. "
            "The first target becomes the palm-center start point."
        )

    def transform_target_point(self, msg):
        p = PointStamped()
        p.header = msg.header
        p.point = msg.pose.position

        if not p.header.frame_id:
            p.header.frame_id = self.base_frame

        if p.header.frame_id == self.base_frame:
            return copy_point(p.point)

        p.header.stamp = rospy.Time(0)
        transformed = self.listener.transformPoint(self.base_frame, p)
        return copy_point(transformed.point)

    def target_callback(self, msg):
        try:
            p = self.transform_target_point(msg)
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                1.0, "Cannot transform target pose to %s: %s",
                self.base_frame, str(exc))
            return

        with self.lock:
            if (
                self.armed
                and self.auto_reset_jump > 0.0
                and self.current_target is not None
                and distance3(p, self.current_target) > self.auto_reset_jump
            ):
                rospy.logwarn(
                    "Target jump %.1f mm > %.1f mm: resetting path.",
                    1000.0 * distance3(p, self.current_target),
                    1000.0 * self.auto_reset_jump,
                )
                self._clear_locked()

            if not self.armed:
                self.armed = True
                self.start_point = copy_point(p)
                self.current_target = copy_point(p)
                self.target_points = [copy_point(p)]
                self.actual_points = []
                rospy.loginfo(
                    "Palm-center start point set: [%.4f %.4f %.4f] in %s",
                    p.x, p.y, p.z, self.base_frame,
                )
                return

            self.current_target = copy_point(p)
            if (
                not self.target_points
                or distance3(p, self.target_points[-1])
                >= self.min_target_distance
            ):
                self.target_points.append(copy_point(p))
                if len(self.target_points) > self.max_points:
                    self.target_points.pop(0)

    def lookup_actual_point(self):
        trans, _ = self.listener.lookupTransform(
            self.base_frame, self.ee_frame, rospy.Time(0))
        p = Point()
        p.x = float(trans[0])
        p.y = float(trans[1])
        p.z = float(trans[2])
        return p

    def timer_callback(self, _event):
        with self.lock:
            armed = self.armed

        if not armed:
            return

        try:
            actual = self.lookup_actual_point()
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                1.0, "Cannot read TF %s -> %s: %s",
                self.base_frame, self.ee_frame, str(exc))
            return

        with self.lock:
            self.current_actual = copy_point(actual)
            if (
                not self.actual_points
                or distance3(actual, self.actual_points[-1])
                >= self.min_actual_distance
            ):
                self.actual_points.append(copy_point(actual))
                if len(self.actual_points) > self.max_points:
                    self.actual_points.pop(0)

            markers = self.build_markers_locked()

        self.pub.publish(markers)

    def make_base_marker(self, marker_id, marker_type, namespace):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = rospy.Time.now()
        m.ns = namespace
        m.id = int(marker_id)
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = rospy.Duration(0.0)
        m.frame_locked = False
        return m

    def build_markers_locked(self):
        array = MarkerArray()

        # 0: mapped hand target path, rainbow.
        target = self.make_base_marker(
            0, Marker.LINE_STRIP, "hand_target_path")
        target.scale.x = self.target_line_width
        target.points = [copy_point(p) for p in self.target_points]
        target.colors = [
            rainbow_color(i, len(target.points))
            for i in range(len(target.points))
        ]
        array.markers.append(target)

        # 1: actual tool0 path, bright green.
        actual = self.make_base_marker(
            1, Marker.LINE_STRIP, "robot_actual_path")
        actual.scale.x = self.actual_line_width
        actual.color = rgba(0.1, 1.0, 0.1, 1.0)
        actual.points = [copy_point(p) for p in self.actual_points]
        array.markers.append(actual)

        if self.start_point is not None:
            # 2: mapped palm-center start sphere.
            start = self.make_base_marker(
                2, Marker.SPHERE, "palm_center_start")
            start.pose.position = copy_point(self.start_point)
            start.scale.x = self.start_sphere_scale
            start.scale.y = self.start_sphere_scale
            start.scale.z = self.start_sphere_scale
            start.color = rgba(0.0, 1.0, 1.0, 1.0)
            array.markers.append(start)

            # 3: start label.
            label = self.make_base_marker(
                3, Marker.TEXT_VIEW_FACING, "palm_center_start")
            label.pose.position = copy_point(self.start_point)
            label.pose.position.z += 0.055
            label.scale.z = 0.035
            label.color = rgba(0.0, 1.0, 1.0, 1.0)
            label.text = "Hand palm center / zero"
            array.markers.append(label)

        if self.current_target is not None:
            # 4: current target.
            current_target = self.make_base_marker(
                4, Marker.SPHERE, "current_target")
            current_target.pose.position = copy_point(self.current_target)
            current_target.scale.x = self.current_sphere_scale
            current_target.scale.y = self.current_sphere_scale
            current_target.scale.z = self.current_sphere_scale
            current_target.color = rgba(0.1, 0.3, 1.0, 1.0)
            array.markers.append(current_target)

        if self.current_actual is not None:
            # 5: current actual tool0.
            current_actual = self.make_base_marker(
                5, Marker.SPHERE, "current_actual")
            current_actual.pose.position = copy_point(self.current_actual)
            current_actual.scale.x = self.current_sphere_scale
            current_actual.scale.y = self.current_sphere_scale
            current_actual.scale.z = self.current_sphere_scale
            current_actual.color = rgba(0.1, 1.0, 0.1, 1.0)
            array.markers.append(current_actual)

        if self.current_target is not None and self.current_actual is not None:
            # 6: instantaneous tracking error.
            error = self.make_base_marker(
                6, Marker.LINE_LIST, "tracking_error")
            error.scale.x = 0.004
            error.color = rgba(1.0, 0.0, 0.0, 0.9)
            error.points = [
                copy_point(self.current_actual),
                copy_point(self.current_target),
            ]
            array.markers.append(error)

            # 7: numerical position error label.
            error_text = self.make_base_marker(
                7, Marker.TEXT_VIEW_FACING, "tracking_error")
            error_text.pose.position = copy_point(self.current_actual)
            error_text.pose.position.z += 0.045
            error_text.scale.z = 0.025
            error_text.color = rgba(1.0, 0.2, 0.2, 1.0)
            error_text.text = "error={:.1f} mm".format(
                1000.0 * distance3(
                    self.current_target, self.current_actual))
            array.markers.append(error_text)

        return array

    def _clear_locked(self):
        self.armed = False
        self.start_point = None
        self.current_target = None
        self.current_actual = None
        self.target_points = []
        self.actual_points = []

    def reset_callback(self, _request):
        with self.lock:
            self._clear_locked()

        delete = Marker()
        delete.header.frame_id = self.base_frame
        delete.header.stamp = rospy.Time.now()
        delete.action = Marker.DELETEALL
        array = MarkerArray()
        array.markers.append(delete)
        self.pub.publish(array)

        rospy.loginfo(
            "Trajectory cleared. Keep the tag visible and press c again.")
        return EmptyResponse()


if __name__ == "__main__":
    node = HandArmTrajectoryVisualizer()
    rospy.spin()

