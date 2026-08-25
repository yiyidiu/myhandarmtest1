#!/usr/bin/env python3
"""RViz MarkerArray output for three-finger grasp-pose development."""

import math

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def rotation_to_quaternion(rotation):
    rotation = np.asarray(rotation, dtype=float)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            ) * 2.0
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            ) * 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            ) * 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("rotation produced invalid quaternion")
    return Quaternion(x=x / norm, y=y / norm, z=z / norm, w=w / norm)


def matrix_pose(matrix):
    matrix = np.asarray(matrix, dtype=float)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = matrix[:3, 3]
    pose.orientation = rotation_to_quaternion(matrix[:3, :3])
    return pose


def _point(value):
    value = np.asarray(value, dtype=float)
    return Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def _color(red, green, blue, alpha=1.0):
    return ColorRGBA(r=red, g=green, b=blue, a=alpha)


class GraspPoseVisualizer:
    def __init__(self, topic="/handarm_sim_demo/three_finger_grasp_markers"):
        self.publisher = rospy.Publisher(topic, MarkerArray, queue_size=1, latch=True)
        self.frame_id = "world"
        self._markers = []
        self._next_id = 0

    def _new(self, namespace, marker_type):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = self._next_id
        self._next_id += 1
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0.0)
        self._markers.append(marker)
        return marker

    def reset(self):
        delete = Marker()
        delete.action = Marker.DELETEALL
        self.publisher.publish(MarkerArray(markers=[delete]))
        self._markers = []
        self._next_id = 0

    def add_obb(self, T_world_object, size_m):
        marker = self._new("object_obb", Marker.CUBE)
        marker.pose = matrix_pose(T_world_object)
        marker.scale = Vector3(*[float(value) for value in size_m])
        marker.color = _color(0.1, 0.8, 0.1, 0.28)

    def add_frame(self, namespace, matrix, length=0.06, width=0.006):
        matrix = np.asarray(matrix, dtype=float)
        origin = matrix[:3, 3]
        marker = self._new(namespace, Marker.LINE_LIST)
        marker.scale.x = float(width)
        colors = (
            _color(1.0, 0.1, 0.1),
            _color(0.1, 1.0, 0.1),
            _color(0.1, 0.3, 1.0),
        )
        for axis in range(3):
            marker.points.extend(
                [_point(origin), _point(origin + float(length) * matrix[:3, axis])]
            )
            marker.colors.extend([colors[axis], colors[axis]])

    def add_candidates(self, candidates, pregrasp_distance_m, max_markers=400):
        # Show the best representative of every family/direction/roll/validity
        # group so offset variants do not flood RViz with identical arrows.
        representatives = {}
        for candidate in candidates:
            key = (
                candidate.family,
                candidate.direction,
                round(candidate.roll_deg, 6),
                candidate.enclosure.valid,
            )
            current = representatives.get(key)
            if current is None or candidate.enclosure.table_clearance_m > current.enclosure.table_clearance_m:
                representatives[key] = candidate
        for candidate in list(representatives.values())[: int(max_markers)]:
            approach = candidate.T_world_hand[:3, 2]
            grasp = candidate.T_world_grasp_center[:3, 3]
            pregrasp = grasp - float(pregrasp_distance_m) * approach
            marker = self._new("accepted_candidates" if candidate.enclosure.valid else "rejected_candidates", Marker.ARROW)
            marker.points = [_point(pregrasp), _point(grasp)]
            marker.scale = Vector3(0.004, 0.009, 0.012)
            marker.color = (
                _color(0.1, 0.9, 0.2, 0.75)
                if candidate.enclosure.valid
                else _color(0.9, 0.15, 0.1, 0.20)
            )

    def add_selected(self, candidate, pregrasp_distance_m, joint_6_rad=None):
        self.add_frame("selected_grasp_center", candidate.T_world_grasp_center, 0.09, 0.009)
        self.add_frame("selected_tool0", candidate.T_world_tool0, 0.065, 0.006)
        approach = candidate.T_world_hand[:3, 2]
        grasp = candidate.T_world_grasp_center[:3, 3]
        pregrasp = grasp - float(pregrasp_distance_m) * approach
        arrow = self._new("selected_approach", Marker.ARROW)
        arrow.points = [_point(pregrasp), _point(grasp)]
        arrow.scale = Vector3(0.009, 0.018, 0.024)
        arrow.color = _color(0.0, 0.9, 1.0, 1.0)
        for family, contact in sorted(candidate.enclosure.contacts.items()):
            point_world = (
                candidate.T_world_hand[:3, :3] @ contact.point_hand_m
                + candidate.T_world_hand[:3, 3]
            )
            sphere = self._new("predicted_contact_{}".format(family), Marker.SPHERE)
            sphere.pose.position = _point(point_world)
            sphere.scale = Vector3(0.018, 0.018, 0.018)
            sphere.color = _color(1.0, 0.85, 0.0, 1.0)
            normal_world = candidate.T_world_hand[:3, :3] @ contact.normal_hand
            normal = self._new("predicted_normal_{}".format(family), Marker.ARROW)
            normal.points = [_point(point_world), _point(point_world + 0.035 * normal_world)]
            normal.scale = Vector3(0.004, 0.008, 0.010)
            normal.color = _color(1.0, 0.55, 0.0, 1.0)
        text = self._new("selected_summary", Marker.TEXT_VIEW_FACING)
        text.pose.position = _point(grasp + np.array([0.0, 0.0, 0.13]))
        text.scale.z = 0.028
        text.color = _color(1.0, 1.0, 1.0, 1.0)
        joint_text = "NOT_EVALUATED" if joint_6_rad is None else "{:.3f} rad".format(joint_6_rad)
        text.text = (
            "{} {}  tilt={:.1f} roll={:.1f} deg\njoint_6={}  table={:.1f} mm\nf1/f2/f3 predicted"
        ).format(
            candidate.family,
            candidate.direction,
            candidate.tilt_deg,
            candidate.roll_deg,
            joint_text,
            1000.0 * candidate.enclosure.table_clearance_m,
        )

    def add_pad_sweeps(self, geometry, candidate):
        colors = {
            "f1": _color(1.0, 0.25, 0.8, 1.0),
            "f2": _color(0.2, 0.9, 1.0, 1.0),
            "f3": _color(1.0, 0.65, 0.1, 1.0),
        }
        for family in ("f1", "f2", "f3"):
            link = geometry.distal_pad_links[family]
            line = self._new("{}_pad_sweep".format(family), Marker.LINE_STRIP)
            line.scale.x = 0.004
            line.color = colors[family]
            for fraction, _, _, _ in geometry._closure_sweep:
                capsules, _ = geometry._collision_proxies(
                    (link,), geometry.joint_values_at(fraction), family
                )
                cylinder = [
                    item
                    for item in capsules
                    if np.linalg.norm(item.second - item.first) > 1.0e-12
                ][0]
                center_hand = 0.5 * (cylinder.first + cylinder.second)
                center_world = (
                    candidate.T_world_hand[:3, :3] @ center_hand
                    + candidate.T_world_hand[:3, 3]
                )
                line.points.append(_point(center_world))

    def publish(self):
        self.publisher.publish(MarkerArray(markers=self._markers))
