#!/usr/bin/env python3
"""Continuous, fail-closed multi-waypoint static-obstacle course."""

import copy
import csv
import datetime
import json
import math
import os
import sys
import threading
import time

import actionlib
import moveit_commander
import rospkg
import rospy
import tf2_ros
import yaml
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import MoveGroupAction
from moveit_msgs.srv import (
    GetPositionFK,
    GetPositionFKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


EXPECTED_CONTROLLERS = {"controller_gazebo", "controller_gazebo_hand"}
CSV_FIELDS = [
    "run_id",
    "run_label",
    "course_id",
    "repetition",
    "segment_id",
    "waypoint_id",
    "start_pose",
    "goal_pose",
    "direct_path_collision",
    "direct_path_unsafe",
    "direct_collision_aware_fraction",
    "direct_ignore_collision_fraction",
    "direct_collision_aware_invalid_states",
    "direct_ignore_collision_invalid_states",
    "planning_success",
    "planning_time_s",
    "planning_wall_time_s",
    "trajectory_points",
    "joint_path_length",
    "end_effector_path_length_m",
    "execution_success",
    "execution_time_s",
    "settle_time_s",
    "position_error_m",
    "orientation_error_deg",
    "collision_validation_pass",
    "failure_reason",
]


def quaternion_angle_deg(a, b):
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def make_pose(spec):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = spec["position"]
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = spec["orientation_xyzw"]
    return pose


def pose_dict(pose):
    return {
        "position": [pose.position.x, pose.position.y, pose.position.z],
        "orientation_xyzw": [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    }


def point_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def nearest_equivalent_within_bounds(target, current, minimum, maximum):
    """Choose the nearest 2*pi-equivalent target that remains inside limits."""
    turns_min = int(math.ceil((minimum - target) / (2.0 * math.pi)))
    turns_max = int(math.floor((maximum - target) / (2.0 * math.pi)))
    candidates = [
        target + turns * 2.0 * math.pi
        for turns in range(turns_min, turns_max + 1)
    ]
    if not candidates:
        raise ValueError("joint target has no equivalent inside configured bounds")
    return min(candidates, key=lambda value: abs(value - current))


def validate_course_config(config, scene):
    waypoints = config.get("waypoints", [])
    if config.get("frame_id") != "world" or len(waypoints) < 5:
        raise ValueError("course requires world frame and at least five waypoints")
    ids = [item.get("id") for item in waypoints]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("course waypoint ids must be unique and non-empty")
    object_by_name = {item["name"]: item for item in scene.get("objects", {}).values()}
    orientation_representatives = []
    heights = []
    roles = set()
    for item in waypoints:
        position = item.get("position", [])
        quaternion = item.get("orientation_xyzw", [])
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("invalid position for {}".format(item["id"]))
        if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
            raise ValueError("invalid quaternion for {}".format(item["id"]))
        norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(norm - 1.0) > 1.0e-5:
            raise ValueError("non-unit quaternion for {}".format(item["id"]))
        pose = make_pose(item)
        if not orientation_representatives or all(
            quaternion_angle_deg(pose.orientation, old) >= 5.0
            for old in orientation_representatives
        ):
            orientation_representatives.append(pose.orientation)
        if not any(abs(position[2] - value) < 0.03 for value in heights):
            heights.append(position[2])
        roles.add(str(item.get("role", "")))
        for rule in item.get("clearance_constraints", []):
            name = rule.get("object")
            if name not in object_by_name:
                raise ValueError("unknown clearance object {}".format(name))
            axis = rule.get("axis")
            if axis not in ("x", "y", "z"):
                raise ValueError("invalid clearance axis")
            direction = rule.get("direction")
            if direction not in ("positive", "negative"):
                raise ValueError("invalid clearance direction")
            index = {"x": 0, "y": 1, "z": 2}[axis]
            obj = object_by_name[name]
            center = float(obj["pose"]["position"][index])
            half_size = float(obj["size"][index]) / 2.0
            coordinate = float(position[index])
            clearance = (
                coordinate - (center + half_size)
                if direction == "positive"
                else (center - half_size) - coordinate
            )
            if clearance + 1.0e-9 < float(rule["minimum_m"]):
                raise ValueError(
                    "{} violates {} {} clearance: {:.4f} m".format(
                        item["id"], name, axis, clearance
                    )
                )
    if len(heights) < 2:
        raise ValueError("course needs at least two distinct heights")
    if len(orientation_representatives) < 3:
        raise ValueError("course needs at least three distinct orientations")
    if not any("positive_y" in role for role in roles) or not any(
        "negative_y" in role for role in roles
    ):
        raise ValueError("course needs both positive-y and negative-y bypasses")
    return True


class MultiAngleAvoidanceCourse:
    def __init__(self):
        self.scene_path = rospy.get_param("~scene_config")
        self.course_path = rospy.get_param("~course_config")
        self.startup_path = rospy.get_param("~startup_config")
        self.repetitions = int(rospy.get_param("~repetitions", 1))
        self.run_label = str(rospy.get_param("~run_label", "development"))
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        with open(self.scene_path, "r", encoding="utf-8") as stream:
            self.scene_config = yaml.safe_load(stream)
        with open(self.course_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        with open(self.startup_path, "r", encoding="utf-8") as stream:
            self.startup = yaml.safe_load(stream)["initial_configuration"]
        validate_course_config(self.config, self.scene_config)
        package = rospkg.RosPack().get_path("handarm_sim_demo")
        self.results_dir = rospy.get_param(
            "~results_dir",
            os.path.abspath(os.path.join(package, "..", "..", "results", "sim_baseline")),
        )
        self.status_pub = rospy.Publisher(
            "/handarm_sim_demo/multi_angle_status", String, queue_size=1, latch=True
        )
        self.waypoint_pub = rospy.Publisher(
            "/handarm_sim_demo/course_waypoints", MarkerArray, queue_size=1, latch=True
        )
        self.planned_pub = rospy.Publisher(
            "/handarm_sim_demo/planned_ee_path", MarkerArray, queue_size=1, latch=True
        )
        self.executed_pub = rospy.Publisher(
            "/handarm_sim_demo/executed_ee_path", MarkerArray, queue_size=1, latch=True
        )
        client = actionlib.SimpleActionClient("/move_group", MoveGroupAction)
        rospy.loginfo("[course] Waiting for MoveIt /move_group...")
        if not client.wait_for_server(rospy.Duration(90.0)):
            raise RuntimeError("move_group action did not become ready")
        self.group = moveit_commander.MoveGroupCommander(self.config["planning_group"])
        self.robot = moveit_commander.RobotCommander()
        self.group.set_end_effector_link(self.config["end_effector_link"])
        self.group.set_planner_id(self.config["planner_id"])
        self.group.set_planning_time(float(self.config["planning_time_s"]))
        self.group.set_num_planning_attempts(int(self.config["planning_attempts"]))
        self.group.set_max_velocity_scaling_factor(float(self.config["velocity_scaling"]))
        self.group.set_max_acceleration_scaling_factor(
            float(self.config["acceleration_scaling"])
        )
        self.group.set_goal_position_tolerance(
            float(self.config["goal_position_tolerance_m"])
        )
        self.group.set_goal_orientation_tolerance(
            float(self.config["goal_orientation_tolerance_rad"])
        )
        rospy.wait_for_service("/check_state_validity", timeout=30.0)
        rospy.wait_for_service("/compute_fk", timeout=30.0)
        self.check_state = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        self.compute_fk = rospy.ServiceProxy("/compute_fk", GetPositionFK)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.planned_markers = MarkerArray()
        self.executed_markers = MarkerArray()

    @staticmethod
    def unpack_plan(result):
        if isinstance(result, tuple):
            success, trajectory, planning_time, error_code = result
            return bool(success), trajectory, float(planning_time), int(error_code.val)
        trajectory = result
        return bool(trajectory.joint_trajectory.points), trajectory, None, None

    def wait_ready(self):
        for topic in ("/handarm_sim_demo/startup_ready", "/handarm_sim_demo/scene_ready"):
            if not rospy.wait_for_message(topic, Bool, timeout=90.0).data:
                raise RuntimeError("{} reported false".format(topic))
        rospy.wait_for_service("/controller_manager/list_controllers", timeout=30.0)
        controllers = rospy.ServiceProxy(
            "/controller_manager/list_controllers", ListControllers
        )().controller
        states = {item.name: item.state for item in controllers}
        if not EXPECTED_CONTROLLERS.issubset(states) or any(
            states[name] != "running" for name in EXPECTED_CONTROLLERS
        ):
            raise RuntimeError("trajectory controllers are not running")
        scene_status = json.loads(
            rospy.wait_for_message("/handarm_sim_demo/scene_status", String, timeout=10.0).data
        )
        if not scene_status.get("ready") or scene_status.get("scenario") != "multi_angle_obstacle_course":
            raise RuntimeError("PlanningScene is not the requested obstacle course")
        if scene_status.get("max_position_error_m", float("inf")) > 1.0e-6 or scene_status.get(
            "max_orientation_error_rad", float("inf")
        ) > 1.0e-6:
            raise RuntimeError("Gazebo and PlanningScene object poses are inconsistent")
        if not all(item.get("synchronized") for item in scene_status.get("objects", [])):
            raise RuntimeError("Gazebo and PlanningScene object geometry is inconsistent")
        return len(scene_status["objects"])

    def home_target(self):
        names = self.startup["arm"]["joint_names"]
        target = list(self.startup["arm"]["positions"])
        current = dict(zip(self.group.get_active_joints(), self.group.get_current_joint_values()))
        for index, name in enumerate(names):
            joint = self.robot.get_joint(name)
            minimum, maximum = joint.bounds()
            if name in self.startup.get("wraparound_joints", []):
                target[index] = nearest_equivalent_within_bounds(
                    target[index], current[name], minimum, maximum
                )
            elif not minimum <= target[index] <= maximum:
                raise ValueError("home target for {} is outside joint bounds".format(name))
        return names, target

    def validate_plan(self, trajectory):
        points = trajectory.joint_trajectory.points
        names = trajectory.joint_trajectory.joint_names
        if not points or not names:
            return False, "empty trajectory"
        state = self.robot.get_current_state()
        positions = list(state.joint_state.position)
        index = {name: i for i, name in enumerate(state.joint_state.name)}
        if any(name not in index for name in names):
            return False, "trajectory contains unknown joints"
        if len(points[0].positions) != len(names) or not all(
            math.isfinite(value) for value in points[0].positions
        ):
            return False, "trajectory start contains invalid positions"
        start_tolerance = float(self.config["trajectory_start_tolerance_rad"])
        if max(
            abs(value - positions[index[name]])
            for name, value in zip(names, points[0].positions)
        ) > start_tolerance:
            return False, "trajectory start does not match current joint state"
        previous_time = -1.0
        for point in points:
            seconds = point.time_from_start.to_sec()
            if not math.isfinite(seconds) or seconds <= previous_time:
                return False, "trajectory time is not strictly increasing"
            previous_time = seconds
            if len(point.positions) != len(names) or not all(
                math.isfinite(value) for value in point.positions
            ):
                return False, "trajectory contains invalid positions"
            for name, value in zip(names, point.positions):
                positions[index[name]] = value
            state.joint_state.position = positions
            response = self.check_state(
                GetStateValidityRequest(
                    robot_state=state, group_name=self.config["planning_group"]
                )
            )
            if not response.valid:
                return False, "trajectory waypoint is in collision"
        return True, ""

    def current_state_is_valid(self):
        response = self.check_state(
            GetStateValidityRequest(
                robot_state=self.robot.get_current_state(),
                group_name=self.config["planning_group"],
            )
        )
        return bool(response.valid)

    def wait_until_settled(self):
        """Require several fresh low-velocity joint samples before verification."""
        deadline = time.monotonic() + float(self.config["settle_timeout_s"])
        maximum_velocity = float(self.config["settle_max_joint_velocity_rad_s"])
        required = int(self.config["settle_consecutive_samples"])
        active = set(self.group.get_active_joints())
        consecutive = 0
        started = time.monotonic()
        while time.monotonic() < deadline and not rospy.is_shutdown():
            try:
                sample = rospy.wait_for_message("/joint_states", JointState, timeout=0.25)
            except rospy.ROSException:
                consecutive = 0
                continue
            velocity_by_name = dict(zip(sample.name, sample.velocity))
            if not active.issubset(velocity_by_name):
                consecutive = 0
                continue
            velocities = [velocity_by_name[name] for name in active]
            if all(math.isfinite(value) for value in velocities) and max(
                abs(value) for value in velocities
            ) <= maximum_velocity:
                consecutive += 1
                if consecutive >= required:
                    return True, time.monotonic() - started
            else:
                consecutive = 0
        return False, time.monotonic() - started

    def move_home(self):
        names, target = self.home_target()
        current = dict(zip(self.group.get_active_joints(), self.group.get_current_joint_values()))
        if max(abs(current[name] - value) for name, value in zip(names, target)) <= float(
            self.config["home_joint_tolerance_rad"]
        ):
            return True
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(dict(zip(names, target)))
        success, trajectory, _, _ = self.unpack_plan(self.group.plan())
        valid, _ = self.validate_plan(trajectory) if success else (False, "planning failed")
        if not success or not valid or not self.group.execute(trajectory, wait=True):
            self.group.stop()
            return False
        self.group.stop()
        current = dict(zip(self.group.get_active_joints(), self.group.get_current_joint_values()))
        return max(abs(current[name] - value) for name, value in zip(names, target)) <= float(
            self.config["home_joint_tolerance_rad"]
        )

    def direct_path_analysis(self, goal):
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()
        aware, aware_fraction = self.group.compute_cartesian_path(
            [goal], float(self.config["direct_path_eef_step_m"]), True
        )
        self.group.set_start_state_to_current_state()
        ignored, ignored_fraction = self.group.compute_cartesian_path(
            [goal], float(self.config["direct_path_eef_step_m"]), False
        )
        aware_invalid_states = self.count_invalid_states(aware)
        ignored_invalid_states = self.count_invalid_states(ignored)
        threshold = float(self.config["direct_path_complete_fraction"])
        collision = ignored_invalid_states > 0
        unsafe = collision or aware_fraction < threshold
        return {
            "collision_aware_fraction": aware_fraction,
            "ignore_collision_fraction": ignored_fraction,
            "direct_path_unsafe": unsafe,
            "direct_path_collision": collision,
            "collision_aware_invalid_states": aware_invalid_states,
            "ignore_collision_invalid_states": ignored_invalid_states,
            "collision_aware_points": len(aware.joint_trajectory.points),
            "ignore_collision_points": len(ignored.joint_trajectory.points),
        }

    def count_invalid_states(self, trajectory):
        """Count PlanningScene-invalid discrete states in one candidate path."""
        invalid = 0
        for state in self.robot_states_for_trajectory(trajectory):
            response = self.check_state(
                GetStateValidityRequest(
                    robot_state=state, group_name=self.config["planning_group"]
                )
            )
            invalid += int(not response.valid)
        return invalid

    def robot_states_for_trajectory(self, trajectory):
        state = self.robot.get_current_state()
        positions = list(state.joint_state.position)
        index = {name: i for i, name in enumerate(state.joint_state.name)}
        output = []
        for point in trajectory.joint_trajectory.points:
            for name, value in zip(trajectory.joint_trajectory.joint_names, point.positions):
                positions[index[name]] = value
            item = copy.deepcopy(state)
            item.joint_state.position = list(positions)
            output.append(item)
        return output

    def fk_points(self, trajectory):
        output = []
        for state in self.robot_states_for_trajectory(trajectory):
            request = GetPositionFKRequest()
            request.header.frame_id = self.config["frame_id"]
            request.fk_link_names = [self.config["end_effector_link"]]
            request.robot_state = state
            response = self.compute_fk(request)
            if response.error_code.val != response.error_code.SUCCESS or not response.pose_stamped:
                raise RuntimeError("FK failed while evaluating planned path")
            output.append(response.pose_stamped[0].pose.position)
        return output

    @staticmethod
    def joint_path_length(trajectory):
        points = trajectory.joint_trajectory.points
        return sum(
            math.sqrt(sum((b - a) ** 2 for a, b in zip(previous.positions, current.positions)))
            for previous, current in zip(points, points[1:])
        )

    @staticmethod
    def cartesian_path_length(points):
        return sum(point_distance(a, b) for a, b in zip(points, points[1:]))

    def sample_tool_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.config["frame_id"],
                self.config["end_effector_link"],
                rospy.Time(0),
                rospy.Duration(0.2),
            )
        except Exception:
            return None
        return Point(
            x=transform.transform.translation.x,
            y=transform.transform.translation.y,
            z=transform.transform.translation.z,
        )

    def execute_with_trace(self, trajectory):
        samples = []
        stop_event = threading.Event()

        def sample_loop():
            while not stop_event.is_set() and not rospy.is_shutdown():
                point = self.sample_tool_pose()
                if point is not None:
                    samples.append(point)
                time.sleep(0.04)

        worker = threading.Thread(target=sample_loop, daemon=True)
        worker.start()
        started = time.monotonic()
        success = bool(self.group.execute(trajectory, wait=True))
        elapsed = time.monotonic() - started
        stop_event.set()
        worker.join(timeout=1.0)
        final = self.sample_tool_pose()
        if final is not None:
            samples.append(final)
        self.group.stop()
        return success, elapsed, samples

    @staticmethod
    def path_marker(points, namespace, marker_id, color):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.006
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.points = list(points)
        marker.lifetime = rospy.Duration(0)
        return marker

    def publish_course_markers(self):
        markers = MarkerArray()
        marker_id = 0
        colors = ((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.3, 1.0))
        for index, item in enumerate(self.config["waypoints"], 1):
            pose = make_pose(item)
            text = Marker()
            text.header.frame_id = self.config["frame_id"]
            text.header.stamp = rospy.Time.now()
            text.ns, text.id, text.type, text.action = "course_labels", marker_id, Marker.TEXT_VIEW_FACING, Marker.ADD
            marker_id += 1
            text.pose = pose
            text.pose.position.z += 0.07
            text.scale.z = 0.035
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = "{}:{}".format(index, item["id"])
            markers.markers.append(text)
            q = item["orientation_xyzw"]
            x, y, z, w = q
            rotation = (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            )
            for axis in range(3):
                arrow = Marker()
                arrow.header = text.header
                arrow.ns, arrow.id, arrow.type, arrow.action = "course_axes", marker_id, Marker.ARROW, Marker.ADD
                marker_id += 1
                arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.006, 0.012, 0.016
                arrow.color.r, arrow.color.g, arrow.color.b = colors[axis]
                arrow.color.a = 1.0
                start = Point(x=pose.position.x, y=pose.position.y, z=pose.position.z)
                end = Point(
                    x=start.x + 0.06 * rotation[0][axis],
                    y=start.y + 0.06 * rotation[1][axis],
                    z=start.z + 0.06 * rotation[2][axis],
                )
                arrow.points = [start, end]
                markers.markers.append(arrow)
        for alias in ("obstacle_a", "obstacle_b"):
            item = self.scene_config["objects"][alias]
            cube = Marker()
            cube.header.frame_id = self.config["frame_id"]
            cube.header.stamp = rospy.Time.now()
            cube.ns, cube.id, cube.type, cube.action = "obstacle_aabbs", marker_id, Marker.CUBE, Marker.ADD
            marker_id += 1
            cube.pose.position.x, cube.pose.position.y, cube.pose.position.z = item["pose"]["position"]
            cube.pose.orientation.w = 1.0
            cube.scale.x, cube.scale.y, cube.scale.z = item["size"]
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = 1.0, 0.5, 0.0, 0.25
            markers.markers.append(cube)
        self.waypoint_pub.publish(markers)

    def plan_segment(self, goal):
        last = None
        total_wall = 0.0
        for _ in range(int(self.config["planning_retries"])):
            self.group.stop()
            self.group.clear_pose_targets()
            self.group.set_start_state_to_current_state()
            self.group.set_pose_target(goal, self.config["end_effector_link"])
            started = time.monotonic()
            result = self.group.plan()
            total_wall += time.monotonic() - started
            success, trajectory, planning_time, error_code = self.unpack_plan(result)
            last = (success, trajectory, planning_time, error_code, total_wall)
            if success and trajectory.joint_trajectory.points:
                return last
        return last

    def run_course(self, repetition, collision_object_count):
        states = [
            "INIT",
            "WAIT_FOR_ROBOT",
            "WAIT_FOR_CONTROLLERS",
            "WAIT_FOR_SCENE",
            "MOVE_HOME",
            "LOAD_COURSE",
        ]
        segments = []
        unsafe_count = 0
        planning_total = 0.0
        execution_total = 0.0
        completed = 0
        failure_reason = ""
        started = time.monotonic()
        try:
            rospy.loginfo("[course] Repetition %d/%d: moving home...", repetition, self.repetitions)
            if not self.move_home():
                raise RuntimeError("MOVE_HOME failed")
            self.publish_course_markers()
            for segment_id, waypoint in enumerate(self.config["waypoints"], 1):
                states.append("PLAN_SEGMENT")
                goal = make_pose(waypoint)
                start_pose = self.group.get_current_pose(self.config["end_effector_link"]).pose
                if not self.current_state_is_valid():
                    raise RuntimeError(
                        "segment {} starts in an invalid PlanningScene state".format(
                            segment_id
                        )
                    )
                rospy.loginfo(
                    "[course] Segment %d/%d -> %s: checking direct path...",
                    segment_id,
                    len(self.config["waypoints"]),
                    waypoint["id"],
                )
                direct = self.direct_path_analysis(goal)
                unsafe_count += int(direct["direct_path_unsafe"])
                row = {field: None for field in CSV_FIELDS}
                row.update(
                    run_id=self.run_id,
                    run_label=self.run_label,
                    course_id=self.config["course_id"],
                    repetition=repetition,
                    segment_id=segment_id,
                    waypoint_id=waypoint["id"],
                    start_pose=json.dumps(pose_dict(start_pose), separators=(",", ":")),
                    goal_pose=json.dumps(pose_dict(goal), separators=(",", ":")),
                    direct_path_collision=direct["direct_path_collision"],
                    direct_path_unsafe=direct["direct_path_unsafe"],
                    direct_collision_aware_fraction=direct["collision_aware_fraction"],
                    direct_ignore_collision_fraction=direct["ignore_collision_fraction"],
                    direct_collision_aware_invalid_states=direct[
                        "collision_aware_invalid_states"
                    ],
                    direct_ignore_collision_invalid_states=direct[
                        "ignore_collision_invalid_states"
                    ],
                    planning_success=False,
                    trajectory_points=0,
                    execution_success=False,
                    collision_validation_pass=False,
                    failure_reason="",
                )
                segments.append(row)
                success, trajectory, planning_time, _, planning_wall = self.plan_segment(goal)
                row["planning_success"] = bool(success)
                row["planning_time_s"] = (
                    planning_time
                    if success and planning_time is not None and math.isfinite(planning_time)
                    else None
                )
                row["planning_wall_time_s"] = planning_wall
                planning_total += planning_wall
                row["trajectory_points"] = len(trajectory.joint_trajectory.points)
                if not success or not trajectory.joint_trajectory.points:
                    raise RuntimeError("segment {} planning failed".format(segment_id))
                states.append("VALIDATE_SEGMENT")
                valid, reason = self.validate_plan(trajectory)
                row["collision_validation_pass"] = valid
                if not valid:
                    raise RuntimeError("segment {}: {}".format(segment_id, reason))
                planned_points = self.fk_points(trajectory)
                row["joint_path_length"] = self.joint_path_length(trajectory)
                row["end_effector_path_length_m"] = self.cartesian_path_length(planned_points)
                self.planned_markers.markers.append(
                    self.path_marker(planned_points, "planned_ee_path", segment_id, (0.1, 0.8, 1.0, 1.0))
                )
                self.planned_pub.publish(self.planned_markers)
                states.append("EXECUTE_SEGMENT")
                rospy.loginfo(
                    "[course] Segment %d: executing %d collision-free points...",
                    segment_id,
                    row["trajectory_points"],
                )
                executed, execution_time, actual_points = self.execute_with_trace(trajectory)
                row["execution_success"] = executed
                row["execution_time_s"] = execution_time
                execution_total += execution_time
                self.executed_markers.markers.append(
                    self.path_marker(actual_points, "executed_ee_path", segment_id, (0.1, 1.0, 0.2, 1.0))
                )
                self.executed_pub.publish(self.executed_markers)
                if not executed:
                    raise RuntimeError("segment {} execution returned false".format(segment_id))
                states.append("VERIFY_SEGMENT")
                settled, settle_time = self.wait_until_settled()
                row["settle_time_s"] = settle_time
                if not settled:
                    raise RuntimeError("segment {} did not settle".format(segment_id))
                actual = self.group.get_current_pose(self.config["end_effector_link"]).pose
                row["position_error_m"] = point_distance(actual.position, goal.position)
                row["orientation_error_deg"] = quaternion_angle_deg(
                    actual.orientation, goal.orientation
                )
                if row["position_error_m"] > float(
                    self.config["verification_position_tolerance_m"]
                ) or row["orientation_error_deg"] > float(
                    self.config["verification_orientation_tolerance_deg"]
                ):
                    raise RuntimeError("segment {} final pose verification failed".format(segment_id))
                completed += 1
                states.append("NEXT_SEGMENT")
            if unsafe_count < int(self.config["minimum_unsafe_direct_segments"]):
                raise RuntimeError(
                    "only {} direct segments were proven unsafe".format(unsafe_count)
                )
            collision_count = sum(bool(row["direct_path_collision"]) for row in segments)
            if collision_count < int(
                self.config["minimum_collision_proven_direct_segments"]
            ):
                raise RuntimeError(
                    "only {} direct segments were proven colliding".format(
                        collision_count
                    )
                )
            states.extend(["HOLD", "DONE"])
            self.group.stop()
            time.sleep(float(self.config["hold_time_s"]))
            course_success = True
        except Exception as exc:
            self.group.stop()
            self.group.clear_pose_targets()
            states.append("FAILED")
            failure_reason = str(exc)
            if segments and not segments[-1]["failure_reason"]:
                segments[-1]["failure_reason"] = failure_reason
            rospy.logerr("[course] Repetition %d FAILED: %s", repetition, exc)
            course_success = False
        return {
            "repetition": repetition,
            "course_success": course_success,
            "waypoint_count": len(self.config["waypoints"]),
            "segment_count": len(self.config["waypoints"]),
            "completed_segments": completed,
            "unsafe_direct_segments": unsafe_count,
            "collision_proven_direct_segments": sum(
                bool(row["direct_path_collision"]) for row in segments
            ),
            "total_planning_time_s": planning_total,
            "total_execution_time_s": execution_total,
            "all_segments_collision_free": bool(segments)
            and all(row["collision_validation_pass"] for row in segments),
            "failure_reason": failure_reason,
            "states": states,
            "wall_time_s": time.monotonic() - started,
            "collision_object_count": collision_object_count,
            "segments": segments,
        }

    def save(self, courses):
        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = os.path.join(self.results_dir, "multi_angle_avoidance_results.csv")
        rows = [row for course in courses for row in course["segments"]]
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_label": self.run_label,
            "course_id": self.config["course_id"],
            "waypoint_count": len(self.config["waypoints"]),
            "segment_count": len(self.config["waypoints"]),
            "repetitions_requested": self.repetitions,
            "repetitions_completed": len(courses),
            "successful_courses": sum(course["course_success"] for course in courses),
            "course_success_rate": sum(course["course_success"] for course in courses)
            / float(self.repetitions),
            "all_segments_collision_free": bool(courses)
            and all(course["all_segments_collision_free"] for course in courses),
            "minimum_unsafe_direct_segments": int(
                self.config["minimum_unsafe_direct_segments"]
            ),
            "minimum_collision_proven_direct_segments": int(
                self.config["minimum_collision_proven_direct_segments"]
            ),
            "courses": courses,
        }
        json_path = os.path.join(self.results_dir, "multi_angle_avoidance_summary.json")
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
        immutable_json_path = os.path.join(
            self.results_dir,
            "multi_angle_avoidance_{}_{}.json".format(
                self.run_label, self.run_id
            ),
        )
        with open(immutable_json_path, "x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
        return csv_path, json_path, immutable_json_path, summary

    def run(self):
        self.run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        collision_count = self.wait_ready()
        courses = []
        for repetition in range(1, self.repetitions + 1):
            course = self.run_course(repetition, collision_count)
            courses.append(course)
            self.status_pub.publish(json.dumps(course, sort_keys=True))
            if not course["course_success"]:
                break
        csv_path, json_path, immutable_json_path, summary = self.save(courses)
        rospy.loginfo(
            "[course] Results: %s, %s and %s",
            csv_path,
            json_path,
            immutable_json_path,
        )
        if summary["successful_courses"] == self.repetitions:
            rospy.loginfo("[course] DONE: %d/%d complete courses passed.", self.repetitions, self.repetitions)
            return True
        return False


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("multi_angle_avoidance_course")
    try:
        success = MultiAngleAvoidanceCourse().run()
    except Exception as exc:
        rospy.logfatal("Multi-angle course initialization failed: %s", exc)
        success = False
    finally:
        moveit_commander.roscpp_shutdown()
    raise SystemExit(0 if success else 9)


if __name__ == "__main__":
    main()
