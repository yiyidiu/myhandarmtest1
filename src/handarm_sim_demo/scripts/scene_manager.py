#!/usr/bin/env python3
"""Synchronize YAML scene geometry between Gazebo and MoveIt."""

import json
import math
import time

import moveit_commander
import rospy
import yaml
from moveit_msgs.msg import AllowedCollisionEntry, CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPlanningSceneRequest
from gazebo_msgs.srv import DeleteModel, GetModelState, GetWorldProperties
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from std_msgs.msg import Bool, String
from tf.transformations import quaternion_from_euler


def quaternion_angle(a, b):
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def quaternion_multiply(a, b):
    return Quaternion(
        x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


def compose_pose(parent, child):
    q = parent.orientation
    p = child.position
    pure = Quaternion(x=p.x, y=p.y, z=p.z, w=0.0)
    inverse = Quaternion(x=-q.x, y=-q.y, z=-q.z, w=q.w)
    rotated = quaternion_multiply(quaternion_multiply(q, pure), inverse)
    result = Pose()
    result.position.x = parent.position.x + rotated.x
    result.position.y = parent.position.y + rotated.y
    result.position.z = parent.position.z + rotated.z
    result.orientation = quaternion_multiply(q, child.orientation)
    return result


def validate_config(data, scenario):
    if data.get("frame_id") != "world":
        raise ValueError("scene frame_id must be world")
    objects = data.get("objects", {})
    aliases = data.get("scenario_object_sets", {}).get(scenario)
    if not aliases:
        raise ValueError("unknown or empty scenario {}".format(scenario))
    for alias in aliases:
        if alias not in objects:
            raise ValueError("scenario references unknown object {}".format(alias))
        item = objects[alias]
        if item.get("type") not in ("box", "cylinder", "compound"):
            raise ValueError("unsupported scene geometry: {}".format(alias))
        if len(item.get("size", [])) != 3 or any(v <= 0 for v in item["size"]):
            raise ValueError("invalid box size for {}".format(alias))
        if item["type"] == "cylinder" and abs(item["size"][0] - item["size"][1]) > 1e-9:
            raise ValueError("cylinder size must be [diameter, diameter, height]")
        padding = item.get("planning_padding_m", [0.0, 0.0, 0.0])
        if len(padding) != 3 or any(v < 0 for v in padding):
            raise ValueError("invalid planning padding for {}".format(alias))
        if len(item.get("pose", {}).get("position", [])) != 3:
            raise ValueError("invalid position for {}".format(alias))
        scene_primitives(item)
        rpy = item.get("pose", {}).get("orientation_rpy", [])
        if len(rpy) != 3 or any(not math.isfinite(v) for v in rpy):
            raise ValueError("invalid orientation for {}".format(alias))
    return [objects[alias] for alias in aliases]


def planning_dimensions(item):
    padding = item.get("planning_padding_m", [0.0, 0.0, 0.0])
    size = [v + 2.0 * p for v, p in zip(item["size"], padding)]
    # shape_msgs/SolidPrimitive stores a cylinder as [height, radius].
    return [size[2], max(size[:2]) / 2.0] if item["type"] == "cylinder" else size


def add_scene_object(scene, item, pose):
    if item["type"] == "compound":
        obj = CollisionObject(id=item["name"], operation=CollisionObject.ADD)
        obj.header = pose.header
        obj.pose = pose.pose
        for primitive, local in scene_primitives(item):
            obj.primitives.append(primitive)
            obj.primitive_poses.append(local)
        scene.add_object(obj)
        return
    dimensions = planning_dimensions(item)
    if item["type"] == "cylinder":
        scene.add_cylinder(item["name"], pose, height=dimensions[0], radius=dimensions[1])
    else:
        scene.add_box(item["name"], pose, size=dimensions)


def scene_primitives(item):
    """Return the same solid parts used by the free Gazebo bottle model."""
    parts = item.get("collision_parts") if item["type"] == "compound" else [
        dict(type=item["type"], dimensions=planning_dimensions(item), position=[0, 0, 0])]
    if not parts:
        raise ValueError("compound object needs collision parts")
    types = {"box": (SolidPrimitive.BOX, 3),
             "cylinder": (SolidPrimitive.CYLINDER, 2),
             "sphere": (SolidPrimitive.SPHERE, 1)}
    result = []
    for part in parts:
        kind, count = types[part["type"]]
        dims, xyz = part["dimensions"], part["position"]
        if (len(dims) != count or len(xyz) != 3 or
                any(not math.isfinite(x) or x <= 0 for x in dims) or
                any(not math.isfinite(x) for x in xyz)):
            raise ValueError("invalid collision part")
        local = Pose()
        local.position.x, local.position.y, local.position.z = xyz
        local.orientation.w = 1.0
        result.append((SolidPrimitive(type=kind, dimensions=dims), local))
    return result


class SceneManager:
    def __init__(self):
        self.timeout = float(rospy.get_param("~startup_timeout", 90.0))
        self.wait_for_startup_ready = bool(rospy.get_param(
            "~wait_for_startup_ready", True))
        self.scene_path = rospy.get_param("~scene_config")
        self.scenario = rospy.get_param("~scenario", "double_obstacle")
        self.allow_grasp_contacts = bool(rospy.get_param("~allow_grasp_contacts", False))
        self.keep_contacts_until_separated = bool(rospy.get_param(
            "/shared_teleop/control/constrained_velocity_projection", False))
        # In the Gazebo constrained controller, grasp targets are physical
        # objects to touch/push, not obstacles that stop the whole arm.
        # Preserve the former selected-finger policy for legacy rollback.
        self.physical_target_contacts = self.keep_contacts_until_separated
        self.finger_object_clearance = {}
        self.clearance_received_at = 0.0
        self.tool_pose = None
        self.grasp_target = None
        self.was_closing = False
        with open(self.scene_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self.objects = validate_config(self.config, self.scenario)
        self.planning_objects = [
            item for item in self.objects
            if item.get("planning_scene_enabled", True)
        ]
        if not self.planning_objects:
            raise ValueError("scenario has no PlanningScene collision objects")
        self.ready_pub = rospy.Publisher(
            "/handarm_sim_demo/scene_ready", Bool, queue_size=1, latch=True
        )
        self.status_pub = rospy.Publisher(
            "/handarm_sim_demo/scene_status", String, queue_size=1, latch=True
        )

    def wait_startup(self):
        if not self.wait_for_startup_ready:
            return
        msg = rospy.wait_for_message(
            "/handarm_sim_demo/startup_ready", Bool, timeout=self.timeout
        )
        if not msg.data:
            raise RuntimeError("simulation startup was not ready")

    def synchronize(self):
        rospy.wait_for_service("/gazebo/get_world_properties", timeout=self.timeout)
        rospy.wait_for_service("/gazebo/get_model_state", timeout=self.timeout)
        rospy.wait_for_service("/gazebo/delete_model", timeout=self.timeout)
        rospy.wait_for_service("/get_planning_scene", timeout=self.timeout)
        get_world = rospy.ServiceProxy(
            "/gazebo/get_world_properties", GetWorldProperties
        )
        get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        scene = moveit_commander.PlanningSceneInterface(synchronous=True)

        deadline = time.monotonic() + self.timeout
        expected_names = {item["name"] for item in self.planning_objects}
        # Logical support surfaces (for example the z=0 ground used by grasp
        # geometry) need not be spawned, deleted, or mirrored as ordinary box
        # models. Existing scene files omit this flag and keep prior behavior.
        managed_items = [
            item for item in self.config["objects"].values()
            if item.get("scene_manager_enabled", True)
        ]
        all_names = {item["name"] for item in managed_items}
        expected_gazebo_names = {
            item["name"] for item in self.objects
            if item.get("scene_manager_enabled", True)
        }
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if expected_gazebo_names.issubset(set(get_world().model_names)):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Gazebo scene models did not appear")

        removed_models = []
        present_models = set(get_world().model_names)
        for name in sorted((all_names - expected_gazebo_names) & present_models):
            response = delete_model(name)
            if not response.success:
                raise RuntimeError(
                    "failed to remove inactive Gazebo model {}: {}".format(
                        name, response.status_message
                    )
                )
            removed_models.append(name)
        while time.monotonic() < deadline and not rospy.is_shutdown():
            world_names = set(get_world().model_names)
            if expected_gazebo_names.issubset(world_names) and not (
                (all_names - expected_gazebo_names) & world_names
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Gazebo active model set did not converge")

        collision_poses = {}
        all_planning_names = {
            item["name"] for item in self.config["objects"].values()
            if item.get("planning_scene_enabled", True)
        }
        for name in sorted(all_planning_names):
            scene.remove_world_object(name)
        for item in self.planning_objects:
            if item.get("scene_manager_enabled", True):
                state = get_state(item["name"], "world")
                if not state.success:
                    raise RuntimeError(
                        "Gazebo pose unavailable for {}: {}".format(
                            item["name"], state.status_message
                        )
                    )
                collision_poses[item["name"]] = state.pose
            else:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = (
                    item["pose"]["position"]
                )
                pose.orientation = Quaternion(*quaternion_from_euler(
                    *item["pose"]["orientation_rpy"]))
                collision_poses[item["name"]] = pose

        for item in self.planning_objects:
            pose = PoseStamped()
            pose.header.frame_id = self.config["frame_id"]
            pose.header.stamp = rospy.Time.now()
            pose.pose = collision_poses[item["name"]]
            add_scene_object(scene, item, pose)

        while time.monotonic() < deadline and not rospy.is_shutdown():
            if expected_names.issubset(set(scene.get_known_object_names())):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("PlanningScene objects did not appear")

        objects = scene.get_objects(list(expected_names))
        if self.allow_grasp_contacts:
            # This changes MoveIt control permission only. Gazebo keeps the
            # target collision shapes, contact forces, friction and mass.
            # Robot/robot and robot/table/ground entries are never changed.
            request = GetPlanningSceneRequest()
            request.components.components = request.components.ALLOWED_COLLISION_MATRIX
            current = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)(request).scene
            acm = current.allowed_collision_matrix
            touch_links = (moveit_commander.RobotCommander().get_link_names()
                           if self.physical_target_contacts else
                           self.config.get("grasp_touch_links", []))
            graspable = [item["name"] for item in self.planning_objects
                         if item.get("graspable", False)]
            for name in touch_links + graspable:
                if name not in acm.entry_names:
                    acm.entry_names.append(name)
                    for entry in acm.entry_values:
                        entry.enabled.append(False)
                    acm.entry_values.append(AllowedCollisionEntry(
                        enabled=[False] * len(acm.entry_names)))
            for name in graspable:
                for link in touch_links:
                    i, j = acm.entry_names.index(name), acm.entry_names.index(link)
                    acm.entry_values[i].enabled[j] = self.physical_target_contacts
                    acm.entry_values[j].enabled[i] = self.physical_target_contacts
            update = PlanningScene(is_diff=True)
            update.allowed_collision_matrix = acm
            if not rospy.ServiceProxy("/apply_planning_scene", ApplyPlanningScene)(update).success:
                raise RuntimeError("could not set graspable-object finger contact permissions")
            self.live_scene = scene
            self.live_objects = {item["name"]: item for item in self.planning_objects
                                 if item.get("graspable", False)}
            self.last_object_poses = dict(collision_poses)
            self.last_object_update = 0.0
        rows = []
        max_position_error = 0.0
        max_orientation_error = 0.0
        for item in self.planning_objects:
            name = item["name"]
            collision = objects.get(name)
            expected_parts = scene_primitives(item)
            if collision is None or len(collision.primitives) != len(expected_parts):
                raise RuntimeError("invalid PlanningScene geometry for {}".format(name))
            padding = item.get("planning_padding_m", [0.0, 0.0, 0.0])
            planning_size = planning_dimensions(item)
            position_error = orientation_error = 0.0
            for index, (expected, expected_local) in enumerate(expected_parts):
                primitive = collision.primitives[index]
                if primitive.type != expected.type or len(primitive.dimensions) != len(expected.dimensions):
                    raise RuntimeError("PlanningScene shape mismatch: {}".format(name))
                if max(abs(a-b) for a, b in zip(primitive.dimensions, expected.dimensions)) > 1e-9:
                    raise RuntimeError("dimension mismatch for {}".format(name))
                planning_pose = compose_pose(collision.pose, collision.primitive_poses[index])
                gazebo_pose = compose_pose(collision_poses[name], expected_local)
                position_error = max(position_error, math.sqrt(sum(
                    (getattr(planning_pose.position, axis) - getattr(gazebo_pose.position, axis))**2
                    for axis in ("x", "y", "z"))))
                orientation_error = max(orientation_error, quaternion_angle(
                    planning_pose.orientation, gazebo_pose.orientation))
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, orientation_error)
            rows.append(
                {
                    "gazebo_name": name,
                    "planning_scene_name": name,
                    "pose_source": (
                        "gazebo" if item.get("scene_manager_enabled", True)
                        else "configured_static_proxy"),
                    "size_m": list(item["size"]),
                    "planning_size_m": planning_size,
                    "primitive_count": len(expected_parts),
                    "planning_padding_m": list(padding),
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    "synchronized": position_error <= 1e-6
                    and orientation_error <= 1e-6,
                }
            )
        if not all(row["synchronized"] for row in rows):
            raise RuntimeError("Gazebo/PlanningScene pose mismatch")
        return {
            "ready": True,
            "scenario": self.scenario,
            "frame_id": self.config["frame_id"],
            "objects": rows,
            "max_position_error_m": max_position_error,
            "max_orientation_error_rad": max_orientation_error,
            "target_is_world_collision_object": "target_object" in expected_names,
            "target_contact_policy": ("GAZEBO_PHYSICAL_CONTACT_NO_ARM_STOP"
                                      if self.physical_target_contacts and self.allow_grasp_contacts
                                      else "SELECTED_GRASP_FINGER_CONTACT"),
            "removed_inactive_gazebo_models": removed_models,
        }

    def tool_pose_callback(self, message):
        try:
            state = json.loads(message.data)
            p, q = state.get("current_position"), state.get("current_quaternion_xyzw")
            if p is None or q is None:
                return
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = p
            pose.orientation = Quaternion(*q)
            self.tool_pose = pose
        except (ValueError, TypeError):
            return

    def finger_clearance_callback(self, message):
        try:
            state = json.loads(message.data)
            distances = state.get("finger_object_clearance_m", {})
            if distances:
                self.finger_object_clearance = distances
                self.clearance_received_at = time.monotonic()
        except (ValueError, TypeError):
            return

    def finger_contact_callback(self, message):
        try:
            state = json.loads(message.data)
            closure = state.get("normalized_robot_closure")
            if not state.get("calibrated") or closure is None:
                return
            closing = bool(max(closure) > 0.10 or state.get("grip_latched")
                           or state.get("release_phase"))
            target = self.grasp_target
            if not closing:
                if self.keep_contacts_until_separated and target is not None:
                    distance = self.finger_object_clearance.get(target)
                    fresh = time.monotonic() - self.clearance_received_at < 0.25
                    if fresh and distance is not None and distance >= 0.002:
                        target = None
                else:
                    target = None
            elif (not self.was_closing or target is None) and self.tool_pose is not None:
                # A user may start closing before entering the grasp envelope.
                # Keep looking until the first nearby target is acquired; do
                # not require reopening just to permit the intended contact.
                local = Pose()
                local.position.x, local.position.y, local.position.z = 0.0085, 0.0101, 0.0757
                local.orientation.w = 1.0
                center = compose_pose(self.tool_pose, local).position
                distances = [(math.sqrt(sum((getattr(center, axis) -
                    getattr(self.last_object_poses[name].position, axis)) ** 2
                    for axis in ("x", "y", "z"))), name) for name in self.live_objects]
                distance, nearest = min(distances, default=(float("inf"), None))
                target = nearest if distance <= 0.08 else None
            if target == self.grasp_target:
                self.was_closing = closing
                return
            request = GetPlanningSceneRequest()
            request.components.components = request.components.ALLOWED_COLLISION_MATRIX
            acm = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)(request).scene.allowed_collision_matrix
            # MoveIt can omit all-false object rows from the returned matrix.
            # Recreate those explicit rows before granting the selected grasp.
            for name in list(self.live_objects) + self.config.get("grasp_touch_links", []):
                if name not in acm.entry_names:
                    acm.entry_names.append(name)
                    for entry in acm.entry_values:
                        entry.enabled.append(False)
                    acm.entry_values.append(AllowedCollisionEntry(
                        enabled=[False] * len(acm.entry_names)))
            for name in self.live_objects:
                for link in self.config.get("grasp_touch_links", []):
                    i, j = acm.entry_names.index(name), acm.entry_names.index(link)
                    acm.entry_values[i].enabled[j] = name == target
                    acm.entry_values[j].enabled[i] = name == target
            update = PlanningScene(is_diff=True)
            update.allowed_collision_matrix = acm
            if rospy.ServiceProxy("/apply_planning_scene", ApplyPlanningScene)(update).success:
                self.grasp_target = target
                self.was_closing = closing
                rospy.loginfo("Grasp finger contact target: %s", target or "none (open-hand collision checks active)")
        except (ValueError, TypeError, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(2.0, "Could not update grasp contact permission: %s", exc)

    def moving_objects_callback(self, message):
        # Keep collision geometry at the object's measured Gazebo position
        # after a physical pick/release; never attach or teleport the object.
        now = time.monotonic()
        if now - self.last_object_update < 0.10:
            return
        self.last_object_update = now
        poses = dict(zip(message.name, message.pose))
        for name, item in self.live_objects.items():
            pose = poses.get(name)
            if pose is None:
                continue
            previous = self.last_object_poses[name]
            distance = math.sqrt(sum((getattr(pose.position, axis) -
                                      getattr(previous.position, axis)) ** 2
                                     for axis in ("x", "y", "z")))
            if distance < 0.001 and quaternion_angle(
                    pose.orientation, previous.orientation) < 0.01:
                continue
            stamped = PoseStamped()
            stamped.header.frame_id = "world"
            stamped.pose = pose
            add_scene_object(self.live_scene, item, stamped)
            self.last_object_poses[name] = pose

    def run(self):
        try:
            rospy.loginfo(
                "[scene] Waiting for Gazebo, robot and controllers to become ready..."
            )
            self.wait_startup()
            rospy.loginfo(
                "[scene] Startup ready. Synchronizing Gazebo objects with MoveIt PlanningScene..."
            )
            status = self.synchronize()
            self.status_pub.publish(json.dumps(status, sort_keys=True))
            self.ready_pub.publish(True)
            if self.allow_grasp_contacts:
                if not self.physical_target_contacts:
                    self.tool_subscriber = rospy.Subscriber(
                        "/shared_teleop/trend_diagnostics", String,
                        self.tool_pose_callback, queue_size=1)
                    self.finger_subscriber = rospy.Subscriber(
                        "/shared_teleop/finger_diagnostics", String,
                        self.finger_contact_callback, queue_size=1)
                self.object_subscriber = rospy.Subscriber(
                    "/gazebo/model_states", ModelStates,
                    self.moving_objects_callback, queue_size=1)
            rospy.loginfo("[scene] READY. Gazebo and PlanningScene synchronized: %s", status)
            rospy.spin()
        except Exception as exc:
            failure = {"ready": False, "failure_reason": str(exc)}
            self.status_pub.publish(json.dumps(failure, sort_keys=True))
            self.ready_pub.publish(False)
            rospy.logfatal("Scene synchronization failed: %s", exc)
            raise SystemExit(4)


def main():
    moveit_commander.roscpp_initialize([])
    rospy.init_node("scene_manager")
    SceneManager().run()


if __name__ == "__main__":
    main()
