#!/usr/bin/env python3
"""Synchronize YAML scene geometry between Gazebo and MoveIt."""

import json
import math
import time

import moveit_commander
import rospy
import yaml
from gazebo_msgs.srv import DeleteModel, GetModelState, GetWorldProperties
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from std_msgs.msg import Bool, String


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
        if item.get("type") != "box":
            raise ValueError("only box geometry is supported: {}".format(alias))
        if len(item.get("size", [])) != 3 or any(v <= 0 for v in item["size"]):
            raise ValueError("invalid box size for {}".format(alias))
        padding = item.get("planning_padding_m", [0.0, 0.0, 0.0])
        if len(padding) != 3 or any(v < 0 for v in padding):
            raise ValueError("invalid planning padding for {}".format(alias))
        if len(item.get("pose", {}).get("position", [])) != 3:
            raise ValueError("invalid position for {}".format(alias))
        if item.get("pose", {}).get("orientation_rpy") != [0.0, 0.0, 0.0]:
            raise ValueError("stage-3 boxes require zero RPY: {}".format(alias))
    return [objects[alias] for alias in aliases]


class SceneManager:
    def __init__(self):
        self.timeout = float(rospy.get_param("~startup_timeout", 90.0))
        self.scene_path = rospy.get_param("~scene_config")
        self.scenario = rospy.get_param("~scenario", "double_obstacle")
        with open(self.scene_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self.objects = validate_config(self.config, self.scenario)
        self.ready_pub = rospy.Publisher(
            "/handarm_sim_demo/scene_ready", Bool, queue_size=1, latch=True
        )
        self.status_pub = rospy.Publisher(
            "/handarm_sim_demo/scene_status", String, queue_size=1, latch=True
        )

    def wait_startup(self):
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
        expected_names = {item["name"] for item in self.objects}
        # Logical support surfaces (for example the z=0 ground used by grasp
        # geometry) need not be spawned, deleted, or mirrored as ordinary box
        # models. Existing scene files omit this flag and keep prior behavior.
        managed_items = [
            item for item in self.config["objects"].values()
            if item.get("scene_manager_enabled", True)
        ]
        all_names = {item["name"] for item in managed_items}
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if all_names.issubset(set(get_world().model_names)):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Gazebo scene models did not appear")

        removed_models = []
        for name in sorted(all_names - expected_names):
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
            if expected_names.issubset(world_names) and not (
                (all_names - expected_names) & world_names
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Gazebo active model set did not converge")

        gazebo_poses = {}
        for name in sorted(all_names):
            scene.remove_world_object(name)
        for item in self.objects:
            state = get_state(item["name"], "world")
            if not state.success:
                raise RuntimeError(
                    "Gazebo pose unavailable for {}: {}".format(
                        item["name"], state.status_message
                    )
                )
            gazebo_poses[item["name"]] = state.pose

        for item in self.objects:
            pose = PoseStamped()
            pose.header.frame_id = self.config["frame_id"]
            pose.header.stamp = rospy.Time.now()
            pose.pose = gazebo_poses[item["name"]]
            padding = item.get("planning_padding_m", [0.0, 0.0, 0.0])
            planning_size = tuple(
                size + 2.0 * margin for size, margin in zip(item["size"], padding)
            )
            scene.add_box(item["name"], pose, size=planning_size)

        while time.monotonic() < deadline and not rospy.is_shutdown():
            if expected_names.issubset(set(scene.get_known_object_names())):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("PlanningScene objects did not appear")

        objects = scene.get_objects(list(expected_names))
        rows = []
        max_position_error = 0.0
        max_orientation_error = 0.0
        for item in self.objects:
            name = item["name"]
            collision = objects.get(name)
            if collision is None or len(collision.primitives) != 1:
                raise RuntimeError("invalid PlanningScene geometry for {}".format(name))
            primitive = collision.primitives[0]
            if primitive.type != primitive.BOX:
                raise RuntimeError("PlanningScene object is not a box: {}".format(name))
            padding = item.get("planning_padding_m", [0.0, 0.0, 0.0])
            planning_size = [
                size + 2.0 * margin for size, margin in zip(item["size"], padding)
            ]
            dimension_error = max(
                abs(a - b) for a, b in zip(primitive.dimensions, planning_size)
            )
            if dimension_error > 1e-9:
                raise RuntimeError("dimension mismatch for {}".format(name))
            local_pose = Pose()
            local_pose.orientation.w = 1.0
            if collision.primitive_poses:
                local_pose = collision.primitive_poses[0]
            planning_pose = compose_pose(collision.pose, local_pose)
            gazebo_pose = gazebo_poses[name]
            position_error = math.sqrt(
                (planning_pose.position.x - gazebo_pose.position.x) ** 2
                + (planning_pose.position.y - gazebo_pose.position.y) ** 2
                + (planning_pose.position.z - gazebo_pose.position.z) ** 2
            )
            orientation_error = quaternion_angle(
                planning_pose.orientation, gazebo_pose.orientation
            )
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, orientation_error)
            rows.append(
                {
                    "gazebo_name": name,
                    "planning_scene_name": name,
                    "size_m": list(item["size"]),
                    "planning_size_m": planning_size,
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
            "removed_inactive_gazebo_models": removed_models,
        }

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
