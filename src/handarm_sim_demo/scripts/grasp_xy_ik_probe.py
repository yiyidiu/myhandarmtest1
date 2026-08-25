#!/usr/bin/env python3
"""Non-executing virtual XY coarse screen for strict top-down candidates.

This node never changes Gazebo and never executes a robot trajectory. For each
configured XY it synchronously replaces only the MoveIt target proxy, runs the
same geometry/quality/complete-IK/continuous-approach/pregrasp-plan checks as
the real plan-only node, and records candidates that require a later cold-world
plan-only confirmation.
"""

import datetime
import json
import os
import sys

import moveit_commander
import numpy as np
import rospy

import rospkg


PACKAGE_PATH = rospkg.RosPack().get_path("handarm_sim_demo")
sys.path.insert(0, os.path.join(PACKAGE_PATH, "scripts"))

from grasp_candidate_quality import evaluate_candidate_quality
from grasp_geometry import transform
from grasp_pose_planner import ThreeFingerPosePlanner, matrix_pose


def validate_probes(value):
    if not isinstance(value, list) or not value:
        raise ValueError("probes must be a non-empty list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("probe {} must contain exactly x,y".format(index))
        xy = np.asarray(item, dtype=float)
        if not np.all(np.isfinite(xy)):
            raise ValueError("probe {} must be finite".format(index))
        # Table center/size and the large object's half extents.
        if xy[0] < 0.285 or xy[0] > 1.015 or abs(xy[1]) > 0.410:
            raise ValueError("probe {} places the object outside the table".format(index))
        result.append([float(xy[0]), float(xy[1])])
    if len({tuple(item) for item in result}) != len(result):
        raise ValueError("probes must be unique")
    return result


class GraspXYIKProbe:
    def __init__(self):
        self.planner = ThreeFingerPosePlanner()
        if self.planner.grasp_family != "top_down":
            raise ValueError("virtual XY probe requires grasp_family=top_down")
        self.probes = validate_probes(rospy.get_param("~probes"))
        self.object_z = float(rospy.get_param("~object_center_z_m", 0.44))
        self.results_dir = rospy.get_param(
            "~results_dir",
            os.path.abspath(
                os.path.join(PACKAGE_PATH, "..", "..", "results", "sim_baseline")
            ),
        )

    def run(self):
        self.planner.wait_ready()
        rows = []
        for index, (x_value, y_value) in enumerate(self.probes):
            T_world_object = transform(
                np.eye(3), [x_value, y_value, self.object_z]
            )
            pose = matrix_pose(T_world_object)
            sync_position, sync_orientation = self.planner.synchronize_exact_object(
                pose
            )
            self.planner._T_world_object = T_world_object
            candidates = self.planner.geometry.coarse_geometry_candidates(
                T_world_object,
                self.planner.target_spec["size"],
                self.planner.table_z,
                "top_down",
            )
            selected, moveit, failure = self.planner.select_candidate(candidates)
            row = {
                "index": index,
                "probe_xy_m": [x_value, y_value],
                "T_world_object": T_world_object.tolist(),
                "candidate_count": len(candidates),
                "geometry_valid_count": sum(item.enclosure.valid for item in candidates),
                "planning_scene_position_error_m": sync_position,
                "planning_scene_orientation_error_deg": sync_orientation,
                "success": selected is not None,
                "failure": failure,
                "selected_candidate": selected.as_dict() if selected else None,
                "selected_quality": (
                    evaluate_candidate_quality(
                        selected, self.planner.geometry_config
                    ).as_dict()
                    if selected
                    else None
                ),
                "moveit": moveit,
                "robot_executed": False,
            }
            rows.append(row)
            rospy.loginfo(
                "[xy-ik-probe] xy=(%.3f, %.3f) success=%s failure=%s",
                x_value,
                y_value,
                row["success"],
                failure.get("failure_reason", ""),
            )
        record = {
            "schema_version": 1,
            "mode": "virtual_target_xy_plan_only_coarse_screen",
            "virtual_target_probe": True,
            "gazebo_target_untouched": True,
            "contact_forbidden": True,
            "robot_executed": False,
            "object_size_m": list(self.planner.target_spec["size"]),
            "object_center_z_m": self.object_z,
            "grasp_family": "top_down",
            "rows": rows,
            "cold_world_confirmation_required": True,
        }
        os.makedirs(self.results_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        path = os.path.join(
            self.results_dir, "three_finger_virtual_xy_probe_{}.json".format(stamp)
        )
        with open(path, "x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
        rospy.loginfo("[xy-ik-probe] results=%s", path)
        return record, path


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("three_finger_xy_ik_probe")
    try:
        GraspXYIKProbe().run()
    except Exception as exc:
        rospy.logfatal("Virtual XY IK probe failed: %s", exc)
        raise SystemExit(8)
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
