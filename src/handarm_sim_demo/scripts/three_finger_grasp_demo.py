#!/usr/bin/env python3
"""Execute only approach and verified three-finger physical contact.

This development node consumes the object-relative plan produced by
ThreeFingerPosePlanner.  Success requires simultaneous, stable target-object
contact from exactly f1/f2/f3.  It never lifts, places, attaches or releases a
successful grasp; those phases remain disabled until this contact gate passes.
"""

import copy
import datetime
import json
import math
import os
import re
import sys
import threading
import time

import moveit_commander
import rospkg
import rospy
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Pose


PACKAGE_PATH = rospkg.RosPack().get_path("handarm_sim_demo")
sys.path.insert(0, os.path.join(PACKAGE_PATH, "scripts"))

from grasp_pose_planner import (
    ThreeFingerPosePlanner,
    matrix_pose,
    pose_matrix,
    position_distance,
    quaternion_distance_deg,
)
from hand_commander import HandCommander


FINGER_LINK_PATTERN = re.compile(r"(?:^|::)f([123])link[123](?:::|$)")


def pose_as_dict(pose):
    return {
        "position_m": [pose.position.x, pose.position.y, pose.position.z],
        "orientation_xyzw": [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    }


def has_target_table_support(states, target_model, table_model="work_table"):
    target_prefix = target_model + "::"
    table_prefix = table_model + "::"
    for state in states:
        first = str(state.collision1_name)
        second = str(state.collision2_name)
        if (target_prefix in first and table_prefix in second) or (
            target_prefix in second and table_prefix in first
        ):
            return True
    return False


def classify_contacts(states, target_model):
    families, pairs, unexpected = set(), set(), set()
    target_prefix = target_model + "::"
    for state in states:
        first = str(state.collision1_name)
        second = str(state.collision2_name)
        if target_prefix not in first and target_prefix not in second:
            continue
        other = second if target_prefix in first else first
        pair = "{} <-> {}".format(first, second)
        match = FINGER_LINK_PATTERN.search(other)
        if match is not None:
            families.add("f{}".format(match.group(1)))
            pairs.add(pair)
        elif "work_table::" not in other:
            unexpected.add(pair)
    return families, pairs, unexpected


def contact_obstruction_candidate(result):
    return (
        not bool(result.get("success"))
        and result.get("failure_reason")
        in {
            "active joint verification failed",
            "mimic joint relation verification failed",
        }
    )


class ThreeFingerContactDemo:
    def __init__(self):
        self.planner = ThreeFingerPosePlanner()
        self.hand = HandCommander(
            rospy.get_param(
                "~hand_config",
                os.path.join(PACKAGE_PATH, "config", "hand_commands.yaml"),
            )
        )
        runtime = self.planner.geometry_config["runtime_acceptance"]
        self.required = set(runtime["required_actual_finger_families"])
        if self.required != {"f1", "f2", "f3"}:
            raise ValueError("required actual contacts must be exactly f1/f2/f3")
        self.contact_stability_s = float(runtime["contact_stability_s"])
        self.contact_wait_timeout_s = float(runtime["contact_wait_timeout_s"])
        self._lock = threading.Lock()
        self._latest_monotonic = None
        self._latest_families = set()
        self._all_families = set()
        self._pairs = set()
        self._unexpected = set()
        self._qualifying_since = None
        self._best_duration_s = 0.0
        self._target_table_support = False
        self.subscriber = rospy.Subscriber(
            "/handarm_sim_demo/target_contacts",
            ContactsState,
            self._contact_callback,
            queue_size=50,
        )

    def _contact_callback(self, message):
        now = time.monotonic()
        families, pairs, unexpected = classify_contacts(
            message.states, self.planner.target_name
        )
        target_table_support = has_target_table_support(
            message.states, self.planner.target_name
        )
        with self._lock:
            self._latest_monotonic = now
            self._latest_families = families
            self._all_families.update(families)
            self._pairs.update(pairs)
            self._unexpected.update(unexpected)
            self._target_table_support = target_table_support
            if families == self.required and not unexpected:
                if self._qualifying_since is None:
                    self._qualifying_since = now
                self._best_duration_s = max(
                    self._best_duration_s, now - self._qualifying_since
                )
            else:
                self._qualifying_since = None

    def reset_contacts(self):
        with self._lock:
            self._latest_monotonic = None
            self._latest_families = set()
            self._all_families = set()
            self._pairs = set()
            self._unexpected = set()
            self._qualifying_since = None
            self._best_duration_s = 0.0

    def contact_snapshot(self):
        with self._lock:
            current_duration = (
                0.0
                if self._qualifying_since is None
                else time.monotonic() - self._qualifying_since
            )
            return {
                "latest_monotonic": self._latest_monotonic,
                "latest_families": set(self._latest_families),
                "all_families": set(self._all_families),
                "pairs": set(self._pairs),
                "unexpected": set(self._unexpected),
                "best_duration_s": self._best_duration_s,
                "current_duration_s": current_duration,
                "target_table_support": self._target_table_support,
            }

    def wait_for_three_finger_contact(self):
        deadline = time.monotonic() + self.contact_wait_timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            snapshot = self.contact_snapshot()
            if (
                snapshot["current_duration_s"] >= self.contact_stability_s
                and snapshot["latest_families"] == self.required
                and not snapshot["unexpected"]
            ):
                return snapshot
            rospy.sleep(0.02)
        snapshot = self.contact_snapshot()
        missing = sorted(self.required - snapshot["all_families"])
        raise RuntimeError(
            "THREE_FINGER_CONTACT_NOT_ACHIEVED: missing={} all={} "
            "latest={} stable={:.3f}s unexpected={}".format(
                missing,
                sorted(snapshot["all_families"]),
                sorted(snapshot["latest_families"]),
                snapshot["best_duration_s"],
                sorted(snapshot["unexpected"]),
            )
        )

    def close_until_three_finger_contact(self):
        """Run CLOSE concurrently and stop at stable physical enclosure."""
        result_holder = {}

        def command_worker():
            try:
                result_holder["result"] = self.hand.command("GRASP")
            except Exception as exc:
                result_holder["exception"] = exc

        self.reset_contacts()
        worker = threading.Thread(target=command_worker, daemon=True)
        worker.start()
        timeout_s = float(
            self.planner.geometry_config["runtime_acceptance"]
            ["contact_limited_close_timeout_s"]
        )
        deadline = time.monotonic() + timeout_s
        acquired = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            snapshot = self.contact_snapshot()
            if (
                snapshot["current_duration_s"] >= self.contact_stability_s
                and snapshot["latest_families"] == self.required
                and not snapshot["unexpected"]
            ):
                acquired = snapshot
                # Contact, not a blind terminal angle, ends flexion.  The
                # trajectory controller then generates its normal stop ramp.
                self.hand.client.cancel_goal()
                break
            if not worker.is_alive():
                break
            rospy.sleep(0.01)
        if acquired is None:
            self.hand.client.cancel_goal()
        worker.join(timeout=3.0)
        if worker.is_alive():
            raise RuntimeError("contact-limited CLOSE worker did not stop")
        if "exception" in result_holder:
            raise RuntimeError(
                "contact-limited CLOSE raised: {}".format(
                    result_holder["exception"]
                )
            )
        if acquired is None:
            snapshot = self.contact_snapshot()
            missing = sorted(self.required - snapshot["all_families"])
            raise RuntimeError(
                "CONTACT_LIMITED_CLOSE_FAILED: missing={} all={} latest={} "
                "best={:.3f}s".format(
                    missing,
                    sorted(snapshot["all_families"]),
                    sorted(snapshot["latest_families"]),
                    snapshot["best_duration_s"],
                )
            )
        stopped = self.hand.command("STOP")
        if not stopped["success"]:
            raise RuntimeError(
                "contact-limited STOP failed: {}".format(
                    stopped["failure_reason"]
                )
            )
        # Prove that the controller stop/hold state still encloses the object;
        # transient contact during a moving trajectory is not acceptance.
        self.reset_contacts()
        held = self.wait_for_three_finger_contact()
        return acquired, held, result_holder.get("result", {}), stopped

    def execute_pregrasp(self, candidate):
        trajectory = self.planner._planned_pregrasp_trajectory
        if not trajectory.joint_trajectory.points:
            raise RuntimeError("pregrasp trajectory is empty")
        if not self.planner.group.execute(trajectory, wait=True):
            raise RuntimeError("pregrasp execution returned false")
        self.planner.group.stop()
        actual = self.planner.group.get_current_pose(
            self.planner.end_effector_link
        ).pose
        target = matrix_pose(
            self.planner.selected_ik_metrics["pregrasp_T_world_tool0"]
        )
        position_error = position_distance(actual.position, target.position)
        orientation_error = quaternion_distance_deg(
            actual.orientation, target.orientation
        )
        runtime = self.planner.geometry_config["runtime_acceptance"]
        if position_error > float(runtime["pregrasp_position_tolerance_m"]):
            raise RuntimeError(
                "pregrasp position error {:.4f}m".format(position_error)
            )
        if orientation_error > float(
            runtime["pregrasp_orientation_tolerance_deg"]
        ):
            raise RuntimeError(
                "pregrasp orientation error {:.3f}deg".format(
                    orientation_error
                )
            )
        return position_error, orientation_error

    def execute_approach(self, candidate):
        runtime = self.planner.geometry_config["runtime_acceptance"]
        target = matrix_pose(candidate.T_world_tool0)
        trajectory, fraction = self.planner.group.compute_cartesian_path(
            [target], float(runtime["cartesian_eef_step_m"]), True
        )
        if fraction < float(runtime["cartesian_fraction_min"]):
            raise RuntimeError(
                "APPROACH_INCOMPLETE: Cartesian fraction {:.6f}".format(
                    fraction
                )
            )
        trajectory = self.planner.group.retime_trajectory(
            self.planner.robot.get_current_state(),
            trajectory,
            velocity_scaling_factor=float(runtime["approach_velocity_scaling"]),
            acceleration_scaling_factor=float(
                runtime["approach_acceleration_scaling"]
            ),
            algorithm="iterative_time_parameterization",
        )
        if not trajectory.joint_trajectory.points:
            raise RuntimeError("approach trajectory is empty after retiming")
        if not self.planner.group.execute(trajectory, wait=True):
            raise RuntimeError("approach execution returned false")
        self.planner.group.stop()
        return {
            "approach_fraction": fraction,
            "approach_trajectory_points": len(
                trajectory.joint_trajectory.points
            ),
            "approach_trajectory_duration_s": trajectory.joint_trajectory.points[
                -1
            ].time_from_start.to_sec(),
        }

    def run(self):
        plan_record, plan_path = self.planner.run()
        candidate = self.planner.selected_candidate
        states = ["PLAN_ONLY_PASS", "RELEASE_HAND"]
        record = {
            "schema_version": 1,
            "mode": "three_finger_contact_only",
            "plan_result": plan_path,
            "selected_candidate": candidate.as_dict(),
            "states": states,
            "success": False,
            "lift_executed": False,
            "place_executed": False,
            "attachment_used": False,
            "failure_reason": "",
        }
        object_before = self.planner.get_model_state(
            self.planner.target_name, "world"
        ).pose
        try:
            opened = self.hand.command("RELEASE")
            record["release_success"] = bool(opened["success"])
            if not opened["success"]:
                raise RuntimeError("RELEASE failed: {}".format(opened["failure_reason"]))
            states.append("EXECUTE_PREGRASP")
            (
                record["pregrasp_position_error_m"],
                record["pregrasp_orientation_error_deg"],
            ) = self.execute_pregrasp(candidate)
            object_preapproach = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["object_pose_before_approach"] = pose_as_dict(
                object_preapproach
            )
            record["object_pregrasp_displacement_m"] = position_distance(
                object_before.position, object_preapproach.position
            )
            runtime = self.planner.geometry_config["runtime_acceptance"]
            if record["object_pregrasp_displacement_m"] > float(
                runtime["maximum_pregrasp_object_displacement_m"]
            ):
                raise RuntimeError("pregrasp transit disturbed target object")
            states.append("EXECUTE_APPROACH")
            record.update(self.execute_approach(candidate))
            object_after_approach = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["object_pose_after_approach"] = pose_as_dict(
                object_after_approach
            )
            record["object_approach_displacement_m"] = position_distance(
                object_preapproach.position, object_after_approach.position
            )
            if record["object_approach_displacement_m"] > float(
                runtime["maximum_approach_object_displacement_m"]
            ):
                raise RuntimeError("approach disturbed target object")
            states.append("CONTACT_LIMITED_GRASP_HAND")
            acquired_contact, contact, closed, stopped_after_contact = (
                self.close_until_three_finger_contact()
            )
            object_after_grasp = self.planner.get_model_state(
                self.planner.target_name, "world"
            ).pose
            record["object_pose_after_grasp"] = pose_as_dict(object_after_grasp)
            record["grasp_command_success"] = bool(closed["success"])
            record["grasp_command_failure_reason"] = closed.get(
                "failure_reason", ""
            )
            record["hand_target_joint_positions"] = closed.get(
                "target_joint_positions"
            )
            record["hand_actual_joint_positions"] = closed.get(
                "actual_joint_positions"
            )
            record["contact_limited_stop"] = stopped_after_contact
            record["contact_acquisition_families"] = sorted(
                acquired_contact["latest_families"]
            )
            record["contact_acquisition_stability_s"] = acquired_contact[
                "current_duration_s"
            ]
            record["object_grasp_displacement_m"] = position_distance(
                object_after_approach.position, object_after_grasp.position
            )
            if record["object_grasp_displacement_m"] > float(
                runtime["maximum_grasp_object_displacement_m"]
            ):
                raise RuntimeError("grasp displaced target object too far")
            states.append("VERIFY_HELD_THREE_FINGER_CONTACT")
            record["actual_contact_families"] = sorted(
                contact["latest_families"]
            )
            record["all_observed_contact_families"] = sorted(
                contact["all_families"]
            )
            record["contact_pairs"] = sorted(contact["pairs"])
            record["unexpected_target_contacts"] = sorted(
                contact["unexpected"]
            )
            record["contact_stability_s"] = contact["current_duration_s"]
            states.append("STOP_HAND")
            stopped = self.hand.command("STOP")
            if not stopped["success"]:
                raise RuntimeError("STOP failed: {}".format(stopped["failure_reason"]))
            states.append("THREE_FINGER_CONTACT_ONLY_PASS")
            record["success"] = True
        except Exception as exc:
            self.planner.group.stop()
            try:
                self.hand.command("RELEASE")
                self.hand.command("STOP")
            except Exception as release_exc:
                rospy.logerr("Fail-safe hand release also failed: %s", release_exc)
            states.append("FAILED")
            record["failure_reason"] = str(exc)
        record["states"] = states
        record["object_initial_xyz_m"] = [
            object_before.position.x,
            object_before.position.y,
            object_before.position.z,
        ]
        os.makedirs(self.planner.results_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        path = os.path.join(
            self.planner.results_dir,
            "three_finger_contact_only_{}.json".format(stamp),
        )
        with open(path, "x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
        self.planner.status.publish(json.dumps(record, sort_keys=True))
        if not record["success"]:
            raise RuntimeError(record["failure_reason"])
        rospy.loginfo(
            "[three-finger-contact] PASS actual f1/f2/f3 contact; results=%s",
            path,
        )
        return record, path


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("three_finger_grasp_contact_demo")
    try:
        demo = ThreeFingerContactDemo()
        demo.run()
        rospy.loginfo(
            "[three-finger-contact] Contact pose is held for observation; Ctrl-C to exit."
        )
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("Three-finger contact-only failed: %s", exc)
        raise SystemExit(8)
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
