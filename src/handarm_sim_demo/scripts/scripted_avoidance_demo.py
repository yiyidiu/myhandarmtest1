#!/usr/bin/env python3
"""Fail-closed static-obstacle avoidance state machine and recorder."""

import csv
import datetime
import json
import math
import os
import sys
import time

import actionlib
import moveit_commander
import rospkg
import rospy
import yaml
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveGroupAction
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from std_msgs.msg import Bool, String


EXPECTED_CONTROLLERS = {"controller_gazebo", "controller_gazebo_hand"}
CSV_FIELDS = [
    "run_id", "scenario", "trial", "state", "expected_outcome",
    "outcome_pass", "plan_success", "planning_time_s", "planning_wall_time_s",
    "trajectory_points", "trajectory_collision_free", "straight_path_invalid_samples",
    "execution_attempted", "execution_success", "final_position_error_m",
    "final_orientation_error_deg", "collision_object_count", "return_home_success",
    "failure_reason", "states",
]


def quaternion_angle_deg(a, b):
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


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


class AvoidanceDemo:
    def __init__(self):
        self.scene_path = rospy.get_param("~scene_config")
        self.config_path = rospy.get_param("~avoidance_config")
        self.startup_path = rospy.get_param("~startup_config")
        self.scenario = rospy.get_param("~scenario")
        self.repetitions = int(rospy.get_param("~repetitions", 1))
        with open(self.scene_path, "r", encoding="utf-8") as stream:
            self.scene_config = yaml.safe_load(stream)
        with open(self.config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        with open(self.startup_path, "r", encoding="utf-8") as stream:
            self.startup = yaml.safe_load(stream)["initial_configuration"]
        if self.scenario not in self.scene_config["avoidance_goals"]:
            raise ValueError("unknown avoidance scenario {}".format(self.scenario))
        self.goal_spec = self.scene_config["avoidance_goals"][self.scenario]
        self.expected_outcome = self.goal_spec["expected_outcome"]
        package = rospkg.RosPack().get_path("handarm_sim_demo")
        default_results = os.path.abspath(
            os.path.join(package, "..", "..", "results", "sim_baseline")
        )
        self.results_dir = rospy.get_param("~results_dir", default_results)
        self.status_pub = rospy.Publisher(
            "/handarm_sim_demo/avoidance_status", String, queue_size=1, latch=True
        )
        move_group_client = actionlib.SimpleActionClient(
            "/move_group", MoveGroupAction
        )
        rospy.loginfo(
            "[avoidance] Waiting for MoveIt /move_group (cold GUI start can take 20-30 s)..."
        )
        if not move_group_client.wait_for_server(rospy.Duration(90.0)):
            raise RuntimeError("move_group action did not become ready")
        rospy.loginfo("[avoidance] MoveIt action is available.")
        self.group = moveit_commander.MoveGroupCommander(
            self.config["planning_group"]
        )
        self.robot = moveit_commander.RobotCommander()
        self.check_state = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity
        )
        self.group.set_end_effector_link(self.config["end_effector_link"])
        self.group.set_planner_id(self.config["planner_id"])
        self.group.set_planning_time(float(self.config["planning_time_s"]))
        self.group.set_num_planning_attempts(int(self.config["planning_attempts"]))
        self.group.set_max_velocity_scaling_factor(
            float(self.config["velocity_scaling"])
        )
        self.group.set_max_acceleration_scaling_factor(
            float(self.config["acceleration_scaling"])
        )
        self.group.set_goal_position_tolerance(
            float(self.config["goal_position_tolerance_m"])
        )
        self.group.set_goal_orientation_tolerance(
            float(self.config["goal_orientation_tolerance_rad"])
        )

    def wait_ready(self):
        rospy.loginfo("[avoidance] Waiting for startup and synchronized scene...")
        for topic in (
            "/handarm_sim_demo/startup_ready",
            "/handarm_sim_demo/scene_ready",
        ):
            if not rospy.wait_for_message(topic, Bool, timeout=90.0).data:
                raise RuntimeError("{} reported false".format(topic))
        rospy.wait_for_service("/controller_manager/list_controllers", timeout=30.0)
        rospy.wait_for_service("/check_state_validity", timeout=30.0)
        controllers = rospy.ServiceProxy(
            "/controller_manager/list_controllers", ListControllers
        )().controller
        states = {item.name: item.state for item in controllers}
        if not EXPECTED_CONTROLLERS.issubset(states) or any(
            states[name] != "running" for name in EXPECTED_CONTROLLERS
        ):
            raise RuntimeError("trajectory controllers are not running")
        scene_status = json.loads(
            rospy.wait_for_message(
                "/handarm_sim_demo/scene_status", String, timeout=10.0
            ).data
        )
        if not scene_status.get("ready") or scene_status.get("scenario") != self.scenario:
            raise RuntimeError("scene status does not match requested scenario")
        rospy.loginfo(
            "[avoidance] READY: scenario=%s, collision objects=%d.",
            self.scenario,
            len(scene_status["objects"]),
        )
        return len(scene_status["objects"])

    @staticmethod
    def unpack_plan(result):
        if isinstance(result, tuple):
            success, trajectory, planning_time, error_code = result
            return bool(success), trajectory, float(planning_time), int(error_code.val)
        trajectory = result
        return bool(trajectory.joint_trajectory.points), trajectory, None, None

    def validate_plan(self, trajectory):
        points = trajectory.joint_trajectory.points
        names = trajectory.joint_trajectory.joint_names
        if not points or not names:
            return False, "empty trajectory"
        previous_time = -1.0
        state = self.robot.get_current_state()
        positions = list(state.joint_state.position)
        index = {name: i for i, name in enumerate(state.joint_state.name)}
        if any(name not in index for name in names):
            return False, "trajectory contains unknown joint"
        for point in points:
            seconds = point.time_from_start.to_sec()
            if not math.isfinite(seconds) or seconds <= previous_time:
                return False, "trajectory time is not strictly increasing"
            previous_time = seconds
            if len(point.positions) != len(names) or not all(
                math.isfinite(value) for value in point.positions
            ):
                return False, "trajectory positions are invalid"
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

    def straight_path_invalid_samples(self, trajectory, samples=101):
        names = trajectory.joint_trajectory.joint_names
        final = trajectory.joint_trajectory.points[-1].positions
        current = dict(
            zip(self.group.get_active_joints(), self.group.get_current_joint_values())
        )
        start = [current[name] for name in names]
        state = self.robot.get_current_state()
        positions = list(state.joint_state.position)
        index = {name: i for i, name in enumerate(state.joint_state.name)}
        invalid = 0
        for sample in range(samples):
            alpha = sample / float(samples - 1)
            for name, begin, end in zip(names, start, final):
                positions[index[name]] = begin + alpha * (end - begin)
            state.joint_state.position = positions
            if not self.check_state(
                GetStateValidityRequest(
                    robot_state=state, group_name=self.config["planning_group"]
                )
            ).valid:
                invalid += 1
        return invalid

    def home_target(self):
        names = self.startup["arm"]["joint_names"]
        target = list(self.startup["arm"]["positions"])
        current = dict(zip(self.group.get_active_joints(), self.group.get_current_joint_values()))
        for i, name in enumerate(names):
            if name in self.startup.get("wraparound_joints", []):
                target[i] += round((current[name] - target[i]) / (2.0 * math.pi)) * (
                    2.0 * math.pi
                )
        return names, target

    def move_home(self):
        names, target = self.home_target()
        current = dict(zip(self.group.get_active_joints(), self.group.get_current_joint_values()))
        if max(abs(current[n] - value) for n, value in zip(names, target)) <= float(
            self.config["home_joint_tolerance_rad"]
        ):
            return True
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(dict(zip(names, target)))
        success, plan, _, _ = self.unpack_plan(self.group.plan())
        valid, _ = self.validate_plan(plan) if success else (False, "planning failed")
        if not success or not valid:
            return False
        executed = bool(self.group.execute(plan, wait=True))
        self.group.stop()
        if not executed:
            return False
        current = dict(zip(self.group.get_active_joints(), self.group.get_current_joint_values()))
        return max(abs(current[n] - value) for n, value in zip(names, target)) <= float(
            self.config["home_joint_tolerance_rad"]
        )

    def run_trial(self, trial, collision_object_count):
        states = ["INIT", "WAIT_FOR_ROBOT", "WAIT_FOR_CONTROLLERS", "WAIT_FOR_SCENE"]
        row = {field: None for field in CSV_FIELDS}
        row.update(
            run_id=self.run_id,
            scenario=self.scenario,
            trial=trial,
            expected_outcome=self.expected_outcome,
            outcome_pass=False,
            plan_success=False,
            trajectory_points=0,
            trajectory_collision_free=False,
            straight_path_invalid_samples=0,
            execution_attempted=False,
            execution_success=False,
            collision_object_count=collision_object_count,
            return_home_success=False,
            failure_reason="",
        )
        try:
            rospy.loginfo(
                "[avoidance] Trial %d/%d: moving to home position...",
                trial,
                self.repetitions,
            )
            states.append("MOVE_HOME")
            if not self.move_home():
                raise RuntimeError("MOVE_HOME failed")
            states.extend(["SET_GOAL", "PLAN"])
            rospy.loginfo("[avoidance] Trial %d: planning collision-free path...", trial)
            goal = make_pose(self.goal_spec)
            self.group.stop()
            self.group.clear_pose_targets()
            self.group.set_start_state_to_current_state()
            self.group.set_pose_target(goal, self.config["end_effector_link"])
            start_wall = time.monotonic()
            result = self.group.plan()
            row["planning_wall_time_s"] = time.monotonic() - start_wall
            success, plan, planning_time, _ = self.unpack_plan(result)
            row["planning_time_s"] = (
                planning_time
                if success
                and planning_time is not None
                and math.isfinite(planning_time)
                and planning_time >= 0.0
                else None
            )
            row["plan_success"] = success
            row["trajectory_points"] = len(plan.joint_trajectory.points)
            if not success or not plan.joint_trajectory.points:
                raise RuntimeError("planning failed or returned empty trajectory")
            states.append("VALIDATE_PLAN")
            valid, reason = self.validate_plan(plan)
            row["trajectory_collision_free"] = valid
            if not valid:
                raise RuntimeError(reason)
            row["straight_path_invalid_samples"] = self.straight_path_invalid_samples(plan)
            if self.expected_outcome == "safe_failure":
                raise RuntimeError("unexpected valid plan in safe-failure scenario")
            states.append("EXECUTE")
            rospy.loginfo(
                "[avoidance] Trial %d: executing %d trajectory points...",
                trial,
                row["trajectory_points"],
            )
            row["execution_attempted"] = True
            row["execution_success"] = bool(self.group.execute(plan, wait=True))
            self.group.stop()
            if not row["execution_success"]:
                raise RuntimeError("trajectory execution returned false")
            states.append("VERIFY_GOAL")
            actual = self.group.get_current_pose(self.config["end_effector_link"]).pose
            row["final_position_error_m"] = math.sqrt(
                (actual.position.x - goal.position.x) ** 2
                + (actual.position.y - goal.position.y) ** 2
                + (actual.position.z - goal.position.z) ** 2
            )
            row["final_orientation_error_deg"] = quaternion_angle_deg(
                actual.orientation, goal.orientation
            )
            if row["final_position_error_m"] > float(
                self.config["verification_position_tolerance_m"]
            ) or row["final_orientation_error_deg"] > float(
                self.config["verification_orientation_tolerance_deg"]
            ):
                raise RuntimeError("final end-effector verification failed")
            states.append("RETURN_HOME")
            rospy.loginfo("[avoidance] Trial %d: goal reached; returning home...", trial)
            row["return_home_success"] = self.move_home()
            if not row["return_home_success"]:
                raise RuntimeError("RETURN_HOME failed")
            states.append("DONE")
            row["state"] = "DONE"
            row["outcome_pass"] = True
            rospy.loginfo("[avoidance] Trial %d DONE.", trial)
        except Exception as exc:
            self.group.stop()
            self.group.clear_pose_targets()
            states.append("FAILED")
            row["state"] = "FAILED"
            row["failure_reason"] = str(exc)
            row["outcome_pass"] = (
                self.expected_outcome == "safe_failure"
                and not row["execution_attempted"]
                and not row["execution_success"]
            )
            rospy.logerr("[avoidance] Trial %d FAILED: %s", trial, exc)
        row["states"] = json.dumps(states, separators=(",", ":"))
        return row

    def save(self, rows):
        os.makedirs(self.results_dir, exist_ok=True)
        json_path = os.path.join(
            self.results_dir,
            "avoidance_{}_{}.json".format(self.scenario, self.run_id),
        )
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "run_id": self.run_id,
                    "scenario": self.scenario,
                    "results": rows,
                    "all_outcomes_pass": all(row["outcome_pass"] for row in rows),
                },
                stream,
                indent=2,
                sort_keys=True,
            )
        csv_path = os.path.join(self.results_dir, "avoidance_trials.csv")
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        return json_path, csv_path

    def run(self):
        self.run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        collision_count = self.wait_ready()
        rows = []
        for trial in range(1, self.repetitions + 1):
            row = self.run_trial(trial, collision_count)
            rows.append(row)
            self.status_pub.publish(json.dumps(row, sort_keys=True))
            if not row["outcome_pass"]:
                break
        json_path, csv_path = self.save(rows)
        rospy.loginfo("[avoidance] DONE. Results: %s and %s", json_path, csv_path)
        rospy.loginfo(
            "[avoidance] Gazebo/RViz remain open. Press Ctrl-C in this terminal when finished."
        )
        return all(row["outcome_pass"] for row in rows) and len(rows) == self.repetitions


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("scripted_avoidance_demo")
    try:
        success = AvoidanceDemo().run()
    except Exception as exc:
        rospy.logfatal("Avoidance demo setup failed: %s", exc)
        raise SystemExit(5)
    raise SystemExit(0 if success else 5)


if __name__ == "__main__":
    main()
