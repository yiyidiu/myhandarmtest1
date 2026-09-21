"""Bounded, operator-cancellable Cartesian detour around the robot itself.

Only supplies temporary pose waypoints to the existing pose servo. It never
executes a joint trajectory, changes C, or bypasses the live collision guard.
"""
import copy
import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import JointConstraint
from moveit_msgs.srv import (
    GetCartesianPath, GetCartesianPathRequest,
    GetPlanningScene, GetPlanningSceneRequest,
    GetPositionFK, GetPositionFKRequest,
    GetPositionIK, GetPositionIKRequest,
)
from .shared_teleop_core import matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix, so3_log, so3_exp


class SelfDetour:
    def __init__(self, enabled, joint_limits):
        self.enabled = enabled
        self.joint_limits = joint_limits
        self.lock = threading.Lock()
        self.generation = 0
        self.revision = None
        self.worker_running = False
        self.result = None
        self.active = None
        self.failed_goal = None
        self.blocked_since = None
        self.status = "DIRECT"
        self.last_planning_ms = 0.0

    @staticmethod
    def changed(a, b):
        return (np.linalg.norm(a[0]-b[0]) > .020 or
                np.linalg.norm(so3_log(a[1] @ b[1].T)) > math.radians(15))

    def cancel(self, status="DIRECT"):
        with self.lock:
            self.generation += 1
            self.result = None
            self.active = None
            self.blocked_since = None
            self.status = status

    def target(self, goal_p, goal_r, current_p, current_r, feedback,
               feedback_fresh, revision, observation_valid):
        if not self.enabled:
            return None
        now = time.monotonic()
        goal = (goal_p.copy(), goal_r.copy())
        if revision != self.revision:
            self.cancel()
            self.failed_goal = None
            self.revision = revision
        if not observation_valid:
            self.cancel("CANCELLED_INPUT_UNAVAILABLE")
            return None
        with self.lock:
            result, self.result = self.result, None
            if result is not None:
                generation, planned_goal, waypoints, start_p, why = result
                if generation == self.generation and not self.changed(goal, planned_goal):
                    if waypoints and np.linalg.norm(current_p-start_p) < .015:
                        self.active = dict(goal=planned_goal, points=waypoints, index=0, start=now)
                    elif waypoints:
                        # A valid path became stale while the robot moved.
                        # Do not cache this as an impossible human target.
                        self.failed_goal = None
                        self.status = "REPLAN_ROBOT_MOVED"
                    else:
                        self.failed_goal = planned_goal
                        self.status = why
                elif generation == self.generation:
                    self.status = "CANCELLED_NEW_HAND_TARGET"
            active = self.active
        if active is not None:
            if self.changed(goal, active["goal"]):
                self.cancel("CANCELLED_NEW_HAND_TARGET")
                self.failed_goal = None
                return None
            if now-active["start"] > 8.0:
                self.failed_goal = active["goal"]
                self.cancel("DETOUR_NOT_COMPLETED")
                return None
            index = active["index"]
            # The final waypoint follows small, fresh hand adjustments; the
            # existing swept guard checks every executed command again.
            last = len(active["points"]) - 1
            p, r = goal if index == last else active["points"][index]
            if (np.linalg.norm(p-current_p) < .003 and
                    np.linalg.norm(so3_log(r @ current_r.T)) < math.radians(2)):
                active["index"] += 1
                if active["index"] == len(active["points"]):
                    self.cancel("DETOUR_COMPLETE")
                    return None
                index = active["index"]
                p, r = goal if index == last else active["points"][index]
            self.status = ("DETOUR_RETURN_TO_HAND_TARGET" if index == last else
                           "DETOUR_CLEAR_SELF" if index == 0 else "DETOUR_FOLLOW_CHECKED_POSES")
            return p.copy(), r.copy()
        if self.worker_running:
            self.status = "CHECKING_SHORT_SELF_DETOUR"
            return None
        if self.failed_goal is not None:
            if not self.changed(goal, self.failed_goal):
                return None
            self.failed_goal = None
        requested = np.asarray(feedback.get("requested_velocity", [0.]*6))
        approved = np.asarray(feedback.get("approved_velocity", [0.]*6))
        blocked = (feedback_fresh and feedback.get("input_valid", False) and
                   0 <= feedback.get("self_distance_m", float("inf")) < .012 and
                   np.linalg.norm(requested[3:]) > .25 and
                   np.linalg.norm(approved[3:]) < .08 and
                   np.linalg.norm(approved[3:]) < .15*np.linalg.norm(requested[3:]) and
                   np.linalg.norm(so3_log(goal_r @ current_r.T)) > math.radians(10))
        if not blocked:
            self.blocked_since = None
            if self.status == "CHECKING_SHORT_SELF_DETOUR":
                self.status = "DIRECT"
            return None
        if self.blocked_since is None:
            self.blocked_since = now
        if now-self.blocked_since < .30:
            return None
        self.blocked_since = None
        with self.lock:
            self.worker_running = True
            generation = self.generation
        threading.Thread(target=self._plan, args=(generation, goal), daemon=True).start()
        self.status = "CHECKING_SHORT_SELF_DETOUR"
        return None

    @staticmethod
    def pose(p, r):
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = p
        q = matrix_to_quaternion_xyzw(r)
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
        return msg

    def _plan(self, generation, goal):
        began = time.monotonic()
        waypoints = None
        start_p = np.zeros(3)
        why = "NO_SAFE_SHORT_DETOUR"
        try:
            for name in ("/get_planning_scene", "/compute_fk", "/compute_cartesian_path"):
                rospy.wait_for_service(name, timeout=.20)
            req = GetPlanningSceneRequest()
            req.components.components = 2 | 16 | 32
            scene = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)(req).scene
            if scene.world.collision_objects or scene.world.octomap.octomap.data:
                raise RuntimeError("environment geometry is forbidden in this control mode")
            fk = GetPositionFKRequest()
            fk.header.frame_id = "base_link"
            fk.fk_link_names = ["tool0"]
            fk.robot_state = scene.robot_state
            out = rospy.ServiceProxy("/compute_fk", GetPositionFK)(fk)
            if out.error_code.val != 1:
                raise RuntimeError("current FK unavailable")
            pose = out.pose_stamped[0].pose
            start_p = np.array([pose.position.x, pose.position.y, pose.position.z])
            start_r = quaternion_xyzw_to_matrix([pose.orientation.x, pose.orientation.y,
                                                 pose.orientation.z, pose.orientation.w])
            # One bounded route in the front workspace, not an unconstrained
            # joint-space plan that can move the endpoint tens of centimetres.
            clear_p = start_p + np.array([.04, 0., 0.])
            points = [(clear_p, start_r), (clear_p, goal[1]), goal]
            request = GetCartesianPathRequest()
            request.header.frame_id = "base_link"
            request.group_name = "abbarm"
            request.link_name = "tool0"
            request.start_state = copy.deepcopy(scene.robot_state)
            request.max_step = .005
            request.jump_threshold = 1.0
            request.avoid_collisions = True
            request.waypoints = [self.pose(p, r) for p, r in points]
            for name, (lo, hi) in self.joint_limits.items():
                constraint = JointConstraint()
                constraint.joint_name = name
                constraint.position = math.radians((lo+hi)/2)
                constraint.tolerance_above = constraint.tolerance_below = math.radians((hi-lo)/2)
                constraint.weight = 1.
                request.path_constraints.joint_constraints.append(constraint)
            path = rospy.ServiceProxy("/compute_cartesian_path", GetCartesianPath)(request)
            if path.error_code.val == 1 and path.fraction > .999:
                waypoints = points
                why = "SHORT_DETOUR_CHECKED"
            else:
                # A feasible target can need coordinated arm/wrist motion;
                # the fixed forward/rotate/return route need not exist.
                # Use one seeded IK solution to propose intermediate poses,
                # bounded near the direct operator path. The same Cartesian
                # collision/jump check and live guard still own execution.
                rospy.wait_for_service("/compute_ik", timeout=.20)
                ik = GetPositionIKRequest()
                ik.ik_request.group_name = "abbarm"
                ik.ik_request.ik_link_name = "tool0"
                ik.ik_request.robot_state = copy.deepcopy(scene.robot_state)
                ik.ik_request.pose_stamped.header.frame_id = "base_link"
                ik.ik_request.pose_stamped.pose = self.pose(*goal)
                ik.ik_request.constraints = request.path_constraints
                ik.ik_request.avoid_collisions = True
                ik.ik_request.timeout = rospy.Duration(.10)
                solution = rospy.ServiceProxy("/compute_ik", GetPositionIK)(ik)
                if solution.error_code.val == 1:
                    initial = dict(zip(scene.robot_state.joint_state.name,
                                       scene.robot_state.joint_state.position))
                    final = dict(zip(solution.solution.joint_state.name,
                                     solution.solution.joint_state.position))
                    names = ["joint_%d" % i for i in range(1, 7)]
                    delta = np.array([final[n]-initial[n] for n in names])
                    if np.max(np.abs(delta)) <= math.pi:
                        proposed = []
                        rotation_delta = so3_log(start_r.T @ goal[1])
                        for fraction in (.25, .5, .75):
                            sample = copy.deepcopy(fk)
                            values = sample.robot_state.joint_state.position
                            sample.robot_state.joint_state.position = list(values)
                            for i, name in enumerate(names):
                                index = sample.robot_state.joint_state.name.index(name)
                                sample.robot_state.joint_state.position[index] = initial[name] + fraction*delta[i]
                            result = rospy.ServiceProxy("/compute_fk", GetPositionFK)(sample)
                            if result.error_code.val != 1:
                                break
                            value = result.pose_stamped[0].pose
                            p = np.array([value.position.x, value.position.y, value.position.z])
                            r = quaternion_xyzw_to_matrix([value.orientation.x, value.orientation.y,
                                                           value.orientation.z, value.orientation.w])
                            line_p = (1.-fraction)*start_p + fraction*goal[0]
                            line_r = start_r @ so3_exp(fraction*rotation_delta)
                            if (np.linalg.norm(p-line_p) > .041 or
                                    np.linalg.norm(so3_log(r @ line_r.T)) > math.radians(5.)):
                                break
                            proposed.append((p, r))
                        if len(proposed) == 3:
                            proposed.append(goal)
                            request.waypoints = [self.pose(p, r) for p, r in proposed]
                            path = rospy.ServiceProxy("/compute_cartesian_path", GetCartesianPath)(request)
                            if path.error_code.val == 1 and path.fraction > .999:
                                waypoints = proposed
                                why = "BOUNDED_IK_POSE_DETOUR_CHECKED"
        except Exception as exc:
            why = "SHORT_DETOUR_CHECK_UNAVAILABLE"
            rospy.logwarn_throttle(2., "short self detour unavailable: %s", exc)
        finally:
            with self.lock:
                self.worker_running = False
                self.last_planning_ms = (time.monotonic()-began)*1000.
                self.result = (generation, goal, waypoints, start_p, why)
