#!/usr/bin/env python3
"""Track a robot tool pose from a hand pose relative to an explicit zero."""

import json
import math
import threading
import time

import numpy as np
import rospy
import tf
import yaml
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from std_msgs.msg import Float64, Int8, String
from std_srvs.srv import Empty, Trigger, TriggerResponse

from handarm_moveit_demo.msg import HamerHandPose, HandCommand
from handarm_moveit_demo.shared_teleop_core import (
    AprilTagV3PoseContinuityFilter, CollisionRetreatGuard, PoseSample,
    CameraRangeWorkspaceMapper, GroundSectorWorkspace,
    RelativePoseMapper, RelativePoseServoController,
    SixDofTrendEstimator, StationaryFeedforwardGate,
    SymmetricSideGraspProjector,
    interpolate_pose_ray,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix, so3_log,
)


class SixDofTrendNode:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        trend = config.get("trend", {})
        limits = config.get("limits", {})
        mapping = config.get("mapping", {})
        reference = config.get("reference", {})
        control = config.get("control", {})
        frames = config.get("frames", {})
        self.base_frame = frames.get("base", "base_link")
        self.control_frame = frames.get("servo_control", "tool0")
        self.reference_frame = reference.get(
            "frame", frames.get("camera", "camera_color_optical_frame"))
        self.mapping_profile_name = str(rospy.get_param(
            "~mapping_profile", "current_linear"))
        self.camera_workspace_calibration_status = "NOT_USED"
        self.pose_reachability_enabled = False
        self.pose_reachability_cache = []
        self.pose_reachability_last = {
            "limit": 1.0, "cache_hit": False, "ik_calls": 0,
            "processing_ms": 0.0,
        }
        self.require_confirmation = bool(reference.get("require_confirmation", True))
        self.reference_armed = not self.require_confirmation
        self.reference_ready = False
        self.reference_policy = reference.get(
            "policy", "EXPLICIT_CONFIRM_NEXT_VALID_FRAME")
        self.control_mode = str(control.get(
            "mode", "RELATIVE_POSE_TRACKING"))
        if self.control_mode != "RELATIVE_POSE_TRACKING":
            raise ValueError("control.mode must be RELATIVE_POSE_TRACKING")
        self.pose_continuity = AprilTagV3PoseContinuityFilter(
            maximum_position_step_m=trend.get(
                "maximum_pose_step_m", 0.035),
            maximum_rotation_step_rad=math.radians(trend.get(
                "maximum_pose_rotation_step_deg", 15.0)),
            position_alpha=trend.get("pose_position_lowpass_alpha", 0.30),
            rotation_alpha=trend.get("pose_rotation_lowpass_alpha", 0.28),
            maximum_rotation_innovation_rad=math.radians(trend.get(
                "maximum_pose_rotation_innovation_deg", 35.0)),
            maximum_position_innovation_m=trend.get(
                "maximum_pose_position_innovation_m", float("inf")),
            position_alpha_max=trend.get(
                "pose_position_lowpass_alpha_max"),
            rotation_alpha_max=trend.get(
                "pose_rotation_lowpass_alpha_max"),
            position_quiet_step_m=trend.get(
                "pose_position_quiet_step_m", 0.0),
            position_responsive_step_m=trend.get(
                "pose_position_responsive_step_m"),
            rotation_quiet_step_rad=math.radians(trend.get(
                "pose_rotation_quiet_step_deg", 0.0)),
            rotation_responsive_step_rad=(
                None if trend.get(
                    "pose_rotation_responsive_step_deg") is None else
                math.radians(trend.get(
                    "pose_rotation_responsive_step_deg"))),
        )
        self.estimator = SixDofTrendEstimator(
            window_size=trend.get("window_size", 4),
            translation_deadband_m=trend.get("translation_deadband_m", [0.0015]*3),
            rotation_deadband_rad=trend.get("rotation_deadband_rad", [0.015]*3),
            jump_translation_m=trend.get("jump_translation_m", 0.08),
            jump_rotation_rad=math.radians(trend.get("jump_rotation_deg", 45.0)),
            minimum_dt_s=trend.get("minimum_dt_s", 0.008),
            maximum_dt_s=trend.get("maximum_dt_s", 0.45),
            smoothing_alpha=trend.get("causal_smoothing_alpha", 0.45),
            reanchor_after_rejections=trend.get("reanchor_after_rejections", 3),
        )
        if self.mapping_profile_name == "current_linear":
            self.mapper = RelativePoseMapper(
                mapping.get("translation_matrix", np.eye(3)),
                mapping.get("rotation_matrix", np.eye(3)),
                mapping.get("translation_gain", [1, 1, 1]),
                mapping.get("rotation_gain", [1, 1, 1]),
                control.get("maximum_relative_translation_m", [0.20]*3),
                math.radians(control.get(
                    "maximum_relative_rotation_deg", 90.0)),
            )
        else:
            profiles = config.get("mapping_profiles", {})
            profile = profiles.get(self.mapping_profile_name)
            if not isinstance(profile, dict):
                raise ValueError(
                    "unknown mapping profile {}".format(
                        self.mapping_profile_name))
            if profile.get("mode") != "CAMERA_RANGE_TO_GROUND_SECTOR":
                raise ValueError(
                    "unsupported mapping profile mode {}".format(
                        profile.get("mode")))
            calibration_path = str(rospy.get_param(
                "~camera_workspace_calibration_file", ""))
            if not calibration_path:
                raise ValueError(
                    "camera workspace mapping requires a calibration file")
            with open(calibration_path, "r", encoding="utf-8") as stream:
                calibration = yaml.safe_load(stream)
            if not isinstance(calibration, dict):
                raise ValueError("camera workspace calibration is empty")
            if int(calibration.get("schema_version", 0)) != 1:
                raise ValueError(
                    "unsupported camera workspace calibration schema")
            if calibration.get("frame_id") != self.reference_frame:
                raise ValueError(
                    "camera workspace calibration frame does not match {}".format(
                        self.reference_frame))
            human_workspace = calibration.get("human_workspace", {})
            human_orientation = calibration.get("human_orientation", {})
            robot_workspace_config = profile.get("robot_workspace", {})
            normalized_pose = profile.get("normalized_pose_mapping", {})
            normalized_pose_enabled = bool(
                normalized_pose.get("enabled", False))
            combine_translation_rotation = bool(normalized_pose.get(
                "combine_translation_rotation", normalized_pose_enabled))
            human_orientation_negative = human_orientation.get(
                "negative_extent_deg")
            human_orientation_positive = human_orientation.get(
                "positive_extent_deg")
            if normalized_pose_enabled:
                if (not isinstance(human_orientation_negative, list) or
                        not isinstance(human_orientation_positive, list) or
                        len(human_orientation_negative) != 3 or
                        len(human_orientation_positive) != 3 or
                        min(human_orientation_negative +
                            human_orientation_positive) <= 0.0):
                    raise ValueError(
                        "normalized camera pose mapping requires measured "
                        "positive and negative orientation extents")
            robot_workspace = GroundSectorWorkspace(
                robot_workspace_config.get("center_base_m"),
                robot_workspace_config.get("radii_m"),
                robot_workspace_config.get("minimum_forward_x_m"),
                robot_workspace_config.get("minimum_tool_z_m"),
                robot_workspace_config.get("utilization", 1.0),
                robot_workspace_config.get("boundary_margin_m", 0.0),
            )
            self.mapper = CameraRangeWorkspaceMapper(
                profile.get("translation_matrix", np.eye(3)),
                profile.get("rotation_matrix", np.eye(3)),
                profile.get("rotation_gain", [1, 1, 1]),
                math.radians(control.get(
                    "maximum_relative_rotation_deg", 90.0)),
                human_workspace.get("negative_extent_m"),
                human_workspace.get("positive_extent_m"),
                robot_workspace,
                profile.get("response_exponent", 1.0),
                (None if not normalized_pose_enabled else np.radians(
                    human_orientation_negative)),
                (None if not normalized_pose_enabled else np.radians(
                    human_orientation_positive)),
                (None if not normalized_pose_enabled else np.radians(
                    normalized_pose.get(
                        "robot_orientation_negative_extent_deg"))),
                (None if not normalized_pose_enabled else np.radians(
                    normalized_pose.get(
                        "robot_orientation_positive_extent_deg"))),
                combine_translation_rotation,
            )
            projection = normalized_pose.get(
                "reachability_projection", {})
            self.pose_reachability_enabled = bool(
                normalized_pose_enabled and projection.get("enabled", False))
            if (self.pose_reachability_enabled and
                    not combine_translation_rotation):
                raise ValueError(
                    "pose-ray reachability projection requires combined "
                    "translation/rotation mapping")
            self.pose_reachability_ik_service_name = str(projection.get(
                "ik_service", "/compute_ik"))
            self.pose_reachability_ik_group = str(projection.get(
                "ik_group", "abbarm"))
            self.pose_reachability_ik_timeout = float(projection.get(
                "ik_timeout_s", 0.012))
            self.pose_reachability_bisection_iterations = int(projection.get(
                "bisection_iterations", 6))
            self.pose_reachability_safety_factor = float(projection.get(
                "boundary_safety_factor", 0.96))
            self.pose_reachability_cache_cosine = float(projection.get(
                "cache_direction_cosine", 0.99))
            self.pose_reachability_cache_size = int(projection.get(
                "cache_size", 96))
            if self.pose_reachability_enabled and (
                    self.pose_reachability_ik_timeout <= 0.0 or
                    self.pose_reachability_bisection_iterations < 1 or
                    not 0.0 < self.pose_reachability_safety_factor <= 1.0 or
                    not 0.0 < self.pose_reachability_cache_cosine <= 1.0 or
                    self.pose_reachability_cache_size < 1):
                raise ValueError(
                    "invalid normalized-pose reachability projection settings")
            self.camera_workspace_calibration_status = str(
                calibration.get("status", "UNKNOWN"))
            if self.camera_workspace_calibration_status.startswith(
                    "PROVISIONAL"):
                rospy.logwarn(
                    "camera workspace mapping is using provisional ranges: %s",
                    calibration_path)
            else:
                rospy.loginfo(
                    "camera workspace calibration loaded: %s (%s)",
                    calibration_path,
                    self.camera_workspace_calibration_status)
        rospy.loginfo("teleoperation mapping profile: %s",
                      self.mapping_profile_name)
        side_grasp = config.get("side_grasp_projection", {})
        self.side_grasp_projector = SymmetricSideGraspProjector(
            enabled=side_grasp.get("enabled", False),
            axis=side_grasp.get("local_axis", "x"),
            blend_start_rad=math.radians(side_grasp.get(
                "blend_start_deg", 30.0)),
            blend_full_rad=math.radians(side_grasp.get(
                "blend_full_deg", 55.0)),
            dominance_start_ratio=side_grasp.get(
                "dominance_start_ratio", 0.90),
            dominance_full_ratio=side_grasp.get(
                "dominance_full_ratio", 1.15),
        )
        stability = config.get("stability", {})
        self.feedforward_gate = StationaryFeedforwardGate(
            linear_quiet_mps=stability.get(
                "feedforward_linear_quiet_mps", 0.012),
            linear_full_mps=stability.get(
                "feedforward_linear_full_mps", 0.040),
            angular_quiet_radps=stability.get(
                "feedforward_angular_quiet_radps", 0.18),
            angular_full_radps=stability.get(
                "feedforward_angular_full_radps", 0.60),
        )
        self.pose_servo = RelativePoseServoController(
            control.get("translation_error_gain_per_s", [4.0]*3),
            control.get("rotation_error_gain_per_s", [5.0]*3),
            limits.get("maximum_linear_velocity_mps", [0.1]*3),
            limits.get("maximum_angular_velocity_radps", [0.6]*3),
            control.get("maximum_linear_speed_norm_mps", 0.10),
            control.get("maximum_angular_speed_norm_radps", 1.20),
            control.get("translation_feedforward_gain", [0.0]*3),
            control.get("rotation_feedforward_gain", [0.0]*3),
        )
        self.target_linear_speed_limit = float(
            control.get("target_linear_speed_limit_mps", 0.30))
        self.target_angular_speed_limit = float(
            control.get("target_angular_speed_limit_radps", 2.0))
        if (self.target_linear_speed_limit <= 0.0 or
                self.target_angular_speed_limit <= 0.0):
            raise ValueError("target pose speed limits must be positive")
        self.position_tolerance = np.asarray(
            control.get("position_tolerance_m", [0.002]*3), dtype=float)
        self.rotation_tolerance_rad = math.radians(float(
            control.get("rotation_tolerance_deg", 1.5)))
        if self.position_tolerance.shape != (3,) or np.any(
                self.position_tolerance < 0.0):
            raise ValueError("control.position_tolerance_m must be three non-negative values")
        self.simulation = bool(rospy.get_param("~simulation", False))
        self.repeat_last_target_pose = bool(
            control.get("repeat_last_target_pose", True)) and self.simulation
        self.target_hold_timeout = float(
            control.get("target_hold_timeout_s", 0.0))
        if self.target_hold_timeout < 0.0:
            raise ValueError("control.target_hold_timeout_s must be >= 0")
        self.collision_retreat_guard = CollisionRetreatGuard(
            enter_scale=control.get("collision_guard_enter_scale", 0.20),
            release_scale=control.get("collision_guard_release_scale", 0.80),
            translation_progress_m=control.get(
                "collision_guard_translation_progress_m", 0.001),
            rotation_progress_rad=math.radians(control.get(
                "collision_guard_rotation_progress_deg", 1.0)),
        )
        topics = config.get("topics", {})
        self.publisher = rospy.Publisher(
            topics.get("raw_command", "/shared_teleop/raw_hand_command"),
            HandCommand, queue_size=1)
        self.diagnostics = rospy.Publisher(
            topics.get("trend_diagnostics", "/shared_teleop/trend_diagnostics"),
            String, queue_size=1)
        self.listener = tf.TransformListener()
        self.lock = threading.Lock()
        self.robot_zero_position = None
        self.robot_zero_rotation = None
        self.latest_tracking = None
        self.latest_target_update_ros = None
        self.target_hold_reason = None
        self.reference_revision = 0
        self.active_reference_token = None
        self.blocked_reference_token = None
        self.servo_interlock_status = None
        self.last_servo_status = 0
        self.collision_velocity_scale = 1.0
        self.servo_interlock_statuses = {
            int(value) for value in control.get(
                "servo_interlock_statuses", [4])
        }
        if not self.servo_interlock_statuses.issubset(set(range(-1, 6))):
            raise ValueError(
                "control.servo_interlock_statuses contains an unknown status")
        self.servo_retreat_statuses = {
            int(value) for value in control.get(
                "servo_retreat_statuses", [2, 5])
        }
        if not self.servo_retreat_statuses.issubset(set(range(-1, 6))):
            raise ValueError(
                "control.servo_retreat_statuses contains an unknown status")
        self.servo_auto_reset_statuses = {
            int(value) for value in control.get(
                "servo_auto_reset_statuses", [2, 5])
        }
        if not self.servo_auto_reset_statuses.issubset(
                self.servo_retreat_statuses):
            raise ValueError(
                "control.servo_auto_reset_statuses must be a subset of "
                "control.servo_retreat_statuses")
        self.servo_reset_service_name = str(control.get(
            "servo_reset_service", "/servo_server/reset_servo_status"))
        self.servo_reset_min_interval_s = float(control.get(
            "servo_reset_min_interval_s", 0.25))
        self.servo_reset_fresh_target_s = float(control.get(
            "servo_reset_fresh_target_s", 0.40))
        self.servo_reset_min_command_norm = float(control.get(
            "servo_reset_min_command_norm", 1.0e-4))
        if (self.servo_reset_min_interval_s <= 0.0 or
                self.servo_reset_fresh_target_s <= 0.0 or
                self.servo_reset_min_command_norm < 0.0):
            raise ValueError("Servo reset recovery parameters are invalid")
        self.servo_reset = rospy.ServiceProxy(
            self.servo_reset_service_name, Empty, persistent=False)
        self.last_servo_reset_attempt_monotonic = -float("inf")
        self.servo_reset_count = 0
        self.collision_disarm_scale = float(
            control.get("collision_disarm_scale", 0.0))
        if not 0.0 <= self.collision_disarm_scale < 1.0:
            raise ValueError("control.collision_disarm_scale must be in [0, 1)")
        self.require_target_ik = bool(
            control.get("require_collision_free_target_ik", True))
        self.target_ik_service_name = str(
            control.get("target_ik_service", "/compute_ik"))
        self.target_ik_group = str(
            control.get("target_ik_group", "abbarm"))
        self.target_ik_timeout = float(
            control.get("target_ik_timeout_s", 0.06))
        self.target_ik = rospy.ServiceProxy(
            self.target_ik_service_name, GetPositionIK, persistent=True)
        self.pose_reachability_ik = (
            rospy.ServiceProxy(
                self.pose_reachability_ik_service_name,
                GetPositionIK, persistent=True)
            if self.pose_reachability_enabled else None)
        self.shutting_down = False
        rospy.Subscriber(
            topics.get("hamer_pose", "/shared_teleop/hamer_pose"),
            HamerHandPose, self.callback, queue_size=1)
        rospy.Subscriber(
            topics.get("servo_status", "/servo_server/status"),
            Int8, self.servo_status_callback, queue_size=1)
        rospy.Subscriber(
            topics.get(
                "servo_collision_scale",
                "/servo_server/internal/collision_velocity_scale"),
            Float64, self.collision_scale_callback, queue_size=1)
        rospy.Service("/shared_teleop/reset_hand_zero", Trigger, self.reset)
        rospy.Service("/shared_teleop/confirm_hand_reference", Trigger, self.reset)
        rospy.Service("/shared_teleop/clear_hand_reference", Trigger,
                      self.clear_reference)
        rate = float(control.get("rate_hz", 50.0))
        self.timer = rospy.Timer(rospy.Duration(1.0/rate), self.tick)
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        self.shutting_down = True
        self.timer.shutdown()

    def current_control_pose(self):
        translation, quaternion = self.listener.lookupTransform(
            self.base_frame, self.control_frame, rospy.Time(0))
        return (np.asarray(translation, dtype=float),
                quaternion_xyzw_to_matrix(quaternion))

    def query_pose_ik(self, position, rotation, service, group_name,
                      timeout_s, avoid_collisions):
        request = GetPositionIKRequest()
        request.ik_request.group_name = group_name
        request.ik_request.ik_link_name = self.control_frame
        request.ik_request.avoid_collisions = bool(avoid_collisions)
        request.ik_request.timeout = rospy.Duration(timeout_s)
        request.ik_request.robot_state.is_diff = True
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.base_frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        quaternion = matrix_to_quaternion_xyzw(rotation)
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = quaternion
        request.ik_request.pose_stamped = pose
        response = service(request)
        return response.error_code.val == MoveItErrorCodes.SUCCESS

    def target_is_collision_free(self, position, rotation):
        if not self.require_target_ik:
            return True
        try:
            return self.query_pose_ik(
                position, rotation, self.target_ik,
                self.target_ik_group, self.target_ik_timeout, True)
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0, "collision-free target IK unavailable: %s", exc)
            return False

    def _compute_reachable_pose_ray_limit(
            self, robot_zero_position, robot_zero_rotation,
            boundary_position, boundary_rotation):
        calls = 1
        if self.query_pose_ik(
                boundary_position, boundary_rotation,
                self.pose_reachability_ik,
                self.pose_reachability_ik_group,
                self.pose_reachability_ik_timeout, False):
            raw_limit = 1.0
        else:
            low = 0.0
            high = 1.0
            for _ in range(self.pose_reachability_bisection_iterations):
                middle = 0.5 * (low + high)
                position, rotation = interpolate_pose_ray(
                    robot_zero_position, robot_zero_rotation,
                    boundary_position, boundary_rotation, middle)
                calls += 1
                if self.query_pose_ik(
                        position, rotation, self.pose_reachability_ik,
                        self.pose_reachability_ik_group,
                        self.pose_reachability_ik_timeout, False):
                    low = middle
                else:
                    high = middle
            raw_limit = low
        return raw_limit * self.pose_reachability_safety_factor, calls

    def project_normalized_camera_pose(
            self, robot_zero_position, robot_zero_rotation):
        """Map the calibrated human radius onto the attainable 6-D pose ray.

        This is a kinematic projection, not a per-frame hard rejection. If the
        paper's position-only shell is incompatible with the requested wrist
        orientation, the outer endpoint is shortened along the same 6-D ray;
        returning the hand to C-zero always returns exactly to robot C-zero.
        """

        boundary_position, boundary_rotation, human_fraction, direction = (
            self.mapper.reachability_boundary())
        if human_fraction <= 1.0e-12:
            self.pose_reachability_last = {
                "limit": 1.0, "cache_hit": False, "ik_calls": 0,
                "processing_ms": 0.0,
            }
            return (np.asarray(robot_zero_position, dtype=float).copy(),
                    np.asarray(robot_zero_rotation, dtype=float).copy())

        began = time.perf_counter()
        cache_hit = False
        cache_index = None
        limit = None
        best_cosine = -1.0
        for index, entry in enumerate(self.pose_reachability_cache):
            cosine = float(np.dot(direction, entry["direction"]))
            if cosine > best_cosine:
                best_cosine = cosine
                cache_index = index
        if (cache_index is not None and
                best_cosine >= self.pose_reachability_cache_cosine):
            limit = float(
                self.pose_reachability_cache[cache_index]["limit"])
            cache_hit = True

        ik_calls = 0
        if limit is None:
            limit, ik_calls = self._compute_reachable_pose_ray_limit(
                robot_zero_position, robot_zero_rotation,
                boundary_position, boundary_rotation)
            self.pose_reachability_cache.append({
                "direction": direction.copy(), "limit": float(limit)})
            if len(self.pose_reachability_cache) > (
                    self.pose_reachability_cache_size):
                self.pose_reachability_cache.pop(0)

        projected_fraction = float(np.clip(
            human_fraction * limit, 0.0, 1.0))
        target_position, target_rotation = interpolate_pose_ray(
            robot_zero_position, robot_zero_rotation,
            boundary_position, boundary_rotation, projected_fraction)

        # A nearby cached direction is normally enough. Near the outer shell,
        # verify it once; if the local boundary bends inward, recompute the
        # exact ray rather than passing an infeasible target to Servo.
        if cache_hit and projected_fraction >= 0.80:
            ik_calls += 1
            if not self.query_pose_ik(
                    target_position, target_rotation,
                    self.pose_reachability_ik,
                    self.pose_reachability_ik_group,
                    self.pose_reachability_ik_timeout, False):
                limit, extra_calls = self._compute_reachable_pose_ray_limit(
                    robot_zero_position, robot_zero_rotation,
                    boundary_position, boundary_rotation)
                ik_calls += extra_calls
                cache_hit = False
                self.pose_reachability_cache.append({
                    "direction": direction.copy(), "limit": float(limit)})
                if len(self.pose_reachability_cache) > (
                        self.pose_reachability_cache_size):
                    self.pose_reachability_cache.pop(0)
                projected_fraction = float(np.clip(
                    human_fraction * limit, 0.0, 1.0))
                target_position, target_rotation = interpolate_pose_ray(
                    robot_zero_position, robot_zero_rotation,
                    boundary_position, boundary_rotation,
                    projected_fraction)

        self.pose_reachability_last = {
            "limit": float(limit),
            "projected_fraction": projected_fraction,
            "cache_hit": bool(cache_hit),
            "cache_direction_cosine": float(best_cosine),
            "ik_calls": int(ik_calls),
            "processing_ms": (time.perf_counter() - began) * 1000.0,
        }
        return target_position, target_rotation

    def clear_reference_locked(self, armed=False, token=None):
        self.reference_revision += 1
        self.pose_continuity.reset()
        self.estimator.reset_zero()
        self.reference_armed = bool(armed)
        self.reference_ready = False
        self.robot_zero_position = None
        self.robot_zero_rotation = None
        self.latest_tracking = None
        self.latest_target_update_ros = None
        self.target_hold_reason = None
        self.collision_retreat_guard.reset()
        self.pose_reachability_cache = []
        self.pose_reachability_last = {
            "limit": 1.0, "cache_hit": False, "ik_calls": 0,
            "processing_ms": 0.0,
        }
        self.active_reference_token = token

    def reset(self, _request):
        with self.lock:
            self.clear_reference_locked(armed=True, token=None)
            self.blocked_reference_token = None
            self.servo_interlock_status = None
        self.publish_hold_now("REFERENCE_CAPTURE_ARMED")
        return TriggerResponse(success=True, message=(
            "reference capture armed; next valid hand pose locks both hand zero "
            "and current tool0 pose"))

    def clear_reference(self, _request):
        with self.lock:
            self.clear_reference_locked(
                armed=not self.require_confirmation, token=None)
            self.blocked_reference_token = None
            self.servo_interlock_status = None
        self.publish_hold_now("REFERENCE_CLEARED")
        return TriggerResponse(
            success=True, message="hand and robot references cleared; arm output held")

    def prepare_operator_reference(self, message):
        """Apply the camera C-key gate before accepting any live pose."""

        if not message.control_gate_present:
            return True
        token = str(message.control_reference_token)
        enabled = bool(message.control_enabled and token)
        if not enabled:
            with self.lock:
                state_changed = bool(
                    self.reference_armed or self.reference_ready or
                    self.active_reference_token is not None or
                    self.latest_tracking is not None)
                if state_changed:
                    self.clear_reference_locked(armed=False, token=None)
                self.blocked_reference_token = None
                self.servo_interlock_status = None
            if state_changed:
                self.publish_hold_now("OPERATOR_C_GATE_LOCKED")
            self.publish_waiting(message, "WAITING_FOR_OPERATOR_C_REFERENCE")
            rospy.logwarn_throttle(
                2.0, "Gazebo control LOCKED: hold a neutral hand pose and "
                "press C in the camera window")
            return False

        with self.lock:
            if token == self.blocked_reference_token:
                blocked_status = self.servo_interlock_status
                token_changed = False
            else:
                blocked_status = None
                token_changed = token != self.active_reference_token
                if token_changed:
                    self.clear_reference_locked(armed=True, token=token)
                    self.blocked_reference_token = None
                    self.servo_interlock_status = None
        if blocked_status is not None:
            self.publish_waiting(
                message, "SERVO_SAFETY_REQUIRES_NEW_C_REFERENCE")
            rospy.logerr_throttle(
                2.0, "Servo safety status %d locked this C reference; move "
                "the hand to neutral and press C again", blocked_status)
            return False
        if token_changed:
            self.publish_hold_now("NEW_C_REFERENCE_CAPTURE")
            rospy.loginfo(
                "C reference token %s accepted; next valid pose captures "
                "both hand zero and current %s pose", token,
                self.control_frame)
        return True

    def engage_servo_interlock(self, status):
        with self.lock:
            active = bool(
                self.reference_armed or self.reference_ready or
                self.latest_tracking is not None)
            if not active:
                return
            token = self.active_reference_token
            self.clear_reference_locked(armed=False, token=token)
            self.blocked_reference_token = (
                token if token is not None else "__MANUAL_REFERENCE__")
            self.servo_interlock_status = status
        self.publish_hold_now("SERVO_STATUS_{}_REFERENCE_LOCKED".format(status))
        rospy.logerr(
            "Servo status %d at collision scale %.3f disarmed hand tracking "
            "before further motion; return the hand to neutral and press C "
            "again", status, self.collision_velocity_scale)

    def servo_status_callback(self, message):
        """Latch only configured hard faults; let Servo recover transients."""

        status = int(message.data)
        self.last_servo_status = status
        if status in self.servo_interlock_statuses or (
                self.collision_disarm_scale > 0.0 and
                status == 3 and
                self.collision_velocity_scale <= self.collision_disarm_scale):
            self.engage_servo_interlock(status)

    def collision_scale_callback(self, message):
        self.collision_velocity_scale = float(np.clip(message.data, 0.0, 1.0))
        if (self.collision_disarm_scale > 0.0 and
                self.last_servo_status == 3 and
                self.collision_velocity_scale <= self.collision_disarm_scale):
            self.engage_servo_interlock(3)

    def maybe_reset_recoverable_servo_halt(
            self, target_age_s, velocity, collision_retreat):
        """Reset a latched Servo halt only for a fresh, safe recovery command.

        MoveIt Servo status 2/5 remains published after the triggering command
        has disappeared.  Our C-zero retreat guard then sees that status and
        emits zero forever, so Servo never receives the nonzero command needed
        to recalculate its condition.  Resetting blindly would make it chase a
        stale target.  This gate requires a fresh MANO target and a command
        already approved by the C-zero retreat guard before calling Servo's
        own reset service.
        """

        status = int(self.last_servo_status)
        if status not in self.servo_auto_reset_statuses:
            return False, False, "NONE"
        if target_age_s > self.servo_reset_fresh_target_s:
            return False, False, "WAITING_FOR_FRESH_HAND_TARGET"
        if collision_retreat is None or not collision_retreat.active:
            return False, False, "WAITING_FOR_C_ZERO_RETREAT_GUARD"
        if float(np.linalg.norm(velocity)) <= self.servo_reset_min_command_norm:
            return False, False, "WAITING_FOR_NONZERO_RECOVERY_COMMAND"
        now_monotonic = time.monotonic()
        if (now_monotonic - self.last_servo_reset_attempt_monotonic <
                self.servo_reset_min_interval_s):
            return False, False, "RESET_RETRY_THROTTLED"
        self.last_servo_reset_attempt_monotonic = now_monotonic
        try:
            self.servo_reset()
        except rospy.ServiceException as exc:
            rospy.logerr_throttle(
                1.0, "failed to reset recoverable Servo status %d: %s",
                status, exc)
            return True, False, "RESET_SERVICE_FAILED"
        self.servo_reset_count += 1
        rospy.logwarn(
            "reset recoverable Servo status %d for a fresh C-zero-safe "
            "command (reset count=%d)", status, self.servo_reset_count)
        return True, True, "RESET_REQUESTED"

    def publish_hold_now(self, reason):
        command = HandCommand()
        command.header.stamp = rospy.Time.now()
        command.header.frame_id = self.base_frame
        command.confidence = [0.0] * 6
        command.valid = False
        self.publisher.publish(command)
        self.publish_diagnostic({
            "stamp": command.header.stamp.to_sec(),
            "valid": False,
            "reason": reason,
            "reference_ready": False,
        })

    def publish_diagnostic(self, values):
        defaults = {
            "control_mode": self.control_mode,
            "reference_frame": self.reference_frame,
            "reference_policy": self.reference_policy,
            "mapping_profile": self.mapping_profile_name,
            "camera_workspace_calibration_status": (
                self.camera_workspace_calibration_status),
            "reference_ready": self.reference_ready,
            "active_reference_token": self.active_reference_token,
            "servo_interlock_status": self.servo_interlock_status,
            "servo_status": self.last_servo_status,
            "servo_auto_reset_attempted": False,
            "servo_auto_reset_succeeded": False,
            "servo_auto_reset_count": self.servo_reset_count,
            "servo_recovery_reason": "NONE",
            "collision_velocity_scale": self.collision_velocity_scale,
            "target_input_age_s": None,
            "target_hold_active": False,
            "collision_retreat_guard_active": False,
            "collision_retreat_reason": "NONE",
            "linear_retreat_allowed": True,
            "angular_retreat_allowed": True,
            "raw_velocity": [0.0] * 6,
            "mapped_velocity": [0.0] * 6,
            "target_feedforward_velocity": [0.0] * 6,
            "feedforward_linear_weight": 0.0,
            "feedforward_angular_weight": 0.0,
            "pose_position_filter_alpha": self.pose_continuity.last_position_alpha,
            "pose_rotation_filter_alpha": self.pose_continuity.last_rotation_alpha,
            "pose_continuity_reason": self.pose_continuity.last_reason,
            "relative_position": [0.0] * 3,
            "relative_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "raw_relative_rotation_vector_deg": [0.0] * 3,
            "projected_relative_rotation_vector_deg": [0.0] * 3,
            "side_grasp_projection_active": False,
            "side_grasp_projection_weight": 0.0,
            "side_grasp_local_axis": self.side_grasp_projector.axis_name,
            "side_grasp_sign": 0,
            "workspace_mapping": self.mapper.mapping_diagnostics(),
            "confidence_camera_axes": [0.0] * 6,
            "confidence": [0.0] * 6,
            "robot_zero_position": None,
            "robot_zero_quaternion_xyzw": None,
            "target_position": None,
            "target_quaternion_xyzw": None,
            "current_position": None,
            "current_quaternion_xyzw": None,
            "position_error_m": None,
            "rotation_error_rad": None,
            "processing_ms": 0.0,
        }
        defaults.update(values)
        self.diagnostics.publish(String(data=json.dumps(
            defaults, separators=(",", ":"))))

    def publish_waiting(self, message, reason):
        with self.lock:
            self.latest_tracking = None
            self.target_hold_reason = None
        command = HandCommand()
        command.header.stamp = message.header.stamp
        command.header.frame_id = self.base_frame
        command.confidence = [0.0] * 6
        command.valid = False
        command.gesture = message.gesture
        command.gesture_confidence = message.gesture_confidence
        self.publisher.publish(command)
        self.publish_diagnostic({
            "stamp": message.source_timestamp,
            "valid": False,
            "reason": reason,
            "reference_ready": False,
        })

    def callback(self, message):
        began = time.perf_counter()
        if message.header.frame_id != self.reference_frame:
            self.publish_waiting(message, "REFERENCE_FRAME_MISMATCH")
            rospy.logwarn_throttle(
                1.0, "HaMeR reference frame changed: expected %s, got %s",
                self.reference_frame, message.header.frame_id)
            return
        if not self.prepare_operator_reference(message):
            return
        with self.lock:
            reference_armed = self.reference_armed
            reference_revision = self.reference_revision
        if not reference_armed:
            self.publish_waiting(message, "WAITING_FOR_REFERENCE_CONFIRMATION")
            rospy.logwarn_throttle(
                1.0, "Hand reference not confirmed; hold hand steady then call "
                "/shared_teleop/confirm_hand_reference")
            return
        try:
            source_timestamp = (
                message.source_timestamp
                if math.isfinite(message.source_timestamp) and
                message.source_timestamp > 0.0
                else message.header.stamp.to_sec())
            sample = PoseSample(
                source_timestamp,
                [message.wrist_pose.position.x, message.wrist_pose.position.y,
                 message.wrist_pose.position.z],
                quaternion_xyzw_to_matrix([
                    message.wrist_pose.orientation.x,
                    message.wrist_pose.orientation.y,
                    message.wrist_pose.orientation.z,
                    message.wrist_pose.orientation.w]),
                message.confidence, message.valid, message.gesture,
                message.gesture_confidence,
            )
            with self.lock:
                hand_zero_position_for_retreat = (
                    None if self.estimator.zero_position is None else
                    self.estimator.zero_position.copy())
                hand_zero_rotation_for_retreat = (
                    None if self.estimator.zero_rotation is None else
                    self.estimator.zero_rotation.copy())
            sample = self.pose_continuity.update(
                sample,
                return_reference_position=hand_zero_position_for_retreat,
                return_reference_rotation=hand_zero_rotation_for_retreat)
            pose_continuity_reason = self.pose_continuity.last_reason
            with self.lock:
                result = self.estimator.update(sample)
                needs_robot_zero = self.robot_zero_position is None
                hand_zero_rotation = (
                    None if self.estimator.zero_rotation is None else
                    self.estimator.zero_rotation.copy())
            if not result.valid:
                invalid_reason = (
                    self.pose_continuity.last_reason or result.reason)
                with self.lock:
                    hold_existing_target = bool(
                        self.repeat_last_target_pose and
                        self.reference_armed and self.reference_ready and
                        self.latest_tracking is not None and
                        self.latest_tracking["valid"])
                    if hold_existing_target:
                        self.target_hold_reason = "HOLD_LAST_{}".format(
                            invalid_reason)
                if hold_existing_target:
                    rospy.logwarn_throttle(
                        1.0, "MANO observation invalid (%s); holding the last "
                        "valid target pose", invalid_reason)
                    return
                self.publish_waiting(message, invalid_reason)
                return
            if result.reason == "ZERO_INITIALIZED" or needs_robot_zero:
                try:
                    robot_zero_position, robot_zero_rotation = (
                        self.current_control_pose())
                except Exception:
                    with self.lock:
                        self.estimator.reset_zero()
                        self.reference_ready = False
                    raise
                with self.lock:
                    if (self.reference_revision != reference_revision or
                            not self.reference_armed):
                        return
                    self.robot_zero_position = robot_zero_position
                    self.robot_zero_rotation = robot_zero_rotation
                    self.reference_ready = True
            with self.lock:
                robot_zero_position = self.robot_zero_position.copy()
                robot_zero_rotation = self.robot_zero_rotation.copy()
            side_projection = self.side_grasp_projector.project(
                result.relative_rotation)
            target_position, target_rotation = self.mapper.map(
                result.relative_position, side_projection.rotation,
                robot_zero_position, robot_zero_rotation)
            if self.pose_reachability_enabled:
                target_position, target_rotation = (
                    self.project_normalized_camera_pose(
                        robot_zero_position, robot_zero_rotation))
                with self.lock:
                    previous_tracking = self.latest_tracking
                target_velocity = np.zeros(6)
                if previous_tracking is not None:
                    target_dt = (
                        result.timestamp -
                        previous_tracking["source_stamp"])
                    if 0.5 * self.estimator.minimum_dt_s <= target_dt <= (
                            1.5 * self.estimator.maximum_dt_s):
                        target_velocity[:3] = (
                            target_position -
                            previous_tracking["target_position"]) / target_dt
                        target_velocity[3:] = so3_log(
                            target_rotation @
                            previous_tracking["target_rotation"].T) / target_dt
            else:
                target_velocity = self.mapper.map_target_velocity(
                    result.raw_velocity, hand_zero_rotation,
                    robot_zero_rotation,
                    relative_hand_position=result.relative_position,
                    robot_zero_position=robot_zero_position)
            mapping_diagnostics = self.mapper.mapping_diagnostics()
            if self.pose_reachability_enabled:
                mapping_diagnostics["reachability_projection"] = dict(
                    self.pose_reachability_last)
            if side_projection.active and not self.pose_reachability_enabled:
                angular_tool_zero = robot_zero_rotation.T @ target_velocity[3:]
                angular_tool_zero = (
                    self.side_grasp_projector.project_local_angular_velocity(
                        angular_tool_zero, side_projection))
                target_velocity[3:] = robot_zero_rotation @ angular_tool_zero
            feedforward = self.feedforward_gate.apply(target_velocity)
            target_velocity = feedforward.velocity
            target_velocity[:3] = RelativePoseServoController.clamp_norm(
                target_velocity[:3], self.target_linear_speed_limit)
            target_velocity[3:] = RelativePoseServoController.clamp_norm(
                target_velocity[3:], self.target_angular_speed_limit)
            if not self.target_is_collision_free(
                    target_position, target_rotation):
                self.publish_waiting(
                    message, "TARGET_IK_COLLISION_OR_UNREACHABLE")
                rospy.logwarn_throttle(
                    1.0, "mapped hand target rejected by collision-free IK; "
                    "move the hand back toward its C-zero")
                return
            mapped_confidence = self.mapper.map_confidence(result.confidence)
            tracking = {
                "header_stamp": message.header.stamp,
                "source_stamp": result.timestamp,
                "valid": bool(result.valid),
                "reason": result.reason,
                "target_position": target_position,
                "target_rotation": target_rotation,
                "target_velocity": target_velocity,
                "robot_zero_position": robot_zero_position,
                "robot_zero_rotation": robot_zero_rotation,
                "relative_position": result.relative_position.copy(),
                "relative_rotation": result.relative_rotation.copy(),
                "projected_relative_rotation": side_projection.rotation.copy(),
                "raw_relative_rotation_vector": (
                    side_projection.input_rotation_vector.copy()),
                "projected_relative_rotation_vector": (
                    side_projection.projected_rotation_vector.copy()),
                "side_grasp_projection_weight": side_projection.weight,
                "side_grasp_projection_active": side_projection.active,
                "side_grasp_sign": side_projection.side_sign,
                "workspace_mapping": mapping_diagnostics,
                "feedforward_linear_weight": feedforward.linear_weight,
                "feedforward_angular_weight": feedforward.angular_weight,
                "pose_position_filter_alpha": (
                    self.pose_continuity.last_position_alpha),
                "pose_rotation_filter_alpha": (
                    self.pose_continuity.last_rotation_alpha),
                "pose_continuity_reason": pose_continuity_reason,
                "raw_velocity": result.raw_velocity.copy(),
                "camera_confidence": result.confidence.copy(),
                "confidence": mapped_confidence,
                "gesture": result.gesture,
                "gesture_confidence": result.gesture_confidence,
                "callback_processing_ms": (
                    time.perf_counter() - began) * 1000.0,
            }
            with self.lock:
                if (self.reference_revision != reference_revision or
                        not self.reference_armed or
                        not self.reference_ready):
                    return
                self.latest_tracking = tracking
                self.latest_target_update_ros = rospy.Time.now().to_sec()
                self.target_hold_reason = None
        except Exception as exc:
            with self.lock:
                hold_existing_target = bool(
                    self.repeat_last_target_pose and
                    self.reference_armed and self.reference_ready and
                    self.latest_tracking is not None and
                    self.latest_tracking["valid"])
                if hold_existing_target:
                    self.target_hold_reason = "HOLD_LAST_POSE_TARGET_ERROR"
            if not hold_existing_target:
                self.publish_waiting(message, "POSE_TARGET_ERROR")
            rospy.logwarn_throttle(
                1.0, "relative hand pose rejected%s: %s",
                "; holding last valid target" if hold_existing_target else "",
                exc)
            return
        if result.reason != "NONE":
            rospy.logwarn_throttle(
                1.0, "relative-pose tracking status: %s", result.reason)
        if pose_continuity_reason == "C_ZERO_RETREAT_OVERRIDE":
            rospy.logwarn_throttle(
                1.0, "large hand-pose innovation accepted only because it "
                "makes progress toward the operator C-zero")

    def tick(self, _event):
        if self.shutting_down or rospy.is_shutdown():
            return
        began = time.perf_counter()
        with self.lock:
            state = self.latest_tracking
            latest_target_update_ros = self.latest_target_update_ros
            target_hold_reason = self.target_hold_reason
        if state is None:
            return
        now_ros = rospy.Time.now()
        target_age_s = max(
            0.0,
            now_ros.to_sec() - latest_target_update_ros
            if latest_target_update_ros is not None else float("inf"))
        target_hold_allowed = bool(
            self.repeat_last_target_pose and
            (self.target_hold_timeout <= 0.0 or
             target_age_s <= self.target_hold_timeout))
        holding_last_target = bool(
            target_hold_allowed and
            (target_hold_reason is not None or target_age_s > 0.05))
        valid = state["valid"]
        reason = (
            target_hold_reason if holding_last_target and
            target_hold_reason is not None else state["reason"])
        current_position = None
        current_rotation = None
        position_error = None
        rotation_error = None
        velocity = np.zeros(6)
        collision_retreat = None
        servo_reset_attempted = False
        servo_reset_succeeded = False
        servo_recovery_reason = "NONE"
        if valid:
            try:
                current_position, current_rotation = self.current_control_pose()
                position_error = state["target_position"] - current_position
                rotation_error = so3_log(
                    state["target_rotation"] @ current_rotation.T)
                # A V3 HOLD_LAST packet repeats a fixed target pose.  Its
                # target velocity therefore decays to zero; retaining the
                # final measured hand velocity would drive past that target.
                target_velocity = (
                    np.zeros(6) if holding_last_target else
                    state["target_velocity"])
                velocity = self.pose_servo.command(
                    current_position, current_rotation,
                    state["target_position"], state["target_rotation"],
                    target_velocity)
                velocity[:3][
                    np.abs(position_error) <= self.position_tolerance] = 0.0
                if np.linalg.norm(rotation_error) <= self.rotation_tolerance_rad:
                    velocity[3:] = 0.0
                servo_retreat_active = bool(
                    self.last_servo_status in self.servo_retreat_statuses)
                retreat_scale = (
                    0.0 if servo_retreat_active else
                    self.collision_velocity_scale)
                retreat_reason = (
                    "SERVO_STATUS_{}_RETURN_TOWARD_C_ZERO".format(
                        self.last_servo_status)
                    if servo_retreat_active else
                    "COLLISION_PROXIMITY_RETURN_TOWARD_C_ZERO")
                with self.lock:
                    collision_retreat = self.collision_retreat_guard.apply(
                        retreat_scale,
                        current_position, current_rotation,
                        state["robot_zero_position"],
                        state["robot_zero_rotation"], velocity,
                        active_reason=retreat_reason)
                velocity = collision_retreat.velocity
                (servo_reset_attempted, servo_reset_succeeded,
                 servo_recovery_reason) = (
                    self.maybe_reset_recoverable_servo_halt(
                        target_age_s, velocity, collision_retreat))
            except Exception as exc:
                valid = False
                reason = "CONTROL_TF_ERROR"
                rospy.logwarn_throttle(
                    1.0, "relative-pose controller waiting for TF %s -> %s: %s",
                    self.base_frame, self.control_frame, exc)
        command = HandCommand()
        # ros_udp_target_pose_receiver_apriltag_v3 refreshes the timestamp
        # whenever it republishes the last target.  Do the same in Gazebo so
        # brief MANO/MediaPipe gaps hold the last pose instead of repeatedly
        # braking and restarting the speed loop.
        command.header.stamp = (
            now_ros if target_hold_allowed else state["header_stamp"])
        command.header.frame_id = self.base_frame
        command.confidence = state["confidence"].tolist()
        command.valid = valid
        command.gesture = state["gesture"]
        command.gesture_confidence = state["gesture_confidence"]
        values = velocity if valid else np.zeros(6)
        (command.twist.linear.x, command.twist.linear.y,
         command.twist.linear.z, command.twist.angular.x,
         command.twist.angular.y, command.twist.angular.z) = values
        self.publisher.publish(command)
        self.publish_diagnostic({
            "stamp": state["source_stamp"],
            "valid": valid,
            "reason": reason,
            "reference_ready": self.reference_ready,
            "servo_status": self.last_servo_status,
            "servo_auto_reset_attempted": servo_reset_attempted,
            "servo_auto_reset_succeeded": servo_reset_succeeded,
            "servo_auto_reset_count": self.servo_reset_count,
            "servo_recovery_reason": servo_recovery_reason,
            "target_input_age_s": target_age_s,
            "target_hold_active": holding_last_target,
            "collision_retreat_guard_active": bool(
                collision_retreat is not None and collision_retreat.active),
            "collision_retreat_reason": (
                "NONE" if collision_retreat is None else
                collision_retreat.reason),
            "linear_retreat_allowed": bool(
                collision_retreat is None or
                collision_retreat.linear_retreat_allowed),
            "angular_retreat_allowed": bool(
                collision_retreat is None or
                collision_retreat.angular_retreat_allowed),
            "raw_velocity": state["raw_velocity"].tolist(),
            "mapped_velocity": values.tolist(),
            "target_feedforward_velocity": (
                np.zeros(6) if holding_last_target else
                state["target_velocity"]).tolist(),
            "feedforward_linear_weight": float(
                state["feedforward_linear_weight"]),
            "feedforward_angular_weight": float(
                state["feedforward_angular_weight"]),
            "pose_position_filter_alpha": float(
                state["pose_position_filter_alpha"]),
            "pose_rotation_filter_alpha": float(
                state["pose_rotation_filter_alpha"]),
            "pose_continuity_reason": state["pose_continuity_reason"],
            "relative_position": state["relative_position"].tolist(),
            "relative_quaternion_xyzw": matrix_to_quaternion_xyzw(
                state["relative_rotation"]).tolist(),
            "raw_relative_rotation_vector_deg": np.degrees(
                state["raw_relative_rotation_vector"]).tolist(),
            "projected_relative_rotation_vector_deg": np.degrees(
                state["projected_relative_rotation_vector"]).tolist(),
            "side_grasp_projection_active": bool(
                state["side_grasp_projection_active"]),
            "side_grasp_projection_weight": float(
                state["side_grasp_projection_weight"]),
            "side_grasp_local_axis": self.side_grasp_projector.axis_name,
            "side_grasp_sign": int(state["side_grasp_sign"]),
            "workspace_mapping": state["workspace_mapping"],
            "confidence_camera_axes": state["camera_confidence"].tolist(),
            "confidence": state["confidence"].tolist(),
            "robot_zero_position": state["robot_zero_position"].tolist(),
            "robot_zero_quaternion_xyzw": matrix_to_quaternion_xyzw(
                state["robot_zero_rotation"]).tolist(),
            "target_position": state["target_position"].tolist(),
            "target_quaternion_xyzw": matrix_to_quaternion_xyzw(
                state["target_rotation"]).tolist(),
            "current_position": (
                None if current_position is None else current_position.tolist()),
            "current_quaternion_xyzw": (
                None if current_rotation is None else
                matrix_to_quaternion_xyzw(current_rotation).tolist()),
            "position_error_m": (
                None if position_error is None else position_error.tolist()),
            "rotation_error_rad": (
                None if rotation_error is None else rotation_error.tolist()),
            "processing_ms": (
                state["callback_processing_ms"] +
                (time.perf_counter() - began) * 1000.0),
        })


def main():
    rospy.init_node("six_dof_trend")
    SixDofTrendNode()
    rospy.spin()


if __name__ == "__main__":
    main()
