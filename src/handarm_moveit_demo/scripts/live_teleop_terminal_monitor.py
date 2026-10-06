#!/usr/bin/env python3
"""Print operator-visible proof that C reaches the simulated robot."""

import json
import math
import os
from pathlib import Path
import time
import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64, Int8, String
from sensor_msgs.msg import JointState
from handarm_moveit_demo.msg import HamerHandPose


class LiveTeleopTerminalMonitor:
    def __init__(self):
        self.buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.reference_token = ""
        self.reference_pose = None
        self.nonzero_command_seen = False
        self.command_seen_monotonic = None
        self.actual_motion_reported = False
        self.no_motion_warning_reported = False
        self.servo_status = 0
        self.collision_scale = 1.0
        self.latest_diagnostic = {}
        self.last_diagnostic_at = 0.0
        self.hand_input_valid = None
        self.hand_orientation_valid = None
        self.hand_input_reason = ""
        self.geometry = {}
        self.finger_closure = None
        self.grip_latched = False
        self.release_phase = None
        self.hand_safety_status = ""
        self.robot_safety_status = ""
        self.feedback_file = rospy.get_param("~feedback_file", "")
        self.last_feedback_write = 0.0
        # Start from the operator's actual C event, not a startup countdown.
        # The previous ten-minute capture ended before C and missed the fault.
        self.trace_lock = threading.Lock()
        self.trace_stream = None
        self.trace_started = False
        self.trace_closed = False
        self.trace_flush_time = 0.0
        self.trace_path = (Path(self.feedback_file).with_name("control_trace.jsonl")
                           if self.feedback_file else None)
        rospy.on_shutdown(self.close_trace)
        rospy.Subscriber("/shared_teleop/hamer_pose", HamerHandPose,
                         self.hand_trace_callback, queue_size=100, tcp_nodelay=True)
        rospy.Subscriber("/joint_states", JointState, self.joint_trace_callback,
                         queue_size=300, tcp_nodelay=True)
        for key, topic in (
                ("egm", "/egm_position_reference/diagnostics"),
                ("direct_velocity", "/direct_velocity/diagnostics"),
                ("geometry", "/shared_teleop/geometric_filter_diagnostics"),
                ("motor", "/handarm_sim_demo/physical_hand_diagnostics"),
                ("finger", "/shared_teleop/finger_diagnostics"),
                ("guard", "/full_robot_self_collision_guard/command_diagnostic"),
                ("hand_guard", "/full_robot_self_collision_guard/hand_diagnostic")):
            rospy.Subscriber(topic, String,
                             lambda m, k=key: self.trace(k, m.data),
                             queue_size=200, tcp_nodelay=True)
        self.direction_blocked_since = None
        rospy.Subscriber("/shared_teleop/geometric_filter_diagnostics", String,
                         self.geometry_callback, queue_size=1)
        rospy.Subscriber("/shared_teleop/finger_diagnostics", String,
                         self.finger_callback, queue_size=1)
        rospy.Subscriber("/full_robot_self_collision_guard/hand_diagnostic", String,
                         lambda m: setattr(self, 'hand_safety_status', m.data), queue_size=1)
        rospy.Subscriber("/full_robot_self_collision_guard/diagnostic", String,
                         lambda m: setattr(self, 'robot_safety_status', m.data), queue_size=1)
        rospy.Subscriber(
            "/shared_teleop/trend_diagnostics",
            String,
            self.diagnostic_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/servo_server/delta_twist_cmds",
            TwistStamped,
            self.twist_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/servo_server/status", Int8, self.status_callback, queue_size=1
        )
        rospy.Subscriber(
            "/servo_server/internal/collision_velocity_scale",
            Float64,
            self.collision_scale_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(0.05), self.tick)

    def trace(self, key, value):
        if not self.trace_started or self.trace_path is None or self.trace_closed:
            return
        with self.trace_lock:
            if self.trace_closed:
                return
            try:
                if self.trace_stream is None:
                    self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                    self.trace_stream = self.trace_path.open("a", buffering=1024 * 1024)
                self.trace_stream.write(json.dumps(dict(
                    topic=key, wall=time.time(), mono=time.monotonic(),
                    sim=rospy.Time.now().to_sec(), data=value), separators=(",", ":")) + "\n")
                if time.monotonic() - self.trace_flush_time >= 1.0:
                    self.trace_stream.flush()
                    self.trace_flush_time = time.monotonic()
            except OSError as exc:
                # Recording must never interrupt the physical control loop.
                rospy.logwarn_throttle(5.0, "Control trace write failed: %s", exc)

    def close_trace(self):
        with self.trace_lock:
            self.trace_closed = True
            if self.trace_stream is not None:
                self.trace_stream.close()
                self.trace_stream = None

    def hand_trace_callback(self, message):
        self.hand_input_valid = bool(message.valid)
        self.hand_orientation_valid = bool(
            message.valid and len(message.confidence) == 6 and
            all(math.isfinite(value) and value > 0.0
                for value in message.confidence[3:]))
        self.hand_input_reason = str(message.invalid_reason)
        if message.control_enabled and message.valid:
            self.trace_started = True
        def pose(p):
            return [p.position.x, p.position.y, p.position.z,
                    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        self.trace("hand_input", dict(
            source=message.source_timestamp, sequence=message.source_capture_sequence,
            valid=message.valid, reason=message.invalid_reason,
            enabled=message.control_enabled, token=message.control_reference_token,
            latency_s=message.capture_to_publish_s, pose=pose(message.wrist_pose),
            confidence=list(message.confidence), finger_valid=message.finger_tracking_valid,
            finger_confidence=message.finger_tracking_confidence,
            flexion=list(message.finger_flexion),
            reference_present=message.control_reference_pose_present,
            reference_sequence=message.control_reference_sequence,
            reference_pose=pose(message.control_reference_wrist_pose)))

    def joint_trace_callback(self, message):
        self.trace("joints", dict(stamp=message.header.stamp.to_sec(),
                                 names=list(message.name), q=list(message.position),
                                 v=list(message.velocity), effort=list(message.effort)))

    def tool_pose(self):
        transform = self.buffer.lookup_transform(
            "base_link", "tool0", rospy.Time(0), rospy.Duration(0.03)
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            np.asarray(
                [translation.x, translation.y, translation.z], dtype=float
            ),
            np.asarray(
                [rotation.x, rotation.y, rotation.z, rotation.w], dtype=float
            ),
        )

    def diagnostic_callback(self, message):
        try:
            diagnostic = json.loads(message.data)
        except (TypeError, ValueError):
            return
        self.trace("trend", diagnostic)
        self.latest_diagnostic = diagnostic
        self.last_diagnostic_at = time.monotonic()
        token = str(diagnostic.get("active_reference_token", ""))
        accepted = bool(diagnostic.get("reference_ready", False) and token)
        if not accepted or token == self.reference_token:
            return
        self.reference_token = token
        self.reference_pose = None
        try:
            self.reference_pose = self.tool_pose()
        except Exception:
            pass
        self.nonzero_command_seen = False
        self.command_seen_monotonic = None
        self.actual_motion_reported = False
        self.no_motion_warning_reported = False
        print("[链路 1/3] C 已被 ROS 接受，并完成机械臂参考捕获。", flush=True)

    def twist_callback(self, message):
        twist = message.twist
        linear_norm = math.sqrt(
            twist.linear.x ** 2
            + twist.linear.y ** 2
            + twist.linear.z ** 2
        )
        angular_norm = math.sqrt(
            twist.angular.x ** 2
            + twist.angular.y ** 2
            + twist.angular.z ** 2
        )
        if (
            self.reference_token
            and (linear_norm >= 0.005 or angular_norm >= 0.05)
            and not self.nonzero_command_seen
        ):
            self.nonzero_command_seen = True
            self.command_seen_monotonic = time.monotonic()
            print("[链路 2/3] 控制链已生成非零运动指令。", flush=True)

    def status_callback(self, message):
        self.servo_status = int(message.data)

    def collision_scale_callback(self, message):
        self.collision_scale = float(message.data)

    def geometry_callback(self, message):
        try:
            self.geometry = json.loads(message.data)
        except (TypeError, ValueError):
            pass

    def finger_callback(self, message):
        try:
            state = json.loads(message.data)
            self.finger_closure = state.get("normalized_robot_closure")
            self.grip_latched = bool(state.get("grip_latched"))
            self.release_phase = state.get("release_phase")
        except (TypeError, ValueError):
            pass

    def write_feedback(self):
        now = time.monotonic()
        if not self.feedback_file or now - self.last_feedback_write < 0.25:
            return
        self.last_feedback_write = now
        d, g = self.latest_diagnostic, self.geometry
        p = d.get("current_position")
        target = d.get("requested_target_position", d.get("target_position"))
        gap = None if p is None or target is None else float(np.linalg.norm(np.array(target)-p)*100)
        error = d.get("requested_rotation_error_rad", d.get("rotation_error_rad"))
        rotation_gap = (None if error is None else
                        float(np.degrees(np.linalg.norm(error))))
        approved = g.get("approved_velocity")
        command_available = bool(
            d.get("valid") and g.get("input_valid") and approved is not None
            and len(approved) == 6)
        translation_blocked = bool(command_available and gap is not None
            and gap > 1.0 and np.linalg.norm(approved[:3]) < 0.0001)
        rotation_blocked = bool(command_available and self.hand_orientation_valid is not False
            and rotation_gap is not None
            and rotation_gap > 5.0 and np.linalg.norm(approved[3:]) < 0.001)
        blocked_now = translation_blocked or rotation_blocked
        if not blocked_now:
            self.direction_blocked_since = None
        elif self.direction_blocked_since is None:
            self.direction_blocked_since = now
        direction_blocked = bool(self.direction_blocked_since is not None
                                 and now - self.direction_blocked_since >= 0.4)
        limited = bool(g.get("filtered") or (gap is not None and gap > 3))
        status, level = "正在跟随", "normal"
        if now - self.last_diagnostic_at > 1.0:
            status, level = "控制状态未更新，等待连接", "waiting"
        elif not d.get("reference_ready"):
            status, level = "张手放稳后按 C，建立零位", "waiting"
        elif self.robot_safety_status.startswith(("SELF_COLLISION:", "INVALID_JOINT_STATE:")):
            status, level = "机器人自身保护已锁定，请结束本轮并检查", "limited"
        elif (not d.get("valid") or d.get("position_reference_hold_active")
              or self.hand_input_valid is False):
            status, level = "手部输入暂不可用，保持上次目标", "limited"
        elif self.hand_safety_status.startswith("HAND_PATH_SELF_COLLISION:"):
            status, level = "手指开合受本体空间限制，请先将末端移开", "limited"
        elif self.hand_safety_status == "HAND_CHECK_EXPIRED":
            status, level = "手指安全检查暂未完成，保持原指令", "limited"
        elif self.hand_orientation_valid is False:
            status, level = "手部朝向未更新，旋转目标保持", "limited"
        elif d.get("self_detour_status", "").startswith("DETOUR_") and d.get("self_detour_status") in (
                "DETOUR_CLEAR_SELF", "DETOUR_ROTATE", "DETOUR_RETURN_TO_HAND_TARGET"):
            status, level = {
                "DETOUR_CLEAR_SELF": "正在小幅绕开自身，保留手部目标",
                "DETOUR_ROTATE": "绕开后正在调整姿态",
                "DETOUR_RETURN_TO_HAND_TARGET": "正在回到手部指定的目标",
            }[d["self_detour_status"]], "normal"
        elif d.get("self_detour_status") == "CHECKING_SHORT_SELF_DETOUR":
            status, level = "自身阻挡，正在检查小幅绕行路径", "limited"
        elif direction_blocked and d.get("self_detour_status") == "NO_SAFE_SHORT_DETOUR":
            status, level = "未找到安全的小幅绕行，请移动手部调整目标", "limited"
        elif direction_blocked:
            blocked_axes = ("平移和旋转" if translation_blocked and rotation_blocked
                            else "旋转" if rotation_blocked else "平移")
            status, level = blocked_axes + "方向受限，请反向或调整姿态", "limited"
        elif limited and g.get("self_distance_m", 1) < 0.005:
            status, level = "接近自身碰撞，请撤回或调整姿态", "limited"
        elif limited and g.get("world_distance_m", 1) < 0.03:
            status, level = "接近物体或台面，请调整接近位置", "limited"
        elif limited and (g.get("singularity_limited") or g.get("joint_limit_limited")):
            status, level = "当前姿态已接近运动边界，请撤回或调整姿态", "limited"
        elif gap is not None and gap > 5:
            status, level = "目标尚未到达，请先停手观察实际末端", "limited"
        elif rotation_gap is not None and rotation_gap > 10.0:
            status, level = "旋转目标尚未到达", "limited"
        value = dict(monotonic_s=now, status_text=status, level=level,
                     motion_direction_blocked=direction_blocked,
                     target_gap_cm=gap, rotation_gap_deg=rotation_gap,
                     orientation_input_valid=self.hand_orientation_valid,
                     input_reason=self.hand_input_reason, actual_position_m=p,
                     closure=self.finger_closure, grip_latched=self.grip_latched,
                     release_phase=self.release_phase,
                     hand_safety_status=self.hand_safety_status,
                     robot_safety_status=self.robot_safety_status)
        value["self_detour_status"] = d.get("self_detour_status", "DIRECT")
        path = Path(self.feedback_file)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False))
        os.replace(str(temporary), str(path))

    def tick(self, _event):
        self.write_feedback()
        if not self.reference_token:
            return
        try:
            position, quaternion = self.tool_pose()
        except Exception:
            return
        if self.reference_pose is None:
            self.reference_pose = (position, quaternion)
            return
        if not self.nonzero_command_seen or self.actual_motion_reported:
            return
        reference_position, reference_quaternion = self.reference_pose
        displacement = float(np.linalg.norm(position - reference_position))
        quaternion_dot = float(abs(np.dot(
            quaternion / np.linalg.norm(quaternion),
            reference_quaternion / np.linalg.norm(reference_quaternion),
        )))
        rotation_rad = 2.0 * math.acos(float(np.clip(
            quaternion_dot, 0.0, 1.0
        )))
        rotation_deg = math.degrees(rotation_rad)
        if displacement >= 0.002 or rotation_deg >= 1.0:
            self.actual_motion_reported = True
            print(
                "[链路 3/3] Gazebo 机械臂已实际移动 {:.1f} mm / {:.1f}°。".format(
                    1000.0 * displacement, rotation_deg
                ),
                flush=True,
            )
            return
        if (
            not self.no_motion_warning_reported
            and self.command_seen_monotonic is not None
            and time.monotonic() - self.command_seen_monotonic >= 3.0
        ):
            self.no_motion_warning_reported = True
            print(
                "[链路 3/3 等待] 已有运动指令，但尚未检测到 2 mm / 1° 的"
                "实际动作；请查看相机窗口中的控制状态提示。",
                flush=True,
            )


def main():
    rospy.init_node("live_teleop_terminal_monitor")
    LiveTeleopTerminalMonitor()
    print("[链路 0/3] 监视器已就绪，等待相机窗口中的 C。", flush=True)
    rospy.spin()


if __name__ == "__main__":
    main()
