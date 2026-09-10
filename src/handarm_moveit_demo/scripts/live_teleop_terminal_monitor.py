#!/usr/bin/env python3
"""Print operator-visible proof that C reaches the simulated robot."""

import json
import math
import os
from pathlib import Path
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64, Int8, String


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
        self.geometry = {}
        self.finger_closure = None
        self.grip_latched = False
        self.release_phase = None
        self.feedback_file = rospy.get_param("~feedback_file", "")
        self.last_feedback_write = 0.0
        self.direction_blocked_since = None
        rospy.Subscriber("/shared_teleop/geometric_filter_diagnostics", String,
                         self.geometry_callback, queue_size=1)
        rospy.Subscriber("/shared_teleop/finger_diagnostics", String,
                         self.finger_callback, queue_size=1)
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
            print("[链路 2/3] MoveIt Servo 已收到非零运动指令。", flush=True)

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
        approved = g.get("approved_velocity")
        blocked_now = bool(
            d.get("valid") and g.get("input_valid") and approved is not None
            and len(approved) == 6 and gap is not None and gap > 1.0
            and np.linalg.norm(approved[:3]) < 0.0001
            and np.linalg.norm(approved[3:]) < 0.001
        )
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
        elif not d.get("valid") or d.get("position_reference_hold_active"):
            status, level = "手部输入暂不可用，机械臂保持", "limited"
        elif direction_blocked:
            status, level = "当前指令方向受限，请撤回或换方向；等待不会自动到达", "limited"
        elif limited and g.get("self_distance_m", 1) < 0.005:
            status, level = "接近自身碰撞，请撤回或调整姿态", "limited"
        elif limited and g.get("world_distance_m", 1) < 0.03:
            status, level = "接近物体或台面，请调整接近位置", "limited"
        elif limited and (g.get("singularity_limited") or g.get("joint_limit_limited")):
            status, level = "当前姿态已接近运动边界，请撤回或调整姿态", "limited"
        elif gap is not None and gap > 5:
            status, level = "目标尚未到达，请先停手观察实际末端", "limited"
        value = dict(monotonic_s=now, status_text=status, level=level,
                     motion_direction_blocked=direction_blocked,
                     target_gap_cm=gap, actual_position_m=p,
                     closure=self.finger_closure, grip_latched=self.grip_latched,
                     release_phase=self.release_phase)
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
                "实际动作；请继续做明显动作。Servo 状态={}，碰撞速度比例={:.3f}。".format(
                    self.servo_status, self.collision_scale
                ),
                flush=True,
            )


def main():
    rospy.init_node("live_teleop_terminal_monitor")
    LiveTeleopTerminalMonitor()
    print("[链路 0/3] 监视器已就绪，等待相机窗口中的 C。", flush=True)
    rospy.spin()


if __name__ == "__main__":
    main()
