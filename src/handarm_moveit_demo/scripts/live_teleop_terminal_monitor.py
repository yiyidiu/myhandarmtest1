#!/usr/bin/env python3
"""Print operator-visible proof that C reaches the simulated robot."""

import json
import math
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

    def tick(self, _event):
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
