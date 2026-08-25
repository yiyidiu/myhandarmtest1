#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge MoveIt Servo joint-velocity trajectory output to a position-based
JointTrajectoryController.

Why this bridge is needed here:
- The current Gazebo URDF uses hardware_interface::PositionJointInterface.
- controller_gazebo is a position_controllers/JointTrajectoryController.
- MoveIt Servo may output meaningful joint velocities while keeping positions
  nearly constant. A position-only controller will not visibly move if only the
  velocity field changes.

This node subscribes to Servo's raw JointTrajectory, extracts the joint velocity
command, integrates it into a small position step, and publishes that position
step to /controller_gazebo/command.
"""

import math
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ServoVelocityToPositionBridge:
    def __init__(self):
        self.raw_topic = rospy.get_param("~raw_servo_topic", "/servo_server/raw_joint_cmds")
        self.controller_topic = rospy.get_param("~controller_topic", "/controller_gazebo/command")
        self.arm_joints = rospy.get_param("~arm_joints", [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ])
        self.integration_dt = float(rospy.get_param("~integration_dt", 0.12))
        self.time_from_start = float(rospy.get_param("~time_from_start", 0.10))
        self.max_joint_step = float(rospy.get_param("~max_joint_step", 0.02))

        self.current_positions = {}
        self.last_publish_time = rospy.Time(0)

        self.pub = rospy.Publisher(self.controller_topic, JointTrajectory, queue_size=1)
        rospy.Subscriber("/joint_states", JointState, self.joint_state_cb, queue_size=1)
        rospy.Subscriber(self.raw_topic, JointTrajectory, self.raw_cmd_cb, queue_size=1)

        rospy.loginfo("Servo velocity-to-position bridge started.")
        rospy.loginfo("  raw_servo_topic: %s", self.raw_topic)
        rospy.loginfo("  controller_topic: %s", self.controller_topic)
        rospy.loginfo("  arm_joints: %s", self.arm_joints)

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_positions[name] = pos

    def _clamp(self, value, limit):
        if value > limit:
            return limit
        if value < -limit:
            return -limit
        return value

    def raw_cmd_cb(self, msg):
        if not self.current_positions:
            rospy.logwarn_throttle(1.0, "Waiting for /joint_states before publishing controller commands.")
            return

        if not msg.points:
            return

        joint_names = list(msg.joint_names) if msg.joint_names else list(self.arm_joints)
        # Keep only the arm joints and preserve arm order.
        joint_names = [j for j in self.arm_joints if j in joint_names]
        if len(joint_names) != len(self.arm_joints):
            rospy.logwarn_throttle(1.0, "Raw Servo joint names do not match expected arm joints. raw=%s expected=%s",
                                   msg.joint_names, self.arm_joints)
            return

        # Use the first point with non-empty velocities. If none, fall back to the first point.
        point = None
        for p in msg.points:
            if len(p.velocities) >= len(msg.joint_names):
                point = p
                break
        if point is None:
            point = msg.points[0]

        # Map velocity array by incoming joint name.
        qdot_by_name = {}
        if len(point.velocities) >= len(msg.joint_names):
            for name, vel in zip(msg.joint_names, point.velocities):
                qdot_by_name[name] = vel
        else:
            rospy.logwarn_throttle(1.0, "Raw Servo command has no velocities. Cannot integrate to positions.")
            return

        positions = []
        velocities = []
        any_motion = False

        for name in self.arm_joints:
            if name not in self.current_positions:
                rospy.logwarn_throttle(1.0, "Joint %s not found in /joint_states yet.", name)
                return
            q = self.current_positions[name]
            qdot = qdot_by_name.get(name, 0.0)
            if not math.isfinite(qdot):
                qdot = 0.0
            step = self._clamp(qdot * self.integration_dt, self.max_joint_step)
            if abs(step) > 1e-6:
                any_motion = True
            positions.append(q + step)
            velocities.append(qdot)

        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = list(self.arm_joints)

        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.velocities = velocities
        pt.time_from_start = rospy.Duration(self.time_from_start)
        traj.points.append(pt)

        self.pub.publish(traj)
        if any_motion:
            rospy.loginfo_throttle(1.0, "Bridge publishing position steps from Servo velocities.")


def main():
    rospy.init_node("servo_velocity_to_position_bridge")
    ServoVelocityToPositionBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
