#!/usr/bin/env python3
"""Move the simulated arm and left hand through their standard ROS action interfaces."""
from math import isfinite, radians
import sys
import time

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryResult, JointTolerance, JointTrajectoryControllerState
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


NS = "/fr3_allegro"
REACH = [radians(v) for v in (65.92199, -71.11189, -49.48612, -128.18966, -114.06764, 132.10537, -65.66378)]
LIFT = [radians(v) for v in (88.83213, -54.94861, -37.02535, -110.13073, -97.36297, 136.57132, -86.17151)]
OPEN = [-0.10, 0.07, 0.06, 0.06, 0.0, 0.07, 0.06, 0.06,
        0.10, 0.07, 0.06, 0.06, 0.53, 0.12, 0.13, 0.09]
CLOSED = [0.10, 0.90, 1.08, 0.83, -0.10, 0.92, 1.02, 0.79,
          -0.10, 0.84, 0.94, 0.77, 0.95, 0.54, 0.72, 0.68]


def connect(name):
    base = NS + "/" + name + "_controller"
    # An action server can exist before its controller starts, while physics is still paused.
    rospy.wait_for_message(base + "/state", JointTrajectoryControllerState, timeout=40)
    client = actionlib.SimpleActionClient(base + "/follow_joint_trajectory", FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration(35)):
        raise RuntimeError("控制器未就绪 " + base)
    return client, rospy.get_param(base + "/joints")


def wait_for_home(targets, end_stamp):
    """Accept a settled Gazebo arm and hand from measured joints before enabling teleop."""
    wanted, tolerance = {}, {}
    for (_client, names), angles in targets:
        wanted.update(zip(names, angles))
        tolerance.update((name, radians(.25 if name.startswith("fr3_") else .40)) for name in names)
    deadline = time.monotonic() + 36
    first, previous, anchor = None, None, {}
    worst = "等待关节反馈"
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        for (client, _), _angles in targets:
            status = client.get_state()
            if status in (GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.SUCCEEDED):
                continue
            result = client.get_result()
            if status != GoalStatus.ABORTED or result is None or result.error_code != FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED:
                raise RuntimeError("回初始位的轨迹中断 " + client.get_goal_status_text())
        try:
            msg = rospy.wait_for_message(NS + "/joint_states", JointState, timeout=.5)
        except rospy.ROSInterruptException:
            raise
        except rospy.ROSException:
            first = None
            continue
        stamp = msg.header.stamp.to_sec()
        q = dict(zip(msg.name, msg.position))
        if (stamp < end_stamp or not 0 <= rospy.Time.now().to_sec() - stamp <= .15
                or any(name not in q or not isfinite(q[name]) for name in wanted)):
            first = None
            continue
        worst = max(wanted, key=lambda name: abs(q[name] - wanted[name]) / tolerance[name])
        if abs(q[worst] - wanted[worst]) > tolerance[worst]:
            first = None
            continue
        if (first is None or not 0 < stamp - previous < .15
                or any(abs(q[n] - anchor[n]) > radians(.12) for n in wanted)):
            first, anchor = stamp, q
        previous = stamp
        if stamp - first >= .40:
            return
    if rospy.is_shutdown():
        raise rospy.ROSInterruptException("ROS shutdown during homing")
    raise RuntimeError("仿真臂手未在初始位持续稳定，检查关节 " + worst)


def move(label, duration, targets, wait_for_settle=False, verify_home=False):
    rospy.loginfo(label)
    start = rospy.Time.now() + rospy.Duration(0.12)
    settle_time = 25.0 if wait_for_settle else 0.0
    for (client, joints), positions in targets:
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = start
        goal.trajectory.joint_names = joints
        if wait_for_settle:
            # Allow the effort controllers to build holding torque before declaring the hand ready.
            goal.goal_tolerance = [JointTolerance(name=joint, position=radians(0.12)) for joint in joints]
            goal.goal_time_tolerance = rospy.Duration(settle_time)
        point = JointTrajectoryPoint(positions=positions, velocities=[0.0] * len(joints),
                                     time_from_start=rospy.Duration(duration))
        goal.trajectory.points = [point]
        client.send_goal(goal)
    if verify_home:
        wait_for_home(targets, start.to_sec() + duration)
        return
    for (client, _), _positions in targets:
        if not client.wait_for_result(rospy.Duration(duration + settle_time + 6.0)):
            client.cancel_goal()
            raise RuntimeError("仿真轨迹超时")
        if client.get_state() != GoalStatus.SUCCEEDED:
            result = client.get_result()
            raise RuntimeError("轨迹未完成 " + (result.error_string if result else client.get_goal_status_text()))


def main():
    rospy.init_node("fr3_allegro_motion")
    arm, hand = connect("arm"), connect("hand")
    ready = rospy.get_param(NS + "/ready_arm")
    demonstrate = rospy.get_param("~demonstrate", True)
    move("掌心朝下，中指朝前，等待四指张开的就绪位稳定", 2.8,
         [(arm, ready), (hand, OPEN)], wait_for_settle=True, verify_home=not demonstrate)
    if not demonstrate:
        rospy.set_param(NS + "/open_hand", OPEN)
        rospy.set_param(NS + "/ready", True)
        return
    rospy.sleep(0.65)
    move("机械臂伸出并转腕，机械手保持张开", 3.1, [(arm, REACH)])
    rospy.sleep(0.55)
    pinch = CLOSED[:4] + OPEN[4:12] + CLOSED[12:]
    move("机械臂保持，拇指和食指弯曲", 1.6, [(hand, pinch)])
    move("中指和环指合拢", 1.4, [(hand, CLOSED)])
    rospy.sleep(0.70)
    move("机械臂抬起并转向，四指同时张开", 3.3, [(arm, LIFT), (hand, OPEN)])
    rospy.sleep(0.6)
    move("回到掌心朝下的就绪姿态", 2.8, [(arm, ready)], wait_for_settle=True)
    rospy.loginfo("演示完成，控制器继续保持姿态。")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except (RuntimeError, rospy.ROSException) as error:
        if not rospy.is_shutdown():
            rospy.set_param(NS + "/homing_error", str(error))
            rospy.logerr(str(error))
            sys.exit(1)
