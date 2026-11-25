#!/usr/bin/env python3
"""Adapt the guarded joint reference to the FR3 Gazebo trajectory controller."""
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from tf.transformations import quaternion_from_matrix, quaternion_matrix
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def main():
    rospy.init_node('fr3_reference_output')
    joints = rospy.get_param('/fr3_allegro/arm_controller/joints')
    arm = rospy.Publisher('/fr3_allegro/arm_controller/command', JointTrajectory, queue_size=1)
    palm = rospy.Publisher('/fr3_allegro/palm_target', PoseStamped, queue_size=1)
    listener = tf.TransformListener()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and not rospy.get_param('/fr3_allegro/ready', False):
        rate.sleep()
    if rospy.is_shutdown():
        return

    def publish_arm(message):
        count = len(joints)
        if len(message.data) != 2 * count:
            return
        # Position and signed velocity come from the same guarded integration step.
        point = JointTrajectoryPoint(positions=message.data[:count], velocities=message.data[count:],
                                     time_from_start=rospy.Duration(.016))
        arm.publish(JointTrajectory(joint_names=joints, points=[point]))

    tool_to_palm = None

    def publish_palm(target):
        nonlocal tool_to_palm
        if tool_to_palm is None:
            try:
                offset, rotation = listener.lookupTransform('tool0', 'palm_tcp', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                rospy.logwarn_throttle(2., '等待机械腕到掌心的固定变换 %s', exc)
                return
            tool_to_palm = quaternion_matrix(rotation)
            tool_to_palm[:3, 3] = offset
        p, q = target.pose.position, target.pose.orientation
        base_to_tool = quaternion_matrix((q.x, q.y, q.z, q.w))
        base_to_tool[:3, 3] = (p.x, p.y, p.z)
        base_to_palm = base_to_tool @ tool_to_palm
        out = PoseStamped()
        out.header.stamp = target.header.stamp
        out.header.frame_id = 'fr3_link0'
        out.pose.position.x, out.pose.position.y, out.pose.position.z = base_to_palm[:3, 3]
        rotation = out.pose.orientation
        rotation.x, rotation.y, rotation.z, rotation.w = quaternion_from_matrix(base_to_palm)
        palm.publish(out)

    rospy.Subscriber('/egm_position_reference/command_state', Float64MultiArray, publish_arm,
                     queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/shared_teleop/pose_target', PoseStamped, publish_palm, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
