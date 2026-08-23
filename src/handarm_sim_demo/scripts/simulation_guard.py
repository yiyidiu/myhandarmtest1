#!/usr/bin/env python3
"""Fail-closed guard for simulation-only and attachment-isolated launches."""

import os

import rospy
from std_msgs.msg import Bool


def main():
    rospy.init_node("simulation_guard")
    simulation = bool(rospy.get_param("~simulation", True))
    use_real_robot = bool(rospy.get_param("~use_real_robot", False))
    use_real_hand = bool(rospy.get_param("~use_real_hand", False))
    grasp_mode = str(rospy.get_param("~grasp_mode", "approach_only"))
    allow_nonphysical_attachment = bool(
        rospy.get_param("~allow_nonphysical_attachment", False)
    )
    world_name = os.path.basename(str(rospy.get_param("~world_name", "")))

    if not simulation or use_real_robot or use_real_hand:
        rospy.logfatal(
            "Rejected unsafe launch arguments: simulation=%s, "
            "use_real_robot=%s, use_real_hand=%s",
            simulation,
            use_real_robot,
            use_real_hand,
        )
        raise SystemExit(2)

    nonphysical_mode = grasp_mode == "fixed_attachment_demo_nonphysical"
    nonphysical_world = world_name == "handarm_pick_obstacle_nonphysical_attachment.world"
    if nonphysical_mode != allow_nonphysical_attachment:
        rospy.logfatal(
            "Rejected attachment mode mismatch: grasp_mode=%s, "
            "allow_nonphysical_attachment=%s",
            grasp_mode,
            allow_nonphysical_attachment,
        )
        raise SystemExit(3)
    if nonphysical_world != allow_nonphysical_attachment:
        rospy.logfatal(
            "Rejected attachment world mismatch: world=%s, "
            "allow_nonphysical_attachment=%s",
            world_name,
            allow_nonphysical_attachment,
        )
        raise SystemExit(4)

    ready = rospy.Publisher(
        "/handarm_sim_demo/simulation_guard_ready", Bool, queue_size=1, latch=True
    )
    ready.publish(True)
    rospy.loginfo(
        "Simulation-only guard active; real hardware disabled; "
        "nonphysical_attachment=%s",
        allow_nonphysical_attachment,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
