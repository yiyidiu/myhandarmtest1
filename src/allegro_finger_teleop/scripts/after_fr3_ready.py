#!/usr/bin/env python3
"""Do not preempt the simulator's homing trajectory during controller startup."""
import os
import subprocess
import sys
import time
import rospy

def pin_gazebo_physics():
    if os.environ.get("FR3_PHYSICS_CPUS"):
        script = os.path.join(os.environ["ALLEGRO_TELEOP_PROJECT"], "scripts", "teleop_cpu_profile.sh")
        subprocess.run(["bash", "-c", 'source "$1"; teleop_physics_cpus="$2"; teleop_pin_gazebo_physics',
                        "fr3", script, os.environ["FR3_PHYSICS_CPUS"]], check=True)


def main():
    rospy.init_node("wait_for_fr3_home", anonymous=True, argv=[sys.argv[0]], disable_signals=True)
    arguments = sys.argv[1:]
    ready = "/fr3_allegro/ready"
    if arguments[0] == "--fingers":
        arguments = arguments[1:]
        ready = "/fr3_allegro/fingers_ready"
    while not rospy.is_shutdown() and not rospy.get_param(ready, False):
        time.sleep(.1)
    if not rospy.is_shutdown():
        if ready.endswith("fingers_ready"):
            pin_gazebo_physics()
        os.execvp(arguments[0], arguments)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
