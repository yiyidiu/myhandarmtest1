# ABB IRB120 MoveIt Servo Test Notes

This workspace has been extended with a first-pass real-time following chain:

```text
desired motion / target pose
        ↓
TwistStamped Cartesian velocity command
        ↓
MoveIt Servo
        ↓
/controller_gazebo/command  (JointTrajectory stream)
        ↓
Gazebo JointTrajectoryController
```

## Why output JointTrajectory instead of a JointGroupVelocityController?

The current `gazebo_handarm.urdf` exposes `hardware_interface/PositionJointInterface` transmissions.
A `velocity_controllers/JointGroupVelocityController` would require changing the URDF transmissions to
`VelocityJointInterface`. To avoid breaking the existing Gazebo/MoveIt setup, the first usable version keeps
the existing `position_controllers/JointTrajectoryController` and lets MoveIt Servo stream short
`trajectory_msgs/JointTrajectory` commands to `/controller_gazebo/command`.

## Required package

Install MoveIt Servo for ROS Noetic if it is not already installed:

```bash
sudo apt install ros-noetic-moveit-servo
```

## Build

From the workspace root:

```bash
cd ~/myhandarmtest1
catkin_make
source devel/setup.bash
```

## Run complete simulation + Servo

```bash
roslaunch abb120_moveit_config1 demo_gazebo_servo.launch
```

Wait until Gazebo, MoveIt, controllers, and `servo_server` have started.

Check controller status:

```bash
rosservice call /controller_manager/list_controllers
```

You should see `controller_gazebo` running.

## Test 1: raw velocity pulse

In a second terminal:

```bash
source ~/myhandarmtest1/devel/setup.bash
rosrun handarm_moveit_demo servo_twist_pulse_test.py _axis:=x _speed:=0.03 _duration:=2.0
```

The end effector should move smoothly for about 2 seconds and then stop.

## Test 2: small relative position target

```bash
source ~/myhandarmtest1/devel/setup.bash
roslaunch handarm_moveit_demo servo_pose_step_test.launch dx:=0.05 dy:=0.0 dz:=0.0
```

This node reads `base_link -> tool0` from TF, creates a target 5 cm away, and publishes Twist commands using
`v = Kp * (p_des - p_now)`.

## If it immediately stops

1. If the terminal reports collision halt, temporarily set `check_collisions: false` in
   `abb120_moveit_config1/config/servo_abbarm.yaml` to verify the control chain. Then fix SRDF collision pairs.
2. If the terminal reports singularity halt, try a different starting pose or reduce commanded axis/speed.
3. If `/servo_server/delta_twist_cmds` is published but `/controller_gazebo/command` is not, check that
   `moveit_servo` is installed and `servo_server` is running.
4. If `/controller_gazebo/command` is published but the arm does not move, check the controller list and whether
   `controller_gazebo` is running.
