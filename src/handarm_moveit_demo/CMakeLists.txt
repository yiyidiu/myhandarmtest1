cmake_minimum_required(VERSION 3.0.2)
project(handarm_moveit_demo)

catkin_python_setup()

find_package(catkin REQUIRED COMPONENTS
  rospy
  moveit_msgs
  geometry_msgs
  std_msgs
  std_srvs
  message_generation
)

add_message_files(
  FILES
  HamerHandPose.msg
  HandCommand.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime moveit_msgs rospy std_msgs std_srvs
)

catkin_install_python(PROGRAMS
  scripts/01_print_moveit_info.py
  scripts/02_move_arm_named.py
  scripts/03_move_arm_joint.py
  scripts/04_move_arm_relative_pose.py
  scripts/05_move_hand_named.py
  scripts/dynamic_arm_controller.py
  scripts/keyboard_ee_delta_control.py
  scripts/servo_twist_pulse_test.py
  scripts/servo_pose_step_test.py
  scripts/servo_velocity_to_position_bridge.py
  scripts/servo_dynamic_target_test.py
  scripts/servo_pose_tracking_node.py
  scripts/publish_target_pose_test.py
  scripts/publish_dynamic_target_pose_test.py
  scripts/servo_pose_tracking_node_v2.py
  scripts/joint_target_servo_controller.py
  scripts/ros_udp_target_pose_receiver_servo_local.py
  scripts/d455_conda_udp_sender_servo_v3.py
  scripts/ros_udp_target_pose_receiver_apriltag.py
  scripts/ros_udp_target_pose_receiver_apriltag_v3.py
  scripts/handarm_trajectory_rviz.py
  scripts/hamer_input_adapter.py
  scripts/six_dof_trend_node.py
  scripts/gesture_isolation_node.py
  scripts/shared_control_node.py
  scripts/moveit_servo_output_adapter.py
  scripts/mock_three_finger_hand_adapter.py
  scripts/synthetic_hamer_pose_publisher.py
  scripts/synthetic_direction_sequence.py
  scripts/teleop_pose_replay.py
  scripts/hamer_recording_replay.py
  scripts/teleop_csv_logger.py
  scripts/offline_shared_teleop_demo.py
  scripts/grasp_tolerance_data_collector.py
  scripts/check_direction_mapping.py
  scripts/direction_mapping_monitor.py
  scripts/gazebo_direction_validator.py
  scripts/gazebo_x90_response_validator.py
  scripts/live_gazebo_follow_validator.py
  scripts/udp_c_gate_validator.py
  scripts/calibrate_camera_workspace_udp.py
  scripts/analyze_irb120_ground_workspace.py
  scripts/check_ground_workspace_mapping.py
  scripts/validate_ground_workspace_gazebo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_shared_teleop_core.py)
endif()
