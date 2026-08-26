cmake_minimum_required(VERSION 3.0.2)
project(handarm_sim_demo)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  control_msgs
  controller_manager_msgs
  gazebo_msgs
  geometry_msgs
  moveit_commander
  moveit_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_geometry_msgs
  tf2_ros
  trajectory_msgs
  visualization_msgs
)
find_package(gazebo REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(handarm_deterministic_attach_plugin
  src/deterministic_attach_world_plugin.cpp)
target_link_libraries(handarm_deterministic_attach_plugin
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(handarm_stable_hand_spring_plugin
  src/stable_hand_spring_plugin.cpp)
target_link_libraries(handarm_stable_hand_spring_plugin
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

catkin_install_python(PROGRAMS
  scripts/simulation_guard.py
  scripts/startup_coordinator.py
  scripts/scene_manager.py
  scripts/scripted_avoidance_demo.py
  scripts/multi_angle_avoidance_course.py
  scripts/hand_commander.py
  scripts/run_hand_cycle_tests.py
  scripts/run_hand_stability_tests.py
  scripts/render_teleop_hand_urdf.py
  scripts/physical_hand_trajectory_action_server.py
  scripts/validate_hand_transport_stability.py
  scripts/scripted_pick_demo.py
  scripts/grasp_candidate_quality.py
  scripts/grasp_geometry.py
  scripts/grasp_size_pose_diagnostic.py
  scripts/grasp_xy_ik_probe.py
  scripts/grasp_pose_visualizer.py
  scripts/grasp_pose_planner.py
  scripts/three_finger_grasp_demo.py
  scripts/three_finger_pick_place_demo.py
  scripts/task_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS handarm_deterministic_attach_plugin
                handarm_stable_hand_spring_plugin
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_ground_scene.py)
  catkin_add_nosetests(test/test_hand_stability_profile.py)
endif()
