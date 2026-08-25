cmake_minimum_required(VERSION 3.0.2)
project(handarmtest1)

find_package(catkin REQUIRED)

catkin_package()

# This package contains the robot description and controller configuration;
# it does not build code. Install every runtime asset so the package works
# outside the source workspace as well as from devel space.
install(DIRECTORY config launch meshes urdf xacro
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
