#!/usr/bin/env bash
set -eo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_directory/.." && pwd)"
cd "$workspace"
source /opt/ros/noetic/setup.bash

catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash

python3 -m unittest discover -s src/handarm_moveit_demo/test -p 'test_shared_teleop_core.py' -v
python3 -m unittest discover -s perception_hamer/tests -v
catkin_make run_tests_handarm_moveit_demo
catkin_test_results --all build/test_results/handarm_moveit_demo
