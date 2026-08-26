#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/workspace.sh"
cd "$PROJECT_ROOT"
source_ros

if [[ ! -e src/CMakeLists.txt ]]; then
  catkin_init_workspace src
fi

catkin_make -DCMAKE_BUILD_TYPE=Release
source_workspace

python3 -m unittest discover -s perception_hamer/tests -v
python3 -m unittest discover -s src/handarm_sim_demo/test -v
catkin_make run_tests_handarm_moveit_demo
catkin_test_results --all build/test_results/handarm_moveit_demo
