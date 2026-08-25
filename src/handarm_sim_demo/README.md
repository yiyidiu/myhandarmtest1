# handarm_sim_demo

Simulation-only ABB IRB120 and three-finger-hand baseline for deterministic static-scene planning and scripted approach/close demonstrations.

Safety defaults are fail-closed:

- `simulation:=true`
- `use_real_robot:=false`
- `use_real_hand:=false`
- no perception or teleoperation input

The launch guard keeps the launch alive only while all three conditions remain simulation-safe. This package never launches a real ABB or real-hand driver.

Implemented entry points:

- `simulation_baseline.launch`: environment only, with optional task flags;
- `avoidance_demo.launch`: known-pose static obstacle planning and return-home;
- `hand_demo.launch`: repeated OPEN/CLOSE/neutral pre-shape checks;
- `scripted_pick_demo.launch`: known-object pregrasp, Cartesian approach and
  deterministic simulation-only 10 cm lift while retaining the initial hand shape;
- `three_finger_grasp_pose_demo.launch`: live-object-pose grasp candidate search;
- `three_finger_grasp_contact_demo.launch`: approach and exact f1/f2/f3 contact gate;
- `three_finger_pick_place_demo.launch`: contact-only physical pick, airborne hold,
  supported placement, release and retreat without a fixed attachment.

The deterministic lift uses a simulation-only fixed joint. Physical-contact grasp
is not claimed by that legacy launch. The separate three-finger pick/place launch
has passed two cold-start physical-contact runs without an attachment. See
`docs/SIM_*.md` and `results/sim_baseline/summary.json` for measured evidence.

## Running the demos

Build and source once in each new terminal:

```bash
cd /home/diu/myhandarmtest1
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

Run exactly one full-stack launch at a time. Cold MoveIt startup with Gazebo and
RViz normally takes about 20--30 seconds; the task nodes print their current wait,
planning and execution state to the terminal.

```bash
# Static double-obstacle avoidance, then return home.
roslaunch handarm_sim_demo avoidance_demo.launch \
  gazebo_gui:=true rviz:=true \
  scenario:=double_obstacle repetitions:=1

# Three-finger hand command cycles. MoveIt and RViz are intentionally not started.
roslaunch handarm_sim_demo hand_demo.launch \
  gazebo_gui:=true cycles:=1

# Known-pose pregrasp, Cartesian approach, fixed simulation attachment and lift.
roslaunch handarm_sim_demo scripted_pick_demo.launch \
  gazebo_gui:=true rviz:=true \
  grasp_mode:=deterministic_lift repetitions:=1

# Live-object-pose, exact three-finger physical pick/place (no attachment).
roslaunch handarm_sim_demo three_finger_pick_place_demo.launch \
  gazebo_gui:=true rviz:=true grasp_family:=auto
```

`deterministic_lift` keeps the startup finger angles unchanged, attaches the known
target with a Gazebo fixed joint, transfers the box to MoveIt's attached collision
objects and lifts it by 0.10 m. It is a deterministic simulation transport baseline,
not a claim of friction/contact-based physical grasping. The legacy
`grasp_mode:=approach_only` behavior remains available.

After a task prints `DONE`, Gazebo/RViz remain open for inspection. Press Ctrl-C
in that launch terminal and wait for the final `done` line before starting another
demo. Starting two full stacks at once creates duplicate `/move_group`,
`/robot_state_publisher`, Gazebo model and controller names. Automated tests that
should close immediately after the task may add `shutdown_on_task_exit:=true`.

The verified copy-and-run sequence, expected progress messages and stale-process
check are collected in `docs/SIM_FINAL_RUN_COMMANDS.md`.
