Source: https://codeload.github.com/moveit/moveit/tar.gz/refs/tags/1.1.16
Version: 1.1.16 (matches the installed Noetic package). Upstream BSD licenses are retained.

Local simulation-profile changes in servo_calcs.cpp/.h:
- Test actual joint-command direction on a copied RobotState instead of a 1 cm weak-direction probe that mutated the live state.
- The opt-in `inverse_damping: 0.005` profile bounds the inverse Jacobian and relies on actual joint velocity limits instead of stopping every Cartesian direction at an arbitrary condition number. Both the Cartesian geometry filter and Servo use the same inverse. The upstream condition-number behavior remains available with damping zero.
- The opt-in `directional_collision_retreat` profile removes proximity slowdown only after the actual next joint step increases active clearance without entering another collision. Actual collision stops, joint limits and the downstream full arm/hand swept gate remain active.
- Collision-body transforms are explicitly updated before calling the const FCL distance API.

September 14 configured Servo comparison (live environment truth disabled):
- The optional `stop_distance` profile uses a fixed 15 mm pre-contact margin and stopping-time factor 1.5. The initial exponential-scale profile reached wrist contact, and a 5 mm stopping-distance trial reached finger/arm contact; those trials are retained separately from the final comparison.
- `CollisionCheck` retains the upstream stopping-distance decision. A trial that held unconditionally inside the fixed margin was rejected because recorded motions stalled for most of the trial. ServoCalcs additionally permits a geometrically checked separating step when this monitor reports a pre-contact zero scale. Actual negative clearance remains blocked by this test and by the shared full arm/hand prediction.
- A released pre-contact hold clears Servo's collision halt status; it never clears the shared measured self-collision fault latch.
- The EGM comparison selects `joint_velocity_solver=servo` while retaining the modern feedback thread, hand-motion checks, prediction clocks and 250 Hz reference. It does not select the historical `legacy` controller.

The Gazebo EGM gate additionally checks already-committed position references and acceleration-limited reference motion. Its private Float64MultiArray output contains six joint velocities followed by an absolute Unix-time expiry; the EGM adapter never renews that expiry on receipt. These changes are not authorization for physical-robot operation.
