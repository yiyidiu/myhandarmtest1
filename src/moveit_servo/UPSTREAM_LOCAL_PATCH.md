Source: https://codeload.github.com/moveit/moveit/tar.gz/refs/tags/1.1.16
Version: 1.1.16 (matches the installed Noetic package). Upstream BSD licenses are retained.

Local simulation-profile changes in servo_calcs.cpp/.h:
- Test actual joint-command direction on a copied RobotState instead of a 1 cm weak-direction probe that mutated the live state.
- The opt-in `inverse_damping: 0.005` profile bounds the inverse Jacobian and relies on actual joint velocity limits instead of stopping every Cartesian direction at an arbitrary condition number. Both the Cartesian geometry filter and Servo use the same inverse. The upstream condition-number behavior remains available with damping zero.
- The opt-in `directional_collision_retreat` profile removes proximity slowdown only after the actual next joint step increases active clearance without entering another collision. Actual collision stops, joint limits and the downstream full arm/hand swept gate remain active.
- Collision-body transforms are explicitly updated before calling the const FCL distance API.

The Gazebo EGM gate additionally checks already-committed position references and acceleration-limited reference motion. Its private Float64MultiArray output contains six joint velocities followed by an absolute Unix-time expiry; the EGM adapter never renews that expiry on receipt. These changes are not authorization for physical-robot operation.
