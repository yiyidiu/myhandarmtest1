# P0 remediation 01: fail-closed camera-to-control state

This change set addresses the unsafe state-continuity defects measured in the
2026-08-26 human-teleoperation baseline. It does **not** claim that tracking or
finger dynamics are acceptable yet.

## Implemented

- Bind every operator `C` reference to presence generation, active-hand
  generation, and handedness. A reacquired, switched, or relabelled hand cannot
  inherit the old human/robot zero.
- Reject discontinuous MANO wrist innovations above the configured 45-degree
  hard threshold and require a new `C` reference.
- Emit one schema-valid UDP status on every processed camera frame. Invalid
  samples contain no pose geometry, and camera shutdown emits an immediate
  fail-closed status.
- Validate UDP schema, ordering, frame, finite pose data, identity binding, and
  the full reference token before committing receiver state.
- Publish repeated ROS fail-closed messages after 0.40 seconds without accepted
  UDP input.
- Limit simulation target replay to 0.40 seconds instead of holding an old
  target indefinitely.
- Record observed hand identity and C-reference fields in synchronized CSV
  evidence.

## Verification

- Perception/unit suite: 205 tests passed.
- Catkin package suite: 91 tests passed, 0 errors and 0 failures.
- Python compilation and `git diff --check`: passed.
- ROS integration probe: valid identity-bound packet accepted; geometry-free
  invalid heartbeat and UDP-loss watchdog both published locked
  `HamerHandPose` states.

## Still unresolved

- The measured 4.54 Hz HaMeR rate, long inference gaps, and end-to-end lag.
- The fixed-target finger oscillation caused by the v0.3 Gazebo hand plant.
- The default arm path's incomplete MoveIt Servo/collision safety coverage.
- Actual human-finger-to-robot-finger retargeting; the baseline gesture channel
  remains inactive.

The next remediation must first A/B-test a stable hand plant under a fixed
finger target while the arm moves, then restore the constraint-complete arm
safety path. No human teleoperation quality claim is justified before those
tests pass.
