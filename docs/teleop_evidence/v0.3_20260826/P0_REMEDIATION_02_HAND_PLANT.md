# P0 remediation 02: stable finite-compliance Gazebo hand plant

This change isolates and fixes the execution-side finger oscillation measured
in the 2026-08-26 human-teleoperation evidence. It does **not** claim that the
perception, human-finger retargeting, contact grasp, or complete end-to-end
teleoperation system is acceptable yet.

## Root cause and implementation

The original v0.3 Gazebo hand split ownership across four ros_control position
loops and four independent mimic PID plugins. The low-inertia links, moving
arm base, finite efforts, and separate constraints formed a visibly excited
plant even while every commanded finger target was constant.

The default `physical_grasp` profile now:

- removes the four hand transmissions and four legacy mimic plugins only from
  the runtime-rendered `robot_description`; the canonical URDF is unchanged;
- assigns all four active and four coupled joints to one Gazebo model plugin;
- uses ODE implicit spring/damper constraints with finite effort limits;
- softens the eight finger contact surfaces to avoid a stiff contact bounce
  loop while preserving their established friction settings;
- retains the standard `FollowJointTrajectory` endpoint through a validation
  and measured-state compatibility server;
- publishes measured hand joint states and a compact physical-hand diagnostic;
- preserves `hand_stability_profile:=original` as an immediate A/B rollback.

## Reproducible A/B method

Both profiles received the same constant active-joint target
`[0.051, 0.0317, 0.0227, 0.0363] rad`. Arm joints 1, 4, and 6 received bounded
sinusoidal references for 12 seconds. Gazebo positions for all eight hand
joints were sampled 601 times at 50 Hz. Joint 1 moved through approximately
`0.602 rad` in every run.

The acceptance limits were fixed before the final run:

- every finger position peak-to-peak: `<= 0.01 rad`;
- every causal 50 Hz position-derived velocity P95: `<= 0.20 rad/s`;
- active-joint fixed-target error P95: `<= 0.01 rad`;
- mimic relation error P95: `<= 0.03 rad`;
- measured arm-joint-1 excitation range: `>= 0.40 rad`.

| Result | Original v0.3 | `physical_grasp` run 1 | `physical_grasp` run 2 |
|---|---:|---:|---:|
| Verdict | FAIL (12 conditions) | PASS | PASS |
| Samples | 601 | 601 | 601 |
| Arm joint-1 range (rad) | 0.60190 | 0.60176 | 0.60196 |
| Worst finger peak-to-peak (rad) | 0.03512 | 0.00640 | 0.00639 |
| Worst derived velocity P95 (rad/s) | 0.22746 | 0.04525 | 0.04525 |
| Worst active target-error P95 (rad) | 0.01553 | 0.00521 | 0.00525 |
| Worst mimic-error P95 (rad) | 0.01397 | 0.00313 | 0.00316 |

Local machine-readable records are intentionally ignored by Git and reside at
`.runtime/hand_transport_ab_20260826/` as `original_final50hz.json`,
`physical_grasp_tuned2_final50hz.json`, and
`physical_grasp_tuned2_repeat50hz.json`.

## Verification

- Catkin build with testing enabled: passed.
- Catkin aggregate: 98 tests, 0 errors, 0 failures, 0 skipped.
- Python compilation, launch XML parsing, and `git diff --check`: passed.
- Live Gazebo `FollowJointTrajectory` probe: action state `SUCCEEDED`, result
  code `SUCCESSFUL`; the worst of four measured final-position errors was
  `0.00261 rad` against a `0.05 rad` tolerance.
- Fixed-target transport A/B: original profile failed; the new profile passed
  twice without changing an acceptance limit.

## Measurement caveat

The primary velocity metric matches the formal evidence methodology by
causally differentiating positions sampled at 50 Hz. Gazebo's instantaneous
joint-rate service was also retained as a diagnostic and still observed
sub-sample P95 rates up to `1.59 rad/s` in the new profile (versus
`6.19 rad/s` in the original profile). This is a substantial reduction, but
it prevents claiming that all physics-step chatter is eliminated. A later
contact/load experiment must inspect physics-step effort, contact force, and
object slip before this plant is called grasp-qualified.

## Remaining P0 work

- Restore a constraint-complete MoveIt Servo/collision safety path for the
  default arm-control route.
- Implement and validate actual human-finger-to-robot-finger retargeting.
- Reduce perception latency and long HaMeR inference gaps.
- Repeat the user-controlled C-to-Q synchronized video/data trial after those
  changes, including free-space tracking, singularity, workspace boundary,
  hand loss/reacquisition, and loaded contact tests.
