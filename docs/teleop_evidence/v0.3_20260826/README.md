# v0.3 human-teleoperation diagnostic baseline

This directory records the immutable code and evidence identity used before
the P0 teleoperation remediation work started.

## Code provenance

- Repository: `https://github.com/yiyidiu/myhandarmtest1.git`
- Tested tag: `v0.3.0-egm-stable-hold`
- Tested commit: `4a5e195288ccd37731507f9ce3bcf04fb06a4a55`
- Formal capture date: `2026-08-26` (Asia/Shanghai)
- C-to-Q wall-clock duration: `177.36252514 s`

The ROS, Gazebo, mapping, and control stack came from the tested tag. The
camera process used the local D435i-compatible launcher. Its direct diff from
the tagged camera launcher was limited to RealSense/device/environment path
compatibility; hand selection, C-gating, orientation filtering, and HaMeR
inference behavior were unchanged.

## Baseline verdict

This baseline is not acceptable for arm-hand teleoperation:

- HaMeR input averaged `4.5408 Hz`; 29.1% of intervals exceeded `250 ms`.
- The same physical hand changed from RIGHT to LEFT without invalidating the
  active C-zero session; the maximum orientation step was `68.46 deg`.
- Position/orientation tracking error P95 was `29.88 cm / 42.54 deg`, with an
  estimated positional lag of roughly `0.35-0.37 s`.
- EGM spent `15.754 s` in nine singularity-recovery episodes.
- All four active finger targets stayed constant, but the finger-speed norm
  reached `35.47 rad/s`; arm/finger speed correlation was `0.9733`.
- Collision scaling and retreat never activated, and shared autonomy remained
  inactive for the entire run.

The detailed Chinese report is stored with the local evidence at:

```text
.runtime/v03_human_teleop_evidence_20260826T1938/review/DIAGNOSIS_REPORT_ZH.md
```

Large evidence files remain intentionally excluded from Git. Their SHA-256
identities are recorded in `EVIDENCE_MANIFEST.sha256` so a local copy can be
verified without committing hundreds of megabytes of generated data.

## Remediation scope started from this baseline

1. Bind the C-zero session to hand identity and presence generation; any
   identity discontinuity must explicitly invalidate control and require a new
   C confirmation.
2. Publish explicit INVALID heartbeats from the camera path and implement a
   real ROS-side input watchdog.
3. Restore a constraint-complete arm safety path and replace the unstable
   legacy hand plant in subsequent changes.

## Tracked remediation records

- `P0_REMEDIATION_01.md`: hand identity, presence generation, invalid
  heartbeat, and C-session fail-closed behavior.
- `P0_REMEDIATION_02_HAND_PLANT.md`: finite-compliance Gazebo hand plant and
  fixed-target arm-transport A/B evidence.
- `P0_REMEDIATION_03_ARM_SAFETY.md`: default Servo/FCL arm path, swept
  self-collision and joint-limit acceptance, PlanningScene synchronization,
  and watchdog evidence.
- `P0_REMEDIATION_04_PERCEPTION_TIMING.md`: 20 Hz D435i timing path,
  latest-only asynchronous forearm fusion, producer latency provenance, and
  startup/timeout C-token interlock evidence.

These records deliberately separate subsystem acceptance from complete human
teleoperation acceptance. A new synchronized C-to-Q human trial is still
required after hand retargeting is implemented and the complete live path is
revalidated.
