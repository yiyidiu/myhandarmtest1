# perception_hamer

Low-memory, crop-only HaMeR integration for an RTX 2060 6 GB.  The runtime API
accepts RGB + bbox + handedness; it does not start ViTDet, ViTPose, Detectron2,
or a renderer.  D455 depth—not HaMeR `pred_cam_t`—provides metric palm position.

The bbox contract is continuous half-open `xyxy` in the original RGB image.
The result distinguishes requested and visible boxes and returns an
`affine_original_to_crop` transform. Left-hand point geometry is available in
both native MANO_RIGHT-canonical and x-unreflected source-camera axes. MANO
rotations remain canonical priors; they are validated as SO(3) but are not a
D455 palm frame. Raw MANO points retain the model's native origin (the API does
not subtract the wrist/root). All HaMeR camera fields are projection-only and
are neither D455 intrinsics nor metric palm translation.

## Current status

The crop preprocessing, fail-closed asset gates, official checkpoint and licensed
`MANO_RIGHT.pkl` are installed and verified. Real batch-1 FP32/FP16 inference and
Gazebo-headless coexistence benchmarks pass the 85% total-VRAM gate. The closest
case is FP16+Gazebo at 4962/6144 MiB, so GUI renderers and GPU detectors must remain
off. The HaMeR+ROI benchmark is deferred until the real P3 ROI provider exists. See
`docs/03_HAMER_INSTALL_AND_BENCHMARK.md`.

D455 capture and lossless RGB/raw-depth/aligned-depth recording are implemented
in `src/d455_capture.py` and `scripts/record_rgbd_session.py`. The connected
camera is currently on USB 2.1 through a 480M hub, so the successful short
recording is accepted only as an explicit degraded development mode. The
recorder defaults to SuperSpeed and requires `--allow-usb2` on this setup. See
`docs/04_D455_CAPTURE_AND_RECORDING.md`.

## Environment

```bash
cd /home/diu/myhandarmtest1
conda env create -f perception_hamer/environment/hamer_rtx2060.yml

conda run -n hamer_rtx2060 python -m pip install --no-deps \
  'hamer @ git+https://github.com/geopavlakos/hamer.git@3a01849f4148352e9260b69bf28b65d1671a4905'
```

The official model bundle is large (6,037,554,929 bytes). Download it from the
official HaMeR instructions, then place the separately licensed MANO model at:

```text
perception_hamer/_DATA/data/mano/MANO_RIGHT.pkl
```

Required assets:

```text
perception_hamer/_DATA/data/mano_mean_params.npz
perception_hamer/_DATA/hamer_ckpts/model_config.yaml
perception_hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
```

Do not commit or redistribute MANO unless its license permits it.

When ROS has already populated `PYTHONPATH`, remove it while freezing or testing
the standalone environment so system ROS packages are not mistaken for Conda
dependencies:

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH \
  conda run -n hamer_rtx2060 python -m pip freeze --all
```

## Tests

```bash
cd /home/diu/myhandarmtest1
python3 -m unittest discover -s perception_hamer/tests -v
```

## Archive V8.3 route (audit/reference only)

`/home/diu/teleoperation_ubuntu_core.tar.gz` was audited and its complete MANO
renderer, robust 16-vertex wrist opening, causal SO(3) filtering, and local
RGB-D forearm principles were migrated into the live runner below.  The
standalone `scripts/run_teleoperation_core_v83_observer.sh` route is retained
only for reproducibility: on this D455 view its original live forearm gate was
valid in 0/1973 frames, and its frozen four-way mutually exclusive intent model
does not satisfy the simultaneous six-axis command contract.  It is therefore
not the current acceptance command and is not connected to ROS/UDP control.

The archive SHA-256 is
`87fa1fd27adb67a07e7aaf97509837e49a831260434fa0d2bfe62d61bb783bc9`.
Its focused tests remain available through
`./scripts/test_teleoperation_core_v83.sh`; passing those tests does not claim
that the standalone observer works on the current live scene.

## Live D455 + MANO display

The HaMeR environment intentionally contains `opencv-python-headless`. The live
runner detects that build and streams its overlay to the Qt5-enabled
`mediapipe_env` display helper, so a GUI error cannot stop teleoperation UDP:

```bash
cd /home/diu/myhandarmtest1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n hamer_rtx2060 \
  python perception_hamer/scripts/run_d455_hamer_crop.py \
  --auto-roi-mediapipe \
  --mesh-renderer teleoperation-core \
  --control-reference mano-wrist-ring \
  --hand-presence-timeout-s 0.25 \
  --teleop-udp-host 127.0.0.1 \
  --teleop-udp-port 5010
```

This is the only primary live command.  It loads exactly one HaMeR model.  The
left panel is the exact inference RGB; the right panel is the complete
778-vertex/1538-face MANO rendering from the same inference frame; the appended
panel reports metric XYZ and relative pitch/yaw/roll.  Hold a neutral pose and
press `C` once to define the zero.  That zero survives a temporary no-hand
interval and an automatic reacquisition; only another `C` or process restart
changes it.  `R` only asks the ROI tracker to reacquire and does not clear zero.

Keep the complete hand and wrist inside the image.  The first screen is only a
MediaPipe preflight: HaMeR/MANO is deliberately off there.  A stable complete
hand is selected automatically and starts HaMeR; no `C` key is required.
`--require-hand-confirmation` is available only when an intentional manual
confirmation gate is desired.

MediaPipe can report both physical hands, while a single automatic active-hand
selector feeds only one crop to the single HaMeR/MANO model.  A simultaneously
visible other hand is ignored.  If the active hand disappears and the opposite
hand remains stable for three detector results, the selector automatically
clears the old track and switches handedness.

The complete live renderer from the supplied archive is now the default.  It
projects all 778 HaMeR vertices and all 1538 MANO faces with HaMeR's crop camera
onto the exact RGB frame used for inference, then shows source and mesh panels
side by side.  The former display that warped an old mesh to a newer KLT bbox
is available only through `--mesh-renderer legacy-depth`.

The translation/control reference is the archive-derived MANO wrist-opening
geometry, not a floating point in the palm. Its position is the mean of the 16
vertices on MANO's open wrist edge;
that centre is projected into the aligned D455 depth image to obtain metric
camera coordinates. Its orientation is the IRLS/Huber robust Kabsch fit from a
neutral side-specific wrist ring to the current ring, followed by the causal
SO(3) filter. A white point and RGB axes mark this reference in both display
panels. Finger articulation is not used to rebuild the live axes.

The window now reports whether the orientation is `FOREARM FUSED` or
`FOREARM MANO-ONLY`.  MediaPipe wrist/palm pixels only propose a broad local
ROI.  The actual elbow-to-wrist axis is fitted from D455 aligned metric depth
using robust point-cloud and cross-section checks, including a 19 cm wrist-local
bound that rejects a forearm connected to a face or torso.  At most 20% of that
axis regularizes MANO's longitudinal direction; MANO still supplies full wrist
orientation and all roll because one forearm axis cannot observe twist about
itself.  If depth/forearm quality fails, the exact MANO rotation is retained and
the hand is not hidden.

At startup the ROS side still requires explicit hand-reference confirmation;
that accepted wrist-ring pose becomes the relative zero. The fixed camera-axis
map then turns its relative translation and rotation into simultaneous flange
translation, pitch, yaw and roll. The legacy joint-0/MCP definition remains
available for A/B comparison with `--control-reference mano-joint-palm`.

The UDP packet retains simultaneous six-axis wrist mapping and never consumes
the archive's mutually exclusive V8.3 intent label.  It includes the raw MANO
rotation and all forearm fit/fusion diagnostics for audit.

Small D455 depth holes are searched causally around the same projected wrist
centre.  If an entire frame still has no wrist depth, the previous valid depth
may be held for at most 0.12 s at confidence 0.08 while XYZ is recomputed on the
current wrist-centre ray.  A held sample never refreshes its own age; after
0.12 s the 6-D packet becomes invalid.  Diagnostics expose
`depth_reference_hold_used` and `depth_reference_age_s`.

The response-first live ROI uses a direct `1.0` new-frame weight. Palm
orientation sent to teleoperation uses a motion- and quality-adaptive SO(3)
filter: the default 0.08 s stationary time constant attenuates MANO jitter,
while 1.5--8 degree innovations raise the gain up to 0.95 for intentional
motion. A one-frame innovation at 70 degrees is rejected. Crop border/jitter
quality still scales its confidence. A no-hand
interval resets that local filter and never reuses a held orientation.  See
[`docs/15_TELEOPERATION_UBUNTU_CORE_AUDIT.md`](../docs/15_TELEOPERATION_UBUNTU_CORE_AUDIT.md)
for the archive comparison and the reasons the V8.3 mutually exclusive model
is not connected to Servo.

MediaPipe continues checking every D455 frame independently after startup; KLT
is used only to interpolate crop motion between detector results.  One isolated
detector miss is tolerated for at most 0.08 s to prevent fast-motion flicker.
The second consecutive miss, or 0.25 s without a fresh detector result, hides
both the crop box and previous MANO mesh and suppresses teleoperation UDP.  A
hand must then be detected in two spatially consistent frames before a new KLT
track and HaMeR inference are allowed.  Use `--hand-miss-grace-frames 0` for
strict first-miss hiding.  In the window, sustained loss is directly visible as
`REAL HAND presence=NO`, `ROI ... valid=False`, and `MANO mesh=OFF`; a crop or
mesh remaining visible after the stated grace/timeout is a regression.

The exact source/mesh pair updates only when a real HaMeR inference completes;
it does not claim that a 30 Hz D455 makes the MANO model 30 Hz. This prevents a
stale pose from appearing to follow the current crop. Press `q`/Escape in the
display to stop, or `r` to manually reinitialize the ROI. Use `--no-display`
only for an intentional headless run.

## Benchmark matrix

Run both `fp32` and `fp16` for each applicable scenario.  Example:

```bash
conda run -n hamer_rtx2060 python perception_hamer/scripts/benchmark_hamer.py \
  --checkpoint perception_hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --data-root perception_hamer/_DATA \
  --precision fp16 \
  --scenario hamer_only \
  --is-right
```

For `hamer_gazebo_headless`, start `gzserver` without gzclient or RViz first.
The script refuses to label that scenario headless if either GUI process exists.
It returns nonzero unless the full iteration count completes and peak total GPU
memory measured through `nvidia-smi` remains at or below 85%.
