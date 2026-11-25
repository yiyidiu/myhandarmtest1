"""Exercise the live entry up to tracking without opening a camera or a window."""
from concurrent.futures import Future
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from perception_hamer.scripts import run_d455_hamer_crop as live


class TrackingStarted(Exception):
    pass


class FR3CameraStartupTest(unittest.TestCase):
    def test_hidden_mesh_keeps_the_full_surface_for_wrist_measurement(self):
        runner = Mock()
        topology = np.asarray([[0, 1, 2], [2, 3, 0]])
        runner.mano_faces.return_value = topology
        runner.neutral_mano_geometry.return_value = (np.zeros((4, 3)), np.zeros((21, 3)))
        runner.warmup.return_value = .1
        args = SimpleNamespace(control_reference="mano-wrist-ring", finger_landmarks=True,
                               no_mesh_overlay=True)
        with patch.object(live, "build_mano_wrist_definition", side_effect=["left", "right"]) as build:
            _, measurement_faces, wrists = live._prepare_live_model(runner, args, live.HAMER_ROOT_WRIST)
        self.assertIs(measurement_faces, topology)
        self.assertEqual(wrists, {False: "left", True: "right"})
        self.assertTrue(all(call.args[2] is topology for call in build.call_args_list))

    def tracking_start_frame(self, robot_frame, model_frame, manual=False, keys=(3, 9)):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            ready = Path(directory) / "ready"
            stack.enter_context(patch.dict(os.environ, FR3_CAMERA_READY_FILE=str(ready)))
            argv = ["camera", "--auto-roi-mediapipe", "--fixed-left-hand", "--disable-forearm-fusion",
                    "--finger-landmarks", "--no-mesh-overlay", "--teleop-udp-host", "127.0.0.1"]
            if manual:
                argv.append("--require-hand-confirmation")
            stack.enter_context(patch.object(sys, "argv", argv))
            def instance(name):
                return stack.enter_context(patch.object(live, name)).return_value

            instance("ProcessSingleton")
            stack.enter_context(patch.object(live.socket, "socket"))
            gate = stack.enter_context(patch.object(live, "TeleopControlGate", wraps=live.TeleopControlGate))
            runner = instance("HamerCropInference")
            runner.asset_status.return_value = dict.fromkeys(
                ("checkpoint_exists", "model_config_exists", "mano_right_exists", "mano_mean_params_exists"), True)
            capture = instance("D455Capture")
            image = SimpleNamespace(rgb=np.zeros((180, 300, 3), dtype=np.uint8))
            capture.wait_for_stable_frames.return_value = capture.wait_for_frame.return_value = image
            model_executor = instance("ThreadPoolExecutor")
            startup = Future()
            model_executor.submit.return_value = startup
            instance("MediaPipeDetectionSidecar")
            detector = instance("AsyncMediaPipeDetection")
            frame = 0
            request_frames = []
            request_zero = live._request_camera_zero

            def request(control_gate, display_pose):
                request_frames.append(frame)
                return request_zero(control_gate, display_pose)

            stack.enter_context(patch.object(live, "_request_camera_zero", side_effect=request))

            def observe(previous):
                return previous + 1, {"valid": True, "is_right": False, "bbox": [10, 10, 60, 80]}, 0.

            detector.latest_after.side_effect = observe
            display = instance("LiveDisplay")
            display.stop_requested = False

            def show(_image):
                nonlocal frame
                frame += 1
                self.assertLessEqual(frame, 12)
                if frame == model_frame:
                    startup.set_result((0., None, {}))
                if frame == robot_frame:
                    ready.touch()
                return True

            display.show.side_effect = show
            display.pop_confirm_request.side_effect = lambda: frame in keys
            stack.enter_context(patch.object(live, "KLTTrackerROIProvider", side_effect=TrackingStarted))
            with self.assertRaises(TrackingStarted):
                live.main()
            model_executor.submit.assert_called_once()
            capture.start.assert_called_once()
            capture.stop.assert_called_once()
            display.close.assert_called_once()
            self.assertTrue(gate.call_args.kwargs["require_hand_landmarks"])
            return frame, request_frames

    def test_preview_while_homing_does_not_latch_early_confirm(self):
        self.assertEqual(self.tracking_start_frame(robot_frame=6, model_frame=2, manual=True), (9, [9]))

    def test_home_before_model_still_waits_for_model_to_start_tracking(self):
        self.assertEqual(self.tracking_start_frame(robot_frame=2, model_frame=6), (7, []))

    def test_one_c_on_first_ready_tracking_frame_reaches_the_operator_gate(self):
        self.assertEqual(self.tracking_start_frame(robot_frame=2, model_frame=6, keys=(3, 7)), (7, [7]))
