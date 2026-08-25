#!/usr/bin/env python3

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.hamer_crop_inference import (  # noqa: E402
    HamerAssetError,
    HamerInferenceError,
    HamerCropInference,
    prepare_hamer_crop,
    restore_source_camera_point_axes,
    validate_bbox,
    validate_rotation_matrices,
)


class CropValidationTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        self.rgb[:, :50, 0] = 255
        self.rgb[:, 50:, 2] = 255

    def test_valid_bbox_and_quality(self):
        clipped, quality = validate_bbox([-10, 5, 60, 70], self.rgb.shape)
        np.testing.assert_allclose(clipped, [0, 5, 60, 70])
        self.assertGreater(quality["bbox_visible_fraction"], 0.0)
        self.assertLess(quality["bbox_visible_fraction"], 1.0)

    def test_invalid_bbox_rejected(self):
        for bbox in (
            [1, 1, 1, 10],
            [10, 10, 1, 20],
            [math.nan, 1, 2, 3],
            [math.inf, 1, 2, 3],
            [-20, -20, -10, -10],
        ):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                validate_bbox(bbox, self.rgb.shape)

    def test_crop_is_finite_chw(self):
        crop, affine, quality = prepare_hamer_crop(
            self.rgb, [20, 10, 80, 70], True, image_size=256
        )
        self.assertEqual(crop.shape, (3, 256, 256))
        self.assertEqual(crop.dtype, np.float32)
        self.assertEqual(affine.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(crop)))
        self.assertIn("bbox_area_fraction", quality)

    def test_left_crop_is_mirrored(self):
        right, _, _ = prepare_hamer_crop(
            self.rgb, [0, 0, 100, 80], True, image_size=64, rescale_factor=1.0
        )
        left, _, _ = prepare_hamer_crop(
            self.rgb, [0, 0, 100, 80], False, image_size=64, rescale_factor=1.0
        )
        # OpenCV's affine sampling differs at the outermost pixel after a
        # horizontal flip, so verify the handedness convention by spatial
        # colour mass instead of demanding bit-exact border interpolation.
        self.assertGreater(right[0, :, :32].mean(), right[0, :, 32:].mean())
        self.assertLess(left[0, :, :32].mean(), left[0, :, 32:].mean())
        self.assertLess(right[2, :, :32].mean(), right[2, :, 32:].mean())
        self.assertGreater(left[2, :, :32].mean(), left[2, :, 32:].mean())

    def test_left_affine_maps_original_asymmetric_bbox_center(self):
        _, affine, quality = prepare_hamer_crop(
            self.rgb, [10, 20, 40, 60], False, image_size=64, rescale_factor=1.0
        )
        original_center = np.array([25.0, 40.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(affine @ original_center, [32.0, 32.0], atol=1e-5)
        self.assertTrue(quality["input_reflected_to_mano_right"])

    def test_border_crop_keeps_requested_geometry(self):
        _, affine, quality = prepare_hamer_crop(
            self.rgb, [-20, 10, 40, 70], True, image_size=64, rescale_factor=1.0
        )
        requested_center = np.array([10.0, 40.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(affine @ requested_center, [32.0, 32.0], atol=1e-5)
        np.testing.assert_allclose(quality["visible_bbox_xyxy"], [0, 10, 40, 70])
        self.assertEqual(
            quality["bbox_coordinate_convention"], "continuous_xyxy_half_open"
        )

    def test_handedness_must_be_boolean(self):
        with self.assertRaises(TypeError):
            prepare_hamer_crop(self.rgb, [1, 1, 20, 20], "left")

    def test_dtype_and_timestamp_contracts(self):
        with self.assertRaises(ValueError):
            prepare_hamer_crop(self.rgb.astype(np.float32), [1, 1, 20, 20], True)
        with self.assertRaises(ValueError):
            prepare_hamer_crop(
                self.rgb,
                [1, 1, 20, 20],
                True,
                image_mean=(math.nan, 0.4, 0.5),
            )
        with tempfile.TemporaryDirectory() as directory:
            runner = HamerCropInference(str(Path(directory) / "hamer.ckpt"))
            with self.assertRaises(ValueError):
                runner.infer(self.rgb, [1, 1, 20, 20], True, math.nan)


class GeometryContractTest(unittest.TestCase):
    def test_rotation_shape_and_so3_contract(self):
        rotation = np.eye(3, dtype=np.float32)
        np.testing.assert_array_equal(
            validate_rotation_matrices(rotation, (3, 3), "rotation"), rotation
        )
        with self.assertRaises(HamerInferenceError):
            validate_rotation_matrices(rotation[None], (3, 3), "rotation")
        reflection = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        with self.assertRaises(HamerInferenceError):
            validate_rotation_matrices(reflection, (3, 3), "rotation")

    def test_left_point_axes_restore_is_not_a_rotation_transform(self):
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            restore_source_camera_point_axes(points, True), points
        )
        expected_left = points.copy()
        expected_left[:, 0] *= -1.0
        np.testing.assert_array_equal(
            restore_source_camera_point_axes(points, False), expected_left
        )


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeCuda:
    @staticmethod
    def synchronize(_device):
        return None


class _FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def from_numpy(value):
        return _FakeInputTensor(value)

    @staticmethod
    def zeros(shape, device=None):
        del device
        return _FakeInputTensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def inference_mode():
        return _FakeInferenceMode()

    @staticmethod
    def autocast(**_kwargs):
        return _FakeInferenceMode()


class _FakeInputTensor:
    def __init__(self, value):
        self.value = value

    def unsqueeze(self, _axis):
        return self

    def to(self, _device):
        return self


class _FakeDevice:
    type = "cpu"

    def __str__(self):
        return "cpu"


class _FakeCfgModel(dict):
    IMAGE_SIZE = 64
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)


class _FakeCfg:
    MODEL = _FakeCfgModel(BBOX_SHAPE=[48, 64])


class InferenceResultContractTest(unittest.TestCase):
    def test_dummy_model_success_path_contract(self):
        runner = HamerCropInference(
            "/unused/hamer.ckpt",
            freeze_betas=False,
            source_frame="camera_color_optical_frame",
            timestamp_clock_domain="ros_time",
        )
        runner._torch = _FakeTorch()
        runner._device = _FakeDevice()
        runner._cfg = _FakeCfg()

        model_calls = []

        def fake_model(_batch):
            model_calls.append(True)
            eye = np.eye(3, dtype=np.float32)
            return {
                "pred_vertices": _FakeTensor(np.ones((1, 778, 3))),
                "pred_keypoints_3d": _FakeTensor(np.ones((1, 21, 3))),
                "pred_mano_params": {
                    "global_orient": _FakeTensor(eye.reshape(1, 1, 3, 3)),
                    "hand_pose": _FakeTensor(np.tile(eye, (1, 15, 1, 1))),
                    "betas": _FakeTensor(np.zeros((1, 10))),
                },
                "pred_cam": _FakeTensor(np.ones((1, 3))),
                "pred_cam_t": _FakeTensor(np.ones((1, 3))),
                "focal_length": _FakeTensor(np.ones((1, 2))),
                "pred_keypoints_2d": _FakeTensor(np.zeros((1, 21, 2))),
            }

        runner._model = fake_model
        self.assertGreaterEqual(runner.warmup(), 0.0)
        rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        result = runner.infer(rgb, [-5, 10, 40, 70], False, 123.5)
        self.assertEqual(result.global_orient.shape, (3, 3))
        self.assertEqual(result.hand_pose.shape, (15, 3, 3))
        self.assertEqual(result.pred_vertices.shape, (778, 3))
        self.assertEqual(result.pred_keypoints_2d_crop_normalized.shape, (21, 2))
        self.assertTrue(np.all(result.pred_vertices[:, 0] == -1.0))
        self.assertEqual(result.source_frame, "camera_color_optical_frame")
        self.assertEqual(result.timestamp_clock_domain, "ros_time")
        self.assertFalse(result.quality["camera_translation_metric_valid"])
        self.assertIn("pred_vertices", result.as_dict())
        self.assertEqual(len(model_calls), 2)


class AssetGateTest(unittest.TestCase):
    def test_missing_licensed_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
            runner = HamerCropInference(str(checkpoint), data_root=directory)
            status = runner.asset_status()
            self.assertFalse(status["checkpoint_exists"])
            self.assertFalse(status["mano_right_exists"])
            with self.assertRaises(HamerAssetError):
                runner.load()


if __name__ == "__main__":
    unittest.main()
