#!/usr/bin/env python3
"""Crop-only HaMeR inference without a detector, ViTPose, or renderer.

The public input image is RGB.  A caller supplies one hand bounding box and its
handedness.  Heavy dependencies are imported only when :meth:`load` is called,
so validation and preprocessing can be tested without a HaMeR installation.

HaMeR camera translation is exposed for diagnostics only.  It must not be used
as the teleoperation system's metric palm position; aligned D455 depth owns
that quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


class HamerInferenceError(RuntimeError):
    """Raised when the crop API cannot produce a trustworthy result."""


class HamerAssetError(HamerInferenceError):
    """Raised when licensed or downloaded model assets are missing."""


@dataclass(frozen=True)
class HamerInferenceResult:
    """One batch-1 HaMeR result, converted to CPU NumPy arrays.

    ``global_orient`` and ``hand_pose`` remain MANO_RIGHT-canonical rotation
    priors.  They are not a D455 palm frame and must not be sent to a robot.
    The ``*_source_camera_axes`` point sets undo the input-image reflection for
    a left hand, but remain raw HaMeR/MANO geometry without D455 extrinsic
    translation.  The API does not subtract the wrist/root joint.
    """

    timestamp: float
    timestamp_clock_domain: str
    source_frame: str
    is_right: bool
    requested_bbox_xyxy: np.ndarray
    visible_bbox_xyxy: np.ndarray
    image_size_hw: np.ndarray
    pred_vertices_mano_right_canonical: np.ndarray
    pred_keypoints_3d_mano_right_canonical: np.ndarray
    pred_vertices_source_camera_axes: np.ndarray
    pred_keypoints_3d_source_camera_axes: np.ndarray
    global_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    hamer_weak_perspective_cam: np.ndarray
    hamer_crop_projection_translation: np.ndarray
    hamer_nominal_crop_focal_length: np.ndarray
    pred_keypoints_2d_crop_normalized: np.ndarray
    inference_time_s: float
    quality: Mapping[str, Any]

    @property
    def pred_vertices(self) -> np.ndarray:
        """Required public point output, in documented source-camera axes."""

        return self.pred_vertices_source_camera_axes

    @property
    def pred_keypoints_3d(self) -> np.ndarray:
        """Required public joint output, in documented source-camera axes."""

        return self.pred_keypoints_3d_source_camera_axes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_clock_domain": self.timestamp_clock_domain,
            "source_frame": self.source_frame,
            "is_right": self.is_right,
            "requested_bbox_xyxy": self.requested_bbox_xyxy,
            "visible_bbox_xyxy": self.visible_bbox_xyxy,
            "image_size_hw": self.image_size_hw,
            "pred_vertices_mano_right_canonical": (
                self.pred_vertices_mano_right_canonical
            ),
            "pred_keypoints_3d_mano_right_canonical": (
                self.pred_keypoints_3d_mano_right_canonical
            ),
            "pred_vertices_source_camera_axes": (
                self.pred_vertices_source_camera_axes
            ),
            "pred_keypoints_3d_source_camera_axes": (
                self.pred_keypoints_3d_source_camera_axes
            ),
            "pred_vertices": self.pred_vertices,
            "pred_keypoints_3d": self.pred_keypoints_3d,
            "global_orient": self.global_orient,
            "hand_pose": self.hand_pose,
            "betas": self.betas,
            "hamer_weak_perspective_cam": self.hamer_weak_perspective_cam,
            "hamer_crop_projection_translation": (
                self.hamer_crop_projection_translation
            ),
            "hamer_nominal_crop_focal_length": (
                self.hamer_nominal_crop_focal_length
            ),
            "pred_keypoints_2d_crop_normalized": (
                self.pred_keypoints_2d_crop_normalized
            ),
            "inference_time_s": self.inference_time_s,
            "quality": dict(self.quality),
        }


def _finite_scalar(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_rgb_image(rgb: np.ndarray) -> np.ndarray:
    """Validate an RGB uint8 ndarray without silently rescaling it."""

    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a numpy.ndarray")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must have dtype uint8")
    if rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise ValueError("rgb is too small")
    return rgb


def validate_bbox(
    bbox: Sequence[float], image_shape: Sequence[int]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Validate a continuous, half-open ``[x1, y1, x2, y2)`` box.

    The returned box is the visible intersection with ``[0,W) x [0,H)``.  It
    is only a quality diagnostic: crop geometry must continue to use the
    requested box so border padding does not silently move or shrink a hand.
    """

    array = np.asarray(bbox, dtype=np.float64)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError("bbox must contain four finite values [x1,y1,x2,y2]")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height < 2 or width < 2:
        raise ValueError("invalid image dimensions")
    x1, y1, x2, y2 = array.tolist()
    requested_area = (x2 - x1) * (y2 - y1)
    if x2 <= x1 or y2 <= y1 or requested_area < 4.0:
        raise ValueError("bbox must have positive width, height, and area")
    clipped = np.array(
        [
            np.clip(x1, 0.0, float(width)),
            np.clip(y1, 0.0, float(height)),
            np.clip(x2, 0.0, float(width)),
            np.clip(y2, 0.0, float(height)),
        ],
        dtype=np.float32,
    )
    clipped_area = float(
        max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    )
    if clipped_area < 4.0:
        raise ValueError("bbox lies outside the image after clipping")
    visible_fraction = min(1.0, clipped_area / requested_area)
    return clipped, {
        "bbox_visible_fraction": float(visible_fraction),
        "bbox_area_fraction": clipped_area / float(width * height),
    }


def _expand_to_aspect_ratio(
    width: float, height: float, target_width: float, target_height: float
) -> Tuple[float, float]:
    if height / width < target_height / target_width:
        return width, width * target_height / target_width
    return height * target_width / target_height, height


def prepare_hamer_crop(
    rgb: np.ndarray,
    bbox: Sequence[float],
    is_right: bool,
    image_size: int = 256,
    bbox_shape: Sequence[float] = (192.0, 256.0),
    rescale_factor: float = 2.0,
    image_mean: Sequence[float] = (0.485, 0.456, 0.406),
    image_std: Sequence[float] = (0.229, 0.224, 0.225),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Create HaMeR's normalized CHW crop using OpenCV only.

    Left hands are flipped to the right-hand MANO convention, matching the
    official HaMeR inference dataset.  The returned affine always maps the
    *original* RGB image to the crop, including that reflection.  No detector
    or renderer is imported.
    """

    rgb = validate_rgb_image(rgb)
    if not isinstance(is_right, (bool, np.bool_)):
        raise TypeError("is_right must be bool")
    requested = np.asarray(bbox, dtype=np.float64)
    visible, diagnostics = validate_bbox(requested, rgb.shape)
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    rescale_factor = _finite_scalar(rescale_factor, "rescale_factor")
    if rescale_factor <= 0.0:
        raise ValueError("rescale_factor must be positive")
    bbox_shape_arr = np.asarray(bbox_shape, dtype=np.float64)
    if (
        bbox_shape_arr.shape != (2,)
        or not np.all(np.isfinite(bbox_shape_arr))
        or np.any(bbox_shape_arr <= 0.0)
    ):
        raise ValueError("bbox_shape must contain two positive finite values")

    x1, y1, x2, y2 = requested
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    expanded_w, expanded_h = _expand_to_aspect_ratio(
        x2 - x1, y2 - y1, bbox_shape_arr[0], bbox_shape_arr[1]
    )
    bbox_size = rescale_factor * max(expanded_w, expanded_h)

    work = rgb
    original_to_work = np.eye(3, dtype=np.float64)
    if not is_right:
        work = np.ascontiguousarray(rgb[:, ::-1, :])
        center_x = rgb.shape[1] - center_x - 1.0
        original_to_work = np.array(
            [[-1.0, 0.0, rgb.shape[1] - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    # Match the official inference preprocessing's conditional anti-aliasing
    # before a substantial downsample, without importing skimage at runtime.
    downsampling_factor = (bbox_size / float(image_size)) / 2.0
    if downsampling_factor > 1.1:
        sigma = (downsampling_factor - 1.0) / 2.0
        work = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma, sigmaY=sigma)

    src = np.array(
        [
            [center_x, center_y],
            [center_x, center_y + 0.5 * bbox_size],
            [center_x + 0.5 * bbox_size, center_y],
        ],
        dtype=np.float32,
    )
    half = 0.5 * image_size
    dst = np.array(
        [[half, half], [half, image_size], [image_size, half]], dtype=np.float32
    )
    affine_work_to_crop = cv2.getAffineTransform(src, dst)
    patch = cv2.warpAffine(
        work,
        affine_work_to_crop,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).astype(np.float32)
    mean = 255.0 * np.asarray(image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = 255.0 * np.asarray(image_std, dtype=np.float32).reshape(1, 1, 3)
    if (
        mean.shape != (1, 1, 3)
        or std.shape != (1, 1, 3)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0)
    ):
        raise ValueError("image_mean and image_std must contain three valid values")
    normalized = ((patch - mean) / std).transpose(2, 0, 1)
    affine_original_to_crop = (
        np.vstack([affine_work_to_crop, [0.0, 0.0, 1.0]]) @ original_to_work
    )[:2].astype(np.float32)
    diagnostics.update(
        {
            "crop_bbox_size_px": float(bbox_size),
            "crop_padding_expected": float(
                bbox_size > min(rgb.shape[0], rgb.shape[1])
            ),
            "bbox_coordinate_convention": "continuous_xyxy_half_open",
            "visible_bbox_xyxy": visible.astype(float).tolist(),
            "input_reflected_to_mano_right": bool(not is_right),
        }
    )
    return (
        np.ascontiguousarray(normalized, dtype=np.float32),
        affine_original_to_crop,
        diagnostics,
    )


def validate_rotation_matrices(
    matrices: np.ndarray, expected_shape: Tuple[int, ...], name: str
) -> np.ndarray:
    """Reject malformed/non-SO(3) HaMeR rotation output."""

    value = np.asarray(matrices, dtype=np.float32)
    if value.shape != expected_shape or not np.all(np.isfinite(value)):
        raise HamerInferenceError(f"{name} must be finite with shape {expected_shape}")
    flat = value.reshape(-1, 3, 3).astype(np.float64)
    identities = np.matmul(np.swapaxes(flat, -1, -2), flat)
    determinants = np.linalg.det(flat)
    if not np.allclose(identities, np.eye(3), atol=5e-3, rtol=5e-3):
        raise HamerInferenceError(f"{name} contains a non-orthonormal rotation")
    if not np.allclose(determinants, 1.0, atol=5e-3, rtol=5e-3):
        raise HamerInferenceError(f"{name} contains a rotation with det(R) != +1")
    return value.copy()


def restore_source_camera_point_axes(points: np.ndarray, is_right: bool) -> np.ndarray:
    """Undo HaMeR's left-image x reflection for native MANO point geometry.

    This helper is only for points.  Applying the same one-sided reflection to
    a rotation would produce ``det=-1`` and is intentionally unsupported.
    """

    if not isinstance(is_right, (bool, np.bool_)):
        raise TypeError("is_right must be bool")
    value = np.asarray(points, dtype=np.float32)
    if value.ndim < 2 or value.shape[-1] != 3 or not np.all(np.isfinite(value)):
        raise ValueError("points must be a finite array with final dimension 3")
    restored = value.copy()
    if not is_right:
        restored[..., 0] *= -1.0
    return restored


class HamerCropInference:
    """Lazy batch-1 HaMeR runner for a 6 GB RTX 2060.

    This class deliberately has no ROI detector and no rendering API.  Callers
    must provide a bbox and schedule HaMeR at a lower rate than the RGB-D rigid
    tracker when required by measured latency and memory.
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_root: Optional[str] = None,
        device: str = "cuda:0",
        precision: str = "fp16",
        rescale_factor: float = 2.0,
        freeze_betas: bool = True,
        minimum_visible_fraction: float = 0.60,
        source_frame: str = "unspecified_rgb_camera_frame",
        timestamp_clock_domain: str = "caller_rgb_capture_clock",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root
            else self.checkpoint_path.parent.parent.parent
        )
        if precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be 'fp32' or 'fp16'")
        self.device_name = str(device)
        self.precision = precision
        self.rescale_factor = _finite_scalar(rescale_factor, "rescale_factor")
        self.freeze_betas = bool(freeze_betas)
        self.source_frame = str(source_frame).strip()
        self.timestamp_clock_domain = str(timestamp_clock_domain).strip()
        if not self.source_frame or not self.timestamp_clock_domain:
            raise ValueError("source_frame and timestamp_clock_domain must be non-empty")
        self.minimum_visible_fraction = _finite_scalar(
            minimum_visible_fraction, "minimum_visible_fraction"
        )
        if not 0.0 <= self.minimum_visible_fraction <= 1.0:
            raise ValueError("minimum_visible_fraction must be in [0,1]")
        self._torch: Any = None
        self._model: Any = None
        self._cfg: Any = None
        self._device: Any = None
        self._perspective_projection: Any = None
        self._frozen_betas: Dict[bool, Any] = {}
        self._neutral_mano_canonical: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def asset_status(self) -> Dict[str, Any]:
        model_config = self.checkpoint_path.parent.parent / "model_config.yaml"
        mano = self.data_root / "data" / "mano" / "MANO_RIGHT.pkl"
        mean_params = self.data_root / "data" / "mano_mean_params.npz"
        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.is_file(),
            "model_config": str(model_config),
            "model_config_exists": model_config.is_file(),
            "mano_right": str(mano),
            "mano_right_exists": mano.is_file(),
            "mano_mean_params": str(mean_params),
            "mano_mean_params_exists": mean_params.is_file(),
        }

    def _require_assets(self) -> None:
        status = self.asset_status()
        missing = [
            status[key]
            for key in ("checkpoint", "model_config", "mano_right", "mano_mean_params")
            if not status[f"{key}_exists"]
        ]
        if missing:
            raise HamerAssetError("missing required HaMeR assets: " + ", ".join(missing))

    def load(self) -> None:
        """Load only HaMeR and MANO; renderer construction is disabled."""

        if self.loaded:
            return
        self._require_assets()
        try:
            import torch
            from hamer.configs import get_config
            from hamer.models import HAMER
            from hamer.utils.geometry import perspective_projection
        except Exception as exc:
            raise HamerInferenceError(
                "HaMeR runtime is unavailable; activate hamer_rtx2060"
            ) from exc

        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise HamerInferenceError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device(self.device_name)
        model_config = self.checkpoint_path.parent.parent / "model_config.yaml"
        cfg = get_config(str(model_config), update_cachedir=False)
        cfg.defrost()
        if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
            if int(cfg.MODEL.IMAGE_SIZE) != 256:
                raise HamerInferenceError("unsupported HaMeR ViT input size")
            cfg.MODEL.BBOX_SHAPE = [192, 256]
        if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
            cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        cfg.MANO.MODEL_PATH = str(self.data_root / "data" / "mano")
        cfg.MANO.MEAN_PARAMS = str(self.data_root / "data" / "mano_mean_params.npz")
        cfg.freeze()

        try:
            model = HAMER.load_from_checkpoint(
                str(self.checkpoint_path),
                strict=False,
                cfg=cfg,
                init_renderer=False,
                map_location="cpu",
            )
            model = model.to(device)
            model.eval()
        except Exception as exc:
            raise HamerInferenceError(f"failed to load HaMeR checkpoint: {exc}") from exc
        self._torch = torch
        self._model = model
        self._cfg = cfg
        self._device = device
        self._perspective_projection = perspective_projection

    def mano_faces(self) -> np.ndarray:
        """Return the fixed MANO triangle topology for lightweight display."""

        if not self.loaded:
            self.load()
        faces = np.asarray(self._model.mano.faces, dtype=np.int64)
        if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
                or np.min(faces) < 0 or np.max(faces) >= 778):
            raise HamerInferenceError("loaded MANO model has invalid triangle faces")
        return faces.copy()

    def neutral_mano_geometry(self, is_right: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Return neutral MANO vertices/joints in physical source-camera axes.

        The model is evaluated only once with identity rotations and zero
        shape.  A physical left-hand definition is the same neutral MANO
        topology reflected back from HaMeR's right-hand canonical input, just
        like live ``pred_*_source_camera_axes`` outputs.
        """

        if not isinstance(is_right, (bool, np.bool_)):
            raise TypeError("is_right must be bool")
        if not self.loaded:
            self.load()
        if self._neutral_mano_canonical is None:
            torch = self._torch
            identity = torch.eye(3, dtype=torch.float32, device=self._device)
            global_orient = identity.reshape(1, 1, 3, 3)
            hand_pose = identity.reshape(1, 1, 3, 3).repeat(1, 15, 1, 1)
            betas = torch.zeros(
                (1, int(self._model.mano.num_betas)),
                dtype=torch.float32,
                device=self._device,
            )
            try:
                with torch.inference_mode():
                    neutral = self._model.mano(
                        global_orient=global_orient,
                        hand_pose=hand_pose,
                        betas=betas,
                        pose2rot=False,
                    )
                vertices = neutral.vertices[0].detach().float().cpu().numpy()
                joints = neutral.joints[0].detach().float().cpu().numpy()
            except Exception as exc:
                raise HamerInferenceError(
                    "failed to create neutral MANO wrist definition: {}".format(exc)
                ) from exc
            if (
                vertices.shape != (778, 3)
                or joints.ndim != 2
                or joints.shape[0] < 18
                or joints.shape[1] != 3
                or not np.all(np.isfinite(vertices))
                or not np.all(np.isfinite(joints))
            ):
                raise HamerInferenceError(
                    "neutral MANO geometry has unexpected shape or non-finite values"
                )
            self._neutral_mano_canonical = (
                vertices.astype(np.float32, copy=True),
                joints.astype(np.float32, copy=True),
            )
        vertices, joints = self._neutral_mano_canonical
        return (
            restore_source_camera_point_axes(vertices, bool(is_right)),
            restore_source_camera_point_axes(joints, bool(is_right)),
        )

    def warmup(self) -> float:
        """Run the archive live viewer's one-pass model warm-up.

        The first real D455 frame must not pay CUDA kernel/module setup costs.
        The returned value is elapsed wall time in seconds and is diagnostic;
        no synthetic output is exposed to rendering or teleoperation.
        """

        if not self.loaded:
            self.load()
        torch = self._torch
        image_size = int(self._cfg.MODEL.IMAGE_SIZE)
        dummy = torch.zeros(
            (1, 3, image_size, image_size),
            device=self._device,
        )
        use_fp16 = self.precision == "fp16" and self._device.type == "cuda"
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.float16 if use_fp16 else None,
                    enabled=use_fp16,
                ):
                    self._model({"img": dummy})
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
        except torch.cuda.OutOfMemoryError as exc:
            raise HamerInferenceError("HaMeR CUDA out of memory during warm-up") from exc
        return float(time.perf_counter() - started)

    def reset_shape(self, is_right: Optional[bool] = None) -> None:
        if is_right is None:
            self._frozen_betas.clear()
        else:
            self._frozen_betas.pop(bool(is_right), None)

    def set_frozen_betas(self, is_right: bool, betas: np.ndarray) -> None:
        """Freeze a robust externally calibrated 10-D session shape."""

        if not isinstance(is_right, (bool, np.bool_)):
            raise TypeError("is_right must be bool")
        values = np.asarray(betas, dtype=np.float32)
        if values.shape != (10,) or not np.all(np.isfinite(values)):
            raise ValueError("betas must be a finite 10-vector")
        if not self.loaded:
            self.load()
        self.freeze_betas = True
        self._frozen_betas[bool(is_right)] = self._torch.from_numpy(
            values[None].copy()
        ).to(self._device)

    def frozen_betas(self, is_right: bool) -> Optional[np.ndarray]:
        value = self._frozen_betas.get(bool(is_right))
        if value is None:
            return None
        return value.detach().float().cpu().numpy()[0].copy()

    def _apply_frozen_betas(self, output: Dict[str, Any], is_right: bool) -> None:
        if not self.freeze_betas:
            return
        torch = self._torch
        mano_params = output["pred_mano_params"]
        if is_right not in self._frozen_betas:
            self._frozen_betas[is_right] = mano_params["betas"].detach().clone()
            return
        mano_params["betas"] = self._frozen_betas[is_right].to(
            device=mano_params["betas"].device, dtype=mano_params["betas"].dtype
        )
        batch_size = int(mano_params["betas"].shape[0])
        with torch.autocast(device_type=self._device.type, enabled=False):
            mano_output = self._model.mano(
                **{key: value.float() for key, value in mano_params.items()},
                pose2rot=False,
            )
        output["pred_keypoints_3d"] = mano_output.joints.reshape(batch_size, -1, 3)
        output["pred_vertices"] = mano_output.vertices.reshape(batch_size, -1, 3)

    def _forward_with_frozen_betas(
        self, batch: Dict[str, Any], is_right: bool
    ) -> Dict[str, Any]:
        """Run HaMeR once while injecting the calibrated shape before MANO.

        Upstream HaMeR evaluates MANO inside ``forward_step``. Replacing betas
        only after that call required a second MANO evaluation on every live
        frame. This batch-one equivalent preserves upstream output semantics
        while applying the frozen shape before the single MANO call.
        """

        if bool(is_right) not in self._frozen_betas:
            raise HamerInferenceError("frozen betas are unavailable")
        torch = self._torch
        model = self._model
        image = batch["img"]
        batch_size = int(image.shape[0])
        if batch_size != 1:
            raise HamerInferenceError("optimized frozen-betas path requires batch one")
        conditioning = model.backbone(image[:, :, :, 32:-32])
        predicted, pred_cam, _ = model.mano_head(conditioning)
        predicted = dict(predicted)
        template = predicted["betas"]
        predicted["betas"] = self._frozen_betas[bool(is_right)].to(
            device=template.device,
            dtype=template.dtype,
        )
        output = {
            "pred_cam": pred_cam,
            "pred_mano_params": {
                key: value.clone() for key, value in predicted.items()
            },
        }
        device = predicted["hand_pose"].device
        dtype = predicted["hand_pose"].dtype
        focal_length = (
            self._cfg.EXTRA.FOCAL_LENGTH
            * torch.ones(batch_size, 2, device=device, dtype=dtype)
        )
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2.0
                * focal_length[:, 0]
                / (
                    self._cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0]
                    + 1.0e-9
                ),
            ],
            dim=-1,
        )
        output["pred_cam_t"] = pred_cam_t
        output["focal_length"] = focal_length
        mano_params = {
            "global_orient": predicted["global_orient"].reshape(
                batch_size, -1, 3, 3
            ),
            "hand_pose": predicted["hand_pose"].reshape(
                batch_size, -1, 3, 3
            ),
            "betas": predicted["betas"].reshape(batch_size, -1),
        }
        with torch.autocast(device_type=self._device.type, enabled=False):
            mano_output = model.mano(
                **{key: value.float() for key, value in mano_params.items()},
                pose2rot=False,
            )
            joints = mano_output.joints.reshape(batch_size, -1, 3)
            output["pred_keypoints_3d"] = joints
            output["pred_vertices"] = mano_output.vertices.reshape(
                batch_size, -1, 3
            )
            output["pred_keypoints_2d"] = self._perspective_projection(
                joints,
                translation=pred_cam_t.reshape(-1, 3).float(),
                focal_length=(
                    focal_length.reshape(-1, 2).float()
                    / self._cfg.MODEL.IMAGE_SIZE
                ),
            ).reshape(batch_size, -1, 2)
        return output

    @staticmethod
    def _numpy_first(value: Any) -> np.ndarray:
        return value.detach().float().cpu().numpy()[0].copy()

    def infer(
        self, rgb: np.ndarray, bbox: Sequence[float], is_right: bool, timestamp: float
    ) -> HamerInferenceResult:
        """Run one hand crop and return MANO outputs without rendering."""

        timestamp = _finite_scalar(timestamp, "timestamp")
        if not isinstance(is_right, (bool, np.bool_)):
            raise TypeError("is_right must be bool")
        if not self.loaded:
            self.load()
        cfg = self._cfg
        crop, affine, quality = prepare_hamer_crop(
            rgb,
            bbox,
            bool(is_right),
            image_size=int(cfg.MODEL.IMAGE_SIZE),
            bbox_shape=tuple(cfg.MODEL.get("BBOX_SHAPE", [192, 256])),
            rescale_factor=self.rescale_factor,
            image_mean=tuple(cfg.MODEL.IMAGE_MEAN),
            image_std=tuple(cfg.MODEL.IMAGE_STD),
        )
        requested_bbox = np.asarray(bbox, dtype=np.float32).copy()
        visible_bbox, _ = validate_bbox(requested_bbox, rgb.shape)
        if quality["bbox_visible_fraction"] < self.minimum_visible_fraction:
            raise HamerInferenceError(
                "bbox visible fraction is below the configured quality gate"
            )

        torch = self._torch
        tensor = torch.from_numpy(crop).unsqueeze(0).to(self._device)
        batch = {"img": tensor}
        use_fp16 = self.precision == "fp16" and self._device.type == "cuda"
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.float16 if use_fp16 else None,
                    enabled=use_fp16,
                ):
                    if self.freeze_betas and bool(is_right) in self._frozen_betas:
                        output = self._forward_with_frozen_betas(
                            batch, bool(is_right)
                        )
                    else:
                        output = self._model(batch)
                        self._apply_frozen_betas(output, bool(is_right))
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
        except torch.cuda.OutOfMemoryError as exc:
            raise HamerInferenceError("HaMeR CUDA out of memory") from exc
        elapsed = time.perf_counter() - started

        mano = output["pred_mano_params"]
        vertices_native = self._numpy_first(output["pred_vertices"])
        keypoints_native = self._numpy_first(output["pred_keypoints_3d"])
        global_orient_raw = self._numpy_first(mano["global_orient"])
        hand_pose_raw = self._numpy_first(mano["hand_pose"])
        if global_orient_raw.shape != (1, 3, 3):
            raise HamerInferenceError(
                "global_orient must have native shape (1,3,3) after batch removal"
            )
        global_orient = validate_rotation_matrices(
            global_orient_raw[0], (3, 3), "global_orient"
        )
        hand_pose = validate_rotation_matrices(
            hand_pose_raw, (15, 3, 3), "hand_pose"
        )
        arrays = {
            "pred_vertices_mano_right_canonical": vertices_native,
            "pred_keypoints_3d_mano_right_canonical": keypoints_native,
            "betas": self._numpy_first(mano["betas"]),
            "hamer_weak_perspective_cam": self._numpy_first(output["pred_cam"]),
            "hamer_crop_projection_translation": self._numpy_first(
                output["pred_cam_t"]
            ),
            "hamer_nominal_crop_focal_length": self._numpy_first(
                output["focal_length"]
            ),
            "pred_keypoints_2d_crop_normalized": self._numpy_first(
                output["pred_keypoints_2d"]
            ),
        }
        expected_shapes = {
            "pred_vertices_mano_right_canonical": (778, 3),
            "pred_keypoints_3d_mano_right_canonical": (21, 3),
            "betas": (10,),
            "hamer_weak_perspective_cam": (3,),
            "hamer_crop_projection_translation": (3,),
            "hamer_nominal_crop_focal_length": (2,),
            "pred_keypoints_2d_crop_normalized": (21, 2),
        }
        if not all(
            array.shape == expected_shapes[name] and np.all(np.isfinite(array))
            for name, array in arrays.items()
        ):
            detail = {name: value.shape for name, value in arrays.items()}
            raise HamerInferenceError(
                f"HaMeR emitted NaN/Inf or an unexpected output shape: {detail}"
            )
        vertices_source = restore_source_camera_point_axes(vertices_native, is_right)
        keypoints_source = restore_source_camera_point_axes(keypoints_native, is_right)
        quality.update(
            {
                "valid": True,
                "precision": self.precision,
                "device": str(self._device),
                "betas_frozen": self.freeze_betas,
                "affine_original_to_crop": affine.astype(float).tolist(),
                "camera_translation_metric_valid": False,
                "camera_parameters_usage": "hamer_crop_projection_only",
                "camera_parameters_frame": "mano_right_canonical_crop",
                "focal_length_is_d455_intrinsics": False,
                "point_geometry_units": "mano_model_coordinates_not_d455_translation",
                "point_geometry_origin": "native_mano_output_not_root_normalized",
                "point_geometry_frame": "source_camera_axes_after_left_x_unreflection",
                "rotation_convention": "mano_right_canonical_native_prior",
                "rotation_is_d455_palm_frame": False,
                "timestamp_semantics": "caller_supplied_rgb_capture_timestamp",
            }
        )
        return HamerInferenceResult(
            timestamp=timestamp,
            timestamp_clock_domain=self.timestamp_clock_domain,
            source_frame=self.source_frame,
            is_right=bool(is_right),
            requested_bbox_xyxy=requested_bbox,
            visible_bbox_xyxy=visible_bbox,
            image_size_hw=np.asarray(rgb.shape[:2], dtype=np.int32),
            pred_vertices_source_camera_axes=vertices_source,
            pred_keypoints_3d_source_camera_axes=keypoints_source,
            global_orient=global_orient,
            hand_pose=hand_pose,
            inference_time_s=float(elapsed),
            quality=quality,
            **arrays,
        )
