#!/usr/bin/env python3
"""Run the archived V8.3 observer with a 6-GiB-safe HaMeR loader.

Only checkpoint placement and CUDA autocast are adapted.  The archived MANO
geometry, V5.3 forearm estimator, V8.3 feature builder/models/thresholds,
pose overlay and observer-only output contract remain the implementation used
at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import MethodType


def _runtime_root() -> Path:
    value = os.environ.get("TELEOP_CORE_RUNTIME_ROOT", "")
    if not value:
        raise RuntimeError("TELEOP_CORE_RUNTIME_ROOT was not set by the launcher")
    root = Path(value).expanduser().resolve()
    required = root / "hamer-win" / "live_v83_pose_observer_windows.py"
    if not required.is_file():
        raise RuntimeError("invalid Teleoperation Core runtime: " + str(root))
    return root


def _install_archive_paths(root: Path) -> None:
    source = root / "hamer-win"
    legacy = root / "src" / "legacy"
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(legacy))


def _load_hamer_cpu_then_cuda(checkpoint_path: str, init_renderer: bool = False):
    """Archive load_hamer equivalent without a transient second CUDA copy."""

    import torch
    from hamer.configs import get_config
    from hamer.models import HAMER

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    model_cfg = get_config(
        str(checkpoint.parent.parent / "model_config.yaml"),
        update_cachedir=True,
    )
    if (
        model_cfg.MODEL.BACKBONE.TYPE == "vit"
        and "BBOX_SHAPE" not in model_cfg.MODEL
    ):
        model_cfg.defrost()
        if int(model_cfg.MODEL.IMAGE_SIZE) != 256:
            raise RuntimeError("archived HaMeR model image size must be 256")
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(
        str(checkpoint),
        strict=False,
        cfg=model_cfg,
        init_renderer=init_renderer,
        map_location="cpu",
    )

    original_forward = model.forward

    def _autocast_forward(self, *args, **kwargs):
        enabled = next(self.parameters()).device.type == "cuda"
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=enabled,
        ):
            return original_forward(*args, **kwargs)

    model.forward = MethodType(_autocast_forward, model)
    return model, model_cfg


def main() -> int:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    root = _runtime_root()
    _install_archive_paths(root)

    import live_v83_pose_observer_windows as pose_observer

    pose_observer.observer.load_hamer = _load_hamer_cpu_then_cuda
    archived_main = pose_observer.observer.main

    def _compatible_main() -> int:
        # The pose entry point assigns its own build label immediately before
        # invoking observer.main(), so append our loader label at that boundary.
        pose_observer.observer.BUILD_ID += "+ubuntu-6g-cpu-load-fp16-forward"
        return int(archived_main())

    pose_observer.observer.main = _compatible_main
    print(
        "6-GiB compatibility: CPU checkpoint load + CUDA FP16 autocast; "
        "V8.3 classifier/thresholds unchanged.",
        flush=True,
    )
    return int(pose_observer.main())


if __name__ == "__main__":
    raise SystemExit(main())
