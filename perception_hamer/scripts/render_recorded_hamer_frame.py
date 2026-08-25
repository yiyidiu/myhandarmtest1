#!/usr/bin/env python3
"""Run current HaMeR and the teleoperation-core renderer on one saved RGB frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = PACKAGE_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from perception_hamer.src.hamer_crop_inference import HamerCropInference  # noqa: E402
from perception_hamer.src.mano_wrist_reference import (  # noqa: E402
    build_mano_wrist_definition,
    estimate_mano_wrist_frame,
)
from perception_hamer.src.teleop_pose_packet import (  # noqa: E402
    metric_wrist_ring_from_arrays,
)
from perception_hamer.src.teleoperation_core_mano_renderer import (  # noqa: E402
    render_inference_frame,
)


def _read_record(session: Path, requested_index: int) -> dict:
    records = session / "frames.jsonl"
    if not records.is_file():
        raise FileNotFoundError("recording has no frames.jsonl: " + str(records))
    with records.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if int(record.get("index", -1)) == requested_index:
                return record
    raise IndexError("recording frame index not found: {}".format(requested_index))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    session = args.session.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit("output already exists; use --overwrite: " + str(output))
    record = _read_record(session, args.index)
    rgb_path = session / str(record["rgb_path"])
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("failed to read recorded RGB: " + str(rgb_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    roi = record.get("roi") or {}
    bbox = record.get("bbox") or roi.get("bbox")
    if bbox is None:
        raise ValueError("record does not contain a HaMeR bbox")
    is_right = bool(roi.get("is_right", True))

    runner = HamerCropInference(
        str(PACKAGE_DIR / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"),
        data_root=str(PACKAGE_DIR / "_DATA"),
        device="cuda:0",
        precision=args.precision,
        freeze_betas=False,
        source_frame="recorded_d455_color_optical_frame",
        timestamp_clock_domain=str(record.get("timestamp_domain", "recorded")),
    )
    runner.load()
    warmup_s = runner.warmup()
    result = runner.infer(
        rgb,
        bbox,
        is_right,
        float(record.get("timestamp", 0.0)),
    )
    faces = runner.mano_faces()
    neutral_vertices, neutral_joints = runner.neutral_mano_geometry(is_right)
    wrist_definition = build_mano_wrist_definition(
        neutral_vertices, neutral_joints, faces, is_right
    )
    wrist_frame = estimate_mano_wrist_frame(
        result.pred_vertices_source_camera_axes, wrist_definition
    )
    if not wrist_frame.valid:
        raise RuntimeError(
            "recorded frame has no valid MANO wrist reference: "
            + wrist_frame.failure_reason
        )
    depth_path = session / str(record["aligned_depth_path"])
    aligned_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if aligned_depth is None or aligned_depth.dtype != np.uint16:
        raise FileNotFoundError(
            "failed to read recorded uint16 aligned depth: " + str(depth_path)
        )
    summary_path = session / "summary.json"
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    device = summary["device"]
    wrist_center_m, depth_confidence, depth_diagnostics, _ = (
        metric_wrist_ring_from_arrays(
            result.pred_vertices_mano_right_canonical,
            wrist_definition.wrist_loop,
            result.hamer_crop_projection_translation,
            result.hamer_nominal_crop_focal_length,
            result.quality["affine_original_to_crop"],
            aligned_depth,
            float(device["depth_scale_m_per_unit"]),
            device["color_intrinsics"],
        )
    )
    rendered = render_inference_frame(rgb, result, faces)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rendered.side_by_side_bgr()):
        raise RuntimeError("failed to write rendered image: " + str(output))
    print(json.dumps({
        "schema": "teleoperation_core_recorded_render_v1",
        "session": str(session),
        "index": int(args.index),
        "source_rgb": str(rgb_path),
        "output": str(output),
        "is_right": is_right,
        "inference_ms": 1000.0 * result.inference_time_s,
        "warmup_ms": 1000.0 * warmup_s,
        "vertices": int(len(result.pred_vertices_mano_right_canonical)),
        "faces": int(len(faces)),
        "control_reference": "MANO_WRIST_RING_16",
        "wrist_ring_vertex_count": int(len(wrist_definition.wrist_loop)),
        "wrist_ring_vertex_indices": wrist_definition.wrist_loop.tolist(),
        "wrist_center_d455_m": wrist_center_m.tolist(),
        "wrist_rotation_row_major": wrist_frame.rotation.reshape(-1).tolist(),
        "wrist_geometric_confidence": float(
            wrist_frame.quality["geometric_confidence"]
        ),
        "wrist_depth_confidence": float(depth_confidence),
        "wrist_depth_diagnostics": depth_diagnostics,
        "renderer": "TELEOPERATION_CORE_EXACT_FRAME",
        "robot_output": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
