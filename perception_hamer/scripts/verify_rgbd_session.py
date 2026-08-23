#!/usr/bin/env python3
"""Offline fail-closed verification for a recorded D455 RGB-D session."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.record_rgbd_session import _sha256, _write_json, summarize  # noqa: E402
from src.d455_capture import D455CaptureError, D455Frame  # noqa: E402


class SessionVerificationError(RuntimeError):
    """Raised when an on-disk session violates the replay contract."""


# Schema-v2 sessions recorded before the USB2-development/formal-USB3 split do
# not contain the later derived gate labels.  Their immutable frame evidence is
# still verifiable.  These original fields remain mandatory; newer recorded
# fields, when present, must also match the authoritative recomputation.
_SUMMARY_V2_REQUIRED_FIELDS = {
    "capture_candidate_eligible",
    "captured_frames",
    "color_frame_number_drops",
    "color_frame_number_non_increasing",
    "color_period",
    "color_timestamp_non_increasing",
    "data_integrity_pass",
    "deployment_link_pass",
    "depth_frame_number_drops",
    "depth_frame_number_non_increasing",
    "depth_period",
    "depth_timestamp_non_increasing",
    "device_to_host_time_mapping",
    "formal_dataset_eligibility_reason",
    "formal_dataset_eligible",
    "frame_rate_pass",
    "host_arrival_period",
    "host_cadence_pass",
    "host_frameset_received_monotonic_non_increasing",
    "max_device_timestamp_skew_ms",
    "maximum_allowed_skew_ms",
    "mean_valid_depth_fraction",
    "minimum_valid_depth_fraction",
    "planned_capture_complete",
    "quality_pass",
    "requested_fps",
    "requested_frames",
    "requested_period_ms",
    "session_acceptance",
    "timestamp_domain",
    "timestamp_domain_pass",
    "usb2_degraded_mode_explicitly_allowed",
    "usb_superspeed",
    "writer",
    "writer_integrity_pass",
}


def _recorded_projection_mismatches(
    recorded: Any, recomputed: Any, path: str = "summary"
) -> List[str]:
    """Compare immutable recorded evidence against an enriched recomputation.

    New derived keys may be added to the recomputed schema, including nested
    statistics such as p99.  Existing recorded keys may not disappear, change,
    or claim a different gate.  Tiny floating differences across NumPy versions
    are tolerated; booleans, integers, strings, lists, and paths remain exact.
    """

    if isinstance(recorded, dict):
        if not isinstance(recomputed, dict):
            return [path]
        mismatches: List[str] = []
        for key, value in recorded.items():
            child_path = f"{path}.{key}"
            if key not in recomputed:
                mismatches.append(child_path)
            else:
                mismatches.extend(
                    _recorded_projection_mismatches(value, recomputed[key], child_path)
                )
        return mismatches
    if isinstance(recorded, list):
        if not isinstance(recomputed, list) or len(recorded) != len(recomputed):
            return [path]
        mismatches = []
        for index, (old_value, new_value) in enumerate(zip(recorded, recomputed)):
            mismatches.extend(
                _recorded_projection_mismatches(
                    old_value, new_value, f"{path}[{index}]"
                )
            )
        return mismatches
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        return [] if type(recorded) is type(recomputed) and recorded == recomputed else [path]
    if isinstance(recorded, (int, float)) and isinstance(recomputed, (int, float)):
        if isinstance(recorded, int) and isinstance(recomputed, int):
            return [] if recorded == recomputed else [path]
        return (
            []
            if math.isfinite(float(recorded))
            and math.isfinite(float(recomputed))
            and math.isclose(
                float(recorded), float(recomputed), rel_tol=1e-12, abs_tol=1e-9
            )
            else [path]
        )
    return [] if recorded == recomputed else [path]


def _session_file(session_dir: Path, relative_value: Any, field: str) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise SessionVerificationError(f"{field} is not a safe relative path")
    candidate = (session_dir / relative).resolve()
    try:
        candidate.relative_to(session_dir.resolve())
    except ValueError as exc:
        raise SessionVerificationError(f"{field} escapes the session directory") from exc
    if not candidate.is_file():
        raise SessionVerificationError(f"{field} does not exist: {relative}")
    return candidate


def _read_image(path: Path, name: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SessionVerificationError(f"cannot decode {name}: {path.name}")
    return np.ascontiguousarray(image)


def verify_session(session_directory: Path) -> Dict[str, Any]:
    session_dir = session_directory.expanduser().resolve()
    manifest_path = session_dir / "session.json"
    frames_path = session_dir / "frames.jsonl"
    if not manifest_path.is_file() or not frames_path.is_file():
        raise SessionVerificationError("session.json or frames.jsonl is missing")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "CAPTURE_COMPLETE_UNVERIFIED"
    ):
        raise SessionVerificationError(
            "only CAPTURE_COMPLETE_UNVERIFIED schema_version=2 sessions are valid"
        )
    records: List[Dict[str, Any]] = []
    with frames_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionVerificationError(
                    f"invalid JSON at frames.jsonl line {line_number}"
                ) from exc
            records.append(record)
    if not records:
        raise SessionVerificationError("recording contains no frames")

    expected_paths = set()
    for expected_index, record in enumerate(records):
        if record.get("schema_version") != 2 or record.get("index") != expected_index:
            raise SessionVerificationError(
                f"frame index/schema mismatch at expected index {expected_index}"
            )
        paths = {
            "rgb": _session_file(session_dir, record.get("rgb_path"), "rgb_path"),
            "raw_depth": _session_file(
                session_dir, record.get("raw_depth_path"), "raw_depth_path"
            ),
            "aligned_depth": _session_file(
                session_dir, record.get("aligned_depth_path"), "aligned_depth_path"
            ),
        }
        expected_relative_paths = {
            "rgb": f"rgb/{expected_index:06d}.png",
            "raw_depth": f"raw_depth/{expected_index:06d}.png",
            "aligned_depth": f"aligned_depth/{expected_index:06d}.png",
        }
        for name, expected_relative in expected_relative_paths.items():
            if Path(str(record.get(f"{name}_path"))).as_posix() != expected_relative:
                raise SessionVerificationError(
                    f"{name}_path is not canonical at frame {expected_index}"
                )
            if paths[name] in expected_paths:
                raise SessionVerificationError(f"reused image path at frame {expected_index}")
            expected_paths.add(paths[name])
        for name, path in paths.items():
            expected_hash = record.get(f"{name}_sha256")
            if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
                raise SessionVerificationError(f"{name} hash mismatch at frame {expected_index}")
        rgb_bgr = _read_image(paths["rgb"], "RGB")
        raw_depth = _read_image(paths["raw_depth"], "raw depth")
        aligned_depth = _read_image(paths["aligned_depth"], "aligned depth")
        if rgb_bgr.dtype != np.uint8 or rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
            raise SessionVerificationError(f"RGB dtype/shape mismatch at frame {expected_index}")
        try:
            rebuilt = D455Frame(
                rgb=np.ascontiguousarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)),
                raw_depth_raw=raw_depth,
                aligned_depth_raw=aligned_depth,
                depth_scale_m_per_unit=record["depth_scale_m_per_unit"],
                raw_depth_intrinsics=record["raw_depth_intrinsics"],
                color_intrinsics=record["color_intrinsics"],
                depth_to_color_extrinsics=record["depth_to_color_extrinsics"],
                # Added after early schema-v2 recordings.  Those recordings were
                # produced only after the capture API had already enforced exact
                # aligned-depth/color intrinsics equality, so zero differences
                # are the only compatible reconstruction.
                aligned_to_color_intrinsics_differences=record.get(
                    "aligned_to_color_intrinsics_differences",
                    {
                        "width": 0.0,
                        "height": 0.0,
                        "fx": 0.0,
                        "fy": 0.0,
                        "ppx": 0.0,
                        "ppy": 0.0,
                    },
                ),
                color_frame_number=record["color_frame_number"],
                depth_frame_number=record["depth_frame_number"],
                raw_color_frame_number=record["raw_color_frame_number"],
                raw_depth_frame_number=record["raw_depth_frame_number"],
                color_timestamp_ms=record["color_timestamp_ms"],
                depth_timestamp_ms=record["depth_timestamp_ms"],
                raw_color_timestamp_ms=record["raw_color_timestamp_ms"],
                raw_depth_timestamp_ms=record["raw_depth_timestamp_ms"],
                color_timestamp_domain=record["color_timestamp_domain"],
                depth_timestamp_domain=record["depth_timestamp_domain"],
                raw_color_timestamp_domain=record["raw_color_timestamp_domain"],
                raw_depth_timestamp_domain=record["raw_depth_timestamp_domain"],
                host_monotonic_ns_before_wait=record[
                    "host_monotonic_ns_before_wait"
                ],
                host_monotonic_ns_frameset_received=record[
                    "host_monotonic_ns_frameset_received"
                ],
                host_monotonic_ns_alignment_completed=record[
                    "host_monotonic_ns_alignment_completed"
                ],
                host_wall_time_ns_alignment_completed=record[
                    "host_wall_time_ns_alignment_completed"
                ],
                device_serial=record["device_serial"],
                firmware_version=record["firmware_version"],
                usb_type_descriptor=record["usb_type_descriptor"],
            )
        except (D455CaptureError, KeyError, TypeError, ValueError) as exc:
            raise SessionVerificationError(
                f"frame/calibration contract failed at frame {expected_index}: {exc}"
            ) from exc
        rebuilt_metadata = rebuilt.metadata()
        for key, value in rebuilt_metadata.items():
            if key == "aligned_to_color_intrinsics_differences" and key not in record:
                continue
            if record.get(key) != value:
                raise SessionVerificationError(
                    f"derived/raw metadata mismatch for {key} at frame {expected_index}"
                )

    actual_png_paths = {
        path.resolve()
        for directory_name in ("rgb", "raw_depth", "aligned_depth")
        for path in (session_dir / directory_name).glob("*.png")
    }
    if actual_png_paths != expected_paths:
        raise SessionVerificationError("session contains missing or orphan PNG files")

    plan = manifest.get("capture_plan", {})
    recorded_summary = manifest.get("summary", {})
    required_manifest_fields = {
        "scenario",
        "session_name",
        "created_host_wall_time_ns",
        "completed_host_wall_time_ns",
        "device",
        "capture_plan",
        "summary",
        "data_contract",
    }
    if not required_manifest_fields.issubset(manifest):
        raise SessionVerificationError("session manifest is incomplete")
    required_plan_fields = {
        "requested_frames",
        "requested_fps",
        "maximum_skew_ms",
        "usb2_degraded_mode_explicitly_allowed",
    }
    if not required_plan_fields.issubset(plan) or "writer" not in recorded_summary:
        raise SessionVerificationError("capture plan or writer statistics are incomplete")
    device = manifest["device"]
    session_constants = {
        "device_serial": device.get("device_serial"),
        "firmware_version": device.get("firmware_version"),
        "usb_type_descriptor": device.get("usb_type_descriptor"),
        "depth_scale_m_per_unit": device.get("depth_scale_m_per_unit"),
        "raw_depth_intrinsics": device.get("raw_depth_intrinsics"),
        "color_intrinsics": device.get("color_intrinsics"),
        "depth_to_color_extrinsics": device.get("depth_to_color_extrinsics"),
    }
    for key, value in session_constants.items():
        if value is None or any(record.get(key) != value for record in records):
            raise SessionVerificationError(f"per-frame/session constant mismatch for {key}")
    recomputed = summarize(
        records,
        allow_usb2=bool(plan.get("usb2_degraded_mode_explicitly_allowed", False)),
        maximum_skew_ms=float(plan["maximum_skew_ms"]),
        writer_statistics=recorded_summary.get("writer"),
        requested_frames=int(plan["requested_frames"]),
        requested_fps=float(plan["requested_fps"]),
    )
    missing_summary_fields = _SUMMARY_V2_REQUIRED_FIELDS - set(recorded_summary)
    if missing_summary_fields:
        raise SessionVerificationError(
            "recorded summary is missing schema-v2 fields: "
            + ", ".join(sorted(missing_summary_fields))
        )
    mismatched_summary_fields = _recorded_projection_mismatches(
        recorded_summary, recomputed
    )
    if mismatched_summary_fields:
        raise SessionVerificationError(
            "recomputed summary does not match session manifest fields: "
            + ", ".join(sorted(mismatched_summary_fields))
        )
    usb3_session_candidate_verified = bool(recomputed["capture_candidate_eligible"])
    development_eligible = bool(
        recomputed["data_integrity_pass"]
        and recomputed["session_acceptance"]
        in ("PASS", "DEGRADED_USB2_ACCEPTED_FOR_DEVELOPMENT")
    )
    return {
        "schema_version": 1,
        "verification": "PASS",
        "session": str(session_dir),
        "session_json_sha256": _sha256(manifest_path),
        "frames_jsonl_sha256": _sha256(frames_path),
        "verified_frames": len(records),
        "session_acceptance": recomputed["session_acceptance"],
        "p2_development_gate": "PASS" if development_eligible else "NOT_PASS",
        "development_dataset_eligible": development_eligible,
        "phase_progression_allowed": development_eligible,
        "evidence_scope": (
            "FORMAL_USB3_ONLY"
            if recomputed["usb_superspeed"]
            else "USB2_DEVELOPMENT_ONLY"
        ),
        "usb3_session_candidate_verified": usb3_session_candidate_verified,
        "p2_formal_usb3_gate": (
            "PENDING_FULL_USB3_CERTIFICATION"
            if recomputed["usb_superspeed"]
            else "DEFERRED_USB3_UNAVAILABLE"
        ),
        "formal_P2_pass": False,
        "formal_dataset_eligible": False,
        "formal_dataset_eligibility_reason": (
            "REQUIRES_AGGREGATED_USB3_CERTIFICATION"
            if usb3_session_candidate_verified
            else "USB_LINK_OR_CAPTURE_GATE_NOT_FORMAL"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_directory")
    parser.add_argument(
        "--write-result",
        action="store_true",
        help="atomically write verification.json inside the session",
    )
    args = parser.parse_args()
    try:
        result = verify_session(Path(args.session_directory))
        if args.write_result:
            _write_json(Path(args.session_directory).resolve() / "verification.json", result)
    except (OSError, ValueError, KeyError, SessionVerificationError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
