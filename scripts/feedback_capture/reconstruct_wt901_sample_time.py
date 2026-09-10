#!/usr/bin/env python3
"""Reconstruct an explicit *estimated* WT901 sample clock from raw batches.

The WT901 BLE frame has no device timestamp or sequence number.  This tool
therefore keeps the raw receive evidence unchanged and fits only an effective
uniform clock to sample order versus serial-batch arrival time.  It cannot
prove that hidden packet loss did not occur, and it does not perform the later
camera-to-IMU latency alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RAW_REQUIRED = (
    "receive_monotonic_ns",
    "receive_wall_time_ns",
    "acc_x_g",
    "acc_y_g",
    "acc_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "sample_index",
    "batch_index",
    "sample_in_batch",
    "batch_size",
    "raw_frame_hex",
)

DERIVED_FIELDS = (
    "estimated_sample_monotonic_ns",
    "estimated_sample_wall_time_ns",
    "effective_sample_period_ns",
    "batch_gap_before_ms",
    "arrival_model_residual_ms",
    "arrival_delay_proxy_ms",
    "timestamp_uncertainty_ms",
    "time_valid",
    "time_invalid_reason",
    # These stay empty until a later camera/IMU motion-alignment experiment.
    "camera_aligned_sample_monotonic_ns",
    "camera_time_offset_ns",
)


class ReconstructionError(RuntimeError):
    pass


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"minimum": None, "median": None, "p95": None, "maximum": None}
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def _load_raw(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise ReconstructionError("raw CSV is missing: " + str(path))
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        missing = [name for name in RAW_REQUIRED if name not in fields]
        if missing:
            raise ReconstructionError(
                "raw CSV lacks batch-aware fields: " + ", ".join(missing)
            )
        overlap = [name for name in DERIVED_FIELDS if name in fields]
        if overlap:
            raise ReconstructionError(
                "input already contains derived time fields: " + ", ".join(overlap)
            )
        rows = list(reader)
    if len(rows) < 20:
        raise ReconstructionError("fewer than 20 raw IMU samples")
    return rows, fields


def _validate_and_group(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows])
    if not np.array_equal(sample_indices, np.arange(len(rows))):
        raise ReconstructionError("sample_index must be contiguous from zero")

    batches: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(rows):
        batch_index = int(rows[cursor]["batch_index"])
        end = cursor + 1
        while end < len(rows) and int(rows[end]["batch_index"]) == batch_index:
            end += 1
        members = rows[cursor:end]
        if batch_index != len(batches):
            raise ReconstructionError("batch_index must be contiguous from zero")
        expected_size = len(members)
        if any(int(row["batch_size"]) != expected_size for row in members):
            raise ReconstructionError(f"batch {batch_index} has inconsistent batch_size")
        if [int(row["sample_in_batch"]) for row in members] != list(
            range(expected_size)
        ):
            raise ReconstructionError(
                f"batch {batch_index} has non-contiguous sample_in_batch"
            )
        receive_mono = {int(row["receive_monotonic_ns"]) for row in members}
        receive_wall = {int(row["receive_wall_time_ns"]) for row in members}
        if len(receive_mono) != 1 or len(receive_wall) != 1:
            raise ReconstructionError(
                f"batch {batch_index} does not share one receive timestamp"
            )
        for row in members:
            raw = row["raw_frame_hex"].strip().lower()
            if len(raw) != 40 or not raw.startswith("5561"):
                raise ReconstructionError(
                    f"sample {row['sample_index']} has an invalid raw frame"
                )
            try:
                bytes.fromhex(raw)
            except ValueError as exc:
                raise ReconstructionError(
                    f"sample {row['sample_index']} raw frame is not hex"
                ) from exc
            numeric = [float(row[name]) for name in RAW_REQUIRED[2:11]]
            if not all(math.isfinite(value) for value in numeric):
                raise ReconstructionError(
                    f"sample {row['sample_index']} contains non-finite sensor data"
                )
        batches.append(
            {
                "index": batch_index,
                "start": cursor,
                "end": end,
                "size": expected_size,
                "last_sample_index": end - 1,
                "receive_monotonic_ns": receive_mono.pop(),
                "receive_wall_time_ns": receive_wall.pop(),
            }
        )
        cursor = end

    receive = np.asarray(
        [batch["receive_monotonic_ns"] for batch in batches], dtype=np.int64
    )
    if len(batches) < 10 or np.any(np.diff(receive) <= 0):
        raise ReconstructionError("batch receive times are too few or non-monotonic")
    return batches


def _robust_clock_fit(
    batches: list[dict[str, Any]], nominal_period_s: float
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    x = np.asarray(
        [batch["last_sample_index"] for batch in batches], dtype=np.float64
    )
    receive_ns = np.asarray(
        [batch["receive_monotonic_ns"] for batch in batches], dtype=np.int64
    )
    x0 = x[0]
    y0_ns = int(receive_ns[0])
    y = (receive_ns - y0_ns).astype(np.float64) / 1.0e9
    centred_x = x - x0
    keep = np.ones(len(x), dtype=bool)
    for _ in range(10):
        design = np.column_stack((np.ones(np.count_nonzero(keep)), centred_x[keep]))
        intercept_s, period_s = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        predicted = intercept_s + period_s * centred_x
        residual_s = y - predicted
        median_s = float(np.median(residual_s[keep]))
        robust_sigma_s = float(
            1.4826 * np.median(np.abs(residual_s[keep] - median_s))
        )
        threshold_s = max(3.0 * robust_sigma_s, nominal_period_s)
        updated = np.abs(residual_s - median_s) <= threshold_s
        if np.count_nonzero(updated) < max(10, len(x) // 2):
            raise ReconstructionError("robust clock fit rejected too many batches")
        if np.array_equal(updated, keep):
            break
        keep = updated

    design = np.column_stack((np.ones(np.count_nonzero(keep)), centred_x[keep]))
    intercept_s, period_s = np.linalg.lstsq(design, y[keep], rcond=None)[0]
    predicted = intercept_s + period_s * centred_x
    residual_s = y - predicted
    median_s = float(np.median(residual_s[keep]))
    robust_sigma_s = float(
        1.4826 * np.median(np.abs(residual_s[keep] - median_s))
    )
    if not math.isfinite(period_s) or period_s <= 0.0:
        raise ReconstructionError("fitted sample period is not positive and finite")

    # Anchor the estimated sample clock no later than every observed batch.
    # The unknown minimum transport latency remains a constant-offset ambiguity
    # and is resolved only by the later camera/IMU motion-alignment experiment.
    period_ns = period_s * 1.0e9
    origins = receive_ns.astype(np.float64) - period_ns * x
    origin_ns = float(np.min(origins))
    return period_s, origin_ns, residual_s, keep, robust_sigma_s


def reconstruct(
    input_path: Path,
    output_path: Path,
    nominal_hz: float,
    maximum_rate_error_fraction: float,
    stall_periods: float,
) -> tuple[Path, Path, dict[str, Any]]:
    rows, input_fields = _load_raw(input_path)
    batches = _validate_and_group(rows)
    nominal_period_s = 1.0 / nominal_hz
    period_s, origin_ns, residual_s, fit_inliers, robust_sigma_s = _robust_clock_fit(
        batches, nominal_period_s
    )
    effective_hz = float(1.0 / period_s)
    rate_error_fraction = float((effective_hz - nominal_hz) / nominal_hz)
    rate_model_in_range = bool(
        abs(rate_error_fraction) <= maximum_rate_error_fraction
    )

    receive_ns = np.asarray(
        [batch["receive_monotonic_ns"] for batch in batches], dtype=np.int64
    )
    receive_wall_ns = np.asarray(
        [batch["receive_wall_time_ns"] for batch in batches], dtype=np.int64
    )
    batch_gaps_ms = np.empty(len(batches), dtype=np.float64)
    batch_gaps_ms[0] = np.nan
    batch_gaps_ms[1:] = np.diff(receive_ns).astype(np.float64) / 1.0e6
    residual_ms = residual_s * 1000.0
    residual_centre_ms = float(np.median(residual_ms[fit_inliers]))
    residual_threshold_ms = max(
        3.0 * robust_sigma_s * 1000.0, 2.0 * period_s * 1000.0
    )
    stall_threshold_ms = stall_periods * period_s * 1000.0
    model_inlier = np.abs(residual_ms - residual_centre_ms) <= residual_threshold_ms
    no_stall = np.ones(len(batches), dtype=bool)
    no_stall[1:] = batch_gaps_ms[1:] <= stall_threshold_ms
    batch_valid = model_inlier & no_stall & rate_model_in_range
    uncertainty_ms = float(
        np.percentile(np.abs(residual_ms[fit_inliers] - residual_centre_ms), 95)
    )
    wall_minus_mono_ns = np.median(receive_wall_ns - receive_ns)
    period_ns = int(round(period_s * 1.0e9))

    output_path = output_path.expanduser().resolve()
    summary_path = output_path.with_suffix(".summary.json")
    if output_path.exists() or summary_path.exists():
        raise ReconstructionError("output or summary already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_by_index = {batch["index"]: index for index, batch in enumerate(batches)}
    output_fields = input_fields + list(DERIVED_FIELDS)
    valid_samples = 0
    with output_path.open(
        "x", encoding="utf-8", newline="", buffering=1024 * 1024
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            sample_index = int(row["sample_index"])
            batch_position = batch_by_index[int(row["batch_index"])]
            estimated_ns = int(round(origin_ns + period_ns * sample_index))
            receive_sample_ns = int(row["receive_monotonic_ns"])
            reasons = []
            if not rate_model_in_range:
                reasons.append("EFFECTIVE_RATE_OUT_OF_RANGE")
            if not model_inlier[batch_position]:
                reasons.append("ARRIVAL_MODEL_OUTLIER")
            if not no_stall[batch_position]:
                reasons.append("LONG_RECEIVE_GAP_BEFORE_BATCH")
            valid = not reasons
            valid_samples += int(valid)
            writer.writerow(
                {
                    **row,
                    "estimated_sample_monotonic_ns": estimated_ns,
                    "estimated_sample_wall_time_ns": int(
                        round(estimated_ns + wall_minus_mono_ns)
                    ),
                    "effective_sample_period_ns": period_ns,
                    "batch_gap_before_ms": (
                        ""
                        if math.isnan(float(batch_gaps_ms[batch_position]))
                        else f"{batch_gaps_ms[batch_position]:.9f}"
                    ),
                    "arrival_model_residual_ms": (
                        f"{residual_ms[batch_position]:.9f}"
                    ),
                    "arrival_delay_proxy_ms": (
                        f"{(receive_sample_ns - estimated_ns) / 1.0e6:.9f}"
                    ),
                    "timestamp_uncertainty_ms": (
                        f"{max(uncertainty_ms, abs(residual_ms[batch_position] - residual_centre_ms)):.9f}"
                    ),
                    "time_valid": "1" if valid else "0",
                    "time_invalid_reason": "|".join(reasons),
                    "camera_aligned_sample_monotonic_ns": "",
                    "camera_time_offset_ns": "",
                }
            )

    summary = {
        "schema": "wt901_unsequenced_effective_sample_clock_v1",
        "status": "ESTIMATED_NOT_CAMERA_ALIGNED",
        "input_raw_csv": str(input_path.expanduser().resolve()),
        "output_derived_csv": str(output_path),
        "sample_count": len(rows),
        "batch_count": len(batches),
        "batch_size_distribution": {
            str(size): int(sum(batch["size"] == size for batch in batches))
            for size in sorted({batch["size"] for batch in batches})
        },
        "nominal_rate_hz": float(nominal_hz),
        "effective_rate_hz": effective_hz,
        "effective_sample_period_ns": period_ns,
        "rate_error_fraction": rate_error_fraction,
        "maximum_rate_error_fraction": float(maximum_rate_error_fraction),
        "rate_model_in_range": rate_model_in_range,
        "fit_inlier_batches": int(np.count_nonzero(fit_inliers)),
        "arrival_model_residual_ms": _percentiles(
            np.abs(residual_ms - residual_centre_ms)
        ),
        "batch_gap_ms": _percentiles(batch_gaps_ms[1:]),
        "stall_threshold_ms": stall_threshold_ms,
        "invalid_batch_count": int(np.count_nonzero(~batch_valid)),
        "valid_sample_count": valid_samples,
        "invalid_sample_count": len(rows) - valid_samples,
        "timestamp_uncertainty_ms_role": (
            "empirical arrival-model residual, not a guaranteed physical-time bound"
        ),
        "device_timestamp_available": False,
        "device_sequence_available": False,
        "hidden_packet_loss_excluded": False,
        "absolute_sensor_latency_known": False,
        "camera_time_offset_applied": False,
        "evidence_boundary": [
            "Raw receive timestamps and batch membership are preserved unchanged.",
            "Estimated sample time assumes a uniform effective cadence between received samples.",
            "The packet format cannot distinguish clock-rate error from hidden packet loss.",
            "Camera/IMU constant latency offset is still unknown and must be measured before HaMeR comparison.",
        ],
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output_path, summary_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="batch-aware raw WT901 CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal-hz", type=float, required=True)
    parser.add_argument("--maximum-rate-error-fraction", type=float, default=0.05)
    parser.add_argument(
        "--stall-periods",
        type=float,
        default=5.0,
        help="mark a batch invalid when its receive gap exceeds this many fitted periods",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.nominal_hz) or args.nominal_hz <= 0.0:
        raise SystemExit("--nominal-hz must be positive and finite")
    if not 0.0 < args.maximum_rate_error_fraction < 1.0:
        raise SystemExit("--maximum-rate-error-fraction must be in (0,1)")
    if not math.isfinite(args.stall_periods) or args.stall_periods < 2.0:
        raise SystemExit("--stall-periods must be finite and at least 2")
    try:
        output, summary_path, summary = reconstruct(
            args.input,
            args.output,
            args.nominal_hz,
            args.maximum_rate_error_fraction,
            args.stall_periods,
        )
    except (OSError, ValueError, ReconstructionError) as exc:
        print("time reconstruction failed: " + str(exc))
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "summary": str(summary_path),
                "status": summary["status"],
                "effective_rate_hz": summary["effective_rate_hz"],
                "valid_sample_count": summary["valid_sample_count"],
                "invalid_sample_count": summary["invalid_sample_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
