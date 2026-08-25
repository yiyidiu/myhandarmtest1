#!/usr/bin/env python3
"""Benchmark crop-only HaMeR latency and memory with truthful failure output."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.hamer_crop_inference import HamerCropInference  # noqa: E402


class NvidiaSmiSampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples_mib: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _read() -> Optional[float]:
        try:
            text = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            return float(text.splitlines()[0].strip())
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._read()
            if value is not None:
                self.samples_mib.append(value)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "NvidiaSmiSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def peak_mib(self) -> Optional[float]:
        return max(self.samples_mib) if self.samples_mib else None


def nvidia_smi_gpu_info() -> Dict[str, Any]:
    """Return the physical VRAM figure used by the 85% acceptance gate."""

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
        name, total, driver = [part.strip() for part in output.splitlines()[0].split(",")]
        return {
            "gpu_name": name,
            "gpu_total_mib": float(total),
            "driver_version": driver,
        }
    except (OSError, ValueError, subprocess.SubprocessError, IndexError) as exc:
        raise RuntimeError(f"cannot query physical GPU memory through nvidia-smi: {exc}")


def percentile(values: List[float], percent: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def detect_processes() -> Dict[str, bool]:
    try:
        process_text = subprocess.check_output(["ps", "-eo", "comm,args"], text=True)
    except (OSError, subprocess.SubprocessError):
        process_text = ""
    lower = process_text.lower()
    return {
        "gazebo_headless_detected": "gzserver" in lower,
        "gazebo_gui_detected": "gzclient" in lower,
        "rviz_detected": "rviz" in lower,
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    if args.input:
        bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read input image: {args.input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        # Valid for memory/latency only, never for accuracy claims.
        rgb = np.full((480, 640, 3), 127, dtype=np.uint8)
    bbox = [float(value) for value in args.bbox]
    runner = HamerCropInference(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        precision=args.precision,
        rescale_factor=args.rescale_factor,
        freeze_betas=True,
    )
    processes = detect_processes()
    result: Dict[str, Any] = {
        "scenario": args.scenario,
        "precision": args.precision,
        "batch_size": 1,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "input": args.input or "synthetic_constant_image",
        "accuracy_input": bool(args.input),
        "oom": False,
        "error": None,
        **processes,
    }
    if args.scenario == "hamer_gazebo_headless":
        if not processes["gazebo_headless_detected"]:
            raise RuntimeError("scenario requires a running gzserver")
        if processes["gazebo_gui_detected"] or processes["rviz_detected"]:
            raise RuntimeError("headless scenario forbids gzclient and RViz")
    if args.scenario == "hamer_roi":
        raise RuntimeError(
            "P3 ROI provider is not implemented; refusing to mislabel an external bbox as HaMeR+ROI"
        )

    gpu_props = torch.cuda.get_device_properties(torch.device(args.device))
    result.update(nvidia_smi_gpu_info())
    result["torch_reported_total_mib"] = gpu_props.total_memory / 2**20
    result["gpu_85_percent_limit_mib"] = 0.85 * result["gpu_total_mib"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    times: List[float] = []
    with NvidiaSmiSampler() as sampler:
        try:
            runner.load()
            for index in range(args.warmup + args.iterations):
                output = runner.infer(
                    rgb, bbox, bool(args.is_right), timestamp=time.time()
                )
                if index >= args.warmup:
                    times.append(output.inference_time_s)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            result["oom"] = "out of memory" in str(exc).lower()
            result["error"] = f"{type(exc).__name__}: {exc}"

    result.update(
        {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "nvidia_smi_peak_mib": sampler.peak_mib,
            "mean_ms": 1000.0 * statistics.fmean(times) if times else None,
            "median_ms": 1000.0 * statistics.median(times) if times else None,
            "p95_ms": 1000.0 * percentile(times, 95.0) if times else None,
            "effective_fps": 1.0 / statistics.fmean(times) if times else None,
            "successful_iterations": len(times),
        }
    )
    peak = result["nvidia_smi_peak_mib"]
    result["within_85_percent_total_vram"] = bool(
        peak is not None
        and peak <= result["gpu_85_percent_limit_mib"]
        and not result["oom"]
        and len(times) == args.iterations
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--input")
    parser.add_argument("--bbox", nargs=4, type=float, default=[160, 80, 480, 440])
    parser.add_argument("--is-right", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument(
        "--scenario",
        choices=["hamer_only", "hamer_roi", "hamer_gazebo_headless"],
        default="hamer_only",
    )
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "benchmark_results"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any]
    try:
        result = run_benchmark(args)
    except Exception as exc:
        result = {
            "scenario": args.scenario,
            "precision": args.precision,
            "batch_size": 1,
            "oom": "out of memory" in str(exc).lower(),
            "error": f"{type(exc).__name__}: {exc}",
            "successful_iterations": 0,
            "within_85_percent_total_vram": False,
        }
    stem = f"{args.scenario}_{args.precision}_batch1"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("within_85_percent_total_vram") else 2


if __name__ == "__main__":
    raise SystemExit(main())
