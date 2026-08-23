#!/usr/bin/env python3
"""Asynchronous live P5: D455 -> palm ROI -> RGB-D Kabsch.

HaMeR runs on a separate latest-only mailbox and can publish only hand_pose,
gesture and ROI context. Its global/root orientation is discarded and cannot
enter pairwise or accumulated RGB-D orientation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import resource
import sys
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(ROOT))

from perception_hamer.src.d455_capture import D455Capture
from perception_hamer.src.hamer_crop_inference import HamerCropInference
from perception_hamer.src.p5_async_runtime import (
    HamerContextState, LatestOnlySlot, P5CapturePacket, SequentialCaptureQueue,
)
from perception_hamer.src.rgbd_rigid_tracker import (
    RGBDRelativeOrientationTracker, RGBDRigidTrackerConfig,
    build_rigid_palm_mask, rgbd_tracker_frame_from_d455, robust_palm_center_m,
)
from perception_hamer.src.roi_provider import KLTTrackerROIProvider


SCENARIOS = ("P5_STATIC", "P5_TRANSLATION", "P5_ROTATION", "P5_GESTURE")
SCENARIO_INSTRUCTIONS = {
    "P5_STATIC": "Keep the hand shape and wrist completely still.",
    "P5_TRANSLATION": "Keep wrist orientation fixed; translate left/right, up/down, then forward/back.",
    "P5_ROTATION": "Keep the palm center nearly fixed; rotate about camera X, Y, then Z in three equal time segments.",
    "P5_GESTURE": "Keep palm pose fixed; repeatedly open and close the hand.",
}


def safe(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    return value


def expand_bbox(bbox: Any, shape: Any, factor: float = 2.2) -> np.ndarray:
    box = np.asarray(bbox, dtype=np.float64)
    center = 0.5 * (box[:2] + box[2:]); extent = factor * (box[2:] - box[:2])
    result = np.concatenate((center - extent / 2, center + extent / 2))
    result[[0, 2]] = np.clip(result[[0, 2]], 0, shape[1])
    result[[1, 3]] = np.clip(result[[1, 3]], 0, shape[0])
    return result


def select_roi(rgb: np.ndarray) -> np.ndarray:
    import subprocess
    helper = SCRIPT_DIR / "manual_select_roi_once.py"
    process = subprocess.run([
        "/home/diu/anaconda3/envs/mediapipe_env/bin/python", str(helper),
        "--width", str(rgb.shape[1]), "--height", str(rgb.shape[0])],
        input=rgb.tobytes(), capture_output=True, timeout=120)
    payload = json.loads(process.stdout.decode().strip().splitlines()[-1])
    if process.returncode or not payload.get("valid"):
        raise RuntimeError("manual palm ROI cancelled")
    return np.asarray(payload["bbox"], dtype=np.float64)


def capture_worker(capture: D455Capture, initial_roi: np.ndarray,
                   fifo: SequentialCaptureQueue, latest: LatestOnlySlot,
                   stop: threading.Event, stats: dict) -> None:
    tracker = KLTTrackerROIProvider(
        initial_bbox=initial_roi, bbox_smoothing_alpha=0.35,
        minimum_visible_fraction=0.40, min_tracked_points=8)
    initialized = False
    try:
        while not stop.is_set():
            frame = capture.wait_for_frame()
            roi = (tracker.reinitialize(frame.rgb, initial_roi)
                   if not initialized else tracker.update(frame.rgb))
            if roi.lost or roi.bbox is None:
                stats["roi_lost"] += 1
                raise RuntimeError(
                    "PALM_ROI_LOST_REQUIRES_REDETECTION_AND_NEW_CLUTCH"
                )
            initialized = True
            packet = P5CapturePacket(
                frame=frame, palm_roi=roi,
                hand_roi=expand_bbox(roi.bbox, frame.rgb.shape),
                sequence=stats["captured"])
            stats["captured"] += 1
            fifo.publish(packet)
            latest.publish(packet)
    except BaseException as exc:
        stats["error"] = f"{type(exc).__name__}:{exc}"
        stop.set()
    finally:
        latest.close()


def rotation_change_deg(previous: np.ndarray, current: np.ndarray) -> float:
    relative = np.swapaxes(previous, -1, -2) @ current
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1.0)/2.0, -1, 1)
    return float(np.percentile(np.degrees(np.arccos(cosine)), 75))


def hamer_worker(runner: HamerCropInference, latest: LatestOnlySlot,
                 context: HamerContextState, stop: threading.Event,
                 is_right: bool) -> None:
    version = 0; previous_pose: Optional[np.ndarray] = None
    while not stop.is_set():
        try: version, packet = latest.get_after(version, 1.0)
        except TimeoutError: continue
        if packet is None: break
        try:
            output = runner.infer(
                packet.frame.rgb, packet.hand_roi, is_right,
                packet.frame.color_timestamp_ms / 1000.0)
            gesture_score = (0.0 if previous_pose is None else
                             rotation_change_deg(previous_pose, output.hand_pose))
            previous_pose = output.hand_pose.copy()
            # Deliberately omit output.global_orient, vertices and palm frames.
            context.update({
                "valid": True,
                "hand_pose": output.hand_pose.tolist(),
                "gesture_changing": gesture_score > 7.5,
                "gesture_change_p75_deg": gesture_score,
                "timestamp": output.timestamp,
                "inference_ms": output.inference_time_s * 1000.0,
                "failure_reason": "NONE",
                "usage": "ROI_HAND_POSE_GESTURE_ONLY",
            })
        except Exception as exc:
            context.update({"valid": False, "hand_pose": None,
                "gesture_changing": False, "timestamp": None,
                "inference_ms": None,
                "failure_reason": f"{type(exc).__name__}:{exc}"})


class AsyncWriter:
    def __init__(self, output: Path) -> None:
        self.output = output; self.queue: queue.Queue = queue.Queue(maxsize=64)
        self.error: Optional[BaseException] = None; self.stop = False
        self.handle = (output / "frames.jsonl").open("x", encoding="utf-8")
        self.video = cv2.VideoWriter(str(output / "tracking_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640,480))
        if not self.video.isOpened(): raise RuntimeError("video open failed")
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def submit(self, index: int, rgb: np.ndarray, depth: np.ndarray,
               overlay: np.ndarray, record: dict) -> None:
        if self.error: raise RuntimeError("writer failed") from self.error
        self.queue.put((index, rgb.copy(), depth.copy(), overlay.copy(), record), timeout=2)

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None: break
                index,rgb,depth,overlay,record=item
                rp=f"rgb/{index:06d}.png"; dp=f"aligned_depth/{index:06d}.png"
                if not cv2.imwrite(str(self.output/rp), cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                    raise RuntimeError("rgb write failed")
                if not cv2.imwrite(str(self.output/dp), depth): raise RuntimeError("depth write failed")
                record.update({"rgb_path":rp,"aligned_depth_path":dp})
                self.handle.write(json.dumps(safe(record),separators=(",",":"))+"\n")
                self.video.write(overlay)
        except BaseException as exc: self.error=exc

    def close(self) -> None:
        self.queue.put(None); self.thread.join(); self.handle.close(); self.video.release()
        if self.error: raise RuntimeError("writer failed") from self.error


def overlay_image(frame: Any, palm_roi: Any, result: Any,
                  tracker_frame: Any, config: Any) -> np.ndarray:
    image=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR); box=np.rint(palm_roi.bbox).astype(int)
    cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(0,255,255),2)
    contours=cv2.findContours(build_rigid_palm_mask(tracker_frame,config),
                              cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(image,contours,-1,(255,100,0),1)
    pixels=result.pairwise.tracked_pixels_current
    if pixels is not None:
        for point in pixels: cv2.circle(image,tuple(np.rint(point).astype(int)),2,(0,255,0),-1)
    rotation=result.accumulated_rotation; origin=((box[0]+box[2])//2,(box[1]+box[3])//2)
    if rotation is not None:
        for i,color in enumerate(((0,0,255),(0,255,0),(255,0,0))):
            d=rotation[:2,i]; n=np.linalg.norm(d)
            if n>1e-8:
                end=tuple(np.rint(np.asarray(origin)+40*d/n).astype(int))
                cv2.arrowedLine(image,origin,end,color,2,tipLength=.2)
    cv2.putText(image,f"{result.state.value} valid={result.pairwise.valid} "
        f"inliers={result.pairwise.ransac_inliers}",(8,22),
        cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return image


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario",choices=SCENARIOS,required=True)
    parser.add_argument("--bbox",nargs=4,type=float)
    parser.add_argument("--duration-s",type=float,default=25.0)
    parser.add_argument("--countdown-s",type=float,default=4.0)
    parser.add_argument("--left-hand",action="store_true")
    parser.add_argument("--output-root",default=str(ROOT/"datasets/development_usb2/p5_rgbd_relative_orientation"))
    args=parser.parse_args()
    sdk="/home/diu/anaconda3/envs/mediapipe_env/lib/python3.10/site-packages"
    if sdk not in sys.path: sys.path.append(sdk)
    runner=HamerCropInference(str(PACKAGE_DIR/"_DATA/hamer_ckpts/checkpoints/hamer.ckpt"),
        data_root=str(PACKAGE_DIR/"_DATA"),precision="fp16",freeze_betas=True,
        source_frame="camera_color_optical_frame",
        timestamp_clock_domain="d455_device_global_time_ms")
    runner.load()
    capture=D455Capture(640,480,30,require_superspeed=False); capture.start()
    device_metadata=safe(capture.device_metadata)
    seed=capture.wait_for_stable_frames(8)
    roi=np.asarray(args.bbox,dtype=np.float64) if args.bbox else select_roi(seed.rgb)
    output=Path(args.output_root).resolve()/(args.scenario+"_"+time.strftime("%Y%m%dT%H%M%S"))
    output.mkdir(parents=True,exist_ok=False); (output/"rgb").mkdir(); (output/"aligned_depth").mkdir()
    writer=AsyncWriter(output); fifo=SequentialCaptureQueue(8); latest=LatestOnlySlot()
    context=HamerContextState(); stop=threading.Event(); stats={"captured":0,"roi_lost":0,"error":None}
    config=RGBDRigidTrackerConfig(maximum_frame_gap=1,maximum_dt_s=.12,
        palm_bbox_erosion_fraction=.15,maximum_rotation_increment_deg=30)
    tracker=RGBDRelativeOrientationTracker(config,lost_after_s=.25); tracker.engage_clutch()
    capture_thread=threading.Thread(target=capture_worker,args=(capture,roi,fifo,latest,stop,stats),daemon=True)
    hamer_thread=threading.Thread(target=hamer_worker,args=(runner,latest,context,stop,not args.left_hand),daemon=True)
    print(f"Selected central palm ROI: {roi.tolist()}", flush=True)
    print(SCENARIO_INSTRUCTIONS[args.scenario], flush=True)
    countdown_deadline=time.monotonic()+max(0.0,args.countdown_s); shown=None
    while time.monotonic()<countdown_deadline:
        remaining=max(1,int(np.ceil(countdown_deadline-time.monotonic())))
        if remaining!=shown:
            print(f"Starting in {remaining}...", flush=True); shown=remaining
        capture.wait_for_frame()  # drain the SDK while the operator prepares
    capture_thread.start(); hamer_thread.start()
    started=time.monotonic(); records=[]; times=[]
    usage0=resource.getrusage(resource.RUSAGE_SELF); rss0=usage0.ru_maxrss
    try:
        while time.monotonic()-started<args.duration_s:
            packet=fifo.get(3); tf=rgbd_tracker_frame_from_d455(packet.frame,packet.palm_roi.bbox)
            if packet.palm_roi.reinitialized and packet.sequence > 0:
                tracker.mark_roi_reacquired()
            h=context.snapshot(); fresh=(h.get("timestamp") is not None and
                tf.timestamp_s-float(h["timestamp"])<=.30)
            gesture=bool(args.scenario=="P5_GESTURE" and fresh and h.get("gesture_changing"))
            tick=time.perf_counter(); result=tracker.process(tf,externally_frozen=gesture,
                freeze_reason="HAMER_HAND_POSE_CHANGING" if gesture else "NONE")
            try: center=robust_palm_center_m(tf,config)
            except Exception: center=None
            ms=(time.perf_counter()-tick)*1000; times.append(ms)
            record={"index":len(records),"capture_sequence":packet.sequence,
                "color_frame_number":packet.frame.color_frame_number,
                "depth_frame_number":packet.frame.depth_frame_number,
                "timestamp_s":tf.timestamp_s,"timestamp_domain":tf.timestamp_domain,
                "palm_roi":packet.palm_roi.as_dict(),"hand_roi_xyxy":packet.hand_roi,
                "palm_center_m":center,"processing_ms":ms,"result":result.as_dict(),
                "hamer_context":{"valid":h.get("valid",False),"timestamp":h.get("timestamp"),
                    "inference_ms":h.get("inference_ms"),"gesture_changing":h.get("gesture_changing",False),
                    "gesture_change_p75_deg":h.get("gesture_change_p75_deg"),
                    "usage":"ROI_HAND_POSE_GESTURE_ONLY","orientation_present":False}}
            overlay=overlay_image(packet.frame,packet.palm_roi,result,tf,config)
            writer.submit(len(records),packet.frame.rgb,packet.frame.aligned_depth_raw,overlay,record)
            records.append(record)
    finally:
        stop.set(); latest.close(); capture_thread.join(1.0)
        capture.stop(); capture_thread.join(3.0); hamer_thread.join(3.0); writer.close()
    wall=max(time.monotonic()-started,1e-9); pair=[r["result"]["pairwise"] for r in records]
    if not records:
        raise RuntimeError("no RGB-D frames were processed")
    valid=sum(r["valid"] for r in pair); states=[r["result"]["state"] for r in records]
    usage1=resource.getrusage(resource.RUSAGE_SELF)
    summary={"schema_version":1,"scenario":args.scenario,
        "profile":"D455 RGB8 + aligned Z16 640x480@30","device":device_metadata,
        "usb_type_descriptor":device_metadata["usb_type_descriptor"],
        "duration_s":wall,"captured_frames":stats["captured"],"processed_frames":len(records),
        "raw_capture_hz":stats["captured"]/wall,"kabsch_processing_hz":len(records)/wall,
        "capture_queue_drops":fifo.dropped,"capture_queue_high_water":fifo.maximum_size,
        "kabsch_valid_frames":valid,"kabsch_valid_coverage":valid/max(len(records),1),
        "frozen_frames":states.count("FROZEN"),"lost_frames":states.count("LOST"),
        "reinitialization_count":records[-1]["result"]["reinitialization_count"],
        "kabsch_processing_ms":{"mean":float(np.mean(times)),"p50":float(np.percentile(times,50)),
            "p95":float(np.percentile(times,95)),"maximum":float(np.max(times))},
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "rss_growth_mib":(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss0)/1024,
        "process_cpu_seconds":(usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime),
        "process_cpu_utilization_percent":100.0*(usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime)/wall,
        "hamer_latest_only":latest.stats,"hamer_orientation_used":False,
        "orientation_source":"RGBD_KLT_RANSAC_KABSCH_ONLY","config":safe(config.__dict__),
        "palm_roi_initial_xyxy":roi.tolist(),"status":"COMPLETE","capture_error":stats["error"]}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output_dir":str(output),**summary},ensure_ascii=False)); return 0


if __name__=="__main__": raise SystemExit(main())
