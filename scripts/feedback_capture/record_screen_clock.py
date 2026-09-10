#!/usr/bin/env python3
"""Record the X11 desktop and bind actual MP4 frame PTS to host clocks."""
import argparse
import csv
import json
from pathlib import Path
import re
import signal
import subprocess
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def bind_container_times(directory):
    """Read MP4 sample timestamps without decoding; encoder has no B-frames."""
    source = (directory / "screen_capture_times.csv").open()
    rows = iter(csv.DictReader(source))
    output = (directory / "screen_frame_timeline.csv").open("w", newline="")
    fields = ["video_frame_index", "video_pts_ns", "video_time_s", "capture_pts_ns",
              "estimated_frame_monotonic_ns", "estimated_frame_wall_ns", "probe_monotonic_ns", "probe_wall_ns"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    pipeline = Gst.parse_launch("filesrc name=source ! qtdemux ! fakesink name=samples sync=false")
    pipeline.get_by_name("source").set_property("location", str(directory / "screen.mp4"))
    state = dict(mapped_frames=0, missing_source_rows=0, first_pts=None, first_capture_pts=None, max_interval_residual_ns=0)

    def probe(_pad, info):
        buffer = info.get_buffer()
        if buffer is None or buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.PadProbeReturn.OK
        row = next(rows, None)
        if row is None:
            state["missing_source_rows"] += 1
            return Gst.PadProbeReturn.OK
        pts = int(buffer.pts)
        capture_pts = int(row["capture_pts_ns"])
        if state["first_pts"] is None:
            state["first_pts"], state["first_capture_pts"] = pts, capture_pts
        residual = abs((pts-state["first_pts"]) - (capture_pts-state["first_capture_pts"]))
        state["max_interval_residual_ns"] = max(state["max_interval_residual_ns"], residual)
        writer.writerow(dict(video_frame_index=state["mapped_frames"], video_pts_ns=pts,
            video_time_s=pts / 1e9, **{k: row[k] for k in fields if k in row}))
        state["mapped_frames"] += 1
        return Gst.PadProbeReturn.OK

    pipeline.get_by_name("samples").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe)
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(30 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipeline.set_state(Gst.State.NULL)
    state["unmapped_capture_rows"] = sum(1 for _ in rows)
    state["demux_eos"] = message is not None and message.type == Gst.MessageType.EOS
    state["usable_frame_mapping"] = bool(state["demux_eos"] and state["mapped_frames"] > 0
        and not state["missing_source_rows"] and not state["unmapped_capture_rows"]
        and state["max_interval_residual_ns"] < 2_000_000)
    state["mapping_check_role"] = "Frame count and relative PTS consistency only; not a physical camera/IMU synchronization certificate."
    output.close()
    source.close()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    Gst.init(None)
    geometry = subprocess.check_output(["xrandr", "--current"], universal_newlines=True)
    match = re.search(r"current (\d+) x (\d+)", geometry)
    if not match:
        raise RuntimeError("Cannot identify the current desktop dimensions")
    width, height = map(int, match.groups())
    out_width = min(width, 1920) // 2 * 2
    out_height = round(height * out_width / width / 2) * 2
    pipeline = Gst.parse_launch(
        "ximagesrc use-damage=false show-pointer=true do-timestamp=true "
        # Both live views render at 15 Hz. Recording 30 copies/s increased
        # inference time and caused actual control dropouts on this host.
        "! video/x-raw,framerate=15/1 ! queue max-size-buffers=3 leaky=downstream "
        "! videoscale ! video/x-raw,width=%d,height=%d,pixel-aspect-ratio=1/1 "
        "! videoconvert ! clockoverlay time-format=\"%%H:%%M:%%S\" "
        "valignment=bottom halignment=left font-desc=\"Sans 16\" shaded-background=true "
        "! video/x-raw,format=I420 "
        "! x264enc name=encoder threads=2 speed-preset=ultrafast tune=zerolatency "
        "bframes=0 bitrate=6000 key-int-max=30 "
        "! mp4mux fragment-duration=1000 ! filesink name=destination" % (out_width, out_height))
    pipeline.get_by_name("destination").set_property("location", str(directory / "screen.mp4"))
    clock = Gst.SystemClock.obtain()
    pipeline.use_clock(clock)
    handle = (directory / "screen_capture_times.csv").open("x", newline="")
    writer = csv.writer(handle)
    writer.writerow(["capture_frame_index", "capture_pts_ns", "duration_ns", "estimated_frame_monotonic_ns",
                     "estimated_frame_wall_ns", "probe_monotonic_ns", "probe_wall_ns", "clock_pair_span_ns"])
    state = dict(capture_frames=0, started_wall_ns=time.time_ns(), started_monotonic_ns=time.monotonic_ns(),
        desktop_size=[width, height], video_size=[out_width, out_height], requested_fps=15,
        timestamp_role="ximagesrc PTS mapped through its pipeline clock to host monotonic time; screen sampling and display latency remain.",
        audio_recorded=False, eos=False, error=None)
    loop = GLib.MainLoop()
    stopping = [False]

    def capture_probe(_pad, info):
        buffer = info.get_buffer()
        if buffer is None or buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.PadProbeReturn.OK
        m0 = time.monotonic_ns()
        clock_now = clock.get_time()
        m1 = time.monotonic_ns()
        monotonic = (m0+m1)//2
        wall = time.time_ns()
        estimated = int(pipeline.get_base_time()) + int(buffer.pts) + monotonic - int(clock_now)
        writer.writerow([state["capture_frames"], int(buffer.pts), int(buffer.duration), estimated,
                         wall+estimated-monotonic, monotonic, wall, m1-m0])
        state["capture_frames"] += 1
        state["last_frame_monotonic_ns"] = estimated
        if state["capture_frames"] == 1:
            handle.flush()
            atomic_json(directory / "screen_ready.json", state)
        if state["capture_frames"] % 30 == 0:
            handle.flush()
            atomic_json(directory / "screen_status.json", state)
        return Gst.PadProbeReturn.OK

    def stop(*_args):
        if not stopping[0]:
            stopping[0] = True
            pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(10, force_stop)
        return False

    def force_stop():
        if not state["eos"]:
            state["error"] = state["error"] or "EOS timeout; file may be incomplete"
            loop.quit()
        return False

    def on_message(_bus, message):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            state["error"] = str(error) + "; " + str(debug)
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            state["eos"] = True
            loop.quit()

    pipeline.get_by_name("encoder").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, capture_probe)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if args.duration > 0:
        GLib.timeout_add(int(args.duration * 1000), stop)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        handle.close()
        state.update(stopped_wall_ns=time.time_ns(), stopped_monotonic_ns=time.monotonic_ns())
        if state["eos"]:
            state["container_timeline"] = bind_container_times(directory)
        atomic_json(directory / "screen_summary.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state["eos"] and not state["error"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
