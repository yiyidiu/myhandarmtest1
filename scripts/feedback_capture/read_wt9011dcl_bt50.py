#!/usr/bin/env python3
"""Scan, connect, and read a WT9011DCL-BT50 through its BLE5 USB adapter.

The WITMotion BLE5 adapter enumerates as a USB CDC serial device.  It does not
start scanning merely because the port is opened; the adapter first needs its
ASCII scan/connect commands.  Once linked, the sensor emits 20-byte ``55 61``
frames containing acceleration, angular velocity, and roll/pitch/yaw.
"""

from __future__ import annotations

import argparse
import csv
from glob import glob
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
from typing import Iterable, List, NamedTuple, Optional, TextIO

try:
    import serial
except ImportError as exc:  # pragma: no cover - depends on the host environment
    raise SystemExit(
        "pyserial is required; install it with: python3 -m pip install --user pyserial"
    ) from exc


DEFAULT_DEVICE_NAME = "WT901BLE67"
DEFAULT_PORT_GLOB = "/dev/serial/by-id/*nRF52_USB_CDC_BLE_Demo*"
FRAME_HEADER = b"\x55\x61"
FRAME_SIZE = 20
DEVICE_LINE = re.compile(
    rb'WIT-LIST-#\s+(\d+):"([^"]*)"\s+(0x[0-9A-Fa-f]+)\s+(-?\d+)'
)


class BleDevice(NamedTuple):
    index: int
    name: str
    address: str
    rssi_dbm: int


class Sample(NamedTuple):
    # These are host receive timestamps for the serial read that completed the
    # frame.  They are deliberately not called sample timestamps: the BLE/USB
    # adapter can deliver several device samples in one burst.
    receive_monotonic_ns: int
    receive_wall_time_ns: int
    acc_x_g: float
    acc_y_g: float
    acc_z_g: float
    gyro_x_dps: float
    gyro_y_dps: float
    gyro_z_dps: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    sample_index: int
    batch_index: int
    sample_in_batch: int
    batch_size: int
    raw_frame_hex: str

    @property
    def host_monotonic_ns(self) -> int:
        """Backward-compatible alias; this is a receive, not sample, time."""

        return self.receive_monotonic_ns

    @property
    def host_wall_time_ns(self) -> int:
        """Backward-compatible alias; this is a receive, not sample, time."""

        return self.receive_wall_time_ns


CSV_HEADER = list(Sample._fields)


def resolve_port(requested: Optional[str]) -> str:
    if requested:
        port = Path(requested)
        if not port.exists():
            raise RuntimeError(f"serial port does not exist: {port}")
        return str(port)

    matches = sorted(glob(DEFAULT_PORT_GLOB))
    if len(matches) == 1:
        return matches[0]
    if not matches and Path("/dev/ttyACM0").exists():
        return "/dev/ttyACM0"
    if not matches:
        raise RuntimeError("WITMotion nRF52 BLE5 USB adapter was not found")
    raise RuntimeError("multiple WITMotion adapters found; select one with --port")


def read_for(port: "serial.Serial", seconds: float) -> bytes:
    deadline = time.monotonic() + seconds
    result = bytearray()
    while time.monotonic() < deadline:
        waiting = int(port.in_waiting)
        chunk = port.read(waiting if waiting > 0 else 1)
        if chunk:
            result.extend(chunk)
    return bytes(result)


def parse_scan(data: bytes) -> List[BleDevice]:
    devices = []
    for match in DEVICE_LINE.finditer(data):
        devices.append(
            BleDevice(
                index=int(match.group(1)),
                name=match.group(2).decode("utf-8", errors="replace").strip(),
                address=match.group(3).decode("ascii"),
                rssi_dbm=int(match.group(4)),
            )
        )
    return devices


def choose_device(devices: Iterable[BleDevice], requested: str) -> BleDevice:
    candidates = list(devices)
    exact = [device for device in candidates if device.name == requested]
    if len(exact) == 1:
        return exact[0]

    prefix = [device for device in candidates if device.name.startswith(requested)]
    if len(prefix) == 1:
        return prefix[0]

    available = ", ".join(
        f"{device.name} ({device.address}, {device.rssi_dbm} dBm)"
        for device in candidates
    )
    if not prefix:
        raise RuntimeError(f"device {requested!r} not found; scan saw: {available}")
    raise RuntimeError(
        f"device prefix {requested!r} is ambiguous; use an exact name: {available}"
    )


def scan_and_connect(
    port: "serial.Serial", device_name: str, scan_seconds: float
) -> bytes:
    port.reset_input_buffer()
    port.write(b"AT+SCAN=1\r\n")
    port.flush()
    scan_data = read_for(port, scan_seconds)
    port.write(b"AT+SCAN=0\r\n")
    port.flush()
    scan_data += read_for(port, 0.5)

    if b"Scan_start" not in scan_data:
        raise RuntimeError("BLE adapter did not acknowledge AT+SCAN=1")
    devices = parse_scan(scan_data)
    for device in devices:
        print(
            f"scan: index={device.index} name={device.name!r} "
            f"address={device.address} rssi={device.rssi_dbm} dBm",
            file=sys.stderr,
        )
    selected = choose_device(devices, device_name)
    print(
        f"connecting: {selected.name} {selected.address} "
        f"({selected.rssi_dbm} dBm)",
        file=sys.stderr,
    )

    port.reset_input_buffer()
    port.write(f"AT+CONNECT={selected.index}\r\n".encode("ascii"))
    port.flush()
    link_data = read_for(port, 8.0)
    if b"WIT-LINK" not in link_data or FRAME_HEADER not in link_data:
        readable = link_data[:300].decode("utf-8", errors="replace")
        raise RuntimeError(f"adapter did not establish a data link; reply={readable!r}")
    print("connected: streaming 55 61 frames", file=sys.stderr)
    return link_data


def decode_frame(
    frame: bytes,
    receive_monotonic_ns: int,
    receive_wall_time_ns: int,
    sample_index: int = 0,
    batch_index: int = 0,
    sample_in_batch: int = 0,
    batch_size: int = 1,
) -> Sample:
    if len(frame) != FRAME_SIZE or not frame.startswith(FRAME_HEADER):
        raise ValueError("not a complete WT9011DCL 55 61 frame")
    values = struct.unpack("<9h", frame[2:])
    return Sample(
        int(receive_monotonic_ns),
        int(receive_wall_time_ns),
        values[0] * 16.0 / 32768.0,
        values[1] * 16.0 / 32768.0,
        values[2] * 16.0 / 32768.0,
        values[3] * 2000.0 / 32768.0,
        values[4] * 2000.0 / 32768.0,
        values[5] * 2000.0 / 32768.0,
        values[6] * 180.0 / 32768.0,
        values[7] * 180.0 / 32768.0,
        values[8] * 180.0 / 32768.0,
        int(sample_index),
        int(batch_index),
        int(sample_in_batch),
        int(batch_size),
        frame.hex(),
    )


def extract_samples(
    buffer: bytearray,
    *,
    receive_monotonic_ns: Optional[int] = None,
    receive_wall_time_ns: Optional[int] = None,
    sample_index_start: int = 0,
    batch_index: int = 0,
) -> List[Sample]:
    """Decode every complete frame made available by one serial read.

    All returned frames share the receive timestamp of that read.  Their
    device sampling times are unknown and must be reconstructed offline from
    the batch/order evidence; this function never fabricates them.
    """

    if (receive_monotonic_ns is None) != (receive_wall_time_ns is None):
        raise ValueError("receive monotonic and wall timestamps must be supplied together")
    if sample_index_start < 0 or batch_index < 0:
        raise ValueError("sample and batch indices must be nonnegative")
    frames: List[bytes] = []
    while True:
        header_index = buffer.find(FRAME_HEADER)
        if header_index < 0:
            if buffer[-1:] == FRAME_HEADER[:1]:
                del buffer[:-1]
            else:
                buffer.clear()
            break
        if header_index:
            del buffer[:header_index]
        if len(buffer) < FRAME_SIZE:
            break
        frame = bytes(buffer[:FRAME_SIZE])
        del buffer[:FRAME_SIZE]
        frames.append(frame)
    if not frames:
        return []
    if receive_monotonic_ns is None:
        receive_monotonic_ns = time.monotonic_ns()
        receive_wall_time_ns = time.time_ns()
    batch_size = len(frames)
    return [
        decode_frame(
            frame,
            receive_monotonic_ns,
            receive_wall_time_ns,
            sample_index=sample_index_start + sample_in_batch,
            batch_index=batch_index,
            sample_in_batch=sample_in_batch,
            batch_size=batch_size,
        )
        for sample_in_batch, frame in enumerate(frames)
    ]


def open_csv(path: Optional[Path]) -> tuple[Optional[TextIO], Optional[csv.writer]]:
    if path is None:
        return None, None
    # Do not line-flush every 10 Hz sample while the RGB-D recorder is writing
    # thousands of PNG files to the same disk.  A blocked line flush can leave
    # BLE notifications queued in the USB CDC driver; when the process runs
    # again, those old frames receive nearly identical host timestamps and
    # corrupt angular-rate integration.  The complete log is far below this
    # buffer size and is flushed when the handle closes, including Ctrl-C.
    handle = path.open(
        "x", encoding="utf-8", newline="", buffering=1024 * 1024
    )
    writer = csv.writer(handle)
    writer.writerow(CSV_HEADER)
    return handle, writer


def run(args: argparse.Namespace) -> int:
    serial_path = resolve_port(args.port)
    print(f"port: {serial_path}", file=sys.stderr)
    try:
        port = serial.Serial(serial_path, 115200, timeout=0.1)
    except PermissionError as exc:
        raise RuntimeError(
            "serial permission denied; log in again after joining the dialout group"
        ) from exc

    csv_handle: Optional[TextIO] = None
    try:
        port.dtr = True
        initial = read_for(port, 0.6)
        if FRAME_HEADER in initial:
            print("connected: reusing the existing BLE link", file=sys.stderr)
        else:
            initial = scan_and_connect(port, args.device, args.scan_seconds)

        # Do not count frames buffered during link probing/connection as part
        # of the timed acquisition interval.
        port.reset_input_buffer()
        csv_handle, writer = open_csv(args.csv)
        buffer = bytearray()
        preview = None
        if getattr(args, "orientation_preview_json", None) is not None:
            from imu_posture_guidance import GyroPreview
            if getattr(args, "gyro_bias_reference", None) is None:
                raise RuntimeError("姿态提示需要明确冻结的静止零偏文件，不能再单独估计另一套零偏")
            bias_reference = json.loads(args.gyro_bias_reference.read_text())
            preview = GyroPreview(args.orientation_preview_json, fixed_bias_dps=bias_reference["gyro_bias_dps"])
        started = time.monotonic()
        sample_count = 0
        batch_count = 0
        ready_written = False
        next_display = started
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            waiting = int(port.in_waiting)
            chunk = port.read(waiting if waiting > 0 else 1)
            receive_monotonic_ns = time.monotonic_ns()
            receive_wall_time_ns = time.time_ns()
            if chunk:
                buffer.extend(chunk)
            samples = extract_samples(
                buffer,
                receive_monotonic_ns=receive_monotonic_ns,
                receive_wall_time_ns=receive_wall_time_ns,
                sample_index_start=sample_count,
                batch_index=batch_count,
            )
            if samples:
                batch_count += 1
            for sample in samples:
                sample_count += 1
                if writer is not None:
                    writer.writerow(sample)
                if preview is not None:
                    preview.update(sample)
                if args.ready_file is not None and not ready_written:
                    if args.ready_file.exists():
                        raise RuntimeError(
                            "ready file already exists: " + str(args.ready_file)
                        )
                    if csv_handle is not None:
                        csv_handle.flush()
                    with args.ready_file.open("x", encoding="utf-8") as stream:
                        json.dump(
                            {
                                "schema": "wt901_reader_ready_v1",
                                "pid": os.getpid(),
                                "first_sample_index": sample.sample_index,
                                "first_batch_index": sample.batch_index,
                                "first_receive_monotonic_ns": (
                                    sample.receive_monotonic_ns
                                ),
                                "timestamp_role": "serial_read_receive_time",
                            },
                            stream,
                            ensure_ascii=False,
                            indent=2,
                        )
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    ready_written = True
                now = time.monotonic()
                if now >= next_display:
                    print(
                        "acc_g=(% .3f,% .3f,% .3f) "
                        "gyro_dps=(% .2f,% .2f,% .2f) "
                        "rpy_deg=(% .2f,% .2f,% .2f)" % sample[2:11],
                        flush=True,
                    )
                    next_display = now + 1.0 / args.display_hz

        elapsed = time.monotonic() - started
        print(
            f"summary: frames={sample_count} batches={batch_count} elapsed_s={elapsed:.3f} "
            f"received_rate_hz={sample_count / elapsed:.2f}",
            file=sys.stderr,
        )
        return 0 if sample_count else 2
    finally:
        if csv_handle is not None:
            csv_handle.close()
        port.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="USB CDC port; auto-detected by default")
    parser.add_argument("--orientation-preview-json", type=Path,
                        help="optional nominal-cadence gyro orientation for operator guidance")
    parser.add_argument("--gyro-bias-reference", type=Path,
                        help="frozen zero-rate calibration shared by operator guidance and offline reference")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE_NAME,
        help=f"exact BLE name or unique prefix (default: {DEFAULT_DEVICE_NAME})",
    )
    parser.add_argument(
        "--scan-seconds", type=float, default=8.0, help="BLE scan duration"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="read duration in seconds; 0 means until interrupted",
    )
    parser.add_argument(
        "--display-hz", type=float, default=5.0, help="terminal display rate"
    )
    parser.add_argument(
        "--csv", type=Path, help="write every decoded frame to a new CSV file"
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="write a one-time JSON marker after the first sample is safely recorded",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.scan_seconds <= 0:
        raise SystemExit("--scan-seconds must be positive")
    if args.duration < 0:
        raise SystemExit("--duration must be nonnegative")
    if args.display_hz <= 0:
        raise SystemExit("--display-hz must be positive")
    if args.ready_file is not None and args.csv is not None:
        if args.ready_file.expanduser().resolve() == args.csv.expanduser().resolve():
            raise SystemExit("--ready-file and --csv must be different paths")
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nstopped by user", file=sys.stderr)
        return 130
    except (RuntimeError, serial.SerialException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
