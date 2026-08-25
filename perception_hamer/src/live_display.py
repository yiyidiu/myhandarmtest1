"""OpenCV display adapter that keeps GUI dependencies out of HaMeR.

The inference environment intentionally uses ``opencv-python-headless``.  In
interactive mode frames are therefore sent to a small helper running in the
MediaPipe environment, where OpenCV has Qt support.  In environments with a
working local OpenCV GUI the same API uses ``cv2.imshow`` directly.
"""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess
import threading
from typing import Optional

import cv2
import numpy as np


def opencv_gui_available(build_information: Optional[str] = None) -> bool:
    """Return whether this OpenCV build reports a usable GUI backend."""

    information = cv2.getBuildInformation() if build_information is None else build_information
    for line in information.splitlines():
        if line.strip().startswith("GUI:"):
            backend = line.split(":", 1)[1].strip().upper()
            return bool(backend and backend != "NONE")
    return False


class LiveDisplay:
    """Display BGR frames locally or through a GUI-enabled Python sidecar."""

    def __init__(
        self,
        title: str,
        helper_python: str,
        helper_script: Path,
        backend: str = "auto",
        jpeg_quality: int = 88,
    ) -> None:
        if backend not in ("auto", "local", "sidecar"):
            raise ValueError("display backend must be auto, local, or sidecar")
        if backend == "auto":
            backend = "local" if opencv_gui_available() else "sidecar"
        self.backend = backend
        self.title = str(title)
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 100))
        self.stop_requested = False
        self._reinitialize_requested = False
        self._confirm_requested = False
        self._process: Optional[subprocess.Popen] = None
        self._control_thread: Optional[threading.Thread] = None
        self._closed = False

        if self.backend == "sidecar":
            helper = Path(helper_script).resolve()
            if not Path(helper_python).is_file():
                raise RuntimeError("display helper Python does not exist: " + helper_python)
            if not helper.is_file():
                raise RuntimeError("display helper script does not exist: " + str(helper))
            self._process = subprocess.Popen(
                [helper_python, "-u", str(helper), "--title", self.title],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=0,
            )
            self._control_thread = threading.Thread(
                target=self._read_controls, name="hamer-display-controls", daemon=True
            )
            self._control_thread.start()
            print(
                "OpenCV GUI is unavailable in the HaMeR environment; "
                "using the MediaPipe display sidecar.",
                flush=True,
            )

    def _read_controls(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            command = raw_line.decode("utf-8", errors="replace").strip().upper()
            if command == "QUIT":
                self.stop_requested = True
            elif command == "REINITIALIZE":
                self._reinitialize_requested = True
            elif command == "CONFIRM":
                self._confirm_requested = True

    def pop_reinitialize_request(self) -> bool:
        requested = self._reinitialize_requested
        self._reinitialize_requested = False
        return requested

    def pop_confirm_request(self) -> bool:
        requested = self._confirm_requested
        self._confirm_requested = False
        return requested

    def show(self, bgr_frame: np.ndarray) -> bool:
        """Show one frame; return False if the external viewer has exited."""

        if self._closed:
            return False
        if self.backend == "local":
            cv2.imshow(self.title, bgr_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self.stop_requested = True
            elif key in (ord("r"), ord("R")):
                self._reinitialize_requested = True
            elif key in (ord("c"), ord("C"), 10, 13, 32):
                self._confirm_requested = True
            return True

        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            return False
        ok, encoded = cv2.imencode(
            ".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return False
        payload = encoded.tobytes()
        try:
            process.stdin.write(struct.pack("!I", len(payload)))
            process.stdin.write(payload)
            return True
        except (BrokenPipeError, OSError):
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.backend == "local":
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
            return
        process = self._process
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)

    def __enter__(self) -> "LiveDisplay":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
