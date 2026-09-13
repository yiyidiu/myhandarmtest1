import unittest

from perception_hamer.src.live_display import opencv_gui_available, write_all


class OpenCVGUIAvailabilityTest(unittest.TestCase):
    def test_headless_build_is_rejected(self):
        self.assertFalse(opencv_gui_available("  GUI:                           NONE\n"))

    def test_qt_build_is_accepted(self):
        self.assertTrue(opencv_gui_available("  GUI:                           QT5\n"))

    def test_missing_gui_field_fails_closed(self):
        self.assertFalse(opencv_gui_available("OpenCV 4.10 build information\n"))


class FramePipeTest(unittest.TestCase):
    def test_interrupted_short_writes_preserve_consecutive_frames(self):
        import struct
        class ShortPipe:
            data = bytearray()
            def write(self, payload):
                count = min(7, len(payload))
                self.data.extend(payload[:count])
                return count
        pipe = ShortPipe()
        frames = [b"first RGB frame" * 9000, b"second RGB frame" * 11000]
        expected = b""
        for payload in frames:
            header = struct.pack("!I", len(payload))
            write_all(pipe, header)
            write_all(pipe, payload)
            expected += header + payload
        self.assertEqual(bytes(pipe.data), expected)

    def test_closed_pipe_does_not_spin(self):
        class ClosedPipe:
            def write(self, _payload): return 0
        with self.assertRaises(BrokenPipeError):
            write_all(ClosedPipe(), b"frame")


if __name__ == "__main__":
    unittest.main()
