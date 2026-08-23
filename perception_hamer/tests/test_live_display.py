import unittest

from perception_hamer.src.live_display import opencv_gui_available


class OpenCVGUIAvailabilityTest(unittest.TestCase):
    def test_headless_build_is_rejected(self):
        self.assertFalse(opencv_gui_available("  GUI:                           NONE\n"))

    def test_qt_build_is_accepted(self):
        self.assertTrue(opencv_gui_available("  GUI:                           QT5\n"))

    def test_missing_gui_field_fails_closed(self):
        self.assertFalse(opencv_gui_available("OpenCV 4.10 build information\n"))


if __name__ == "__main__":
    unittest.main()
