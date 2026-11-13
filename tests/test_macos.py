from __future__ import annotations

import unittest
from unittest import mock

from scflow import config
from scflow.device import ScrcpyWindow, _applescript_string, _macos_window_geometry


class MacOSSupportTests(unittest.TestCase):
    def test_platform_label_and_tokens(self):
        with mock.patch("scflow.config.sys.platform", "darwin"), mock.patch(
            "scflow.config.os.name", "posix"
        ):
            self.assertEqual(config.platform_label(), "macOS")
            self.assertIn("macos", config._platform_tokens())

    def test_applescript_string_escapes(self):
        self.assertEqual(_applescript_string('a"b\\c'), '"a\\"b\\\\c"')

    def test_macos_find_window(self):
        window = ScrcpyWindow("scflow-test", scrcpy="scrcpy")
        with mock.patch("scflow.device.IS_MACOS", True), mock.patch(
            "scflow.device.IS_WINDOWS", False
        ), mock.patch(
            "scflow.device._macos_find_window", return_value="scflow-test"
        ) as finder:
            self.assertEqual(window._find_window(), "scflow-test")
        finder.assert_called_once_with("scflow-test")

    def test_macos_geometry_parsing(self):
        with mock.patch("scflow.device._run_osascript", return_value="10,20,300,600"):
            self.assertEqual(_macos_window_geometry("scflow-test"), (10, 20, 300, 600))


if __name__ == "__main__":
    unittest.main()
