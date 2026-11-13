from __future__ import annotations

import ctypes
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageGrab

from . import config, vision

def build_monkey_script(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    count: int = 100,
    moves: int = 1,
) -> str:
    """生成 monkey 脚本：count 次滑动（起终点相同就是连点），一次注入。

    monkey 在设备上单进程注入，实测 100 次滑动约 0.5 秒，可达 100+/秒，
    比每条 adb input 快一个数量级。
    """
    count = max(1, int(count))
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    lines = [
        "type= raw events",
        f"count= {(moves + 2) * count}",
        "speed= 1.0",
        "start data >>",
    ]
    for _ in range(count):
        lines.append(f"DispatchPointer(0,0,0,{x1},{y1},0,0,0,0,0,0,0)")
        for step in range(1, moves + 1):
            ratio = step / (moves + 1)
            x = int(round(x1 + (x2 - x1) * ratio))
            y = int(round(y1 + (y2 - y1) * ratio))
            lines.append(f"DispatchPointer(0,0,2,{x},{y},0,0,0,0,0,0,0)")
        lines.append(f"DispatchPointer(0,0,1,{x2},{y2},0,0,0,0,0,0,0)")
    return "\n".join(lines) + "\n"


KEYCODES = {
    "home": 3,
    "back": 4,
    "call": 5,
    "endcall": 6,
    "up": 19,
    "down": 20,
    "left": 21,
    "right": 22,
    "center": 23,
    "volume_up": 24,
    "volume_down": 25,
    "power": 26,
    "camera": 27,
    "tab": 61,
    "space": 62,
    "enter": 66,
    "del": 67,
    "menu": 82,
    "notification": 83,
    "search": 84,
    "play_pause": 85,
    "mute": 164,
}

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"


class DeviceError(RuntimeError):
    pass


def list_adb_devices(adb: str | None = None) -> list[dict]:
    adb = adb or config.resolve_adb()
    try:
        output = subprocess.run(
            [adb, "devices", "-l"], capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1]
        model = ""
        for part in parts[2:]:
            if part.startswith("model:"):
                model = part.split(":", 1)[1]
        devices.append({"serial": serial, "state": state, "model": model})
    return devices


def parse_bounds(value: str | None) -> tuple[int, int, int, int] | None:
    match = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", value or "")
    if not match:
        return None
    x1, y1, x2, y2 = (int(item) for item in match.groups())
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def parse_ui_dump(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, ValueError):
        return []
    elements: list[dict] = []
    for index, node in enumerate(root.iter("node")):
        bounds = parse_bounds(node.get("bounds"))
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        elements.append(
            {
                "index": index,
                "text": node.get("text", "") or "",
                "description": node.get("content-desc", "") or "",
                "resource_id": node.get("resource-id", "") or "",
                "class": node.get("class", "") or "",
                "package": node.get("package", "") or "",
                "clickable": (node.get("clickable", "") or "").lower() == "true",
                "enabled": (node.get("enabled", "") or "").lower() != "false",
                "bounds": [x1, y1, x2, y2],
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            }
        )
    return elements


def match_ui_element(elements: list[dict], selector: dict | None) -> dict | None:
    if not selector:
        return None

    def normalized(value) -> str:
        return str(value or "").strip().lower()

    wanted_index = selector.get("index")
    matches: list[dict] = []
    for element in elements:
        if not element.get("enabled", True):
            continue
        if wanted_index is not None:
            try:
                if int(element.get("index", -1)) != int(wanted_index):
                    continue
            except (TypeError, ValueError):
                continue
        if selector.get("clickable_only") and not element.get("clickable"):
            continue
        matched = True
        for key in ("text", "description", "resource_id", "class", "package"):
            wanted = normalized(selector.get(key))
            if wanted and wanted not in normalized(element.get(key)):
                matched = False
                break
        if matched:
            matches.append(element)
    if not matches:
        return None
    return next((item for item in matches if item.get("clickable")), matches[0])


def ui_selector_from_element(element: dict) -> dict:
    selector = {}
    for key in ("text", "description", "resource_id", "class"):
        value = str(element.get(key, "") or "").strip()
        if value:
            selector[key] = value
    if element.get("clickable"):
        selector["clickable_only"] = True
    return selector


def _popen_options() -> dict:
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class AdbDevice:
    def __init__(self, serial: str | None = None, adb: str | None = None):
        self.serial = serial
        self.adb = adb or config.resolve_adb()
        self._screen_size: tuple[int, int] | None = None
        self._burst_script: str | None = None
        self._burst_remote: str = ""

    def _cmd(self, args: list[str]) -> list[str]:
        return [self.adb] + (["-s", self.serial] if self.serial else []) + args

    def run(self, args: list[str], timeout: float = 20, binary: bool = False):
        try:
            result = subprocess.run(
                self._cmd(args), capture_output=True, timeout=timeout
            )
        except FileNotFoundError as exc:
            raise DeviceError(f"找不到 adb：{self.adb}") from exc
        except subprocess.TimeoutExpired as exc:
            raise DeviceError(f"adb 超时：{' '.join(args)}") from exc
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "replace").strip()
            raise DeviceError(message or f"adb 失败：{' '.join(args)}")
        return result.stdout if binary else result.stdout.decode("utf-8", "replace")

    def check(self) -> None:
        state = self.run(["get-state"], timeout=10).strip()
        if state != "device":
            raise DeviceError(f"设备不可用：{state}")

    def screen_size(self) -> tuple[int, int]:
        if self._screen_size:
            return self._screen_size
        output = self.run(["shell", "wm", "size"], timeout=10)
        override = re.search(r"Override size:\s*(\d+)x(\d+)", output)
        physical = re.search(r"Physical size:\s*(\d+)x(\d+)", output)
        match = override or physical
        if not match:
            raise DeviceError(f"无法解析屏幕尺寸：{output.strip()}")
        self._screen_size = (int(match.group(1)), int(match.group(2)))
        return self._screen_size

    def screenshot(self) -> np.ndarray:
        try:
            raw = self.run(["exec-out", "screencap", "-p"], timeout=30, binary=True)
            if raw:
                with Image.open(io.BytesIO(raw)) as handle:
                    return np.asarray(handle.convert("RGB"))
        except Exception:
            pass
        remote = "/sdcard/scflow_shot.png"
        self.run(["shell", "screencap", "-p", remote], timeout=30)
        raw = self.run(["exec-out", "cat", remote], timeout=30, binary=True)
        try:
            return vision.decode_image(raw)
        except Exception as exc:
            raise DeviceError(f"截图失败：{exc}") from exc

    def ui_elements(self) -> list[dict]:
        remote = "/sdcard/scflow_ui.xml"
        try:
            self.run(["shell", "uiautomator", "dump", remote], timeout=25)
        except DeviceError:
            self.run(
                ["shell", "uiautomator", "dump", "--compressed", remote],
                timeout=25,
            )
        raw = self.run(["shell", "cat", remote], timeout=20)
        return parse_ui_dump(raw)

    def find_ui_element(self, selector: dict | None) -> dict | None:
        return match_ui_element(self.ui_elements(), selector)

    def tap(self, x: float, y: float) -> None:
        self.run(["shell", "input", "tap", str(int(round(x))), str(int(round(y)))], timeout=10)

    def swipe(
        self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300
    ) -> None:
        self.run(
            [
                "shell",
                "input",
                "swipe",
                str(int(round(x1))),
                str(int(round(y1))),
                str(int(round(x2))),
                str(int(round(y2))),
                str(int(duration_ms)),
            ],
            timeout=20,
        )

    def swipe_repeat(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        duration_ms: int = 300,
        count: int = 1,
    ) -> None:
        """在一条 adb shell 里连续执行多次 swipe。

        逐条调用 adb 每次都要启动一个进程（约 100~200ms 开销），
        批量发送才能做到一秒钟滑动十几次以上。
        """
        count = max(1, int(count))
        if count == 1:
            self.swipe(x1, y1, x2, y2, duration_ms)
            return
        command = "input swipe %d %d %d %d %d" % (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
            int(duration_ms),
        )
        script = "; ".join([command] * count)
        timeout = max(20.0, count * (int(duration_ms) + 150) / 1000.0 + 10.0)
        self.run(["shell", "sh", "-c", script], timeout=timeout)

    def burst_swipe(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        count: int = 100,
        duration_ms: int = 50,
    ) -> int:
        """用 monkey 脚本一次注入 count 次滑动/连点（100+/秒）。"""
        count = max(1, int(count))
        script = build_monkey_script(x1, y1, x2, y2, count)
        serial_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(self.serial or "default"))
        remote = f"/data/local/tmp/scflow_burst_{serial_tag}.txt"
        if script != self._burst_script or remote != self._burst_remote:
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            )
            try:
                handle.write(script)
                handle.close()
                self.run(["push", handle.name, remote], timeout=60)
                self._burst_script = script
                self._burst_remote = remote
            finally:
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass
        output = self.run(
            ["shell", "monkey", "-f", remote, "1"],
            timeout=max(30.0, count * 0.05 + 30.0),
        )
        match = re.search(r"Events injected:\s*(\d+)", output or "")
        if match and int(match.group(1)) == 0:
            raise DeviceError("Monkey 没有注入事件：手机可能锁屏或不在前台")
        return count

    def key(self, key: str) -> None:
        code = KEYCODES.get(str(key).lower(), key)
        self.run(["shell", "input", "keyevent", str(code)], timeout=10)

    def launch_app(self, package: str, fresh: bool = True) -> None:
        package = str(package).strip()
        if not package:
            raise DeviceError("缺少应用包名")
        if fresh:
            self.run(["shell", "am", "force-stop", package], timeout=15)
        self.run(
            [
                "shell",
                "monkey",
                "-p",
                package,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ],
            timeout=25,
        )

    def clear_text(self, count: int = 60) -> None:
        self.run(
            ["shell", "input", "keyevent", "123"] + ["67"] * max(1, int(count)),
            timeout=30,
        )

    def text(self, value: str) -> None:
        escaped = str(value).replace(" ", "%s")
        escaped = re.sub(r'([()<>|;&*\\~"\'`$!])', r"\\\1", escaped)
        self.run(["shell", "input", "text", escaped], timeout=20)


def _windows_find_window(title: str):
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if title.lower() in buffer.value.lower():
                found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found[-1] if found else None


def _windows_client_geometry(hwnd) -> tuple[int, int, int, int]:
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise DeviceError("读取 scrcpy 窗口尺寸失败")
    point = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        raise DeviceError("读取 scrcpy 窗口位置失败")
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        raise DeviceError("scrcpy 窗口尺寸无效")
    return (point.x, point.y, width, height)


def _applescript_string(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _run_osascript(script: str, timeout: float = 6.0) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeviceError(f"无法执行 osascript：{exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise DeviceError(
            f"macOS 窗口控制失败：{detail or '未知错误'}。"
            "请在「系统设置 → 隐私与安全性 → 辅助功能」中允许终端或 Python 控制窗口。"
        )
    return result.stdout.strip()


def _macos_find_window(title: str) -> str | None:
    title_literal = _applescript_string(title)
    script = f"""
tell application "System Events"
    repeat with p in (every process whose background only is false)
        try
            repeat with w in (every window of p)
                try
                    set windowName to name of w as text
                    set processName to name of p as text
                    if (windowName contains {title_literal}) or (processName contains "scrcpy") then return windowName
                end try
            end repeat
        end try
    end repeat
end tell
return ""
"""
    name = _run_osascript(script)
    return name or None


def _macos_window_geometry(title: str) -> tuple[int, int, int, int]:
    title_literal = _applescript_string(title)
    script = f"""
tell application "System Events"
    repeat with p in (every process whose background only is false)
        try
            repeat with w in (every window of p)
                try
                    set windowName to name of w as text
                    if windowName contains {title_literal} then
                        set windowPosition to position of w
                        set windowSize to size of w
                        return (item 1 of windowPosition as text) & "," & (item 2 of windowPosition as text) & "," & (item 1 of windowSize as text) & "," & (item 2 of windowSize as text)
                    end if
                end try
            end repeat
        end try
    end repeat
end tell
return ""
"""
    output = _run_osascript(script)
    if not output:
        raise DeviceError("scrcpy 窗口不存在")
    try:
        x, y, width, height = (int(value.strip()) for value in output.split(",")[:4])
    except (TypeError, ValueError) as exc:
        raise DeviceError(f"无法解析 scrcpy 窗口位置：{output}") from exc
    if width <= 0 or height <= 0:
        raise DeviceError("scrcpy 窗口尺寸无效")
    return (x, y, width, height)


def _macos_activate_window(title: str) -> None:
    title_literal = _applescript_string(title)
    script = f"""
tell application "System Events"
    repeat with p in (every process whose background only is false)
        try
            repeat with w in (every window of p)
                try
                    if (name of w as text) contains {title_literal} then
                        set frontmost of p to true
                        return "ok"
                    end if
                end try
            end repeat
        end try
    end repeat
end tell
return ""
"""
    _run_osascript(script)


def _macos_click(screen_x: int, screen_y: int) -> None:
    cliclick = shutil.which("cliclick")
    if cliclick:
        try:
            result = subprocess.run(
                [cliclick, f"c:{int(screen_x)},{int(screen_y)}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return
        except (OSError, subprocess.SubprocessError):
            pass
    script = (
        'tell application "System Events" to click at '
        f"{{{int(screen_x)}, {int(screen_y)}}}"
    )
    try:
        _run_osascript(script)
    except DeviceError as exc:
        raise DeviceError(
            "scrcpy 窗口点击失败：请安装 cliclick（brew install cliclick），"
            "或在「系统设置 → 隐私与安全性 → 辅助功能」中授权。"
        ) from exc


class ScrcpyWindow:
    def __init__(
        self,
        title: str,
        serial: str | None = None,
        scrcpy: str | None = None,
        extra_args: list[str] | None = None,
    ):
        self.title = title
        self.serial = serial
        self.scrcpy = scrcpy or config.resolve_scrcpy()
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen | None = None
        self.window_id = None
        self._geometry: tuple[int, int, int, int] | None = None
        self._geometry_time = 0.0

    @property
    def ready(self) -> bool:
        if self.proc is not None and self.proc.poll() is not None:
            return False
        try:
            return bool(self.window_id) or self._find_window() is not None
        except DeviceError:
            return False

    def start(self, timeout: float = 12.0) -> None:
        executable = self.scrcpy
        if not Path(executable).exists() and not shutil.which(executable):
            raise DeviceError(f"找不到 scrcpy：{executable}")
        cmd = [executable, f"--window-title={self.title}", "--stay-awake"]
        if IS_MACOS:
            cmd.append("--window-borderless")
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += self.extra_args
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_popen_options(),
        )
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise DeviceError("scrcpy 启动失败或已退出")
                window_id = self._find_window()
                if window_id:
                    self.window_id = window_id
                    self._refresh_geometry(force=True)
                    return
                time.sleep(0.25)
            raise DeviceError("等待 scrcpy 窗口超时")
        except DeviceError:
            self.stop()
            raise

    def _find_window(self):
        if self.window_id:
            return self.window_id
        if IS_WINDOWS:
            self.window_id = _windows_find_window(self.title)
            return self.window_id
        if IS_MACOS:
            self.window_id = _macos_find_window(self.title)
            return self.window_id
        xdotool = shutil.which("xdotool")
        if xdotool:
            try:
                output = subprocess.run(
                    [xdotool, "search", "--name", self.title],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
                ids = [line.strip() for line in output.splitlines() if line.strip()]
                if ids:
                    self.window_id = ids[-1]
                    return self.window_id
            except (OSError, subprocess.SubprocessError):
                pass
        if not shutil.which("xwininfo"):
            return None
        try:
            output = subprocess.run(
                ["xwininfo", "-root", "-tree"], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in output.splitlines():
            match = re.match(r'\s*(0x[0-9a-fA-F]+)\s+"([^"]*)"', line)
            if match and self.title in match.group(2):
                self.window_id = match.group(1)
                return self.window_id
        return None

    def _refresh_geometry(self, force: bool = False) -> tuple[int, int, int, int]:
        if not force and self._geometry and time.time() - self._geometry_time < 2.0:
            return self._geometry
        window_id = self._find_window()
        if not window_id:
            raise DeviceError("scrcpy 窗口不存在")
        if IS_WINDOWS:
            self._geometry = _windows_client_geometry(window_id)
        elif IS_MACOS:
            self._geometry = _macos_window_geometry(window_id)
        else:
            try:
                output = subprocess.run(
                    ["xwininfo", "-id", str(window_id)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
            except (OSError, subprocess.SubprocessError) as exc:
                raise DeviceError("读取 scrcpy 窗口位置失败") from exc
            values = {}
            for key, pattern in (
                ("x", r"Absolute upper-left X:\s*(-?\d+)"),
                ("y", r"Absolute upper-left Y:\s*(-?\d+)"),
                ("w", r"Width:\s*(\d+)"),
                ("h", r"Height:\s*(\d+)"),
            ):
                match = re.search(pattern, output)
                if not match:
                    raise DeviceError("窗口信息不完整")
                values[key] = int(match.group(1))
            if values["w"] <= 0 or values["h"] <= 0:
                raise DeviceError("scrcpy 窗口尺寸无效")
            self._geometry = (values["x"], values["y"], values["w"], values["h"])
        self._geometry_time = time.time()
        return self._geometry

    def capture(self) -> np.ndarray:
        x, y, width, height = self._refresh_geometry()
        try:
            import mss
        except ImportError:
            mss = None
        if mss is not None:
            try:
                with mss.mss() as sct:
                    shot = sct.grab({"left": x, "top": y, "width": width, "height": height})
                    return np.asarray(shot)[:, :, [2, 1, 0]].copy()
            except Exception:
                pass
        try:
            if IS_WINDOWS or IS_MACOS:
                image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            else:
                image = ImageGrab.grab(
                    bbox=(x, y, x + width, y + height),
                    xdisplay=os.environ.get("DISPLAY"),
                )
            return np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise DeviceError(f"截取 scrcpy 窗口失败：{exc}") from exc

    def _tap_windows(self, screen_x: int, screen_y: int) -> None:
        user32 = ctypes.windll.user32
        user32.SetForegroundWindow(self._find_window())
        user32.SetCursorPos(int(screen_x), int(screen_y))
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)

    def tap_phone(self, x: float, y: float, device_size: tuple[int, int]) -> None:
        origin_x, origin_y, win_w, win_h = self._refresh_geometry()
        dev_w, dev_h = device_size
        scale = min(win_w / dev_w, win_h / dev_h)
        video_w, video_h = dev_w * scale, dev_h * scale
        offset_x = (win_w - video_w) / 2
        offset_y = (win_h - video_h) / 2
        px = round(offset_x + x * scale)
        py = round(offset_y + y * scale)
        if IS_WINDOWS:
            self._tap_windows(origin_x + px, origin_y + py)
            return
        if IS_MACOS:
            _macos_activate_window(self.window_id or self.title)
            _macos_click(origin_x + px, origin_y + py)
            return
        xdotool = shutil.which("xdotool")
        if not xdotool:
            raise DeviceError("scrcpy 窗口输入需要安装 xdotool")
        subprocess.run(
            [
                xdotool,
                "mousemove",
                "--window",
                str(self._find_window()),
                str(px),
                str(py),
                "click",
                "1",
            ],
            capture_output=True,
            timeout=10,
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.window_id = None
        self._geometry = None


class DeviceController:
    def __init__(
        self,
        serial: str | None = None,
        capture: str = "scrcpy",
        input_mode: str = "adb",
        manage_scrcpy: bool = True,
        scrcpy_title: str | None = None,
        scrcpy_path: str | None = None,
        adb_path: str | None = None,
    ):
        self.adb = AdbDevice(serial=serial, adb=adb_path or config.resolve_adb())
        self.serial = serial
        self.capture = capture
        self.input_mode = input_mode
        self.manage_scrcpy = manage_scrcpy
        self.scrcpy_path = scrcpy_path or config.resolve_scrcpy()
        self.window: ScrcpyWindow | None = None
        self.warnings: list[str] = []
        self._owned_window = False
        self.scrcpy_title = scrcpy_title or f"scflow-{os.getpid()}"

    @property
    def device_size(self) -> tuple[int, int]:
        return self.adb.screen_size()

    def ui_elements(self) -> list[dict]:
        return self.adb.ui_elements()

    def find_ui_element(self, selector: dict | None) -> dict | None:
        return self.adb.find_ui_element(selector)

    def start(self) -> None:
        self.adb.check()
        self.adb.screen_size()
        try:
            self.adb.run(["shell", "svc", "power", "stayon", "usb"], timeout=10)
            self.adb.run(["shell", "input", "keyevent", "224"], timeout=5)
            self.adb.run(["shell", "wm", "dismiss-keyguard"], timeout=5)
        except DeviceError:
            pass
        wants_scrcpy = (
            self.capture == "scrcpy"
            or self.input_mode == "scrcpy"
            or self.manage_scrcpy
        )
        if not wants_scrcpy:
            return
        try:
            window = ScrcpyWindow(
                title=self.scrcpy_title,
                serial=self.serial,
                scrcpy=self.scrcpy_path,
            )
            window.start()
            self.window = window
            self._owned_window = True
        except DeviceError as exc:
            self.warnings.append(str(exc))
            if self.capture == "scrcpy":
                self.warnings.append("scrcpy 画面不可用，已回退到 adb 截图")
                self.capture = "adb"
            if self.input_mode == "scrcpy":
                self.warnings.append("scrcpy 窗口输入不可用，已回退到 adb 输入")
                self.input_mode = "adb"

    def screenshot(self) -> np.ndarray:
        if self.capture == "scrcpy" and self.window and self.window.ready:
            try:
                frame = self.window.capture()
                return vision.fit_frame(frame, self.adb.screen_size())
            except DeviceError as exc:
                if exc.args and exc.args[0] not in self.warnings:
                    self.warnings.append(exc.args[0])
                self.warnings.append("scrcpy 截图失败，后续改用 adb 截图")
                self.capture = "adb"
        return self.adb.screenshot()

    def tap(self, x: float, y: float) -> None:
        if self.input_mode == "scrcpy" and self.window and self.window.ready:
            try:
                self.window.tap_phone(x, y, self.adb.screen_size())
                return
            except DeviceError as exc:
                if str(exc) not in self.warnings:
                    self.warnings.append(str(exc))
                self.warnings.append("scrcpy 输入失败，后续改用 adb 输入")
                self.input_mode = "adb"
        self.adb.tap(x, y)

    def swipe(
        self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300
    ) -> None:
        self.adb.swipe(x1, y1, x2, y2, duration_ms)

    def swipe_repeat(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        duration_ms: int = 300,
        count: int = 1,
    ) -> None:
        if self.input_mode == "scrcpy" and self.window and self.window.ready:
            for _ in range(max(1, int(count))):
                self.swipe(x1, y1, x2, y2, duration_ms)
            return
        self.adb.swipe_repeat(x1, y1, x2, y2, duration_ms, count)

    def burst_swipe(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        count: int = 100,
        duration_ms: int = 50,
    ) -> int:
        return self.adb.burst_swipe(x1, y1, x2, y2, count, duration_ms)

    def key(self, key: str) -> None:
        self.adb.key(key)

    def text(self, value: str) -> None:
        self.adb.text(value)

    def clear_text(self, count: int = 60) -> None:
        self.adb.clear_text(count)

    def launch_app(self, package: str, fresh: bool = True) -> None:
        self.adb.launch_app(package, fresh)

    def stop(self) -> None:
        if self._owned_window and self.window:
            self.window.stop()
        self.window = None


class DryRunDevice:
    def __init__(self, inner):
        self.inner = inner
        self.actions: list[tuple] = []

    def screenshot(self):
        return self.inner.screenshot()

    def screen_size(self):
        return self.inner.screen_size()

    @property
    def device_size(self):
        return self.inner.device_size

    def ui_elements(self) -> list[dict]:
        """模拟运行时仍然允许 AI 读取控件树（只是不真正点击）。"""
        getter = getattr(self.inner, "ui_elements", None)
        return list(getter() or []) if getter is not None else []

    def tap(self, x: float, y: float):
        self.actions.append(("tap", round(x), round(y)))

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300):
        self.actions.append(("swipe", round(x1), round(y1), round(x2), round(y2), duration_ms))

    def swipe_repeat(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        duration_ms: int = 300,
        count: int = 1,
    ):
        for _ in range(max(1, int(count))):
            self.swipe(x1, y1, x2, y2, duration_ms)

    def burst_swipe(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        count: int = 100,
        duration_ms: int = 50,
    ) -> int:
        count = max(1, int(count))
        for _ in range(count):
            self.actions.append(("swipe", round(x1), round(y1), round(x2), round(y2), duration_ms))
        return count

    def key(self, key: str):
        self.actions.append(("key", key))

    def text(self, value: str):
        self.actions.append(("text", value))

    def stop(self):
        pass
