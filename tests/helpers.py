from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def temp_home() -> tempfile.TemporaryDirectory:
    handle = tempfile.TemporaryDirectory(prefix="scflow-test-")
    os.environ["SCFLOW_HOME"] = handle.name
    return handle


def make_image(kind: str, size: tuple[int, int] = (180, 400)) -> Image.Image:
    width, height = size
    if kind == "blue":
        image = Image.new("RGB", size, (25, 70, 190))
    elif kind == "red":
        image = Image.new("RGB", size, (200, 45, 45))
    elif kind == "green":
        image = Image.new("RGB", size, (30, 160, 80))
    elif kind == "black":
        image = Image.new("RGB", size, (5, 5, 5))
    else:
        image = Image.new("RGB", size, (120, 120, 120))
    draw = ImageDraw.Draw(image)
    if kind == "blue":
        draw.rectangle((20, 30, width - 20, 130), fill=(245, 245, 245))
        draw.rectangle((30, 220, width - 30, 300), fill=(90, 160, 230))
    elif kind == "red":
        draw.ellipse((25, 220, width - 25, 360), outline=(20, 20, 20), width=8)
        draw.text((40, 50), "MESSAGE", fill=(255, 255, 255))
    elif kind == "green":
        draw.rectangle((15, 150, width - 15, 260), fill=(240, 220, 80))
        draw.line((0, height - 30, width, height - 30), fill=(255, 255, 255), width=6)
    elif kind == "roi":
        image = Image.new("RGB", size, (240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width // 2, height), fill=(220, 50, 50))
        draw.rectangle((width // 2, 0, width, height), fill=(40, 80, 220))
    elif kind == "roi-changed":
        image = Image.new("RGB", size, (240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width // 2, height), fill=(220, 50, 50))
        draw.rectangle((width // 2, 0, width, height), fill=(40, 200, 80))
    return image


def save_image(path: Path, image: Image.Image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


class FakeDevice:
    def __init__(self, frame):
        self.frame = frame
        self.taps: list[tuple[int, int]] = []
        self.swipes: list[tuple] = []
        self.keys: list[str] = []
        self.texts: list[str] = []
        self.cleared = False
        self.launched: list[str] = []
        self.bursts: list[tuple] = []

    def screenshot(self):
        return self.frame

    def set_frame(self, frame):
        self.frame = frame

    def screen_size(self):
        return self.frame.shape[1], self.frame.shape[0]

    def tap(self, x, y):
        self.taps.append((round(x), round(y)))

    def swipe(self, x1, y1, x2, y2, duration_ms=300):
        self.swipes.append((round(x1), round(y1), round(x2), round(y2), duration_ms))

    def swipe_repeat(self, x1, y1, x2, y2, duration_ms=300, count=1):
        for _ in range(max(1, int(count))):
            self.swipe(x1, y1, x2, y2, duration_ms)

    def burst_swipe(self, x1, y1, x2, y2, count=100, duration_ms=50):
        self.bursts.append((round(x1), round(y1), round(x2), round(y2), int(count), int(duration_ms)))
        return int(count)

    def key(self, key):
        self.keys.append(key)

    def text(self, value):
        self.texts.append(value)

    def clear_text(self, count=60):
        self.cleared = True

    def launch_app(self, package, fresh=True):
        self.launched.append(package)


class ScriptedDevice:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.taps: list[tuple[int, int]] = []
        self.keys: list[str] = []

    def screenshot(self):
        if not self.frames:
            raise RuntimeError("no frames")
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def screen_size(self):
        return self.frames[0].shape[1], self.frames[0].shape[0]

    def tap(self, x, y):
        self.taps.append((round(x), round(y)))

    def swipe(self, *args, **kwargs):
        pass

    def key(self, key):
        self.keys.append(key)

    def text(self, value):
        pass
