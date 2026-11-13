from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

from .embeddings import to_pil


def decode_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as handle:
        return np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))


def save_rgb(path: str | Path, frame: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(frame).save(path)


def save_thumbnail(path: str | Path, frame: np.ndarray, max_width: int = 540, quality: int = 82) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = to_pil(frame)
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    image.save(path, quality=quality)
    return str(path)


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = int(size[0]), int(size[1])
    height, width = frame.shape[:2]
    if (width, height) == (target_w, target_h):
        return frame
    target_ratio = target_w / target_h
    ratio = width / height
    if abs(ratio - target_ratio) > 1e-4:
        if ratio > target_ratio:
            new_w = max(1, round(height * target_ratio))
            left = (width - new_w) // 2
            frame = frame[:, left : left + new_w]
        else:
            new_h = max(1, round(width / target_ratio))
            top = (height - new_h) // 2
            frame = frame[top : top + new_h, :]
    image = to_pil(frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.asarray(image)


def _ratio(size: tuple[int, int]) -> float:
    return size[0] / max(1, size[1])


def needs_template_match(node_size: tuple[int, int], screen_size: tuple[int, int]) -> bool:
    if not node_size or not screen_size:
        return False
    ratio_node = _ratio(node_size)
    ratio_screen = _ratio(screen_size)
    return abs(ratio_node - ratio_screen) / ratio_screen > 0.03


def template_locate(
    screen: np.ndarray, template: np.ndarray, min_score: float = 0.55
) -> tuple[int, int, int, int, float] | None:
    if cv2 is None:
        return None
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    screen_h, screen_w = screen_gray.shape[:2]
    work_scale = min(1.0, 800.0 / max(screen_w, screen_h))
    if work_scale < 1.0:
        work = cv2.resize(
            screen_gray, (round(screen_w * work_scale), round(screen_h * work_scale))
        )
    else:
        work = screen_gray
    work_h, work_w = work.shape[:2]
    best = (-1.0, 0, 0, 0, 0)
    for scale in np.linspace(0.3, 1.4, 12):
        tpl_w = max(4, round(template_gray.shape[1] * scale * work_scale))
        tpl_h = max(4, round(template_gray.shape[0] * scale * work_scale))
        if tpl_w >= work_w or tpl_h >= work_h:
            continue
        tpl = cv2.resize(template_gray, (tpl_w, tpl_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(work, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score > best[0]:
            best = (score, location[0], location[1], tpl_w, tpl_h)
    score, x, y, w, h = best
    if score < min_score:
        return None
    inv = 1.0 / work_scale
    return (round(x * inv), round(y * inv), round(w * inv), round(h * inv), float(score))


def resolve_point(
    node_click: tuple[float, float] | None,
    node_size: tuple[int, int] | None,
    screen: np.ndarray,
    node_image: np.ndarray | None = None,
) -> tuple[tuple[int, int] | None, tuple[int, int, int, int] | None]:
    screen_h, screen_w = screen.shape[:2]
    if node_click is None:
        return (screen_w // 2, screen_h // 2), None
    rel_x, rel_y = float(node_click[0]), float(node_click[1])
    if node_size and needs_template_match(node_size, (screen_w, screen_h)) and node_image is not None:
        found = template_locate(screen, node_image)
        if found:
            x, y, w, h, _ = found
            return (round(x + rel_x * w), round(y + rel_y * h)), (x, y, w, h)
    return (round(rel_x * screen_w), round(rel_y * screen_h)), None


def crop_region(frame: np.ndarray, region: tuple[int, int, int, int] | None) -> np.ndarray:
    if not region:
        return frame
    x, y, w, h = region
    return frame[max(0, y) : y + h, max(0, x) : x + w]


def draw_click_marker(frame: np.ndarray, point: tuple[int, int] | None) -> np.ndarray:
    if point is None:
        return frame
    if cv2 is None:
        return frame
    out = frame.copy()
    x, y = point
    cv2.circle(out, (int(x), int(y)), 28, (255, 60, 60), 4)
    cv2.line(out, (x - 40, y), (x + 40, y), (255, 60, 60), 2)
    cv2.line(out, (x, y - 40), (x, y + 40), (255, 60, 60), 2)
    return out


def frame_to_pil(frame: Any) -> Image.Image:
    return to_pil(frame)


def cv2_available() -> bool:
    return cv2 is not None


def crop_relative(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray | None:
    if frame is None:
        return None
    height, width = frame.shape[:2]
    x = int(round(float(roi[0]) * width))
    y = int(round(float(roi[1]) * height))
    w = int(round(float(roi[2]) * width))
    h = int(round(float(roi[3]) * height))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    crop = frame[y : y + h, x : x + w]
    return crop if crop.size else None


def locate_target(
    screen: np.ndarray,
    node_image: np.ndarray | None,
    target_roi: tuple[float, float, float, float] | None,
    min_score: float = 0.55,
) -> tuple[tuple[int, int] | None, tuple[int, int, int, int] | None, bool]:
    if node_image is None or target_roi is None:
        return None, None, False
    crop = crop_relative(node_image, target_roi)
    if crop is None or crop.shape[0] < 6 or crop.shape[1] < 6:
        return None, None, False
    found = template_locate(screen, crop, min_score=min_score)
    if found:
        x, y, w, h, _score = found
        return (x + w // 2, y + h // 2), (x, y, w, h), True
    height, width = screen.shape[:2]
    center_x = (float(target_roi[0]) + float(target_roi[2]) / 2) * width
    center_y = (float(target_roi[1]) + float(target_roi[3]) / 2) * height
    return (round(center_x), round(center_y)), None, False
