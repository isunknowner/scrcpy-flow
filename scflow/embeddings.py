from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

HYBRID_KIND = "hybrid-v2"
DEFAULT_DIM = 960


def to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return ImageOps.exif_transpose(handle).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return Image.fromarray(array, "RGB")


def _dhash(gray: Image.Image, size: int = 8) -> np.ndarray:
    small = gray.resize((size + 1, size), Image.Resampling.LANCZOS)
    values = np.asarray(small, dtype=np.float32)
    bits = (values[:, 1:] > values[:, :-1]).astype(np.float32).ravel()
    return bits * 2.0 - 1.0


def _color_hist(rgb: Image.Image) -> np.ndarray:
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)
    hist, _ = np.histogramdd(
        (hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()),
        bins=(8, 8, 8),
        range=((0, 256), (0, 256), (0, 256)),
    )
    flat = hist.astype(np.float32).ravel()
    return flat / (np.linalg.norm(flat) + 1e-8)


def _gradient_hist(gray: Image.Image) -> np.ndarray:
    data = np.asarray(gray.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    gx = np.zeros_like(data)
    gy = np.zeros_like(data)
    gx[:, 1:-1] = data[:, 2:] - data[:, :-2]
    gy[1:-1, :] = data[2:, :] - data[:-2, :]
    magnitude = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) % np.pi) / np.pi * 8.0
    bins = np.clip(angle.astype(np.int32), 0, 7)
    vec = np.zeros((4, 4, 8), dtype=np.float32)
    height, width = data.shape
    for row in range(4):
        for col in range(4):
            block_mag = magnitude[
                row * height // 4 : (row + 1) * height // 4,
                col * width // 4 : (col + 1) * width // 4,
            ].ravel()
            block_bins = bins[
                row * height // 4 : (row + 1) * height // 4,
                col * width // 4 : (col + 1) * width // 4,
            ].ravel()
            np.add.at(vec[row, col], block_bins, block_mag)
    flat = vec.ravel()
    return flat / (np.linalg.norm(flat) + 1e-8)


def _thumbnail(gray: Image.Image) -> np.ndarray:
    small = gray.resize((16, 16), Image.Resampling.LANCZOS)
    values = np.asarray(small, dtype=np.float32) / 255.0
    values = (values - values.mean()) / (values.std() + 1e-6)
    flat = values.ravel()
    return flat / (np.linalg.norm(flat) + 1e-8)


class HybridEmbedder:
    kind = HYBRID_KIND
    dim = DEFAULT_DIM
    weights = (0.15, 0.25, 0.25, 0.35)

    def embed(self, image: Any) -> np.ndarray:
        rgb = to_pil(image)
        gray = rgb.convert("L")
        parts = (_dhash(gray), _color_hist(rgb), _gradient_hist(gray), _thumbnail(gray))
        blocks = []
        for part, weight in zip(parts, self.weights):
            part = part / (np.linalg.norm(part) + 1e-8)
            blocks.append(part.astype(np.float32) * np.sqrt(weight))
        vector = np.concatenate(blocks)
        vector /= np.linalg.norm(vector) + 1e-8
        return vector.astype(np.float32)


class ClipEmbedder:
    def __init__(self, model_name: str | None = None, pretrained: str | None = None):
        self.model_name = model_name or os.environ.get("SCFLOW_CLIP_MODEL", "ViT-B-32")
        self.pretrained = pretrained or os.environ.get("SCFLOW_CLIP_PRETRAINED", "openai")
        self.kind = f"clip:{self.model_name}:{self.pretrained}"
        self.dim = 0
        self._model = None
        self._preprocess = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "使用 CLIP 向量需要安装 open_clip_torch 与 torch：pip install open_clip_torch torch"
            ) from exc
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = model.to(device).eval()
        self._preprocess = preprocess
        self._torch = torch
        self._device = device
        with torch.no_grad():
            dummy = self._preprocess(Image.new("RGB", (224, 224), "white")).unsqueeze(0).to(device)
            self.dim = int(self._model.encode_image(dummy).shape[-1])

    def embed(self, image: Any) -> np.ndarray:
        self._load()
        torch = self._torch
        tensor = self._preprocess(to_pil(image)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().numpy().astype(np.float32)


def get_embedder(name: str | None = None):
    name = (name or os.environ.get("SCFLOW_EMBEDDER", "hybrid")).strip().lower()
    if name in ("clip", "open_clip", "clip-vit"):
        return ClipEmbedder()
    return HybridEmbedder()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left is None or right is None or left.size == 0 or right.size == 0:
        return 0.0
    if left.shape != right.shape:
        return 0.0
    value = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-8))
    return max(-1.0, min(1.0, value))


def embed_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_embed(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()
