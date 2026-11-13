from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from . import similarity, vision


def _as_pil(frame: np.ndarray) -> Image.Image:
    array = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def augment_positive(frame: np.ndarray) -> list[np.ndarray]:
    """给正样本做轻量增强，覆盖亮度、模糊、轻微缩放等真实变化。"""
    image = _as_pil(frame)
    results: list[np.ndarray] = []
    for factor in (0.85, 1.15):
        results.append(np.asarray(ImageEnhance.Brightness(image).enhance(factor)))
    results.append(np.asarray(image.filter(ImageFilter.GaussianBlur(0.6))))
    for scale in (0.97, 1.03):
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        results.append(np.asarray(resized.resize(image.size, Image.Resampling.BILINEAR)))
    return results


def collect_negatives(
    db,
    node,
    max_negatives: int = 80,
    include_other_flows: bool = True,
) -> list[np.ndarray]:
    """收集负样本：同流程其它节点图片（难负样本）+ 可选其它流程节点图片。"""
    frames: list[np.ndarray] = []
    nodes = list(db.list_nodes(node.flow_id))
    if include_other_flows:
        try:
            for flow in db.list_flows():
                if flow.id == node.flow_id:
                    continue
                nodes.extend(db.list_nodes(flow.id))
        except Exception:
            pass
    for other in nodes:
        if getattr(other, "id", None) == node.id:
            continue
        path = getattr(other, "image_path", None)
        if not path or not Path(path).exists():
            continue
        try:
            frames.append(vision.load_rgb(path))
        except Exception:
            continue
        if len(frames) >= max_negatives:
            break
    return frames


def calibrate_node(
    db,
    node,
    model: dict | None = None,
    fpr: float = 0.01,
    margin: float = 0.02,
    min_negatives: int = 3,
    max_negatives: int = 80,
) -> dict:
    """为单个节点自动校准阈值。

    正样本：节点图片 + 轻量增强；
    负样本：同流程/其它流程的其它节点图片；
    输出：similarity.build_calibration 的校准字典。
    """
    path = getattr(node, "image_path", None)
    if not path or not Path(path).exists():
        raise ValueError("节点没有可用图片，无法校准")

    base_model = dict(model or similarity.get_model(node))
    base_model.pop("calibration", None)

    node_image = vision.load_rgb(path)
    positives = [node_image, *augment_positive(node_image)]
    negatives = collect_negatives(db, node, max_negatives=max_negatives)
    if len(negatives) < min_negatives:
        raise ValueError(
            f"负样本不足：需要至少 {min_negatives} 个带图片的其它节点，"
            f"当前只有 {len(negatives)} 个"
        )

    positive_scores = [
        similarity.raw_score_node(node, frame, base_model) for frame in positives
    ]
    negative_scores = [
        similarity.raw_score_node(node, frame, base_model) for frame in negatives
    ]
    positive_scores = [score for score in positive_scores if score is not None]
    negative_scores = [score for score in negative_scores if score is not None]
    if not negative_scores:
        raise ValueError("负样本打分失败，无法校准")
    return similarity.build_calibration(
        positive_scores,
        negative_scores,
        fpr=fpr,
        margin=margin,
    )
