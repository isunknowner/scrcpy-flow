from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

COMPONENTS = ("color", "texture", "structure", "edge")
DEFAULT_COMPONENT_WEIGHT = 0.25
PREVIEW_WIDTH = 320

# 改进项：空间分块直方图，避免“无位置信息”的全局直方图误匹配。
SPATIAL_GRID = 4
SPATIAL_MIN_CELL = 12
# 改进项：几何平均之后的“短板惩罚”，任何一项过低都会拉低总分。
SHORT_PENALTY = 0.35
SHORT_TAU = 0.5
# 改进项：边缘密度过低（空白/纯色/加载页）视为无效，返回 0 而不是满分。
EDGE_MIN_DENSITY = 0.004
CALIBRATION_VERSION = 1


def default_component_weights() -> dict[str, float]:
    return {name: DEFAULT_COMPONENT_WEIGHT for name in COMPONENTS}


def default_model(mode: str = "image", roi_weight: float = 0.72, has_target: bool = False) -> dict:
    roi_only = mode == "detect"
    roi_weight = 1.0 if roi_only else max(0.0, min(1.0, float(roi_weight)))
    model = {
        "version": 1,
        "mode": "roi_only" if roi_only else "roi_plus_full",
        "region_weights": {
            "roi": roi_weight,
            "non_roi": 0.0 if roi_only else max(0.0, 1.0 - roi_weight),
        },
        "roi": default_component_weights(),
        "non_roi": default_component_weights(),
        "component_thresholds": {name: 0.0 for name in COMPONENTS},
    }
    if has_target:
        model["target"] = default_component_weights()
    return model


def _merge_weights(value: dict | None) -> dict[str, float]:
    merged = default_component_weights()
    if isinstance(value, dict):
        for name in COMPONENTS:
            try:
                merged[name] = max(0.0, float(value.get(name, DEFAULT_COMPONENT_WEIGHT)))
            except (TypeError, ValueError):
                pass
    return merged


def normalize_model(value: dict | None, mode: str = "image", roi_weight: float = 0.72, has_target: bool = False) -> dict:
    base = default_model(mode, roi_weight, has_target)
    if not isinstance(value, dict):
        return base
    base["version"] = 1
    if value.get("mode") in ("roi_only", "roi_plus_full"):
        base["mode"] = value["mode"]
    region = value.get("region_weights")
    if isinstance(region, dict):
        for key in ("roi", "non_roi"):
            try:
                base["region_weights"][key] = max(0.0, float(region.get(key, base["region_weights"][key])))
            except (TypeError, ValueError):
                pass
    base["roi"] = _merge_weights(value.get("roi"))
    base["non_roi"] = _merge_weights(value.get("non_roi"))
    thresholds = value.get("component_thresholds")
    if isinstance(thresholds, dict):
        for name in COMPONENTS:
            try:
                base["component_thresholds"][name] = max(
                    0.0, min(1.0, float(thresholds.get(name, 0.0)))
                )
            except (TypeError, ValueError):
                pass
    if has_target:
        base["target"] = _merge_weights(value.get("target"))
    calibration = value.get("calibration")
    if isinstance(calibration, dict) and calibration:
        base["calibration"] = dict(calibration)
    return base


def get_model(node) -> dict:
    mode = "detect" if getattr(node, "kind", "") == "detect" else "image"
    action = getattr(node, "action", {}) or {}
    return normalize_model(
        action.get("similarity_model"),
        mode=mode,
        roi_weight=getattr(node, "roi_weight", 0.72) or 0.72,
        has_target=bool(
            getattr(node, "has_target", False)
            or getattr(node, "relative_roi", None) is not None
        ),
    )


def _to_rgb(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.float32)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array.astype(np.float32)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")
    return np.asarray(pil.resize((max(1, width), max(1, height)), Image.Resampling.BILINEAR), dtype=np.float32)


def prepare_pair(node_image, live_image) -> tuple[np.ndarray, np.ndarray]:
    node = _to_rgb(node_image)
    live = _to_rgb(live_image)
    height, width = node.shape[:2]
    if width > PREVIEW_WIDTH:
        new_height = max(1, round(height * PREVIEW_WIDTH / width))
        node = _resize(node, PREVIEW_WIDTH, new_height)
        live = _resize(live, PREVIEW_WIDTH, new_height)
    else:
        live = _resize(live, width, height)
    return node, live


def _gray(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def _region_mask(shape: tuple[int, int], roi: tuple[float, float, float, float] | None) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if roi is None:
        return mask
    height, width = shape
    x = int(round(float(roi[0]) * width))
    y = int(round(float(roi[1]) * height))
    w = int(round(float(roi[2]) * width))
    h = int(round(float(roi[3]) * height))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    mask[y : y + h, x : x + w] = False
    return mask


def _crop_region(image: np.ndarray, roi: tuple[float, float, float, float] | None) -> np.ndarray:
    if roi is None:
        return image
    height, width = image.shape[:2]
    x = int(round(float(roi[0]) * width))
    y = int(round(float(roi[1]) * height))
    w = int(round(float(roi[2]) * width))
    h = int(round(float(roi[3]) * height))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return image[y : y + h, x : x + w]


def _crop_pixels_region(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = region
    height, width = image.shape[:2]
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))
    w = max(1, min(width - x, int(w)))
    h = max(1, min(height - y, int(h)))
    return image[y : y + h, x : x + w]


def _hist(values: np.ndarray, mask: np.ndarray | None, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    data = values if mask is None else values[mask]
    if data.size == 0:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(data, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def _hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    total_a = float(a.sum())
    total_b = float(b.sum())
    if total_a <= 0 or total_b <= 0:
        # 改进：过去“两边都为空”返回 1.0，会让两张空图/纯色图误判为完全一致。
        # 现在统一返回 0，交给上层按“无效/低分”处理。
        return 0.0
    a = a / total_a
    b = b / total_b
    return float(np.clip(np.minimum(a, b).sum(), 0.0, 1.0))


def _color_score_block(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    scores = []
    for channel in range(3):
        hist_a = _hist(a[..., channel], mask, 16, (0, 255))
        hist_b = _hist(b[..., channel], mask, 16, (0, 255))
        scores.append(_hist_similarity(hist_a, hist_b))
    return float(np.mean(scores))


def _laplacian(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    lap = np.abs(
        4 * padded[1:-1, 1:-1]
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
    )
    return lap


def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    magnitude = np.sqrt(gx * gx + gy * gy)
    return gx, gy, magnitude


def _texture_score_block(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    lap_a = _laplacian(_gray(a))
    lap_b = _laplacian(_gray(b))
    hist_a = _hist(lap_a, mask, 16, (0, 80))
    hist_b = _hist(lap_b, mask, 16, (0, 80))
    return _hist_similarity(hist_a, hist_b)


def _structure_score_block(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    gx_a, gy_a, mag_a = _gradients(_gray(a))
    gx_b, gy_b, mag_b = _gradients(_gray(b))
    angle_a = (np.arctan2(gy_a, gx_a) + np.pi) % np.pi
    angle_b = (np.arctan2(gy_b, gx_b) + np.pi) % np.pi
    inner = mask
    bin_a = np.minimum(7, (angle_a / np.pi * 8).astype(np.int32)).ravel()
    bin_b = np.minimum(7, (angle_b / np.pi * 8).astype(np.int32)).ravel()
    weight_a = mag_a.ravel()
    weight_b = mag_b.ravel()
    if inner is not None:
        flat = inner.ravel()
        bin_a = bin_a[flat]
        bin_b = bin_b[flat]
        weight_a = weight_a[flat]
        weight_b = weight_b[flat]
    hist_a = np.bincount(bin_a, weights=weight_a, minlength=8).astype(np.float32)
    hist_b = np.bincount(bin_b, weights=weight_b, minlength=8).astype(np.float32)
    # 两个块都没有梯度（纯色区域）时，结构上视为“相同”，返回 1.0；
    # 整图是否为空/无效由 edge 成分和几何平均负责压分。
    if float(hist_a.sum()) <= 0 and float(hist_b.sum()) <= 0:
        return 1.0
    return _hist_similarity(hist_a, hist_b)


def _edge_score(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    _, _, mag_a = _gradients(_gray(a))
    _, _, mag_b = _gradients(_gray(b))

    def _reference(values: np.ndarray) -> float:
        return float(np.mean(values) + np.std(values)) if values.size else 0.0

    if mask is not None:
        ref_a = _reference(mag_a[mask])
        ref_b = _reference(mag_b[mask])
    else:
        ref_a = _reference(mag_a.ravel())
        ref_b = _reference(mag_b.ravel())
    # 改进：阈值由两图共同决定（对称），不再只依赖节点图。
    threshold = max(8.0, 0.6 * (ref_a + ref_b) / 2.0)
    edge_a = mag_a > threshold
    edge_b = mag_b > threshold
    cells = 4
    scores = []
    densities_a: list[float] = []
    densities_b: list[float] = []
    height, width = edge_a.shape
    for row in range(cells):
        for column in range(cells):
            y1 = row * height // cells
            y2 = (row + 1) * height // cells
            x1 = column * width // cells
            x2 = (column + 1) * width // cells
            mask_cell = mask[y1:y2, x1:x2] if mask is not None else None
            count = int(mask_cell.sum()) if mask_cell is not None else (y2 - y1) * (x2 - x1)
            if count <= 0:
                continue
            density_a = float(edge_a[y1:y2, x1:x2][mask_cell].sum()) / count if mask_cell is not None else float(edge_a[y1:y2, x1:x2].sum()) / count
            density_b = float(edge_b[y1:y2, x1:x2][mask_cell].sum()) / count if mask_cell is not None else float(edge_b[y1:y2, x1:x2].sum()) / count
            densities_a.append(density_a)
            densities_b.append(density_b)
            scores.append(1.0 - min(1.0, abs(density_a - density_b)))
    if not scores:
        return 0.0
    # 改进：两边几乎没有边缘（空白/纯色/加载页）时判为无效，避免 1.0 满分。
    if max(max(densities_a), max(densities_b)) < EDGE_MIN_DENSITY:
        return 0.0
    return float(np.mean(scores))


def _grid_average(score_fn, a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    """把图像切成 SPATIAL_GRID×SPATIAL_GRID 块，逐块算分再平均。

    这样保留了一部分空间布局信息，可以显著拉开“排版不同”的两张图；
    总像素数不变，复杂度仍是 O(HW)。
    """
    if a.shape[:2] != b.shape[:2]:
        return score_fn(a, b, mask)
    height, width = a.shape[:2]
    cell_h = height // SPATIAL_GRID
    cell_w = width // SPATIAL_GRID
    if cell_h < SPATIAL_MIN_CELL or cell_w < SPATIAL_MIN_CELL:
        return score_fn(a, b, mask)
    scores: list[float] = []
    for row in range(SPATIAL_GRID):
        y1 = row * height // SPATIAL_GRID
        y2 = (row + 1) * height // SPATIAL_GRID
        for column in range(SPATIAL_GRID):
            x1 = column * width // SPATIAL_GRID
            x2 = (column + 1) * width // SPATIAL_GRID
            cell_mask = mask[y1:y2, x1:x2] if mask is not None else None
            if cell_mask is not None and not cell_mask.any():
                continue
            scores.append(
                score_fn(
                    a[y1:y2, x1:x2],
                    b[y1:y2, x1:x2],
                    cell_mask,
                )
            )
    return float(np.mean(scores)) if scores else score_fn(a, b, mask)


def _color_score(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    return _grid_average(_color_score_block, a, b, mask)


def _texture_score(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    return _grid_average(_texture_score_block, a, b, mask)


def _structure_score(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    return _grid_average(_structure_score_block, a, b, mask)


def component_scores(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    return {
        "color": _color_score(a, b, mask),
        "texture": _texture_score(a, b, mask),
        "structure": _structure_score(a, b, mask),
        "edge": _edge_score(a, b, mask),
    }


def weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    """加权几何平均 + 短板惩罚。

    与旧的算术平均相比，任何一项很低都会明显拉低总分：
    无关图只要某个成分崩掉，就不会再被其它高成分“平均”成高分。
    """
    total = 0.0
    log_sum = 0.0
    weighted_values: list[float] = []
    for name in COMPONENTS:
        weight = max(0.0, float(weights.get(name, 0.0) or 0.0))
        if weight <= 0:
            continue
        value = max(0.0, min(1.0, float(components.get(name, 0.0))))
        total += weight
        log_sum += weight * math.log(max(value, 1e-6))
        weighted_values.append(value)
    if total <= 0:
        return float(np.mean([components.get(name, 0.0) for name in COMPONENTS]))
    geometric = math.exp(log_sum / total)
    worst = min(weighted_values)
    penalty = SHORT_PENALTY * max(0.0, SHORT_TAU - worst)
    return float(np.clip(geometric - penalty, 0.0, 1.0))


def _finalize_score(result: dict, raw: float, model: dict, calibrated: bool) -> dict:
    raw = float(np.clip(raw, 0.0, 1.0))
    result["raw_score"] = raw
    calibration = model.get("calibration") if calibrated else None
    result["score"] = apply_calibration(raw, calibration)
    return result


def explain_node(
    node, live_image, model: dict | None = None, calibrated: bool = True
) -> dict | None:
    image_path = getattr(node, "image_path", None)
    if not image_path or not Path(image_path).exists():
        return None
    if model is None:
        model = get_model(node)
    node_image, live = prepare_pair(image_path, live_image)
    roi = getattr(node, "roi", None)
    result: dict = {"model": model}
    thresholds = model.get("component_thresholds")
    if model["mode"] == "roi_only":
        components = component_scores(_crop_region(node_image, roi), _crop_region(live, roi))
        result["roi"] = components
        low = check_component_thresholds(components, thresholds)
        result["low_components"] = low
        raw = 0.0 if low else weighted_score(components, model["roi"])
        return _finalize_score(result, raw, model, calibrated)
    if roi is None:
        components = component_scores(node_image, live)
        result["roi"] = components
        low = check_component_thresholds(components, thresholds)
        result["low_components"] = low
        raw = 0.0 if low else weighted_score(components, model["roi"])
        return _finalize_score(result, raw, model, calibrated)
    roi_components = component_scores(_crop_region(node_image, roi), _crop_region(live, roi))
    roi_score = weighted_score(roi_components, model["roi"])
    result["roi"] = roi_components
    mask = _region_mask(node_image.shape[:2], roi)
    if not mask.any():
        # ROI 覆盖整张图，没有“剩余部分”，只用 ROI 分数。
        low = check_component_thresholds(roi_components, thresholds)
        result["low_components"] = low
        raw = 0.0 if low else roi_score
        return _finalize_score(result, raw, model, calibrated)
    non_components = component_scores(node_image, live, mask)
    result["non_roi"] = non_components
    low = sorted(
        set(check_component_thresholds(roi_components, thresholds))
        | set(check_component_thresholds(non_components, thresholds))
    )
    result["low_components"] = low
    non_score = weighted_score(non_components, model["non_roi"])
    weight_roi = max(0.0, float(model["region_weights"].get("roi", 0.72)))
    weight_non = max(0.0, float(model["region_weights"].get("non_roi", 0.28)))
    total = weight_roi + weight_non
    raw = 0.0 if low else (
        roi_score if total <= 0 else float(
            np.clip((weight_roi * roi_score + weight_non * non_score) / total, 0.0, 1.0)
        )
    )
    return _finalize_score(result, raw, model, calibrated)


def check_component_thresholds(
    components: dict[str, float], thresholds: dict | None
) -> list[str]:
    if not isinstance(thresholds, dict):
        return []
    low = []
    for name in COMPONENTS:
        try:
            threshold = max(0.0, float(thresholds.get(name, 0.0)))
        except (TypeError, ValueError):
            threshold = 0.0
        if threshold > 0 and float(components.get(name, 0.0)) < threshold:
            low.append(name)
    return low


def score_node(node, live_image, model: dict | None = None) -> float | None:
    result = explain_node(node, live_image, model)
    return None if result is None else float(result["score"])


def roi_explain(
    node,
    live_image,
    roi: tuple[float, float, float, float] | None,
    weights: dict[str, float],
    thresholds: dict | None = None,
    region: tuple[int, int, int, int] | None = None,
) -> dict | None:
    image_path = getattr(node, "image_path", None)
    if not image_path or not Path(image_path).exists() or roi is None:
        return None
    node_image = _to_rgb(image_path)
    live = _to_rgb(live_image)
    node_crop = _crop_region(node_image, roi)
    if region is not None:
        live_crop = _crop_pixels_region(live, region)
    else:
        live_crop = _crop_region(live, roi)
    if node_crop.size == 0 or live_crop.size == 0:
        return {"score": 0.0, "components": {}, "low_components": []}
    height, width = node_crop.shape[:2]
    live_crop = _resize(live_crop, width, height)
    components = component_scores(node_crop.astype(np.float32), live_crop.astype(np.float32))
    low = check_component_thresholds(components, thresholds)
    return {
        "score": 0.0 if low else weighted_score(components, weights),
        "components": components,
        "low_components": low,
    }


def roi_score(
    node,
    live_image,
    roi: tuple[float, float, float, float] | None,
    weights: dict[str, float],
    region: tuple[int, int, int, int] | None = None,
    thresholds: dict | None = None,
) -> float | None:
    result = roi_explain(node, live_image, roi, weights, thresholds, region)
    return None if result is None else float(result["score"])


def target_score(
    node,
    live_image,
    region: tuple[int, int, int, int] | None = None,
    model: dict | None = None,
) -> float | None:
    if not getattr(node, "has_target", False):
        return None
    if model is None:
        model = get_model(node)
    weights = model.get("target")
    if not isinstance(weights, dict):
        return None
    return roi_score(
        node,
        live_image,
        getattr(node, "target_region", None),
        weights,
        region,
        model.get("component_thresholds"),
    )


def non_roi_mask(shape: tuple[int, int], roi: tuple[float, float, float, float] | None) -> np.ndarray:
    if roi is None:
        return np.ones(shape, dtype=bool)
    return _region_mask(shape, roi)


def combine_maps(
    maps: dict[str, np.ndarray],
    weights: dict[str, float],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    base = next(iter(maps.values()))
    combined = np.zeros_like(base, dtype=np.float32)
    total = 0.0
    for name in COMPONENTS:
        weight = max(0.0, float(weights.get(name, 0.0) or 0.0))
        combined += weight * maps.get(name, np.zeros_like(base))
        total += weight
    if total > 0:
        combined /= total
    if mask is not None:
        combined = np.where(mask, combined, 0.0)
    return np.clip(combined, 0.0, 1.0)


def region_component_scores(
    node,
    live_image,
    roi: tuple[float, float, float, float] | None,
    region: tuple[int, int, int, int],
) -> dict[str, float] | None:
    image_path = getattr(node, "image_path", None)
    if not image_path or not Path(image_path).exists():
        return None
    node_image = _to_rgb(image_path)
    live = _to_rgb(live_image)
    node_crop = _crop_region(node_image, roi)
    live_crop = _crop_pixels_region(live, region)
    if node_crop.size == 0 or live_crop.size == 0:
        return None
    height, width = node_crop.shape[:2]
    live_crop = _resize(live_crop, width, height)
    return component_scores(node_crop.astype(np.float32), live_crop.astype(np.float32))


def region_component_score(
    node,
    live_image,
    roi: tuple[float, float, float, float] | None,
    region: tuple[int, int, int, int],
    weights: dict[str, float],
) -> float | None:
    components = region_component_scores(node, live_image, roi, region)
    if components is None:
        return None
    return weighted_score(components, weights)


def chart_data(node_image, model: dict | None = None) -> dict[str, np.ndarray]:
    image, _ = prepare_pair(node_image, node_image)
    gray = _gray(image)
    lap = _laplacian(gray)
    _, _, magnitude = _gradients(gray)
    color = image.max(axis=2) - image.min(axis=2)
    edge = np.where(magnitude > max(8.0, float(np.mean(magnitude) + np.std(magnitude)) * 0.6), 255.0, 0.0)
    maps = {
        "color": color,
        "texture": lap,
        "structure": magnitude,
        "edge": edge,
    }
    normalized = {}
    for name, values in maps.items():
        maximum = float(values.max())
        normalized[name] = values / maximum if maximum > 0 else np.zeros_like(values)
    if isinstance(model, dict):
        combined = np.zeros_like(gray, dtype=np.float32)
        total = 0.0
        for name in COMPONENTS:
            weight = max(0.0, float(model.get("roi", {}).get(name, 0.0) or 0.0))
            combined += weight * normalized[name]
            total += weight
        if total > 0:
            combined /= total
        normalized["combined"] = np.clip(combined, 0.0, 1.0)
    return normalized


# ---------------------------------------------------------------------------
# 阈值校准（负样本基线）
# ---------------------------------------------------------------------------
def _z_for_fpr(fpr: float) -> float:
    """常见误匹配率对应的正态分位数（不引入 scipy）。"""
    fpr = max(1e-6, min(0.5, float(fpr)))
    table = (
        (0.2, 0.8416),
        (0.1, 1.2816),
        (0.05, 1.6449),
        (0.02, 2.0537),
        (0.01, 2.3263),
        (0.005, 2.5758),
        (0.001, 3.0902),
    )
    for limit, z in table:
        if fpr <= limit:
            return z
    return 3.0902


def get_calibration(node) -> dict | None:
    model = get_model(node)
    calibration = model.get("calibration")
    return calibration if isinstance(calibration, dict) and calibration else None


def apply_calibration(score: float, calibration: dict | None) -> float:
    """把原始相似度映射到“超出负样本基线”的分数。

    默认使用 excess 模式：
        s = clip((s_raw - mu) / (1 - mu), 0, 1)
    无关图的 s_raw 约等于 mu，因此会被压到 0 附近；真实匹配仍然接近 1。
    """
    if not isinstance(calibration, dict) or not calibration:
        return float(np.clip(score, 0.0, 1.0))
    mu = float(calibration.get("mu", 0.0) or 0.0)
    mode = str(calibration.get("mode", "excess"))
    if mode == "z":
        sigma = max(1e-6, float(calibration.get("sigma", 1.0) or 1.0))
        z = (float(score) - mu) / sigma
        a = float(calibration.get("sigmoid_a", 1.5) or 1.5)
        z0 = float(calibration.get("sigmoid_z0", 1.5) or 1.5)
        return float(np.clip(1.0 / (1.0 + math.exp(-a * (z - z0))), 0.0, 1.0))
    denominator = max(1e-6, 1.0 - mu)
    return float(np.clip((float(score) - mu) / denominator, 0.0, 1.0))


def build_calibration(
    positive_scores,
    negative_scores,
    fpr: float = 0.01,
    margin: float = 0.02,
) -> dict:
    """根据正/负样本原始分数计算每节点阈值。

    r_raw = max(负样本 (1-fpr) 百分位, mu + z*sigma) + margin
    并同时保存归一化后的阈值 r_norm，运行时用归一化分数比较。
    """
    negatives = [float(value) for value in negative_scores if value is not None]
    positives = [float(value) for value in positive_scores if value is not None]
    if not negatives:
        raise ValueError("负样本不足，无法校准阈值")
    neg = np.asarray(negatives, dtype=np.float32)
    mu = float(neg.mean())
    sigma = float(neg.std())
    z = _z_for_fpr(fpr)
    percentile = float(np.percentile(neg, 100.0 * (1.0 - fpr)))
    r_raw = float(
        np.clip(max(percentile, mu + z * sigma) + max(0.0, float(margin)), 0.0, 1.0)
    )
    calibration = {
        "version": CALIBRATION_VERSION,
        "mode": "excess",
        "mu": mu,
        "sigma": sigma,
        "threshold_raw": r_raw,
        "fpr": float(fpr),
        "margin": float(margin),
        "z": z,
        "n_pos": len(positives),
        "n_neg": len(negatives),
        "pos_mean": float(np.mean(positives)) if positives else None,
        "pos_min": float(np.min(positives)) if positives else None,
        "neg_max": float(np.max(neg)) if negatives else None,
    }
    calibration["threshold_norm"] = apply_calibration(r_raw, calibration)
    return calibration


def raw_score_node(node, live_image, model: dict | None = None) -> float | None:
    """不应用校准的原始相似度，供离线校准使用。"""
    result = explain_node(node, live_image, model, calibrated=False)
    if result is None:
        return None
    return float(result.get("raw_score", result.get("score", 0.0)))


def calibrated_weighted_score(
    components: dict[str, float], weights: dict[str, float], model: dict
) -> float:
    """对“检测重点”分支的加权分应用同一套校准。"""
    raw = weighted_score(components, weights)
    return apply_calibration(raw, model.get("calibration"))
