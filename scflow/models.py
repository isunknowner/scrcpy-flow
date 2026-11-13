from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _as_dict(value: Any, default: dict | None = None) -> dict:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return dict(default or {})
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return dict(default or {})
    return parsed if isinstance(parsed, dict) else dict(default or {})


@dataclass
class Flow:
    id: int
    name: str
    description: str
    settings: dict
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row) -> "Flow":
        return cls(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            settings=_as_dict(row["settings"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @property
    def fallback(self) -> dict:
        value = self.settings.get("fallback")
        return value if isinstance(value, dict) else {}


@dataclass
class Node:
    id: int
    flow_id: int
    name: str
    kind: str
    image_path: str | None
    img_w: int | None
    img_h: int | None
    click_x: float | None
    click_y: float | None
    roi_x: float | None
    roi_y: float | None
    roi_w: float | None
    roi_h: float | None
    roi_weight: float
    target_x: float | None
    target_y: float | None
    target_w: float | None
    target_h: float | None
    action_type: str
    action: dict
    threshold: float
    is_entry: bool
    is_end: bool
    x: float
    y: float
    notes: str
    embed_kind: str | None = None
    embed_dim: int | None = None
    created_at: str = ""

    @classmethod
    def from_row(cls, row) -> "Node":
        return cls(
            id=row["id"],
            flow_id=row["flow_id"],
            name=row["name"],
            kind=row["kind"],
            image_path=row["image_path"],
            img_w=row["img_w"],
            img_h=row["img_h"],
            click_x=row["click_x"],
            click_y=row["click_y"],
            roi_x=row["roi_x"],
            roi_y=row["roi_y"],
            roi_w=row["roi_w"],
            roi_h=row["roi_h"],
            roi_weight=float(row["roi_weight"] if row["roi_weight"] is not None else 0.72),
            target_x=row["target_x"],
            target_y=row["target_y"],
            target_w=row["target_w"],
            target_h=row["target_h"],
            action_type=row["action_type"] or "tap",
            action=_as_dict(row["action_json"]),
            threshold=float(row["threshold"] or 0.88),
            is_entry=bool(row["is_entry"]),
            is_end=bool(row["is_end"]),
            x=float(row["x"] or 0),
            y=float(row["y"] or 0),
            notes=row["notes"] or "",
            embed_kind=row["embed_kind"],
            embed_dim=row["embed_dim"],
            created_at=row["created_at"] or "",
        )

    @property
    def has_image(self) -> bool:
        return bool(self.image_path)

    @property
    def has_click_point(self) -> bool:
        return self.click_x is not None and self.click_y is not None

    @property
    def has_roi(self) -> bool:
        return (
            self.roi_x is not None
            and self.roi_y is not None
            and self.roi_w is not None
            and self.roi_h is not None
            and self.roi_w > 0.01
            and self.roi_h > 0.01
        )

    @property
    def roi(self) -> tuple[float, float, float, float] | None:
        if not self.has_roi:
            return None
        return (float(self.roi_x), float(self.roi_y), float(self.roi_w), float(self.roi_h))

    @property
    def has_target(self) -> bool:
        return (
            self.target_x is not None
            and self.target_y is not None
            and self.target_w is not None
            and self.target_h is not None
            and self.target_w > 0.01
            and self.target_h > 0.01
        )

    @property
    def target_region(self) -> tuple[float, float, float, float] | None:
        if not self.has_target:
            return None
        return (
            float(self.target_x),
            float(self.target_y),
            float(self.target_w),
            float(self.target_h),
        )

    @property
    def relative_roi(self) -> tuple[float, float, float, float] | None:
        value = (self.action or {}).get("rel_roi")
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            x, y, w, h = (float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if w <= 0.01 or h <= 0.01:
            return None
        return (x, y, w, h)

    @property
    def relative_offset(self) -> tuple[float, float]:
        value = (self.action or {}).get("rel_offset")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                pass
        return (0.0, 0.0)

    @property
    def target_threshold(self) -> float:
        value = (self.action or {}).get("target_threshold")
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.55

    @property
    def action_summary(self) -> str:
        if self.action_type == "tap":
            if self.has_click_point:
                return f"点击 ({self.click_x:.3f}, {self.click_y:.3f})"
            return "点击(居中)"
        if self.action_type == "swipe":
            return "滑动"
        if self.action_type == "key":
            return f"按键 {self.action.get('key', 'back')}"
        if self.action_type == "text":
            text = str(self.action.get("text", ""))
            return f"输入 {text[:16]}"
        if self.action_type == "launch":
            package = str(self.action.get("package", ""))
            return f"启动 {package}" if package else "启动应用"
        if self.action_type == "wait":
            return f"等待 {self.action.get('seconds', 1)}s"
        return "无动作"


@dataclass
class Edge:
    id: int
    flow_id: int
    src_id: int
    dst_id: int
    weight: float
    label: str

    @classmethod
    def from_row(cls, row) -> "Edge":
        return cls(
            id=row["id"],
            flow_id=row["flow_id"],
            src_id=row["src_id"],
            dst_id=row["dst_id"],
            weight=float(row["weight"]),
            label=row["label"] or "",
        )


@dataclass
class Match:
    node: Node
    similarity: float
    bonus: float
    score: float
    roi_similarity: float | None = None
    point: tuple[int, int] | None = None
    region: tuple[int, int, int, int] | None = None


@dataclass
class Session:
    id: int
    flow_id: int
    started_at: str
    ended_at: str | None
    status: str
    steps: int
    fallbacks: int
    note: str

    @classmethod
    def from_row(cls, row) -> "Session":
        return cls(
            id=row["id"],
            flow_id=row["flow_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=row["status"],
            steps=row["steps"] or 0,
            fallbacks=row["fallbacks"] or 0,
            note=row["note"] or "",
        )


@dataclass
class EngineResult:
    status: str
    steps: int = 0
    fallbacks: int = 0
    session_id: int | None = None
    error: str | None = None


@dataclass
class Project:
    id: int
    name: str
    description: str
    python_code: str
    variables: dict
    settings: dict
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row) -> "Project":
        return cls(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            python_code=row["python_code"] or "",
            variables=_as_dict(row["variables_json"]),
            settings=_as_dict(row["settings"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class ProjectBlock:
    id: int
    project_id: int
    flow_id: int
    position: int
    alias: str
    flow_name: str = ""
    node_count: int = 0

    @classmethod
    def from_row(cls, row) -> "ProjectBlock":
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            flow_id=row["flow_id"],
            position=row["position"],
            alias=row["alias"] or "",
            flow_name=row["flow_name"] if "flow_name" in row.keys() else "",
            node_count=row["node_count"] if "node_count" in row.keys() else 0,
        )
