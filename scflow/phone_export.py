from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import vision
from .db import Database

FORMAT = "scflow.phone"
VERSION = 1


def _safe_package(package: str) -> str:
    parts = [re.sub(r"[^A-Za-z0-9_]", "_", part) for part in (package or "").split(".") if part]
    if not parts:
        return "com.scflow.phone.runner"
    if not parts[0].isalpha():
        parts.insert(0, "com")
    return ".".join(parts).lower()


def _default_package(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]", "_", (name or "flow").lower())
    slug = re.sub(r"_+", "_", slug).strip("_") or "flow"
    if not slug[0].isalpha():
        slug = "app_" + slug
    return f"com.scflow.phone.{slug[:24]}"


def _default_app_name(name: str) -> str:
    return (name or "scflow 手机流程").strip()[:40]


def _node_payload(node, flow_key: str, index: int, key_of: dict[int, str], images: dict[str, bytes]) -> dict:
    image_name = None
    img_w = node.img_w
    img_h = node.img_h
    if node.image_path and Path(node.image_path).exists():
        source = Path(node.image_path)
        try:
            frame = vision.load_rgb(source)
            image_name = f"images/{flow_key}_n{index}.png"
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="PNG")
            images[image_name] = buffer.getvalue()
            img_h, img_w = frame.shape[:2]
        except Exception:
            image_name = None
    action = dict(node.action or {})
    branches = []
    for branch in action.get("if_branches") or []:
        try:
            key = key_of.get(int(branch))
        except (TypeError, ValueError):
            key = None
        if key:
            branches.append(key)
    action["if_branches"] = branches
    default = action.get("if_default")
    if default is not None:
        try:
            action["if_default"] = key_of.get(int(default))
        except (TypeError, ValueError):
            action["if_default"] = None
    return {
        "id": key_of[node.id],
        "name": node.name,
        "kind": node.kind,
        "image": image_name,
        "imgW": img_w,
        "imgH": img_h,
        "threshold": node.threshold,
        "roi": list(node.roi) if node.roi else None,
        "roiWeight": node.roi_weight,
        "target": list(node.target_region) if node.target_region else None,
        "targetThreshold": node.target_threshold,
        "relativeRoi": list(node.relative_roi) if node.relative_roi else None,
        "relativeOffset": list(node.relative_offset),
        "click": (
            [node.click_x, node.click_y] if node.has_click_point else None
        ),
        "actionType": node.action_type,
        "action": action,
        "isEntry": bool(node.is_entry),
        "isEnd": bool(node.is_end),
        "notes": node.notes,
    }


def _flow_payload(db: Database, flow_id: int, flow_key: str, images: dict[str, bytes]) -> dict:
    flow = db.get_flow(flow_id)
    if flow is None:
        raise ValueError("流程不存在")
    nodes = db.list_nodes(flow_id)
    key_of = {node.id: f"n{index}" for index, node in enumerate(nodes)}
    node_payloads = [
        _node_payload(node, flow_key, index, key_of, images)
        for index, node in enumerate(nodes)
    ]
    edges = []
    for edge in db.list_edges(flow_id):
        src = key_of.get(edge.src_id)
        dst = key_of.get(edge.dst_id)
        if src and dst:
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "weight": edge.weight,
                    "label": edge.label,
                }
            )
    entry = next((node for node in nodes if node.is_entry), None)
    if entry is None and nodes:
        entry = nodes[0]
    return {
        "key": flow_key,
        "name": flow.name,
        "entryId": key_of.get(entry.id) if entry else None,
        "settings": flow.settings or {},
        "nodes": node_payloads,
        "edges": edges,
    }


def export_phone_package(
    db: Database,
    path: str | Path,
    flow_id: int | None = None,
    project_id: int | None = None,
    app_name: str | None = None,
    package_name: str | None = None,
    version_name: str = "1.0.0",
    version_code: int = 1,
) -> Path:
    if flow_id is None and project_id is None:
        raise ValueError("请选择要导出的流程或项目")

    flow_ids: list[int] = []
    project_payload = None
    entry_type = "flow"
    entry_key = None
    entry_name = ""
    if project_id is not None:
        project = db.get_project(project_id)
        if project is None:
            raise ValueError("项目不存在")
        blocks = db.list_project_blocks(project_id)
        flow_ids = [block.flow_id for block in blocks]
        entry_type = "project"
        entry_name = project.name
        base_name = project.name
    else:
        flow = db.get_flow(int(flow_id))
        if flow is None:
            raise ValueError("流程不存在")
        flow_ids = [flow.id]
        entry_type = "flow"
        entry_name = flow.name
        base_name = flow.name

    if not flow_ids:
        raise ValueError("项目中没有流程区块")

    unique_flow_ids = []
    seen = set()
    for flow_id_value in flow_ids:
        if flow_id_value not in seen:
            seen.add(flow_id_value)
            unique_flow_ids.append(flow_id_value)
    flow_ids = unique_flow_ids

    key_of_flow = {flow_id_value: f"f{index}" for index, flow_id_value in enumerate(flow_ids)}
    images: dict[str, bytes] = {}
    flows = [
        _flow_payload(db, flow_id_value, key_of_flow[flow_id_value], images)
        for flow_id_value in flow_ids
    ]
    entry_key = key_of_flow[flow_ids[0]] if entry_type == "flow" else None
    if project_id is not None:
        project_payload = {
            "name": entry_name,
            "flowKeys": [key_of_flow[flow_id_value] for flow_id_value in flow_ids],
        }

    display_name = _default_app_name(app_name or base_name)
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "exportedAt": datetime.now().isoformat(timespec="seconds"),
        "app": {
            "name": display_name,
            "package": _safe_package(package_name or _default_package(display_name)),
            "versionName": version_name or "1.0.0",
            "versionCode": int(version_code or 1),
        },
        "entry": {
            "type": entry_type,
            "key": entry_key,
            "name": entry_name,
        },
        "flows": flows,
        "projects": [project_payload] if project_payload else [],
    }

    target = Path(path)
    if target.suffix.lower() != ".scflowpkg":
        target = target.with_suffix(".scflowpkg")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        for name, data in images.items():
            archive.writestr(name, data)
    return target
