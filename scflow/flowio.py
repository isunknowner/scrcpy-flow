from __future__ import annotations

import base64
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import config
from .db import Database
from .embeddings import embed_to_blob, get_embedder

FORMAT = "scflow.flow"
VERSION = 1


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", name, flags=re.UNICODE).strip("._")
    return cleaned[:48] or "image"


def export_flow(db: Database, flow_id: int, path: str | Path) -> Path:
    flow = db.get_flow(flow_id)
    if flow is None:
        raise ValueError("流程不存在")
    nodes = db.list_nodes(flow_id)
    edges = db.list_edges(flow_id)
    key_of = {node.id: f"n{index + 1}" for index, node in enumerate(nodes)}
    images: dict[str, dict] = {}
    exported_nodes = []
    for node in nodes:
        item = {
            "key": key_of[node.id],
            "name": node.name,
            "kind": node.kind,
            "img_w": node.img_w,
            "img_h": node.img_h,
            "click_x": node.click_x,
            "click_y": node.click_y,
            "roi_x": node.roi_x,
            "roi_y": node.roi_y,
            "roi_w": node.roi_w,
            "roi_h": node.roi_h,
            "roi_weight": node.roi_weight,
            "target_x": node.target_x,
            "target_y": node.target_y,
            "target_w": node.target_w,
            "target_h": node.target_h,
            "action_type": node.action_type,
            "action": node.action,
            "threshold": node.threshold,
            "is_entry": node.is_entry,
            "is_end": node.is_end,
            "x": node.x,
            "y": node.y,
            "notes": node.notes,
            "image": None,
        }
        if node.image_path and Path(node.image_path).exists():
            source = Path(node.image_path)
            filename = f"{item['key']}_{_slug(source.stem)}{source.suffix or '.png'}"
            data = base64.b64encode(source.read_bytes()).decode("ascii")
            images[filename] = {"data": data, "mime": "image/png"}
            item["image"] = filename
        exported_nodes.append(item)
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "flow": {
            "name": flow.name,
            "description": flow.description,
            "settings": flow.settings,
        },
        "nodes": exported_nodes,
        "edges": [
            {
                "src": key_of[edge.src_id],
                "dst": key_of[edge.dst_id],
                "weight": edge.weight,
                "label": edge.label,
            }
            for edge in edges
        ],
        "images": images,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _save_imported_image(b64: str, flow_id: int, name: str) -> tuple[str, int, int]:
    raw = base64.b64decode(b64)
    suffix = Path(name).suffix or ".png"
    target = config.images_dir() / f"flow{flow_id}_{uuid.uuid4().hex[:10]}_{_slug(Path(name).stem)}{suffix}"
    target.write_bytes(raw)
    with Image.open(target) as handle:
        width, height = handle.size
    return str(target), width, height


def connect_nodes(
    db: Database, src_node_id: int, dst_node_id: int, weight: float = 1.0, label: str = ""
) -> int:
    src = db.get_node(src_node_id)
    dst = db.get_node(dst_node_id)
    if src is None or dst is None:
        raise ValueError("节点不存在")
    if src.flow_id != dst.flow_id:
        raise ValueError("只能连接同一个流程中的节点")
    return db.add_edge(src.flow_id, src_node_id, dst_node_id, weight, label)


def import_flow(
    db: Database,
    path: str | Path,
    target_flow_id: int | None = None,
    link_from_node_id: int | None = None,
    link_to_node_id: int | None = None,
    link_weight: float = 1.0,
    new_flow_name: str | None = None,
) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError("不是 scflow 流程文件")
    exported_flow = payload.get("flow", {})
    if target_flow_id is None:
        name = db.unique_flow_name(new_flow_name or exported_flow.get("name") or "导入流程")
        target_flow_id = db.create_flow(
            name,
            exported_flow.get("description", ""),
            exported_flow.get("settings") or {},
        )
    else:
        if db.get_flow(target_flow_id) is None:
            raise ValueError("目标流程不存在")

    images = payload.get("images", {})
    embedder = get_embedder()
    key_to_id: dict[str, int] = {}
    imported_nodes = []
    for item in payload.get("nodes", []):
        image_path = None
        img_w = item.get("img_w")
        img_h = item.get("img_h")
        image_name = item.get("image")
        if image_name and image_name in images:
            data = images[image_name]
            raw = data.get("data", "") if isinstance(data, dict) else str(data)
            image_path, img_w, img_h = _save_imported_image(raw, target_flow_id, image_name)
        node_id = db.add_node(
            target_flow_id,
            item.get("name") or "导入节点",
            kind=item.get("kind", "image"),
            image_path=image_path,
            img_w=img_w,
            img_h=img_h,
            click_x=item.get("click_x"),
            click_y=item.get("click_y"),
            roi_x=item.get("roi_x"),
            roi_y=item.get("roi_y"),
            roi_w=item.get("roi_w"),
            roi_h=item.get("roi_h"),
            roi_weight=float(item.get("roi_weight", 0.72) or 0.72),
            target_x=item.get("target_x"),
            target_y=item.get("target_y"),
            target_w=item.get("target_w"),
            target_h=item.get("target_h"),
            action_type=item.get("action_type", "tap"),
            action=item.get("action") or {},
            threshold=float(item.get("threshold", 0.88) or 0.88),
            is_entry=False,
            is_end=bool(item.get("is_end", False) or item.get("kind") == "end"),
            x=float(item.get("x", 0) or 0),
            y=float(item.get("y", 0) or 0),
            notes=item.get("notes", ""),
        )
        key_to_id[item.get("key")] = node_id
        imported_nodes.append((item, node_id))
        if image_path:
            try:
                vector = embedder.embed(image_path)
                db.set_node_embedding(node_id, embed_to_blob(vector), embedder.kind, int(vector.shape[0]))
            except Exception:
                pass

    for edge in payload.get("edges", []):
        src_id = key_to_id.get(edge.get("src"))
        dst_id = key_to_id.get(edge.get("dst"))
        if src_id and dst_id:
            db.add_edge(
                target_flow_id,
                src_id,
                dst_id,
                float(edge.get("weight", 1.0) or 1.0),
                edge.get("label", ""),
            )

    entries = [
        node_id for item, node_id in imported_nodes if item.get("is_entry")
    ] or [node_id for item, node_id in imported_nodes if item.get("image")]
    target_flow = db.get_flow(target_flow_id)
    target_nodes = db.list_nodes(target_flow_id)
    has_entry = any(node.is_entry for node in target_nodes if node.id not in key_to_id.values())
    entry_id = entries[0] if entries else None
    if entry_id and not has_entry:
        db.set_entry(target_flow_id, entry_id)

    linked = 0
    if link_from_node_id and entry_id:
        connect_nodes(db, link_from_node_id, entry_id, link_weight, "导入衔接")
        linked += 1
    if link_to_node_id:
        imported_out = {edge["src"] for edge in payload.get("edges", [])}
        terminals = [
            node_id
            for item, node_id in imported_nodes
            if item.get("key") not in imported_out
        ]
        for node_id in terminals:
            connect_nodes(db, node_id, link_to_node_id, link_weight, "导入衔接")
            linked += 1

    return {
        "flow_id": target_flow_id,
        "flow_name": target_flow.name if target_flow else "",
        "nodes": len(imported_nodes),
        "edges": len(payload.get("edges", [])),
        "entry": entry_id,
        "links": linked,
    }
