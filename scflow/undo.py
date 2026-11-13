from __future__ import annotations

import json
from typing import Any

from .db import Database


def snapshot_flow(db: Database, flow_id: int) -> dict | None:
    flow = db.get_flow(flow_id)
    if flow is None:
        return None
    nodes = db.list_nodes(flow_id)
    edges = db.list_edges(flow_id)
    embeddings = db.get_embeddings(flow_id)
    node_data = []
    for node in nodes:
        item = {
            "id": node.id,
            "name": node.name,
            "kind": node.kind,
            "image_path": node.image_path,
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
            "action_json": json.dumps(node.action, ensure_ascii=False),
            "threshold": node.threshold,
            "is_entry": 1 if node.is_entry else 0,
            "is_end": 1 if node.is_end else 0,
            "x": node.x,
            "y": node.y,
            "notes": node.notes,
            "created_at": node.created_at,
            "embed": None,
            "embed_kind": node.embed_kind,
            "embed_dim": node.embed_dim,
        }
        embedding = embeddings.get(node.id)
        if embedding:
            item["embed"], item["embed_kind"], item["embed_dim"] = embedding
        node_data.append(item)
    edge_data = [
        {
            "id": edge.id,
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "weight": edge.weight,
            "label": edge.label,
        }
        for edge in edges
    ]
    return {
        "flow": {
            "id": flow.id,
            "name": flow.name,
            "description": flow.description,
            "settings": flow.settings,
            "created_at": flow.created_at,
            "updated_at": flow.updated_at,
        },
        "nodes": node_data,
        "edges": edge_data,
    }


def restore_flow(db: Database, flow_id: int, snapshot: dict) -> None:
    flow = snapshot["flow"]
    db.restore_flow_row(
        flow_id,
        flow["name"],
        flow["description"],
        flow["settings"],
        flow["created_at"],
        flow["updated_at"],
    )
    db.replace_flow_content(flow_id, snapshot["nodes"], snapshot["edges"])


def snapshot_project(db: Database, project_id: int) -> dict | None:
    project = db.get_project(project_id)
    if project is None:
        return None
    blocks = db.list_project_blocks(project_id)
    return {
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "python_code": project.python_code,
            "variables": project.variables,
            "settings": project.settings,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
        },
        "blocks": [
            {
                "id": block.id,
                "flow_id": block.flow_id,
                "position": block.position,
                "alias": block.alias,
            }
            for block in blocks
        ],
    }


def restore_project(db: Database, project_id: int, snapshot: dict) -> None:
    project = snapshot["project"]
    db.restore_project_row(
        project_id,
        project["name"],
        project["description"],
        project["python_code"],
        project["variables"],
        project["settings"],
        project["created_at"],
        project["updated_at"],
    )
    blocks = [
        block
        for block in snapshot["blocks"]
        if db.get_flow(block["flow_id"]) is not None
    ]
    db.replace_project_blocks(project_id, blocks)


class UndoStack:
    def __init__(self, limit: int = 50):
        self.limit = limit
        self.items: list[tuple[str, Any]] = []

    def push(self, label: str, snapshot: Any) -> None:
        if snapshot is None:
            return
        self.items.append((label, snapshot))
        if len(self.items) > self.limit:
            del self.items[0 : len(self.items) - self.limit]

    def pop(self) -> tuple[str, Any] | None:
        if not self.items:
            return None
        return self.items.pop()

    def clear(self) -> None:
        self.items.clear()

    def __bool__(self) -> bool:
        return bool(self.items)

    def __len__(self) -> int:
        return len(self.items)
