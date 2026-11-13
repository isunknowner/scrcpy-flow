from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import config
from .db import Database
from .embeddings import embed_to_blob, get_embedder

FORMAT = "scflow-engineering"
VERSION = 1
MANIFEST = "manifest.json"
IMAGE_PREFIX = "images/"


def _node_payload(node) -> dict:
    return {
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
        "action": dict(node.action or {}),
        "threshold": node.threshold,
        "is_entry": bool(node.is_entry),
        "is_end": bool(node.is_end),
        "x": node.x,
        "y": node.y,
        "notes": node.notes,
    }


def export_engineering(
    db: Database,
    path: str | Path,
    flow_ids: list[int],
    project_ids: list[int],
    pack_images: bool = True,
) -> Path:
    path = Path(path)
    flows = {flow.id: flow for flow in db.list_flows()}
    projects = {project.id: project for project in db.list_projects()}

    selected_flows: list[int] = []
    seen_flows: set[int] = set()
    for flow_id in flow_ids:
        if flow_id in flows and flow_id not in seen_flows:
            seen_flows.add(flow_id)
            selected_flows.append(flow_id)
    selected_projects = [pid for pid in project_ids if pid in projects]

    # Projects need their block flows even if they were not checked explicitly.
    project_blocks: dict[int, list] = {}
    for project_id in selected_projects:
        blocks = db.list_project_blocks(project_id)
        project_blocks[project_id] = blocks
        for block in blocks:
            if block.flow_id in flows and block.flow_id not in seen_flows:
                seen_flows.add(block.flow_id)
                selected_flows.append(block.flow_id)

    flow_key = {flow_id: f"f{index}" for index, flow_id in enumerate(selected_flows)}
    manifest: dict = {
        "format": FORMAT,
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pack_images": bool(pack_images),
        "flows": [],
        "projects": [],
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for flow_id in selected_flows:
            flow = flows[flow_id]
            nodes = db.list_nodes(flow_id)
            index_of = {node.id: index for index, node in enumerate(nodes)}
            node_payloads = []
            for index, node in enumerate(nodes):
                payload = _node_payload(node)
                image_path = node.image_path
                if pack_images and image_path and Path(image_path).exists():
                    suffix = Path(image_path).suffix or ".png"
                    member = f"{IMAGE_PREFIX}{flow_key[flow_id]}_{index}{suffix}"
                    archive.write(image_path, member)
                    payload["image_archive"] = member
                    payload["image_path"] = None
                else:
                    payload["image_archive"] = None
                node_payloads.append(payload)
            edge_payloads = []
            for edge in db.list_edges(flow_id):
                if edge.src_id in index_of and edge.dst_id in index_of:
                    edge_payloads.append(
                        {
                            "src": index_of[edge.src_id],
                            "dst": index_of[edge.dst_id],
                            "weight": edge.weight,
                            "label": edge.label,
                        }
                    )
            manifest["flows"].append(
                {
                    "key": flow_key[flow_id],
                    "name": flow.name,
                    "description": flow.description,
                    "settings": flow.settings,
                    "nodes": node_payloads,
                    "edges": edge_payloads,
                }
            )

        for project_id in selected_projects:
            project = projects[project_id]
            manifest["projects"].append(
                {
                    "name": project.name,
                    "description": project.description,
                    "python_code": project.python_code,
                    "variables": project.variables,
                    "settings": project.settings,
                    "blocks": [
                        {
                            "flow_key": flow_key.get(block.flow_id),
                            "position": block.position,
                            "alias": block.alias,
                        }
                        for block in project_blocks.get(project_id, [])
                        if block.flow_id in flow_key
                    ],
                }
            )

        archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
    return path


def import_engineering(db: Database, path: str | Path) -> dict:
    path = Path(path)
    with zipfile.ZipFile(path, "r") as archive:
        try:
            manifest = json.loads(archive.read(MANIFEST).decode("utf-8"))
        except KeyError as exc:
            raise ValueError("不是有效的工程文件（缺少 manifest.json）") from exc
        if manifest.get("format") != FORMAT:
            raise ValueError("不是有效的 scflow 工程文件")

        images_dir = config.images_dir()
        flow_id_by_key: dict[str, int] = {}
        imported_flows = 0
        imported_projects = 0

        for flow_entry in manifest.get("flows", []):
            name = db.unique_flow_name(flow_entry.get("name") or "流程")
            flow_id = db.create_flow(
                name,
                description=flow_entry.get("description") or "",
                settings=flow_entry.get("settings") or {},
            )
            flow_id_by_key[flow_entry.get("key")] = flow_id
            imported_flows += 1
            node_ids = []
            for node_entry in flow_entry.get("nodes", []):
                image_path = node_entry.get("image_path")
                member = node_entry.get("image_archive")
                if member:
                    try:
                        data = archive.read(member)
                    except KeyError:
                        data = None
                    if data:
                        suffix = Path(member).suffix or ".png"
                        target = images_dir / (
                            f"eng_{flow_id}_{len(node_ids)}_{uuid4().hex[:8]}{suffix}"
                        )
                        target.write_bytes(data)
                        image_path = str(target)
                node_id = db.add_node(
                    flow_id,
                    name=node_entry.get("name") or "导入节点",
                    kind=node_entry.get("kind", "image"),
                    image_path=image_path,
                    img_w=node_entry.get("img_w"),
                    img_h=node_entry.get("img_h"),
                    click_x=node_entry.get("click_x"),
                    click_y=node_entry.get("click_y"),
                    roi_x=node_entry.get("roi_x"),
                    roi_y=node_entry.get("roi_y"),
                    roi_w=node_entry.get("roi_w"),
                    roi_h=node_entry.get("roi_h"),
                    roi_weight=float(node_entry.get("roi_weight", 0.72) or 0.72),
                    target_x=node_entry.get("target_x"),
                    target_y=node_entry.get("target_y"),
                    target_w=node_entry.get("target_w"),
                    target_h=node_entry.get("target_h"),
                    action_type=node_entry.get("action_type", "tap"),
                    action=node_entry.get("action") or {},
                    threshold=float(node_entry.get("threshold", 0.88) or 0.88),
                    is_entry=bool(node_entry.get("is_entry", False)),
                    is_end=bool(node_entry.get("is_end", False)),
                    x=float(node_entry.get("x", 0) or 0),
                    y=float(node_entry.get("y", 0) or 0),
                    notes=node_entry.get("notes", ""),
                )
                node_ids.append(node_id)
                if image_path and Path(image_path).exists():
                    try:
                        embedder = get_embedder()
                        vector = embedder.embed(image_path)
                        db.set_node_embedding(
                            node_id,
                            embed_to_blob(vector),
                            embedder.kind,
                            int(vector.shape[0]),
                        )
                    except Exception:
                        pass
            for edge_entry in flow_entry.get("edges", []):
                src_index = edge_entry.get("src")
                dst_index = edge_entry.get("dst")
                if (
                    isinstance(src_index, int)
                    and isinstance(dst_index, int)
                    and 0 <= src_index < len(node_ids)
                    and 0 <= dst_index < len(node_ids)
                ):
                    db.add_edge(
                        flow_id,
                        node_ids[src_index],
                        node_ids[dst_index],
                        float(edge_entry.get("weight", 1.0) or 1.0),
                        edge_entry.get("label", ""),
                    )

        for project_entry in manifest.get("projects", []):
            name = db.unique_project_name(project_entry.get("name") or "项目")
            project_id = db.create_project(
                name,
                description=project_entry.get("description") or "",
                python_code=project_entry.get("python_code") or "",
                variables=project_entry.get("variables") or {},
                settings=project_entry.get("settings") or {},
            )
            imported_projects += 1
            for block in project_entry.get("blocks", []):
                flow_id = flow_id_by_key.get(block.get("flow_key"))
                if flow_id is None:
                    continue
                try:
                    db.add_project_block(
                        project_id,
                        flow_id,
                        alias=block.get("alias"),
                        position=block.get("position"),
                    )
                except Exception:
                    pass

    return {"flows": imported_flows, "projects": imported_projects}
