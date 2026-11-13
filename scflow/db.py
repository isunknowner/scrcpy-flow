from __future__ import annotations

import json
import keyword
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from . import config
from .models import Edge, Flow, Node, Project, ProjectBlock, Session

SCHEMA = """
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    settings TEXT NOT NULL DEFAULT '{}',
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'image',
    image_path TEXT,
    img_w INTEGER,
    img_h INTEGER,
    click_x REAL,
    click_y REAL,
    roi_x REAL,
    roi_y REAL,
    roi_w REAL,
    roi_h REAL,
    roi_weight REAL NOT NULL DEFAULT 0.72,
    target_x REAL,
    target_y REAL,
    target_w REAL,
    target_h REAL,
    action_type TEXT NOT NULL DEFAULT 'tap',
    action_json TEXT NOT NULL DEFAULT '{}',
    threshold REAL NOT NULL DEFAULT 0.88,
    is_entry INTEGER NOT NULL DEFAULT 0,
    is_end INTEGER NOT NULL DEFAULT 0,
    x REAL NOT NULL DEFAULT 0,
    y REAL NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    embed BLOB,
    embed_kind TEXT,
    embed_dim INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_flow ON nodes(flow_id);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
    src_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    weight REAL NOT NULL DEFAULT 1.0,
    label TEXT NOT NULL DEFAULT '',
    UNIQUE(src_id, dst_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_flow ON edges(flow_id);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    steps INTEGER NOT NULL DEFAULT 0,
    fallbacks INTEGER NOT NULL DEFAULT 0,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sessions_flow ON sessions(flow_id);

CREATE TABLE IF NOT EXISTS step_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    ts TEXT NOT NULL,
    current_node_id INTEGER,
    matched_node_id INTEGER,
    similarity REAL,
    score REAL,
    action TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT '',
    screenshot_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_logs_session ON step_logs(session_id);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    python_code TEXT NOT NULL DEFAULT '',
    variables_json TEXT NOT NULL DEFAULT '{}',
    settings TEXT NOT NULL DEFAULT '{}',
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS project_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    flow_id INTEGER NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    alias TEXT NOT NULL DEFAULT '',
    UNIQUE(project_id, flow_id)
);
CREATE INDEX IF NOT EXISTS idx_project_blocks_project ON project_blocks(project_id);

CREATE TABLE IF NOT EXISTS project_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    project_name TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    blocks INTEGER NOT NULL DEFAULT 0,
    steps INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    variables_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_project_runs_project ON project_runs(project_id);
"""

NODE_FIELDS = {
    "name",
    "kind",
    "image_path",
    "img_w",
    "img_h",
    "click_x",
    "click_y",
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "roi_weight",
    "target_x",
    "target_y",
    "target_w",
    "target_h",
    "action_type",
    "action_json",
    "threshold",
    "is_entry",
    "is_end",
    "x",
    "y",
    "notes",
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else config.db_file()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self.connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def init(self) -> None:
        with self.tx() as conn:
            conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        with self.tx() as conn:
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()
            }
            for name in ("target_x", "target_y", "target_w", "target_h"):
                if name not in columns:
                    conn.execute(f"ALTER TABLE nodes ADD COLUMN {name} REAL")
            if "is_end" not in columns:
                conn.execute(
                    "ALTER TABLE nodes ADD COLUMN is_end INTEGER NOT NULL DEFAULT 0"
                )
            conn.execute("UPDATE nodes SET is_end=1 WHERE kind='end' AND is_end=0")

            flow_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(flows)").fetchall()
            }
            if "position" not in flow_columns:
                conn.execute(
                    "ALTER TABLE flows ADD COLUMN position INTEGER NOT NULL DEFAULT 0"
                )
            conn.execute("UPDATE flows SET position=id WHERE position=0")

            project_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(projects)").fetchall()
            }
            if "position" not in project_columns:
                conn.execute(
                    "ALTER TABLE projects ADD COLUMN position INTEGER NOT NULL DEFAULT 0"
                )
            conn.execute("UPDATE projects SET position=id WHERE position=0")

    def create_flow(self, name: str, description: str = "", settings: dict | None = None) -> int:
        ts = now()
        with self.tx() as conn:
            position = conn.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM flows"
            ).fetchone()[0]
            cur = conn.execute(
                """INSERT INTO flows(name, description, settings, position, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    name,
                    description,
                    json.dumps(settings or {}, ensure_ascii=False),
                    position,
                    ts,
                    ts,
                ),
            )
            return int(cur.lastrowid)

    def unique_flow_name(self, base: str) -> str:
        base = (base or "流程").strip() or "流程"
        names = {flow.name for flow in self.list_flows()}
        if base not in names:
            return base
        i = 2
        while f"{base} ({i})" in names:
            i += 1
        return f"{base} ({i})"

    def get_flow(self, flow_id: int) -> Flow | None:
        with self.read() as conn:
            row = conn.execute("SELECT * FROM flows WHERE id=?", (flow_id,)).fetchone()
        return Flow.from_row(row) if row else None

    def get_flow_by_name(self, name: str) -> Flow | None:
        with self.read() as conn:
            row = conn.execute("SELECT * FROM flows WHERE name=?", (name,)).fetchone()
        return Flow.from_row(row) if row else None

    def list_flows(self) -> list[Flow]:
        with self.read() as conn:
            rows = conn.execute("SELECT * FROM flows ORDER BY position, id").fetchall()
        return [Flow.from_row(row) for row in rows]

    def reorder_flows(self, flow_ids: list[int]) -> None:
        with self.tx() as conn:
            for position, flow_id in enumerate(flow_ids, start=1):
                conn.execute(
                    "UPDATE flows SET position=? WHERE id=?", (position, flow_id)
                )

    def update_flow(self, flow_id: int, **fields: Any) -> None:
        allowed = {"name", "description", "settings", "position"}
        data = {key: value for key, value in fields.items() if key in allowed}
        if not data:
            return
        if "settings" in data and not isinstance(data["settings"], str):
            data["settings"] = json.dumps(data["settings"], ensure_ascii=False)
        data["updated_at"] = now()
        columns = ", ".join(f"{key}=?" for key in data)
        with self.tx() as conn:
            conn.execute(f"UPDATE flows SET {columns} WHERE id=?", (*data.values(), flow_id))

    def delete_flow(self, flow_id: int) -> None:
        with self.tx() as conn:
            conn.execute("DELETE FROM flows WHERE id=?", (flow_id,))

    def add_node(
        self,
        flow_id: int,
        name: str,
        kind: str = "image",
        image_path: str | None = None,
        img_w: int | None = None,
        img_h: int | None = None,
        click_x: float | None = None,
        click_y: float | None = None,
        roi_x: float | None = None,
        roi_y: float | None = None,
        roi_w: float | None = None,
        roi_h: float | None = None,
        roi_weight: float = 0.72,
        target_x: float | None = None,
        target_y: float | None = None,
        target_w: float | None = None,
        target_h: float | None = None,
        action_type: str = "tap",
        action: dict | None = None,
        threshold: float = 0.88,
        is_entry: bool = False,
        is_end: bool = False,
        x: float = 0,
        y: float = 0,
        notes: str = "",
    ) -> int:
        with self.tx() as conn:
            cur = conn.execute(
                """INSERT INTO nodes(flow_id,name,kind,image_path,img_w,img_h,click_x,click_y,
                   roi_x,roi_y,roi_w,roi_h,roi_weight,
                   target_x,target_y,target_w,target_h,
                   action_type,action_json,threshold,is_entry,is_end,x,y,notes,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    flow_id,
                    name,
                    kind,
                    image_path,
                    img_w,
                    img_h,
                    click_x,
                    click_y,
                    roi_x,
                    roi_y,
                    roi_w,
                    roi_h,
                    roi_weight,
                    target_x,
                    target_y,
                    target_w,
                    target_h,
                    action_type,
                    json.dumps(action or {}, ensure_ascii=False),
                    threshold,
                    1 if is_entry else 0,
                    1 if is_end else 0,
                    x,
                    y,
                    notes,
                    now(),
                ),
            )
            node_id = int(cur.lastrowid)
            if is_entry:
                conn.execute(
                    "UPDATE nodes SET is_entry=0 WHERE flow_id=? AND id<>?", (flow_id, node_id)
                )
            return node_id

    def get_node(self, node_id: int) -> Node | None:
        with self.read() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return Node.from_row(row) if row else None

    def list_nodes(self, flow_id: int) -> list[Node]:
        with self.read() as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE flow_id=? ORDER BY id", (flow_id,)
            ).fetchall()
        return [Node.from_row(row) for row in rows]

    def update_node(self, node_id: int, **fields: Any) -> None:
        data = {key: value for key, value in fields.items() if key in NODE_FIELDS}
        if "action" in fields and "action_json" not in data:
            data["action_json"] = json.dumps(fields["action"] or {}, ensure_ascii=False)
        if not data:
            return
        if "is_entry" in data:
            data["is_entry"] = 1 if data["is_entry"] else 0
        if "is_end" in data:
            data["is_end"] = 1 if data["is_end"] else 0
        columns = ", ".join(f"{key}=?" for key in data)
        with self.tx() as conn:
            conn.execute(f"UPDATE nodes SET {columns} WHERE id=?", (*data.values(), node_id))
            if data.get("is_entry"):
                flow_id = conn.execute(
                    "SELECT flow_id FROM nodes WHERE id=?", (node_id,)
                ).fetchone()[0]
                conn.execute(
                    "UPDATE nodes SET is_entry=0 WHERE flow_id=? AND id<>?", (flow_id, node_id)
                )

    def set_entry(self, flow_id: int, node_id: int) -> None:
        with self.tx() as conn:
            conn.execute("UPDATE nodes SET is_entry=0 WHERE flow_id=?", (flow_id,))
            conn.execute(
                "UPDATE nodes SET is_entry=1 WHERE id=? AND flow_id=?", (node_id, flow_id)
            )

    def delete_node(self, node_id: int) -> None:
        with self.tx() as conn:
            conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))

    def set_node_embedding(self, node_id: int, blob: bytes, kind: str, dim: int) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE nodes SET embed=?, embed_kind=?, embed_dim=? WHERE id=?",
                (sqlite3.Binary(blob), kind, dim, node_id),
            )

    def get_embeddings(self, flow_id: int) -> dict[int, tuple[bytes, str, int]]:
        with self.read() as conn:
            rows = conn.execute(
                "SELECT id, embed, embed_kind, embed_dim FROM nodes WHERE flow_id=? AND embed IS NOT NULL",
                (flow_id,),
            ).fetchall()
        return {
            row["id"]: (bytes(row["embed"]), row["embed_kind"], int(row["embed_dim"] or 0))
            for row in rows
        }

    def add_edge(
        self, flow_id: int, src_id: int, dst_id: int, weight: float = 1.0, label: str = ""
    ) -> int:
        with self.tx() as conn:
            conn.execute(
                """INSERT INTO edges(flow_id,src_id,dst_id,weight,label) VALUES (?,?,?,?,?)
                   ON CONFLICT(src_id,dst_id) DO UPDATE SET weight=excluded.weight,label=excluded.label""",
                (flow_id, src_id, dst_id, weight, label),
            )
            row = conn.execute(
                "SELECT id FROM edges WHERE src_id=? AND dst_id=?", (src_id, dst_id)
            ).fetchone()
            return int(row["id"])

    def update_edge(self, edge_id: int, **fields: Any) -> None:
        data = {key: value for key, value in fields.items() if key in {"weight", "label"}}
        if not data:
            return
        columns = ", ".join(f"{key}=?" for key in data)
        with self.tx() as conn:
            conn.execute(f"UPDATE edges SET {columns} WHERE id=?", (*data.values(), edge_id))

    def delete_edge(self, edge_id: int) -> None:
        with self.tx() as conn:
            conn.execute("DELETE FROM edges WHERE id=?", (edge_id,))

    def list_edges(self, flow_id: int) -> list[Edge]:
        with self.read() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE flow_id=? ORDER BY id", (flow_id,)
            ).fetchall()
        return [Edge.from_row(row) for row in rows]

    def get_edge(self, edge_id: int) -> Edge | None:
        with self.read() as conn:
            row = conn.execute("SELECT * FROM edges WHERE id=?", (edge_id,)).fetchone()
        return Edge.from_row(row) if row else None

    def create_session(self, flow_id: int, note: str = "") -> int:
        with self.tx() as conn:
            cur = conn.execute(
                "INSERT INTO sessions(flow_id,started_at,status,note) VALUES (?,?,?,?)",
                (flow_id, now(), "running", note),
            )
            return int(cur.lastrowid)

    def finish_session(
        self, session_id: int, status: str, steps: int, fallbacks: int, note: str = ""
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=?, status=?, steps=?, fallbacks=?, note=? WHERE id=?",
                (now(), status, steps, fallbacks, note, session_id),
            )

    def log_step(
        self,
        session_id: int,
        idx: int,
        current_node_id: int | None,
        matched_node_id: int | None,
        similarity: float | None,
        score: float | None,
        action: str,
        result: str,
        screenshot_path: str | None = None,
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                """INSERT INTO step_logs(session_id,idx,ts,current_node_id,matched_node_id,
                   similarity,score,action,result,screenshot_path) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    idx,
                    now(),
                    current_node_id,
                    matched_node_id,
                    similarity,
                    score,
                    action,
                    result,
                    screenshot_path,
                ),
            )

    def list_sessions(self, flow_id: int | None = None, limit: int = 100) -> list[Session]:
        with self.read() as conn:
            if flow_id is None:
                rows = conn.execute(
                    "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE flow_id=? ORDER BY id DESC LIMIT ?",
                    (flow_id, limit),
                ).fetchall()
        return [Session.from_row(row) for row in rows]

    def list_logs(self, session_id: int) -> list[sqlite3.Row]:
        with self.read() as conn:
            return conn.execute(
                """SELECT l.*, c.name AS current_name, m.name AS matched_name
                   FROM step_logs l
                   LEFT JOIN nodes c ON c.id=l.current_node_id
                   LEFT JOIN nodes m ON m.id=l.matched_node_id
                   WHERE l.session_id=? ORDER BY l.idx, l.id""",
                (session_id,),
            ).fetchall()


    def create_project(
        self,
        name: str,
        description: str = "",
        python_code: str = "",
        variables: dict | None = None,
        settings: dict | None = None,
    ) -> int:
        ts = now()
        with self.tx() as conn:
            position = conn.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM projects"
            ).fetchone()[0]
            cur = conn.execute(
                """INSERT INTO projects(name,description,python_code,variables_json,settings,position,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    name,
                    description,
                    python_code,
                    json.dumps(variables or {}, ensure_ascii=False),
                    json.dumps(settings or {}, ensure_ascii=False),
                    position,
                    ts,
                    ts,
                ),
            )
            return int(cur.lastrowid)

    def unique_project_name(self, base: str) -> str:
        base = (base or "项目").strip() or "项目"
        names = {project.name for project in self.list_projects()}
        if base not in names:
            return base
        index = 2
        while f"{base} ({index})" in names:
            index += 1
        return f"{base} ({index})"

    def get_project(self, project_id: int) -> Project | None:
        with self.read() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        return Project.from_row(row) if row else None

    def get_project_by_name(self, name: str) -> Project | None:
        with self.read() as conn:
            row = conn.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()
        return Project.from_row(row) if row else None

    def list_projects(self) -> list[Project]:
        with self.read() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY position, id").fetchall()
        return [Project.from_row(row) for row in rows]

    def reorder_projects(self, project_ids: list[int]) -> None:
        with self.tx() as conn:
            for position, project_id in enumerate(project_ids, start=1):
                conn.execute(
                    "UPDATE projects SET position=? WHERE id=?", (position, project_id)
                )

    def update_project(self, project_id: int, **fields: Any) -> None:
        allowed = {
            "name",
            "description",
            "python_code",
            "variables_json",
            "settings",
            "position",
        }
        data = {key: value for key, value in fields.items() if key in allowed}
        if "variables" in fields:
            data["variables_json"] = json.dumps(fields["variables"] or {}, ensure_ascii=False)
        if "settings" in fields and not isinstance(fields["settings"], str):
            data["settings"] = json.dumps(fields["settings"] or {}, ensure_ascii=False)
        if not data:
            return
        data["updated_at"] = now()
        columns = ", ".join(f"{key}=?" for key in data)
        with self.tx() as conn:
            conn.execute(f"UPDATE projects SET {columns} WHERE id=?", (*data.values(), project_id))

    def delete_project(self, project_id: int) -> None:
        with self.tx() as conn:
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))

    def add_project_block(
        self, project_id: int, flow_id: int, alias: str | None = None, position: int | None = None
    ) -> int:
        flow = self.get_flow(flow_id)
        if flow is None:
            raise ValueError("流程不存在")
        with self.tx() as conn:
            existing = conn.execute(
                "SELECT id FROM project_blocks WHERE project_id=? AND flow_id=?",
                (project_id, flow_id),
            ).fetchone()
            if existing:
                return int(existing["id"])
            if position is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 AS pos FROM project_blocks WHERE project_id=?",
                    (project_id,),
                ).fetchone()
                position = int(row["pos"])
            if not alias:
                used = {
                    row["alias"]
                    for row in conn.execute(
                        "SELECT alias FROM project_blocks WHERE project_id=?", (project_id,)
                    ).fetchall()
                }
                alias = make_alias(flow.name, used)
            cur = conn.execute(
                "INSERT INTO project_blocks(project_id,flow_id,position,alias) VALUES (?,?,?,?)",
                (project_id, flow_id, position, alias),
            )
            return int(cur.lastrowid)

    def list_project_blocks(self, project_id: int) -> list[ProjectBlock]:
        with self.read() as conn:
            rows = conn.execute(
                """SELECT pb.*, f.name AS flow_name,
                          (SELECT COUNT(*) FROM nodes WHERE flow_id=pb.flow_id) AS node_count
                   FROM project_blocks pb JOIN flows f ON f.id=pb.flow_id
                   WHERE pb.project_id=? ORDER BY pb.position, pb.id""",
                (project_id,),
            ).fetchall()
        return [ProjectBlock.from_row(row) for row in rows]

    def update_project_block(self, block_id: int, **fields: Any) -> None:
        data = {key: value for key, value in fields.items() if key in {"alias", "position"}}
        if not data:
            return
        columns = ", ".join(f"{key}=?" for key in data)
        with self.tx() as conn:
            conn.execute(f"UPDATE project_blocks SET {columns} WHERE id=?", (*data.values(), block_id))

    def remove_project_block(self, block_id: int) -> None:
        with self.tx() as conn:
            conn.execute("DELETE FROM project_blocks WHERE id=?", (block_id,))

    def reorder_project_blocks(self, project_id: int, block_ids: list[int]) -> None:
        with self.tx() as conn:
            for position, block_id in enumerate(block_ids):
                conn.execute(
                    "UPDATE project_blocks SET position=? WHERE id=? AND project_id=?",
                    (position, block_id, project_id),
                )

    def create_project_run(self, project_id: int, project_name: str) -> int:
        with self.tx() as conn:
            cur = conn.execute(
                "INSERT INTO project_runs(project_id,project_name,started_at,status) VALUES (?,?,?,?)",
                (project_id, project_name, now(), "running"),
            )
            return int(cur.lastrowid)

    def finish_project_run(
        self,
        run_id: int,
        status: str,
        blocks: int,
        steps: int,
        error: str = "",
        variables: dict | None = None,
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                """UPDATE project_runs SET ended_at=?, status=?, blocks=?, steps=?, error=?, variables_json=?
                   WHERE id=?""",
                (
                    now(),
                    status,
                    blocks,
                    steps,
                    error,
                    json.dumps(variables or {}, ensure_ascii=False, default=str),
                    run_id,
                ),
            )

    def list_project_runs(self, project_id: int | None = None, limit: int = 100):
        with self.read() as conn:
            if project_id is None:
                return conn.execute(
                    "SELECT * FROM project_runs ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return conn.execute(
                "SELECT * FROM project_runs WHERE project_id=? ORDER BY id DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()

    def restore_flow_row(
        self,
        flow_id: int,
        name: str,
        description: str,
        settings: dict,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                """INSERT INTO flows(id,name,description,settings,created_at,updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, description=excluded.description,
                   settings=excluded.settings, updated_at=excluded.updated_at""",
                (
                    flow_id,
                    name,
                    description,
                    json.dumps(settings or {}, ensure_ascii=False),
                    created_at,
                    updated_at,
                ),
            )

    def replace_flow_content(self, flow_id: int, nodes: list[dict], edges: list[dict]) -> None:
        with self.tx() as conn:
            conn.execute("DELETE FROM edges WHERE flow_id=?", (flow_id,))
            conn.execute("DELETE FROM nodes WHERE flow_id=?", (flow_id,))
            for node in nodes:
                conn.execute(
                    """INSERT INTO nodes(id,flow_id,name,kind,image_path,img_w,img_h,click_x,click_y,
                       roi_x,roi_y,roi_w,roi_h,roi_weight,
                       target_x,target_y,target_w,target_h,
                       action_type,action_json,threshold,is_entry,is_end,
                       x,y,notes,embed,embed_kind,embed_dim,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        node["id"],
                        flow_id,
                        node["name"],
                        node["kind"],
                        node["image_path"],
                        node["img_w"],
                        node["img_h"],
                        node["click_x"],
                        node["click_y"],
                        node["roi_x"],
                        node["roi_y"],
                        node["roi_w"],
                        node["roi_h"],
                        node["roi_weight"],
                        node.get("target_x"),
                        node.get("target_y"),
                        node.get("target_w"),
                        node.get("target_h"),
                        node["action_type"],
                        node["action_json"],
                        node["threshold"],
                        node["is_entry"],
                        node.get("is_end", 0),
                        node["x"],
                        node["y"],
                        node["notes"],
                        node.get("embed"),
                        node.get("embed_kind"),
                        node.get("embed_dim"),
                        node["created_at"],
                    ),
                )
            for edge in edges:
                conn.execute(
                    """INSERT INTO edges(id,flow_id,src_id,dst_id,weight,label) VALUES (?,?,?,?,?,?)""",
                    (
                        edge["id"],
                        flow_id,
                        edge["src_id"],
                        edge["dst_id"],
                        edge["weight"],
                        edge["label"],
                    ),
                )

    def restore_project_row(
        self,
        project_id: int,
        name: str,
        description: str,
        python_code: str,
        variables: dict,
        settings: dict,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                """INSERT INTO projects(id,name,description,python_code,variables_json,settings,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, description=excluded.description,
                   python_code=excluded.python_code, variables_json=excluded.variables_json,
                   settings=excluded.settings, updated_at=excluded.updated_at""",
                (
                    project_id,
                    name,
                    description,
                    python_code,
                    json.dumps(variables or {}, ensure_ascii=False),
                    json.dumps(settings or {}, ensure_ascii=False),
                    created_at,
                    updated_at,
                ),
            )

    def replace_project_blocks(self, project_id: int, blocks: list[dict]) -> None:
        with self.tx() as conn:
            conn.execute("DELETE FROM project_blocks WHERE project_id=?", (project_id,))
            for block in blocks:
                conn.execute(
                    """INSERT INTO project_blocks(id,project_id,flow_id,position,alias)
                       VALUES (?,?,?,?,?)""",
                    (
                        block["id"],
                        project_id,
                        block["flow_id"],
                        block["position"],
                        block["alias"],
                    ),
                )


def make_alias(name: str, used: set[str]) -> str:
    candidate = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in (name or "")).strip("_")
    if not candidate:
        candidate = "block"
    if candidate[0].isdigit() or not candidate.isidentifier() or keyword.iskeyword(candidate):
        candidate = f"b_{candidate}"
    if not candidate.isidentifier():
        candidate = "block"
    base = candidate
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    return candidate
