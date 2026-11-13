from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image

from ..db import Database
from .widgets import ScreenshotViewer


class HistoryView(ttk.Frame):
    def __init__(self, master, app, db: Database):
        super().__init__(master, padding=10)
        self.app = app
        self.db = db
        self.flow_map: dict[str, int] = {}
        self.screenshot_map: dict[str, str] = {}
        self._build()
        self.refresh_flows()

    def _build(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(toolbar, text="流程").pack(side="left")
        self.flow_combo = ttk.Combobox(toolbar, width=28, state="readonly")
        self.flow_combo.pack(side="left", padx=4)
        self.flow_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_sessions())
        ttk.Button(toolbar, text="刷新", command=self.refresh_sessions).pack(side="left", padx=4)
        ttk.Label(toolbar, text="双击日志可查看截图").pack(side="right")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        flow_tab = ttk.Frame(self.notebook)
        project_tab = ttk.Frame(self.notebook)
        self.notebook.add(flow_tab, text="流程会话")
        self.notebook.add(project_tab, text="项目运行")
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self.refresh_project_runs())

        paned = ttk.PanedWindow(flow_tab, orient="vertical")
        paned.pack(fill="both", expand=True)

        sessions_frame = ttk.LabelFrame(paned, text="运行会话", padding=6)
        columns = ("id", "flow", "start", "status", "steps", "fallbacks", "note")
        self.sessions = ttk.Treeview(
            sessions_frame, columns=columns, show="headings", height=8
        )
        for key, title, width in (
            ("id", "会话", 60),
            ("flow", "流程", 180),
            ("start", "开始时间", 150),
            ("status", "状态", 90),
            ("steps", "步数", 60),
            ("fallbacks", "保底", 60),
            ("note", "备注", 320),
        ):
            self.sessions.heading(key, text=title)
            self.sessions.column(key, width=width, anchor="w")
        self.sessions.pack(fill="both", expand=True)
        self.sessions.bind("<<TreeviewSelect>>", lambda _event: self.load_logs())
        paned.add(sessions_frame, weight=1)

        logs_frame = ttk.LabelFrame(paned, text="步骤日志", padding=6)
        log_columns = ("idx", "time", "current", "matched", "sim", "action", "result")
        self.logs = ttk.Treeview(logs_frame, columns=log_columns, show="headings")
        for key, title, width in (
            ("idx", "步", 50),
            ("time", "时间", 150),
            ("current", "操作前状态", 150),
            ("matched", "匹配节点", 150),
            ("sim", "相似度", 80),
            ("action", "动作", 300),
            ("result", "结果", 160),
        ):
            self.logs.heading(key, text=title)
            self.logs.column(key, width=width, anchor="w")
        self.logs.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(logs_frame, orient="vertical", command=self.logs.yview)
        self.logs.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.logs.bind("<Double-1>", self._open_screenshot)
        paned.add(logs_frame, weight=2)

        project_toolbar = ttk.Frame(project_tab)
        project_toolbar.pack(fill="x", pady=(6, 4))
        ttk.Button(project_toolbar, text="刷新项目运行", command=self.refresh_project_runs).pack(
            side="left"
        )
        ttk.Label(project_toolbar, text="项目级会话（区块内部仍可在「流程会话」查看）").pack(
            side="right"
        )
        project_columns = ("id", "project", "start", "status", "blocks", "steps", "error")
        self.project_runs = ttk.Treeview(
            project_tab, columns=project_columns, show="headings", height=10
        )
        for key, title, width in (
            ("id", "会话", 60),
            ("project", "项目", 180),
            ("start", "开始时间", 150),
            ("status", "状态", 90),
            ("blocks", "区块", 60),
            ("steps", "总步数", 70),
            ("error", "错误", 320),
        ):
            self.project_runs.heading(key, text=title)
            self.project_runs.column(key, width=width, anchor="w")
        self.project_runs.pack(fill="both", expand=True)
        self.project_runs.bind("<<TreeviewSelect>>", lambda _event: self._show_project_vars())
        self.project_vars = tk.Text(project_tab, height=5, wrap="word")
        self.project_vars.pack(fill="x", pady=(4, 0))
        self.project_vars.configure(state="disabled")

    def refresh_flows(self):
        flows = self.db.list_flows()
        self.flow_map = {flow.name: flow.id for flow in flows}
        self.flow_combo["values"] = ["全部流程"] + [flow.name for flow in flows]
        if not self.flow_combo.get():
            self.flow_combo.set("全部流程")
        self.refresh_sessions()

    def refresh_sessions(self):
        if not hasattr(self, "sessions"):
            return
        self.refresh_project_runs()
        self.sessions.delete(*self.sessions.get_children())
        name = self.flow_combo.get()
        flow_id = self.flow_map.get(name) if name and name != "全部流程" else None
        flow_names = {flow.id: flow.name for flow in self.db.list_flows()}
        for session in self.db.list_sessions(flow_id, limit=200):
            self.sessions.insert(
                "",
                "end",
                iid=str(session.id),
                values=(
                    session.id,
                    flow_names.get(session.flow_id, ""),
                    session.started_at,
                    session.status,
                    session.steps,
                    session.fallbacks,
                    session.note,
                ),
            )

    def load_logs(self):
        selection = self.sessions.selection()
        if not selection:
            return
        try:
            session_id = int(selection[0])
        except ValueError:
            return
        self.logs.delete(*self.logs.get_children())
        self.screenshot_map = {}
        for index, row in enumerate(self.db.list_logs(session_id)):
            iid = f"log-{session_id}-{index}"
            self.logs.insert(
                "",
                "end",
                iid=iid,
                values=(
                    row["idx"],
                    row["ts"],
                    row["current_name"] or "",
                    row["matched_name"] or "",
                    "-" if row["similarity"] is None else f"{row['similarity']:.3f}",
                    row["action"],
                    row["result"],
                ),
            )
            if row["screenshot_path"]:
                self.screenshot_map[iid] = row["screenshot_path"]

    def _open_screenshot(self, _event=None):
        selection = self.logs.selection()
        if not selection:
            return
        path = self.screenshot_map.get(selection[0])
        if not path or not Path(path).exists():
            return
        with Image.open(path) as image:
            viewer = ScreenshotViewer(self, image.copy(), title=f"截图 · {Path(path).name}")
        viewer.grab_set()


    def refresh_project_runs(self):
        if not hasattr(self, "project_runs"):
            return
        self.project_runs.delete(*self.project_runs.get_children())
        for row in self.db.list_project_runs(limit=200):
            self.project_runs.insert(
                "",
                "end",
                iid=f"pr-{row['id']}",
                values=(
                    row["id"],
                    row["project_name"],
                    row["started_at"],
                    row["status"],
                    row["blocks"],
                    row["steps"],
                    row["error"],
                ),
            )

    def _show_project_vars(self):
        selection = self.project_runs.selection()
        if not selection:
            return
        run_id = int(selection[0].split("-", 1)[1])
        rows = [
            row
            for row in self.db.list_project_runs(limit=500)
            if int(row["id"]) == run_id
        ]
        text = ""
        if rows:
            try:
                variables = json.loads(rows[0]["variables_json"] or "{}")
            except (TypeError, ValueError):
                variables = {}
            text = json.dumps(variables, ensure_ascii=False, indent=2)
        self.project_vars.configure(state="normal")
        self.project_vars.delete("1.0", tk.END)
        self.project_vars.insert("1.0", text)
        self.project_vars.configure(state="disabled")
