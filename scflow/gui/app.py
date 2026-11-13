from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk

from .. import __version__, config
from ..db import Database
from .edit_view import EditView
from .history_view import HistoryView
from .run_view import RunView


class App(tk.Tk):
    def __init__(self, db: Database | None = None):
        super().__init__()
        self.db = db or Database()
        self.title(f"scrcpy-flow · 手机自动化流程编辑器 v{__version__}")
        self.geometry("1460x940")
        self.minsize(1180, 760)
        self._configure_style()

        header = ttk.Frame(self, padding=(14, 10, 14, 4))
        header.pack(fill="x")
        ttk.Label(header, text="scrcpy-flow", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="图片识别 · 流程图 · 状态偏好 · 保底机制 · scrcpy 控制",
            style="Subtitle.TLabel",
        ).pack(side="left", padx=(12, 0), pady=(6, 0))
        self.status_var = tk.StringVar(value=f"数据目录：{config.home()}")
        ttk.Label(header, textvariable=self.status_var, style="Subtitle.TLabel").pack(
            side="right", pady=(6, 0)
        )

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(4, 8))
        self.run_view = RunView(self.notebook, self, self.db)
        self.edit_view = EditView(self.notebook, self, self.db)
        self.history_view = HistoryView(self.notebook, self, self.db)
        self.notebook.add(self.run_view, text="  运行模式  ")
        self.notebook.add(self.edit_view, text="  编辑模式  ")
        self.notebook.add(self.history_view, text="  历史记录  ")

        footer = ttk.Frame(self, padding=(14, 2, 14, 8))
        footer.pack(fill="x")
        ttk.Label(
            footer,
            text="提示：先在编辑模式导入手机截图、设置检测重点与流程图，再到运行模式启动。",
            style="Subtitle.TLabel",
        ).pack(side="left")
        ttk.Label(footer, text=f"v{__version__}", style="Subtitle.TLabel").pack(side="right")

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        if os.name == "nt":
            base = ("Microsoft YaHei UI", 10)
        elif sys.platform == "darwin":
            base = ("PingFang SC", 10)
        else:
            base = ("Noto Sans CJK SC", 10)
        self.option_add("*Font", base)
        style.configure("TLabel", foreground="#24303f")
        style.configure("Title.TLabel", font=(base[0], 18, "bold"), foreground="#1c64d9")
        style.configure("Subtitle.TLabel", foreground="#65758b", font=(base[0], 9))
        style.configure("TLabelframe", background="#f7f9fc")
        style.configure("TLabelframe.Label", foreground="#334155", font=(base[0], 10, "bold"))
        style.configure("TFrame", background="#f2f5f9")
        style.configure("TNotebook", background="#e8eef6")
        style.configure("TNotebook.Tab", padding=(18, 9), font=(base[0], 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")])
        style.configure("Treeview", rowheight=25, background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=(base[0], 9, "bold"))
        style.configure("TButton", padding=(10, 5))
        self.configure(background="#f2f5f9")
        self.option_add("*Text.font", base)
        self.option_add("*Listbox.font", base)
        self.option_add("*TCombobox*Listbox.font", base)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def refresh_all(self) -> None:
        self.run_view.refresh_flows()
        self.edit_view.refresh_flows(keep_selection=True)
        self.history_view.refresh_flows()

    def refresh_history(self) -> None:
        self.history_view.refresh_sessions()

    def open_project(self, name: str) -> None:
        project = self.db.get_project_by_name(name)
        if project is None:
            return
        self.notebook.select(self.edit_view)
        self.edit_view.open_project(project.id)

    def open_flow(self, name: str) -> None:
        flow = self.db.get_flow_by_name(name)
        if flow is None:
            return
        self.notebook.select(self.edit_view)
        self.edit_view.open_flow(flow.id)

    def _on_tab_changed(self, _event=None):
        try:
            if self.notebook.select() != str(self.edit_view):
                self.edit_view.flush()
        except Exception:
            pass

    def _on_close(self):
        try:
            self.run_view.stop()
        except Exception:
            pass
        try:
            self.edit_view.flush()
        except Exception:
            pass
        self.destroy()
