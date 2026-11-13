from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4

from PIL import Image

from .. import config, vision
from ..db import Database
from ..device import AdbDevice, list_adb_devices
from ..embeddings import embed_to_blob, get_embedder
from ..engine import merged_settings
from ..flowio import export_flow, import_flow
from ..models import Flow, Node, Project, ProjectBlock
from ..undo import UndoStack, restore_flow, restore_project, snapshot_flow, snapshot_project
from .dialogs import ConnectDialog, EdgeDialog, FlowDialog, ImportDialog, VariableDialog
from .widgets import FallbackStepDialog, ImageCanvas, ScrollableFrame, photo_image

NODE_W = 214
NODE_H = 108

KIND_LABELS = {"image": "图片步骤", "action": "动作步骤", "end": "结束步骤"}
KIND_VALUES = {value: key for key, value in KIND_LABELS.items()}
ACTION_LABELS = {
    "tap": "点击",
    "swipe": "滑动",
    "key": "按键",
    "text": "输入文本",
    "launch": "启动应用",
    "wait": "等待",
    "none": "无动作",
}
ACTION_VALUES = {value: key for key, value in ACTION_LABELS.items()}
SELECTION_LABELS = {
    "best": "最相似优先",
    "weighted_random": "加权随机（每天换一家）",
}
SELECTION_VALUES = {value: key for key, value in SELECTION_LABELS.items()}
SCOPE_LABELS = {
    "next": "只匹配下一步（推荐）",
    "reachable": "只匹配当前可达节点",
    "global": "全局匹配（允许跳步/弹窗）",
}
SCOPE_VALUES = {value: key for key, value in SCOPE_LABELS.items()}
FALLBACK_LABELS = {
    "key": "按键",
    "wait": "等待",
    "tap": "点击",
    "swipe": "滑动",
    "text": "输入",
}

IMAGE_TYPES = [
    ("图片", "*.png *.jpg *.jpeg *.webp *.bmp"),
    ("所有文件", "*.*"),
]


def parse_literal(value: str):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def format_literal(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _tag_id(tags, prefix: str) -> int | None:
    for tag in tags:
        if tag.startswith(prefix):
            try:
                return int(tag.split(":", 1)[1])
            except ValueError:
                return None
    return None


class EditView(ttk.Frame):
    def __init__(self, master, app, db: Database):
        super().__init__(master, padding=8)
        self.app = app
        self.db = db
        self.flows: list[Flow] = []
        self.flow_id: int | None = None
        self.nodes: list[Node] = []
        self.edges = []
        self.nodes_by_id: dict[int, Node] = {}
        self.edges_by_id = {}
        self.selected_node_id: int | None = None
        self.selected_edge_id: int | None = None
        self.current_click: tuple[float, float] | None = None
        self.current_roi: tuple[float, float, float, float] | None = None
        self.current_target: tuple[float, float, float, float] | None = None
        self._last_action_label = ACTION_LABELS["tap"]
        self._syncing_no_click = False
        self._loading = True
        self._dirty = False
        self._save_job = None
        self._refreshing_flows = False
        self._drag = None
        self._thumb_cache: dict[tuple, object] = {}
        self._thumb_refs: dict[int, object] = {}
        self.action_vars: dict[str, tk.Variable] = {}
        self.fallback_steps: list[dict] = []
        self.flow_undo = UndoStack()
        self.project_undo = UndoStack()
        self._node_edit_open = False
        self.projects: list[Project] = []
        self.project_id: int | None = None
        self.project_blocks: list[ProjectBlock] = []
        self.selected_block_id: int | None = None
        self.project_vars: dict = {}
        self._project_loading = False
        self._drag_block = None
        self._project_dirty = False

        self._create_vars()
        self._build()
        self._loading = False
        self.refresh_flows()
        self.refresh_projects()
        self.after(300, self.refresh_devices)

    def _create_vars(self):
        self.name_var = tk.StringVar()
        self.kind_var = tk.StringVar(value=KIND_LABELS["image"])
        self.threshold_var = tk.StringVar(value="0.88")
        self.roi_weight_var = tk.StringVar(value="0.72")
        self.entry_var = tk.BooleanVar(value=False)
        self.action_type_var = tk.StringVar(value=ACTION_LABELS["tap"])
        self.pre_wait_var = tk.StringVar(value="0")
        self.post_wait_var = tk.StringVar(value="0.6")
        self.image_mode_var = tk.StringVar(value="view")
        self.click_info_var = tk.StringVar(value="点击点：未设置")
        self.roi_info_var = tk.StringVar(value="检测重点：未设置")
        self.target_info_var = tk.StringVar(value="点击对象：未设置")
        self.zoom_label_var = tk.StringVar(value="100%")
        self.no_click_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪")
        for var in (
            self.name_var,
            self.threshold_var,
            self.roi_weight_var,
            self.entry_var,
            self.action_type_var,
            self.pre_wait_var,
            self.post_wait_var,
        ):
            var.trace_add("write", self._mark_dirty)
        self.action_type_var.trace_add("write", self._on_action_type_changed)

    def _build(self):
        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill="both", expand=True)
        self.project_tab = ttk.Frame(self.main_notebook, padding=4)
        self.flow_tab = ttk.Frame(self.main_notebook, padding=4)
        self.main_notebook.add(self.project_tab, text="  区块编排（项目）  ")
        self.main_notebook.add(self.flow_tab, text="  流程画布（区块内部）  ")
        self._build_flow_tab(self.flow_tab)
        self._build_project_tab(self.project_tab)
        self.main_notebook.bind("<<NotebookTabChanged>>", lambda _event: self.flush())

    def _build_flow_tab(self, parent):
        paned = ttk.PanedWindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, width=250)
        paned.add(left, weight=0)
        self._build_flow_panel(left)
        self._build_device_panel(left)

        center = ttk.Frame(paned)
        paned.add(center, weight=3)
        self._build_graph(center)

        right = ttk.Frame(paned, width=460)
        paned.add(right, weight=0)
        self.right_notebook = ttk.Notebook(right)
        self.right_notebook.pack(fill="both", expand=True)
        self._build_node_tab()
        self._build_settings_tab()
        self._build_info_tab()

    def _build_flow_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="流程库", padding=6)
        frame.pack(fill="both", expand=True, padx=(0, 6))
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True)
        self.flow_list = tk.Listbox(
            list_frame,
            exportselection=False,
            activestyle="none",
            height=14,
            borderwidth=1,
            relief="solid",
        )
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.flow_list.yview)
        self.flow_list.configure(yscrollcommand=scroll.set)
        self.flow_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.flow_list.bind("<<ListboxSelect>>", self._on_flow_select)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(6, 0))
        for index, (text, command) in enumerate(
            (
                ("新建", self._new_flow),
                ("编辑", self._rename_flow),
                ("删除", self._delete_flow),
                ("导入", self._import_flow),
                ("导出", self._export_flow),
                ("刷新", lambda: self.refresh_flows(keep_selection=True)),
            )
        ):
            ttk.Button(buttons, text=text, command=command, width=6).grid(
                row=index // 3, column=index % 3, padx=2, pady=2, sticky="ew"
            )
        for column in range(3):
            buttons.columnconfigure(column, weight=1)

    def _build_device_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="设备（取图 / 模拟点击）", padding=6)
        frame.pack(fill="x", padx=(0, 6), pady=(8, 0))
        self.device_combo = ttk.Combobox(frame, width=26, state="readonly")
        self.device_combo.pack(fill="x")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="刷新设备", command=self.refresh_devices).pack(
            side="left", expand=True, fill="x", padx=(0, 2)
        )
        ttk.Button(row, text="手机截屏建节点", command=self._screenshot_node).pack(
            side="left", expand=True, fill="x", padx=(2, 0)
        )
        ttk.Label(frame, textvariable=self.status_var, wraplength=230, foreground="#526173").pack(
            fill="x", pady=(8, 0)
        )

    def _build_graph(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 4))
        for text, command in (
            ("添加节点", self._add_node),
            ("添加结束节点", lambda: self._add_node(kind="end")),
            ("连接 / 下一步", self._connect),
            ("取消连接/下一步", self._cancel_connection),
            ("删除选中", self._delete_selected),
        ):
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=(0, 4))
        toolbar2 = ttk.Frame(parent)
        toolbar2.pack(fill="x", pady=(0, 6))
        for text, command in (
            ("自动布局", self._auto_layout),
            ("撤回", self._undo_flow_action),
            ("刷新", self._reload_graph),
        ):
            ttk.Button(toolbar2, text=text, command=command).pack(side="left", padx=(0, 4))
        ttk.Label(
            toolbar2,
            text="拖动节点排版 · 单击选中 · 双击连线编辑",
            foreground="#64748b",
        ).pack(side="right")

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            background="#e7edf5",
            highlightthickness=0,
            scrollregion=(0, 0, 3200, 3200),
        )
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._graph_press)
        self.canvas.bind("<B1-Motion>", self._graph_drag)
        self.canvas.bind("<ButtonRelease-1>", self._graph_release)
        self.canvas.bind("<Double-Button-1>", self._graph_double)
        self.canvas.bind("<Delete>", lambda _event: self._delete_selected())
        self.canvas.bind("<Control-s>", lambda _event: self._save_node())

    def _build_node_tab(self):
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="节点属性")
        scroll = ScrollableFrame(tab)
        scroll.pack(fill="both", expand=True)
        body = ttk.Frame(scroll.inner, padding=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(body, text="名称").grid(row=row, column=0, sticky="w", pady=3)
        self.name_entry = ttk.Entry(body, textvariable=self.name_var)
        self.name_entry.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        ttk.Label(body, text="节点类型").grid(row=row, column=0, sticky="w", pady=3)
        self.kind_combo = ttk.Combobox(
            body,
            textvariable=self.kind_var,
            values=list(KIND_LABELS.values()),
            state="readonly",
            width=18,
        )
        self.kind_combo.grid(row=row, column=1, sticky="w", pady=3)
        row += 1
        ttk.Label(body, text="匹配阈值").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(
            body,
            from_=0.3,
            to=0.999,
            increment=0.01,
            textvariable=self.threshold_var,
            width=10,
            format="%.2f",
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1
        self.entry_check = ttk.Checkbutton(
            body, text="设为流程入口（初始状态）", variable=self.entry_var
        )
        self.entry_check.grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        row += 1

        ttk.Label(body, text="状态图片（Ctrl+滚轮缩放，中键拖动）").grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        image_holder = ttk.Frame(body)
        image_holder.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        self.image_canvas = ImageCanvas(
            image_holder,
            width=340,
            height=500,
            mode="view",
            on_click=self._on_image_click,
            on_roi=self._on_image_roi,
            on_target=self._on_image_target,
            on_zoom=self._on_zoom_changed,
        )
        image_hbar = ttk.Scrollbar(
            image_holder, orient="horizontal", command=self.image_canvas.xview
        )
        image_vbar = ttk.Scrollbar(
            image_holder, orient="vertical", command=self.image_canvas.yview
        )
        self.image_canvas.configure(
            xscrollcommand=image_hbar.set, yscrollcommand=image_vbar.set
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        image_vbar.grid(row=0, column=1, sticky="ns")
        image_hbar.grid(row=1, column=0, sticky="ew")
        image_holder.rowconfigure(0, weight=1)
        image_holder.columnconfigure(0, weight=1)
        row += 1
        zoom_row = ttk.Frame(body)
        zoom_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Button(zoom_row, text="放大", width=6, command=self.image_canvas.zoom_in).pack(
            side="left", padx=(0, 3)
        )
        ttk.Button(zoom_row, text="缩小", width=6, command=self.image_canvas.zoom_out).pack(
            side="left", padx=3
        )
        ttk.Button(
            zoom_row, text="适应窗口", width=9, command=self.image_canvas.zoom_fit
        ).pack(side="left", padx=3)
        ttk.Label(zoom_row, textvariable=self.zoom_label_var, foreground="#64748b").pack(
            side="right"
        )
        row += 1
        modes = ttk.Frame(body)
        modes.grid(row=row, column=0, columnspan=2, sticky="w")
        for text, value in (
            ("仅查看", "view"),
            ("取点击点", "click"),
            ("框选检测重点", "roi"),
            ("框选点击对象", "target"),
        ):
            ttk.Radiobutton(
                modes,
                text=text,
                value=value,
                variable=self.image_mode_var,
                command=self._on_image_mode,
            ).pack(side="left", padx=(0, 8))
        row += 1
        ttk.Label(body, textvariable=self.click_info_var, foreground="#b23c3c").grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Label(body, textvariable=self.roi_info_var, foreground="#1c64d9").grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Label(body, textvariable=self.target_info_var, foreground="#e67700").grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        action_row = ttk.Frame(body)
        action_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        self.no_click_check = ttk.Checkbutton(
            action_row,
            text="不模拟点击",
            variable=self.no_click_var,
            command=self._on_no_click_toggle,
            style="Toolbutton",
        )
        self.no_click_check.pack(side="left", padx=(0, 4))
        ttk.Button(action_row, text="框选重点点击对象", command=self._start_target_mode).pack(
            side="left", padx=4
        )
        self.simulate_button = ttk.Button(
            action_row, text="模拟点击", command=self._simulate_click
        )
        self.simulate_button.pack(side="right")
        row += 1
        image_buttons = ttk.Frame(body)
        image_buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(image_buttons, text="更换图片", command=self._replace_image).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(image_buttons, text="清除检测重点", command=self._clear_roi).pack(
            side="left", padx=4
        )
        ttk.Button(image_buttons, text="清除点击对象", command=self._clear_target).pack(
            side="left", padx=4
        )
        row += 1

        ttk.Label(body, text="检测重点权重").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(
            body,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.roi_weight_var,
            width=10,
            format="%.2f",
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1
        ttk.Label(
            body,
            text="权重越高，判断时越只看选区；0.72 表示选区占主导。",
            foreground="#64748b",
            wraplength=320,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Separator(body).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(body, text="动作类型").grid(row=row, column=0, sticky="w", pady=3)
        self.action_combo = ttk.Combobox(
            body,
            textvariable=self.action_type_var,
            values=list(ACTION_LABELS.values()),
            state="readonly",
            width=18,
        )
        self.action_combo.grid(row=row, column=1, sticky="w", pady=3)
        row += 1
        self.action_dynamic = ttk.Frame(body)
        self.action_dynamic.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
        row += 1
        wait_frame = ttk.Frame(body)
        wait_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(wait_frame, text="动作前等待").pack(side="left")
        ttk.Entry(wait_frame, textvariable=self.pre_wait_var, width=6).pack(
            side="left", padx=(4, 14)
        )
        ttk.Label(wait_frame, text="动作后等待").pack(side="left")
        ttk.Entry(wait_frame, textvariable=self.post_wait_var, width=6).pack(side="left", padx=4)
        row += 1

        ttk.Label(body, text="备注").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.notes_text = tk.Text(body, height=4, wrap="word")
        self.notes_text.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
        self.notes_text.bind("<<Modified>>", self._on_notes_modified)
        row += 1

        edge_frame = ttk.LabelFrame(body, text="下一步 / 偏好（可添加多个）", padding=6)
        edge_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        self.outgoing = ttk.Treeview(
            edge_frame,
            columns=("dst", "weight", "label"),
            show="headings",
            height=5,
        )
        for key, title, width in (
            ("dst", "下一步节点", 150),
            ("weight", "偏好权重", 80),
            ("label", "说明", 100),
        ):
            self.outgoing.heading(key, text=title)
            self.outgoing.column(key, width=width, anchor="w")
        self.outgoing.pack(fill="x")
        self.outgoing.bind("<Double-1>", lambda _event: self._edit_outgoing_edge())
        edge_buttons = ttk.Frame(edge_frame)
        edge_buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(edge_buttons, text="添加下一步", command=self._connect).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(edge_buttons, text="编辑连线", command=self._edit_outgoing_edge).pack(
            side="left", padx=4
        )
        ttk.Button(edge_buttons, text="删除连线", command=self._delete_outgoing_edge).pack(
            side="left", padx=4
        )

        action_buttons = ttk.Frame(body)
        action_buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(action_buttons, text="保存节点", command=self._save_node).pack(
            side="right", padx=(4, 0)
        )
        ttk.Button(action_buttons, text="删除节点", command=self._delete_selected).pack(
            side="right", padx=4
        )
        self._clear_inspector()

    def _build_settings_tab(self):
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="保底 / 运行参数")
        scroll = ScrollableFrame(tab)
        scroll.pack(fill="both", expand=True)
        body = ttk.Frame(scroll.inner, padding=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        self.settings_vars = {
            "beta": tk.StringVar(value="0.12"),
            "stay_penalty": tk.StringVar(value="0.06"),
            "match_scope": tk.StringVar(value=SCOPE_LABELS["next"]),
            "save_screenshots": tk.BooleanVar(value=True),
            "selection_mode": tk.StringVar(value=SELECTION_LABELS["best"]),
            "max_retries": tk.StringVar(value="2"),
            "threshold_step": tk.StringVar(value="0.04"),
            "min_threshold": tk.StringVar(value="0.55"),
            "max_fallbacks": tk.StringVar(value="3"),
            "retry_wait": tk.StringVar(value="0.6"),
        }

        row = 0
        ttk.Label(body, text="路径偏好权重 β").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(body, textvariable=self.settings_vars["beta"], width=10).grid(
            row=row, column=1, sticky="w", pady=3
        )
        row += 1
        ttk.Label(body, text="重复停留惩罚").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(body, textvariable=self.settings_vars["stay_penalty"], width=10).grid(
            row=row, column=1, sticky="w", pady=3
        )
        row += 1
        ttk.Label(body, text="匹配范围").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(
            body,
            textvariable=self.settings_vars["match_scope"],
            values=list(SCOPE_LABELS.values()),
            state="readonly",
            width=28,
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1
        ttk.Label(
            body,
            text="「只匹配下一步」时，未到步骤的节点及其检测重点不会参与计算。",
            foreground="#64748b",
            wraplength=320,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(
            body, text="保存每一步截图", variable=self.settings_vars["save_screenshots"]
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        row += 1
        ttk.Label(body, text="候选选择方式").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(
            body,
            textvariable=self.settings_vars["selection_mode"],
            values=list(SELECTION_LABELS.values()),
            state="readonly",
            width=26,
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1
        ttk.Label(
            body,
            text="加权随机会在多个达标候选间按偏好权重选择，适合每天换一家。",
            foreground="#64748b",
            wraplength=320,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        fallback_frame = ttk.LabelFrame(body, text="保底机制", padding=8)
        fallback_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        inner = ttk.Frame(fallback_frame)
        inner.pack(fill="x")
        labels = [
            ("max_retries", "最大重试"),
            ("threshold_step", "每次降阈值"),
            ("min_threshold", "最低阈值"),
            ("max_fallbacks", "最多保底次数"),
            ("retry_wait", "重试等待秒"),
        ]
        for index, (key, label) in enumerate(labels):
            ttk.Label(inner, text=label).grid(row=index, column=0, sticky="w", pady=2)
            ttk.Entry(inner, textvariable=self.settings_vars[key], width=10).grid(
                row=index, column=1, sticky="w", pady=2, padx=6
            )
        ttk.Label(
            fallback_frame,
            text="连续未匹配 -> 降低阈值重试 -> 执行下面的保底步骤。",
            foreground="#64748b",
        ).pack(anchor="w", pady=(4, 2))
        columns = ("index", "type", "detail")
        self.fallback_tree = ttk.Treeview(
            fallback_frame, columns=columns, show="headings", height=6
        )
        for key, title, width in (
            ("index", "#", 40),
            ("type", "类型", 80),
            ("detail", "参数", 240),
        ):
            self.fallback_tree.heading(key, text=title)
            self.fallback_tree.column(key, width=width, anchor="w")
        self.fallback_tree.pack(fill="x", pady=4)
        self.fallback_tree.bind("<Double-1>", lambda _event: self._edit_fallback_step())
        step_buttons = ttk.Frame(fallback_frame)
        step_buttons.pack(fill="x")
        ttk.Button(step_buttons, text="添加", command=self._add_fallback_step).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(step_buttons, text="编辑", command=self._edit_fallback_step).pack(
            side="left", padx=4
        )
        ttk.Button(step_buttons, text="删除", command=self._remove_fallback_step).pack(
            side="left", padx=4
        )
        ttk.Button(step_buttons, text="上移", command=lambda: self._move_fallback(-1)).pack(
            side="left", padx=4
        )
        ttk.Button(step_buttons, text="下移", command=lambda: self._move_fallback(1)).pack(
            side="left", padx=4
        )

        ttk.Button(body, text="保存运行设置", command=self._save_settings).grid(
            row=row, column=0, columnspan=2, sticky="e", pady=8
        )

    def _build_info_tab(self):
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="流程信息")
        body = ttk.Frame(tab, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="流程说明").pack(anchor="w")
        self.flow_desc = tk.Text(body, height=8, wrap="word")
        self.flow_desc.pack(fill="x", pady=(4, 8))
        ttk.Button(body, text="保存说明", command=self._save_flow_info).pack(anchor="e")
        ttk.Separator(body).pack(fill="x", pady=12)
        self.flow_stats = ttk.Label(body, text="", justify="left")
        self.flow_stats.pack(anchor="w")
        ttk.Separator(body).pack(fill="x", pady=12)
        ttk.Button(body, text="导出当前流程", command=self._export_flow).pack(
            fill="x", pady=3
        )
        ttk.Button(body, text="导入并衔接流程", command=self._import_flow).pack(
            fill="x", pady=3
        )
        ttk.Button(body, text="打开数据目录", command=self._open_data_dir).pack(
            fill="x", pady=3
        )
        ttk.Separator(body).pack(fill="x", pady=12)
        ttk.Button(
            body, text="删除当前流程", command=self._delete_flow
        ).pack(fill="x", pady=3)
        ttk.Label(
            body,
            text="提示：复制整个项目文件夹（含 data/ 与 tools/）即可迁移到另一台电脑。",
            foreground="#64748b",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

    def refresh_flows(self, keep_selection: bool = True, select_id: int | None = None):
        if self._refreshing_flows:
            return
        self._refreshing_flows = True
        self.flush()
        self.flows = self.db.list_flows()
        self.flow_list.delete(0, tk.END)
        for flow in self.flows:
            self.flow_list.insert(tk.END, flow.name)
        index = None
        if select_id is not None:
            index = next((i for i, f in enumerate(self.flows) if f.id == select_id), None)
        elif keep_selection and self.flow_id is not None:
            index = next((i for i, f in enumerate(self.flows) if f.id == self.flow_id), None)
        if index is None and self.flows:
            index = 0
        if index is not None:
            self.flow_list.selection_set(index)
            self.flow_list.activate(index)
            self.flow_list.see(index)
        self._refreshing_flows = False
        if index is not None:
            self._load_flow(self.flows[index].id)
        else:
            self._clear_all()
        self._refresh_project_after_flow_change()

    def _on_flow_select(self, _event=None):
        if self._refreshing_flows:
            return
        selection = self.flow_list.curselection()
        if not selection:
            return
        flow = self.flows[selection[0]]
        if flow.id == self.flow_id:
            return
        self.flush()
        self._load_flow(flow.id)

    def _load_flow(self, flow_id: int):
        flow = self.db.get_flow(flow_id)
        if flow is None:
            return
        self.flow_id = flow_id
        self.nodes = self.db.list_nodes(flow_id)
        self.edges = self.db.list_edges(flow_id)
        self.nodes_by_id = {node.id: node for node in self.nodes}
        self.edges_by_id = {edge.id: edge for edge in self.edges}
        self.selected_node_id = None
        self.selected_edge_id = None
        self.current_click = None
        self.current_roi = None
        self._node_edit_open = False
        self.flow_undo.clear()
        if (
            len(self.nodes) > 1
            and len({(node.x, node.y) for node in self.nodes}) == 1
        ):
            for node_id, (x, y) in self._compute_layout().items():
                self.db.update_node(node_id, x=x, y=y)
                node = self.nodes_by_id.get(node_id)
                if node:
                    node.x, node.y = x, y
        self._draw_graph()
        self._load_settings(flow)
        self._load_flow_info(flow)
        self._clear_inspector()
        if self.nodes:
            self._select_node(self.nodes[0].id)
        self._status(f"已加载流程：{flow.name} · {len(self.nodes)} 个节点 · {len(self.edges)} 条连线")

    def _clear_all(self):
        self.flow_id = None
        self.nodes = []
        self.edges = []
        self.nodes_by_id = {}
        self.edges_by_id = {}
        self.selected_node_id = None
        self.selected_edge_id = None
        self.flow_undo.clear()
        self._node_edit_open = False
        self.canvas.delete("all")
        self.canvas.create_text(
            360,
            200,
            text="还没有流程\n点击左侧「新建」或在运行环境导入一个流程",
            fill="#64748b",
            justify="center",
            font=("TkDefaultFont", 14),
        )
        self._clear_inspector()
        self._status("尚未选择流程")

    def _clear_inspector(self):
        self._loading = True
        self.name_var.set("")
        self.threshold_var.set("0.88")
        self.roi_weight_var.set("0.72")
        self.entry_var.set(False)
        self.action_type_var.set(ACTION_LABELS["tap"])
        self.pre_wait_var.set("0")
        self.post_wait_var.set("0.6")
        self.click_info_var.set("点击点：未设置")
        self.roi_info_var.set("检测重点：未设置")
        self.target_info_var.set("点击对象：未设置")
        self.no_click_var.set(False)
        self._last_action_label = ACTION_LABELS["tap"]
        self.image_mode_var.set("view")
        self.image_canvas.set_mode("view")
        self.image_canvas.set_source(None)
        self.image_canvas.zoom_fit()
        self.image_canvas.set_markers(None, None, None)
        self.notes_text.delete("1.0", tk.END)
        self.notes_text.edit_modified(False)
        self.outgoing.delete(*self.outgoing.get_children())
        self._rebuild_action_fields(None)
        self._loading = False
        self._dirty = False

    def _status(self, text: str):
        self.status_var.set(text)

    def flush(self):
        if self._save_job is not None:
            try:
                self.after_cancel(self._save_job)
            except Exception:
                pass
            self._save_job = None
        if self._dirty and self.selected_node_id is not None:
            self._save_node(silent=True, redraw=False)
        self._flush_project()

    def _mark_dirty(self, *_args):
        if self._loading:
            return
        if not self._node_edit_open:
            self._push_flow_undo("编辑节点")
            self._node_edit_open = True
        self._dirty = True
        if self._save_job is not None:
            try:
                self.after_cancel(self._save_job)
            except Exception:
                pass
        self._save_job = self.after(700, self._auto_save)

    def _auto_save(self):
        self._save_job = None
        if self._dirty and self.selected_node_id is not None:
            self._save_node(silent=True)

    def _on_notes_modified(self, _event=None):
        if self._loading:
            return
        self.notes_text.edit_modified(False)
        self._mark_dirty()

    def _on_action_type_changed(self, *_args):
        if self._loading:
            return
        no_click = self._action_type_key() == "none"
        if not self._syncing_no_click:
            self.no_click_var.set(no_click)
        self.simulate_button.configure(state="disabled" if no_click else "normal")
        node = self.nodes_by_id.get(self.selected_node_id) if self.selected_node_id else None
        self._rebuild_action_fields(node)
        self._mark_dirty()

    def _on_no_click_toggle(self):
        if self._loading:
            return
        self._syncing_no_click = True
        try:
            if self.no_click_var.get():
                if self._action_type_key() != "none":
                    self._last_action_label = self.action_type_var.get()
                self.action_type_var.set(ACTION_LABELS["none"])
            else:
                self.action_type_var.set(self._last_action_label or ACTION_LABELS["tap"])
        finally:
            self._syncing_no_click = False
        self.simulate_button.configure(state="disabled" if self.no_click_var.get() else "normal")
        self._mark_dirty()

    def _on_image_mode(self):
        mode = self.image_mode_var.get()
        self.image_canvas.set_mode(mode)
        if mode == "target":
            self._start_target_mode(show_warning=False)
            if not vision.cv2_available():
                self._status("未安装 OpenCV：点击对象运行时会回退到相对位置")

    def _start_target_mode(self, show_warning: bool = True):
        self.image_mode_var.set("target")
        self.image_canvas.set_mode("target")
        if show_warning and not vision.cv2_available():
            self._status("未安装 OpenCV：运行时会按相对位置点击，请安装 requirements.txt")
            messagebox.showwarning(
                "缺少 OpenCV",
                "识别点击对象需要 OpenCV。\n请安装：pip install -r requirements.txt\n"
                "未安装时运行会退化为按相对位置点击。",
                parent=self,
            )

    def _on_zoom_changed(self, zoom: float):
        self.zoom_label_var.set(f"{int(round(zoom * 100))}%")

    def _on_image_click(self, x: float, y: float):
        if not self._node_edit_open:
            self._push_flow_undo("设置点击点")
            self._node_edit_open = True
        self.current_click = (x, y)
        self.click_info_var.set(f"点击点：x={x:.4f}  y={y:.4f}")
        self.image_canvas.set_markers(self.current_click, self.current_roi, self.current_target)
        self._save_node(silent=True)

    def _on_image_roi(self, roi):
        if not self._node_edit_open:
            self._push_flow_undo("设置检测重点")
            self._node_edit_open = True
        self.current_roi = roi
        self.roi_info_var.set(
            f"检测重点：x={roi[0]:.3f} y={roi[1]:.3f} w={roi[2]:.3f} h={roi[3]:.3f}"
        )
        self.image_canvas.set_markers(self.current_click, self.current_roi, self.current_target)
        self._save_node(silent=True)

    def _clear_roi(self):
        if not self._node_edit_open:
            self._push_flow_undo("清除检测重点")
            self._node_edit_open = True
        self.current_roi = None
        self.roi_info_var.set("检测重点：未设置")
        self.image_canvas.set_markers(self.current_click, None, self.current_target)
        self._save_node(silent=True)

    def _on_image_target(self, region):
        if not self._node_edit_open:
            self._push_flow_undo("框选点击对象")
            self._node_edit_open = True
        self.current_target = region
        self.target_info_var.set(
            f"点击对象：x={region[0]:.3f} y={region[1]:.3f} "
            f"w={region[2]:.3f} h={region[3]:.3f}"
        )
        self.image_canvas.set_markers(self.current_click, self.current_roi, self.current_target)
        self._save_node(silent=True)
        self._status("已设置点击对象：运行时会识别该区域并点击中心")
        if not vision.cv2_available():
            self._status("已设置点击对象，但缺少 OpenCV，运行时会按相对位置点击")

    def _clear_target(self):
        if not self._node_edit_open:
            self._push_flow_undo("清除点击对象")
            self._node_edit_open = True
        self.current_target = None
        self.target_info_var.set("点击对象：未设置")
        self.image_canvas.set_markers(self.current_click, self.current_roi, None)
        self._save_node(silent=True)

    def _rebuild_action_fields(self, node: Node | None):
        for child in self.action_dynamic.winfo_children():
            child.destroy()
        self.action_vars = {}
        if node is None:
            self.action_dynamic.configure(height=1)
            return
        action = node.action or {}
        action_type = ACTION_VALUES.get(self.action_type_var.get(), node.action_type)
        if action_type == "tap":
            ttk.Label(
                self.action_dynamic,
                text="运行时会点击图片上设置的点击点（未设置则点屏幕中心）。",
                foreground="#64748b",
                wraplength=320,
            ).pack(anchor="w")
        elif action_type == "swipe":
            frame = ttk.Frame(self.action_dynamic)
            frame.pack(fill="x")
            for index, (name, label, default) in enumerate(
                (
                    ("x1", "起点x", 0.5),
                    ("y1", "起点y", 0.8),
                    ("x2", "终点x", 0.5),
                    ("y2", "终点y", 0.2),
                    ("duration", "时长ms", 300),
                )
            ):
                ttk.Label(frame, text=label).grid(
                    row=index // 2, column=(index % 2) * 2, sticky="w", pady=2
                )
                var = tk.StringVar(value=str(action.get(name, default)))
                var.trace_add("write", self._mark_dirty)
                ttk.Entry(frame, textvariable=var, width=9).grid(
                    row=index // 2, column=(index % 2) * 2 + 1, sticky="w", padx=(0, 8), pady=2
                )
                self.action_vars[name] = var
        elif action_type == "key":
            frame = ttk.Frame(self.action_dynamic)
            frame.pack(fill="x")
            ttk.Label(frame, text="按键").pack(side="left")
            var = tk.StringVar(value=str(action.get("key", "back")))
            var.trace_add("write", self._mark_dirty)
            ttk.Combobox(
                frame,
                textvariable=var,
                values=["back", "home", "enter", "menu", "power", "volume_up", "volume_down"],
                width=16,
            ).pack(side="left", padx=6)
            self.action_vars["key"] = var
        elif action_type == "text":
            frame = ttk.Frame(self.action_dynamic)
            frame.pack(fill="x")
            ttk.Label(frame, text="发送文本").pack(anchor="w")
            var = tk.StringVar(value=str(action.get("text", "")))
            var.trace_add("write", self._mark_dirty)
            ttk.Entry(frame, textvariable=var).pack(fill="x", pady=2)
            options = ttk.Frame(frame)
            options.pack(fill="x", pady=2)
            clear_var = tk.BooleanVar(value=bool(action.get("clear", False)))
            clear_var.trace_add("write", self._mark_dirty)
            ttk.Checkbutton(options, text="先清空输入框", variable=clear_var).pack(
                side="left", padx=(0, 10)
            )
            submit_var = tk.BooleanVar(value=bool(action.get("submit", False)))
            submit_var.trace_add("write", self._mark_dirty)
            ttk.Checkbutton(options, text="输入后回车", variable=submit_var).pack(side="left")
            ttk.Label(
                frame,
                text="中文输入依赖手机输入法；批量中文回复建议安装 ADBKeyboard。",
                foreground="#64748b",
                wraplength=320,
            ).pack(anchor="w")
            self.action_vars["text"] = var
            self.action_vars["clear"] = clear_var
            self.action_vars["submit"] = submit_var
        elif action_type == "launch":
            frame = ttk.Frame(self.action_dynamic)
            frame.pack(fill="x")
            ttk.Label(frame, text="应用包名").pack(anchor="w")
            package_var = tk.StringVar(value=str(action.get("package", "")))
            package_var.trace_add("write", self._mark_dirty)
            ttk.Entry(frame, textvariable=package_var).pack(fill="x", pady=2)
            fresh_var = tk.BooleanVar(value=bool(action.get("fresh", True)))
            fresh_var.trace_add("write", self._mark_dirty)
            ttk.Checkbutton(
                frame, text="先强制停止再启动（回到首页）", variable=fresh_var
            ).pack(anchor="w")
            ttk.Label(
                frame,
                text="例如拼多多 com.xunmeng.pinduoduo，淘宝 com.taobao.taobao。",
                foreground="#64748b",
                wraplength=320,
            ).pack(anchor="w")
            self.action_vars["package"] = package_var
            self.action_vars["fresh"] = fresh_var
        elif action_type == "wait":
            frame = ttk.Frame(self.action_dynamic)
            frame.pack(fill="x")
            ttk.Label(frame, text="等待秒数").pack(side="left")
            var = tk.StringVar(value=str(action.get("seconds", 1)))
            var.trace_add("write", self._mark_dirty)
            ttk.Entry(frame, textvariable=var, width=10).pack(side="left", padx=6)
            self.action_vars["seconds"] = var
        else:
            ttk.Label(
                self.action_dynamic,
                text="该步骤只匹配画面，不执行动作。",
                foreground="#64748b",
            ).pack(anchor="w")

    def _action_type_key(self) -> str:
        return ACTION_VALUES.get(self.action_type_var.get(), "tap")

    def _collect_action(self) -> dict:
        action: dict = {}
        for name, var in self.action_vars.items():
            value = var.get()
            if name in {"key", "text", "package"}:
                action[name] = value
            elif name in {"clear", "submit", "fresh"}:
                action[name] = bool(value)
            else:
                try:
                    number = float(value)
                    action[name] = int(number) if name == "duration" else number
                except ValueError:
                    action[name] = 1 if name == "duration" else 0.0
        for name, var in (("pre_wait", self.pre_wait_var), ("post_wait", self.post_wait_var)):
            try:
                action[name] = float(var.get())
            except ValueError:
                action[name] = 0.0
        return action

    def _save_node(self, silent: bool = False, redraw: bool = True):
        if self._loading or self.selected_node_id is None:
            return
        node = self.nodes_by_id.get(self.selected_node_id)
        if node is None:
            return
        if not silent and not self._node_edit_open:
            self._push_flow_undo("保存节点")
            self._node_edit_open = True
        name = self.name_var.get().strip() or node.name
        kind = KIND_VALUES.get(self.kind_var.get(), node.kind)
        try:
            threshold = min(0.999, max(0.3, float(self.threshold_var.get())))
        except ValueError:
            threshold = node.threshold
        try:
            roi_weight = min(1.0, max(0.0, float(self.roi_weight_var.get())))
        except ValueError:
            roi_weight = node.roi_weight
        click_x = self.current_click[0] if self.current_click else None
        click_y = self.current_click[1] if self.current_click else None
        roi = self.current_roi
        target = self.current_target
        self.db.update_node(
            node.id,
            name=name,
            kind=kind,
            threshold=threshold,
            roi_weight=roi_weight,
            click_x=click_x,
            click_y=click_y,
            roi_x=roi[0] if roi else None,
            roi_y=roi[1] if roi else None,
            roi_w=roi[2] if roi else None,
            roi_h=roi[3] if roi else None,
            target_x=target[0] if target else None,
            target_y=target[1] if target else None,
            target_w=target[2] if target else None,
            target_h=target[3] if target else None,
            action_type=self._action_type_key(),
            action=self._collect_action(),
            is_entry=self.entry_var.get(),
            notes=self.notes_text.get("1.0", "end").strip(),
        )
        self.nodes = self.db.list_nodes(self.flow_id)
        self.nodes_by_id = {item.id: item for item in self.nodes}
        self._fill_outgoing()
        if redraw:
            self._draw_graph()
        self._dirty = False
        if not silent:
            self._status(f"节点已保存：{name}")

    def _select_node(self, node_id: int):
        if node_id == self.selected_node_id and self.selected_edge_id is None:
            return
        self.flush()
        self._node_edit_open = False
        node = self.nodes_by_id.get(node_id)
        if node is None:
            return
        self.selected_node_id = node_id
        self.selected_edge_id = None
        self._loading = True
        self.name_var.set(node.name)
        self.kind_var.set(KIND_LABELS.get(node.kind, KIND_LABELS["image"]))
        self.threshold_var.set(f"{node.threshold:.2f}")
        self.roi_weight_var.set(f"{node.roi_weight:.2f}")
        self.entry_var.set(node.is_entry)
        self.action_type_var.set(ACTION_LABELS.get(node.action_type, ACTION_LABELS["tap"]))
        if node.action_type != "none":
            self._last_action_label = ACTION_LABELS.get(
                node.action_type, ACTION_LABELS["tap"]
            )
        else:
            self._last_action_label = ACTION_LABELS["tap"]
        self.pre_wait_var.set(str(node.action.get("pre_wait", 0)))
        self.post_wait_var.set(str(node.action.get("post_wait", node.action.get("wait", 0.6))))
        self.current_click = (
            (node.click_x, node.click_y) if node.has_click_point else None
        )
        self.current_roi = node.roi
        self.current_target = node.target_region
        self.click_info_var.set(
            "点击点：未设置"
            if self.current_click is None
            else f"点击点：x={self.current_click[0]:.4f}  y={self.current_click[1]:.4f}"
        )
        self.roi_info_var.set(
            "检测重点：未设置"
            if self.current_roi is None
            else f"检测重点：x={self.current_roi[0]:.3f} y={self.current_roi[1]:.3f} "
            f"w={self.current_roi[2]:.3f} h={self.current_roi[3]:.3f}"
        )
        self.target_info_var.set(
            "点击对象：未设置"
            if self.current_target is None
            else f"点击对象：x={self.current_target[0]:.3f} y={self.current_target[1]:.3f} "
            f"w={self.current_target[2]:.3f} h={self.current_target[3]:.3f}"
        )
        self.no_click_var.set(node.action_type == "none")
        self.simulate_button.configure(
            state="disabled" if node.action_type == "none" else "normal"
        )
        self.image_mode_var.set("view")
        self.image_canvas.set_mode("view")
        if node.image_path and Path(node.image_path).exists():
            self.image_canvas.set_source(node.image_path)
        else:
            self.image_canvas.set_source(None)
        self.image_canvas.zoom_fit()
        self.image_canvas.set_markers(self.current_click, self.current_roi, self.current_target)
        self.notes_text.delete("1.0", tk.END)
        if node.notes:
            self.notes_text.insert("1.0", node.notes)
        self.notes_text.edit_modified(False)
        self._rebuild_action_fields(node)
        self._fill_outgoing()
        self._loading = False
        self._dirty = False
        self._draw_graph()
        self._status(
            f"已选中节点：{node.name} · {KIND_LABELS.get(node.kind, node.kind)} · "
            f"{node.action_summary}"
        )

    def _fill_outgoing(self):
        self.outgoing.delete(*self.outgoing.get_children())
        if self.selected_node_id is None:
            return
        for edge in self.edges:
            if edge.src_id != self.selected_node_id:
                continue
            target = self.nodes_by_id.get(edge.dst_id)
            self.outgoing.insert(
                "",
                "end",
                iid=str(edge.id),
                values=(
                    target.name if target else f"#{edge.dst_id}",
                    f"{edge.weight:g}",
                    edge.label,
                ),
            )

    def _reload_graph(self):
        if self.flow_id is None:
            return
        self.flush()
        self.nodes = self.db.list_nodes(self.flow_id)
        self.edges = self.db.list_edges(self.flow_id)
        self.nodes_by_id = {node.id: node for node in self.nodes}
        self.edges_by_id = {edge.id: edge for edge in self.edges}
        if self.selected_node_id not in self.nodes_by_id:
            self.selected_node_id = None
        if self.selected_edge_id not in self.edges_by_id:
            self.selected_edge_id = None
        self._draw_graph()
        self._fill_outgoing()
        self._status("已刷新")

    def _draw_graph(self):
        self.canvas.delete("all")
        self._draw_edges()
        for node in self.nodes:
            self._draw_node(node)
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(
                scrollregion=(0, 0, max(1200, bbox[2] + 120), max(900, bbox[3] + 120))
            )
        if not self.nodes:
            self.canvas.create_text(
                380,
                220,
                text="此流程还没有节点\n点击上方「添加节点」导入手机截图",
                fill="#64748b",
                justify="center",
                font=("TkDefaultFont", 13),
            )

    def _draw_edges(self):
        self.canvas.delete("edge")
        for edge in self.edges:
            src = self.nodes_by_id.get(edge.src_id)
            dst = self.nodes_by_id.get(edge.dst_id)
            if src is None or dst is None:
                continue
            src_center = (src.x + NODE_W / 2, src.y + NODE_H / 2)
            dst_center = (dst.x + NODE_W / 2, dst.y + NODE_H / 2)
            start = self._border_point(src, dst_center)
            end = self._border_point(dst, src_center)
            selected = edge.id == self.selected_edge_id
            color = "#1c64d9" if selected else "#7a8ba3"
            width = 3 if selected else 2
            self.canvas.create_line(
                *start,
                *end,
                fill=color,
                width=width,
                arrow=tk.LAST,
                arrowshape=(12, 15, 5),
                smooth=True,
                tags=("edge", f"edge:{edge.id}"),
            )
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            text = f"×{edge.weight:g}" if edge.weight != 1 else ""
            if edge.label:
                text = f"{edge.label} {text}".strip()
            if text:
                self.canvas.create_text(
                    mid_x,
                    mid_y - 8,
                    text=text,
                    fill="#334155",
                    font=("TkDefaultFont", 8, "bold"),
                    tags=("edge", f"edge:{edge.id}"),
                )

    @staticmethod
    def _border_point(node: Node, target: tuple[float, float]) -> tuple[float, float]:
        cx = node.x + NODE_W / 2
        cy = node.y + NODE_H / 2
        dx = target[0] - cx
        dy = target[1] - cy
        if dx == 0 and dy == 0:
            return cx, cy
        tx = (NODE_W / 2) / abs(dx) if dx else float("inf")
        ty = (NODE_H / 2) / abs(dy) if dy else float("inf")
        t = min(tx, ty)
        return cx + dx * t, cy + dy * t

    def _thumb(self, node: Node):
        if not node.image_path or not Path(node.image_path).exists():
            return None
        try:
            mtime = os.path.getmtime(node.image_path)
            key = (node.image_path, mtime)
            if key not in self._thumb_cache:
                with Image.open(node.image_path) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((54, 74), Image.Resampling.LANCZOS)
                    self._thumb_cache[key] = photo_image(thumb)
                if len(self._thumb_cache) > 300:
                    for old_key in list(self._thumb_cache)[:100]:
                        self._thumb_cache.pop(old_key, None)
            return self._thumb_cache[key]
        except Exception:
            return None

    def _draw_node(self, node: Node):
        x, y = node.x, node.y
        selected = node.id == self.selected_node_id
        body_fill = {
            "image": "#ffffff",
            "action": "#ffffff",
            "end": "#fff5f5",
        }.get(node.kind, "#ffffff")
        header_fill = {
            "image": "#dbe9ff",
            "action": "#e8f1da",
            "end": "#ffdede",
        }.get(node.kind, "#dbe9ff")
        outline = "#1c64d9" if selected else "#9fb2c9"
        width = 3 if selected else 1
        tags = ("node", f"node:{node.id}")
        self.canvas.create_rectangle(
            x, y, x + NODE_W, y + NODE_H, fill=body_fill, outline=outline, width=width, tags=tags
        )
        self.canvas.create_rectangle(
            x + 1,
            y + 1,
            x + NODE_W - 1,
            y + 24,
            fill=header_fill,
            outline="",
            tags=tags,
        )
        title = node.name if len(node.name) <= 16 else node.name[:15] + "…"
        self.canvas.create_text(
            x + 8,
            y + 6,
            text=title,
            anchor="nw",
            fill="#1f2d3d",
            font=("TkDefaultFont", 10, "bold"),
            tags=tags,
        )
        subtitle = f"{KIND_LABELS.get(node.kind, node.kind)} · {node.action_summary}"
        if node.has_roi:
            subtitle += " · 重点区"
        self.canvas.create_text(
            x + 8,
            y + 32,
            text=subtitle,
            anchor="nw",
            width=NODE_W - 80,
            fill="#465a70",
            font=("TkDefaultFont", 8),
            tags=tags,
        )
        if node.has_click_point:
            self.canvas.create_text(
                x + 8,
                y + 70,
                text=f"点击 {node.click_x:.2f},{node.click_y:.2f}",
                anchor="nw",
                fill="#b23c3c",
                font=("TkDefaultFont", 8),
                tags=tags,
            )
        if node.is_entry:
            self.canvas.create_rectangle(
                x + 8,
                y + NODE_H - 22,
                x + 48,
                y + NODE_H - 7,
                fill="#1c64d9",
                outline="",
                tags=tags,
            )
            self.canvas.create_text(
                x + 28,
                y + NODE_H - 15,
                text="入口",
                fill="white",
                font=("TkDefaultFont", 8, "bold"),
                tags=tags,
            )
        thumb = self._thumb(node)
        if thumb is not None:
            self.canvas.create_image(
                x + NODE_W - 60,
                y + 30,
                image=thumb,
                anchor="nw",
                tags=tags,
            )
        else:
            self.canvas.create_rectangle(
                x + NODE_W - 60,
                y + 30,
                x + NODE_W - 10,
                y + 82,
                outline="#c8d3e0",
                dash=(3, 2),
                tags=tags,
            )
            self.canvas.create_text(
                x + NODE_W - 35,
                y + 56,
                text="无图",
                fill="#94a3b8",
                font=("TkDefaultFont", 8),
                tags=tags,
            )

    def _graph_press(self, event):
        self.canvas.focus_set()
        item = self.canvas.find_withtag("current")
        tags = self.canvas.gettags(item) if item else ()
        node_id = _tag_id(tags, "node:")
        edge_id = _tag_id(tags, "edge:")
        if node_id is not None:
            self._select_node(node_id)
            node = self.nodes_by_id.get(node_id)
            if node:
                self._drag = {
                    "id": node_id,
                    "last_x": event.x,
                    "last_y": event.y,
                    "moved": False,
                    "snapshot": snapshot_flow(self.db, self.flow_id) if self.flow_id else None,
                }
                self.canvas.tag_raise(f"node:{node_id}")
        elif edge_id is not None:
            self.selected_edge_id = edge_id
            self._draw_graph()
            self._status(f"已选中连线 #{edge_id}，双击可编辑权重，Delete 可删除")
        else:
            self.selected_node_id = None
            self.selected_edge_id = None
            self._draw_graph()
            self._status("已取消选择")

    def _graph_drag(self, event):
        if not self._drag:
            return
        node = self.nodes_by_id.get(self._drag["id"])
        if node is None:
            return
        dx = event.x - self._drag["last_x"]
        dy = event.y - self._drag["last_y"]
        if abs(dx) + abs(dy) < 1:
            return
        self._drag["last_x"] = event.x
        self._drag["last_y"] = event.y
        self._drag["moved"] = True
        new_x = max(0, min(3000, node.x + dx))
        new_y = max(0, min(3000, node.y + dy))
        move_x = new_x - node.x
        move_y = new_y - node.y
        node.x = new_x
        node.y = new_y
        self.canvas.move(f"node:{node.id}", move_x, move_y)
        self._draw_edges()
        self.canvas.tag_lower("edge")

    def _graph_release(self, _event):
        if not self._drag:
            return
        drag = self._drag
        self._drag = None
        if drag["moved"]:
            node = self.nodes_by_id.get(drag["id"])
            if node:
                self.flow_undo.push("移动节点", drag.get("snapshot"))
                self.db.update_node(node.id, x=node.x, y=node.y)
                self._status(f"已移动节点：{node.name}")

    def _graph_double(self, event):
        item = self.canvas.find_withtag("current")
        tags = self.canvas.gettags(item) if item else ()
        edge_id = _tag_id(tags, "edge:")
        node_id = _tag_id(tags, "node:")
        if edge_id is not None:
            self._edit_edge(edge_id)
        elif node_id is not None:
            self.name_entry.focus_set()
            self.name_entry.selection_range(0, tk.END)

    def _next_position(self):
        count = len(self.nodes)
        return 70 + (count % 4) * 270, 70 + (count // 4) * 160

    def _copy_image(self, source: str) -> tuple[str, int, int]:
        source_path = Path(source)
        suffix = source_path.suffix.lower() or ".png"
        target = config.images_dir() / f"flow{self.flow_id}_{uuid4().hex[:10]}{suffix}"
        shutil.copy2(source_path, target)
        with Image.open(target) as image:
            width, height = image.size
        return str(target), width, height

    def _embed(self, node_id: int, image_path: str):
        try:
            embedder = get_embedder()
            vector = embedder.embed(image_path)
            self.db.set_node_embedding(
                node_id, embed_to_blob(vector), embedder.kind, int(vector.shape[0])
            )
        except Exception as exc:
            self._status(f"向量计算失败：{exc}")

    def _add_node(self, kind: str = "image"):
        if self.flow_id is None:
            messagebox.showinfo("提示", "请先新建或选择一个流程", parent=self)
            return
        path = filedialog.askopenfilename(
            title="选择手机截图", filetypes=IMAGE_TYPES, parent=self
        )
        if not path:
            return
        try:
            image_path, width, height = self._copy_image(path)
        except Exception as exc:
            messagebox.showerror("错误", f"复制图片失败：{exc}", parent=self)
            return
        self._push_flow_undo("添加节点")
        x, y = self._next_position()
        name = Path(path).stem[:24] or "新步骤"
        node_id = self.db.add_node(
            self.flow_id,
            name=name,
            kind=kind,
            image_path=image_path,
            img_w=width,
            img_h=height,
            click_x=0.5,
            click_y=0.5,
            action_type="none" if kind == "end" else "tap",
            action={"post_wait": 0.0 if kind == "end" else 0.6},
            threshold=0.88,
            is_entry=not any(node.is_entry for node in self.nodes),
            x=x,
            y=y,
        )
        self._embed(node_id, image_path)
        self._reload_graph()
        self._select_node(node_id)
        self._status(f"已添加节点：{name}")

    def _replace_image(self):
        if self.selected_node_id is None:
            return
        node = self.nodes_by_id.get(self.selected_node_id)
        if node is None:
            return
        path = filedialog.askopenfilename(
            title="选择新的状态图片", filetypes=IMAGE_TYPES, parent=self
        )
        if not path:
            return
        try:
            image_path, width, height = self._copy_image(path)
        except Exception as exc:
            messagebox.showerror("错误", f"复制图片失败：{exc}", parent=self)
            return
        self._push_flow_undo("更换图片")
        self.db.update_node(
            node.id,
            image_path=image_path,
            img_w=width,
            img_h=height,
            click_x=None,
            click_y=None,
            roi_x=None,
            roi_y=None,
            roi_w=None,
            roi_h=None,
            target_x=None,
            target_y=None,
            target_w=None,
            target_h=None,
        )
        self._embed(node.id, image_path)
        self.current_click = None
        self.current_roi = None
        self.current_target = None
        self._reload_graph()
        self._select_node(node.id)
        self._status("图片已更换，请重新取点击点、检测重点与点击对象")

    def _simulate_click(self):
        if self.selected_node_id is None:
            messagebox.showinfo("提示", "请先选中一个节点", parent=self)
            return
        if self.no_click_var.get():
            messagebox.showinfo("提示", "当前节点已设置为「不模拟点击」", parent=self)
            return
        if self.current_click is None:
            messagebox.showinfo("提示", "请先在图片上取点击点", parent=self)
            return
        serial = self._selected_serial()
        try:
            device = AdbDevice(serial=serial)
            width, height = device.screen_size()
            x = self.current_click[0] * width
            y = self.current_click[1] * height
            device.tap(x, y)
        except Exception as exc:
            messagebox.showerror("模拟点击失败", str(exc), parent=self)
            return
        self._status(f"已发送点击：({int(x)}, {int(y)}) · 设备 {width}x{height}")

    def _screenshot_node(self):
        if self.flow_id is None:
            messagebox.showinfo("提示", "请先选择一个流程", parent=self)
            return
        serial = self._selected_serial()
        try:
            device = AdbDevice(serial=serial)
            frame = device.screenshot()
        except Exception as exc:
            messagebox.showerror("截图失败", str(exc), parent=self)
            return
        stamp = time.strftime("%m%d_%H%M%S")
        target = config.images_dir() / f"flow{self.flow_id}_{stamp}_{uuid4().hex[:6]}.png"
        vision.save_rgb(target, frame)
        height, width = frame.shape[:2]
        x, y = self._next_position()
        node_id = self.db.add_node(
            self.flow_id,
            name=f"手机截图 {stamp}",
            kind="image",
            image_path=str(target),
            img_w=width,
            img_h=height,
            click_x=0.5,
            click_y=0.5,
            action_type="tap",
            action={"post_wait": 0.6},
            threshold=0.88,
            is_entry=not any(node.is_entry for node in self.nodes),
            x=x,
            y=y,
        )
        self._embed(node_id, str(target))
        self._reload_graph()
        self._select_node(node_id)
        self._status("已用手机当前画面创建节点")

    def _connect(self):
        if self.flow_id is None or not self.nodes:
            messagebox.showinfo("提示", "请先在流程中添加节点", parent=self)
            return
        dialog = ConnectDialog(self, self.nodes, src=self.selected_node_id)
        self.wait_window(dialog)
        if not dialog.result:
            return
        self._push_flow_undo("添加连线")
        self.db.add_edge(
            self.flow_id,
            dialog.result["src"],
            dialog.result["dst"],
            dialog.result["weight"],
            dialog.result["label"],
        )
        self._reload_graph()
        self._status("已添加下一步 / 偏好连线")

    def _edit_outgoing_edge(self):
        selection = self.outgoing.selection()
        if not selection:
            return
        try:
            edge_id = int(selection[0])
        except ValueError:
            return
        self._edit_edge(edge_id)

    def _edit_edge(self, edge_id: int):
        edge = self.db.get_edge(edge_id)
        if edge is None:
            return
        dialog = EdgeDialog(self, edge.weight, edge.label)
        self.wait_window(dialog)
        if not dialog.result:
            return
        self._push_flow_undo("编辑连线")
        self.db.update_edge(edge_id, **dialog.result)
        self._reload_graph()
        self._status("连线已更新")

    def _delete_outgoing_edge(self):
        selection = self.outgoing.selection()
        if not selection:
            return
        edge_id = int(selection[0])
        if not messagebox.askyesno("确认", "删除这条连线？", parent=self):
            return
        self._push_flow_undo("删除连线")
        self.db.delete_edge(edge_id)
        self._reload_graph()

    def _delete_selected(self):
        if self.selected_edge_id is not None:
            edge_id = self.selected_edge_id
            if not messagebox.askyesno("确认", "删除选中的连线？", parent=self):
                return
            self._push_flow_undo("删除连线")
            self.db.delete_edge(edge_id)
            self.selected_edge_id = None
            self._reload_graph()
            return
        if self.selected_node_id is None:
            return
        node = self.nodes_by_id.get(self.selected_node_id)
        if node is None:
            return
        if not messagebox.askyesno(
            "确认删除", f"删除节点「{node.name}」以及相关连线？", parent=self
        ):
            return
        self._push_flow_undo("删除节点")
        self.db.delete_node(node.id)
        self.selected_node_id = None
        self._clear_inspector()
        self._reload_graph()
        self._status("节点已删除")

    def _compute_layout(self) -> dict[int, tuple[float, float]]:
        if not self.nodes:
            return {}
        outgoing = {}
        indegree = {node.id: 0 for node in self.nodes}
        for edge in self.edges:
            outgoing.setdefault(edge.src_id, []).append(edge.dst_id)
            indegree[edge.dst_id] = indegree.get(edge.dst_id, 0) + 1
        entry = next((node.id for node in self.nodes if node.is_entry), None)
        roots = [entry] if entry else [nid for nid, count in indegree.items() if count == 0]
        if not roots:
            roots = [self.nodes[0].id]
        levels: dict[int, int] = {}
        queue = [(root, 0) for root in roots]
        while queue:
            node_id, level = queue.pop(0)
            if node_id in levels:
                continue
            levels[node_id] = level
            for nxt in outgoing.get(node_id, []):
                if nxt not in levels:
                    queue.append((nxt, level + 1))
        next_level = max(levels.values(), default=0) + 1
        for node in self.nodes:
            if node.id not in levels:
                levels[node.id] = next_level
                next_level += 1
        counts: dict[int, int] = {}
        positions: dict[int, tuple[float, float]] = {}
        for node in self.nodes:
            level = levels[node.id]
            index = counts.get(level, 0)
            counts[level] = index + 1
            positions[node.id] = (70 + index * 260, 70 + level * 150)
        return positions

    def _auto_layout(self):
        if not self.nodes:
            return
        self._push_flow_undo("自动布局")
        for node_id, (x, y) in self._compute_layout().items():
            self.db.update_node(node_id, x=x, y=y)
        self._reload_graph()
        self._status("已自动布局")

    def _load_settings(self, flow: Flow):
        settings = merged_settings(flow)
        fallback = settings.get("fallback", {})
        self._loading = True
        self.settings_vars["beta"].set(str(settings.get("beta", 0.12)))
        self.settings_vars["stay_penalty"].set(str(settings.get("stay_penalty", 0.06)))
        scope = settings.get("match_scope")
        if not scope:
            scope = "reachable" if settings.get("only_reachable") else "next"
        self.settings_vars["match_scope"].set(SCOPE_LABELS.get(scope, SCOPE_LABELS["next"]))
        self.settings_vars["save_screenshots"].set(bool(settings.get("save_screenshots", True)))
        mode = str(settings.get("selection_mode", "best"))
        self.settings_vars["selection_mode"].set(SELECTION_LABELS.get(mode, mode))
        for key in ("max_retries", "threshold_step", "min_threshold", "max_fallbacks", "retry_wait"):
            self.settings_vars[key].set(str(fallback.get(key, "")))
        self.fallback_steps = [dict(step) for step in fallback.get("steps", [])]
        self._refresh_fallback_tree()
        self._loading = False

    def _save_settings(self):
        if self.flow_id is None:
            return
        self._push_flow_undo("保存运行设置")
        flow = self.db.get_flow(self.flow_id)
        if flow is None:
            return
        settings = dict(flow.settings)
        try:
            settings["beta"] = float(self.settings_vars["beta"].get())
            settings["stay_penalty"] = float(self.settings_vars["stay_penalty"].get())
        except ValueError:
            pass
        settings["match_scope"] = SCOPE_VALUES.get(
            self.settings_vars["match_scope"].get(), "next"
        )
        settings.pop("only_reachable", None)
        settings["save_screenshots"] = self.settings_vars["save_screenshots"].get()
        settings["selection_mode"] = SELECTION_VALUES.get(
            self.settings_vars["selection_mode"].get(), "best"
        )
        fallback = dict(settings.get("fallback") or {})
        for key in ("max_retries", "max_fallbacks"):
            try:
                fallback[key] = int(float(self.settings_vars[key].get()))
            except ValueError:
                pass
        for key in ("threshold_step", "min_threshold", "retry_wait"):
            try:
                fallback[key] = float(self.settings_vars[key].get())
            except ValueError:
                pass
        fallback["steps"] = [dict(step) for step in self.fallback_steps]
        settings["fallback"] = fallback
        self.db.update_flow(self.flow_id, settings=settings)
        self._status("运行设置已保存")

    def _refresh_fallback_tree(self):
        self.fallback_tree.delete(*self.fallback_tree.get_children())
        for index, step in enumerate(self.fallback_steps):
            self.fallback_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(index + 1, FALLBACK_LABELS.get(step.get("type"), step.get("type")), self._fallback_detail(step)),
            )

    @staticmethod
    def _fallback_detail(step: dict) -> str:
        kind = step.get("type")
        if kind == "key":
            return f"按键 {step.get('key', 'back')}"
        if kind == "wait":
            return f"等待 {step.get('seconds', 1)} 秒"
        if kind == "tap":
            return f"点击 {step.get('x', 0.5)}, {step.get('y', 0.5)}"
        if kind == "swipe":
            return (
                f"({step.get('x1', 0.5)},{step.get('y1', 0.8)}) -> "
                f"({step.get('x2', 0.5)},{step.get('y2', 0.2)})"
            )
        if kind == "text":
            return f"输入 {str(step.get('text', ''))[:24]}"
        return str(step)

    def _add_fallback_step(self):
        dialog = FallbackStepDialog(self)
        self.wait_window(dialog)
        if dialog.result:
            self.fallback_steps.append(dialog.result)
            self._refresh_fallback_tree()

    def _edit_fallback_step(self):
        selection = self.fallback_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        dialog = FallbackStepDialog(self, self.fallback_steps[index])
        self.wait_window(dialog)
        if dialog.result:
            self.fallback_steps[index] = dialog.result
            self._refresh_fallback_tree()

    def _remove_fallback_step(self):
        selection = self.fallback_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        self.fallback_steps.pop(index)
        self._refresh_fallback_tree()

    def _move_fallback(self, direction: int):
        selection = self.fallback_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        target = index + direction
        if not (0 <= target < len(self.fallback_steps)):
            return
        self.fallback_steps[index], self.fallback_steps[target] = (
            self.fallback_steps[target],
            self.fallback_steps[index],
        )
        self._refresh_fallback_tree()
        self.fallback_tree.selection_set(str(target))

    def _load_flow_info(self, flow: Flow):
        self._loading = True
        self.flow_desc.delete("1.0", tk.END)
        if flow.description:
            self.flow_desc.insert("1.0", flow.description)
        counts = {}
        for node in self.nodes:
            counts[node.kind] = counts.get(node.kind, 0) + 1
        self.flow_stats.configure(
            text=(
                f"流程：{flow.name}\n"
                f"节点：{len(self.nodes)} 个  ·  连线：{len(self.edges)} 条\n"
                f"图片步骤：{counts.get('image', 0)}  ·  动作步骤：{counts.get('action', 0)}  ·  "
                f"结束步骤：{counts.get('end', 0)}\n"
                f"创建：{flow.created_at}\n更新：{flow.updated_at}"
            )
        )
        self._loading = False

    def _save_flow_info(self):
        if self.flow_id is None:
            return
        self._push_flow_undo("保存流程说明")
        self.db.update_flow(
            self.flow_id, description=self.flow_desc.get("1.0", "end").strip()
        )
        self._status("流程说明已保存")

    def _new_flow(self):
        dialog = FlowDialog(self, "新建流程")
        self.wait_window(dialog)
        if not dialog.result:
            return
        flow_id = self.db.create_flow(dialog.result["name"], dialog.result["description"])
        self.refresh_flows(select_id=flow_id)
        self._notify_other_views()
        self._status("流程已创建")

    def _rename_flow(self):
        if self.flow_id is None:
            return
        flow = self.db.get_flow(self.flow_id)
        if flow is None:
            return
        dialog = FlowDialog(self, "编辑流程", flow.name, flow.description)
        self.wait_window(dialog)
        if not dialog.result:
            return
        self.db.update_flow(
            flow.id,
            name=dialog.result["name"],
            description=dialog.result["description"],
        )
        self.refresh_flows(select_id=flow.id)
        self._notify_other_views()

    def _delete_flow(self):
        if self.flow_id is None:
            return
        flow = self.db.get_flow(self.flow_id)
        if flow is None:
            return
        if not messagebox.askyesno(
            "确认删除",
            f"删除流程「{flow.name}」以及所有节点、连线、运行记录？\n图片文件会保留在 data/images。",
            parent=self,
        ):
            return
        self.db.delete_flow(flow.id)
        self.flow_id = None
        self.flow_undo.clear()
        self._node_edit_open = False
        self.refresh_flows(keep_selection=False)
        self._notify_other_views()

    def _export_flow(self):
        if self.flow_id is None:
            return
        flow = self.db.get_flow(self.flow_id)
        if flow is None:
            return
        default_name = f"{flow.name}.scflow.json"
        path = filedialog.asksaveasfilename(
            title="导出流程",
            defaultextension=".scflow.json",
            initialdir=str(config.exports_dir()),
            initialfile=default_name,
            filetypes=[("scflow 流程", "*.scflow.json"), ("JSON", "*.json")],
            parent=self,
        )
        if not path:
            return
        try:
            target = export_flow(self.db, self.flow_id, path)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc), parent=self)
            return
        self._status(f"已导出：{target}")
        messagebox.showinfo("导出成功", f"流程已导出到：\n{target}", parent=self)

    def _import_flow(self):
        path = filedialog.askopenfilename(
            title="选择流程文件",
            filetypes=[("scflow 流程", "*.scflow.json"), ("JSON", "*.json")],
            parent=self,
        )
        if not path:
            return
        dialog = ImportDialog(self, self.db, path, self.flow_id, self.nodes)
        self.wait_window(dialog)
        if not dialog.result:
            return
        try:
            result = import_flow(
                self.db,
                path,
                target_flow_id=dialog.result["target_flow_id"],
                link_from_node_id=dialog.result["link_from"],
                link_to_node_id=dialog.result["link_to"],
                link_weight=dialog.result["link_weight"],
                new_flow_name=dialog.result["new_flow_name"],
            )
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc), parent=self)
            return
        self.refresh_flows(select_id=result["flow_id"])
        self._notify_other_views()
        self._status(
            f"导入完成：{result['nodes']} 个节点，{result['edges']} 条连线，"
            f"{result['links']} 条衔接"
        )
        messagebox.showinfo(
            "导入成功",
            f"流程：{result['flow_name']}\n节点：{result['nodes']}\n"
            f"连线：{result['edges']}\n人工衔接：{result['links']}",
            parent=self,
        )

    def _notify_other_views(self):
        try:
            self.app.run_view.refresh_flows()
            self.app.history_view.refresh_flows()
        except Exception:
            pass

    def _open_data_dir(self):
        path = config.home()
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("无法打开", str(exc), parent=self)

    def refresh_devices(self):
        devices = list_adb_devices()
        values = ["（自动选择）"]
        for device in devices:
            if device["state"] == "device":
                model = f" · {device['model']}" if device["model"] else ""
                values.append(f"{device['serial']}{model}")
        self.device_combo["values"] = values
        if not self.device_combo.get():
            self.device_combo.current(0)

    def _selected_serial(self) -> str | None:
        value = self.device_combo.get()
        if not value or value.startswith("（"):
            return None
        return value.split(" · ", 1)[0]

    def _build_project_tab(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="新建项目", command=self._new_project).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(toolbar, text="删除项目", command=self._delete_project).pack(
            side="left", padx=4
        )
        ttk.Button(toolbar, text="保存项目", command=self._save_project).pack(
            side="left", padx=4
        )
        ttk.Button(toolbar, text="撤回", command=self._undo_project).pack(
            side="left", padx=4
        )
        ttk.Button(
            toolbar, text="刷新", command=lambda: self.refresh_projects(keep_selection=True)
        ).pack(side="left", padx=4)
        ttk.Label(
            toolbar,
            text="拖动卡片排序 · 双击卡片打开区块画布",
            foreground="#64748b",
        ).pack(side="right")

        vertical = ttk.PanedWindow(parent, orient="vertical")
        vertical.pack(fill="both", expand=True)
        top = ttk.Frame(vertical)
        bottom = ttk.Frame(vertical)
        vertical.add(top, weight=3)
        vertical.add(bottom, weight=2)

        horizontal = ttk.PanedWindow(top, orient="horizontal")
        horizontal.pack(fill="both", expand=True)
        left = ttk.Frame(horizontal, width=250)
        center = ttk.Frame(horizontal)
        right = ttk.Frame(horizontal, width=300)
        horizontal.add(left, weight=0)
        horizontal.add(center, weight=3)
        horizontal.add(right, weight=0)

        self._build_project_left(left)
        self._build_project_center(center)
        self._build_project_right(right)
        self._build_project_bottom(bottom)

    def _build_project_left(self, parent):
        frame = ttk.LabelFrame(parent, text="项目库", padding=6)
        frame.pack(fill="both", expand=True, pady=(0, 6))
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True)
        self.project_list = tk.Listbox(
            list_frame, exportselection=False, activestyle="none", height=7
        )
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.project_list.yview)
        self.project_list.configure(yscrollcommand=scroll.set)
        self.project_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.project_list.bind("<<ListboxSelect>>", self._on_project_select)

        frame2 = ttk.LabelFrame(parent, text="区块库（所有流程）", padding=6)
        frame2.pack(fill="both", expand=True)
        list_frame2 = ttk.Frame(frame2)
        list_frame2.pack(fill="both", expand=True)
        self.block_library = tk.Listbox(
            list_frame2, exportselection=False, activestyle="none", height=10
        )
        scroll2 = ttk.Scrollbar(list_frame2, orient="vertical", command=self.block_library.yview)
        self.block_library.configure(yscrollcommand=scroll2.set)
        self.block_library.pack(side="left", fill="both", expand=True)
        scroll2.pack(side="right", fill="y")
        buttons = ttk.Frame(frame2)
        buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(buttons, text="刷新", command=self.refresh_block_library).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(buttons, text="添加为区块", command=self._add_block_from_library).pack(
            side="left", padx=4
        )

    def _build_project_center(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(toolbar, text="上移", command=lambda: self._move_block(-1)).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(toolbar, text="下移", command=lambda: self._move_block(1)).pack(
            side="left", padx=4
        )
        ttk.Button(toolbar, text="移除区块", command=self._remove_block).pack(
            side="left", padx=4
        )
        ttk.Button(toolbar, text="打开区块画布", command=self._open_selected_block).pack(
            side="left", padx=4
        )
        ttk.Label(
            toolbar,
            text="区块按顺序执行；勾选 Python 编排后由脚本控制",
            foreground="#64748b",
        ).pack(side="right")

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True)
        self.block_canvas = tk.Canvas(
            canvas_frame,
            background="#e7edf5",
            highlightthickness=0,
            scrollregion=(0, 0, 3200, 700),
            height=330,
        )
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.block_canvas.xview)
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.block_canvas.yview)
        self.block_canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.block_canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.block_canvas.bind("<ButtonPress-1>", self._project_canvas_press)
        self.block_canvas.bind("<B1-Motion>", self._project_canvas_drag)
        self.block_canvas.bind("<ButtonRelease-1>", self._project_canvas_release)
        self.block_canvas.bind("<Double-Button-1>", self._project_canvas_double)
        self.project_status_var = tk.StringVar(value="就绪")
        ttk.Label(parent, textvariable=self.project_status_var, foreground="#526173").pack(
            fill="x", pady=(4, 0)
        )

    def _build_project_right(self, parent):
        settings = ttk.LabelFrame(parent, text="项目设置", padding=8)
        settings.pack(fill="x", pady=(0, 6))
        self.project_mode_var = tk.StringVar(value="sequence")
        self.stop_on_error_var = tk.BooleanVar(value=True)
        self.block_steps_var = tk.StringVar(value="100")
        self.max_runs_var = tk.StringVar(value="200")
        self.script_timeout_var = tk.StringVar(value="3600")
        ttk.Radiobutton(
            settings,
            text="按区块顺序执行",
            value="sequence",
            variable=self.project_mode_var,
            command=self._mark_project_dirty,
        ).pack(anchor="w")
        ttk.Radiobutton(
            settings,
            text="使用 Python 编排",
            value="python",
            variable=self.project_mode_var,
            command=self._mark_project_dirty,
        ).pack(anchor="w")
        ttk.Checkbutton(
            settings,
            text="区块未完成时停止",
            variable=self.stop_on_error_var,
        ).pack(anchor="w", pady=(4, 0))
        for label, var in (
            ("单区块最大步数", self.block_steps_var),
            ("最多区块执行次数", self.max_runs_var),
            ("脚本超时（秒）", self.script_timeout_var),
        ):
            row = ttk.Frame(settings)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=var, width=9).pack(side="right")
        for var in (self.stop_on_error_var, self.block_steps_var, self.max_runs_var, self.script_timeout_var):
            var.trace_add("write", self._mark_project_dirty)

        info = ttk.LabelFrame(parent, text="当前区块", padding=8)
        info.pack(fill="x", pady=(0, 6))
        self.block_info_var = tk.StringVar(value="未选择区块")
        ttk.Label(info, textvariable=self.block_info_var, wraplength=270, justify="left").pack(
            anchor="w"
        )
        row = ttk.Frame(info)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="上移", width=6, command=lambda: self._move_block(-1)).pack(
            side="left", padx=(0, 3)
        )
        ttk.Button(row, text="下移", width=6, command=lambda: self._move_block(1)).pack(
            side="left", padx=3
        )
        ttk.Button(row, text="移除", width=6, command=self._remove_block).pack(
            side="left", padx=3
        )
        ttk.Button(row, text="打开", width=6, command=self._open_selected_block).pack(
            side="left", padx=3
        )

        alias_frame = ttk.LabelFrame(parent, text="Python 可用变量", padding=8)
        alias_frame.pack(fill="both", expand=True)
        self.alias_help = ttk.Label(
            alias_frame,
            text="添加区块后，区块名会自动变成可调用的 Python 变量。",
            wraplength=270,
            justify="left",
            foreground="#334155",
        )
        self.alias_help.pack(anchor="w")

    def _build_project_bottom(self, parent):
        paned = ttk.PanedWindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=(6, 0))
        code_frame = ttk.LabelFrame(paned, text="Python 编排（区块名 = 可调用变量）", padding=6)
        vars_frame = ttk.LabelFrame(paned, text="变量", padding=6)
        paned.add(code_frame, weight=4)
        paned.add(vars_frame, weight=1)

        self.python_text = tk.Text(
            code_frame, height=10, wrap="none", undo=True, font=("Consolas", 10)
        )
        code_scroll_y = ttk.Scrollbar(code_frame, orient="vertical", command=self.python_text.yview)
        code_scroll_x = ttk.Scrollbar(code_frame, orient="horizontal", command=self.python_text.xview)
        self.python_text.configure(
            yscrollcommand=code_scroll_y.set, xscrollcommand=code_scroll_x.set
        )
        self.python_text.grid(row=0, column=0, sticky="nsew")
        code_scroll_y.grid(row=0, column=1, sticky="ns")
        code_scroll_x.grid(row=1, column=0, sticky="ew")
        code_frame.rowconfigure(0, weight=1)
        code_frame.columnconfigure(0, weight=1)
        self.python_text.bind("<<Modified>>", self._on_project_code_modified)

        code_buttons = ttk.Frame(code_frame)
        code_buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(code_buttons, text="保存项目", command=self._save_project).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(code_buttons, text="生成顺序脚本", command=self._generate_sequence_script).pack(
            side="left", padx=4
        )
        ttk.Button(code_buttons, text="检查语法", command=self._check_project_code).pack(
            side="left", padx=4
        )
        ttk.Label(
            code_buttons,
            text="示例：if 首页(): 点餐()   /   while 次数 < 3: ...",
            foreground="#64748b",
        ).pack(side="right")

        self.var_tree = ttk.Treeview(
            vars_frame, columns=("name", "value"), show="headings", height=8
        )
        self.var_tree.heading("name", text="变量名")
        self.var_tree.heading("value", text="值")
        self.var_tree.column("name", width=100, anchor="w")
        self.var_tree.column("value", width=120, anchor="w")
        self.var_tree.pack(fill="both", expand=True)
        self.var_tree.bind("<Double-1>", lambda _event: self._edit_variable())
        var_buttons = ttk.Frame(vars_frame)
        var_buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(var_buttons, text="添加", command=self._add_variable).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(var_buttons, text="编辑", command=self._edit_variable).pack(
            side="left", padx=4
        )
        ttk.Button(var_buttons, text="删除", command=self._delete_variable).pack(
            side="left", padx=4
        )

    def refresh_projects(self, keep_selection: bool = True, select_id: int | None = None):
        self._flush_project()
        self.projects = self.db.list_projects()
        self.project_list.delete(0, tk.END)
        for project in self.projects:
            self.project_list.insert(tk.END, project.name)
        index = None
        if select_id is not None:
            index = next((i for i, p in enumerate(self.projects) if p.id == select_id), None)
        elif keep_selection and self.project_id is not None:
            index = next((i for i, p in enumerate(self.projects) if p.id == self.project_id), None)
        if index is None and self.projects:
            index = 0
        self.refresh_block_library()
        if index is not None:
            self._project_loading = True
            self.project_list.selection_set(index)
            self.project_list.activate(index)
            self.project_list.see(index)
            self._project_loading = False
            self._load_project(self.projects[index].id)
        else:
            self.project_id = None
            self.project_blocks = []
            self.selected_block_id = None
            self.block_canvas.delete("all")
            self.python_text.delete("1.0", tk.END)
            self.var_tree.delete(*self.var_tree.get_children())
            self.project_status_var.set("还没有项目")
            self.alias_help.configure(text="新建项目后，把流程添加为区块。")

    def refresh_block_library(self):
        if not hasattr(self, "block_library"):
            return
        flows = self.db.list_flows()
        self.block_library.delete(0, tk.END)
        for flow in flows:
            self.block_library.insert(tk.END, flow.name)

    def _on_project_select(self, _event=None):
        if self._project_loading:
            return
        selection = self.project_list.curselection()
        if not selection:
            return
        project = self.projects[selection[0]]
        if project.id == self.project_id:
            return
        self._flush_project()
        self.project_undo.clear()
        self._load_project(project.id)

    def _load_project(self, project_id: int):
        project = self.db.get_project(project_id)
        if project is None:
            return
        self.project_id = project.id
        self._project_loading = True
        self.project_mode_var.set(str(project.settings.get("mode", "sequence")))
        self.stop_on_error_var.set(bool(project.settings.get("stop_on_error", True)))
        self.block_steps_var.set(str(project.settings.get("block_max_steps", 100)))
        self.max_runs_var.set(str(project.settings.get("max_block_runs", 200)))
        self.script_timeout_var.set(str(project.settings.get("script_timeout", 3600)))
        self.project_vars = dict(project.variables)
        self.python_text.delete("1.0", tk.END)
        if project.python_code:
            self.python_text.insert("1.0", project.python_code)
        self.python_text.edit_modified(False)
        self._refresh_var_tree()
        self.project_blocks = self.db.list_project_blocks(project.id)
        self.selected_block_id = self.project_blocks[0].id if self.project_blocks else None
        self._project_loading = False
        self._project_dirty = False
        self._draw_project_canvas()
        self._update_block_info()
        self._update_alias_help()
        self.project_status_var.set(
            f"项目：{project.name} · {len(self.project_blocks)} 个区块"
        )

    def _mark_project_dirty(self, *_args):
        if self._project_loading:
            return
        self._project_dirty = True

    def _on_project_code_modified(self, _event=None):
        if self._project_loading:
            return
        self.python_text.edit_modified(False)
        self._project_dirty = True

    def _flush_project(self):
        if self._project_dirty and self.project_id is not None:
            self._save_project(silent=True)

    def _save_project(self, silent: bool = False):
        if self.project_id is None:
            return
        project = self.db.get_project(self.project_id)
        if project is None:
            return
        if silent and not self._project_dirty:
            return
        if not silent:
            self._push_project_undo("保存项目")
        settings = dict(project.settings or {})
        settings["mode"] = self.project_mode_var.get()
        settings["stop_on_error"] = bool(self.stop_on_error_var.get())
        try:
            settings["block_max_steps"] = int(float(self.block_steps_var.get()))
        except ValueError:
            settings["block_max_steps"] = 100
        try:
            settings["max_block_runs"] = int(float(self.max_runs_var.get()))
        except ValueError:
            settings["max_block_runs"] = 200
        try:
            settings["script_timeout"] = float(self.script_timeout_var.get())
        except ValueError:
            settings["script_timeout"] = 3600
        self.db.update_project(
            project.id,
            python_code=self.python_text.get("1.0", "end").rstrip("\n"),
            variables=self.project_vars,
            settings=settings,
        )
        self._project_dirty = False
        if not silent:
            self.project_status_var.set("项目已保存")
            self._notify_other_views()

    def _push_project_undo(self, label: str) -> None:
        if self.project_id is not None:
            self.project_undo.push(label, snapshot_project(self.db, self.project_id))

    def _undo_project(self):
        item = self.project_undo.pop()
        if not item:
            self.project_status_var.set("没有可撤回的操作")
            return
        label, snapshot = item
        if self.project_id is None:
            return
        self._project_dirty = False
        restore_project(self.db, self.project_id, snapshot)
        self.refresh_projects(select_id=self.project_id)
        self.project_status_var.set(f"已撤回：{label}")

    def _add_block_from_library(self):
        self._flush_project()
        if self.project_id is None:
            messagebox.showinfo("提示", "请先新建并选择一个项目", parent=self)
            return
        selection = self.block_library.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先在区块库中选择一个流程", parent=self)
            return
        flows = self.db.list_flows()
        if selection[0] >= len(flows):
            return
        flow = flows[selection[0]]
        self._push_project_undo("添加区块")
        try:
            self.db.add_project_block(self.project_id, flow.id)
        except Exception as exc:
            messagebox.showerror("添加失败", str(exc), parent=self)
            return
        self._load_project(self.project_id)
        self.project_status_var.set(f"已添加区块：{flow.name}")

    def _selected_block(self) -> ProjectBlock | None:
        return next(
            (block for block in self.project_blocks if block.id == self.selected_block_id),
            None,
        )

    def _remove_block(self):
        self._flush_project()
        block = self._selected_block()
        if block is None:
            return
        if not messagebox.askyesno(
            "确认", f"从项目中移除区块「{block.flow_name}」？\n流程本身不会被删除。", parent=self
        ):
            return
        self._push_project_undo("移除区块")
        self.db.remove_project_block(block.id)
        self._load_project(self.project_id)
        self.project_status_var.set("区块已移除")

    def _move_block(self, direction: int):
        self._flush_project()
        block = self._selected_block()
        if block is None:
            return
        ids = [item.id for item in self.project_blocks]
        index = ids.index(block.id)
        target = index + direction
        if not (0 <= target < len(ids)):
            return
        ids[index], ids[target] = ids[target], ids[index]
        self._push_project_undo("移动区块")
        self.db.reorder_project_blocks(self.project_id, ids)
        self.selected_block_id = block.id
        self._load_project(self.project_id)
        self.selected_block_id = block.id
        self._draw_project_canvas()
        self._update_block_info()

    def _move_block_to(self, block_id: int, target: int):
        self._flush_project()
        ids = [item.id for item in self.project_blocks]
        if block_id not in ids:
            return
        current = ids.index(block_id)
        target = max(0, min(len(ids) - 1, target))
        if current == target:
            self._draw_project_canvas()
            return
        ids.pop(current)
        ids.insert(target, block_id)
        self._push_project_undo("拖动区块")
        self.db.reorder_project_blocks(self.project_id, ids)
        self.selected_block_id = block_id
        self._load_project(self.project_id)
        self.selected_block_id = block_id
        self._draw_project_canvas()
        self._update_block_info()
        self.project_status_var.set("区块顺序已更新")

    def _open_selected_block(self):
        block = self._selected_block()
        if block is None:
            return
        self._flush_project()
        self.main_notebook.select(self.flow_tab)
        self.refresh_flows(select_id=block.flow_id)

    def _draw_project_canvas(self):
        self.block_canvas.delete("all")
        card_w, card_h, gap = 230, 130, 70
        if not self.project_blocks:
            self.block_canvas.create_text(
                360,
                150,
                text="还没有区块\n在左侧「区块库」选择流程，点击「添加为区块」",
                fill="#64748b",
                justify="center",
                font=("TkDefaultFont", 13),
            )
            return
        for index, block in enumerate(self.project_blocks):
            x = 40 + index * (card_w + gap)
            y = 70
            selected = block.id == self.selected_block_id
            tags = ("pblock", f"pblock:{block.id}")
            self.block_canvas.create_rectangle(
                x,
                y,
                x + card_w,
                y + card_h,
                fill="#ffffff",
                outline="#1c64d9" if selected else "#9fb2c9",
                width=3 if selected else 1,
                tags=tags,
            )
            self.block_canvas.create_rectangle(
                x + 1, y + 1, x + card_w - 1, y + 30, fill="#dbe9ff", outline="", tags=tags
            )
            self.block_canvas.create_text(
                x + 10,
                y + 8,
                text=f"{index + 1}",
                fill="#1c64d9",
                font=("TkDefaultFont", 11, "bold"),
                tags=tags,
            )
            self.block_canvas.create_text(
                x + 34,
                y + 8,
                text=block.flow_name[:14],
                anchor="nw",
                fill="#1f2d3d",
                font=("TkDefaultFont", 10, "bold"),
                tags=tags,
            )
            self.block_canvas.create_text(
                x + 12,
                y + 44,
                text=f"变量名：{block.alias}",
                anchor="nw",
                fill="#1c64d9",
                font=("TkDefaultFont", 9),
                tags=tags,
            )
            self.block_canvas.create_text(
                x + 12,
                y + 70,
                text=f"节点：{block.node_count} 个",
                anchor="nw",
                fill="#526173",
                font=("TkDefaultFont", 9),
                tags=tags,
            )
            self.block_canvas.create_text(
                x + 12,
                y + 94,
                text="拖动排序 · 双击打开",
                anchor="nw",
                fill="#94a3b8",
                font=("TkDefaultFont", 8),
                tags=tags,
            )
            if index < len(self.project_blocks) - 1:
                nx = x + card_w
                ny = y + card_h // 2
                self.block_canvas.create_line(
                    nx + 10,
                    ny,
                    nx + gap - 10,
                    ny,
                    fill="#7a8ba3",
                    width=2,
                    arrow=tk.LAST,
                    arrowshape=(10, 12, 4),
                    tags=("parrow",),
                )
        last_x = 40 + len(self.project_blocks) * (card_w + gap) + 80
        self.block_canvas.configure(scrollregion=(0, 0, max(1200, last_x), 600))

    def _update_block_info(self):
        block = self._selected_block()
        if block is None:
            self.block_info_var.set("未选择区块")
            return
        index = self.project_blocks.index(block)
        self.block_info_var.set(
            f"顺序：{index + 1} / {len(self.project_blocks)}\n"
            f"流程：{block.flow_name}\n"
            f"Python 变量名：{block.alias}\n"
            f"节点：{block.node_count} 个"
        )

    def _update_alias_help(self):
        if not self.project_blocks:
            self.alias_help.configure(text="添加区块后，区块名会自动变成可调用的 Python 变量。")
            return
        lines = ["可用变量（区块名）："]
        for block in self.project_blocks:
            lines.append(f"  {block.alias}  ←  {block.flow_name}")
        alias = self.project_blocks[0].alias
        lines += [
            "",
            "示例：",
            f"if {alias}():",
            "    变量['次数'] += 1",
            "",
            "循环示例：",
            "while 次数 < 3:",
            f"    {alias}()",
            "    次数 += 1",
        ]
        self.alias_help.configure(text="\n".join(lines))

    def _project_canvas_press(self, event):
        self.block_canvas.focus_set()
        item = self.block_canvas.find_withtag("current")
        tags = self.block_canvas.gettags(item) if item else ()
        block_id = _tag_id(tags, "pblock:")
        if block_id is None:
            return
        self.selected_block_id = block_id
        self._draw_project_canvas()
        self._update_block_info()
        self._drag_block = {
            "id": block_id,
            "last_x": event.x,
            "moved": False,
        }
        self.block_canvas.tag_raise(f"pblock:{block_id}")

    def _project_canvas_drag(self, event):
        if not self._drag_block:
            return
        dx = event.x - self._drag_block["last_x"]
        if not dx:
            return
        self._drag_block["last_x"] = event.x
        self._drag_block["moved"] = True
        self.block_canvas.move(f"pblock:{self._drag_block['id']}", dx, 0)

    def _project_canvas_release(self, _event):
        if not self._drag_block:
            return
        drag = self._drag_block
        self._drag_block = None
        if not drag["moved"]:
            return
        bbox = self.block_canvas.bbox(f"pblock:{drag['id']}")
        if not bbox:
            self._draw_project_canvas()
            return
        center = (bbox[0] + bbox[2]) / 2
        card_w, gap = 230, 70
        target = round((center - (40 + card_w / 2)) / (card_w + gap))
        self._move_block_to(drag["id"], target)

    def _project_canvas_double(self, _event):
        block = self._selected_block()
        if block:
            self._open_selected_block()

    def _refresh_var_tree(self):
        self.var_tree.delete(*self.var_tree.get_children())
        for name, value in self.project_vars.items():
            self.var_tree.insert("", "end", iid=str(name), values=(name, format_literal(value)))

    def _add_variable(self):
        if self.project_id is None:
            return
        dialog = VariableDialog(self)
        self.wait_window(dialog)
        if not dialog.result:
            return
        name = dialog.result["name"]
        if name in self.project_vars:
            messagebox.showwarning("提示", "变量已存在", parent=self)
            return
        self._push_project_undo("添加变量")
        self.project_vars[name] = parse_literal(dialog.result["value"])
        self.db.update_project(self.project_id, variables=self.project_vars)
        self._refresh_var_tree()
        self._update_alias_help()

    def _edit_variable(self):
        selection = self.var_tree.selection()
        if not selection or self.project_id is None:
            return
        old_name = selection[0]
        dialog = VariableDialog(self, old_name, format_literal(self.project_vars.get(old_name)))
        self.wait_window(dialog)
        if not dialog.result:
            return
        new_name = dialog.result["name"]
        if new_name != old_name and new_name in self.project_vars:
            messagebox.showwarning("提示", "变量名已存在", parent=self)
            return
        self._push_project_undo("编辑变量")
        self.project_vars.pop(old_name, None)
        self.project_vars[new_name] = parse_literal(dialog.result["value"])
        self.db.update_project(self.project_id, variables=self.project_vars)
        self._refresh_var_tree()
        self._update_alias_help()

    def _delete_variable(self):
        selection = self.var_tree.selection()
        if not selection or self.project_id is None:
            return
        name = selection[0]
        if not messagebox.askyesno("确认", f"删除变量「{name}」？", parent=self):
            return
        self._push_project_undo("删除变量")
        self.project_vars.pop(name, None)
        self.db.update_project(self.project_id, variables=self.project_vars)
        self._refresh_var_tree()
        self._update_alias_help()

    def _generate_sequence_script(self):
        if not self.project_blocks:
            messagebox.showinfo("提示", "先添加区块", parent=self)
            return
        current = self.python_text.get("1.0", "end").strip()
        if current and not messagebox.askyesno(
            "确认", "当前脚本会被覆盖，继续生成顺序脚本？", parent=self
        ):
            return
        lines = ["# 自动生成：按区块顺序执行，可继续添加 if / while", ""]
        for block in self.project_blocks:
            lines.append(f"{block.alias}()")
        self.python_text.delete("1.0", tk.END)
        self.python_text.insert("1.0", "\n".join(lines) + "\n")
        self.project_mode_var.set("python")
        self._project_dirty = True
        self.project_status_var.set("已生成顺序脚本，保存后生效")

    def _check_project_code(self):
        code = self.python_text.get("1.0", "end")
        try:
            compile(code, "<项目编排>", "exec")
        except SyntaxError as exc:
            messagebox.showerror(
                "语法错误",
                f"第 {exc.lineno} 行：{exc.msg}\n{exc.text or ''}",
                parent=self,
            )
        else:
            messagebox.showinfo("检查通过", "Python 语法正确。", parent=self)

    def _new_project(self):
        dialog = FlowDialog(self, "新建项目")
        self.wait_window(dialog)
        if not dialog.result:
            return
        name = self.db.unique_project_name(dialog.result["name"])
        project_id = self.db.create_project(name, dialog.result["description"])
        self.refresh_projects(select_id=project_id)
        self._notify_other_views()
        self.project_status_var.set("项目已创建")

    def _delete_project(self):
        if self.project_id is None:
            return
        project = self.db.get_project(self.project_id)
        if project is None:
            return
        if not messagebox.askyesno(
            "确认删除", f"删除项目「{project.name}」？\n流程区块本身不会被删除。", parent=self
        ):
            return
        self.project_undo.clear()
        self.project_id = None
        self.db.delete_project(project.id)
        self.refresh_projects(keep_selection=False)
        self._notify_other_views()

    def _refresh_project_after_flow_change(self):
        self.refresh_block_library()
        if self.project_id is not None:
            self.project_blocks = self.db.list_project_blocks(self.project_id)
            ids = {block.id for block in self.project_blocks}
            if self.selected_block_id not in ids:
                self.selected_block_id = self.project_blocks[0].id if self.project_blocks else None
            self._draw_project_canvas()
            self._update_block_info()
            self._update_alias_help()

    def _push_flow_undo(self, label: str) -> None:
        if self.flow_id is None:
            return
        self.flow_undo.push(label, snapshot_flow(self.db, self.flow_id))

    def _undo_flow_action(self):
        if self.flow_id is None:
            return
        if self._save_job is not None:
            try:
                self.after_cancel(self._save_job)
            except Exception:
                pass
            self._save_job = None
        self._dirty = False
        item = self.flow_undo.pop()
        if not item:
            self._status("没有可撤回的操作")
            return
        label, snapshot = item
        try:
            restore_flow(self.db, self.flow_id, snapshot)
        except Exception as exc:
            messagebox.showerror("撤回失败", str(exc), parent=self)
            return
        self._node_edit_open = False
        flow = self.db.get_flow(self.flow_id)
        if flow is not None:
            self._load_settings(flow)
            self._load_flow_info(flow)
        keep = self.selected_node_id
        self._reload_graph()
        if keep in self.nodes_by_id:
            self._select_node(keep)
        else:
            self.selected_node_id = None
            self._clear_inspector()
            self._draw_graph()
        self._status(f"已撤回：{label}")

    def _cancel_connection(self):
        edge_id = self.selected_edge_id
        if edge_id is None:
            selection = self.outgoing.selection()
            if selection:
                edge_id = int(selection[0])
        if edge_id is None:
            messagebox.showinfo(
                "提示",
                "请先在画布上单击一条连线，或在「下一步 / 偏好」列表里选中一条连线。",
                parent=self,
            )
            return
        self._push_flow_undo("取消连线")
        self.db.delete_edge(edge_id)
        self.selected_edge_id = None
        self._reload_graph()
        self._status("已取消该连线 / 下一步")


    def open_project(self, project_id: int):
        self.main_notebook.select(self.project_tab)
        self.refresh_projects(select_id=project_id)

    def open_flow(self, flow_id: int):
        self.main_notebook.select(self.flow_tab)
        self.refresh_flows(select_id=flow_id)
