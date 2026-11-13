from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from ..db import Database
from ..device import DeviceController, DryRunDevice, list_adb_devices
from ..engine import RunEngine
from ..project_runner import ProjectRunner
from .widgets import ImageCanvas

CAPTURE_CHOICES = {"scrcpy 窗口": "scrcpy", "ADB 截图": "adb"}
INPUT_CHOICES = {"ADB 输入": "adb", "scrcpy 窗口输入": "scrcpy"}


class RunView(ttk.Frame):
    def __init__(self, master, app, db: Database):
        super().__init__(master, padding=10)
        self.app = app
        self.db = db
        self.queue: queue.Queue = queue.Queue()
        self.engine: RunEngine | None = None
        self.device: DeviceController | None = None
        self.worker: threading.Thread | None = None
        self.running = False
        self.stop_requested = False
        self._warn_count = 0
        self._log_count = 0
        self._build()
        self.refresh_flows()
        self.after(300, self.refresh_devices)
        self.after(120, self._poll)

    def _build(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(toolbar, text="运行对象").pack(side="left")
        self.flow_combo = ttk.Combobox(toolbar, width=30, state="readonly")
        self.flow_combo.pack(side="left", padx=(4, 8))
        ttk.Button(toolbar, text="刷新", width=5, command=self.refresh_flows).pack(side="left")
        ttk.Label(toolbar, text="设备").pack(side="left", padx=(16, 0))
        self.device_combo = ttk.Combobox(toolbar, width=30, state="readonly")
        self.device_combo.pack(side="left", padx=(4, 4))
        ttk.Button(toolbar, text="刷新", width=5, command=self.refresh_devices).pack(side="left")
        ttk.Label(toolbar, text="画面来源").pack(side="left", padx=(16, 0))
        self.capture_combo = ttk.Combobox(
            toolbar, values=list(CAPTURE_CHOICES), width=12, state="readonly"
        )
        self.capture_combo.set("scrcpy 窗口")
        self.capture_combo.pack(side="left", padx=4)
        ttk.Label(toolbar, text="输入方式").pack(side="left", padx=(12, 0))
        self.input_combo = ttk.Combobox(
            toolbar, values=list(INPUT_CHOICES), width=15, state="readonly"
        )
        self.input_combo.set("ADB 输入")
        self.input_combo.pack(side="left", padx=4)

        options = ttk.Frame(self)
        options.pack(fill="x", pady=(0, 8))
        self.manage_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="启动 scrcpy 窗口", variable=self.manage_var).pack(
            side="left"
        )
        self.dry_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="模拟运行（不点击手机）", variable=self.dry_var).pack(
            side="left", padx=(16, 0)
        )
        ttk.Label(options, text="最大步数").pack(side="left", padx=(16, 0))
        self.max_steps_var = tk.StringVar(value="300")
        ttk.Spinbox(
            options, from_=1, to=100000, textvariable=self.max_steps_var, width=8
        ).pack(side="left", padx=4)
        self.start_button = ttk.Button(options, text="开始运行", command=self.start)
        self.start_button.pack(side="right", padx=4)
        self.stop_button = ttk.Button(
            options, text="停止", command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="right", padx=4)

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.LabelFrame(paned, text="实时画面", padding=6)
        self.preview = ImageCanvas(left, width=380, height=760, mode="view")
        self.preview.pack(fill="both", expand=True)
        self.state_label = ttk.Label(left, text="尚未开始", anchor="w")
        self.state_label.pack(fill="x", pady=(6, 0))
        paned.add(left, weight=1)

        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        candidates_frame = ttk.LabelFrame(right, text="本步候选（下一步可能性）", padding=6)
        candidates_frame.pack(fill="both", expand=False, pady=(0, 8))
        columns = ("node", "sim", "roi", "bonus", "score")
        self.candidates = ttk.Treeview(
            candidates_frame, columns=columns, show="headings", height=6
        )
        headings = {
            "node": ("节点", 180),
            "sim": ("综合相似度", 90),
            "roi": ("检测重点相似度", 110),
            "bonus": ("路径偏好", 80),
            "score": ("最终得分", 80),
        }
        for key, (title, width) in headings.items():
            self.candidates.heading(key, text=title)
            self.candidates.column(key, width=width, anchor="center")
        self.candidates.pack(fill="x")

        logs_frame = ttk.LabelFrame(right, text="运行日志", padding=6)
        logs_frame.pack(fill="both", expand=True)
        log_columns = ("idx", "time", "kind", "node", "sim", "action", "result")
        self.logs = ttk.Treeview(logs_frame, columns=log_columns, show="headings")
        log_headings = {
            "idx": ("步", 50),
            "time": ("时间", 70),
            "kind": ("类型", 70),
            "node": ("状态/节点", 160),
            "sim": ("相似度", 70),
            "action": ("动作", 240),
            "result": ("结果", 120),
        }
        for key, (title, width) in log_headings.items():
            self.logs.heading(key, text=title)
            self.logs.column(key, width=width, anchor="w")
        self.logs.tag_configure("match", foreground="#2f9e44")
        self.logs.tag_configure("no_match", foreground="#e67700")
        self.logs.tag_configure("fallback", foreground="#9c36b5")
        self.logs.tag_configure("error", foreground="#c92a2a")
        self.logs.tag_configure("info", foreground="#495057")
        log_scroll = ttk.Scrollbar(logs_frame, orient="vertical", command=self.logs.yview)
        self.logs.configure(yscrollcommand=log_scroll.set)
        self.logs.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self.session_label = ttk.Label(self, text="", anchor="w")
        self.session_label.pack(fill="x", pady=(6, 0))

    def refresh_flows(self):
        flows = self.db.list_flows()
        projects = self.db.list_projects()
        self.target_map: dict[str, tuple[str, int]] = {}
        values = []
        for project in projects:
            label = f"项目：{project.name}"
            self.target_map[label] = ("project", project.id)
            values.append(label)
        for flow in flows:
            label = f"流程：{flow.name}"
            self.target_map[label] = ("flow", flow.id)
            values.append(label)
        current = self.flow_combo.get()
        self.flow_combo["values"] = values
        if current in self.target_map:
            self.flow_combo.set(current)
        elif values:
            self.flow_combo.current(0)
        else:
            self.flow_combo.set("")

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

    def _selected_target(self):
        value = self.flow_combo.get()
        return self.target_map.get(value) if value else None

    def start(self):
        if self.running:
            return
        try:
            self.app.edit_view.flush()
        except Exception:
            pass
        target = self._selected_target()
        if target is None:
            messagebox.showwarning("提示", "请先选择流程或区块项目", parent=self)
            return
        kind, target_id = target
        if kind == "flow":
            if not self.db.list_nodes(target_id):
                messagebox.showwarning("提示", "该流程还没有步骤节点", parent=self)
                return
        else:
            if not self.db.list_project_blocks(target_id):
                messagebox.showwarning("提示", "该项目还没有区块", parent=self)
                return
        try:
            max_steps = int(self.max_steps_var.get())
        except ValueError:
            max_steps = 300
        capture = CAPTURE_CHOICES.get(self.capture_combo.get(), "scrcpy")
        input_mode = INPUT_CHOICES.get(self.input_combo.get(), "adb")
        self.logs.delete(*self.logs.get_children())
        self.candidates.delete(*self.candidates.get_children())
        self._log_count = 0
        self._warn_count = 0
        self.stop_requested = False
        self._set_running(True)
        self.state_label.configure(text="正在连接设备…")
        self.session_label.configure(text="")
        serial = self._selected_serial()
        manage = self.manage_var.get()
        dry = self.dry_var.get()
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(kind, target_id, serial, capture, input_mode, manage, dry, max_steps),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, kind, target_id, serial, capture, input_mode, manage, dry, max_steps):
        device = None
        try:
            device = DeviceController(
                serial=serial,
                capture=capture,
                input_mode=input_mode,
                manage_scrcpy=manage,
            )
            device.start()
            for warning in device.warnings:
                self.queue.put({"type": "warn", "message": warning})
            self.device = device
            inner = DryRunDevice(device) if dry else device
            if kind == "project":
                runner = ProjectRunner(
                    self.db, target_id, inner, event_cb=self.queue.put
                )
                self.engine = runner
                if self.stop_requested:
                    runner.stop()
                runner.run()
            else:
                engine = RunEngine(
                    self.db, target_id, inner, event_cb=self.queue.put, max_steps=max_steps
                )
                self.engine = engine
                if self.stop_requested:
                    engine.stop()
                engine.run()
        except Exception as exc:
            self.queue.put({"type": "error", "message": str(exc)})
        finally:
            if device is not None:
                try:
                    device.stop()
                except Exception:
                    pass
            self.device = None
            self.queue.put({"type": "worker_done"})

    def stop(self):
        self.stop_requested = True
        if self.engine is not None:
            self.engine.stop()
        self.state_label.configure(text="正在停止…")

    def _set_running(self, running: bool):
        self.running = running
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        state = "disabled" if running else "readonly"
        self.flow_combo.configure(state=state)
        self.device_combo.configure(state=state)
        self.capture_combo.configure(state=state)
        self.input_combo.configure(state=state)

    def _poll(self):
        try:
            for _ in range(80):
                event = self.queue.get_nowait()
                self._handle(event)
        except queue.Empty:
            pass
        device = self.device
        if device is not None:
            warnings = list(device.warnings)
            while self._warn_count < len(warnings):
                self._log("提示", "", None, warnings[self._warn_count], "", "info")
                self._warn_count += 1
        self.after(120, self._poll)

    def _handle(self, event: dict):
        kind = event.get("type")
        if kind == "screenshot":
            self.preview.set_source(event.get("frame"))
        elif kind == "project_start":
            mode = "Python 编排" if event.get("mode") == "python" else "顺序执行"
            self.state_label.configure(text=f"项目运行中 · {event.get('name', '')}")
            self.session_label.configure(
                text=f"项目会话 {event.get('run_id')} · {len(event.get('blocks') or [])} 个区块 · {mode}"
            )
            self._log(
                "项目开始",
                "",
                None,
                f"{event.get('name', '')} · {mode}",
                f"{len(event.get('blocks') or [])} 个区块",
                "info",
            )
        elif kind == "block_start":
            self.state_label.configure(
                text=f"区块：{event.get('name', '')} · 第 {event.get('run_index')}/{event.get('max_runs')} 次"
            )
            self._log(
                "区块开始",
                event.get("name", ""),
                None,
                f"变量名 {event.get('alias', '')}",
                f"{event.get('run_index')}/{event.get('max_runs')}",
                "info",
            )
        elif kind == "block_finish":
            result = event.get("result")
            if result is not None:
                tag = "match" if result.ok else "error"
                self._log(
                    "区块结束",
                    result.name,
                    None,
                    f"状态 {result.status} · 步数 {result.steps} · 保底 {result.fallbacks}",
                    result.error or "",
                    tag,
                )
                self.state_label.configure(
                    text=f"区块 {result.name} 结束：{result.status} · 步数 {result.steps}"
                )
        elif kind == "project_log":
            self._log("脚本", "", None, event.get("message", ""), "", "info")
        elif kind == "project_finish":
            status = event.get("status")
            labels = {
                "finished": "项目完成",
                "stopped": "项目已停止",
                "error": "项目出错停止",
            }
            self.session_label.configure(
                text=f"项目会话 {event.get('run_id')} · {labels.get(status, status)} · "
                f"区块 {event.get('blocks')} · 总步数 {event.get('steps')} · "
                f"保底 {event.get('fallbacks')}"
            )
            self._log(
                "项目结束",
                "",
                None,
                labels.get(status, str(status)),
                event.get("error") or "",
                "error" if status == "error" else "info",
            )
            self.state_label.configure(text=labels.get(status, str(status)))
        elif kind == "start":
            self.state_label.configure(
                text=f"运行中 · {event.get('flow', '')} · 会话 {event.get('session_id', '')}"
            )
            self._log("开始", "", None, "开始运行", f"会话 {event.get('session_id')}", "info")
        elif kind == "match":
            match = event.get("match")
            if match is not None:
                self._fill_candidates(event.get("matches") or [])
            block = event.get("block")
            state_name = event.get("state_name", "")
            label = f"{block} / {state_name}" if block else state_name
            prefix = f"区块 {block} · " if block else ""
            self.state_label.configure(
                text=f"{prefix}当前状态：{state_name} · 相似度 {event.get('similarity', 0):.3f}"
            )
            self._log(
                "匹配",
                label,
                event.get("similarity"),
                event.get("action", ""),
                "成功",
                "match",
            )
        elif kind == "no_match":
            self._fill_candidates(event.get("matches") or [])
            prefix = f"区块 {event.get('block')} · " if event.get("block") else ""
            self.state_label.configure(
                text=f"{prefix}未匹配 · 重试 {event.get('retries')}/{event.get('max_retries')}"
            )
            self._log(
                "未匹配",
                "",
                event.get("similarity"),
                "等待画面稳定",
                f"重试 {event.get('retries')}/{event.get('max_retries')}",
                "no_match",
            )
        elif kind == "fallback":
            self._log(
                "保底",
                "",
                None,
                "执行保底动作",
                f"{event.get('index')}/{event.get('max_fallbacks')}",
                "fallback",
            )
            prefix = f"区块 {event.get('block')} · " if event.get("block") else ""
            self.state_label.configure(
                text=f"{prefix}连续未匹配，执行保底 {event.get('index')}/{event.get('max_fallbacks')}"
            )
        elif kind == "warn":
            message = event.get("message", "")
            self.session_label.configure(text=message)
            self._log("提示", "", None, message, "", "info")
        elif kind == "error":
            message = event.get("message", "")
            prefix = f"区块 {event.get('block')} · " if event.get("block") else ""
            self.state_label.configure(text=f"{prefix}错误：{message}")
            self._log("错误", event.get("block") or "", None, message, "停止", "error")
        elif kind == "end":
            prefix = f"区块 {event.get('block')} · " if event.get("block") else ""
            self.state_label.configure(text=f"{prefix}到达结束节点")
        elif kind == "finish":
            status = event.get("status")
            labels = {
                "finished": "已完成",
                "stopped": "已停止",
                "error": "出错停止",
                "max_steps": "达到最大步数",
            }
            self.session_label.configure(
                text=f"会话 {event.get('session_id')} · {labels.get(status, status)} · "
                f"步数 {event.get('steps')} · 保底 {event.get('fallbacks')}"
            )
            self._log(
                "结束",
                "",
                None,
                labels.get(status, str(status)),
                event.get("error") or "",
                "error" if status == "error" else "info",
            )
        elif kind == "worker_done":
            self._set_running(False)
            self.engine = None
            self.app.refresh_history()

    def _fill_candidates(self, matches):
        self.candidates.delete(*self.candidates.get_children())
        for match in matches:
            roi = match.roi_similarity
            self.candidates.insert(
                "",
                "end",
                values=(
                    match.node.name,
                    f"{match.similarity:.3f}",
                    "-" if roi is None else f"{roi:.3f}",
                    f"{match.bonus:.2f}",
                    f"{match.score:.3f}",
                ),
            )

    def _log(self, kind, node, similarity, action, result, tag):
        self._log_count += 1
        if self._log_count > 1000:
            return
        self.logs.insert(
            "",
            0,
            values=(
                self._log_count,
                time.strftime("%H:%M:%S"),
                kind,
                node,
                "-" if similarity is None else f"{similarity:.3f}",
                action,
                result,
            ),
            tags=(tag,),
        )
        children = self.logs.get_children()
        if len(children) > 600:
            self.logs.delete(children[-1])
