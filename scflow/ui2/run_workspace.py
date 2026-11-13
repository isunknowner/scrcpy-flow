from __future__ import annotations

import queue
import threading
import time

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..db import Database
from ..device import DeviceController, DryRunDevice, list_adb_devices
from ..engine import RunEngine
from ..project_runner import ProjectRunner
from .scrollbar_bar import make_scrollable_bar
from .theme import GREEN, MUTED, ORANGE, RED

CAPTURE_CHOICES = {"ADB 截图": "adb", "scrcpy 窗口": "scrcpy"}
INPUT_CHOICES = {"ADB 输入": "adb", "scrcpy 窗口输入": "scrcpy"}


class RunWorkspace(QWidget):
    status_message = Signal(str)
    running_changed = Signal(bool)
    data_changed = Signal()

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.queue: queue.Queue = queue.Queue()
        self.engine = None
        self.device = None
        self.worker: threading.Thread | None = None
        self.running = False
        self.stop_requested = False
        self._frame: QImage | None = None
        self._pre_run_callback = None
        self._build()
        self.refresh_targets()
        self.after(300, self.refresh_devices)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(120)

    def after(self, ms, fn) -> None:
        QTimer.singleShot(ms, fn)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("运行对象"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(300)
        bar.addWidget(self.target_combo)
        self.refresh_target_btn = QPushButton("刷新")
        bar.addWidget(self.refresh_target_btn)
        bar.addWidget(QLabel("设备"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(220)
        bar.addWidget(self.device_combo)
        self.refresh_device_btn = QPushButton("刷新设备")
        bar.addWidget(self.refresh_device_btn)
        bar.addWidget(QLabel("画面"))
        self.capture_combo = QComboBox()
        self.capture_combo.addItems(list(CAPTURE_CHOICES))
        self.capture_combo.setCurrentText("ADB 截图")
        bar.addWidget(self.capture_combo)
        bar.addWidget(QLabel("输入"))
        self.input_combo = QComboBox()
        self.input_combo.addItems(list(INPUT_CHOICES))
        bar.addWidget(self.input_combo)
        bar.addStretch(1)
        layout.addWidget(make_scrollable_bar(bar))

        opts = QHBoxLayout()
        self.scrcpy_check = QCheckBox("启动 scrcpy 窗口")
        self.dry_check = QCheckBox("模拟运行（不点击手机）")
        opts.addWidget(self.scrcpy_check)
        opts.addWidget(self.dry_check)
        opts.addWidget(QLabel("最大步数"))
        self.max_steps = QSpinBox()
        self.max_steps.setRange(1, 100000)
        self.max_steps.setValue(300)
        opts.addWidget(self.max_steps)
        self.speed_label = QLabel("加速")
        opts.addWidget(self.speed_label)
        self.speed_combo = QComboBox()
        for label, value in (
            ("普通", 1.0),
            ("快速", 5.0),
            ("极速（跳过等待）", 0.0),
        ):
            self.speed_combo.addItem(label, value)
        self.speed_combo.setToolTip(
            "普通＝按原速度；快速＝等待缩到 1/5；极速＝跳过所有等待。\n"
            "滑动节点要连滑几次，在右侧检查器的「重复次数」里设置。"
        )
        opts.addWidget(self.speed_combo)
        self.manual_btn = QPushButton("人工选择")
        self.manual_btn.setToolTip("运行卡住时点这里：手动点击/按键/打开应用/滑动，可记住为下次的保底动作")
        self.manual_btn.setEnabled(False)
        self.manual_btn.clicked.connect(self.request_manual)
        opts.addWidget(self.manual_btn)
        opts.addStretch(1)
        self.start_btn = QPushButton("开始运行")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        opts.addWidget(self.start_btn)
        opts.addWidget(self.stop_btn)
        layout.addWidget(make_scrollable_bar(opts))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.preview = QLabel("等待运行")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#0d0f13; border:1px solid #333842; color:#5c6575;")
        self.preview.setMinimumSize(200, 200)
        left_layout.addWidget(self.preview, 1)
        self.state_label = QLabel("尚未开始")
        self.state_label.setObjectName("subtitle")
        left_layout.addWidget(self.state_label)
        self.session_label = QLabel("")
        self.session_label.setObjectName("subtitle")
        self.session_label.setWordWrap(True)
        left_layout.addWidget(self.session_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("本步候选（下一步可能性）"))
        self.candidates = QTableWidget(0, 5)
        self.candidates.setHorizontalHeaderLabels(
            ["节点", "综合相似度", "重点相似度", "路径偏好", "最终得分"]
        )
        self.candidates.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.candidates.verticalHeader().setVisible(False)
        self.candidates.setMaximumHeight(210)
        right_layout.addWidget(self.candidates)
        right_layout.addWidget(QLabel("运行日志"))
        self.logs = QTableWidget(0, 7)
        self.logs.setHorizontalHeaderLabels(
            ["步", "时间", "类型", "状态/节点", "相似度", "动作", "结果"]
        )
        self.logs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.logs.verticalHeader().setVisible(False)
        right_layout.addWidget(self.logs, 1)
        splitter.addWidget(right)
        splitter.setSizes([420, 780])

        self.refresh_target_btn.clicked.connect(self.refresh_targets)
        self.refresh_device_btn.clicked.connect(self.refresh_devices)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def set_simple(self, simple: bool) -> None:
        for widget in (
            self.capture_combo,
            self.input_combo,
            self.scrcpy_check,
            self.dry_check,
            self.max_steps,
            self.refresh_device_btn,
            self.device_combo,
            self.speed_label,
            self.speed_combo,
        ):
            widget.setVisible(not simple)

    def set_pre_run_callback(self, callback) -> None:
        self._pre_run_callback = callback

    def refresh_targets(self) -> None:
        current = self.target_combo.currentData()
        self.target_combo.clear()
        for project in self.db.list_projects():
            self.target_combo.addItem(f"项目：{project.name}", f"project:{project.id}")
        for flow in self.db.list_flows():
            self.target_combo.addItem(f"流程：{flow.name}", f"flow:{flow.id}")
        if current is not None:
            index = self.target_combo.findData(current)
            if index >= 0:
                self.target_combo.setCurrentIndex(index)

    def refresh_devices(self) -> None:
        current = self.device_combo.currentText()
        self.device_combo.clear()
        self.device_combo.addItem("（自动选择）", None)
        for device in list_adb_devices():
            if device["state"] == "device":
                model = f" · {device['model']}" if device["model"] else ""
                self.device_combo.addItem(f"{device['serial']}{model}", device["serial"])
        index = self.device_combo.findText(current)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)

    def _selected_target(self):
        data = self.target_combo.currentData()
        if not data:
            return None
        kind, _, raw = str(data).partition(":")
        try:
            return kind, int(raw)
        except ValueError:
            return None

    def select_target(self, kind: str, target_id: int) -> int:
        index = self.target_combo.findData(f"{kind}:{target_id}")
        if index >= 0:
            self.target_combo.setCurrentIndex(index)
        return index

    def _selected_serial(self):
        return self.device_combo.currentData()

    def start(self) -> None:
        if self.running:
            return
        if self._pre_run_callback:
            try:
                self._pre_run_callback()
            except Exception:
                pass
        target = self._selected_target()
        if not target:
            QMessageBox.warning(self, "提示", "请选择项目或流程")
            return
        kind, target_id = target
        if kind == "flow" and not self.db.list_nodes(target_id):
            QMessageBox.warning(self, "提示", "该流程还没有节点")
            return
        if kind == "project" and not self.db.list_project_blocks(target_id):
            QMessageBox.warning(self, "提示", "该项目还没有区块")
            return
        self.candidates.setRowCount(0)
        self.logs.setRowCount(0)
        self.stop_requested = False
        self._set_running(True)
        self.state_label.setText("正在连接设备…")
        serial = self._selected_serial()
        capture = CAPTURE_CHOICES.get(self.capture_combo.currentText(), "adb")
        input_mode = INPUT_CHOICES.get(self.input_combo.currentText(), "adb")
        manage = self.scrcpy_check.isChecked()
        dry = self.dry_check.isChecked()
        max_steps = self.max_steps.value()
        speed_value = float(self.speed_combo.currentData() or 1.0)
        skip_waits = speed_value <= 0
        speed = 1.0 if skip_waits else speed_value
        swipe_repeat = 1
        self.worker = threading.Thread(
            target=self._worker,
            args=(
                kind,
                target_id,
                serial,
                capture,
                input_mode,
                manage,
                dry,
                max_steps,
                speed,
                skip_waits,
                swipe_repeat,
            ),
            daemon=True,
        )
        self.worker.start()

    def _worker(
        self,
        kind,
        target_id,
        serial,
        capture,
        input_mode,
        manage,
        dry,
        max_steps,
        speed,
        skip_waits,
        swipe_repeat,
    ):
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
                    self.db,
                    target_id,
                    inner,
                    event_cb=self.queue.put,
                    speed=speed,
                    skip_waits=skip_waits,
                    swipe_repeat=swipe_repeat,
                )
                self.engine = runner
                if self.stop_requested:
                    runner.stop()
                runner.run()
            else:
                engine = RunEngine(
                    self.db,
                    target_id,
                    inner,
                    event_cb=self.queue.put,
                    max_steps=max_steps,
                    speed=speed,
                    skip_waits=skip_waits,
                    swipe_repeat=swipe_repeat,
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

    def stop(self) -> None:
        self.stop_requested = True
        if self.engine is not None:
            self.engine.stop()
        self.state_label.setText("正在停止…")

    def request_manual(self) -> None:
        """运行中请求人工介入：下一步会暂停并弹出手动选择窗口。"""
        engine = self.engine
        if engine is None:
            return
        request = getattr(engine, "request_manual", None)
        if request is None:
            return
        request()
        self.state_label.setText("已请求人工介入，将在下一步暂停…")
        self.manual_btn.setEnabled(False)

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the run and release the device before the app exits."""
        self.stop()
        device = self.device
        if device is not None:
            try:
                device.stop()
            except Exception:
                pass
        worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.target_combo.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        self.manual_btn.setEnabled(running)
        self.running_changed.emit(running)

    def _poll(self) -> None:
        try:
            for _ in range(100):
                event = self.queue.get_nowait()
                self._handle(event)
        except queue.Empty:
            pass

    def _handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "screenshot":
            self._set_frame(event.get("frame"))
        elif kind == "project_start":
            mode = "Python 编排" if event.get("mode") == "python" else "顺序执行"
            self.state_label.setText(f"项目运行中 · {event.get('name', '')}")
            self.session_label.setText(
                f"项目会话 {event.get('run_id')} · {len(event.get('blocks') or [])} 个区块 · {mode}"
            )
            self._log("项目开始", "", None, f"{event.get('name')} · {mode}", "", "info")
        elif kind == "block_start":
            self.state_label.setText(
                f"区块：{event.get('name')} · 第 {event.get('run_index')}/{event.get('max_runs')} 次"
            )
            self._log("区块开始", event.get("name"), None, f"变量名 {event.get('alias')}", "", "info")
        elif kind == "block_finish":
            result = event.get("result")
            if result is not None:
                self._log(
                    "区块结束",
                    result.name,
                    None,
                    f"状态 {result.status} · 步数 {result.steps} · 保底 {result.fallbacks}",
                    result.error or "",
                    "match" if result.ok else "error",
                )
        elif kind == "project_log":
            self._log("脚本", "", None, event.get("message", ""), "", "info")
        elif kind == "project_finish":
            status = event.get("status")
            labels = {"finished": "项目完成", "stopped": "项目已停止", "error": "项目出错停止"}
            self.state_label.setText(labels.get(status, str(status)))
            self.session_label.setText(
                f"项目会话 {event.get('run_id')} · {labels.get(status, status)} · "
                f"区块 {event.get('blocks')} · 总步数 {event.get('steps')} · 保底 {event.get('fallbacks')}"
            )
            self._log(
                "项目结束", "", None, labels.get(status, str(status)), event.get("error") or "",
                "error" if status == "error" else "info",
            )
        elif kind == "start":
            self.state_label.setText("运行中…")
            self.session_label.setText(f"会话 {event.get('session_id')} · {event.get('flow', '')}")
        elif kind == "match":
            matches = event.get("matches") or []
            self._fill_candidates(matches)
            block = event.get("block")
            state_name = event.get("state_name", "")
            label = f"{block} / {state_name}" if block else state_name
            self.state_label.setText(
                f"{'区块 ' + block + ' · ' if block else ''}当前状态：{state_name} · "
                f"相似度 {event.get('similarity', 0):.3f}"
            )
            self._log(
                "匹配", label, event.get("similarity"), event.get("action", ""), "成功", "match"
            )
        elif kind == "no_match":
            self._fill_candidates(event.get("matches") or [])
            prefix = f"区块 {event.get('block')} · " if event.get("block") else ""
            self.state_label.setText(
                f"{prefix}未匹配 · 重试 {event.get('retries')}/{event.get('max_retries')}"
            )
            self._log(
                "未匹配", "", event.get("similarity"), "等待画面稳定",
                f"重试 {event.get('retries')}/{event.get('max_retries')}", "no_match",
            )
        elif kind == "fallback":
            prefix = f"区块 {event.get('block')} · " if event.get("block") else ""
            self.state_label.setText(
                f"{prefix}执行保底 {event.get('index')}/{event.get('max_fallbacks')}"
            )
            self._log(
                "保底", "", None, "执行保底动作",
                f"{event.get('index')}/{event.get('max_fallbacks')}", "warn",
            )
        elif kind == "warn":
            message = event.get("message", "")
            self._log("提示", "", None, message, "", "info")
        elif kind == "ai_decision":
            self._log(
                "AI 决策",
                event.get("node_name") or "",
                event.get("confidence"),
                f"{event.get('page') or ''} · {event.get('reason') or ''}".strip(" ·"),
                f"{event.get('action')} · {event.get('elapsed', 0):.1f}s",
                "ai",
            )
        elif kind == "ai_recover":
            self._log(
                "AI 接管",
                "",
                None,
                event.get("reason") or "",
                event.get("action") or "",
                "ai",
            )
        elif kind == "ai_rerank":
            self._log(
                "AI 选节点",
                event.get("node_name") or "",
                event.get("confidence"),
                event.get("reason") or "",
                "图片相似度不足，由 AI 选定",
                "ai",
            )
        elif kind == "ai_verified":
            self._log(
                "AI 复核",
                event.get("node_name") or "",
                None,
                event.get("reason") or "",
                "目标已达成",
                "ai",
            )
        elif kind == "ai_log":
            self._log("AI", "", None, event.get("message", ""), "", "ai")
        elif kind == "manual_saved":
            self._log("人工已记住", event.get("name") or "", None, "已写入流程画布", "", "warn")
            self.data_changed.emit()
        elif kind == "nodes_changed":
            self.data_changed.emit()
        elif kind == "human_applied":
            self.state_label.setText(f"人工动作：{event.get('action')}")
            self._log(
                "人工",
                "",
                None,
                event.get("action") or "",
                "AI 托管" if event.get("ai") else "已执行",
                "ai" if event.get("ai") else "warn",
            )
        elif kind == "human_monitor_failed":
            self._log("人工检测", "", None, "检测未达标", event.get("action") or "", "error")
        elif kind == "need_human":
            self._handle_human(event)
        elif kind == "error":
            message = event.get("message", "")
            self.state_label.setText(f"错误：{message}")
            self._log("错误", event.get("block") or "", None, message, "停止", "error")
        elif kind == "end":
            self.state_label.setText("到达结束节点")
        elif kind == "timeout":
            self.state_label.setText(
                f"运行时间到，流程自动结束（已运行 {event.get('elapsed', 0):.0f}s）"
            )
            self._log(
                "时间结束", "", None, "运行时间达到上限，流程自动结束", "", "warn"
            )
        elif kind == "finish":
            status = event.get("status")
            labels = {
                "finished": "已完成",
                "stopped": "已停止",
                "error": "出错停止",
                "max_steps": "达到最大步数",
            }
            self.state_label.setText(labels.get(status, str(status)))
            self.session_label.setText(
                f"会话 {event.get('session_id')} · {labels.get(status, status)} · "
                f"步数 {event.get('steps')} · 保底 {event.get('fallbacks')}"
            )
            self._log(
                "结束", "", None, labels.get(status, str(status)), event.get("error") or "",
                "error" if status == "error" else "info",
            )
        elif kind == "worker_done":
            self._set_running(False)
            self.engine = None

    def _handle_human(self, event: dict) -> None:
        from .human_dialog import HumanActionDialog

        if self._frame is None and event.get("frame") is not None:
            self._set_frame(event.get("frame"))
        dialog = HumanActionDialog(
            self,
            self._frame,
            str(event.get("context") or "请选择下一步动作"),
            bool(event.get("allow_ai", True)),
            str(event.get("failed_action") or ""),
        )
        self.state_label.setText("等待人工选择…")
        self.manual_btn.setEnabled(False)
        accepted = dialog.exec()
        self.manual_btn.setEnabled(self.running)
        engine = self.engine
        result = dialog.result if accepted else None
        if engine is not None:
            provide = getattr(engine, "provide_human", None)
            if provide is not None:
                provide(result)
        if result is None:
            self.state_label.setText("已取消人工选择，继续自动运行")
        elif result.get("ai"):
            self.state_label.setText("已选择：让 AI 托管")
        else:
            self.state_label.setText("人工动作已提交，继续运行")

    def _set_frame(self, frame) -> None:
        if frame is None:
            return
        height, width = frame.shape[:2]
        data = np.ascontiguousarray(frame)
        image = QImage(data.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self._frame = image
        self._update_preview()

    def _update_preview(self) -> None:
        if self._frame is None:
            return
        pixmap = QPixmap.fromImage(self._frame)
        size = self.preview.size()
        self.preview.setPixmap(
            pixmap.scaled(
                size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_preview()

    def _fill_candidates(self, matches) -> None:
        self.candidates.setRowCount(0)
        for match in matches:
            row = self.candidates.rowCount()
            self.candidates.insertRow(row)
            values = [
                match.node.name,
                f"{match.similarity:.3f}",
                "-" if match.roi_similarity is None else f"{match.roi_similarity:.3f}",
                f"{match.bonus:.2f}",
                f"{match.score:.3f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.candidates.setItem(row, col, item)

    def _log(self, kind, node, similarity, action, result, tag) -> None:
        row = self.logs.rowCount()
        self.logs.insertRow(row)
        values = [
            row + 1,
            time.strftime("%H:%M:%S"),
            kind,
            node or "",
            "-" if similarity is None else f"{similarity:.3f}",
            action or "",
            result or "",
        ]
        colors = {
            "match": GREEN,
            "no_match": ORANGE,
            "warn": ORANGE,
            "error": RED,
            "info": MUTED,
            "ai": "#a371f7",
        }
        color = QColor(colors.get(tag, MUTED))
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if tag in colors:
                item.setForeground(color)
            self.logs.setItem(row, col, item)
        self.logs.scrollToBottom()
