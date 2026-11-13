from __future__ import annotations

import json
import queue
import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import ai as ai_module
from .. import ollama
from ..ai_project import create_project_from_plan
from ..db import Database
from ..device import AdbDevice
from .ai_dialogs import AIGenerateFlowDialog, AISettingsDialog, start_task
from .theme import GREEN, MUTED, RED


class AIAssistantWidget(QWidget):
    """右侧/底部的 AI 助手面板：连接状态、快捷操作与运行日志。"""

    status_message = Signal(str)

    def __init__(self, db: Database, flow_workspace, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_workspace = flow_workspace
        self.serial: str | None = None
        self.open_flow_callback = None
        self.open_project_callback = None
        self.settings = ollama.load_settings()
        self._loading = False
        self._describe_thread = None
        self._agent_thread: threading.Thread | None = None
        self._agent = None
        self._queue: queue.Queue = queue.Queue()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll)
        self._build()
        self.status_label.setText("点击「刷新」检测本地 Ollama")

    # -- 构建 ------------------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        head = QHBoxLayout()
        self.status_label = QLabel("正在检测本地 Ollama…")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.refresh_btn = QPushButton("刷新")
        self.settings_btn = QPushButton("AI 设置…")
        head.addWidget(self.status_label, 1)
        head.addWidget(self.refresh_btn)
        head.addWidget(self.settings_btn)
        layout.addLayout(head)

        models = QHBoxLayout()
        models.addWidget(QLabel("视觉"))
        self.vision_combo = QComboBox()
        self.vision_combo.setMinimumWidth(150)
        models.addWidget(self.vision_combo)
        models.addWidget(QLabel("文本"))
        self.text_combo = QComboBox()
        self.text_combo.setMinimumWidth(150)
        models.addWidget(self.text_combo)
        models.addStretch(1)
        layout.addLayout(models)

        self.feature_group = QGroupBox("增强（只影响之后的运行）")
        feature_layout = QHBoxLayout(self.feature_group)
        self.ai_nodes_check = QCheckBox("AI 节点")
        self.fallback_check = QCheckBox("失败接管")
        self.rerank_check = QCheckBox("节点重排")
        self.ui_check = QCheckBox("控件树上下文")
        for widget in (
            self.ai_nodes_check,
            self.fallback_check,
            self.rerank_check,
            self.ui_check,
        ):
            feature_layout.addWidget(widget)
        feature_layout.addStretch(1)
        layout.addWidget(self.feature_group)

        actions = QHBoxLayout()
        self.describe_btn = QPushButton("分析当前画面")
        self.generate_btn = QPushButton("AI 生成流程…")
        self.web_label = QLabel("")
        self.web_label.setObjectName("subtitle")
        actions.addWidget(self.describe_btn)
        actions.addWidget(self.generate_btn)
        actions.addWidget(self.web_label, 1)
        layout.addLayout(actions)

        task_row = QHBoxLayout()
        self.task_edit = QLineEdit()
        self.task_edit.setPlaceholderText(
            "让 AI 直接执行一个任务，例如：打开设置，查看电池电量（不需要先建流程）"
        )
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(12)
        self.steps_spin.setSuffix(" 步")
        self.run_task_btn = QPushButton("开始")
        self.run_task_btn.setObjectName("primary")
        self.stop_task_btn = QPushButton("停止")
        self.stop_task_btn.setEnabled(False)
        task_row.addWidget(self.task_edit, 1)
        task_row.addWidget(self.steps_spin)
        task_row.addWidget(self.run_task_btn)
        task_row.addWidget(self.stop_task_btn)
        layout.addLayout(task_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setPlaceholderText("AI 的运行细节会显示在这里")
        layout.addWidget(self.log, 1)

        self.refresh_btn.clicked.connect(self.refresh_status)
        self.settings_btn.clicked.connect(self._open_settings)
        self.describe_btn.clicked.connect(self._describe_screen)
        self.generate_btn.clicked.connect(self._generate_flow)
        self.run_task_btn.clicked.connect(self._start_task)
        self.stop_task_btn.clicked.connect(self._stop_task)
        self.vision_combo.currentTextChanged.connect(self._save_models)
        self.text_combo.currentTextChanged.connect(self._save_models)
        for check, attr in (
            (self.ai_nodes_check, "ai_nodes"),
            (self.fallback_check, "ai_fallback"),
            (self.rerank_check, "ai_rerank"),
            (self.ui_check, "ui_context"),
        ):
            check.toggled.connect(
                lambda checked, name=attr: self._toggle_feature(name, checked)
            )
        self._sync_widgets()

    # -- 状态 ------------------------------------------------------------
    def set_serial(self, serial: str | None) -> None:
        self.serial = serial

    def refresh_status(self) -> None:
        self.settings = ollama.load_settings()
        self._sync_widgets()
        try:
            client = ollama.OllamaClient(settings=self.settings)
            info = client.status()
        except ollama.OllamaError as exc:
            info = {"available": False, "error": str(exc)}
        self._apply_status(info)

    def _apply_status(self, info: dict) -> None:
        if not info.get("available"):
            self.status_label.setText(
                f"本地 Ollama 未连接：{info.get('error') or '请先运行 ollama serve'}"
            )
            self.status_label.setStyleSheet(f"color:{RED}")
            self._set_models(self.settings.vision_model, self.settings.text_model, [])
            return
        models = list(info.get("models") or [])
        self._set_models(
            info.get("vision_model") or self.settings.vision_model,
            info.get("text_model") or self.settings.text_model,
            models,
        )
        warning = info.get("error") or ""
        text = (
            f"已连接 Ollama v{info.get('version')} · 视觉 {info.get('vision_model') or '-'}"
            f" · 文本 {info.get('text_model') or '-'}"
        )
        self.status_label.setText(text + (f" · {warning}" if warning else ""))
        self.status_label.setStyleSheet(f"color:{GREEN if not warning else MUTED}")

    def _set_models(self, vision: str, text: str, models: list[str]) -> None:
        self._loading = True
        try:
            for combo, value in ((self.vision_combo, vision), (self.text_combo, text)):
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("")
                for name in models:
                    combo.addItem(name)
                combo.setCurrentText(value or "")
                combo.blockSignals(False)
        finally:
            self._loading = False

    def _sync_widgets(self) -> None:
        self._loading = True
        try:
            self.ai_nodes_check.setChecked(self.settings.ai_nodes)
            self.fallback_check.setChecked(self.settings.ai_fallback)
            self.rerank_check.setChecked(self.settings.ai_rerank)
            self.ui_check.setChecked(self.settings.ui_context)
        finally:
            self._loading = False

    def _save_models(self, *_args) -> None:
        if self._loading:
            return
        self.settings.vision_model = self.vision_combo.currentText().strip()
        self.settings.text_model = self.text_combo.currentText().strip()
        ollama.save_settings(self.settings)

    def _toggle_feature(self, name: str, checked: bool) -> None:
        if self._loading:
            return
        self.settings = ollama.load_settings()
        setattr(self.settings, name, bool(checked))
        ollama.save_settings(self.settings)
        self.status_message.emit(f"AI 设置已更新：{name} = {checked}")

    def _open_settings(self) -> None:
        dialog = AISettingsDialog(self)
        if dialog.exec():
            self.refresh_status()
            self.status_message.emit("AI 设置已保存")

    # -- 日志 ------------------------------------------------------------
    def _append(self, text: str, color: str | None = None) -> None:
        if color:
            self.log.appendHtml(f'<span style="color:{color}">{text}</span>')
        else:
            self.log.appendPlainText(text)

    # -- 分析当前画面 ----------------------------------------------------
    def _describe_screen(self) -> None:
        self.describe_btn.setEnabled(False)
        self._append("正在截取手机画面并让本地模型分析…")
        settings = ollama.load_settings()
        serial = self.serial

        def job():
            device = AdbDevice(serial) if serial else None
            if device is None:
                raise RuntimeError("请先在「运行」页选择一台设备")
            shot = device.screenshot()
            brain = ai_module.ScreenBrain(settings)
            elements = []
            if settings.ui_context:
                try:
                    elements = device.ui_elements() or []
                except Exception:
                    elements = []
            return brain.describe(shot, elements)

        thread = start_task(self, job)
        thread.succeeded.connect(self._on_described)
        thread.failed.connect(self._on_failed)
        self._describe_thread = thread
        thread.finished.connect(lambda: self.describe_btn.setEnabled(True))

    def _on_described(self, data) -> None:
        self._append("画面理解结果：")
        self._append(json.dumps(data, ensure_ascii=False, indent=2))

    def _on_failed(self, message: str) -> None:
        self._append(f"失败：{message}", RED)

    # -- 生成流程 --------------------------------------------------------
    def _generate_flow(self) -> None:
        dialog = AIGenerateFlowDialog(self)
        if not dialog.exec() or not dialog.plan:
            return
        plan = dialog.plan
        if getattr(dialog, "mode", "flow") == "project":
            try:
                project_id = create_project_from_plan(self.db, plan)
            except Exception as exc:  # noqa: BLE001 - 展示给用户
                self._append(f"创建项目失败：{exc}", RED)
                return
            blocks = plan.get("blocks") or []
            self._append(
                f"已生成项目「{plan.get('name')}」，包含 {len(blocks)} 个区块", GREEN
            )
            self.status_message.emit(f"AI 项目已生成：{plan.get('name')}")
            if self.open_project_callback is not None:
                self.open_project_callback(project_id)
            return
        flow_id = self.flow_workspace.create_flow_from_plan(plan)
        if flow_id is None:
            self._append("创建流程失败：请先在「流程画布」里选择一个流程库", RED)
            return
        self._append(f"已生成流程「{plan.get('name')}」，包含 {len(plan.get('steps') or [])} 个 AI 节点")
        self.status_message.emit(f"AI 流程已生成：{plan.get('name')}")
        if self.open_flow_callback is not None and flow_id is not None:
            self.open_flow_callback(flow_id)

    # -- AI 任务 ---------------------------------------------------------
    def _start_task(self) -> None:
        goal = self.task_edit.text().strip()
        if not goal:
            self.task_edit.setFocus()
            return
        if not self.serial:
            self._append("请先在「运行」页选择一台设备", RED)
            return
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self._append("已有任务在运行", RED)
            return
        settings = ollama.load_settings()
        if not settings.enabled:
            self._append("AI 已在设置中关闭", RED)
            return
        self.run_task_btn.setEnabled(False)
        self.stop_task_btn.setEnabled(True)
        self._append(f"AI 任务开始：{goal}")
        serial = self.serial
        max_steps = int(self.steps_spin.value())

        def job():
            device = AdbDevice(serial)
            runner = ai_module.AgentRunner(
                device,
                goal,
                settings=settings,
                event_cb=self._queue.put,
                max_steps=max_steps,
            )
            self._agent = runner
            try:
                return runner.run()
            finally:
                self._agent = None

        thread = threading.Thread(target=self._run_guarded, args=(job,), daemon=True)
        self._agent_thread = thread
        thread.start()
        self._poll_timer.start()

    def _run_guarded(self, func) -> None:
        try:
            func()
        except Exception as exc:
            self._queue.put({"type": "error", "message": str(exc)})
        finally:
            self._queue.put({"type": "task_done"})

    def _stop_task(self) -> None:
        agent = self._agent
        if agent is not None:
            agent.stop()
        self._append("已请求停止 AI 任务…", MUTED)

    def shutdown(self) -> None:
        """退出程序时停掉正在执行的 AI 任务。"""
        self._stop_task()
        thread = self._agent_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._poll_timer.stop()

    def _poll(self) -> None:
        try:
            while True:
                event = self._queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass

    def _handle_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "ai_step":
            self._append(
                f"第 {event.get('step')} 步 · {event.get('page') or ''} · "
                f"{event.get('action') or ''}"
            )
            if event.get("thought"):
                self._append(f"  思考：{event['thought']}", MUTED)
        elif kind == "warn":
            self._append(f"提示：{event.get('message')}", MUTED)
        elif kind == "error":
            self._append(f"错误：{event.get('message')}", RED)
        elif kind == "ai_agent_finish":
            status = event.get("status")
            self._append(f"AI 任务结束：{status}（{event.get('steps')} 步）")
        elif kind == "task_done":
            self._poll_timer.stop()
            self.run_task_btn.setEnabled(True)
            self.stop_task_btn.setEnabled(False)
            self._agent = None
