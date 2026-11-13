"""云端 AI 面板：配置 OpenAI 兼容 API，并用它看图决策/生成流程/执行任务。"""
from __future__ import annotations

import json
import queue
import threading

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
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
from .. import cloud_ai, ollama
from ..ai_project import create_project_from_plan
from ..db import Database
from ..device import AdbDevice
from .ai_dialogs import AIGenerateFlowDialog
from .theme import GREEN, MUTED, RED


class CloudAIPanel(QWidget):
    status_message = Signal(str)

    def __init__(self, db: Database, flow_workspace, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_workspace = flow_workspace
        self.serial: str | None = None
        self.open_flow_callback = None
        self.open_project_callback = None
        self.settings = cloud_ai.load_settings()
        self._queue: queue.Queue = queue.Queue()
        self._agent = None
        self._agent_thread: threading.Thread | None = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll)
        self._build()

    # -- 构建 ------------------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        head = QHBoxLayout()
        self.status_label = QLabel("云端 API：未启用")
        self.status_label.setObjectName("subtitle")
        head.addWidget(self.status_label, 1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_status)
        head.addWidget(refresh)
        layout.addLayout(head)

        config = QFormLayout()
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("本地 Ollama", "local")
        self.backend_combo.addItem("云端 API", "cloud")
        self.backend_combo.addItem("自动（云端优先，失败回退本地）", "auto")
        self.backend_combo.setToolTip("运行流程和 AI 节点时使用哪个后端")
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        config.addRow("AI 后端", self.backend_combo)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("选择常见服务（可选）", "")
        for label, key in cloud_ai.preset_choices():
            self.preset_combo.addItem(label, key)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        config.addRow("快速配置", self.preset_combo)
        self.base_edit = QLineEdit(self.settings.base_url)
        self.base_edit.setPlaceholderText("https://api.openai.com/v1 或 https://api.deepseek.com/v1")
        config.addRow("Base URL", self.base_edit)
        self.key_edit = QLineEdit(self.settings.api_key)
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("sk-…（只保存在本机 data/cloud.json）")
        config.addRow("API Key", self.key_edit)
        self.model_edit = QLineEdit(self.settings.model)
        self.model_edit.setPlaceholderText("例如 gpt-4o-mini / deepseek-chat / qwen-vl-max")
        config.addRow("模型", self.model_edit)
        layout.addLayout(config)

        features = QHBoxLayout()
        self.enabled_check = QCheckBox("启用云端")
        self.enabled_check.setChecked(self.settings.enabled)
        features.addWidget(self.enabled_check)
        self.advanced_check = QCheckBox("高级选项")
        features.addWidget(self.advanced_check)
        features.addStretch(1)
        save = QPushButton("保存设置")
        save.clicked.connect(self._save)
        test = QPushButton("测试连接")
        test.clicked.connect(self._test)
        features.addWidget(save)
        features.addWidget(test)
        layout.addLayout(features)

        self.features_widget = QWidget()
        feature_layout = QHBoxLayout(self.features_widget)
        feature_layout.setContentsMargins(0, 0, 0, 0)
        self.nodes_check = QCheckBox("AI 决策节点")
        self.nodes_check.setChecked(self.settings.ai_nodes)
        self.fallback_check = QCheckBox("失败时 AI 接管")
        self.fallback_check.setChecked(self.settings.ai_fallback)
        self.rerank_check = QCheckBox("相似度不足时 AI 选节点")
        self.rerank_check.setChecked(self.settings.ai_rerank)
        for widget in (self.nodes_check, self.fallback_check, self.rerank_check):
            feature_layout.addWidget(widget)
        feature_layout.addStretch(1)
        self.features_widget.setVisible(False)
        self.advanced_check.toggled.connect(self.features_widget.setVisible)
        layout.addWidget(self.features_widget)

        actions = QHBoxLayout()
        self.task_edit = QLineEdit()
        self.task_edit.setPlaceholderText("给云端 AI 一个任务，例如：打开设置查看电池电量")
        actions.addWidget(self.task_edit, 1)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(20)
        self.steps_spin.setToolTip("最多决策步数")
        actions.addWidget(self.steps_spin)
        self.run_btn = QPushButton("执行任务")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._start_task)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_task)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.stop_btn)
        layout.addLayout(actions)

        tools = QHBoxLayout()
        describe = QPushButton("分析当前画面")
        describe.clicked.connect(self._describe_screen)
        generate = QPushButton("AI 生成流程 / 项目")
        generate.clicked.connect(self._generate_flow)
        tools.addWidget(describe)
        tools.addWidget(generate)
        tools.addStretch(1)
        layout.addLayout(tools)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        layout.addWidget(self.log, 1)

        self._sync_backend_combo()
        self.refresh_status()

    # -- 设置 ------------------------------------------------------------
    def _sync_backend_combo(self) -> None:
        local = ollama.load_settings()
        index = self.backend_combo.findData(getattr(local, "backend", "local"))
        self.backend_combo.blockSignals(True)
        self.backend_combo.setCurrentIndex(max(0, index))
        self.backend_combo.blockSignals(False)

    def _on_preset_changed(self) -> None:
        key = str(self.preset_combo.currentData() or "")
        values = cloud_ai.preset_values(key) if key else None
        if not values:
            return
        base_url, model = values
        self.base_edit.setText(base_url)
        self.model_edit.setText(model)
        self.enabled_check.setChecked(True)
        self._append(f"已填入 {self.preset_combo.currentText()} 的地址和模型，请填 API Key 后点「保存设置」。", GREEN)

    def _on_backend_changed(self) -> None:
        backend = str(self.backend_combo.currentData() or "local")
        local = ollama.load_settings()
        local.backend = backend
        ollama.save_settings(local)
        self._append(f"AI 后端已切换为：{self.backend_combo.currentText()}", GREEN)
        self.status_message.emit(f"AI 后端：{self.backend_combo.currentText()}")

    def _settings_from_widgets(self) -> cloud_ai.CloudSettings:
        settings = cloud_ai.CloudSettings.from_dict(self.settings.to_dict())
        settings.base_url = self.base_edit.text().strip() or settings.base_url
        settings.api_key = self.key_edit.text().strip()
        settings.model = self.model_edit.text().strip() or settings.model
        settings.enabled = self.enabled_check.isChecked()
        settings.ai_nodes = self.nodes_check.isChecked()
        settings.ai_fallback = self.fallback_check.isChecked()
        settings.ai_rerank = self.rerank_check.isChecked()
        return settings

    def _save(self) -> None:
        self.settings = self._settings_from_widgets()
        cloud_ai.save_settings(self.settings)
        self._append("云端设置已保存到 data/cloud.json", GREEN)
        self.status_message.emit("云端 AI 设置已保存")
        self.refresh_status()

    def refresh_status(self) -> None:
        settings = cloud_ai.load_settings()
        local = ollama.load_settings()
        backend = {
            "local": "本地 Ollama",
            "cloud": "云端 API",
            "auto": "自动（云端优先）",
        }.get(getattr(local, "backend", "local"), "本地 Ollama")
        if not settings.configured:
            self.status_label.setText(f"当前后端：{backend} · 云端未配置")
        elif not settings.enabled:
            self.status_label.setText(f"当前后端：{backend} · 云端已关闭")
        else:
            self.status_label.setText(
                f"当前后端：{backend} · {settings.model} · {settings.base_url}"
            )

    # -- 连接测试 --------------------------------------------------------
    def _test(self) -> None:
        self.settings = self._settings_from_widgets()
        cloud_ai.save_settings(self.settings)
        self._append("正在连接云端 API…")
        settings = self.settings

        def job():
            return cloud_ai.CloudClient(settings).list_models()

        thread = threading.Thread(target=self._guarded, args=(job, self._on_models), daemon=True)
        self._test_thread = thread
        thread.start()

    def _on_models(self, models: list[str]) -> None:
        preview = "、".join(models[:8]) or "（没有返回模型列表）"
        self._append(f"连接成功，可用模型：{preview}", GREEN)
        self.status_message.emit("云端 API 连接成功")

    def _guarded(self, func, on_done=None) -> None:
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001 - 展示给用户
            self._append(f"失败：{exc}", RED)
            return
        if on_done is not None:
            try:
                on_done(result)
            except Exception as exc:  # noqa: BLE001
                self._append(f"处理结果失败：{exc}", RED)

    # -- 画面分析 --------------------------------------------------------
    def _cloud_brain(self):
        settings = self._settings_from_widgets()
        if not settings.configured:
            raise RuntimeError("请先填写 Base URL 和 API Key，并点「保存设置」")
        return ai_module.ScreenBrain(settings, client=cloud_ai.CloudClient(settings))

    def _describe_screen(self) -> None:
        if not self.serial:
            self._append("请先在「运行」页选择一台设备", RED)
            return
        self._append("正在截图并让云端模型分析…")
        serial = self.serial
        brain = None

        def job():
            nonlocal brain
            device = AdbDevice(serial)
            shot = device.screenshot()
            elements = []
            try:
                elements = device.ui_elements() or []
            except Exception:
                elements = []
            brain = self._cloud_brain()
            return brain.describe(shot, elements)

        threading.Thread(
            target=self._guarded, args=(job, self._on_described), daemon=True
        ).start()

    def _on_described(self, data) -> None:
        self._append("画面理解：" + json.dumps(data, ensure_ascii=False)[:1200])

    # -- 生成流程 / 项目 --------------------------------------------------
    def _generate_flow(self) -> None:
        try:
            brain = self._cloud_brain()
        except Exception as exc:  # noqa: BLE001
            self._append(f"失败：{exc}", RED)
            return
        dialog = AIGenerateFlowDialog(self, brain=brain)
        if not dialog.exec() or not dialog.plan:
            return
        plan = dialog.plan
        if getattr(dialog, "mode", "flow") == "project":
            try:
                project_id = create_project_from_plan(self.db, plan)
            except Exception as exc:  # noqa: BLE001
                self._append(f"创建项目失败：{exc}", RED)
                return
            self._append(f"已生成项目「{plan.get('name')}」", GREEN)
            if self.open_project_callback is not None:
                self.open_project_callback(project_id)
            return
        flow_id = self.flow_workspace.create_flow_from_plan(plan)
        if flow_id is None:
            self._append("创建流程失败：请先在「流程画布」里选择一个流程库", RED)
            return
        self._append(f"已生成流程「{plan.get('name')}」", GREEN)
        if self.open_flow_callback is not None:
            self.open_flow_callback(flow_id)

    # -- 执行任务 --------------------------------------------------------
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
        try:
            brain = self._cloud_brain()
        except Exception as exc:  # noqa: BLE001
            self._append(f"失败：{exc}", RED)
            return
        self.settings = self._settings_from_widgets()
        cloud_ai.save_settings(self.settings)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._append(f"云端 AI 任务开始：{goal}", GREEN)
        serial = self.serial
        max_steps = int(self.steps_spin.value())
        settings = self.settings

        def job():
            device = AdbDevice(serial)
            runner = ai_module.AgentRunner(
                device,
                goal,
                settings=settings,
                brain=brain,
                event_cb=self._queue.put,
                max_steps=max_steps,
            )
            self._agent = runner
            try:
                return runner.run()
            finally:
                self._agent = None

        thread = threading.Thread(
            target=self._task_guarded, args=(job,), daemon=True
        )
        self._agent_thread = thread
        thread.start()
        self._poll_timer.start()

    def _task_guarded(self, func) -> None:
        try:
            result = func()
            self._queue.put({"type": "task_done", "result": result})
        except Exception as exc:  # noqa: BLE001
            self._queue.put({"type": "task_error", "message": str(exc)})

    def _stop_task(self) -> None:
        agent = self._agent
        if agent is not None:
            try:
                agent.stop()
            except Exception:
                pass

    def _poll(self) -> None:
        finished = False
        try:
            while True:
                event = self._queue.get_nowait()
                kind = event.get("type")
                if kind == "task_done":
                    finished = True
                    result = event.get("result") or {}
                    self._append(
                        f"任务结束：{result.get('status')} · 步数 {result.get('steps')}",
                        GREEN if result.get("status") == "finished" else RED,
                    )
                elif kind == "task_error":
                    finished = True
                    self._append(f"任务失败：{event.get('message')}", RED)
                elif kind == "ai_decision":
                    self._append(
                        f"决策：{event.get('page') or ''} · {event.get('action')}"
                        f" · {event.get('reason') or ''}".strip(" ·")
                    )
                elif kind in ("ai_agent_start",):
                    self._append(str(event.get("goal") or "开始"))
                elif kind in ("ai_agent_finish",):
                    self._append("任务完成")
                elif kind in ("warn", "error"):
                    self._append(str(event.get("message") or event), RED)
        except queue.Empty:
            pass
        if finished:
            self._poll_timer.stop()
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    # -- 工具 ------------------------------------------------------------
    def set_serial(self, serial: str | None) -> None:
        self.serial = serial

    def _append(self, text: str, color: str = MUTED) -> None:
        prefix = {"#3fb950": "✓ ", "#f85149": "✗ "}.get(color, "")
        self.log.appendPlainText(prefix + str(text))

    def shutdown(self) -> None:
        self._stop_task()
