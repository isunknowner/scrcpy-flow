from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .. import ai as ai_module
from .. import ollama
from .theme import GREEN, MUTED, RED


class TaskThread(QThread):
    """在后台跑一个函数，避免本地大模型阻塞界面。"""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func

    def run(self):  # noqa: D102
        try:
            self.succeeded.emit(self.func())
        except Exception as exc:  # 交给界面显示
            self.failed.emit(str(exc))


def start_task(owner, func) -> TaskThread:
    """启动后台任务并把线程挂在 owner 上，防止被垃圾回收。"""
    thread = TaskThread(func, owner)
    owner._thread = thread
    thread.start()
    return thread


class AISettingsDialog(QDialog):
    """本地 Ollama 连接、模型与功能开关。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = ollama.load_settings()
        self.setWindowTitle("AI 设置 · 本地 Ollama")
        self.resize(560, 640)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "所有推理都在你自己的电脑上完成，不会把截图上传到云端。"
            "Ollama 没启动时，程序会继续使用原来的图片匹配流程。"
        )
        intro.setObjectName("subtitle")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        conn_group = QGroupBox("连接")
        conn_form = QFormLayout(conn_group)
        self.enable_check = QCheckBox("启用本地 AI（AI 决策节点 / 智能兜底 / 节点重排）")
        self.enable_check.setChecked(self.settings.enabled)
        conn_form.addRow("", self.enable_check)
        row = QHBoxLayout()
        self.url_edit = QLineEdit(self.settings.base_url)
        self.test_btn = QPushButton("测试连接")
        row.addWidget(self.url_edit, 1)
        row.addWidget(self.test_btn)
        conn_form.addRow("服务地址", row)
        self.status_label = QLabel("未检测")
        self.status_label.setWordWrap(True)
        conn_form.addRow("状态", self.status_label)
        layout.addWidget(conn_group)

        model_group = QGroupBox("模型")
        model_form = QFormLayout(model_group)
        self.vision_combo = QComboBox()
        self.vision_combo.setEditable(True)
        self.vision_combo.setToolTip("需要能看图的模型，例如 qwen3.5:4b、qwen2.5vl:3b、llava:7b")
        self.text_combo = QComboBox()
        self.text_combo.setEditable(True)
        self.text_combo.setToolTip("生成流程用的文本模型；留空则自动选择")
        self.refresh_btn = QPushButton("刷新模型列表")
        self.model_hint = QLabel("")
        self.model_hint.setObjectName("subtitle")
        self.model_hint.setWordWrap(True)
        model_form.addRow("视觉模型", self.vision_combo)
        model_form.addRow("文本模型", self.text_combo)
        model_form.addRow("", self.refresh_btn)
        model_form.addRow("提示", self.model_hint)
        layout.addWidget(model_group)

        self.advanced_check = QCheckBox("显示高级选项（推理参数、耗时相关）")
        self.advanced_check.setChecked(False)
        self.advanced_check.setToolTip("一般不用改；模型太慢或图片看不清时再调")
        layout.addWidget(self.advanced_check)
        tuning_group = QGroupBox("推理参数")
        tuning_group.setVisible(False)
        self.advanced_check.toggled.connect(tuning_group.setVisible)
        self.tuning_group = tuning_group
        tuning_form = QFormLayout(tuning_group)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 1.5)
        self.temperature.setSingleStep(0.05)
        self.temperature.setDecimals(2)
        self.temperature.setValue(float(self.settings.temperature))
        self.timeout = QSpinBox()
        self.timeout.setRange(10, 900)
        self.timeout.setSuffix(" 秒")
        self.timeout.setValue(int(self.settings.timeout))
        self.image_side = QSpinBox()
        self.image_side.setRange(320, 1600)
        self.image_side.setSingleStep(80)
        self.image_side.setSuffix(" px")
        self.image_side.setValue(int(self.settings.max_image_side))
        self.image_side.setToolTip("发给模型的截图最长边。越小越快，但小字可能看不清。")
        self.think_check = QCheckBox("让模型先思考再回答（更准但更慢）")
        self.think_check.setChecked(self.settings.think)
        self.max_decisions = QSpinBox()
        self.max_decisions.setRange(1, 200)
        self.max_decisions.setValue(int(self.settings.max_decisions))
        self.max_decisions.setToolTip("单个 AI 决策节点一次运行里最多做多少个动作")
        tuning_form.addRow("温度", self.temperature)
        tuning_form.addRow("超时", self.timeout)
        tuning_form.addRow("截图最长边", self.image_side)
        tuning_form.addRow("单节点决策上限", self.max_decisions)
        tuning_form.addRow("", self.think_check)
        layout.addWidget(tuning_group)

        feature_group = QGroupBox("功能")
        feature_layout = QVBoxLayout(feature_group)
        self.ai_nodes_check = QCheckBox("允许运行 AI 决策节点（不需要截图）")
        self.ai_nodes_check.setChecked(self.settings.ai_nodes)
        self.fallback_check = QCheckBox("图片匹配失败时，让 AI 接管一步救场")
        self.fallback_check.setChecked(self.settings.ai_fallback)
        self.rerank_check = QCheckBox("相似度不够时，让 AI 在候选节点里挑一个")
        self.rerank_check.setChecked(self.settings.ai_rerank)
        self.ui_check = QCheckBox("把 Android 控件树（可点文字/按钮）一起发给模型")
        self.ui_check.setChecked(self.settings.ui_context)
        self.text_check = QCheckBox("允许 AI 决定输入文字")
        self.text_check.setChecked(self.settings.allow_text)
        for widget in (
            self.ai_nodes_check,
            self.fallback_check,
            self.rerank_check,
            self.ui_check,
            self.text_check,
        ):
            feature_layout.addWidget(widget)
        layout.addWidget(feature_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.test_btn.clicked.connect(self._test_connection)
        self.refresh_btn.clicked.connect(self._refresh_models)
        self.vision_combo.currentTextChanged.connect(self._update_hint)
        self._populate_models([])
        self._test_connection()

    # -- 数据 ------------------------------------------------------------
    def _populate_models(self, names: list[str]) -> None:
        current_vision = self.vision_combo.currentText() or self.settings.vision_model
        current_text = self.text_combo.currentText() or self.settings.text_model
        self.vision_combo.blockSignals(True)
        self.text_combo.blockSignals(True)
        self.vision_combo.clear()
        self.text_combo.clear()
        self.vision_combo.addItem("")
        self.text_combo.addItem("")
        for name in names:
            self.vision_combo.addItem(name)
            self.text_combo.addItem(name)
        self.vision_combo.setCurrentText(current_vision)
        self.text_combo.setCurrentText(current_text)
        self.vision_combo.blockSignals(False)
        self.text_combo.blockSignals(False)
        self._update_hint()

    def _update_hint(self) -> None:
        model = self.vision_combo.currentText().strip()
        if not model:
            self.model_hint.setText("留空表示自动挑选一个支持看图的模型。")
            return
        try:
            client = ollama.OllamaClient(
                self.url_edit.text().strip(), settings=self.settings
            )
            supports = client.supports_vision(model)
        except Exception:
            return
        if supports:
            self.model_hint.setText(f"{model} 支持看图，可以作为视觉模型。")
            self.model_hint.setStyleSheet(f"color:{GREEN}")
        else:
            self.model_hint.setText(
                f"{model} 不支持看图。请改用 qwen2.5vl:3b / llava:7b 等视觉模型。"
            )
            self.model_hint.setStyleSheet(f"color:{RED}")

    def _set_status(self, text: str, color: str = MUTED) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{color}")

    # -- 交互 ------------------------------------------------------------
    def _test_connection(self) -> None:
        self._set_status("正在检测…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            client = ollama.OllamaClient(
                self.url_edit.text().strip(), timeout=6, settings=self.settings
            )
            ok, detail = client.ping()
            if not ok:
                self._set_status(f"连接失败：{detail}", RED)
                return
            try:
                models = client.model_names(refresh=True)
            except ollama.OllamaError:
                models = []
            self._populate_models(models)
            vision = ""
            try:
                vision = client.resolve_vision_model()
            except ollama.OllamaError as exc:
                self._set_status(f"已连接 v{detail}，但{exc}", RED)
                return
            self._set_status(
                f"已连接 Ollama v{detail} · {len(models)} 个模型 · 视觉模型 {vision}", GREEN
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _refresh_models(self) -> None:
        try:
            client = ollama.OllamaClient(
                self.url_edit.text().strip(), timeout=8, settings=self.settings
            )
            models = client.model_names(refresh=True)
        except ollama.OllamaError as exc:
            self._set_status(f"获取模型失败：{exc}", RED)
            return
        self._populate_models(models)
        self._set_status(f"已刷新，共 {len(models)} 个模型", GREEN)

    def _save(self) -> None:
        self.settings.enabled = self.enable_check.isChecked()
        self.settings.base_url = self.url_edit.text().strip() or ollama.DEFAULT_BASE_URL
        self.settings.vision_model = self.vision_combo.currentText().strip()
        self.settings.text_model = self.text_combo.currentText().strip()
        self.settings.temperature = float(self.temperature.value())
        self.settings.timeout = float(self.timeout.value())
        self.settings.max_image_side = int(self.image_side.value())
        self.settings.think = self.think_check.isChecked()
        self.settings.max_decisions = int(self.max_decisions.value())
        self.settings.ai_nodes = self.ai_nodes_check.isChecked()
        self.settings.ai_fallback = self.fallback_check.isChecked()
        self.settings.ai_rerank = self.rerank_check.isChecked()
        self.settings.ui_context = self.ui_check.isChecked()
        self.settings.allow_text = self.text_check.isChecked()
        ollama.save_settings(self.settings)
        self.accept()


class AIGenerateFlowDialog(QDialog):
    """用一句中文描述生成一条 AI 流程，或一个多区块项目。"""

    def __init__(self, parent=None, brain=None):
        super().__init__(parent)
        self.plan: dict | None = None
        self.mode = "flow"
        self._brain_override = brain
        self.setWindowTitle("AI 生成流程 / 项目")
        self.resize(640, 600)
        self.settings = ollama.load_settings()
        layout = QVBoxLayout(self)

        hint = QLabel(
            "用一句话说明你想让手机做什么，AI 会自动拆成几步，不用先截图。"
            "大多数情况选「单条流程」就够了；多阶段任务再选「多区块项目」。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("创建为"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("单条流程（推荐）", "flow")
        self.mode_combo.addItem("多区块项目（进阶）", "project")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.description = QPlainTextEdit()
        self.description.setPlaceholderText(
            "例如：打开美团，搜索黄焖鸡米饭，把第一家店加入购物车并提交订单"
        )
        self.description.setFixedHeight(90)
        layout.addWidget(self.description)

        row = QHBoxLayout()
        self.generate_btn = QPushButton("生成流程")
        self.generate_btn.setObjectName("primary")
        self.status_label = QLabel("")
        self.status_label.setObjectName("subtitle")
        row.addWidget(self.generate_btn)
        row.addWidget(self.status_label, 1)
        layout.addLayout(row)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("生成结果会显示在这里")
        layout.addWidget(self.preview, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("创建流程")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.generate_btn.clicked.connect(self._generate)

    def _on_mode_changed(self) -> None:
        project = self.mode_combo.currentData() == "project"
        self.generate_btn.setText("生成项目" if project else "生成流程")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "创建项目" if project else "创建流程"
        )

    def _generate(self) -> None:
        description = self.description.toPlainText().strip()
        if not description:
            self.status_label.setText("请先写下需求")
            return
        self.mode = str(self.mode_combo.currentData() or "flow")
        self.generate_btn.setEnabled(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.status_label.setText("本地模型正在思考…")
        self.preview.setPlainText("")
        brain = self._brain_override or ai_module.ScreenBrain(self.settings)
        project_mode = self.mode == "project"

        def job():
            if project_mode:
                return brain.plan_project(description)
            return brain.plan_flow(description)

        start_task(self, job).succeeded.connect(self._on_plan)
        self._thread.failed.connect(self._on_error)

    def _on_plan(self, plan) -> None:
        self.plan = plan if isinstance(plan, dict) else None
        self.generate_btn.setEnabled(True)
        if not self.plan:
            self.status_label.setText("生成失败")
            return
        self.status_label.setText("生成完成，可以直接创建")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        if self.mode == "project":
            lines = [
                f"项目名：{self.plan.get('name', '')}",
                f"说明：{self.plan.get('description', '')}",
                "",
            ]
            for index, block in enumerate(self.plan.get("blocks") or [], start=1):
                lines.append(f"区块 {index}：{block.get('name', '')}")
                lines.append(f"   目标：{block.get('flow_goal', '')}")
                for step_index, step in enumerate(block.get("steps") or [], start=1):
                    lines.append(f"   {step_index}) {step.get('name', '')}：{step.get('goal', '')}")
                lines.append("")
        else:
            lines = [f"流程名：{self.plan.get('name', '')}", f"说明：{self.plan.get('description', '')}", ""]
            for index, step in enumerate(self.plan.get("steps") or [], start=1):
                lines.append(f"{index}. {step.get('name', '')}")
                lines.append(f"   目标：{step.get('goal', '')}")
                if step.get("detail"):
                    lines.append(f"   说明：{step['detail']}")
        self.preview.setPlainText("\n".join(lines))

    def _on_error(self, message: str) -> None:
        self.generate_btn.setEnabled(True)
        self.status_label.setText(f"生成失败：{message}")


class AIGoalDialog(QDialog):
    """编辑一个 AI 决策节点的任务目标。"""

    def __init__(self, goal: str = "", detail: str = "", max_decisions: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 决策节点")
        self.resize(520, 380)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "运行到这个节点时，本地视觉模型会看当前截图，一步步操作直到完成下面的目标。"
            "建议把目标写成**动作**（例如「点击底部「电视台」标签」「在搜索框输入黄焖鸡并回车」）；"
            "写成「进入某页面」这类结果描述时，模型较难判断是否已经完成。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self.goal = QPlainTextEdit(goal)
        self.goal.setPlaceholderText("例如：打开系统设置，进入「电池」页面")
        self.goal.setFixedHeight(110)
        self.detail = QPlainTextEdit(detail)
        self.detail.setPlaceholderText("可选：额外说明，例如「不要点击广告」「用搜索框而不是分类」")
        self.detail.setFixedHeight(70)
        self.limit = QSpinBox()
        self.limit.setRange(0, 200)
        self.limit.setSpecialValueText("使用全局设置")
        self.limit.setValue(int(max_decisions or 0))
        self.limit.setToolTip("0 表示使用「AI 设置」里的单节点决策上限")
        form.addRow("任务目标", self.goal)
        form.addRow("补充说明", self.detail)
        form.addRow("决策次数上限", self.limit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        if not self.goal.toPlainText().strip():
            self.goal.setFocus()
            return
        self.accept()

    def values(self) -> dict:
        return {
            "goal": self.goal.toPlainText().strip(),
            "detail": self.detail.toPlainText().strip(),
            "max_decisions": int(self.limit.value()),
        }
