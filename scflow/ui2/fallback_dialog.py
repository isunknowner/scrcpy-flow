from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..db import Database
from ..engine import DEFAULT_FALLBACK

STEP_LABELS = {
    "tap": "点击",
    "swipe": "滑动",
    "key": "按键",
    "text": "输入文本",
    "wait": "等待",
}
STEP_VALUES = {label: key for key, label in STEP_LABELS.items()}


def _make_double(value: float, low: float, high: float, step: float, decimals: int = 2) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setValue(float(value))
    return box


def _make_int(value: int, low: int, high: int, step: int = 1) -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setValue(int(value))
    return box


def step_summary(step: dict) -> str:
    kind = step.get("type", "wait")
    if kind == "tap":
        return f"点击 ({float(step.get('x', 0.5)):.2f}, {float(step.get('y', 0.5)):.2f})"
    if kind == "swipe":
        return (
            f"滑动 ({float(step.get('x1', 0.5)):.2f},{float(step.get('y1', 0.8)):.2f})"
            f"→({float(step.get('x2', 0.5)):.2f},{float(step.get('y2', 0.2)):.2f})"
        )
    if kind == "key":
        return f"按键 {step.get('key', 'back')}"
    if kind == "text":
        text = str(step.get("text", ""))
        return f"输入 {text[:16]}"
    return f"等待 {float(step.get('seconds', step.get('wait', 1)) or 1):g}s"


class FallbackStepDialog(QDialog):
    """Edit a single fallback action (same kinds as node actions)."""

    def __init__(self, step: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("保底动作")
        self.resize(420, 320)
        self.step = dict(step or {})
        self.fields: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        header = QFormLayout()
        self.type_combo = QComboBox()
        for label in STEP_LABELS.values():
            self.type_combo.addItem(label)
        header.addRow("类型", self.type_combo)
        layout.addLayout(header)

        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        layout.addWidget(self.form_host, 1)
        layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        kind = self.step.get("type", "tap")
        index = self.type_combo.findText(STEP_LABELS.get(kind, STEP_LABELS["tap"]))
        self.type_combo.setCurrentIndex(max(0, index))
        self.type_combo.currentIndexChanged.connect(self._rebuild)
        self._rebuild()

    def _clear_form(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields = {}

    def _add_float(self, key, label, value, low=0.0, high=1.0, step=0.05, decimals=2):
        box = _make_double(value, low, high, step, decimals)
        self.form.addRow(label, box)
        self.fields[key] = box

    def _rebuild(self) -> None:
        self._clear_form()
        kind = STEP_VALUES.get(self.type_combo.currentText(), "tap")
        wait = self.step.get("wait", 0.6)
        if kind == "tap":
            self._add_float("x", "横坐标 x", self.step.get("x", 0.5))
            self._add_float("y", "纵坐标 y", self.step.get("y", 0.5))
            self._add_float("wait", "动作后等待(s)", wait, 0.0, 60.0, 0.1, 2)
        elif kind == "swipe":
            self._add_float("x1", "起点 x", self.step.get("x1", 0.5))
            self._add_float("y1", "起点 y", self.step.get("y1", 0.8))
            self._add_float("x2", "终点 x", self.step.get("x2", 0.5))
            self._add_float("y2", "终点 y", self.step.get("y2", 0.2))
            duration = _make_int(self.step.get("duration", 300), 50, 5000, 50)
            self.form.addRow("时长 ms", duration)
            self.fields["duration"] = duration
            self._add_float("wait", "动作后等待(s)", wait, 0.0, 60.0, 0.1, 2)
        elif kind == "key":
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(["back", "home", "enter", "menu", "power", "volume_up", "volume_down"])
            combo.setCurrentText(str(self.step.get("key", "back")))
            self.form.addRow("按键", combo)
            self.fields["key"] = combo
            self._add_float("wait", "动作后等待(s)", wait, 0.0, 60.0, 0.1, 2)
        elif kind == "text":
            edit = QLineEdit(str(self.step.get("text", "")))
            self.form.addRow("文本", edit)
            self.fields["text"] = edit
            self._add_float("wait", "动作后等待(s)", wait, 0.0, 60.0, 0.1, 2)
        else:
            seconds = self.step.get("seconds", self.step.get("wait", 1))
            self._add_float("seconds", "等待秒数", seconds, 0.0, 600.0, 0.5, 2)

    def values(self) -> dict:
        kind = STEP_VALUES.get(self.type_combo.currentText(), "tap")
        data: dict = {"type": kind}
        for key, widget in self.fields.items():
            if isinstance(widget, QComboBox):
                data[key] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                data[key] = widget.text()
            elif isinstance(widget, QSpinBox):
                data[key] = int(widget.value())
            else:
                data[key] = float(widget.value())
        return data

    def _accept(self) -> None:
        kind = STEP_VALUES.get(self.type_combo.currentText(), "tap")
        if kind == "text" and not self.fields["text"].text().strip():
            QMessageBox.information(self, "提示", "请输入要输入的文本")
            return
        self.accept()


class FallbackDialog(QDialog):
    """Edit a flow's fallback (retry/fallback) settings and fallback actions."""

    def __init__(self, db: Database, flow_id: int, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_id = flow_id
        flow = db.get_flow(flow_id)
        self.flow_name = flow.name if flow else ""
        self.settings = dict(flow.settings) if flow else {}
        fallback = dict(DEFAULT_FALLBACK)
        saved = self.settings.get("fallback")
        if isinstance(saved, dict):
            fallback.update(saved)
        self.fallback = fallback
        self.steps: list[dict] = [dict(step) for step in fallback.get("steps", [])]

        self.setWindowTitle(f"保底机制 · {self.flow_name}")
        self.resize(560, 560)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "未匹配到状态时的处理：先重试，多次失败后执行下面的保底动作。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self.max_retries = _make_int(self.fallback.get("max_retries", 2), 0, 20)
        self.threshold_step = _make_double(
            self.fallback.get("threshold_step", 0.04), 0.0, 0.5, 0.01, 2
        )
        self.min_threshold = _make_double(
            self.fallback.get("min_threshold", 0.55), 0.0, 1.0, 0.01, 2
        )
        self.max_fallbacks = _make_int(self.fallback.get("max_fallbacks", 3), 0, 20)
        self.retry_wait = _make_double(
            self.fallback.get("retry_wait", 0.6), 0.0, 10.0, 0.1, 2
        )
        form.addRow("最大重试次数", self.max_retries)
        form.addRow("阈值递减步长", self.threshold_step)
        form.addRow("最低阈值", self.min_threshold)
        form.addRow("最大保底次数", self.max_fallbacks)
        form.addRow("重试等待(s)", self.retry_wait)
        layout.addLayout(form)

        layout.addWidget(QLabel("保底动作（按顺序执行）"))
        self.step_list = QListWidget()
        layout.addWidget(self.step_list, 1)
        buttons = QHBoxLayout()
        self.add_btn = QPushButton("添加动作")
        self.edit_btn = QPushButton("编辑动作")
        self.delete_btn = QPushButton("删除动作")
        self.up_btn = QPushButton("上移")
        self.down_btn = QPushButton("下移")
        for widget in (self.add_btn, self.edit_btn, self.delete_btn, self.up_btn, self.down_btn):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.save_btn = QPushButton("保存")
        self.save_btn.setObjectName("primary")
        self.cancel_btn = QPushButton("取消")
        bottom.addWidget(self.cancel_btn)
        bottom.addWidget(self.save_btn)
        layout.addLayout(bottom)

        self.add_btn.clicked.connect(self._add_step)
        self.edit_btn.clicked.connect(self._edit_step)
        self.delete_btn.clicked.connect(self._delete_step)
        self.up_btn.clicked.connect(lambda: self._move_step(-1))
        self.down_btn.clicked.connect(lambda: self._move_step(1))
        self.step_list.itemDoubleClicked.connect(lambda _item: self._edit_step())
        self.save_btn.clicked.connect(self._save)
        self.cancel_btn.clicked.connect(self.reject)
        self._reload_steps()

    def _reload_steps(self) -> None:
        self.step_list.clear()
        for index, step in enumerate(self.steps):
            item = QListWidgetItem(f"{index + 1}.  {step_summary(step)}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.step_list.addItem(item)
        if self.steps:
            self.step_list.setCurrentRow(min(len(self.steps) - 1, self.step_list.currentRow()))

    def _selected_index(self) -> int:
        item = self.step_list.currentItem()
        if item is None:
            return -1
        return int(item.data(Qt.ItemDataRole.UserRole))

    def _add_step(self) -> None:
        dialog = FallbackStepDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.steps.append(dialog.values())
        self._reload_steps()
        self.step_list.setCurrentRow(len(self.steps) - 1)

    def _edit_step(self) -> None:
        index = self._selected_index()
        if index < 0:
            return
        dialog = FallbackStepDialog(self.steps[index], self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.steps[index] = dialog.values()
        self._reload_steps()
        self.step_list.setCurrentRow(index)

    def _delete_step(self) -> None:
        index = self._selected_index()
        if index < 0:
            return
        del self.steps[index]
        self._reload_steps()

    def _move_step(self, delta: int) -> None:
        index = self._selected_index()
        target = index + delta
        if index < 0 or not (0 <= target < len(self.steps)):
            return
        self.steps[index], self.steps[target] = self.steps[target], self.steps[index]
        self._reload_steps()
        self.step_list.setCurrentRow(target)

    def _save(self) -> None:
        fallback = dict(self.fallback)
        fallback.update(
            {
                "max_retries": int(self.max_retries.value()),
                "threshold_step": float(self.threshold_step.value()),
                "min_threshold": float(self.min_threshold.value()),
                "max_fallbacks": int(self.max_fallbacks.value()),
                "retry_wait": float(self.retry_wait.value()),
                "steps": [dict(step) for step in self.steps],
            }
        )
        settings = dict(self.settings)
        settings["fallback"] = fallback
        self.db.update_flow(self.flow_id, settings=settings)
        self.accept()
