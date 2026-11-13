from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..models import Node


def parse_literal(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def format_literal(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


class VariableDialog(QDialog):
    def __init__(self, parent=None, name: str = "", value: str = ""):
        super().__init__(parent)
        self.setWindowTitle("变量")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(name)
        self.value_edit = QLineEdit(value)
        form.addRow("变量名", self.name_edit)
        form.addRow("初始值", self.value_edit)
        layout.addLayout(form)
        tip = QLabel("数字、true/false、[1,2] 按 JSON 解析；其他按字符串。")
        tip.setObjectName("subtitle")
        layout.addWidget(tip)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_value(self) -> tuple[str, object] | None:
        name = self.name_edit.text().strip()
        if not name:
            return None
        return name, parse_literal(self.value_edit.text())


class ConnectDialog(QDialog):
    def __init__(
        self,
        nodes: list[Node],
        src: int | None = None,
        dst: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("连接 / 下一步")
        self.setMinimumWidth(440)
        self.nodes = nodes
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.src_combo = QComboBox()
        self.dst_combo = QComboBox()
        for node in nodes:
            text = f"{node.id} · {node.name}"
            self.src_combo.addItem(text, node.id)
            self.dst_combo.addItem(text, node.id)
        if src is not None:
            index = self.src_combo.findData(src)
            if index >= 0:
                self.src_combo.setCurrentIndex(index)
        if dst is not None:
            index = self.dst_combo.findData(dst)
            if index >= 0:
                self.dst_combo.setCurrentIndex(index)
        self.weight = QDoubleSpinBox()
        self.weight.setRange(0, 20)
        self.weight.setSingleStep(0.1)
        self.weight.setValue(1.0)
        self.label = QLineEdit()
        form.addRow("当前步骤", self.src_combo)
        form.addRow("下一步骤", self.dst_combo)
        form.addRow("偏好权重", self.weight)
        form.addRow("连线说明", self.label)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        return {
            "src": int(self.src_combo.currentData()),
            "dst": int(self.dst_combo.currentData()),
            "weight": float(self.weight.value()),
            "label": self.label.text().strip(),
        }
