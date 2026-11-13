from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ..db import Database


class FlowSettingsDialog(QDialog):
    """Per-flow settings, currently the optional run-time limit."""

    def __init__(self, db: Database, flow_id: int, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_id = flow_id
        flow = db.get_flow(flow_id)
        self.flow_name = flow.name if flow else ""
        self.settings = dict(flow.settings) if flow else {}
        seconds = float(self.settings.get("max_run_seconds", 0) or 0)

        self.setWindowTitle(f"流程设置 · {self.flow_name}")
        self.resize(460, 260)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "启用后：流程运行到指定时间会自动结束；若在此之前触发了「结束流程」节点，则按结束节点结束。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self.enable_check = QCheckBox("启用「运行时间结束」")
        self.enable_check.setChecked(seconds > 0)
        form.addRow("", self.enable_check)
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(1.0, 86400.0)
        self.seconds.setSingleStep(10.0)
        self.seconds.setDecimals(1)
        self.seconds.setValue(seconds if seconds > 0 else 60.0)
        form.addRow("最长运行时间(秒)", self.seconds)
        layout.addLayout(form)
        layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.enable_check.toggled.connect(self.seconds.setEnabled)
        self.seconds.setEnabled(self.enable_check.isChecked())

    def _save(self) -> None:
        settings = dict(self.settings)
        if self.enable_check.isChecked():
            settings["max_run_seconds"] = float(self.seconds.value())
        else:
            settings["max_run_seconds"] = 0
        self.db.update_flow(self.flow_id, settings=settings)
        self.accept()
