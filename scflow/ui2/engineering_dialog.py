from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from ..db import Database


class EngineeringExportDialog(QDialog):
    """Choose which projects/flows go into an engineering file."""

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("导出工程文件")
        self.resize(640, 480)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "一个工程文件可以同时包含「项目」和「流程」。接收方用「文件 → 打开工程」即可导入。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        project_box = QGroupBox("包含的项目")
        project_layout = QVBoxLayout(project_box)
        self.project_list = QListWidget()
        project_layout.addWidget(self.project_list)
        row.addWidget(project_box, 1)

        flow_box = QGroupBox("包含的流程")
        flow_layout = QVBoxLayout(flow_box)
        self.flow_list = QListWidget()
        flow_layout.addWidget(self.flow_list)
        row.addWidget(flow_box, 1)
        layout.addLayout(row, 1)

        self.pack_check = QCheckBox(
            "打包图片到工程文件（推荐；接收方无需原图，但文件更大）"
        )
        self.pack_check.setChecked(True)
        layout.addWidget(self.pack_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("下一步：保存工程文件")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate()

    def _populate(self) -> None:
        for project in self.db.list_projects():
            item = QListWidgetItem(project.name)
            item.setData(Qt.ItemDataRole.UserRole, ("project", project.id))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.project_list.addItem(item)
        for flow in self.db.list_flows():
            item = QListWidgetItem(flow.name)
            item.setData(Qt.ItemDataRole.UserRole, ("flow", flow.id))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.flow_list.addItem(item)

    def selection(self) -> tuple[list[int], list[int], bool]:
        project_ids = []
        for row in range(self.project_list.count()):
            item = self.project_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                data = item.data(Qt.ItemDataRole.UserRole)
                if data:
                    project_ids.append(int(data[1]))
        flow_ids = []
        for row in range(self.flow_list.count()):
            item = self.flow_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                data = item.data(Qt.ItemDataRole.UserRole)
                if data:
                    flow_ids.append(int(data[1]))
        return flow_ids, project_ids, self.pack_check.isChecked()
