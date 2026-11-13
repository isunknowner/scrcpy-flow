from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .. import config
from ..db import Database
from ..phone_export import _default_package, export_phone_package


def _slug(name: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", name or "flow", flags=re.UNICODE)
    return value.strip("._-") or "flow"


class PhoneExportDialog(QDialog):
    def __init__(self, db: Database, current_flow_id: int | None = None, parent=None):
        super().__init__(parent)
        self.db = db
        self.exported_path: Path | None = None
        self._auto_app = ""
        self._auto_package = ""
        self.setWindowTitle("导出手机文件")
        self.resize(620, 360)
        self._build(current_flow_id)

    def _build(self, current_flow_id: int | None) -> None:
        layout = QVBoxLayout(self)
        hint = QLabel(
            "导出为 .scflowpkg 手机流程包，可导入手机 APK 运行，"
            "也可以交给 flow-to-apk-sdk 构建成独立 APK。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self.target_combo = QComboBox()
        current_index = 0
        for project in self.db.list_projects():
            self.target_combo.addItem(f"项目：{project.name}", f"project:{project.id}")
        for flow in self.db.list_flows():
            index = self.target_combo.count()
            self.target_combo.addItem(f"流程：{flow.name}", f"flow:{flow.id}")
            if current_flow_id is not None and flow.id == current_flow_id:
                current_index = index
        if self.target_combo.count() == 0:
            self.target_combo.addItem("（没有可导出的流程或项目）", None)
        self.target_combo.setCurrentIndex(current_index)
        form.addRow("导出对象", self.target_combo)

        self.name_edit = QLineEdit()
        self.package_edit = QLineEdit()
        self.version_name = QLineEdit("1.0.0")
        self.version_code = QSpinBox()
        self.version_code.setRange(1, 2_000_000_000)
        self.version_code.setValue(1)
        form.addRow("应用名称", self.name_edit)
        form.addRow("包名", self.package_edit)
        form.addRow("版本名", self.version_name)
        form.addRow("版本号", self.version_code)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.browse_btn = QPushButton("选择…")
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse_btn)
        form.addRow("输出文件", path_row)
        layout.addLayout(form)

        note = QLabel(
            "导出后，可把 .scflowpkg 传到手机由手机运行时导入；"
            "或在 flow-to-apk-sdk 中执行 build_flow_apk.sh 生成独立 APK。"
        )
        note.setObjectName("subtitle")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("导出")
        layout.addWidget(self.buttons)

        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        self.browse_btn.clicked.connect(self._browse)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        self._on_target_changed()

    def _target_name(self) -> str:
        return self.target_combo.currentText().split("：", 1)[-1]

    def _on_target_changed(self) -> None:
        name = self._target_name()
        if not name or self.target_combo.currentData() is None:
            return
        new_app = name[:40]
        if not self.name_edit.text().strip() or self.name_edit.text() == self._auto_app:
            self.name_edit.setText(new_app)
            self._auto_app = new_app
        new_package = _default_package(new_app)
        if not self.package_edit.text().strip() or self.package_edit.text() == self._auto_package:
            self.package_edit.setText(new_package)
            self._auto_package = new_package
        if not self.path_edit.text().strip() or Path(self.path_edit.text()).name.startswith(_slug(self._auto_app)):
            self.path_edit.setText(str(config.exports_dir() / f"{_slug(new_app)}.scflowpkg"))

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出手机文件",
            self.path_edit.text() or str(config.exports_dir() / "flow.scflowpkg"),
            "手机流程包 (*.scflowpkg);;所有文件 (*)",
        )
        if path:
            self.path_edit.setText(path)

    def _accept(self) -> None:
        data = self.target_combo.currentData()
        if not data:
            QMessageBox.warning(self, "提示", "请选择要导出的流程或项目")
            return
        kind, _, raw = str(data).partition(":")
        try:
            target_id = int(raw)
        except ValueError:
            QMessageBox.warning(self, "提示", "导出对象无效")
            return
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请选择输出文件")
            return
        try:
            self.exported_path = export_phone_package(
                self.db,
                path,
                flow_id=target_id if kind == "flow" else None,
                project_id=target_id if kind == "project" else None,
                app_name=self.name_edit.text().strip() or None,
                package_name=self.package_edit.text().strip() or None,
                version_name=self.version_name.text().strip() or "1.0.0",
                version_code=self.version_code.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.accept()
