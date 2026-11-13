from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..db import Database
from .image_editor import ImageEditor


class RecordingReviewDialog(QDialog):
    """Review and edit nodes created by the recorder."""

    def __init__(
        self,
        db: Database,
        flow_id: int,
        node_ids: list[int],
        flow_workspace,
        parent=None,
    ):
        super().__init__(parent)
        self.db = db
        self.flow_id = flow_id
        self.flow_workspace = flow_workspace
        self.order = [node_id for node_id in node_ids if db.get_node(node_id)]
        self.current_id: int | None = None
        self._loading = False
        self.setWindowTitle("编辑录制结果")
        self.resize(920, 620)
        self._build()
        self._refresh_list()
        if self.order:
            self.list.setCurrentRow(0)
            first = self.db.get_node(self.order[0])
            has_entry = any(
                node.is_entry for node in self.db.list_nodes(self.flow_id)
            )
            self.entry_check.setChecked(
                bool(first and first.is_entry) or not has_entry
            )

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        hint = QLabel(
            "左侧是录制顺序，右侧可修改名称、点击位置、动作和等待时间。"
            "「上移 / 下移」会重排录制节点并自动重连；也可以转到属性面板做完整编辑。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("录制节点（执行顺序）"))
        self.list = QListWidget()
        left_layout.addWidget(self.list, 1)
        move_row = QHBoxLayout()
        self.up_btn = QPushButton("上移")
        self.down_btn = QPushButton("下移")
        move_row.addWidget(self.up_btn)
        move_row.addWidget(self.down_btn)
        move_row.addStretch(1)
        left_layout.addLayout(move_row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.image_editor = ImageEditor()
        self.image_editor.setMinimumHeight(230)
        self.image_editor.set_mode("click")
        right_layout.addWidget(self.image_editor, 1)
        self.image_hint = QLabel("在图片上点击可修改点击位置")
        self.image_hint.setObjectName("subtitle")
        right_layout.addWidget(self.image_hint)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.click_x = QDoubleSpinBox()
        self.click_x.setRange(0.0, 100.0)
        self.click_x.setDecimals(2)
        self.click_x.setSuffix(" %")
        self.click_y = QDoubleSpinBox()
        self.click_y.setRange(0.0, 100.0)
        self.click_y.setDecimals(2)
        self.click_y.setSuffix(" %")
        self.action_combo = QComboBox()
        self.action_combo.addItem("点击", "tap")
        self.action_combo.addItem("无动作", "none")
        self.action_combo.addItem("等待", "wait")
        self.wait_seconds = QDoubleSpinBox()
        self.wait_seconds.setRange(0.0, 600.0)
        self.wait_seconds.setDecimals(2)
        self.wait_seconds.setSuffix(" 秒")
        self.post_wait = QDoubleSpinBox()
        self.post_wait.setRange(0.0, 60.0)
        self.post_wait.setDecimals(2)
        self.post_wait.setSuffix(" 秒")
        self.notes = QLineEdit()
        form.addRow("名称", self.name_edit)
        form.addRow("点击 X", self.click_x)
        form.addRow("点击 Y", self.click_y)
        form.addRow("动作", self.action_combo)
        form.addRow("等待秒数", self.wait_seconds)
        form.addRow("动作后等待", self.post_wait)
        form.addRow("备注", self.notes)
        right_layout.addLayout(form)
        self.entry_check = QCheckBox("把第一个录制节点设为流程入口")
        right_layout.addWidget(self.entry_check)

        buttons = QHBoxLayout()
        self.delete_btn = QPushButton("删除节点")
        self.delete_btn.setObjectName("danger")
        self.inspector_btn = QPushButton("在属性面板中编辑")
        self.save_btn = QPushButton("保存当前节点")
        self.finish_btn = QPushButton("完成")
        self.finish_btn.setObjectName("primary")
        buttons.addWidget(self.delete_btn)
        buttons.addWidget(self.inspector_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.finish_btn)
        right_layout.addLayout(buttons)
        splitter.addWidget(right)
        splitter.setSizes([300, 620])

        self.list.currentRowChanged.connect(self._on_select)
        self.up_btn.clicked.connect(lambda: self._move(-1))
        self.down_btn.clicked.connect(lambda: self._move(1))
        self.delete_btn.clicked.connect(self._delete)
        self.inspector_btn.clicked.connect(self._open_inspector)
        self.save_btn.clicked.connect(self._save_current)
        self.finish_btn.clicked.connect(self._finish)
        self.image_editor.point_selected.connect(self._on_image_click)
        self.action_combo.currentIndexChanged.connect(self._update_action_fields)
        self._update_action_fields()

    def _refresh_list(self) -> None:
        current = self.current_id
        self._loading = True
        self.list.clear()
        for index, node_id in enumerate(self.order, start=1):
            node = self.db.get_node(node_id)
            if node is None:
                continue
            item = QListWidgetItem(f"{index}. {node.name}")
            item.setData(Qt.ItemDataRole.UserRole, node_id)
            self.list.addItem(item)
        self._loading = False
        if current in self.order:
            self.list.setCurrentRow(self.order.index(current))
        elif self.order:
            self.list.setCurrentRow(0)
        else:
            self._clear_form()

    def _selected_id(self) -> int | None:
        item = self.list.currentItem()
        return None if item is None else int(item.data(Qt.ItemDataRole.UserRole))

    def _on_select(self, _row: int) -> None:
        if self._loading:
            return
        self._save_current(silent=True)
        node_id = self._selected_id()
        self.current_id = node_id
        if node_id is None:
            self._clear_form()
            return
        node = self.db.get_node(node_id)
        if node is None:
            return
        self._loading = True
        self.name_edit.setText(node.name)
        self.click_x.setValue(float(node.click_x or 0.5) * 100.0)
        self.click_y.setValue(float(node.click_y or 0.5) * 100.0)
        action = node.action or {}
        index = self.action_combo.findData(node.action_type)
        self.action_combo.setCurrentIndex(index if index >= 0 else 0)
        try:
            self.wait_seconds.setValue(float(action.get("seconds", 1) or 1))
        except (TypeError, ValueError):
            self.wait_seconds.setValue(1.0)
        try:
            self.post_wait.setValue(float(action.get("post_wait", 0.6) or 0.6))
        except (TypeError, ValueError):
            self.post_wait.setValue(0.6)
        self.notes.setText(node.notes or "")
        if node.image_path and Path(node.image_path).exists():
            self.image_editor.set_image(QPixmap(node.image_path))
            self.image_editor.set_markers((node.click_x, node.click_y), None, None, None)
            self.image_editor.fit_image()
        else:
            self.image_editor.set_image(None)
        self._update_action_fields()
        self._loading = False

    def _clear_form(self) -> None:
        self._loading = True
        self.name_edit.clear()
        self.click_x.setValue(50.0)
        self.click_y.setValue(50.0)
        self.wait_seconds.setValue(1.0)
        self.post_wait.setValue(0.6)
        self.notes.clear()
        self.image_editor.set_image(None)
        self._loading = False

    def _update_action_fields(self) -> None:
        kind = self.action_combo.currentData()
        self.wait_seconds.setEnabled(kind == "wait")
        clickable = kind == "tap"
        self.click_x.setEnabled(clickable)
        self.click_y.setEnabled(clickable)

    def _on_image_click(self, x: float, y: float) -> None:
        self.click_x.setValue(max(0.0, min(1.0, x)) * 100.0)
        self.click_y.setValue(max(0.0, min(1.0, y)) * 100.0)
        self.image_editor.set_markers((x, y), None, None, None)

    def _save_current(self, silent: bool = False) -> None:
        if self._loading or self.current_id is None:
            return
        node = self.db.get_node(self.current_id)
        if node is None:
            return
        action = dict(node.action or {})
        kind = self.action_combo.currentData()
        if kind == "wait":
            action["seconds"] = float(self.wait_seconds.value())
            action.pop("post_wait", None)
        else:
            action.pop("seconds", None)
        action["post_wait"] = float(self.post_wait.value())
        self.db.update_node(
            self.current_id,
            name=self.name_edit.text().strip() or node.name,
            click_x=float(self.click_x.value()) / 100.0,
            click_y=float(self.click_y.value()) / 100.0,
            action_type=str(kind),
            action=action,
            notes=self.notes.text(),
        )
        if not silent:
            self.flow_workspace.inspector.load_node(self.current_id, force=True)
            self.flow_workspace._reload_graph()
            self._refresh_list()

    def _move(self, delta: int) -> None:
        row = self.list.currentRow()
        target = row + delta
        if row < 0 or not (0 <= target < len(self.order)):
            return
        self._save_current(silent=True)
        node_id = self.order.pop(row)
        self.order.insert(target, node_id)
        self._rebuild_edges()
        self._refresh_list()
        self.list.setCurrentRow(target)

    def _delete(self) -> None:
        node_id = self._selected_id()
        if node_id is None:
            return
        node = self.db.get_node(node_id)
        if node is None:
            return
        answer = QMessageBox.question(
            self, "删除录制节点", f"删除节点「{node.name}」？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.db.delete_node(node_id)
        if node_id in self.order:
            self.order.remove(node_id)
        self.current_id = None
        self._rebuild_edges()
        self._refresh_list()
        self.flow_workspace._reload_graph()
        self.flow_workspace.flow_changed.emit()

    def _rebuild_edges(self) -> None:
        recorded = set(self.order)
        for edge in self.db.list_edges(self.flow_id):
            if edge.src_id in recorded and edge.dst_id in recorded:
                self.db.delete_edge(edge.id)
        for src, dst in zip(self.order, self.order[1:]):
            self.db.add_edge(self.flow_id, src, dst, 1.0, "录制")
        self.flow_workspace._reload_graph()
        self.flow_workspace.flow_changed.emit()

    def _open_inspector(self) -> None:
        node_id = self._selected_id()
        if node_id is None:
            return
        self._save_current(silent=True)
        self.flow_workspace.select_node(node_id)
        self.accept()

    def _finish(self) -> None:
        self._save_current(silent=True)
        self._rebuild_edges()
        if self.entry_check.isChecked() and self.order:
            self.db.set_entry(self.flow_id, self.order[0])
        self.flow_workspace._reload_graph()
        self.flow_workspace.flow_changed.emit()
        self.accept()
