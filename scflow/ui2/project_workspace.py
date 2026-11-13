from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..db import Database
from ..undo import restore_project, snapshot_project
from .dialogs import VariableDialog, format_literal
from .fallback_dialog import FallbackDialog
from .scrollbar_bar import make_scrollable_bar
from .theme import ACCENT

FLOW_MIME = "application/x-scflow-flow-id"


class BlockListWidget(QListWidget):
    flow_dropped = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    @staticmethod
    def _flow_id(event) -> int | None:
        try:
            raw = bytes(event.mimeData().data(FLOW_MIME)).decode("utf-8")
            return int(raw)
        except (TypeError, ValueError):
            return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(FLOW_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(FLOW_MIME):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(FLOW_MIME):
            flow_id = self._flow_id(event)
            if flow_id is None:
                event.ignore()
                return
            self.flow_dropped.emit(flow_id)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class ProjectWorkspace(QWidget):
    open_flow_requested = Signal(int)
    project_changed = Signal()
    project_opened = Signal(int)
    status_message = Signal(str)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.projects = []
        self.project_id: int | None = None
        self.blocks = []
        self.project_vars: dict = {}
        self._loading = False
        self._dirty = False
        self._undo_stack: list[tuple[str, dict]] = []
        self._simple = False
        self.setAcceptDrops(True)
        self._build()
        self.refresh_projects()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("项目"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(260)
        bar.addWidget(self.project_combo)
        self.new_btn = QPushButton("新建项目")
        self.delete_btn = QPushButton("删除项目")
        self.save_btn = QPushButton("保存项目")
        self.save_btn.setObjectName("primary")
        self.undo_btn = QPushButton("撤回")
        self.refresh_btn = QPushButton("刷新")
        for button in (
            self.new_btn,
            self.delete_btn,
            self.save_btn,
            self.undo_btn,
            self.refresh_btn,
        ):
            bar.addWidget(button)
        bar.addStretch(1)
        self.mode_hint = QLabel("")
        self.mode_hint.setObjectName("subtitle")
        bar.addWidget(self.mode_hint)
        root.addWidget(make_scrollable_bar(bar))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("区块顺序（拖动排序；可从左侧工作区拖入流程）"))
        self.block_list = BlockListWidget()
        self.block_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.block_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.block_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.block_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.block_list.customContextMenuRequested.connect(self._on_block_context_menu)
        self.block_list.setSpacing(4)
        self.block_list.setStyleSheet(
            f"QListWidget::item {{ padding: 10px; border: 1px solid #333842; border-radius: 6px; }}"
            f"QListWidget::item:selected {{ border-color: {ACCENT}; background: #24405f; }}"
        )
        left_layout.addWidget(self.block_list, 1)
        add_row = QHBoxLayout()
        self.flow_combo = QComboBox()
        self.flow_combo.setMinimumWidth(180)
        self.flow_combo.setMaxVisibleItems(20)
        self.flow_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        add_row.addWidget(self.flow_combo, 1)
        self.add_block_btn = QPushButton("添加为区块")
        add_row.addWidget(self.add_block_btn)
        left_layout.addLayout(add_row)
        row = QHBoxLayout()
        self.up_btn = QPushButton("上移")
        self.down_btn = QPushButton("下移")
        self.remove_btn = QPushButton("移除")
        self.open_btn = QPushButton("打开区块画布")
        for button in (self.up_btn, self.down_btn, self.remove_btn, self.open_btn):
            row.addWidget(button)
        left_layout.addLayout(row)
        splitter.addWidget(left)

        self.right_tabs = QTabWidget()
        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        code_page = QWidget()
        code_layout = QVBoxLayout(code_page)
        code_layout.setContentsMargins(6, 6, 6, 6)
        tip = QLabel(
            "区块名会变成可调用的 Python 变量。示例：\n"
            "launch_app(\"com.xunmeng.pinduoduo\", fresh=True)\n"
            "if 首页():\n    点餐()\n"
            "while 次数 < 3:\n    选菜()\n    次数 += 1"
        )
        tip.setObjectName("subtitle")
        code_layout.addWidget(tip)
        self.code_edit = QPlainTextEdit()
        self.code_edit.setFont(QFont("JetBrains Mono", 11))
        self.code_edit.setTabStopDistance(32)
        code_layout.addWidget(self.code_edit, 1)
        code_buttons = QHBoxLayout()
        self.generate_btn = QPushButton("生成顺序脚本")
        self.check_btn = QPushButton("检查语法")
        code_buttons.addWidget(self.generate_btn)
        code_buttons.addWidget(self.check_btn)
        code_buttons.addStretch(1)
        code_layout.addLayout(code_buttons)
        self.right_tabs.addTab(code_page, "Python 编排")

        vars_page = QWidget()
        vars_layout = QVBoxLayout(vars_page)
        self.var_table = QTableWidget(0, 2)
        self.var_table.setHorizontalHeaderLabels(["变量名", "值"])
        self.var_table.horizontalHeader().setStretchLastSection(True)
        self.var_table.verticalHeader().setVisible(False)
        vars_layout.addWidget(self.var_table, 1)
        var_buttons = QHBoxLayout()
        self.add_var_btn = QPushButton("添加")
        self.edit_var_btn = QPushButton("编辑")
        self.delete_var_btn = QPushButton("删除")
        for button in (self.add_var_btn, self.edit_var_btn, self.delete_var_btn):
            var_buttons.addWidget(button)
        var_buttons.addStretch(1)
        vars_layout.addLayout(var_buttons)
        self.right_tabs.addTab(vars_page, "变量")

        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("按区块顺序执行", "sequence")
        self.mode_combo.addItem("使用 Python 编排", "python")
        self.stop_check = QCheckBox("区块未完成时停止")
        self.block_steps = QSpinBox()
        self.block_steps.setRange(1, 100000)
        self.block_steps.setValue(100)
        self.max_runs = QSpinBox()
        self.max_runs.setRange(1, 100000)
        self.max_runs.setValue(200)
        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(1, 86400)
        self.timeout.setValue(3600)
        form.addRow("执行模式", self.mode_combo)
        form.addRow("", self.stop_check)
        form.addRow("单区块最大步数", self.block_steps)
        form.addRow("最多区块执行次数", self.max_runs)
        form.addRow("脚本超时（秒）", self.timeout)
        settings_layout.addLayout(form)
        settings_layout.addStretch(1)
        self.right_tabs.addTab(settings_page, "运行设置")
        splitter.setSizes([430, 560])

        self.project_combo.currentIndexChanged.connect(self._on_project_combo)
        self.new_btn.clicked.connect(self._new_project)
        self.delete_btn.clicked.connect(self._delete_project)
        self.save_btn.clicked.connect(lambda: self.save(silent=False))
        self.undo_btn.clicked.connect(self._undo)
        self.refresh_btn.clicked.connect(lambda: self.refresh_projects(keep=True))
        self.add_block_btn.clicked.connect(self._add_block)
        self.remove_btn.clicked.connect(self._remove_block)
        self.up_btn.clicked.connect(lambda: self._move_block(-1))
        self.down_btn.clicked.connect(lambda: self._move_block(1))
        self.open_btn.clicked.connect(self._open_block)
        self.block_list.currentRowChanged.connect(lambda _row: self._update_block_hint())
        self.block_list.model().rowsMoved.connect(self._on_rows_moved)
        self.block_list.flow_dropped.connect(self._add_block_by_id)
        self.generate_btn.clicked.connect(self._generate_script)
        self.check_btn.clicked.connect(self._check_code)
        self.add_var_btn.clicked.connect(self._add_variable)
        self.edit_var_btn.clicked.connect(self._edit_variable)
        self.delete_var_btn.clicked.connect(self._delete_variable)
        self.code_edit.textChanged.connect(self._mark_dirty)
        self.mode_combo.currentIndexChanged.connect(self._mark_dirty)
        self.stop_check.toggled.connect(self._mark_dirty)
        self.block_steps.valueChanged.connect(self._mark_dirty)
        self.max_runs.valueChanged.connect(self._mark_dirty)
        self.timeout.valueChanged.connect(self._mark_dirty)

    @staticmethod
    def _flow_id_from_event(event) -> int | None:
        try:
            raw = bytes(event.mimeData().data(FLOW_MIME)).decode("utf-8")
            return int(raw)
        except (TypeError, ValueError):
            return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(FLOW_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(FLOW_MIME):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(FLOW_MIME):
            flow_id = self._flow_id_from_event(event)
            if flow_id is None:
                event.ignore()
                return
            self._add_block_by_id(flow_id)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def set_simple(self, simple: bool) -> None:
        self._simple = simple
        self.right_tabs.setVisible(not simple)
        self.undo_btn.setVisible(not simple)
        self.mode_hint.setText("简洁版：只看区块顺序与运行" if simple else "普通版")

    def refresh_projects(self, keep: bool = True, select_id: int | None = None) -> None:
        self._flush()
        self.projects = self.db.list_projects()
        current_id = select_id or (self.project_id if keep else None)
        self._loading = True
        self.project_combo.clear()
        for project in self.projects:
            self.project_combo.addItem(project.name, project.id)
        self.refresh_flow_choices()
        index = 0
        if current_id is not None:
            found = self.project_combo.findData(current_id)
            if found >= 0:
                index = found
        if self.projects:
            self.project_combo.setCurrentIndex(index)
            self._loading = False
            self.load_project(self.projects[index].id)
        else:
            self.project_id = None
            self.blocks = []
            self.block_list.clear()
            self.code_edit.clear()
            self.var_table.setRowCount(0)
            self._loading = False

    def refresh_flow_choices(self) -> None:
        current = self.flow_combo.currentData()
        self.flow_combo.blockSignals(True)
        self.flow_combo.clear()
        for flow in self.db.list_flows():
            self.flow_combo.addItem(flow.name, flow.id)
        if current is not None:
            index = self.flow_combo.findData(current)
            if index >= 0:
                self.flow_combo.setCurrentIndex(index)
        self.flow_combo.blockSignals(False)

    def _on_project_combo(self, _index: int) -> None:
        if self._loading:
            return
        self._flush()
        project_id = self.project_combo.currentData()
        if project_id is not None:
            self._undo_stack.clear()
            self.load_project(int(project_id))

    def load_project(self, project_id: int) -> None:
        project = self.db.get_project(project_id)
        if project is None:
            return
        self._loading = True
        self.project_id = project.id
        self.project_vars = dict(project.variables)
        self.code_edit.setPlainText(project.python_code)
        settings = project.settings
        mode = settings.get("mode", "sequence")
        index = self.mode_combo.findData(mode)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.stop_check.setChecked(bool(settings.get("stop_on_error", True)))
        self.block_steps.setValue(int(settings.get("block_max_steps", 100)))
        self.max_runs.setValue(int(settings.get("max_block_runs", 200)))
        self.timeout.setValue(float(settings.get("script_timeout", 3600)))
        self._refresh_variables()
        self._reload_blocks()
        self._loading = False
        self._dirty = False
        self._update_block_hint()
        self.project_opened.emit(project.id)

    def _reload_blocks(self) -> None:
        self.blocks = (
            self.db.list_project_blocks(self.project_id) if self.project_id is not None else []
        )
        selected = self.block_list.currentRow()
        self._loading = True
        self.block_list.clear()
        for index, block in enumerate(self.blocks):
            item = QListWidgetItem(
                f"{index + 1}.  {block.flow_name}\n"
                f"     变量名 {block.alias}   ·   {block.node_count} 个节点"
            )
            item.setData(Qt.ItemDataRole.UserRole, block.id)
            item.setSizeHint(QSize(0, 58))
            self.block_list.addItem(item)
        if self.blocks:
            self.block_list.setCurrentRow(min(max(0, selected), len(self.blocks) - 1))
        self._loading = False

    def _selected_block(self, index: int | None = None):
        row = self.block_list.currentRow() if index is None else index
        if 0 <= row < len(self.blocks):
            return self.blocks[row]
        return None

    def _on_block_context_menu(self, pos) -> None:
        item = self.block_list.itemAt(pos)
        if item is None:
            return
        row = self.block_list.row(item)
        self.block_list.setCurrentRow(row)
        block = self._selected_block(row)
        if block is None:
            return
        menu = QMenu(self)
        fallback_action = menu.addAction("编辑保底机制…")
        chosen = menu.exec(self.block_list.viewport().mapToGlobal(pos))
        if chosen is fallback_action:
            dialog = FallbackDialog(self.db, block.flow_id, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.status_message.emit(f"已保存「{block.flow_name}」的保底机制")

    def _update_block_hint(self) -> None:
        block = self._selected_block()
        if block:
            self.mode_hint.setText(
                f"区块 {self.block_list.currentRow() + 1}/{len(self.blocks)} · {block.flow_name}"
            )

    def _push_undo(self, label: str) -> None:
        if self.project_id is None:
            return
        snapshot = snapshot_project(self.db, self.project_id)
        if snapshot:
            self._undo_stack.append((label, snapshot))
            self._undo_stack = self._undo_stack[-50:]

    def _undo(self) -> None:
        if not self._undo_stack or self.project_id is None:
            self.status_message.emit("没有可撤回的操作")
            return
        label, snapshot = self._undo_stack.pop()
        self._dirty = False
        restore_project(self.db, self.project_id, snapshot)
        project_id = self.project_id
        self.refresh_projects(select_id=project_id)
        self.status_message.emit(f"已撤回：{label}")

    def _mark_dirty(self, *_args) -> None:
        if self._loading:
            return
        self._dirty = True

    def _flush(self) -> None:
        if self._dirty:
            self.save(silent=True)

    def save(self, silent: bool = False) -> None:
        if self.project_id is None:
            return
        project = self.db.get_project(self.project_id)
        if project is None:
            return
        if not silent:
            self._push_undo("保存项目")
        settings = dict(project.settings or {})
        settings["mode"] = self.mode_combo.currentData()
        settings["stop_on_error"] = self.stop_check.isChecked()
        settings["block_max_steps"] = self.block_steps.value()
        settings["max_block_runs"] = self.max_runs.value()
        settings["script_timeout"] = self.timeout.value()
        self.db.update_project(
            project.id,
            python_code=self.code_edit.toPlainText().rstrip("\n"),
            variables=self.project_vars,
            settings=settings,
        )
        self._dirty = False
        self.project_changed.emit()
        if not silent:
            self.status_message.emit("项目已保存")

    def _new_project(self) -> None:
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称：")
        if not ok or not name.strip():
            return
        name = self.db.unique_project_name(name.strip())
        project_id = self.db.create_project(name)
        self.refresh_projects(select_id=project_id)
        self.status_message.emit(f"已创建项目：{name}")
        self.project_changed.emit()

    def _delete_project(self) -> None:
        if self.project_id is None:
            return
        project = self.db.get_project(self.project_id)
        if project is None:
            return
        answer = QMessageBox.question(
            self, "删除项目", f"删除项目「{project.name}」？区块流程本身不会被删除。"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._undo_stack.clear()
        self.project_id = None
        self.db.delete_project(project.id)
        self.refresh_projects(keep=False)
        self.project_changed.emit()

    def _add_block(self) -> None:
        flow_id = self.flow_combo.currentData()
        if flow_id is None:
            return
        self._add_block_by_id(int(flow_id))

    def _add_block_by_id(self, flow_id: int) -> None:
        self._flush()
        if self.project_id is None:
            self.status_message.emit("请先选择一个项目")
            return
        flow = self.db.get_flow(int(flow_id))
        if flow is None:
            self.status_message.emit("流程不存在")
            return
        if any(block.flow_id == int(flow_id) for block in self.blocks):
            self.status_message.emit(f"流程「{flow.name}」已经在区块中")
            return
        self._push_undo("添加区块")
        try:
            self.db.add_project_block(self.project_id, int(flow_id))
        except Exception as exc:
            self.status_message.emit(f"添加区块失败：{exc}")
            return
        self._reload_blocks()
        if self.blocks:
            self.block_list.setCurrentRow(len(self.blocks) - 1)
        self.project_changed.emit()
        self.status_message.emit(f"已添加区块：{flow.name}")

    def _remove_block(self) -> None:
        self._flush()
        block = self._selected_block()
        if block is None:
            return
        answer = QMessageBox.question(
            self, "移除区块", f"从项目中移除「{block.flow_name}」？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._push_undo("移除区块")
        self.db.remove_project_block(block.id)
        self._reload_blocks()
        self.project_changed.emit()

    def _move_block(self, direction: int) -> None:
        self._flush()
        block = self._selected_block()
        if block is None:
            return
        ids = [item.id for item in self.blocks]
        index = ids.index(block.id)
        target = index + direction
        if not (0 <= target < len(ids)):
            return
        ids[index], ids[target] = ids[target], ids[index]
        self._push_undo("移动区块")
        self.db.reorder_project_blocks(self.project_id, ids)
        self._reload_blocks()
        self.block_list.setCurrentRow(target)
        self.project_changed.emit()

    def _on_rows_moved(self, *_args) -> None:
        if self._loading or self.project_id is None:
            return
        ids = [
            int(self.block_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.block_list.count())
        ]
        self._push_undo("拖动区块排序")
        self.db.reorder_project_blocks(self.project_id, ids)
        self._reload_blocks()
        self.project_changed.emit()

    def _open_block(self) -> None:
        block = self._selected_block()
        if block is None:
            return
        self._flush()
        self.open_flow_requested.emit(block.flow_id)

    def _generate_script(self) -> None:
        if not self.blocks:
            self.status_message.emit("先添加区块")
            return
        current = self.code_edit.toPlainText().strip()
        if current:
            answer = QMessageBox.question(
                self, "生成脚本", "当前脚本会被覆盖，继续？"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        lines = ["# 自动生成：按区块顺序执行，可继续添加 if / while", ""]
        for block in self.blocks:
            lines.append(f"{block.alias}()")
        self.code_edit.setPlainText("\n".join(lines) + "\n")
        index = self.mode_combo.findData("python")
        self.mode_combo.setCurrentIndex(index)
        self._mark_dirty()

    def _check_code(self) -> None:
        try:
            compile(self.code_edit.toPlainText(), "<项目编排>", "exec")
        except SyntaxError as exc:
            QMessageBox.critical(
                self, "语法错误", f"第 {exc.lineno} 行：{exc.msg}\n{exc.text or ''}"
            )
        else:
            QMessageBox.information(self, "检查通过", "Python 语法正确。")

    def _refresh_variables(self) -> None:
        self.var_table.setRowCount(len(self.project_vars))
        for row, (name, value) in enumerate(self.project_vars.items()):
            self.var_table.setItem(row, 0, QTableWidgetItem(str(name)))
            self.var_table.setItem(row, 1, QTableWidgetItem(format_literal(value)))
        self.var_table.resizeColumnsToContents()

    def _add_variable(self) -> None:
        dialog = VariableDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = dialog.result_value()
        if not result:
            return
        name, value = result
        if name in self.project_vars:
            QMessageBox.warning(self, "提示", "变量已存在")
            return
        self.project_vars[name] = value
        self._refresh_variables()
        self._mark_dirty()

    def _edit_variable(self) -> None:
        row = self.var_table.currentRow()
        if row < 0 or row >= len(self.project_vars):
            return
        old_name = self.var_table.item(row, 0).text()
        dialog = VariableDialog(
            self, old_name, format_literal(self.project_vars.get(old_name))
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = dialog.result_value()
        if not result:
            return
        name, value = result
        if name != old_name and name in self.project_vars:
            QMessageBox.warning(self, "提示", "变量名已存在")
            return
        self.project_vars.pop(old_name, None)
        self.project_vars[name] = value
        self._refresh_variables()
        self._mark_dirty()

    def _delete_variable(self) -> None:
        row = self.var_table.currentRow()
        if row < 0:
            return
        name = self.var_table.item(row, 0).text()
        self.project_vars.pop(name, None)
        self._refresh_variables()
        self._mark_dirty()
