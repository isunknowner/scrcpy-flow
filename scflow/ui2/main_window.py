from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QMimeData, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStyle,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, config
from ..db import Database
from ..engineering import export_engineering, import_engineering
from .ai_panel import AIAssistantWidget
from .cloud_panel import CloudAIPanel
from .asset_library import AssetLibraryWidget
from .engineering_dialog import EngineeringExportDialog
from .engineering_manager import EngineeringManager
from .error_report import build_error_report
from .usage_guide import UsageGuideDialog
from .flow_workspace import FlowWorkspace
from .fallback_dialog import FallbackDialog
from .flow_settings_dialog import FlowSettingsDialog
from .history_workspace import HistoryWorkspace
from .personalization import PersonalizationDialog, PersonalizationManager
from .phone_export_dialog import PhoneExportDialog
from .pro_workspace import ProWorkspace
from .project_workspace import FLOW_MIME, ProjectWorkspace
from .run_workspace import RunWorkspace
from .scrollbar_bar import make_scrollable_bar
from .similarity_workspace import SimilarityWorkspace
from .theme import apply_theme
from .wheel_guard import WheelFocusGuard


class CommandPalette(QDialog):
    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("命令面板")
        self.resize(620, 440)
        self.actions = actions
        self.selected = None
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("输入命令…  Ctrl+Shift+P")
        layout.addWidget(self.search)
        self.list = QListWidget()
        self.list.setSpacing(2)
        layout.addWidget(self.list, 1)
        hint = QLabel("Enter 执行 · Esc 关闭")
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        self.search.textChanged.connect(self._filter)
        self.list.itemActivated.connect(self._activate)
        self.search.returnPressed.connect(self._activate_current)
        self._populate("")

    def _populate(self, query: str) -> None:
        self.list.clear()
        query = query.lower().strip()
        for label, shortcut, callback in self.actions:
            text = f"{label}    {shortcut}".strip()
            if query and query not in text.lower():
                continue
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, callback)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _filter(self, text: str) -> None:
        self._populate(text)

    def _activate_current(self) -> None:
        item = self.list.currentItem()
        if item:
            self._activate(item)

    def _activate(self, item: QListWidgetItem) -> None:
        callback = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
        if callback:
            QTimer.singleShot(0, callback)


class WorkspaceTree(QTreeWidget):
    """Tree that only allows reordering items within the same root group."""

    reordered = Signal()

    def mimeData(self, items):  # noqa: N802 (Qt naming)
        data = super().mimeData(items)
        if data is None:
            data = QMimeData()
        if items:
            payload = items[0].data(0, Qt.ItemDataRole.UserRole)
            if isinstance(payload, tuple) and len(payload) >= 2 and payload[0] == "flow":
                data.setData(FLOW_MIME, str(payload[1]).encode("utf-8"))
        return data

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        sources = self.selectedItems()
        if not sources:
            event.ignore()
            return
        source = sources[0]
        parent = source.parent()
        if parent is None:
            event.ignore()
            return
        target = self.itemAt(event.position().toPoint())
        if target is None:
            event.ignore()
            return
        if target is parent:
            # Dropped on the group header: append within the same group.
            pass
        elif target.parent() is None or target.parent() is not parent:
            event.ignore()
            return
        elif self.dropIndicatorPosition() == QAbstractItemView.DropIndicatorPosition.OnItem:
            event.ignore()
            return
        super().dropEvent(event)
        self.reordered.emit()


class MainWindow(QMainWindow):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.settings = QSettings("scrcpy-flow", "workspace-2")
        self._mode = self._load_mode()
        self._pro = self._mode in ("normal", "pro")
        self.setWindowTitle(f"scrcpy-flow 1.0 · 工作区 v{__version__}")
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

        self.project_workspace = ProjectWorkspace(db)
        self.flow_workspace = FlowWorkspace(db)
        self.run_workspace = RunWorkspace(db)
        self.history_workspace = HistoryWorkspace(db)
        self.asset_library = AssetLibraryWidget(self.flow_workspace)
        self.pro_workspace = ProWorkspace(db, self.flow_workspace)
        self.ai_panel = AIAssistantWidget(db, self.flow_workspace)
        self.ai_panel.open_flow_callback = self._show_flow_canvas
        self.ai_panel.open_project_callback = self.open_project_by_id
        self.ai_panel.status_message.connect(self.show_status)
        self.cloud_panel = CloudAIPanel(db, self.flow_workspace)
        self.cloud_panel.open_flow_callback = self._show_flow_canvas
        self.cloud_panel.open_project_callback = self.open_project_by_id
        self.cloud_panel.status_message.connect(self.show_status)
        self.similarity_workspace = SimilarityWorkspace(db, self.flow_workspace)
        self.floating_run_window = None

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.project_workspace, "区块编排")
        self.tabs.addTab(self.flow_workspace, "流程画布")
        self.tabs.addTab(self.run_workspace, "运行")
        self.tabs.addTab(self.history_workspace, "历史")
        self.tabs.addTab(self.similarity_workspace, "相似度编辑")
        self.tabs.addTab(self.pro_workspace, "专业工具")
        self.setCentralWidget(self.tabs)

        self._build_docks()

        self._bg_widgets = {
            "project": self.project_workspace,
            "flow": self.flow_workspace.graph,
            "run": self.run_workspace,
            "history": self.history_workspace,
            "inspector": self.flow_workspace.inspector,
            "assets": self.asset_library,
            "console": self.console,
            "ai": self.ai_panel,
        }
        for key, widget in self._bg_widgets.items():
            widget.setObjectName(f"bg_{key}")
        self.personalization = PersonalizationManager(self._bg_widgets, self)

        self._build_actions()
        self._build_toolbar()
        self._wire()
        self._apply_mode(self._mode)
        self.refresh_all()
        self.run_workspace.set_pre_run_callback(self._flush_editors)

        self._wheel_guard = WheelFocusGuard(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._wheel_guard)

        self.engineerings = EngineeringManager()
        saved = self.engineerings.current()
        if (
            saved
            and Path(saved).is_dir()
            and Path(saved).resolve() != Path(config.home()).resolve()
        ):
            self._switch_engineering(Path(saved), announce=False)
        else:
            self._register_current_engineering()

        self._fit_to_screen()
        saved_state = self.settings.value("dock_state")
        if saved_state:
            try:
                self.restoreState(saved_state)
            except Exception:
                pass
        self.statusBar().showMessage(
            f"工程：{config.home()}    ·    {self._mode_label()}"
        )

    def _load_mode(self) -> str:
        saved = self.settings.value("ui_mode", "", type=str)
        if saved in ("concise", "normal", "pro"):
            return saved
        old = self.settings.value("pro_mode", None)
        if old is None:
            return "normal"
        return "normal" if self.settings.value("pro_mode", True, type=bool) else "concise"

    def _mode_label(self) -> str:
        return {"concise": "简洁版", "normal": "普通版", "pro": "专业版"}.get(
            self._mode, "普通版"
        )

    def _build_docks(self) -> None:
        self.workspace_dock = QDockWidget("工作区", self)
        self.workspace_dock.setObjectName("workspace_dock")
        self.workspace_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        workspace_widget = QWidget()
        layout = QVBoxLayout(workspace_widget)
        layout.setContentsMargins(6, 6, 6, 6)
        self.tree = WorkspaceTree()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tree.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        layout.addWidget(self.tree, 1)
        row = QHBoxLayout()
        self.tree_new_project = QPushButton("新建项目")
        self.tree_new_flow = QPushButton("新建流程")
        self.tree_refresh = QPushButton("刷新")
        row.addWidget(self.tree_new_project)
        row.addWidget(self.tree_new_flow)
        row.addWidget(self.tree_refresh)
        layout.addWidget(make_scrollable_bar(row))
        self.workspace_dock.setWidget(workspace_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.workspace_dock)

        self.inspector_dock = QDockWidget("属性 / 检测重点", self)
        self.inspector_dock.setObjectName("inspector_dock")
        self.inspector_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.flow_workspace.inspector.setMinimumWidth(260)
        self.inspector_dock.setMinimumWidth(270)
        self.inspector_dock.setWidget(self.flow_workspace.inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector_dock)

        self.asset_dock = QDockWidget("素材库", self)
        self.asset_dock.setObjectName("asset_dock")
        self.asset_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.asset_dock.setMinimumWidth(210)
        self.asset_dock.setWidget(self.asset_library)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.asset_dock)
        self.tabifyDockWidget(self.inspector_dock, self.asset_dock)
        self.inspector_dock.raise_()
        self.asset_library.status_message.connect(self.show_status)

        self.console_dock = QDockWidget("输出 / 控制台", self)
        self.console_dock.setObjectName("console_dock")
        self.console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        self.console.setMinimumHeight(60)
        self.console_dock.setWidget(self.console)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)

        self.ai_dock = QDockWidget("AI 助手（本地 Ollama）", self)
        self.ai_dock.setObjectName("ai_dock")
        self.ai_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.ai_dock.setWidget(self._scroll_panel(self.ai_panel))
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.ai_dock)
        self.cloud_dock = QDockWidget("AI 助手（云端 API）", self)
        self.cloud_dock.setObjectName("cloud_dock")
        self.cloud_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.cloud_dock.setWidget(self._scroll_panel(self.cloud_panel))
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.cloud_dock)
        self.tabifyDockWidget(self.console_dock, self.ai_dock)
        self.tabifyDockWidget(self.ai_dock, self.cloud_dock)
        self.console_dock.raise_()

        self.resizeDocks([self.console_dock], [150], Qt.Orientation.Vertical)
        self.resizeDocks([self.workspace_dock], [240], Qt.Orientation.Horizontal)
        self.resizeDocks([self.inspector_dock], [400], Qt.Orientation.Horizontal)

    @staticmethod
    def _scroll_panel(widget: QWidget, min_height: int = 120) -> QScrollArea:
        """把面板包进滚动区域：dock 可以自由缩放，内容放不下时出滚动条。"""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setMinimumHeight(min_height)
        area.setWidget(widget)
        return area

    def _build_actions(self) -> None:
        self.action_new_project = QAction("新建项目", self)
        self.action_new_project.triggered.connect(self.project_workspace._new_project)
        self.action_new_flow = QAction("新建流程", self)
        self.action_new_flow.triggered.connect(self.flow_workspace._new_flow)
        self.action_save = QAction("保存全部", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._save_all)
        self.action_run = QAction("开始运行", self)
        self.action_run.setShortcut("F5")
        self.action_run.triggered.connect(self._start_run)
        self.action_stop = QAction("停止", self)
        self.action_stop.setShortcut("Shift+F5")
        self.action_stop.triggered.connect(self.run_workspace.stop)
        self.action_palette = QAction("命令面板", self)
        self.action_palette.setShortcut("Ctrl+Shift+P")
        self.action_palette.triggered.connect(self.show_palette)
        self.action_mode_group = QActionGroup(self)
        self.action_mode_group.setExclusive(True)
        self.action_mode_concise = QAction("简洁版", self, checkable=True)
        self.action_mode_normal = QAction("普通版", self, checkable=True)
        self.action_mode_pro = QAction("专业版", self, checkable=True)
        for action, mode in (
            (self.action_mode_concise, "concise"),
            (self.action_mode_normal, "normal"),
            (self.action_mode_pro, "pro"),
        ):
            action.setData(mode)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=mode: self._set_mode(value)
            )
            self.action_mode_group.addAction(action)
        self.action_demo = QAction("运行示例项目", self)
        self.action_demo.triggered.connect(lambda: self.run_project("4TB SSD 自动比价"))
        self.action_float_inspector = QAction("浮动属性窗口", self)
        self.action_float_inspector.triggered.connect(
            lambda: self._float_dock(self.inspector_dock)
        )
        self.action_float_workspace = QAction("浮动工作区窗口", self)
        self.action_float_workspace.triggered.connect(
            lambda: self._float_dock(self.workspace_dock)
        )
        self.action_float_console = QAction("浮动控制台窗口", self)
        self.action_float_console.triggered.connect(
            lambda: self._float_dock(self.console_dock)
        )
        self.action_float_run = QAction("独立运行窗口", self)
        self.action_float_run.triggered.connect(self.open_floating_run)
        self.action_restore_view = QAction("恢复默认设置", self)
        self.action_restore_view.triggered.connect(self._restore_default_view)

        menu = self.menuBar().addMenu("文件")
        menu.addAction(self.action_new_project)
        menu.addAction(self.action_new_flow)
        menu.addAction(self.action_save)
        menu.addSeparator()
        menu.addAction("新建工程…", self._create_engineering)
        menu.addAction("打开工程文件夹…", self._open_engineering_dir)
        self.switch_engineering_menu = QMenu("切换工程", self)
        self.switch_engineering_menu.aboutToShow.connect(self._populate_switch_menu)
        menu.addMenu(self.switch_engineering_menu)
        menu.addSeparator()
        menu.addAction("导出工程文件…", self._new_engineering)
        menu.addAction("打开工程文件…（新建独立工程）", self._open_engineering)
        menu.addSeparator()
        menu.addAction("导出手机文件…", self._export_phone)
        menu.addSeparator()
        menu.addAction("清理缓存…", self._clear_cache)
        menu.addAction("退出", self.close)
        menu = self.menuBar().addMenu("编辑")
        menu.addAction(self.flow_workspace.undo_btn.text(), self.flow_workspace._undo)
        menu.addAction(self.flow_workspace.redo_btn.text(), self.flow_workspace._redo)
        menu.addAction("删除选中", self.flow_workspace._delete_selected)
        menu.addAction("命令面板", self.show_palette)
        menu = self.menuBar().addMenu("视图")
        mode_menu = menu.addMenu("界面模式")
        mode_menu.addAction(self.action_mode_concise)
        mode_menu.addAction(self.action_mode_normal)
        mode_menu.addAction(self.action_mode_pro)
        menu.addSeparator()
        menu.addAction(self.action_float_workspace)
        menu.addAction(self.action_float_inspector)
        menu.addAction(self.asset_dock.toggleViewAction())
        menu.addAction(self.action_float_console)
        menu.addAction(self.action_float_run)
        menu.addSeparator()
        menu.addAction(self.action_restore_view)
        menu = self.menuBar().addMenu("运行")
        menu.addAction(self.action_run)
        menu.addAction(self.action_stop)
        menu.addSeparator()
        menu.addAction(self.action_demo)
        menu = self.menuBar().addMenu("AI")
        menu.addAction(self.ai_dock.toggleViewAction())
        menu.addAction(self.cloud_dock.toggleViewAction())
        menu.addAction("AI 设置…", self.ai_panel._open_settings)
        menu.addAction("AI 生成流程 / 项目…", self.ai_panel._generate_flow)
        menu.addSeparator()
        menu.addAction("分析当前画面", self.ai_panel._describe_screen)
        menu.addAction("刷新本地模型状态", self.ai_panel.refresh_status)
        menu = self.menuBar().addMenu("帮助")
        menu.addAction("使用说明", self._show_usage_guide)
        menu.addAction("导出错误日志…", self._export_error_log)
        menu.addSeparator()
        menu.addAction("关于", self._about)
        menu = self.menuBar().addMenu("个性化")
        menu.addAction("个性化设置…", self._open_personalization)
        menu.addAction("清除自定义背景", self._clear_personalization)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(True)
        self.addToolBar(toolbar)
        style = self.style()
        self.action_run.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.action_stop.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.action_save.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.action_palette.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        toolbar.addAction(self.action_new_project)
        toolbar.addAction(self.action_new_flow)
        toolbar.addAction(self.action_save)
        toolbar.addSeparator()
        toolbar.addAction(self.action_run)
        toolbar.addAction(self.action_stop)
        toolbar.addSeparator()
        toolbar.addAction(self.action_palette)
        toolbar.addSeparator()
        toolbar.addAction(self.asset_dock.toggleViewAction())
        toolbar.addSeparator()
        toolbar.addAction(self.action_demo)
        toolbar.addSeparator()
        self.mode_badge = QLabel("普通版")
        self.mode_badge.setObjectName("badge")
        toolbar.addWidget(self.mode_badge)
        toolbar.addWidget(QLabel("界面"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("简洁版", "concise")
        self.mode_combo.addItem("普通版", "normal")
        self.mode_combo.addItem("专业版", "pro")
        self.mode_combo.setMinimumWidth(100)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_combo)
        toolbar.addWidget(self.mode_combo)

    def _wire(self) -> None:
        self.tree.itemDoubleClicked.connect(self._on_tree_open)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.tree.reordered.connect(self._on_tree_reordered)
        self.tree_new_project.clicked.connect(self.project_workspace._new_project)
        self.tree_new_flow.clicked.connect(self.flow_workspace._new_flow)
        self.tree_refresh.clicked.connect(self.refresh_all)
        self.project_workspace.status_message.connect(self.show_status)
        self.flow_workspace.status_message.connect(self.show_status)
        self.run_workspace.status_message.connect(self.show_status)
        self.project_workspace.project_changed.connect(self._on_data_changed)
        self.project_workspace.project_opened.connect(self._on_project_opened)
        self.flow_workspace.flow_changed.connect(self._on_data_changed)
        self.project_workspace.open_flow_requested.connect(self.open_flow_by_id)
        self.run_workspace.running_changed.connect(self._on_running_changed)
        self.run_workspace.data_changed.connect(self._on_manual_nodes_changed)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.run_workspace.device_combo.currentIndexChanged.connect(self._sync_serial)
        self.flow_workspace.inspector.assets_changed.connect(self.asset_library.refresh)
        self.flow_workspace.inspector.edit_similarity_requested.connect(self._open_similarity_editor)
        self.flow_workspace.inspector.node_saved.connect(self._on_node_saved_similarity)
        self.flow_workspace.similarity_requested.connect(self._open_similarity_editor)
        self.similarity_workspace.back_requested.connect(self._back_to_flow)
        self.similarity_workspace.status_message.connect(self.show_status)
        self._sync_serial()

    def _sync_serial(self) -> None:
        serial = self.run_workspace.device_combo.currentData()
        self.flow_workspace.inspector.serial = serial
        self.asset_library.set_serial(serial)
        self.ai_panel.set_serial(serial)
        self.cloud_panel.set_serial(serial)

    def _show_flow_canvas(self, flow_id: int) -> None:
        self.open_flow_by_id(flow_id)
        self.tabs.setCurrentWidget(self.flow_workspace)

    def _on_project_opened(self, project_id: int) -> None:
        blocks = self.db.list_project_blocks(project_id)
        if blocks and self.flow_workspace.flow_id != blocks[0].flow_id:
            self.flow_workspace.refresh_flows(select_id=blocks[0].flow_id)

    def _on_data_changed(self) -> None:
        self.refresh_tree()

    def _on_manual_nodes_changed(self) -> None:
        """运行中记住了人工动作：刷新流程画布，让节点立刻可见。"""
        self.refresh_tree()
        try:
            self.flow_workspace.refresh_flows(
                keep=True, select_id=self.flow_workspace.flow_id
            )
        except Exception:
            pass
        self.project_workspace.refresh_flow_choices()
        self.run_workspace.refresh_targets()
        if self.floating_run_window is not None:
            run = self.floating_run_window.centralWidget()
            if isinstance(run, RunWorkspace):
                run.refresh_targets()
        self.history_workspace.refresh()

    def _on_tree_open(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, target_id = data
        if kind == "project":
            self.open_project_by_id(target_id)
        elif kind == "flow":
            self.open_flow_by_id(target_id)

    def _on_tree_reordered(self) -> None:
        project_ids: list[int] = []
        flow_ids: list[int] = []
        for index in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(index)
            data = root.data(0, Qt.ItemDataRole.UserRole)
            if not data or data[0] != "root":
                continue
            target = project_ids if data[1] == "projects" else flow_ids
            for child_index in range(root.childCount()):
                child = root.child(child_index)
                child_data = child.data(0, Qt.ItemDataRole.UserRole)
                if child_data:
                    target.append(int(child_data[1]))
        if project_ids:
            self.db.reorder_projects(project_ids)
        if flow_ids:
            self.db.reorder_flows(flow_ids)
        self.refresh_all()
        self.show_status("已调整项目/流程顺序")

    def _on_tree_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or data[0] == "root":
            return
        kind, target_id = data
        self.tree.setCurrentItem(item)
        menu = QMenu(self)
        rename_action = menu.addAction("重命名")
        delete_action = menu.addAction("删除")
        menu.addSeparator()
        fallback_action = menu.addAction("保底机制…")
        flow_settings_action = None
        if kind == "flow":
            flow_settings_action = menu.addAction("流程设置…")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is rename_action:
            if kind == "project":
                self._rename_project(int(target_id), item.text(0))
            elif kind == "flow":
                self._rename_flow(int(target_id), item.text(0))
        elif chosen is delete_action:
            if kind == "project":
                self._delete_project(int(target_id), item.text(0))
            elif kind == "flow":
                self._delete_flow(int(target_id), item.text(0))
        elif chosen is fallback_action:
            if kind == "flow":
                self._open_flow_fallback(int(target_id))
            elif kind == "project":
                self._open_project_fallback(int(target_id))
        elif flow_settings_action is not None and chosen is flow_settings_action:
            self._open_flow_settings(int(target_id))

    def _rename_project(self, project_id: int, current: str) -> None:
        name, ok = QInputDialog.getText(self, "重命名项目", "项目名称：", text=current)
        if not ok:
            return
        name = name.strip()
        if not name or name == current:
            return
        name = self.db.unique_project_name(name)
        self.db.update_project(project_id, name=name)
        self.refresh_all()
        self.show_status(f"项目已重命名：{name}")

    def _rename_flow(self, flow_id: int, current: str) -> None:
        name, ok = QInputDialog.getText(self, "重命名流程", "流程名称：", text=current)
        if not ok:
            return
        name = name.strip()
        if not name or name == current:
            return
        name = self.db.unique_flow_name(name)
        self.db.update_flow(flow_id, name=name)
        self.refresh_all()
        self.show_status(f"流程已重命名：{name}")

    def _delete_project(self, project_id: int, name: str) -> None:
        answer = QMessageBox.question(
            self, "删除项目", f"删除项目「{name}」？项目内的流程不会被删除。"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.project_workspace.project_id == project_id:
            self.project_workspace.save(silent=True)
            self.project_workspace.project_id = None
        self.db.delete_project(project_id)
        self.refresh_all()
        self.show_status(f"已删除项目：{name}")

    def _delete_flow(self, flow_id: int, name: str) -> None:
        answer = QMessageBox.question(
            self,
            "删除流程",
            f"删除流程「{name}」及其所有节点、连线、运行记录？此操作不可撤销。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.flow_workspace.flow_id == flow_id:
            self.flow_workspace.inspector.flush()
            self.flow_workspace.inspector.clear_node()
            self.flow_workspace.flow_id = None
        self.db.delete_flow(flow_id)
        self.refresh_all()
        self.show_status(f"已删除流程：{name}")

    def _open_flow_fallback(self, flow_id: int) -> None:
        flow = self.db.get_flow(flow_id)
        if flow is None:
            return
        dialog = FallbackDialog(self.db, flow_id, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.show_status(f"已保存流程「{flow.name}」的保底机制")

    def _open_flow_settings(self, flow_id: int) -> None:
        flow = self.db.get_flow(flow_id)
        if flow is None:
            return
        dialog = FlowSettingsDialog(self.db, flow_id, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.show_status(f"已保存流程「{flow.name}」的运行时间设置")

    def _open_project_fallback(self, project_id: int) -> None:
        blocks = self.db.list_project_blocks(project_id)
        if not blocks:
            QMessageBox.information(self, "提示", "该项目还没有区块，无法设置保底机制")
            return
        flows = []
        seen = set()
        for block in blocks:
            if block.flow_id not in seen:
                seen.add(block.flow_id)
                flows.append(block)
        if len(flows) == 1:
            self._open_flow_fallback(flows[0].flow_id)
            return
        names = [block.flow_name for block in flows]
        choice, ok = QInputDialog.getItem(
            self, "选择区块", "为哪个区块的流程设置保底机制：", names, 0, False
        )
        if not ok:
            return
        index = names.index(choice)
        self._open_flow_fallback(flows[index].flow_id)

    def _restore_default_view(self) -> None:
        self.settings.setValue("ui_mode", "normal")
        self.settings.remove("dock_state")
        for dock in (
            self.workspace_dock,
            self.inspector_dock,
            self.asset_dock,
            self.console_dock,
        ):
            dock.setFloating(False)
            dock.show()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.workspace_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.asset_dock)
        self.tabifyDockWidget(self.inspector_dock, self.asset_dock)
        self.inspector_dock.raise_()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        self._fit_to_screen()
        self.resizeDocks([self.workspace_dock], [240], Qt.Orientation.Horizontal)
        self.resizeDocks([self.inspector_dock], [400], Qt.Orientation.Horizontal)
        self.resizeDocks([self.console_dock], [150], Qt.Orientation.Vertical)
        self._apply_mode("normal")
        self.show_status("已恢复默认界面设置")

    def refresh_all(self) -> None:
        self.refresh_tree()
        self.project_workspace.refresh_projects(keep=True)
        self.flow_workspace.refresh_flows(keep=True)
        self.run_workspace.refresh_targets()
        self.run_workspace.refresh_devices()
        self.history_workspace.refresh()
        self.pro_workspace.refresh_all()

    def refresh_tree(self) -> None:
        expanded = set()
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.isExpanded():
                expanded.add(item.text(0))
        self.tree.clear()
        root_flags = (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDropEnabled
        )
        child_flags = (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )
        projects_root = QTreeWidgetItem(["项目"])
        projects_root.setData(0, Qt.ItemDataRole.UserRole, ("root", "projects"))
        projects_root.setFlags(root_flags)
        flows_root = QTreeWidgetItem(["流程"])
        flows_root.setData(0, Qt.ItemDataRole.UserRole, ("root", "flows"))
        flows_root.setFlags(root_flags)
        self.tree.addTopLevelItem(projects_root)
        self.tree.addTopLevelItem(flows_root)
        for project in self.db.list_projects():
            item = QTreeWidgetItem([project.name])
            item.setData(0, Qt.ItemDataRole.UserRole, ("project", project.id))
            item.setFlags(child_flags)
            projects_root.addChild(item)
        for flow in self.db.list_flows():
            item = QTreeWidgetItem([flow.name])
            item.setData(0, Qt.ItemDataRole.UserRole, ("flow", flow.id))
            item.setFlags(child_flags)
            flows_root.addChild(item)
        projects_root.setExpanded(True)
        flows_root.setExpanded(self._pro)
        if not self._pro:
            flows_root.setHidden(True)

    def show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 6000)
        self.console.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def _flush_editors(self) -> None:
        self.project_workspace.save(silent=True)
        self.flow_workspace.inspector.flush()

    def _save_all(self) -> None:
        self.project_workspace.save(silent=False)
        self.flow_workspace.inspector.save_node(silent=False)
        self.show_status("已保存")

    def _clear_cache(self) -> None:
        root = config.screenshots_dir()
        files = [path for path in root.rglob("*") if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        if not files:
            QMessageBox.information(self, "清理缓存", "没有可清理的缓存。")
            return
        answer = QMessageBox.question(
            self,
            "清理缓存",
            f"将删除运行截图缓存：{len(files)} 个文件，约 {total / 1048576:.1f} MB。\n"
            "节点图片、素材和流程数据不会被删除。继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed = 0
        freed = 0
        for path in files:
            try:
                size = path.stat().st_size
                path.unlink()
                removed += 1
                freed += size
            except OSError:
                pass
        for directory in sorted(
            [path for path in root.iterdir() if path.is_dir()], reverse=True
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        self.show_status(f"已清理缓存：{removed} 个文件，约 {freed / 1048576:.1f} MB")

    def _update_engineering_title(self) -> None:
        name = Path(config.home()).name or "工程"
        self.setWindowTitle(
            f"scrcpy-flow 1.0 · 工程：{name} · 工作区 v{__version__}"
        )

    def _register_current_engineering(self) -> None:
        path = Path(config.home())
        self.engineerings.add(path)
        self.engineerings.set_current(path)
        self._update_engineering_title()

    def _create_engineering(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "新建工程（选择或新建一个空文件夹作为工程目录）", str(Path.home())
        )
        if not directory:
            return
        path = Path(directory)
        if (path / "scflow.db").exists():
            answer = QMessageBox.question(
                self,
                "打开已有工程？",
                "该文件夹已包含工程数据（scflow.db），是否直接打开它？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._switch_engineering(path)

    def _open_engineering_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "打开工程文件夹（选择包含 scflow.db 的文件夹）", str(Path.home())
        )
        if not directory:
            return
        path = Path(directory)
        if not (path / "scflow.db").exists():
            bundles = sorted(path.glob("*.scflow"))
            if bundles:
                names = [item.name for item in bundles]
                choice, ok = QInputDialog.getItem(
                    self,
                    "发现工程文件",
                    "该文件夹不是工程目录，但发现工程文件，是否打开它？",
                    names,
                    0,
                    False,
                )
                if ok:
                    self._open_bundle(path / choice)
                return
            QMessageBox.information(
                self,
                "提示",
                "该文件夹不是工程目录（缺少 scflow.db）。\n"
                "如果要打开别人发来的 .scflow 工程文件，请用"
                "「打开工程文件…（新建独立工程）」。",
            )
            return
        self._switch_engineering(path)

    def _populate_switch_menu(self) -> None:
        menu = self.switch_engineering_menu
        menu.clear()
        current = Path(config.home()).resolve()
        paths = self.engineerings.recent_paths()
        if not paths:
            action = menu.addAction("（暂无其它工程）")
            action.setEnabled(False)
            return
        for path in paths:
            action = menu.addAction(path.name + f"  —  {path}")
            action.setCheckable(True)
            action.setChecked(path.resolve() == current)
            action.triggered.connect(
                lambda _checked=False, target=path: self._switch_engineering(target)
            )

    def _switch_engineering(self, path: Path, announce: bool = True) -> None:
        target = Path(path).expanduser().resolve()
        if target == Path(config.home()).resolve():
            self._register_current_engineering()
            self._update_engineering_title()
            return
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "切换工程失败", str(exc))
            return
        try:
            self.run_workspace.stop()
        except Exception:
            pass
        try:
            self._flush_editors()
        except Exception:
            pass

        # Drop per-engineering state before pointing the shared DB at the new file.
        self.flow_workspace.inspector.clear_node()
        self.flow_workspace.inspector._dirty = False
        self.flow_workspace._undo_stack.clear()
        self.flow_workspace._clipboard = None
        self.flow_workspace.flow_id = None
        self.flow_workspace.nodes = []
        self.flow_workspace.edges = []
        self.project_workspace.project_id = None
        self.project_workspace._dirty = False
        self.project_workspace._undo_stack.clear()
        self.project_workspace.blocks = []

        os.environ["SCFLOW_HOME"] = str(target)
        self.db.path = config.db_file()
        self.db.init()
        self.refresh_all()
        self.asset_library.refresh()
        self.engineerings.add(target)
        self.engineerings.set_current(target)
        self._update_engineering_title()
        if announce:
            self.show_status(f"已切换到工程：{target.name}（{target}）")

    def _new_engineering(self) -> None:
        dialog = EngineeringExportDialog(self.db, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        flow_ids, project_ids, pack = dialog.selection()
        if not flow_ids and not project_ids:
            QMessageBox.information(self, "提示", "请至少选择一个流程或项目")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存工程文件", "", "scflow 工程文件 (*.scflow)"
        )
        if not path:
            return
        if not path.lower().endswith(".scflow"):
            path += ".scflow"
        try:
            export_engineering(self.db, path, flow_ids, project_ids, pack)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.show_status(f"工程已保存：{path}")

    def _open_engineering(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开工程文件", "", "scflow 工程文件 (*.scflow);;所有文件 (*)"
        )
        if not path:
            return
        self._open_bundle(Path(path))

    def _open_bundle(self, source: Path) -> None:
        if source.suffix.lower() != ".scflow":
            QMessageBox.information(self, "提示", "请选择 .scflow 工程文件")
            return
        target = source.parent / source.stem
        index = 2
        while target.exists() and any(target.iterdir()):
            target = source.parent / f"{source.stem} ({index})"
            index += 1
        self._switch_engineering(target, announce=False)
        try:
            summary = import_engineering(self.db, source)
        except Exception as exc:
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        self.refresh_all()
        self.show_status(
            f"已创建工程「{target.name}」：{summary['flows']} 个流程、"
            f"{summary['projects']} 个项目"
        )

    def _on_tab_changed(self, index: int) -> None:
        if index != 0:
            self.project_workspace.save(silent=True)
        else:
            self.project_workspace.refresh_flow_choices()
        if index != 1:
            self.flow_workspace.inspector.flush()
        if index == 3:
            self.history_workspace.refresh()
        if index == 4:
            self.similarity_workspace.reload_current()
        if index == 5:
            self.pro_workspace.refresh_all()

    def _start_run(self) -> None:
        self.tabs.setCurrentIndex(2)
        self.run_workspace.start()

    def _on_running_changed(self, running: bool) -> None:
        self.action_run.setEnabled(not running)
        self.action_stop.setEnabled(running)

    def _on_mode_combo(self, _index: int) -> None:
        mode = self.mode_combo.currentData()
        if mode and mode != self._mode:
            self._set_mode(str(mode))

    def _set_mode(self, mode: str) -> None:
        if mode not in ("concise", "normal", "pro"):
            mode = "normal"
        if mode == self._mode:
            self._sync_mode_widgets()
            return
        self._mode = mode
        self.settings.setValue("ui_mode", mode)
        self._apply_mode(mode)
        self.show_status(f"已切换为{self._mode_label()}")

    def _sync_mode_widgets(self) -> None:
        checked = {
            "concise": self.action_mode_concise,
            "normal": self.action_mode_normal,
            "pro": self.action_mode_pro,
        }[self._mode]
        checked.setChecked(True)
        index = self.mode_combo.findData(self._mode)
        if index >= 0 and index != self.mode_combo.currentIndex():
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(False)

    def _apply_mode(self, mode: str) -> None:
        self._mode = mode
        self._pro = mode in ("normal", "pro")
        self._sync_mode_widgets()
        colors = {
            "concise": ("#3a2a1f", "#f0883e"),
            "normal": ("#1f2a3a", "#58a6ff"),
            "pro": ("#1f3a2c", "#3fb950"),
        }
        background, color = colors.get(mode, colors["normal"])
        self.mode_badge.setText(self._mode_label())
        self.mode_badge.setStyleSheet(
            f"background:{background};color:{color};border-radius:4px;padding:2px 8px;"
        )
        self.inspector_dock.setVisible(self._pro)
        self.asset_dock.setVisible(self._pro)
        self.console_dock.setVisible(self._pro)
        self.ai_dock.setVisible(self._pro)
        self.tabs.setTabVisible(1, self._pro)
        self.tabs.setTabVisible(3, self._pro)
        self.tabs.setTabVisible(4, self._pro)
        self.tabs.setTabVisible(5, mode == "pro")
        if mode == "concise" and self.tabs.currentIndex() in (1, 3, 4, 5):
            self.tabs.setCurrentIndex(0)
        elif mode == "normal" and self.tabs.currentIndex() == 5:
            self.tabs.setCurrentIndex(0)
        self.project_workspace.set_simple(mode == "concise")
        self.flow_workspace.set_simple(mode == "concise")
        self.run_workspace.set_simple(mode == "concise")
        if mode != "pro":
            self.pro_workspace.stop_all()
        else:
            self.pro_workspace.refresh_all()
        self.refresh_tree()
        self.statusBar().showMessage(
            f"数据：{config.home()}    ·    {self._mode_label()}"
        )

    def _fit_to_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 800)
            return
        available = screen.availableGeometry()
        width = min(1600, int(available.width() * 0.96))
        height = min(1000, int(available.height() * 0.92))
        self.resize(width, height)
        self.move(
            available.x() + max(0, (available.width() - width) // 2),
            available.y() + max(0, (available.height() - height) // 2),
        )

    def open_floating_run(self) -> None:
        if self.floating_run_window is None:
            window = QMainWindow(self)
            window.setWindowTitle("scrcpy-flow 1.0 · 独立运行面板")
            window.resize(980, 760)
            run = RunWorkspace(self.db)
            run.set_pre_run_callback(self._flush_editors)
            def _sync_floating_serial() -> None:
                serial = run.device_combo.currentData()
                self.flow_workspace.inspector.serial = serial
                self.asset_library.set_serial(serial)

            run.device_combo.currentIndexChanged.connect(
                lambda _index: _sync_floating_serial()
            )
            window.setCentralWidget(run)
            window.destroyed.connect(lambda: setattr(self, "floating_run_window", None))
            self.floating_run_window = window
        self.floating_run_window.show()
        self.floating_run_window.raise_()
        self.floating_run_window.activateWindow()

    def _float_dock(self, dock: QDockWidget) -> None:
        dock.setFloating(True)
        dock.resize(520, 760)
        dock.show()
        dock.raise_()

    def show_palette(self) -> None:
        actions = [
            ("开始运行", "F5", self._start_run),
            ("停止运行", "Shift+F5", self.run_workspace.stop),
            ("保存全部", "Ctrl+S", self._save_all),
            ("新建项目", "", self.project_workspace._new_project),
            ("新建流程", "", self.flow_workspace._new_flow),
            ("运行示例项目", "", lambda: self.run_project("4TB SSD 自动比价")),
            ("切换为简洁版", "", lambda: self._set_mode("concise")),
            ("切换为普通版", "", lambda: self._set_mode("normal")),
            ("切换为专业版", "", lambda: self._set_mode("pro")),
            ("浮动属性窗口", "", lambda: self._float_dock(self.inspector_dock)),
            ("浮动工作区窗口", "", lambda: self._float_dock(self.workspace_dock)),
            ("浮动控制台窗口", "", lambda: self._float_dock(self.console_dock)),
            ("截取手机当前图片（存入素材库）", "", self.asset_library.capture_current),
            ("素材库：导入图片", "", self.asset_library.import_files),
            ("素材库：添加选中到流程", "", self.asset_library.add_selected_to_flow),
            ("导出手机文件", "", self._export_phone),
            ("相似度编辑", "", lambda: self._open_similarity_editor(self.flow_workspace.inspector.node.id) if self.flow_workspace.inspector.node else None),
            ("个性化设置", "", self._open_personalization),
            ("清除自定义背景", "", self._clear_personalization),
            ("清理缓存", "", self._clear_cache),
            ("使用说明", "", self._show_usage_guide),
            ("导出错误日志", "", self._export_error_log),
            ("恢复默认设置", "", self._restore_default_view),
            ("刷新全部", "", self.refresh_all),
        ]
        dialog = CommandPalette(actions, self)
        dialog.exec()

    def open_project_by_id(self, project_id: int) -> None:
        self.tabs.setCurrentIndex(0)
        self.project_workspace.refresh_projects(select_id=project_id)

    def open_flow_by_id(self, flow_id: int) -> None:
        self.flow_workspace.refresh_flows(select_id=flow_id)
        self.tabs.setCurrentIndex(1 if self._pro else 0)

    def open_project(self, name: str) -> None:
        project = self.db.get_project_by_name(name)
        if project:
            self.open_project_by_id(project.id)

    def open_flow(self, name: str) -> None:
        flow = self.db.get_flow_by_name(name)
        if flow:
            self.open_flow_by_id(flow.id)

    def run_project(self, name: str) -> None:
        project = self.db.get_project_by_name(name)
        if project is None:
            QMessageBox.warning(self, "提示", f"找不到项目：{name}")
            return
        self.run_workspace.select_target("project", project.id)
        self._start_run()

    def auto_run(self, target: str, dry_run: bool = False) -> None:
        project = self.db.get_project_by_name(target)
        if project is not None:
            self.run_workspace.select_target("project", project.id)
        else:
            flow = self.db.get_flow_by_name(target)
            if flow is None:
                return
            self.run_workspace.select_target("flow", flow.id)
        self.run_workspace.dry_check.setChecked(dry_run)
        QTimer.singleShot(1200, self._start_run)

    def _open_similarity_editor(self, node_id: int) -> None:
        self.flow_workspace.inspector.flush()
        self.similarity_workspace.load_node(node_id, force=True)
        self.tabs.setCurrentIndex(4)

    def _on_node_saved_similarity(self, node_id: int) -> None:
        if self.similarity_workspace.node_id == node_id:
            self.similarity_workspace.reload_current()

    def _back_to_flow(self) -> None:
        self.tabs.setCurrentIndex(1)
        node_id = self.similarity_workspace.node_id
        if node_id is not None:
            self.flow_workspace.select_node(node_id)

    def _export_phone(self) -> None:
        self._flush_editors()
        dialog = PhoneExportDialog(self.db, self.flow_workspace.flow_id, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.exported_path:
            self.show_status(f"手机文件已导出：{dialog.exported_path}")

    def _open_personalization(self) -> None:
        dialog = PersonalizationDialog(self.personalization, self)
        dialog.exec()

    def _clear_personalization(self) -> None:
        self.personalization.reset_all()
        self.show_status("已清除自定义背景")

    def _show_usage_guide(self) -> None:
        dialog = UsageGuideDialog(self)
        dialog.exec()

    def _export_error_log(self) -> None:
        default = f"scflow-error-{datetime.now():%Y%m%d-%H%M%S}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出错误日志", default, "文本文件 (*.txt *.log);;所有文件 (*)"
        )
        if not path:
            return
        try:
            report = build_error_report(
                self.db, self.console.toPlainText(), config.home(), __version__
            )
            Path(path).write_text(report, encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.show_status(f"错误日志已导出：{path}")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "关于 scrcpy-flow 1.0",
            "scrcpy-flow 1.0\n"
            "普通版：现代化可停靠工作区 · 区块编排 · Python 编排 · scrcpy/ADB 控制\n"
            "专业版：批量设备 · 控件树 / OCR 智能定位 · 录制回放\n\n"
            "所有面板都可以拖动、浮动为独立窗口。",
        )

    def closeEvent(self, event) -> None:
        try:
            self.settings.setValue("dock_state", self.saveState())
        except Exception:
            pass
        try:
            self.pro_workspace.stop_all()
            self.ai_panel.shutdown()
            self.cloud_panel.shutdown()
            self.run_workspace.shutdown()
            self.project_workspace.save(silent=True)
            self.flow_workspace.inspector.flush()
        except Exception:
            pass
        super().closeEvent(event)


def launch(db: Database, args: argparse.Namespace | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("scrcpy-flow")
    app.setApplicationVersion(__version__)
    apply_theme(app)
    window = MainWindow(db)
    window.show()
    QTimer.singleShot(400, window.ai_panel.refresh_status)
    QTimer.singleShot(
        60,
        lambda: window.resizeDocks(
            [window.inspector_dock], [400], Qt.Orientation.Horizontal
        ),
    )
    if args is not None:
        if getattr(args, "open_project", None):
            QTimer.singleShot(300, lambda: window.open_project(args.open_project))
        if getattr(args, "open_flow", None):
            QTimer.singleShot(300, lambda: window.open_flow(args.open_flow))
        if getattr(args, "auto_run", None):
            QTimer.singleShot(600, lambda: window.auto_run(args.auto_run, getattr(args, "dry_run", False)))
    return app.exec()
