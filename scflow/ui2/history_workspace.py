from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..db import Database


class ScreenshotWindow(QDialog):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(path).name)
        self.resize(520, 900)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            label.setPixmap(pixmap)
        scroll.setWidget(label)
        layout.addWidget(scroll)


class HistoryWorkspace(QWidget):
    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.screenshot_map: dict[str, str] = {}
        self._build()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        flow_page = QWidget()
        flow_layout = QVBoxLayout(flow_page)
        bar = QSplitter(Qt.Orientation.Horizontal)
        flow_layout.addWidget(bar)
        self.tabs.addTab(flow_page, "流程会话")
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        top = QSplitter(Qt.Orientation.Horizontal)
        self.flow_combo = QComboBox()
        self.refresh_btn = QPushButton("刷新")
        top.addWidget(self.flow_combo)
        top.addWidget(self.refresh_btn)
        top.setStretchFactor(0, 1)
        left_layout.addWidget(top)
        self.sessions = QTableWidget(0, 7)
        self.sessions.setHorizontalHeaderLabels(
            ["会话", "流程", "开始时间", "状态", "步数", "保底", "备注"]
        )
        self.sessions.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sessions.verticalHeader().setVisible(False)
        left_layout.addWidget(self.sessions, 1)
        bar.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("步骤日志（双击查看截图）"))
        self.logs = QTableWidget(0, 7)
        self.logs.setHorizontalHeaderLabels(
            ["步", "时间", "操作前状态", "匹配节点", "相似度", "动作", "结果"]
        )
        self.logs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.logs.verticalHeader().setVisible(False)
        right_layout.addWidget(self.logs, 1)
        bar.addWidget(right)
        bar.setSizes([500, 700])

        project_page = QWidget()
        project_layout = QVBoxLayout(project_page)
        self.tabs.addTab(project_page, "项目运行")
        top2 = QSplitter(Qt.Orientation.Horizontal)
        self.project_refresh = QPushButton("刷新项目运行")
        top2.addWidget(self.project_refresh)
        top2.setStretchFactor(0, 0)
        project_layout.addWidget(top2)
        self.project_runs = QTableWidget(0, 7)
        self.project_runs.setHorizontalHeaderLabels(
            ["会话", "项目", "开始时间", "状态", "区块", "总步数", "错误"]
        )
        self.project_runs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.project_runs.verticalHeader().setVisible(False)
        project_layout.addWidget(self.project_runs, 1)
        self.project_vars = QLabel("")
        self.project_vars.setWordWrap(True)
        self.project_vars.setObjectName("subtitle")
        project_layout.addWidget(self.project_vars)

        self.flow_combo.currentIndexChanged.connect(self._refresh_sessions)
        self.refresh_btn.clicked.connect(self.refresh)
        self.project_refresh.clicked.connect(self.refresh_project_runs)
        self.sessions.itemSelectionChanged.connect(self._load_logs)
        self.logs.doubleClicked.connect(self._open_screenshot)
        self.project_runs.itemSelectionChanged.connect(self._show_project_vars)
        self.tabs.currentChanged.connect(lambda _index: self.refresh_project_runs())

    def refresh(self) -> None:
        flows = self.db.list_flows()
        current = self.flow_combo.currentData()
        self.flow_combo.blockSignals(True)
        self.flow_combo.clear()
        self.flow_combo.addItem("全部流程", None)
        for flow in flows:
            self.flow_combo.addItem(flow.name, flow.id)
        if current is not None:
            index = self.flow_combo.findData(current)
            if index >= 0:
                self.flow_combo.setCurrentIndex(index)
        self.flow_combo.blockSignals(False)
        self._refresh_sessions()
        self.refresh_project_runs()

    def _refresh_sessions(self) -> None:
        flow_id = self.flow_combo.currentData()
        self.sessions.setRowCount(0)
        flow_names = {flow.id: flow.name for flow in self.db.list_flows()}
        for session in self.db.list_sessions(flow_id, limit=200):
            row = self.sessions.rowCount()
            self.sessions.insertRow(row)
            values = [
                session.id,
                flow_names.get(session.flow_id, ""),
                session.started_at,
                session.status,
                session.steps,
                session.fallbacks,
                session.note,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self.sessions.setItem(row, col, item)
        self.logs.setRowCount(0)

    def _load_logs(self) -> None:
        rows = self.sessions.selectionModel().selectedRows()
        if not rows:
            return
        session_item = self.sessions.item(rows[0].row(), 0)
        if session_item is None:
            return
        session_id = int(session_item.text())
        self.logs.setRowCount(0)
        self.screenshot_map = {}
        for log in self.db.list_logs(session_id):
            row = self.logs.rowCount()
            self.logs.insertRow(row)
            values = [
                log["idx"],
                log["ts"],
                log["current_name"] or "",
                log["matched_name"] or "",
                "-" if log["similarity"] is None else f"{log['similarity']:.3f}",
                log["action"],
                log["result"],
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self.logs.setItem(row, col, item)
            if log["screenshot_path"]:
                self.screenshot_map[str(row)] = log["screenshot_path"]

    def _open_screenshot(self, index) -> None:
        path = self.screenshot_map.get(str(index.row()))
        if path and Path(path).exists():
            dialog = ScreenshotWindow(path, self)
            dialog.exec()

    def refresh_project_runs(self) -> None:
        self.project_runs.setRowCount(0)
        for run in self.db.list_project_runs(limit=200):
            row = self.project_runs.rowCount()
            self.project_runs.insertRow(row)
            values = [
                run["id"],
                run["project_name"],
                run["started_at"],
                run["status"],
                run["blocks"],
                run["steps"],
                run["error"],
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self.project_runs.setItem(row, col, item)

    def _show_project_vars(self) -> None:
        rows = self.project_runs.selectionModel().selectedRows()
        if not rows:
            return
        item = self.project_runs.item(rows[0].row(), 0)
        if item is None:
            return
        run_id = int(item.text())
        for run in self.db.list_project_runs(limit=500):
            if int(run["id"]) == run_id:
                try:
                    variables = json.loads(run["variables_json"] or "{}")
                except (TypeError, ValueError):
                    variables = {}
                self.project_vars.setText(json.dumps(variables, ensure_ascii=False, indent=2))
                return
