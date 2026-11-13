from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..db import Database
from ..embeddings import embed_to_blob, get_embedder
from ..undo import restore_flow, snapshot_flow
from .dialogs import ConnectDialog
from .ai_dialogs import AIGoalDialog
from .flow_graph import FlowGraphView
from .flow_settings_dialog import FlowSettingsDialog
from .layout import compute_layout
from .node_inspector import NodeInspector
from .scrollbar_bar import make_scrollable_bar


class FlowWorkspace(QWidget):
    status_message = Signal(str)
    flow_changed = Signal()
    similarity_requested = Signal(int)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_id: int | None = None
        self.nodes = []
        self.edges = []
        self.selected_edge_id: int | None = None
        self._undo_stack: list[tuple[str, dict]] = []
        self._redo_stack: list[tuple[str, dict]] = []
        self._clipboard: dict | None = None
        self._paste_count = 0
        self._loading = False
        self.inspector = NodeInspector(db)
        self.inspector.undo_callback = self.push_undo
        self._build()
        self.refresh_flows()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("流程"))
        self.flow_combo = QComboBox()
        self.flow_combo.setMinimumWidth(240)
        bar.addWidget(self.flow_combo)
        self.new_flow_btn = QPushButton("新建流程")
        self.settings_btn = QPushButton("流程设置")
        self.add_node_btn = QPushButton("添加节点")
        self.add_end_btn = QPushButton("添加结束节点")
        self.add_nodetect_btn = QPushButton("添加不检测步骤")
        self.add_if_btn = QPushButton("添加 IF 节点")
        self.add_ai_btn = QPushButton("添加 AI 节点")
        self.add_ai_btn.setToolTip("由本地 Ollama 看屏决策的节点，不需要预先截图")
        self.similarity_btn = QPushButton("相似度编辑")
        self.similarity_btn.setCheckable(True)
        self.similarity_btn.setToolTip("开启后点击节点进入相似度编辑")
        self.copy_btn = QPushButton("复制步骤")
        self.paste_btn = QPushButton("粘贴步骤")
        self.connect_btn = QPushButton("连接 / 下一步")
        self.cancel_btn = QPushButton("取消连接/下一步")
        self.delete_btn = QPushButton("删除选中")
        self.layout_btn = QPushButton("自动布局")
        self.undo_btn = QPushButton("撤回")
        self.redo_btn = QPushButton("重做")
        self.fit_btn = QPushButton("适应窗口")
        self.refresh_btn = QPushButton("刷新")
        for button in (
            self.new_flow_btn,
            self.settings_btn,
            self.add_node_btn,
            self.add_end_btn,
            self.add_nodetect_btn,
            self.add_if_btn,
            self.add_ai_btn,
            self.similarity_btn,
            self.copy_btn,
            self.paste_btn,
            self.connect_btn,
            self.cancel_btn,
            self.delete_btn,
            self.layout_btn,
            self.undo_btn,
            self.redo_btn,
            self.fit_btn,
            self.refresh_btn,
        ):
            bar.addWidget(button)
        bar.addStretch(1)
        layout.addWidget(make_scrollable_bar(bar))
        self.graph = FlowGraphView()
        layout.addWidget(self.graph, 1)
        self.hint = QLabel("拖动节点排版 · 单击选中 · 双击连线编辑 · Ctrl+滚轮缩放")
        self.hint.setObjectName("subtitle")
        layout.addWidget(self.hint)

        self.flow_combo.currentIndexChanged.connect(self._on_flow_combo)
        self.new_flow_btn.clicked.connect(self._new_flow)
        self.settings_btn.clicked.connect(self._open_flow_settings)
        self.add_node_btn.clicked.connect(lambda: self._add_node("image"))
        self.add_end_btn.clicked.connect(lambda: self._add_node("end"))
        self.add_nodetect_btn.clicked.connect(self._add_nodetect_node)
        self.add_if_btn.clicked.connect(self._add_if_node)
        self.add_ai_btn.clicked.connect(self._add_ai_node)
        self.similarity_btn.toggled.connect(self._on_similarity_mode)
        self.copy_btn.clicked.connect(self._copy_selected)
        self.paste_btn.clicked.connect(self._paste_nodes)
        self.copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self)
        self.copy_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.copy_shortcut.activated.connect(self._copy_selected)
        self.paste_shortcut = QShortcut(QKeySequence.StandardKey.Paste, self)
        self.paste_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.paste_shortcut.activated.connect(self._paste_nodes)
        self.connect_btn.clicked.connect(self._connect)
        self.cancel_btn.clicked.connect(self._cancel_connection)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.layout_btn.clicked.connect(self._auto_layout)
        self.undo_btn.clicked.connect(self._undo)
        self.redo_btn.clicked.connect(self._redo)
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.undo_shortcut.activated.connect(self._undo)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.redo_shortcut.activated.connect(self._redo)
        self.fit_btn.clicked.connect(self.graph.fit_view)
        self.refresh_btn.clicked.connect(lambda: self.refresh_flows(keep=True))
        self.graph.node_selected.connect(self._on_node_selected)
        self.graph.edge_selected.connect(self._on_edge_selected)
        self.graph.node_moved.connect(self._on_node_moved)
        self.graph.node_drag_started.connect(lambda _node_id: self.push_undo("移动节点"))
        self.graph.edge_double_clicked.connect(self._edit_edge)
        self.graph.node_context_menu.connect(self._on_node_context_menu)
        self.graph.edge_context_menu.connect(self._on_edge_context_menu)
        self.graph.connect_completed.connect(self._on_connect_completed)
        self.graph.connect_cancelled.connect(
            lambda: self.status_message.emit("已取消连线")
        )
        self.inspector.node_saved.connect(self._on_node_saved)
        self.inspector.edges_changed.connect(self._reload_graph)
        self.inspector.delete_requested.connect(self._delete_node)
        self.inspector.status_message.connect(self.status_message)

    def set_simple(self, simple: bool) -> None:
        for button in (
            self.new_flow_btn,
            self.settings_btn,
            self.add_node_btn,
            self.add_end_btn,
            self.add_nodetect_btn,
            self.add_if_btn,
            self.add_ai_btn,
            self.similarity_btn,
            self.copy_btn,
            self.paste_btn,
            self.connect_btn,
            self.cancel_btn,
            self.delete_btn,
            self.layout_btn,
            self.undo_btn,
            self.redo_btn,
        ):
            button.setVisible(not simple)
        self.hint.setText(
            "简洁版：查看流程；切换到普通版或专业版编辑节点"
            if simple
            else "拖动节点排版 · 单击选中 · 双击连线编辑 · 滚轮缩放"
        )

    def refresh_flows(self, keep: bool = True, select_id: int | None = None) -> None:
        self.inspector.flush()
        flows = self.db.list_flows()
        current = select_id or (self.flow_id if keep else None)
        self._loading = True
        self.flow_combo.clear()
        for flow in flows:
            self.flow_combo.addItem(flow.name, flow.id)
        index = 0
        if current is not None:
            found = self.flow_combo.findData(current)
            if found >= 0:
                index = found
        if flows:
            self.flow_combo.setCurrentIndex(index)
            self._loading = False
            self.load_flow(flows[index].id)
        else:
            self.flow_id = None
            self.nodes = []
            self.edges = []
            self.graph.load([], [])
            self.inspector.clear_node()
            self._loading = False

    def _on_flow_combo(self, _index: int) -> None:
        if self._loading:
            return
        self.inspector.flush()
        flow_id = self.flow_combo.currentData()
        if flow_id is not None:
            self._undo_stack.clear()
            self._redo_stack.clear()
            self.load_flow(int(flow_id))

    def load_flow(self, flow_id: int) -> None:
        flow = self.db.get_flow(flow_id)
        if flow is None:
            return
        self.flow_id = flow.id
        self.nodes = self.db.list_nodes(flow.id)
        self.edges = self.db.list_edges(flow.id)
        if len(self.nodes) > 1 and len({(n.x, n.y) for n in self.nodes}) == 1:
            for node_id, (x, y) in compute_layout(self.nodes, self.edges).items():
                self.db.update_node(node_id, x=x, y=y)
            self.nodes = self.db.list_nodes(flow.id)
        self.graph.load(self.nodes, self.edges)
        self.inspector.clear_node()
        entry = next((node for node in self.nodes if node.is_entry), None)
        if entry is None and self.nodes:
            entry = self.nodes[0]
        if entry:
            self.inspector.load_node(entry.id, force=True)
            self.status_message.emit(f"流程：{flow.name} · {len(self.nodes)} 节点")
        else:
            self.status_message.emit(f"流程：{flow.name} · 空流程")

    def _reload_graph(self) -> None:
        if self.flow_id is None:
            return
        self.nodes = self.db.list_nodes(self.flow_id)
        self.edges = self.db.list_edges(self.flow_id)
        self.graph.load(self.nodes, self.edges)

    def _on_similarity_mode(self, enabled: bool) -> None:
        if enabled:
            self.status_message.emit("相似度编辑模式：点击一个节点进入相似度编辑")

    def _on_node_selected(self, node_id: int) -> None:
        if node_id < 0:
            self.inspector.clear_node()
            return
        self.inspector.load_node(node_id)
        if self.similarity_btn.isChecked():
            self.similarity_btn.setChecked(False)
            self.similarity_requested.emit(node_id)

    def _on_edge_selected(self, edge_id: int) -> None:
        self.selected_edge_id = edge_id

    def _on_node_moved(self, node_id: int, x: float, y: float) -> None:
        self.db.update_node(node_id, x=x, y=y)
        self.status_message.emit("节点位置已保存")

    def _on_node_saved(self, node_id: int) -> None:
        self._reload_graph()
        self.flow_changed.emit()

    def _on_node_context_menu(self, node_id: int, global_pos) -> None:
        node = self.db.get_node(node_id)
        if node is None:
            return
        self.select_node(node_id)
        menu = QMenu(self)
        rename_action = menu.addAction("重命名")
        ai_goal_action = (
            menu.addAction("编辑 AI 目标…") if node.kind == "ai" else None
        )
        similarity_action = menu.addAction("相似度编辑…")
        connect_action = menu.addAction("连接下一步")
        cancel_action = menu.addAction("取消连接")
        menu.addSeparator()
        copy_action = menu.addAction("复制")
        paste_action = menu.addAction("粘贴")
        delete_action = menu.addAction("删除选中")
        chosen = menu.exec(global_pos)
        if chosen is rename_action:
            self._rename_node(node_id, node.name)
        elif ai_goal_action is not None and chosen is ai_goal_action:
            self.edit_ai_goal(node_id)
        elif chosen is similarity_action:
            self.similarity_requested.emit(node_id)
        elif chosen is connect_action:
            self.graph.start_connect(node_id)
            self.status_message.emit(
                "连线模式：移动到目标步骤会显示预览线，点击目标步骤确认，Esc/右键取消"
            )
        elif chosen is cancel_action:
            self._cancel_outgoing(node_id, node.name)
        elif chosen is copy_action:
            self._copy_selected()
        elif chosen is paste_action:
            self._paste_nodes()
        elif chosen is delete_action:
            self._delete_node(node_id)

    def _on_connect_completed(self, src_id: int, dst_id: int) -> None:
        if self.flow_id is None:
            return
        self.push_undo("添加连线")
        self.db.add_edge(self.flow_id, src_id, dst_id, 1.0, "")
        self.selected_edge_id = None
        self._reload_graph()
        self.inspector._reload_edges()
        self.flow_changed.emit()
        self.status_message.emit("已添加连线")

    def _on_edge_context_menu(self, edge_id: int, global_pos) -> None:
        if self.db.get_edge(edge_id) is None:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("删除")
        reverse_action = menu.addAction("反向")
        chosen = menu.exec(global_pos)
        if chosen is delete_action:
            self.delete_edge_by_id(edge_id)
        elif chosen is reverse_action:
            self.reverse_edge_by_id(edge_id)

    def delete_edge_by_id(self, edge_id: int) -> None:
        if self.db.get_edge(edge_id) is None:
            return
        self.push_undo("删除连线")
        self.db.delete_edge(edge_id)
        self.selected_edge_id = None
        self._reload_graph()
        self.inspector._reload_edges()
        self.flow_changed.emit()
        self.status_message.emit("已删除连线")

    def reverse_edge_by_id(self, edge_id: int) -> None:
        edge = self.db.get_edge(edge_id)
        if edge is None:
            return
        self.push_undo("反向连线")
        self.db.delete_edge(edge_id)
        self.db.add_edge(
            edge.flow_id, edge.dst_id, edge.src_id, edge.weight, edge.label
        )
        self.selected_edge_id = None
        self._reload_graph()
        self.inspector._reload_edges()
        self.flow_changed.emit()
        self.status_message.emit("已反向连线")

    def _rename_node(self, node_id: int, current: str) -> None:
        name, ok = QInputDialog.getText(self, "重命名节点", "节点名称：", text=current)
        if not ok:
            return
        name = name.strip()
        if not name or name == current:
            return
        self.push_undo("重命名节点")
        self.db.update_node(node_id, name=name)
        self._reload_graph()
        self.inspector.load_node(node_id, force=True)
        self.flow_changed.emit()
        self.status_message.emit(f"节点已重命名：{name}")

    def _cancel_outgoing(self, node_id: int, name: str) -> None:
        edges = [edge for edge in self.db.list_edges(self.flow_id) if edge.src_id == node_id]
        if not edges:
            QMessageBox.information(self, "提示", f"节点「{name}」没有下一步连接")
            return
        answer = QMessageBox.question(
            self,
            "取消连接",
            f"取消节点「{name}」的 {len(edges)} 条下一步连接？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.push_undo("取消连接")
        for edge in edges:
            self.db.delete_edge(edge.id)
        self.selected_edge_id = None
        self._reload_graph()
        self.inspector._reload_edges()
        self.flow_changed.emit()
        self.status_message.emit("已取消连接")

    def push_undo(self, label: str) -> None:
        if self.flow_id is None:
            return
        snapshot = snapshot_flow(self.db, self.flow_id)
        if snapshot:
            self._undo_stack.append((label, snapshot))
            self._undo_stack = self._undo_stack[-50:]
            self._redo_stack.clear()

    def _undo(self) -> None:
        if not self._undo_stack or self.flow_id is None:
            self.status_message.emit("没有可撤回的操作")
            return
        label, snapshot = self._undo_stack.pop()
        current = snapshot_flow(self.db, self.flow_id)
        if current:
            self._redo_stack.append((label, current))
            self._redo_stack = self._redo_stack[-50:]
        self.inspector.clear_node()
        restore_flow(self.db, self.flow_id, snapshot)
        self.load_flow(self.flow_id)
        self.status_message.emit(f"已撤回：{label}")
        self.flow_changed.emit()

    def _redo(self) -> None:
        if not self._redo_stack or self.flow_id is None:
            self.status_message.emit("没有可恢复的操作")
            return
        label, snapshot = self._redo_stack.pop()
        current = snapshot_flow(self.db, self.flow_id)
        if current:
            self._undo_stack.append((label, current))
            self._undo_stack = self._undo_stack[-50:]
        self.inspector.clear_node()
        restore_flow(self.db, self.flow_id, snapshot)
        self.load_flow(self.flow_id)
        self.status_message.emit(f"已恢复：{label}")
        self.flow_changed.emit()

    def _new_flow(self) -> None:
        name, ok = QInputDialog.getText(self, "新建流程", "流程名称：")
        if not ok or not name.strip():
            return
        name = self.db.unique_flow_name(name.strip())
        flow_id = self.db.create_flow(name)
        self.refresh_flows(select_id=flow_id)
        self.status_message.emit(f"已创建流程：{name}")
        self.flow_changed.emit()

    def _open_flow_settings(self) -> None:
        if self.flow_id is None:
            QMessageBox.information(self, "提示", "请先选择一个流程")
            return
        self.inspector.flush()
        dialog = FlowSettingsDialog(self.db, self.flow_id, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.status_message.emit("流程设置已保存")

    def _copy_image(self, path: str) -> tuple[str, int, int]:
        source = Path(path)
        target = config.images_dir() / (
            f"flow{self.flow_id}_{uuid4().hex[:10]}{source.suffix.lower() or '.png'}"
        )
        shutil.copy2(source, target)
        from PySide6.QtGui import QPixmap

        pixmap = QPixmap(str(target))
        return str(target), pixmap.width(), pixmap.height()

    def _add_node(self, kind: str) -> None:
        if self.flow_id is None:
            QMessageBox.information(self, "提示", "请先选择一个流程")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择状态图片", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        self.add_node_from_path(path, kind=kind)

    def add_node_from_path(
        self,
        path: str,
        kind: str = "image",
        name: str | None = None,
        is_end: bool = False,
    ) -> int | None:
        """Create a flow node from an existing image file (dialog or library)."""
        if self.flow_id is None:
            QMessageBox.information(self, "提示", "请先选择一个流程")
            return None
        end_flag = is_end or kind == "end"
        if end_flag:
            kind = "image"
        try:
            image_path, width, height = self._copy_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"复制图片失败：{exc}")
            return None
        self.push_undo("添加节点")
        nodes = self.db.list_nodes(self.flow_id)
        x = 70 + (len(nodes) % 3) * 260
        y = 70 + (len(nodes) // 3) * 150
        node_id = self.db.add_node(
            self.flow_id,
            name=(name or Path(path).stem[:24] or ("结束步骤" if end_flag else "新步骤")),
            kind=kind,
            image_path=image_path,
            img_w=width,
            img_h=height,
            click_x=None,
            click_y=None,
            action_type="none" if end_flag else "tap",
            action={"post_wait": 0.0 if end_flag else 0.6},
            threshold=0.85,
            is_entry=not any(node.is_entry for node in nodes),
            is_end=end_flag,
            x=x,
            y=y,
        )
        try:
            embedder = get_embedder()
            vector = embedder.embed(image_path)
            self.db.set_node_embedding(
                node_id, embed_to_blob(vector), embedder.kind, int(vector.shape[0])
            )
        except Exception as exc:
            self.status_message.emit(f"向量计算失败：{exc}")
        self.load_flow(self.flow_id)
        self.inspector.load_node(node_id, force=True)
        self.flow_changed.emit()
        return node_id

    def _add_nodetect_node(self) -> None:
        node_id = self.add_action_node("nodetect")
        if node_id is not None:
            self.status_message.emit(
                "已添加不检测步骤，请用「连接 / 下一步」把它接入流程（否则不会被执行）"
            )

    def _add_if_node(self) -> None:
        if self.flow_id is None:
            QMessageBox.information(self, "提示", "请先选择一个流程")
            return
        self.push_undo("添加节点")
        nodes = self.db.list_nodes(self.flow_id)
        x = 70 + (len(nodes) % 3) * 260
        y = 70 + (len(nodes) // 3) * 150
        node_id = self.db.add_node(
            self.flow_id,
            name="IF 条件",
            kind="if",
            action_type="none",
            action={"if_branches": [], "if_default": None},
            threshold=0.85,
            is_entry=not any(node.is_entry for node in nodes),
            x=x,
            y=y,
        )
        self.load_flow(self.flow_id)
        self.inspector.load_node(node_id, force=True)
        self.flow_changed.emit()
        self.status_message.emit(
            "已添加 IF 条件节点：在属性面板按顺序添加分支节点并选择默认节点"
        )

    def _add_ai_node(self) -> int | None:
        node_id = self.add_ai_node()
        if node_id is not None:
            self.status_message.emit(
                "已添加 AI 节点：运行时由本地模型看屏决策；请用「连接 / 下一步」接入流程"
            )
        return node_id

    def add_ai_node(
        self,
        goal: str = "",
        name: str | None = None,
        detail: str = "",
        max_decisions: int = 0,
        is_entry: bool | None = None,
    ) -> int | None:
        """创建不需要截图的 AI 决策节点；goal 为空时弹窗填写。"""
        if self.flow_id is None:
            QMessageBox.information(self, "提示", "请先选择一个流程")
            return None
        if not goal:
            dialog = AIGoalDialog(parent=self)
            if not dialog.exec():
                return None
            values = dialog.values()
            goal = values["goal"]
            detail = values["detail"]
            max_decisions = values["max_decisions"]
        self.push_undo("添加 AI 节点")
        nodes = self.db.list_nodes(self.flow_id)
        x = 70 + (len(nodes) % 3) * 260
        y = 70 + (len(nodes) // 3) * 150
        node_id = self.db.add_node(
            self.flow_id,
            name=name or (goal[:12] or "AI 决策"),
            kind="ai",
            action_type="ai",
            action={
                "goal": goal,
                "detail": detail,
                "max_decisions": max_decisions,
                "post_wait": 0.4,
            },
            threshold=0.85,
            is_entry=(
                not any(node.is_entry for node in nodes) if is_entry is None else is_entry
            ),
            x=x,
            y=y,
        )
        self.load_flow(self.flow_id)
        self.inspector.load_node(node_id, force=True)
        self.flow_changed.emit()
        return node_id

    def edit_ai_goal(self, node_id: int) -> None:
        node = self.db.get_node(node_id)
        if node is None or node.kind != "ai":
            return
        action = dict(node.action or {})
        dialog = AIGoalDialog(
            str(action.get("goal") or ""),
            str(action.get("detail") or ""),
            int(action.get("max_decisions") or 0),
            parent=self,
        )
        if not dialog.exec():
            return
        self.push_undo("编辑 AI 目标")
        action.update(dialog.values())
        action.setdefault("post_wait", 0.4)
        self.db.update_node(node_id, action=action)
        self._reload_graph()
        self.inspector.load_node(node_id, force=True)
        self.flow_changed.emit()
        self.status_message.emit(f"AI 节点「{node.name}」目标已更新")

    def create_flow_from_plan(self, plan: dict) -> int | None:
        """按 AI 生成的步骤新建一条流程，每个步骤都是一个 AI 决策节点。"""
        steps = [step for step in (plan.get("steps") or []) if isinstance(step, dict)]
        if not steps:
            return None
        name = self.db.unique_flow_name(str(plan.get("name") or "AI 流程"))
        flow_id = self.db.create_flow(
            name, description=str(plan.get("description") or "")
        )
        previous: int | None = None
        for index, step in enumerate(steps):
            goal = str(step.get("goal") or step.get("name") or "").strip()
            if not goal:
                continue
            node_id = self.db.add_node(
                flow_id,
                name=str(step.get("name") or f"步骤 {index + 1}")[:24],
                kind="ai",
                action_type="ai",
                action={
                    "goal": goal,
                    "detail": str(step.get("detail") or ""),
                    "max_decisions": 0,
                    "post_wait": 0.4,
                },
                threshold=0.85,
                is_entry=previous is None,
                x=90,
                y=70 + index * 130,
            )
            if previous is not None:
                self.db.add_edge(flow_id, previous, node_id, 1.0, "")
            previous = node_id
        self.refresh_flows(select_id=flow_id)
        self.flow_changed.emit()
        return flow_id

    def add_action_node(self, kind: str = "nodetect", name: str | None = None) -> int | None:
        """Create a node without an image that runs its action without detection."""
        if self.flow_id is None:
            QMessageBox.information(self, "提示", "请先选择一个流程")
            return None
        self.push_undo("添加节点")
        nodes = self.db.list_nodes(self.flow_id)
        x = 70 + (len(nodes) % 3) * 260
        y = 70 + (len(nodes) // 3) * 150
        node_id = self.db.add_node(
            self.flow_id,
            name=name or "不检测步骤",
            kind=kind,
            action_type="wait",
            action={"seconds": 1.0, "post_wait": 0.0},
            threshold=0.85,
            is_entry=not any(node.is_entry for node in nodes),
            x=x,
            y=y,
        )
        self.load_flow(self.flow_id)
        self.inspector.load_node(node_id, force=True)
        self.flow_changed.emit()
        return node_id

    def _selected_nodes(self) -> list:
        items = [item for item in self.graph.scene().selectedItems() if hasattr(item, "node")]
        nodes = [item.node for item in items]
        if not nodes and self.inspector.node is not None:
            node = self.db.get_node(self.inspector.node.id)
            if node is not None:
                nodes = [node]
        return [node for node in nodes if node is not None]

    def _copy_selected(self) -> None:
        nodes = self._selected_nodes()
        if not nodes:
            self.status_message.emit("请先选中要复制的步骤")
            return
        index = {node.id: i for i, node in enumerate(nodes)}
        entries = []
        for node in nodes:
            entries.append(
                {
                    "name": node.name,
                    "kind": node.kind,
                    "image_path": node.image_path,
                    "img_w": node.img_w,
                    "img_h": node.img_h,
                    "click_x": node.click_x,
                    "click_y": node.click_y,
                    "roi_x": node.roi_x,
                    "roi_y": node.roi_y,
                    "roi_w": node.roi_w,
                    "roi_h": node.roi_h,
                    "roi_weight": node.roi_weight,
                    "target_x": node.target_x,
                    "target_y": node.target_y,
                    "target_w": node.target_w,
                    "target_h": node.target_h,
                    "action_type": node.action_type,
                    "action": dict(node.action or {}),
                    "threshold": node.threshold,
                    "is_end": node.is_end,
                    "x": node.x,
                    "y": node.y,
                    "notes": node.notes,
                }
            )
        edges = []
        for edge in self.db.list_edges(self.flow_id):
            if edge.src_id in index and edge.dst_id in index:
                edges.append(
                    {
                        "src": index[edge.src_id],
                        "dst": index[edge.dst_id],
                        "weight": edge.weight,
                        "label": edge.label,
                    }
                )
        self._clipboard = {"nodes": entries, "edges": edges}
        self._paste_count = 0
        self.status_message.emit(f"已复制 {len(entries)} 个步骤")

    def _paste_nodes(self) -> None:
        if self.flow_id is None or not self._clipboard:
            self.status_message.emit("没有可粘贴的步骤")
            return
        entries = self._clipboard.get("nodes", [])
        if not entries:
            return
        self.push_undo("粘贴步骤")
        offset = 40 + self._paste_count * 30
        id_map: dict[int, int] = {}
        first_id = None
        for index, item in enumerate(entries):
            node_id = self.db.add_node(
                self.flow_id,
                name=(item.get("name") or "步骤") + " 副本",
                kind=item.get("kind", "image"),
                image_path=item.get("image_path"),
                img_w=item.get("img_w"),
                img_h=item.get("img_h"),
                click_x=item.get("click_x"),
                click_y=item.get("click_y"),
                roi_x=item.get("roi_x"),
                roi_y=item.get("roi_y"),
                roi_w=item.get("roi_w"),
                roi_h=item.get("roi_h"),
                roi_weight=float(item.get("roi_weight", 0.72) or 0.72),
                target_x=item.get("target_x"),
                target_y=item.get("target_y"),
                target_w=item.get("target_w"),
                target_h=item.get("target_h"),
                action_type=item.get("action_type", "tap"),
                action=item.get("action") or {},
                threshold=float(item.get("threshold", 0.88) or 0.88),
                is_entry=False,
                is_end=bool(item.get("is_end", False)),
                x=float(item.get("x", 0)) + offset,
                y=float(item.get("y", 0)) + offset,
                notes=item.get("notes", ""),
            )
            id_map[index] = node_id
            if first_id is None:
                first_id = node_id
            image_path = item.get("image_path")
            if image_path and Path(image_path).exists():
                try:
                    embedder = get_embedder()
                    vector = embedder.embed(image_path)
                    self.db.set_node_embedding(
                        node_id, embed_to_blob(vector), embedder.kind, int(vector.shape[0])
                    )
                except Exception as exc:
                    self.status_message.emit(f"向量计算失败：{exc}")
        for edge in self._clipboard.get("edges", []):
            src = id_map.get(edge.get("src"))
            dst = id_map.get(edge.get("dst"))
            if src and dst:
                self.db.add_edge(
                    self.flow_id,
                    src,
                    dst,
                    float(edge.get("weight", 1.0) or 1.0),
                    edge.get("label", ""),
                )
        self._paste_count += 1
        self.load_flow(self.flow_id)
        if first_id is not None:
            self.inspector.load_node(first_id, force=True)
        self.flow_changed.emit()
        self.status_message.emit(f"已粘贴 {len(entries)} 个步骤")

    def _connect(self) -> None:
        if not self.nodes:
            return
        src = self.inspector.node.id if self.inspector.node else None
        dialog = ConnectDialog(self.nodes, src=src, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.push_undo("添加连线")
        values = dialog.values()
        self.db.add_edge(
            self.flow_id, values["src"], values["dst"], values["weight"], values["label"]
        )
        self._reload_graph()
        self.inspector._reload_edges()
        self.flow_changed.emit()

    def _selected_edge_id(self) -> int | None:
        if self.selected_edge_id is not None:
            return self.selected_edge_id
        edge = self.inspector._selected_edge()
        return edge.id if edge else None

    def _cancel_connection(self) -> None:
        edge_id = self._selected_edge_id()
        if edge_id is None:
            QMessageBox.information(
                self, "提示", "请先在画布上单击一条连线，或在节点的下一步列表选中连线。"
            )
            return
        self.push_undo("取消连线")
        self.db.delete_edge(edge_id)
        self.selected_edge_id = None
        self._reload_graph()
        self.inspector._reload_edges()
        self.flow_changed.emit()
        self.status_message.emit("已取消该连线 / 下一步")

    def _edit_edge(self, edge_id: int) -> None:
        edge = self.db.get_edge(edge_id)
        if edge is None:
            return
        dialog = ConnectDialog(self.nodes, src=edge.src_id, dst=edge.dst_id, parent=self)
        dialog.weight.setValue(edge.weight)
        dialog.label.setText(edge.label)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.push_undo("编辑连线")
        values = dialog.values()
        self.db.add_edge(
            self.flow_id, values["src"], values["dst"], values["weight"], values["label"]
        )
        self._reload_graph()
        self.inspector._reload_edges()
        self.flow_changed.emit()

    def _delete_selected(self) -> None:
        if self.selected_edge_id is not None:
            self._cancel_connection()
            return
        if self.inspector.node:
            self._delete_node(self.inspector.node.id)

    def _delete_node(self, node_id: int) -> None:
        node = self.db.get_node(node_id)
        if node is None:
            return
        answer = QMessageBox.question(
            self, "删除节点", f"删除节点「{node.name}」以及相关连线？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.push_undo("删除节点")
        self.db.delete_node(node_id)
        self.inspector.clear_node()
        self.load_flow(self.flow_id)
        self.flow_changed.emit()

    def _auto_layout(self) -> None:
        if not self.nodes:
            return
        self.push_undo("自动布局")
        for node_id, (x, y) in compute_layout(self.nodes, self.edges).items():
            self.db.update_node(node_id, x=x, y=y)
        self._reload_graph()
        self.status_message.emit("已自动布局")

    def select_node(self, node_id: int) -> None:
        for item_id, item in self.graph._node_items.items():
            item.setSelected(item_id == node_id)
        self.inspector.load_node(node_id, force=True)
