from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView

from .theme import ACCENT, BORDER, MUTED, ORANGE, PANEL, TEXT

NODE_W = 240.0
NODE_H = 126.0

KIND_COLORS = {
    "image": "#2f81f7",
    "end": "#f85149",
    "action": "#2f81f7",
    "detect": "#a371f7",
    "nodetect": "#39c5cf",
    "if": "#d2a8ff",
    "ai": "#3fb950",
    "manual": "#f0883e",
}
KIND_LABELS = {
    "image": "侧重点检测节点",
    "end": "结束步骤",
    "action": "侧重点检测节点",
    "detect": "侧重检测节点",
    "nodetect": "不检测步骤",
    "if": "IF 条件节点",
    "ai": "AI 决策节点",
    "manual": "人工步骤",
}


class NodeItem(QGraphicsItem):
    def __init__(self, node, view: "FlowGraphView"):
        super().__init__()
        self.node = node
        self.view = view
        self._thumb = None
        if node.image_path:
            pixmap = QPixmap(node.image_path)
            if not pixmap.isNull():
                self._thumb = pixmap.scaled(
                    62,
                    82,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
        )
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setZValue(2)
        self._original_pos = QPointF(float(node.x), float(node.y))
        self.setPos(self._original_pos)

    def boundingRect(self) -> QRectF:
        return QRectF(-2, -2, NODE_W + 4, NODE_H + 4)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, NODE_W, NODE_H), 10, 10)
        return path

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.view.update_edges()
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = self.isSelected()
        rect = QRectF(0, 0, NODE_W, NODE_H)
        painter.setPen(QPen(QColor(ACCENT if selected else BORDER), 2.2 if selected else 1.2))
        painter.setBrush(QBrush(QColor(PANEL)))
        painter.drawRoundedRect(rect, 10, 10)

        color = QColor(KIND_COLORS.get(self.node.kind, ACCENT))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color.darker(180)))
        painter.drawRoundedRect(QRectF(0, 0, NODE_W, 30), 9, 9)
        painter.drawRect(QRectF(0, 20, NODE_W, 10))

        painter.setPen(QPen(QColor("#ffffff")))
        font = QFont("Noto Sans CJK SC", 9)
        font.setBold(True)
        painter.setFont(font)
        title = self.node.name
        if painter.fontMetrics().horizontalAdvance(title) > NODE_W - 100:
            title = painter.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, int(NODE_W - 100))
        painter.drawText(QRectF(10, 5, NODE_W - 20, 22), Qt.AlignmentFlag.AlignVCenter, title)

        painter.setFont(QFont("Noto Sans CJK SC", 8))
        painter.setPen(QPen(QColor(MUTED)))
        painter.drawText(
            QRectF(10, 38, NODE_W - 82, 18),
            Qt.AlignmentFlag.AlignVCenter,
            KIND_LABELS.get(self.node.kind, self.node.kind),
        )
        painter.setPen(QPen(QColor(TEXT)))
        summary = self.node.action_summary
        summary = painter.fontMetrics().elidedText(
            summary, Qt.TextElideMode.ElideRight, int(NODE_W - 88)
        )
        painter.drawText(
            QRectF(10, 58, NODE_W - 82, 18), Qt.AlignmentFlag.AlignVCenter, summary
        )
        if self.node.has_roi:
            painter.setPen(QPen(QColor("#4cb1ff")))
            painter.drawText(
                QRectF(10, 78, 100, 16), Qt.AlignmentFlag.AlignVCenter, "检测重点"
            )
        if self.node.has_target:
            painter.setPen(QPen(QColor(ORANGE)))
            painter.drawText(
                QRectF(10, 96, 100, 16), Qt.AlignmentFlag.AlignVCenter, "点击对象"
            )
        if self._thumb is not None:
            x = NODE_W - self._thumb.width() - 10
            y = 38 + max(0, (82 - self._thumb.height()) / 2)
            painter.drawPixmap(int(x), int(y), self._thumb)
        else:
            painter.setPen(QPen(QColor(BORDER), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(NODE_W - 72, 40, 62, 78), 4, 4)
            painter.setPen(QPen(QColor(MUTED)))
            painter.setFont(QFont("Noto Sans CJK SC", 7))
            painter.drawText(
                QRectF(NODE_W - 72, 40, 62, 78),
                Qt.AlignmentFlag.AlignCenter,
                "无图片",
            )
        # Draw the badges last so the thumbnail never hides them.
        if self.node.is_entry:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(ACCENT)))
            painter.drawRoundedRect(QRectF(10, NODE_H - 26, 44, 18), 4, 4)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.setFont(QFont("Noto Sans CJK SC", 7))
            painter.drawText(
                QRectF(10, NODE_H - 26, 44, 18),
                Qt.AlignmentFlag.AlignCenter,
                "入口",
            )
        if bool(getattr(self.node, "is_end", False)) or self.node.kind == "end":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#f85149")))
            painter.drawRoundedRect(QRectF(NODE_W - 54, NODE_H - 26, 44, 18), 4, 4)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.setFont(QFont("Noto Sans CJK SC", 7))
            painter.drawText(
                QRectF(NODE_W - 54, NODE_H - 26, 44, 18),
                Qt.AlignmentFlag.AlignCenter,
                "结束",
            )


class EdgeItem(QGraphicsItem):
    def __init__(self, edge, src: NodeItem, dst: NodeItem):
        super().__init__()
        self.edge = edge
        self.src = src
        self.dst = dst
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(0)
        self._line = QPainterPath()
        self._arrow = QPolygonF()
        self._label_pos = QPointF()
        self.update_path()

    def boundingRect(self) -> QRectF:
        return self._line.boundingRect().united(
            self._arrow.boundingRect()
        ).adjusted(-20, -20, 20, 20)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(16)
        return stroker.createStroke(self._line)

    def center(self) -> QPointF:
        return self.mapToScene(QRectF(0, 0, NODE_W, NODE_H).center())

    def _border_point(self, node: NodeItem, target: QPointF) -> QPointF:
        center = node.pos() + QPointF(NODE_W / 2, NODE_H / 2)
        dx = target.x() - center.x()
        dy = target.y() - center.y()
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return center
        tx = (NODE_W / 2) / abs(dx) if dx else float("inf")
        ty = (NODE_H / 2) / abs(dy) if dy else float("inf")
        t = min(tx, ty)
        return center + QPointF(dx * t, dy * t)

    def update_path(self) -> None:
        self.prepareGeometryChange()
        src_center = self.src.pos() + QPointF(NODE_W / 2, NODE_H / 2)
        dst_center = self.dst.pos() + QPointF(NODE_W / 2, NODE_H / 2)
        start = self._border_point(self.src, dst_center)
        end = self._border_point(self.dst, src_center)
        self._line = QPainterPath(start)
        self._line.lineTo(end)
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        size = 10.0
        p1 = end
        p2 = QPointF(
            end.x() - size * math.cos(angle - math.pi / 7),
            end.y() - size * math.sin(angle - math.pi / 7),
        )
        p3 = QPointF(
            end.x() - size * math.cos(angle + math.pi / 7),
            end.y() - size * math.sin(angle + math.pi / 7),
        )
        self._arrow = QPolygonF([p1, p2, p3])
        self._label_pos = (start + end) / 2
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(ACCENT if self.isSelected() else "#5b6575")
        painter.setPen(QPen(color, 2.4 if self.isSelected() else 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._line)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(self._arrow)
        text = f"×{self.edge.weight:g}" if self.edge.weight != 1 else ""
        if self.edge.label:
            text = f"{self.edge.label} {text}".strip()
        if text:
            painter.setPen(QPen(QColor(MUTED)))
            painter.setFont(QFont("Noto Sans CJK SC", 7))
            rect = painter.fontMetrics().boundingRect(text).adjusted(-4, -2, 4, 2)
            rect.moveCenter(self._label_pos.toPoint())
            painter.setBrush(QBrush(QColor("#181a1f")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(rect), 3, 3)
            painter.setPen(QPen(QColor(MUTED)))
            painter.drawText(QRectF(rect), Qt.AlignmentFlag.AlignCenter, text)


class FlowGraphView(QGraphicsView):
    node_selected = Signal(int)
    edge_selected = Signal(int)
    node_double_clicked = Signal(int)
    edge_double_clicked = Signal(int)
    node_moved = Signal(int, float, float)
    node_drag_started = Signal(int)
    node_context_menu = Signal(int, QPoint)
    edge_context_menu = Signal(int, QPoint)
    connect_completed = Signal(int, int)
    connect_cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_spec = None
        self._bg_pixmap = None
        self._bg_scaled = None
        self._bg_scaled_size = None
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#111318"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setSceneRect(QRectF(-200, -200, 3000, 2000))
        self._node_items: dict[int, NodeItem] = {}
        self._edge_items: dict[int, EdgeItem] = {}
        self._zoom = 1.0
        self._panning = False
        self._pan_pos = QPointF()
        self._connect_src: int | None = None
        self._connect_target: int | None = None
        self._preview_pos: QPointF | None = None
        self.update_edges()

    def load(self, nodes, edges) -> None:
        self.cancel_connect(emit=False)
        scene = self.scene()
        scene.clear()
        self._node_items = {}
        self._edge_items = {}
        for node in nodes:
            item = NodeItem(node, self)
            scene.addItem(item)
            self._node_items[node.id] = item
        for edge in edges:
            src = self._node_items.get(edge.src_id)
            dst = self._node_items.get(edge.dst_id)
            if src and dst:
                item = EdgeItem(edge, src, dst)
                scene.addItem(item)
                self._edge_items[edge.id] = item
        self.update_edges()
        if nodes:
            rect = QRectF()
            for item in self._node_items.values():
                rect = rect.united(item.sceneBoundingRect())
            self.fitInView(rect.adjusted(-80, -80, 80, 80), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    def update_edges(self) -> None:
        for item in self._edge_items.values():
            item.update_path()

    def fit_view(self) -> None:
        if not self._node_items:
            return
        rect = QRectF()
        for item in self._node_items.values():
            rect = rect.united(item.sceneBoundingRect())
        self.fitInView(rect.adjusted(-80, -80, 80, 80), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0

    def set_background_spec(self, spec) -> None:
        self._bg_spec = spec
        self._refresh_background()

    def _refresh_background(self) -> None:
        spec = self._bg_spec or {}
        kind = spec.get("type", "default")
        value = spec.get("value")
        self._bg_scaled = None
        self._bg_scaled_size = None
        if kind == "image" and value:
            pixmap = QPixmap(value)
            self._bg_pixmap = pixmap if not pixmap.isNull() else None
            self.setBackgroundBrush(QColor("#111318"))
        else:
            self._bg_pixmap = None
            if kind == "color" and value:
                self.setBackgroundBrush(QColor(value))
            else:
                self.setBackgroundBrush(QColor("#111318"))
        self.viewport().update()

    def _scaled_background(self):
        size = self.viewport().size()
        if self._bg_pixmap is None or self._bg_pixmap.isNull() or size.width() <= 0:
            return None
        if self._bg_scaled is None or self._bg_scaled_size != size:
            self._bg_scaled = self._bg_pixmap.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._bg_scaled_size = size
        return self._bg_scaled

    def drawBackground(self, painter, rect) -> None:
        scaled = self._scaled_background()
        if scaled is not None:
            painter.save()
            painter.resetTransform()
            viewport = self.viewport().rect()
            x = (viewport.width() - scaled.width()) // 2
            y = (viewport.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.restore()
            return
        super().drawBackground(painter, rect)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_bg_spec", None):
            self._refresh_background()

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if self._zoom * factor < 0.15 or self._zoom * factor > 8:
            return
        self._zoom *= factor
        self.scale(factor, factor)

    def _node_at(self, pos):
        item = self.itemAt(pos)
        while item is not None and not isinstance(item, NodeItem):
            item = item.parentItem()
        return item if isinstance(item, NodeItem) else None

    def start_connect(self, node_id: int) -> None:
        if node_id not in self._node_items:
            return
        self._connect_src = node_id
        self._connect_target = None
        source = self._node_items[node_id]
        self._preview_pos = source.pos() + QPointF(NODE_W / 2, NODE_H / 2)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport().setMouseTracking(True)
        self.viewport().update()

    def cancel_connect(self, emit: bool = True) -> None:
        if self._connect_src is None:
            return
        self._connect_src = None
        self._connect_target = None
        self._preview_pos = None
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()
        if emit:
            self.connect_cancelled.emit()

    def drawForeground(self, painter, rect) -> None:
        if self._connect_src is None or self._preview_pos is None:
            super().drawForeground(painter, rect)
            return
        source = self._node_items.get(self._connect_src)
        if source is None:
            return
        start = source.pos() + QPointF(NODE_W / 2, NODE_H / 2)
        painter.setPen(QPen(QColor(ACCENT), 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(start, self._preview_pos)
        if self._connect_target is not None:
            target = self._node_items.get(self._connect_target)
            if target is not None:
                painter.setPen(QPen(QColor(ACCENT), 2))
                painter.drawRect(target.sceneBoundingRect().adjusted(-4, -4, 4, 4))
        super().drawForeground(painter, rect)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self._connect_src is not None:
            self.cancel_connect()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._connect_src is not None:
            if event.button() == Qt.MouseButton.RightButton:
                self.cancel_connect()
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                item = self._node_at(event.position().toPoint())
                if item is not None and item.node.id != self._connect_src:
                    src = self._connect_src
                    dst = item.node.id
                    self.cancel_connect(emit=False)
                    self.connect_completed.emit(src, dst)
                else:
                    self.cancel_connect()
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            while item and not isinstance(item, (NodeItem, EdgeItem)):
                item = item.parentItem()
            if isinstance(item, NodeItem):
                self.node_drag_started.emit(item.node.id)
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._connect_src is not None:
            self._preview_pos = self.mapToScene(event.position().toPoint())
            item = self._node_at(event.position().toPoint())
            self._connect_target = (
                item.node.id
                if item is not None and item.node.id != self._connect_src
                else None
            )
            self.viewport().update()
            event.accept()
            return
        if self._panning:
            delta = event.position() - self._pan_pos
            self._pan_pos = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
        for node_id, item in self._node_items.items():
            if item.pos() != item._original_pos:
                item._original_pos = QPointF(item.pos())
                self.node_moved.emit(node_id, item.pos().x(), item.pos().y())
        self._emit_selection()

    def _emit_selection(self) -> None:
        for item in self.scene().selectedItems():
            if isinstance(item, NodeItem):
                self.node_selected.emit(item.node.id)
                return
            if isinstance(item, EdgeItem):
                self.edge_selected.emit(item.edge.id)
                return
        self.node_selected.emit(-1)

    def mouseDoubleClickEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        while item and not isinstance(item, (NodeItem, EdgeItem)):
            item = item.parentItem()
        if isinstance(item, NodeItem):
            self.node_double_clicked.emit(item.node.id)
        elif isinstance(item, EdgeItem):
            self.edge_double_clicked.emit(item.edge.id)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        while item is not None and not isinstance(item, (NodeItem, EdgeItem)):
            item = item.parentItem()
        if isinstance(item, NodeItem):
            self.scene().clearSelection()
            item.setSelected(True)
            self.node_context_menu.emit(item.node.id, event.globalPos())
            event.accept()
            return
        if isinstance(item, EdgeItem):
            self.scene().clearSelection()
            item.setSelected(True)
            self.edge_context_menu.emit(item.edge.id, event.globalPos())
            event.accept()
            return
        super().contextMenuEvent(event)
