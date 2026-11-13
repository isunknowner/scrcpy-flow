from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from .theme import ACCENT, GREEN, ORANGE, RED


class ImageEditor(QGraphicsView):
    point_selected = Signal(float, float)
    roi_selected = Signal(QRectF)
    target_selected = Signal(QRectF)
    relative_selected = Signal(QRectF)
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QColor("#111318"))
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._pixmap = QPixmap()
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self._mode = "view"
        self._zoom = 1.0
        self._drag_start: QPointF | None = None
        self._rubber: QGraphicsRectItem | None = None
        self._panning = False
        self._pan_pos = QPointF()
        self._click_item = None
        self._roi_item = None
        self._target_item = None
        self._relative_item = None
        self._relative_point_item = None
        self._roi_label = None
        self._target_label = None
        self._relative_label = None
        self._click = None
        self._roi = None
        self._target = None
        self._relative = None
        self._relative_point = None
        self.setMouseTracking(True)
        self._build_markers()

    def _build_markers(self) -> None:
        self._click_item = self.scene().addEllipse(
            QRectF(0, 0, 1, 1), QPen(QColor(RED), 2), QBrush(Qt.BrushStyle.NoBrush)
        )
        self._click_item.setVisible(False)
        self._roi_item = self.scene().addRect(
            QRectF(), QPen(QColor(ACCENT), 2, Qt.PenStyle.DashLine), QBrush(QColor(47, 129, 247, 45))
        )
        self._roi_item.setVisible(False)
        self._target_item = self.scene().addRect(
            QRectF(), QPen(QColor(ORANGE), 2, Qt.PenStyle.DashLine), QBrush(QColor(240, 136, 62, 40))
        )
        self._target_item.setVisible(False)
        self._roi_label = self.scene().addSimpleText("检测重点")
        self._roi_label.setBrush(QColor(ACCENT))
        self._roi_label.setVisible(False)
        self._target_label = self.scene().addSimpleText("点击对象")
        self._target_label.setBrush(QColor(ORANGE))
        self._target_label.setVisible(False)
        self._relative_item = self.scene().addRect(
            QRectF(), QPen(QColor(GREEN), 2, Qt.PenStyle.DashLine), QBrush(QColor(63, 185, 80, 40))
        )
        self._relative_item.setVisible(False)
        self._relative_label = self.scene().addSimpleText("相对点击图片")
        self._relative_label.setBrush(QColor(GREEN))
        self._relative_label.setVisible(False)
        self._relative_point_item = self.scene().addEllipse(
            QRectF(0, 0, 1, 1), QPen(QColor(GREEN), 3), QBrush(Qt.BrushStyle.NoBrush)
        )
        self._relative_point_item.setVisible(False)

    @property
    def mode(self) -> str:
        return self._mode

    def image_size(self) -> tuple[int, int]:
        if self._pixmap.isNull():
            return (0, 0)
        return (self._pixmap.width(), self._pixmap.height())

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if mode in ("click", "roi", "target", "relative", "relative_point")
            else Qt.CursorShape.ArrowCursor
        )

    def set_image(self, source) -> None:
        if source is None:
            self._pixmap = QPixmap()
        elif isinstance(source, QPixmap):
            self._pixmap = source
        else:
            self._pixmap = QPixmap(str(source))
        self._pixmap_item.setPixmap(self._pixmap)
        if self._pixmap.isNull():
            self.scene().setSceneRect(QRectF(0, 0, 1, 1))
            return
        self.scene().setSceneRect(QRectF(self._pixmap.rect()))
        self.fit_image()

    def fit_image(self) -> None:
        if self._pixmap.isNull():
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0
        self.zoom_changed.emit(1.0)

    def set_markers(
        self,
        click=None,
        roi=None,
        target=None,
        relative=None,
        relative_point=None,
    ) -> None:
        self._click = click
        self._roi = roi
        self._target = target
        self._relative = relative
        self._relative_point = relative_point
        width = max(1, self._pixmap.width())
        height = max(1, self._pixmap.height())
        if click:
            cx, cy = click[0] * width, click[1] * height
            self._click_item.setRect(QRectF(cx - 10, cy - 10, 20, 20))
            self._click_item.setVisible(True)
        else:
            self._click_item.setVisible(False)
        if roi:
            rect = QRectF(roi[0] * width, roi[1] * height, roi[2] * width, roi[3] * height)
            self._roi_item.setRect(rect)
            self._roi_item.setVisible(True)
            self._roi_label.setPos(rect.left() + 4, max(0, rect.top() - 18))
            self._roi_label.setVisible(True)
        else:
            self._roi_item.setVisible(False)
            self._roi_label.setVisible(False)
        if target:
            rect = QRectF(target[0] * width, target[1] * height, target[2] * width, target[3] * height)
            self._target_item.setRect(rect)
            self._target_item.setVisible(True)
            self._target_label.setPos(rect.left() + 4, min(height - 18, rect.bottom() + 3))
            self._target_label.setVisible(True)
        else:
            self._target_item.setVisible(False)
            self._target_label.setVisible(False)
        if relative:
            rect = QRectF(
                relative[0] * width,
                relative[1] * height,
                relative[2] * width,
                relative[3] * height,
            )
            self._relative_item.setRect(rect)
            self._relative_item.setVisible(True)
            self._relative_label.setPos(rect.left() + 4, min(height - 18, rect.bottom() + 3))
            self._relative_label.setVisible(True)
        else:
            self._relative_item.setVisible(False)
            self._relative_label.setVisible(False)
        if relative_point:
            cx = relative_point[0] * width
            cy = relative_point[1] * height
            self._relative_point_item.setRect(QRectF(cx - 9, cy - 9, 18, 18))
            self._relative_point_item.setVisible(True)
        else:
            self._relative_point_item.setVisible(False)

    def _to_relative(self, point: QPointF) -> tuple[float, float] | None:
        if self._pixmap.isNull():
            return None
        rect = QRectF(self._pixmap.rect())
        if not rect.contains(point):
            return None
        return (
            max(0.0, min(1.0, point.x() / max(1, rect.width()))),
            max(0.0, min(1.0, point.y() / max(1, rect.height()))),
        )

    def wheel_zoom(self, direction: int) -> None:
        if self._pixmap.isNull():
            return
        factor = 1.2 if direction > 0 else 1 / 1.2
        new_zoom = self._zoom * factor
        if new_zoom < 0.2 or new_zoom > 12:
            return
        self._zoom = new_zoom
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        self.wheel_zoom(1)

    def zoom_out(self) -> None:
        self.wheel_zoom(-1)

    def wheelEvent(self, event) -> None:
        if self._pixmap.isNull():
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        new_zoom = self._zoom * factor
        if new_zoom < 0.2 or new_zoom > 12:
            return
        self._zoom = new_zoom
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or self._pixmap.isNull():
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._mode in ("click", "relative_point"):
            relative = self._to_relative(scene_pos)
            if relative:
                if self._mode == "click":
                    self._click = relative
                else:
                    self._relative_point = relative
                self.set_markers(
                    self._click,
                    self._roi,
                    self._target,
                    self._relative,
                    self._relative_point,
                )
                self.point_selected.emit(*relative)
            event.accept()
            return
        if self._mode in ("roi", "target", "relative"):
            relative = self._to_relative(scene_pos)
            if not relative:
                return
            self._drag_start = scene_pos
            if self._rubber:
                self.scene().removeItem(self._rubber)
            color = {"target": ORANGE, "relative": GREEN}.get(self._mode, ACCENT)
            self._rubber = self.scene().addRect(
                QRectF(scene_pos, scene_pos),
                QPen(QColor(color), 2, Qt.PenStyle.DashLine),
                QBrush(QColor(color + "20" if len(color) == 7 else color)),
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
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
        if self._drag_start is not None and self._rubber is not None:
            point = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._drag_start, point).normalized()
            pixmap_rect = QRectF(self._pixmap.rect())
            rect = rect.intersected(pixmap_rect)
            self._rubber.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.CrossCursor if self._mode != "view" else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self._drag_start is not None and self._rubber is not None:
            rect = self._rubber.rect().normalized()
            self.scene().removeItem(self._rubber)
            self._rubber = None
            self._drag_start = None
            width = max(1, self._pixmap.width())
            height = max(1, self._pixmap.height())
            if rect.width() >= 4 and rect.height() >= 4:
                normalized = QRectF(
                    rect.left() / width,
                    rect.top() / height,
                    rect.width() / width,
                    rect.height() / height,
                )
                region = (
                    normalized.x(),
                    normalized.y(),
                    normalized.width(),
                    normalized.height(),
                )
                if self._mode == "target":
                    self._target = region
                    self.set_markers(self._click, self._roi, self._target, self._relative, self._relative_point)
                    self.target_selected.emit(normalized)
                elif self._mode == "relative":
                    self._relative = region
                    self.set_markers(self._click, self._roi, self._target, self._relative, self._relative_point)
                    self.relative_selected.emit(normalized)
                else:
                    self._roi = region
                    self.set_markers(self._click, self._roi, self._target, self._relative, self._relative_point)
                    self.roi_selected.emit(normalized)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._zoom <= 1.01 and not self._pixmap.isNull():
            self.fit_image()
