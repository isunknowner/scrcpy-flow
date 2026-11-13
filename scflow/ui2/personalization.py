from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtGui import QBrush, QColor, QPalette, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

SETTINGS_KEY = "personalization"

# key -> display label (order matters for the dialog and application)
TARGETS = [
    ("project", "区块编排"),
    ("flow", "流程画布"),
    ("run", "运行"),
    ("history", "历史"),
    ("inspector", "属性 / 检测重点"),
    ("assets", "素材库"),
    ("console", "控制台"),
]
TARGET_LABELS = dict(TARGETS)

# spec types:
#   missing override            -> inherit unified ("all")
#   {"type": "default"}         -> theme default
#   {"type": "color", "value"}  -> solid colour
#   {"type": "image", "value"}  -> image file
DEFAULT_SPEC = {"type": "default"}


def _as_posix(path: str) -> str:
    return Path(path).as_posix()


class _CoverBackgroundFilter(QObject):
    """Paint an image as a cover background (fill the widget, crop overflow)."""

    def __init__(self, widget, path: str):
        super().__init__(widget)
        self.widget = widget
        self.path = path
        self.target = widget.viewport() if hasattr(widget, "viewport") else widget
        self.target.installEventFilter(self)
        self.apply()

    def apply(self) -> None:
        pixmap = QPixmap(self.path)
        if pixmap.isNull() or not self.widget.objectName():
            return
        size = self.target.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        scaled = pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (scaled.width() - size.width()) // 2
        y = (scaled.height() - size.height()) // 2
        cropped = scaled.copy(x, y, size.width(), size.height())
        palette = self.target.palette()
        palette.setBrush(QPalette.ColorRole.Base, QBrush(cropped))
        self.target.setPalette(palette)
        self.target.setAutoFillBackground(True)
        self.widget.setStyleSheet(
            f"#{self.widget.objectName()} {{ background: palette(base); }}"
        )
        self.target.update()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt naming)
        if obj is self.target and event.type() == QEvent.Type.Resize:
            self.apply()
        return False

    def detach(self) -> None:
        try:
            self.target.removeEventFilter(self)
            palette = self.target.palette()
            palette.setBrush(QPalette.ColorRole.Base, QBrush())
            self.target.setPalette(palette)
        except RuntimeError:
            pass
        try:
            self.widget.setStyleSheet("")
            self.widget.update()
        except RuntimeError:
            pass


class PersonalizationManager(QObject):
    def __init__(self, widgets: dict, parent=None):
        super().__init__(parent)
        self.widgets = widgets
        self.settings = QSettings("scrcpy-flow", "workspace-2")
        raw = self.settings.value(SETTINGS_KEY, "{}")
        try:
            data = json.loads(raw) if isinstance(raw, str) else {}
        except (TypeError, ValueError):
            data = {}
        self._data = data if isinstance(data, dict) else {}
        self.apply_all()

    # -- persistence -------------------------------------------------------
    def _save(self) -> None:
        self.settings.setValue(SETTINGS_KEY, json.dumps(self._data, ensure_ascii=False))

    def _overrides(self) -> dict:
        overrides = self._data.get("overrides")
        if not isinstance(overrides, dict):
            overrides = {}
            self._data["overrides"] = overrides
        return overrides

    # -- specs -------------------------------------------------------------
    def unified_spec(self) -> dict:
        spec = self._data.get("all")
        return spec if isinstance(spec, dict) and spec.get("type") else dict(DEFAULT_SPEC)

    def override_spec(self, key: str) -> dict | None:
        spec = self._overrides().get(key)
        return spec if isinstance(spec, dict) and spec.get("type") else None

    def effective_spec(self, key: str) -> dict:
        spec = self.override_spec(key)
        if spec is not None:
            return spec
        return self.unified_spec()

    def display_spec(self, scope) -> tuple[dict, str]:
        """Return (spec, type) for the editor; type is 'inherit' when following unified."""
        if scope is None:
            spec = self.unified_spec()
            return spec, spec.get("type", "default")
        override = self.override_spec(scope)
        if override is None:
            return self.unified_spec(), "inherit"
        return override, override.get("type", "default")

    def set_spec(self, scope, spec: dict) -> None:
        kind = spec.get("type", "default")
        if scope is None:
            if kind == "default":
                self._data.pop("all", None)
            else:
                self._data["all"] = spec
        else:
            overrides = self._overrides()
            if kind == "inherit":
                overrides.pop(scope, None)
            else:
                overrides[scope] = spec
        self._save()
        self.apply_all()

    def reset_all(self) -> None:
        self._data = {}
        self._save()
        self.apply_all()

    # -- application -------------------------------------------------------
    def apply_all(self) -> None:
        for key, widget in self.widgets.items():
            self._apply(widget, self.effective_spec(key))

    def _apply(self, widget, spec: dict) -> None:
        kind = (spec or {}).get("type", "default")
        value = (spec or {}).get("value")
        previous = getattr(widget, "_bg_cover_filter", None)
        if previous is not None:
            previous.detach()
            widget._bg_cover_filter = None
        if isinstance(widget, QGraphicsView) and hasattr(widget, "set_background_spec"):
            widget.set_background_spec(spec or DEFAULT_SPEC)
            return
        name = widget.objectName()
        if not name:
            return
        if kind == "image" and value and hasattr(widget, "viewport"):
            widget._bg_cover_filter = _CoverBackgroundFilter(widget, value)
            return
        if kind == "color" and value:
            widget.setStyleSheet(f"#{name} {{ background-color: {value}; }}")
        elif kind == "image" and value:
            widget.setStyleSheet(
                f'#{name} {{ border-image: url("{_as_posix(value)}") 0 0 0 0 stretch stretch; }}'
            )
        else:
            widget.setStyleSheet("")


class PersonalizationDialog(QDialog):
    def __init__(self, manager: PersonalizationManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("个性化")
        self.resize(620, 440)
        self._loading = False

        layout = QVBoxLayout(self)
        hint = QLabel(
            "选择「全部窗口」可统一设置所有窗口背景；选择具体窗口可单独设置。"
        )
        hint.setObjectName("subtitle")
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.target_list = QListWidget()
        self.target_list.addItem(QListWidgetItem("全部窗口（统一）"))
        for key, label in TARGETS:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.target_list.addItem(item)
        row.addWidget(self.target_list, 1)

        form = QFormLayout()
        self.type_combo = QComboBox()
        form.addRow("背景类型", self.type_combo)
        self.color_btn = QPushButton("选择颜色…")
        form.addRow("纯色", self.color_btn)
        image_row = QHBoxLayout()
        self.image_edit = QLineEdit()
        self.image_edit.setReadOnly(True)
        self.image_edit.setPlaceholderText("未选择图片")
        self.browse_btn = QPushButton("选择图片…")
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(self.browse_btn)
        form.addRow("图片", image_row)
        self.color_label = QLabel("")
        form.addRow("当前颜色", self.color_label)
        row.addLayout(form, 1)
        layout.addLayout(row, 1)

        bottom = QHBoxLayout()
        self.reset_btn = QPushButton("全部恢复默认")
        self.close_btn = QPushButton("关闭")
        self.close_btn.setObjectName("primary")
        bottom.addStretch(1)
        bottom.addWidget(self.reset_btn)
        bottom.addWidget(self.close_btn)
        layout.addLayout(bottom)

        self.target_list.currentRowChanged.connect(self._on_scope_changed)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        self.color_btn.clicked.connect(self._pick_color)
        self.browse_btn.clicked.connect(self._pick_image)
        self.reset_btn.clicked.connect(self._reset_all)
        self.close_btn.clicked.connect(self.accept)
        self.target_list.setCurrentRow(0)

    # -- helpers -----------------------------------------------------------
    def _scope(self):
        item = self.target_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _on_scope_changed(self, _row: int) -> None:
        self._reload_editor()

    def _reload_editor(self) -> None:
        scope = self._scope()
        self._loading = True
        try:
            spec, kind = self.manager.display_spec(scope)
            self.type_combo.clear()
            if scope is not None:
                self.type_combo.addItem("跟随统一背景", "inherit")
            self.type_combo.addItem("默认主题", "default")
            self.type_combo.addItem("纯色", "color")
            self.type_combo.addItem("图片", "image")
            index = self.type_combo.findData(kind)
            self.type_combo.setCurrentIndex(index if index >= 0 else 0)
            value = spec.get("value") if isinstance(spec, dict) else None
            if kind == "image" and value:
                self.image_edit.setText(str(value))
            else:
                self.image_edit.clear()
            if kind == "color" and value:
                self.color_label.setText(value)
                self.color_btn.setStyleSheet(f"background-color: {value};")
            else:
                self.color_label.setText("")
                self.color_btn.setStyleSheet("")
        finally:
            self._loading = False

    def _on_type_changed(self, _index: int) -> None:
        if self._loading:
            return
        kind = self.type_combo.currentData()
        if kind == "inherit":
            self.manager.set_spec(self._scope(), {"type": "inherit"})
            self._reload_editor()
        elif kind == "default":
            self.manager.set_spec(self._scope(), {"type": "default"})
            self._reload_editor()
        elif kind == "color":
            self._pick_color()
        elif kind == "image":
            self._pick_image()

    def _pick_color(self) -> None:
        current = self.color_label.text() or "#181a1f"
        color = QColorDialog.getColor(QColor(current), self, "选择背景颜色")
        if not color.isValid():
            self._reload_editor()
            return
        self.manager.set_spec(self._scope(), {"type": "color", "value": color.name()})
        self._reload_editor()

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图片", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            self._reload_editor()
            return
        self.manager.set_spec(self._scope(), {"type": "image", "value": path})
        self._reload_editor()

    def _reset_all(self) -> None:
        self.manager.reset_all()
        self._reload_editor()
