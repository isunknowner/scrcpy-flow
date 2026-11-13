from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import config, vision
from ..device import AdbDevice

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_FILTER = "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"


def _unique_target(prefix: str, suffix: str) -> Path:
    suffix = (suffix or ".png").lower()
    if suffix not in IMAGE_EXTS:
        suffix = ".png"
    return config.assets_dir() / f"{prefix}_{uuid4().hex[:12]}{suffix}"


def list_assets() -> list[Path]:
    root = config.assets_dir()
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return files


def add_file_to_assets(source: str | Path, prefix: str = "素材") -> Path:
    source = Path(source)
    target = _unique_target(prefix, source.suffix)
    shutil.copy2(source, target)
    return target


def add_frame_to_assets(frame, prefix: str = "截图") -> Path:
    target = _unique_target(prefix, ".png")
    vision.save_rgb(target, frame)
    return target


def _thumbnail_icon(path: Path, size: int = 120) -> QIcon:
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return QIcon()
    pixmap = pixmap.scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
    return QIcon(pixmap)


def _configure_icon_list(widget: QListWidget) -> None:
    widget.setViewMode(QListWidget.ViewMode.IconMode)
    widget.setResizeMode(QListWidget.ResizeMode.Adjust)
    widget.setMovement(QListWidget.Movement.Static)
    widget.setIconSize(QSize(120, 120))
    widget.setGridSize(QSize(150, 170))
    widget.setWordWrap(True)
    widget.setSpacing(6)
    widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)


def _populate(widget: QListWidget) -> None:
    widget.clear()
    for path in list_assets():
        item = QListWidgetItem(_thumbnail_icon(path), path.name)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setToolTip(str(path))
        item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
        widget.addItem(item)


class AssetPickerDialog(QDialog):
    """Choose a single image from the material library."""

    def __init__(self, parent=None, title: str = "从素材库选择图片"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(660, 520)
        self.selected_path: str | None = None

        layout = QVBoxLayout(self)
        hint = QLabel("双击图片直接选择；素材库为空时可先导入。")
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        self.list = QListWidget()
        _configure_icon_list(self.list)
        layout.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.import_btn = QPushButton("导入图片…")
        self.refresh_btn = QPushButton("刷新")
        self.cancel_btn = QPushButton("取消")
        self.ok_btn = QPushButton("选择")
        self.ok_btn.setObjectName("primary")
        row.addWidget(self.import_btn)
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.ok_btn)
        layout.addLayout(row)

        self.import_btn.clicked.connect(self._import)
        self.refresh_btn.clicked.connect(self._refresh)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn.clicked.connect(self._accept)
        self.list.itemDoubleClicked.connect(lambda _item: self._accept())
        self._refresh()

    def _refresh(self) -> None:
        _populate(self.list)

    def _import(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "导入图片到素材库", "", IMAGE_FILTER)
        for path in paths:
            try:
                add_file_to_assets(path)
            except Exception as exc:
                QMessageBox.warning(self, "导入失败", f"{path}\n{exc}")
        if paths:
            self._refresh()

    def _accept(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()


class AssetLibraryWidget(QWidget):
    """Material library: capture/import images and push them into a flow."""

    status_message = Signal(str)
    assets_changed = Signal()

    def __init__(self, flow_workspace=None, parent=None):
        super().__init__(parent)
        self.flow_workspace = flow_workspace
        self.serial: str | None = None
        self._build()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        top.addWidget(QLabel("素材库"))
        self.count_label = QLabel("0 张")
        self.count_label.setObjectName("subtitle")
        top.addWidget(self.count_label)
        top.addStretch(1)
        layout.addLayout(top)

        self.list = QListWidget()
        _configure_icon_list(self.list)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        layout.addWidget(self.list, 1)

        row1 = QHBoxLayout()
        self.capture_btn = QPushButton("截取手机当前图片")
        self.capture_btn.setObjectName("primary")
        self.import_btn = QPushButton("导入图片")
        self.refresh_btn = QPushButton("刷新")
        row1.addWidget(self.capture_btn)
        row1.addWidget(self.import_btn)
        row1.addWidget(self.refresh_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.add_to_flow_btn = QPushButton("添加到流程")
        self.delete_btn = QPushButton("删除")
        self.delete_btn.setObjectName("danger")
        row2.addWidget(self.add_to_flow_btn)
        row2.addStretch(1)
        row2.addWidget(self.delete_btn)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.open_folder_btn = QPushButton("打开文件所在位置")
        self.redirect_btn = QPushButton("重定向文件所在位置")
        row3.addWidget(self.open_folder_btn)
        row3.addWidget(self.redirect_btn)
        layout.addLayout(row3)

        hint = QLabel("双击素材可直接加入当前流程")
        hint.setObjectName("subtitle")
        layout.addWidget(hint)

        self.capture_btn.clicked.connect(self.capture_current)
        self.import_btn.clicked.connect(self.import_files)
        self.refresh_btn.clicked.connect(self.refresh)
        self.add_to_flow_btn.clicked.connect(self.add_selected_to_flow)
        self.open_folder_btn.clicked.connect(self.open_selected_location)
        self.redirect_btn.clicked.connect(self.redirect_selected)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.list.itemDoubleClicked.connect(lambda _item: self.add_selected_to_flow())
        self.list.customContextMenuRequested.connect(self._on_context_menu)

    # -- helpers -----------------------------------------------------------
    def set_serial(self, serial: str | None) -> None:
        self.serial = serial

    def _selected_path(self) -> Path | None:
        item = self.list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return Path(value) if value else None

    def _select_path(self, path: Path) -> None:
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == str(path):
                self.list.setCurrentItem(item)
                return

    def _on_context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        self.list.setCurrentItem(item)
        menu = QMenu(self)
        rename_action = menu.addAction("重命名图片")
        add_action = menu.addAction("添加到流程")
        open_action = menu.addAction("打开文件所在位置")
        redirect_action = menu.addAction("重定向文件所在位置")
        delete_action = menu.addAction("删除")
        chosen = menu.exec(self.list.viewport().mapToGlobal(pos))
        if chosen is rename_action:
            self.rename_selected()
        elif chosen is add_action:
            self.add_selected_to_flow()
        elif chosen is open_action:
            self.open_selected_location()
        elif chosen is redirect_action:
            self.redirect_selected()
        elif chosen is delete_action:
            self.delete_selected()

    # -- actions -----------------------------------------------------------
    def refresh(self) -> None:
        _populate(self.list)
        self.count_label.setText(f"{self.list.count()} 张")

    def capture_current(self) -> Path | None:
        try:
            device = AdbDevice(serial=self.serial)
            device.check()
            frame = device.screenshot()
        except Exception as exc:
            QMessageBox.critical(self, "截图失败", str(exc))
            return None
        target = add_frame_to_assets(frame)
        self.refresh()
        self._select_path(target)
        self.assets_changed.emit()
        self.status_message.emit(f"已截取手机画面并加入素材库：{target.name}")

        flow = self.flow_workspace
        if flow is not None and getattr(flow, "flow_id", None) is not None:
            answer = QMessageBox.question(
                self,
                "加入流程",
                "检测到正在编辑流程，是否把截取的图片加入当前流程？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                node_id = flow.add_node_from_path(str(target), name=f"截图 {target.stem}")
                if node_id is not None:
                    self.status_message.emit("截取的图片已加入当前流程")
        return target

    def import_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "导入图片到素材库", "", IMAGE_FILTER)
        if not paths:
            return
        last: Path | None = None
        for path in paths:
            try:
                last = add_file_to_assets(path)
            except Exception as exc:
                QMessageBox.warning(self, "导入失败", f"{path}\n{exc}")
        self.refresh()
        if last is not None:
            self._select_path(last)
        self.assets_changed.emit()
        self.status_message.emit(f"已导入 {len(paths)} 张图片到素材库")

    def add_selected_to_flow(self) -> None:
        path = self._selected_path()
        if path is None:
            QMessageBox.information(self, "提示", "请先在素材库中选择一张图片")
            return
        flow = self.flow_workspace
        if flow is None or getattr(flow, "flow_id", None) is None:
            QMessageBox.information(self, "提示", "请先选择一个流程")
            return
        node_id = flow.add_node_from_path(str(path), name=f"素材 {path.stem}")
        if node_id is not None:
            self.status_message.emit(f"已把素材加入流程：{path.name}")

    def open_selected_location(self) -> None:
        path = self._selected_path()
        if path is None:
            QMessageBox.information(self, "提示", "请先在素材库中选择一张图片")
            return
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", f"/select,{path}"])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc))

    def redirect_selected(self) -> None:
        path = self._selected_path()
        if path is None:
            QMessageBox.information(self, "提示", "请先在素材库中选择一张图片")
            return
        source, _ = QFileDialog.getOpenFileName(
            self, "重定向素材到文件", str(path.parent), IMAGE_FILTER
        )
        if not source:
            return
        src = Path(source)
        try:
            if src.resolve() == path.resolve():
                return
        except OSError:
            pass
        answer = QMessageBox.question(
            self, "重定向文件", f"用所选文件替换素材「{path.name}」？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.copy2(src, path)
        except Exception as exc:
            QMessageBox.warning(self, "重定向失败", str(exc))
            return
        self.refresh()
        self._select_path(path)
        self.assets_changed.emit()
        self.status_message.emit(f"已重定向素材：{path.name}")

    def rename_selected(self) -> None:
        path = self._selected_path()
        if path is None:
            return
        name, ok = QInputDialog.getText(self, "重命名图片", "图片名称：", text=path.stem)
        if not ok:
            return
        name = re.sub(r'[\\/:*?"<>|]', "_", name).strip().rstrip(". ")
        for ext in IMAGE_EXTS:
            if name.lower().endswith(ext):
                name = name[: -len(ext)]
                break
        if not name or name == path.stem:
            return
        suffix = path.suffix or ".png"
        target = path.with_name(f"{name}{suffix}")
        if str(target).lower() == str(path).lower():
            return
        if target.exists():
            QMessageBox.warning(self, "重命名失败", f"已存在同名图片：{target.name}")
            return
        try:
            path.rename(target)
        except OSError as exc:
            QMessageBox.warning(self, "重命名失败", str(exc))
            return
        self.refresh()
        self._select_path(target)
        self.assets_changed.emit()
        self.status_message.emit(f"素材已重命名：{target.name}")

    def delete_selected(self) -> None:
        path = self._selected_path()
        if path is None:
            return
        answer = QMessageBox.question(
            self, "删除素材", f"从素材库删除「{path.name}」？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            path.unlink()
        except OSError as exc:
            QMessageBox.warning(self, "删除失败", str(exc))
            return
        self.refresh()
        self.assets_changed.emit()
        self.status_message.emit(f"已删除素材：{path.name}")

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        self.refresh()
