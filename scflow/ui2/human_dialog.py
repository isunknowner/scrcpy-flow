"""人工介入对话框：运行卡住时手动选择点击/按键/打开应用/滑动/输入/等待。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

KEY_CHOICES = [
    ("返回", "back"),
    ("Home", "home"),
    ("最近任务", "app_switch"),
    ("回车", "enter"),
    ("菜单", "menu"),
    ("电源", "power"),
    ("音量+", "volume_up"),
    ("音量-", "volume_down"),
]

SWIPE_CHOICES = [("上滑", "up"), ("下滑", "down"), ("左滑", "left"), ("右滑", "right")]


class ClickableImage(QLabel):
    """显示截图，点击返回 0~1 的相对坐标。"""

    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("无截图")
        self.setObjectName("subtitle")
        self._pixmap: QPixmap | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image: QImage | None) -> None:
        if image is None:
            return
        self._pixmap = QPixmap.fromImage(image)
        self._refresh()

    def _scaled(self) -> QPixmap | None:
        if self._pixmap is None:
            return None
        return self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _refresh(self) -> None:
        scaled = self._scaled()
        if scaled is not None:
            self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def mousePressEvent(self, event) -> None:
        scaled = self._scaled()
        if scaled is None:
            return
        offset_x = (self.width() - scaled.width()) / 2
        offset_y = (self.height() - scaled.height()) / 2
        x = (event.position().x() - offset_x) / max(1, scaled.width())
        y = (event.position().y() - offset_y) / max(1, scaled.height())
        if 0 <= x <= 1 and 0 <= y <= 1:
            self.clicked.emit(float(x), float(y))


class HumanActionDialog(QDialog):
    """运行中的人工选择窗口。result 为动作 dict；取消为 None。"""

    def __init__(
        self,
        parent,
        frame: QImage | None,
        context: str = "",
        allow_ai: bool = True,
        failed_action: str = "",
        parent_window=None,
    ):
        super().__init__(parent_window or parent)
        self.result: dict | None = None
        self._tap_x = 0.5
        self._tap_y = 0.5
        self.setWindowTitle("人工选择")
        self.resize(860, 560)
        layout = QVBoxLayout(self)

        self.context_label = QLabel(context or "请选择下一步动作")
        self.context_label.setObjectName("subtitle")
        self.context_label.setWordWrap(True)
        if failed_action:
            self.context_label.setText(context + f"\n上次动作：{failed_action}")
        layout.addWidget(self.context_label)

        body = QHBoxLayout()
        layout.addLayout(body, 1)

        left = QVBoxLayout()
        self.preview = ClickableImage()
        self.preview.set_image(frame)
        self.preview.clicked.connect(self._on_preview_clicked)
        left.addWidget(self.preview, 1)
        self.tap_label = QLabel("点击左侧截图可选择点击位置（当前 50%, 50%）")
        self.tap_label.setObjectName("subtitle")
        self.tap_label.setWordWrap(True)
        left.addWidget(self.tap_label)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        form = QFormLayout()
        self.action_combo = QComboBox()
        for label, value in (
            ("点击屏幕", "tap"),
            ("按手机按键", "key"),
            ("打开应用", "launch"),
            ("滑动", "swipe"),
            ("输入文字", "text"),
            ("等待", "wait"),
        ):
            self.action_combo.addItem(label, value)
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)
        form.addRow("动作", self.action_combo)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._tap_page())
        self.stack.addWidget(self._key_page())
        self.stack.addWidget(self._launch_page())
        self.stack.addWidget(self._swipe_page())
        self.stack.addWidget(self._text_page())
        self.stack.addWidget(self._wait_page())
        form.addRow(self.stack)
        right.addLayout(form)

        self.remember_check = QCheckBox("记住这个选择（加入流程画布，之后当作保底动作）")
        self.remember_check.setChecked(True)
        self.monitor_check = QCheckBox("执行后检测是否完成（截图比对，不达标可再选/让 AI 托管）")
        self.monitor_check.setChecked(True)
        right.addWidget(self.remember_check)
        right.addWidget(self.monitor_check)
        right.addStretch(1)

        buttons = QHBoxLayout()
        self.ok_btn = QPushButton("执行")
        self.ok_btn.setObjectName("primary")
        self.ok_btn.clicked.connect(self._accept_action)
        cancel = QPushButton("取消（继续自动运行）")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self.ok_btn)
        if allow_ai:
            ai_btn = QPushButton("让 AI 托管")
            ai_btn.clicked.connect(self._accept_ai)
            buttons.addWidget(ai_btn)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        right.addLayout(buttons)
        body.addLayout(right, 2)

    # -- 各动作页 --------------------------------------------------------
    def _tap_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        hint = QLabel("直接点击左侧截图，或下面填相对坐标（0~1）。")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _key_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.key_combo = QComboBox()
        for label, value in KEY_CHOICES:
            self.key_combo.addItem(label, value)
        layout.addWidget(self.key_combo)
        layout.addStretch(1)
        return page

    def _launch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.package_edit = QLineEdit()
        self.package_edit.setPlaceholderText("包名，例如 com.tencent.mm")
        layout.addWidget(self.package_edit)
        hint = QLabel("提示：包名可以在「AI 设置 → 应用列表」里查看，或用 adb shell pm list packages 查询。")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _swipe_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.direction_combo = QComboBox()
        for label, value in SWIPE_CHOICES:
            self.direction_combo.addItem(label, value)
        layout.addWidget(self.direction_combo)
        layout.addStretch(1)
        return page

    def _text_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("要输入的文字")
        layout.addWidget(self.text_edit)
        self.submit_check = QCheckBox("输入后按回车")
        self.submit_check.setChecked(True)
        layout.addWidget(self.submit_check)
        layout.addStretch(1)
        return page

    def _wait_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.wait_spin = QDoubleSpinBox()
        self.wait_spin.setRange(0.1, 60.0)
        self.wait_spin.setDecimals(1)
        self.wait_spin.setValue(1.0)
        self.wait_spin.setSuffix(" 秒")
        layout.addWidget(self.wait_spin)
        layout.addStretch(1)
        return page

    # -- 交互 ------------------------------------------------------------
    def _on_preview_clicked(self, x: float, y: float) -> None:
        self._tap_x = max(0.0, min(1.0, float(x)))
        self._tap_y = max(0.0, min(1.0, float(y)))
        self.tap_label.setText(
            f"点击位置：{self._tap_x * 100:.1f}% , {self._tap_y * 100:.1f}%（可重新点击修改）"
        )
        if self.action_combo.currentData() != "tap":
            self.action_combo.setCurrentIndex(0)

    def _on_action_changed(self) -> None:
        self.stack.setCurrentIndex(max(0, self.action_combo.currentIndex()))

    def _accept_action(self) -> None:
        kind = str(self.action_combo.currentData() or "tap")
        action: dict = {"kind": kind}
        if kind == "tap":
            action["x"] = self._tap_x
            action["y"] = self._tap_y
        elif kind == "key":
            action["key"] = str(self.key_combo.currentData() or "back")
        elif kind == "launch":
            package = self.package_edit.text().strip()
            if not package:
                self.context_label.setText("请先填包名，例如 com.tencent.mm")
                return
            action["package"] = package
            action["fresh"] = True
        elif kind == "swipe":
            action["direction"] = str(self.direction_combo.currentData() or "up")
            action["duration"] = 300
        elif kind == "text":
            text = self.text_edit.text()
            if not text:
                self.context_label.setText("请先输入文字")
                return
            action["text"] = text
            action["clear"] = True
            action["submit"] = self.submit_check.isChecked()
        elif kind == "wait":
            action["seconds"] = float(self.wait_spin.value())
        action["remember"] = self.remember_check.isChecked()
        action["monitor"] = self.monitor_check.isChecked()
        self.result = action
        self.accept()

    def _accept_ai(self) -> None:
        self.result = {"ai": True}
        self.accept()
