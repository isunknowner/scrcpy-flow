from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QSlider,
    QWidget,
)

# Controls whose value is normally changed by the mouse wheel.
VALUE_WIDGETS = (QAbstractSpinBox, QComboBox, QSlider)


def _nearest_scroll_area(widget: QWidget) -> QAbstractScrollArea | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return parent
        parent = parent.parentWidget()
    return None


class WheelFocusGuard(QObject):
    """Make wheel changes opt-in: a control only changes on wheel after it has focus.

    - Wheel over an unfocused combo box / spin box / slider scrolls the nearest
      scrollable container instead of changing the value.
    - Clicking a widget that cannot take focus (blank area, labels, group boxes)
      clears the current focus so subsequent wheel scrolling works again.
    """

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt naming)
        if not isinstance(obj, QWidget):
            return False
        if event.type() == QEvent.Type.Wheel:
            if isinstance(obj, VALUE_WIDGETS) and not obj.hasFocus():
                area = _nearest_scroll_area(obj)
                if area is not None:
                    bar = area.verticalScrollBar()
                    delta = event.angleDelta().y()
                    if delta:
                        bar.setValue(bar.value() - int(delta / 120.0 * 90))
                    else:
                        bar.setValue(bar.value() - int(event.pixelDelta().y() * 3))
                return True
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            if obj.focusPolicy() == Qt.FocusPolicy.NoFocus:
                focused = QApplication.focusWidget()
                if focused is not None:
                    focused.clearFocus()
        return False
