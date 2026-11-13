from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QWidget


def make_scrollable_bar(bar: QHBoxLayout) -> QScrollArea:
    """Wrap a horizontal toolbar layout so it scrolls instead of forcing a wide window."""
    host = QWidget()
    host.setLayout(bar)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(host)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    host.adjustSize()
    scroll.setFixedHeight(host.sizeHint().height())
    return scroll
