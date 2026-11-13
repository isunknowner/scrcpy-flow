from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#2f81f7"
BG = "#181a1f"
PANEL = "#20232a"
PANEL_ALT = "#252932"
BORDER = "#333842"
TEXT = "#d7dce5"
MUTED = "#8b95a7"
GREEN = "#3fb950"
ORANGE = "#f0883e"
RED = "#f85149"

QSS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QMainWindow::separator {{
    background: {BORDER};
    width: 3px;
    height: 3px;
}}
QToolBar {{
    background: {PANEL};
    border: none;
    spacing: 4px;
    padding: 4px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 9px;
}}
QToolButton:hover {{
    background: {PANEL_ALT};
    border-color: {BORDER};
}}
QToolButton:checked {{
    background: {ACCENT};
    color: white;
}}
QPushButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 12px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #2d3542; }}
QPushButton:disabled {{ color: #5c6575; }}
QPushButton#primary {{
    background: {ACCENT};
    color: white;
    border: none;
    font-weight: 600;
}}
QPushButton#danger {{
    background: #3a1f22;
    border-color: #6e2a2f;
    color: #ffb3b8;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: #16181d;
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG};
}}
QTabBar::tab {{
    background: {PANEL};
    color: {MUTED};
    padding: 8px 18px;
    border: 1px solid transparent;
    border-bottom: none;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {BG};
    color: {TEXT};
    border-color: {BORDER};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QDockWidget {{
    font-weight: 600;
}}
QDockWidget::title {{
    background: {PANEL};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 2px;
}}
QListWidget, QTreeWidget, QTableWidget, QTableView {{
    background: #16181d;
    border: 1px solid {BORDER};
    alternate-background-color: #1a1e25;
    gridline-color: {BORDER};
    selection-background-color: #24405f;
    selection-color: white;
}}
QListWidget::item {{ padding: 6px; }}
QListWidget::item:selected {{ background: #24405f; }}
QHeaderView::section {{
    background: {PANEL_ALT};
    color: {MUTED};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 5px;
}}
QScrollBar:vertical {{ background: {BG}; width: 11px; }}
QScrollBar::handle:vertical {{ background: #3c4350; min-height: 24px; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: #4b5668; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; }}
QScrollBar::handle:horizontal {{ background: #3c4350; min-width: 24px; border-radius: 5px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected {{ background: {PANEL_ALT}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QSplitter::handle {{ background: {BORDER}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QLabel#title {{ font-size: 20px; font-weight: 700; color: white; }}
QLabel#subtitle {{ color: {MUTED}; }}
QLabel#badge {{
    background: #1f3a2c;
    color: {GREEN};
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {MUTED};
}}
QGraphicsView {{ border: 1px solid {BORDER}; background: #111318; }}
QDialog {{ background: {PANEL}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor("#16181d"))
    palette.setColor(QPalette.AlternateBase, QColor(PANEL_ALT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(PANEL_ALT))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipBase, QColor(PANEL))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
    app.setStyleSheet(QSS)
