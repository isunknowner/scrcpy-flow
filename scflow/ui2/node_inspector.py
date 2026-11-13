from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import config, vision
from ..db import Database
from ..device import AdbDevice
from ..embeddings import embed_to_blob, get_embedder
from ..models import Node
from .asset_library import AssetPickerDialog, add_file_to_assets, add_frame_to_assets
from .dialogs import ConnectDialog
from .image_editor import ImageEditor
from .theme import ACCENT, GREEN, ORANGE

KIND_LABELS = {
    "image": "侧重点检测节点",
    "detect": "侧重检测节点",
    "nodetect": "不检测步骤",
    "if": "IF 条件节点",
    "ai": "AI 决策节点",
    "manual": "人工步骤",
}
KIND_VALUES = {value: key for key, value in KIND_LABELS.items()}
ACTION_LABELS = {
    "tap": "点击",
    "swipe": "滑动",
    "burst": "极速连滑/连点（100+ 次/秒）",
    "key": "按键",
    "text": "输入文本",
    "launch": "启动应用",
    "wait": "等待",
    "none": "无动作",
    "ai": "AI 决策",
}
ACTION_VALUES = {value: key for key, value in ACTION_LABELS.items()}


class NodeInspector(QWidget):
    node_saved = Signal(int)
    edges_changed = Signal()
    edit_similarity_requested = Signal(int)
    delete_requested = Signal(int)
    status_message = Signal(str)
    assets_changed = Signal()

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.node: Node | None = None
        self.flow_id: int | None = None
        self.serial: str | None = None
        self.undo_callback = None
        self.current_click = None
        self.current_roi = None
        self.current_target = None
        self.current_relative = None
        self.current_relative_point = None
        self._loading = False
        self._dirty = False
        self._image_undo_pushed = False
        self._action_widgets: dict[str, QWidget] = {}
        self._last_action_label = ACTION_LABELS["tap"]
        self._syncing_no_click = False
        self._settings = QSettings("scrcpy-flow", "workspace-2")
        self._auto_save = self._settings.value("auto_save", False, type=bool)
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.setInterval(400)
        self._auto_save_timer.timeout.connect(self._auto_save_now)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        auto_row = QHBoxLayout()
        auto_row.setContentsMargins(10, 8, 10, 0)
        self.auto_save_check = QCheckBox("自动保存每一步修改")
        self.auto_save_check.setToolTip(
            "勾选后，属性有改动会自动保存；不勾选则需点击底部「保存节点」。"
        )
        self.auto_save_check.setChecked(self._auto_save)
        auto_row.addWidget(self.auto_save_check)
        auto_row.addStretch(1)
        outer.addLayout(auto_row)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)
        body = QWidget()
        self.scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 10, 10, 10)

        image_group = QGroupBox("状态图片")
        image_layout = QVBoxLayout(image_group)
        self.image_editor = ImageEditor()
        self.image_editor.setMinimumHeight(220)
        image_layout.addWidget(self.image_editor, 1)
        zoom_row = QHBoxLayout()
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("subtitle")
        self.zoom_in_btn = QPushButton("放大")
        self.zoom_out_btn = QPushButton("缩小")
        self.zoom_fit_btn = QPushButton("适应窗口")
        for button in (self.zoom_in_btn, self.zoom_out_btn, self.zoom_fit_btn):
            zoom_row.addWidget(button)
        zoom_row.addStretch(1)
        zoom_row.addWidget(self.zoom_label)
        image_layout.addLayout(zoom_row)
        mode_grid = QGridLayout()
        self.mode_buttons = {}
        for index, (value, text, hint) in enumerate(
            (
                ("view", "查看", "只查看，不修改"),
                ("click", "取点击点", "在图片上点一下，设置点击位置"),
                ("roi", "框选检测重点", "拖框选择用于识别状态的区域"),
                ("target", "框选点击对象", "拖框选择要识别并点击的对象"),
                (
                    "relative",
                    "框选相对点击",
                    "拖框选择相对图片，运行时会以识别到的图片中心为原点加上偏移量点击",
                ),
            )
        ):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setToolTip(hint)
            if value == "view":
                button.clicked.connect(
                    lambda _checked: self._set_mode(
                        "relative_point" if self.current_relative else "view"
                    )
                )
            else:
                button.clicked.connect(lambda _checked, v=value: self._set_mode(v))
            if value == "relative":
                mode_grid.addWidget(button, index // 2, 0, 1, 2)
            else:
                mode_grid.addWidget(button, index // 2, index % 2)
            self.mode_buttons[value] = button
        image_layout.addLayout(mode_grid)
        self.click_info = QLabel("点击点：未设置")
        self.click_info.setStyleSheet("color:#ff7b72")
        self.roi_info = QLabel("检测重点：未设置")
        self.roi_info.setStyleSheet(f"color:{ACCENT}")
        self.target_info = QLabel("点击对象：未设置")
        self.target_info.setStyleSheet(f"color:{ORANGE}")
        self.relative_info = QLabel("相对点击：未设置")
        self.relative_info.setStyleSheet(f"color:{GREEN}")
        image_layout.addWidget(self.click_info)
        image_layout.addWidget(self.roi_info)
        image_layout.addWidget(self.target_info)
        image_layout.addWidget(self.relative_info)
        self.click_priority_info = QLabel("点击方式：屏幕中心")
        self.click_priority_info.setObjectName("subtitle")
        image_layout.addWidget(self.click_priority_info)
        self.target_strict_check = QCheckBox("仅识别到才点击")
        self.target_strict_check.setToolTip(
            "勾选后：整屏没识别到点击对象就不点击；不勾选：未识别到时退回到框选位置的相对坐标点击。"
        )
        self.target_threshold_row = QGroupBox("点击对象设置")
        target_threshold_layout = QFormLayout(self.target_threshold_row)
        self.target_threshold = QDoubleSpinBox()
        self.target_threshold.setRange(0.3, 0.999)
        self.target_threshold.setSingleStep(0.01)
        self.target_threshold.setDecimals(3)
        self.target_threshold.setValue(0.55)
        self.target_threshold.setToolTip(
            "框选点击对象在整屏定位时所需的最低相似度；只有设置了点击对象时才显示。"
        )
        target_threshold_layout.addRow("最低相似度", self.target_threshold)
        self.target_min_mr = QDoubleSpinBox()
        self.target_min_mr.setRange(0.0, 1.0)
        self.target_min_mr.setSingleStep(0.01)
        self.target_min_mr.setDecimals(3)
        self.target_min_mr.setValue(0.4)
        self.target_min_mr.setToolTip(
            "点击对象定位后，目标区域 MR 低于该值就认为没识别到，"
            "回退到相对位置（或按「仅识别到才点击」跳过）。"
        )
        target_threshold_layout.addRow("识别最低 MR", self.target_min_mr)
        target_threshold_layout.addRow(self.target_strict_check)
        self.target_threshold_row.setVisible(False)
        image_layout.addWidget(self.target_threshold_row)
        image_buttons = QHBoxLayout()
        self.replace_image_btn = QPushButton("更换图片")
        self.capture_image_btn = QPushButton("截取手机画面")
        self.add_to_asset_btn = QPushButton("加入素材库")
        for widget in (
            self.replace_image_btn,
            self.capture_image_btn,
            self.add_to_asset_btn,
        ):
            image_buttons.addWidget(widget)
        image_layout.addLayout(image_buttons)

        clear_buttons = QGridLayout()
        self.clear_click_btn = QPushButton("清除点击点")
        self.clear_roi_btn = QPushButton("清除检测重点")
        self.clear_target_btn = QPushButton("清除点击对象")
        self.clear_relative_main_btn = QPushButton("清除相对点击")
        clear_buttons.addWidget(self.clear_click_btn, 0, 0)
        clear_buttons.addWidget(self.clear_roi_btn, 0, 1)
        clear_buttons.addWidget(self.clear_target_btn, 1, 0)
        clear_buttons.addWidget(self.clear_relative_main_btn, 1, 1)
        image_layout.addLayout(clear_buttons)

        self.no_click_check = QCheckBox("不模拟点击")
        self.simulate_btn = QPushButton("模拟点击")
        self.simulate_btn.setObjectName("primary")
        self.target_simulate_btn = QPushButton("模拟人手点击")
        self.target_simulate_btn.setToolTip(
            "在手机上识别「点击对象」并点击其中心，用于测试框选是否正确"
        )
        self.target_simulate_btn.setVisible(False)
        action_row = QHBoxLayout()
        action_row.addWidget(self.target_simulate_btn)
        action_row.addWidget(self.no_click_check)
        action_row.addStretch(1)
        action_row.addWidget(self.simulate_btn)
        image_layout.addLayout(action_row)
        layout.addWidget(image_group)

        self.relative_group = QGroupBox("相对偏移点击（以相对图片中心为原点）")
        relative_layout = QVBoxLayout(self.relative_group)
        relative_hint = QLabel("运行时先整屏识别相对图片，再在其中心加上下方偏移量点击。")
        relative_hint.setObjectName("subtitle")
        relative_hint.setWordWrap(True)
        relative_layout.addWidget(relative_hint)
        offset_form = QFormLayout()
        self.relative_offset_x = QDoubleSpinBox()
        self.relative_offset_x.setRange(-5000, 5000)
        self.relative_offset_x.setDecimals(1)
        self.relative_offset_x.setSingleStep(1)
        self.relative_offset_x.setSuffix(" px")
        self.relative_offset_x.setToolTip("正数向右，负数向左；单位是手机截图像素")
        self.relative_offset_y = QDoubleSpinBox()
        self.relative_offset_y.setRange(-5000, 5000)
        self.relative_offset_y.setDecimals(1)
        self.relative_offset_y.setSingleStep(1)
        self.relative_offset_y.setSuffix(" px")
        self.relative_offset_y.setToolTip("正数向下，负数向上；单位是手机截图像素")
        offset_form.addRow("偏移 X", self.relative_offset_x)
        offset_form.addRow("偏移 Y", self.relative_offset_y)
        relative_layout.addLayout(offset_form)
        self.relative_point_info = QLabel("相对点击点：未设置")
        self.relative_point_info.setStyleSheet(f"color:{GREEN}")
        self.relative_point_info.setWordWrap(True)
        relative_layout.addWidget(self.relative_point_info)
        relative_buttons = QHBoxLayout()
        self.relative_point_btn = QPushButton("取相对点击点")
        self.clear_relative_btn = QPushButton("清除相对点击")
        self.relative_simulate_btn = QPushButton("模拟偏移点击")
        self.relative_simulate_btn.setToolTip(
            "在手机上识别相对图片，并按当前偏移量点击，用于测试"
        )
        relative_buttons.addWidget(self.relative_point_btn)
        relative_buttons.addWidget(self.relative_simulate_btn)
        relative_buttons.addWidget(self.clear_relative_btn)
        relative_buttons.addStretch(1)
        relative_layout.addLayout(relative_buttons)
        self.relative_group.setVisible(False)
        layout.addWidget(self.relative_group)

        form_group = QGroupBox("节点")
        form = QFormLayout(form_group)
        self.name_edit = QLineEdit()
        self.kind_combo = QComboBox()
        for label in KIND_LABELS.values():
            self.kind_combo.addItem(label)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.3, 0.999)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(3)
        self.roi_weight = QDoubleSpinBox()
        self.roi_weight.setRange(0.0, 1.0)
        self.roi_weight.setSingleStep(0.01)
        self.roi_weight.setDecimals(2)
        self.entry_check = QCheckBox("设为流程入口")
        self.end_check = QCheckBox("结束流程（到此步骤结束）")
        form.addRow("名称", self.name_edit)
        form.addRow("类型", self.kind_combo)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.threshold, 1)
        self.similarity_btn = QPushButton("相似度编辑…")
        self.similarity_btn.setToolTip("打开相似度编辑界面；也可以右键匹配阈值")
        threshold_row.addWidget(self.similarity_btn)
        form.addRow("匹配阈值", threshold_row)
        form.addRow("检测重点权重", self.roi_weight)
        form.addRow("", self.entry_check)
        form.addRow("", self.end_check)
        layout.addWidget(form_group)

        self.if_group = QGroupBox("IF 分支（按顺序判断；全部不中走默认）")
        if_layout = QVBoxLayout(self.if_group)
        self.if_list = QListWidget()
        self.if_list.setMaximumHeight(120)
        if_layout.addWidget(self.if_list)
        if_buttons = QHBoxLayout()
        self.if_add_btn = QPushButton("添加分支")
        self.if_remove_btn = QPushButton("删除分支")
        self.if_up_btn = QPushButton("上移")
        self.if_down_btn = QPushButton("下移")
        for widget in (
            self.if_add_btn,
            self.if_remove_btn,
            self.if_up_btn,
            self.if_down_btn,
        ):
            if_buttons.addWidget(widget)
        if_buttons.addStretch(1)
        if_layout.addLayout(if_buttons)
        default_row = QHBoxLayout()
        default_row.addWidget(QLabel("默认（全部不中）"))
        self.if_default_combo = QComboBox()
        default_row.addWidget(self.if_default_combo, 1)
        if_layout.addLayout(default_row)
        self.if_group.setVisible(False)
        layout.addWidget(self.if_group)

        action_group = QGroupBox("动作")
        action_layout = QVBoxLayout(action_group)
        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("类型"))
        self.action_combo = QComboBox()
        for label in ACTION_LABELS.values():
            self.action_combo.addItem(label)
        action_row.addWidget(self.action_combo, 1)
        action_layout.addLayout(action_row)
        self.action_box = QWidget()
        self.action_form = QFormLayout(self.action_box)
        action_layout.addWidget(self.action_box)
        wait_row = QHBoxLayout()
        wait_row.addWidget(QLabel("动作前等待"))
        self.pre_wait = QLineEdit("0")
        self.pre_wait.setMaximumWidth(70)
        wait_row.addWidget(self.pre_wait)
        wait_row.addWidget(QLabel("动作后等待"))
        self.post_wait = QLineEdit("0.6")
        self.post_wait.setMaximumWidth(70)
        wait_row.addWidget(self.post_wait)
        wait_row.addStretch(1)
        action_layout.addLayout(wait_row)
        layout.addWidget(action_group)

        notes_group = QGroupBox("备注")
        notes_layout = QVBoxLayout(notes_group)
        self.notes = QLineEdit()
        notes_layout.addWidget(self.notes)
        layout.addWidget(notes_group)

        edges_group = QGroupBox("下一步 / 偏好（可多个）")
        edges_layout = QVBoxLayout(edges_group)
        self.edges_list = QListWidget()
        self.edges_list.setMaximumHeight(130)
        edges_layout.addWidget(self.edges_list)
        edge_buttons = QHBoxLayout()
        self.add_edge_btn = QPushButton("添加下一步")
        self.edit_edge_btn = QPushButton("编辑连线")
        self.delete_edge_btn = QPushButton("删除连线")
        self.equalize_edge_btn = QPushButton("均分概率")
        self.equalize_edge_btn.setToolTip("把当前节点的所有下一步连线权重平均分配")
        for button in (
            self.add_edge_btn,
            self.edit_edge_btn,
            self.delete_edge_btn,
            self.equalize_edge_btn,
        ):
            edge_buttons.addWidget(button)
        edge_buttons.addStretch(1)
        edges_layout.addLayout(edge_buttons)
        layout.addWidget(edges_group)

        buttons = QHBoxLayout()
        self.save_btn = QPushButton("保存节点")
        self.save_btn.setObjectName("primary")
        self.delete_btn = QPushButton("删除节点")
        self.delete_btn.setObjectName("danger")
        buttons.addStretch(1)
        buttons.addWidget(self.delete_btn)
        buttons.addWidget(self.save_btn)
        layout.addLayout(buttons)
        layout.addStretch(1)

        self.image_editor.zoom_changed.connect(
            lambda zoom: self.zoom_label.setText(f"{int(round(zoom * 100))}%")
        )
        self.zoom_in_btn.clicked.connect(self.image_editor.zoom_in)
        self.zoom_out_btn.clicked.connect(self.image_editor.zoom_out)
        self.zoom_fit_btn.clicked.connect(self.image_editor.fit_image)
        self.image_editor.point_selected.connect(self._on_click)
        self.image_editor.roi_selected.connect(self._on_roi)
        self.image_editor.target_selected.connect(self._on_target)
        self.image_editor.relative_selected.connect(self._on_relative)
        self.replace_image_btn.clicked.connect(self._replace_image_menu)
        self.capture_image_btn.clicked.connect(self._capture_to_node)
        self.add_to_asset_btn.clicked.connect(self._add_image_to_library)
        self.clear_click_btn.clicked.connect(self._clear_click)
        self.clear_roi_btn.clicked.connect(self._clear_roi)
        self.clear_target_btn.clicked.connect(self._clear_target)
        self.relative_point_btn.clicked.connect(self._enter_relative_point_mode)
        self.clear_relative_btn.clicked.connect(self._clear_relative)
        self.clear_relative_main_btn.clicked.connect(self._clear_relative)
        self.simulate_btn.clicked.connect(self._simulate_click)
        self.target_simulate_btn.clicked.connect(self._simulate_target_click)
        self.relative_simulate_btn.clicked.connect(self._simulate_relative_click)
        self.no_click_check.toggled.connect(self._on_no_click)
        self.target_strict_check.toggled.connect(self._mark_dirty)
        self.target_threshold.valueChanged.connect(self._mark_dirty)
        self.target_min_mr.valueChanged.connect(self._mark_dirty)
        self.relative_offset_x.valueChanged.connect(self._on_relative_offset_changed)
        self.relative_offset_y.valueChanged.connect(self._on_relative_offset_changed)
        self.name_edit.textChanged.connect(self._mark_dirty)
        self.kind_combo.currentIndexChanged.connect(self._mark_dirty)
        self.kind_combo.currentIndexChanged.connect(self._update_if_group)
        self.kind_combo.currentIndexChanged.connect(self._update_kind_constraints)
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        self.if_add_btn.clicked.connect(self._if_add_branch)
        self.if_remove_btn.clicked.connect(self._if_remove_branch)
        self.if_up_btn.clicked.connect(lambda: self._if_move_branch(-1))
        self.if_down_btn.clicked.connect(lambda: self._if_move_branch(1))
        self.if_default_combo.currentIndexChanged.connect(self._mark_dirty)
        self.threshold.valueChanged.connect(self._mark_dirty)
        self.threshold.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.threshold.customContextMenuRequested.connect(self._threshold_menu)
        self.roi_weight.valueChanged.connect(self._mark_dirty)
        self.similarity_btn.clicked.connect(self._open_similarity_editor)
        self.entry_check.toggled.connect(self._mark_dirty)
        self.end_check.toggled.connect(self._mark_dirty)
        self.action_combo.currentIndexChanged.connect(self._on_action_type_changed)
        self.pre_wait.textChanged.connect(self._mark_dirty)
        self.post_wait.textChanged.connect(self._mark_dirty)
        self.notes.textChanged.connect(self._mark_dirty)
        self.auto_save_check.toggled.connect(self._on_auto_save_toggled)
        self.save_btn.clicked.connect(lambda: self.save_node(silent=False))
        self.delete_btn.clicked.connect(self._delete_node)
        self.add_edge_btn.clicked.connect(self._add_edge)
        self.edit_edge_btn.clicked.connect(self._edit_edge)
        self.delete_edge_btn.clicked.connect(self._delete_edge)
        self.equalize_edge_btn.clicked.connect(self._equalize_edge_weights)
        self.edges_list.itemDoubleClicked.connect(lambda _item: self._edit_edge())
        self.clear_node()

    def _enter_relative_point_mode(self) -> None:
        if self.current_relative is None:
            self.status_message.emit("请先「框选相对点击」设置相对图片")
            return
        self._set_mode("relative_point")
        self.status_message.emit("相对点击点模式：请在状态图片上点击要偏移的位置")

    def _set_mode(self, mode: str) -> None:
        if mode == "relative_point" and self.current_relative is None:
            self.status_message.emit("请先「框选相对点击」设置相对图片")
            return
        self.image_editor.set_mode(mode)
        for value, button in self.mode_buttons.items():
            button.setChecked(
                value == mode or (value == "view" and mode in ("view", "relative_point"))
            )
        if mode in ("target", "relative") and not vision.cv2_available():
            what = "点击对象" if mode == "target" else "相对点击图片"
            self.status_message.emit(f"未安装 OpenCV：{what}运行时会回退相对位置")
            QMessageBox.warning(
                self,
                "缺少 OpenCV",
                f"识别{what}需要 OpenCV。\n请安装：pip install -r requirements.txt",
            )

    def clear_node(self) -> None:
        self._auto_save_timer.stop()
        self._loading = True
        self.node = None
        self.image_editor.set_image(None)
        self.image_editor.set_markers(None, None, None, None, None)
        self.name_edit.clear()
        self.threshold.setValue(0.88)
        self.roi_weight.setValue(0.72)
        self.roi_weight.setEnabled(True)
        self.target_threshold.setValue(0.55)
        self.target_min_mr.setValue(0.4)
        self.relative_offset_x.setValue(0.0)
        self.relative_offset_y.setValue(0.0)
        self.current_relative_point = None
        self.entry_check.setChecked(False)
        self.end_check.setChecked(False)
        self.notes.clear()
        self.no_click_check.setChecked(False)
        self.target_strict_check.setChecked(False)
        self.target_strict_check.setEnabled(False)
        self.target_simulate_btn.setVisible(False)
        self.click_info.setText("点击点：未设置")
        self.roi_info.setText("检测重点：未设置")
        self.target_info.setText("点击对象：未设置")
        self.relative_info.setText("相对点击：未设置")
        self.relative_point_info.setText("相对点击点：未设置")
        self.target_threshold_row.setVisible(False)
        self.relative_group.setVisible(False)
        self.if_list.clear()
        self.if_default_combo.blockSignals(True)
        self.if_default_combo.clear()
        self.if_default_combo.addItem("（无）", None)
        self.if_default_combo.blockSignals(False)
        self.if_group.setVisible(False)
        self.edges_list.clear()
        self._loading = False
        self._dirty = False
        self.current_click = self.current_roi = self.current_target = None
        self.current_relative = None

    def load_node(self, node_id: int, force: bool = False) -> None:
        if not force and node_id == (self.node.id if self.node else None):
            return
        self.flush()
        self._auto_save_timer.stop()
        node = self.db.get_node(node_id)
        if node is None:
            self.clear_node()
            return
        self._loading = True
        self.node = node
        self.flow_id = node.flow_id
        self.name_edit.setText(node.name)
        self.kind_combo.setCurrentText(KIND_LABELS.get(node.kind, KIND_LABELS["image"]))
        self._update_if_group()
        self._load_if_branches(node)
        self.threshold.setValue(node.threshold)
        self.roi_weight.setValue(node.roi_weight)
        self._update_kind_constraints()
        self.entry_check.setChecked(node.is_entry)
        self.end_check.setChecked(bool(node.is_end or node.kind == "end"))
        self.notes.setText(node.notes)
        self.current_click = (
            None
            if node.action_type == "none"
            else ((node.click_x, node.click_y) if node.has_click_point else None)
        )
        self.current_roi = node.roi
        self.current_target = node.target_region
        self.current_relative = (
            node.relative_roi if self.current_target is None else None
        )
        offset_x, offset_y = node.relative_offset
        self.target_threshold.setValue(node.target_threshold)
        try:
            self.target_min_mr.setValue(
                float(node.action.get("target_min_mr", 0.4) or 0.4)
            )
        except (TypeError, ValueError):
            self.target_min_mr.setValue(0.4)
        self.relative_offset_x.setValue(offset_x)
        self.relative_offset_y.setValue(offset_y)
        self._last_action_label = (
            ACTION_LABELS.get(node.action_type, ACTION_LABELS["tap"])
            if node.action_type != "none"
            else ACTION_LABELS["tap"]
        )
        self.no_click_check.setChecked(node.action_type == "none")
        self.target_strict_check.setChecked(
            bool(node.action.get("target_strict", False))
        )
        self.simulate_btn.setEnabled(node.action_type != "none")
        self.action_combo.setCurrentText(
            ACTION_LABELS.get(node.action_type, ACTION_LABELS["tap"])
        )
        self.pre_wait.setText(str(node.action.get("pre_wait", 0)))
        self.post_wait.setText(str(node.action.get("post_wait", node.action.get("wait", 0.6))))
        self._rebuild_action_fields(node)
        pixmap = QPixmap()
        if node.image_path and Path(node.image_path).exists():
            pixmap = self._pixmap_for_path(node.image_path)
            if not pixmap.isNull() and (
                node.img_w != pixmap.width() or node.img_h != pixmap.height()
            ):
                self.db.update_node(
                    node.id, img_w=pixmap.width(), img_h=pixmap.height()
                )
                refreshed = self.db.get_node(node.id)
                if refreshed is not None:
                    node = refreshed
                    self.node = node
        self._update_marker_labels()
        self.image_editor.set_image(pixmap if not pixmap.isNull() else None)
        self.image_editor.set_markers(
            self.current_click,
            self.current_roi,
            self.current_target,
            self.current_relative,
            self.current_relative_point,
        )
        self._update_clear_buttons()
        self._update_click_priority()
        self.image_editor.fit_image()
        if self.image_editor.mode == "relative_point" and self.current_relative is None:
            self._set_mode("view")
        self._reload_edges()
        self._loading = False
        self._dirty = False
        self._image_undo_pushed = False

    def _pixmap_for_path(self, path: str) -> QPixmap:
        try:
            frame = vision.load_rgb(path)
            height, width = frame.shape[:2]
            image = QImage(
                frame.data,
                width,
                height,
                3 * width,
                QImage.Format.Format_RGB888,
            ).copy()
            return QPixmap.fromImage(image)
        except Exception:
            return QPixmap(str(path))

    def _image_dimensions(self) -> tuple[float, float]:
        if self.node is not None and self.node.img_w and self.node.img_h:
            return (float(self.node.img_w), float(self.node.img_h))
        width, height = self.image_editor.image_size()
        if width and height:
            return (float(width), float(height))
        return (0.0, 0.0)

    def _update_marker_labels(self) -> None:
        click = self.current_click
        roi = self.current_roi
        target = self.current_target
        relative = self.current_relative
        self.click_info.setText(
            "点击点：未设置"
            if not click
            else f"点击点：x={click[0]:.4f}  y={click[1]:.4f}"
        )
        self.roi_info.setText(
            "检测重点：未设置"
            if not roi
            else f"检测重点：x={roi[0]:.3f} y={roi[1]:.3f} w={roi[2]:.3f} h={roi[3]:.3f}"
        )
        self.target_info.setText(
            "点击对象：未设置"
            if not target
            else f"点击对象：x={target[0]:.3f} y={target[1]:.3f} "
            f"w={target[2]:.3f} h={target[3]:.3f}"
        )
        self.relative_info.setText(
            "相对点击：未设置"
            if not relative
            else f"相对点击图片：x={relative[0]:.3f} y={relative[1]:.3f} "
            f"w={relative[2]:.3f} h={relative[3]:.3f}"
        )
        if relative and self.node is not None:
            center_x = float(relative[0]) + float(relative[2]) / 2.0
            center_y = float(relative[1]) + float(relative[3]) / 2.0
            image_w, image_h = self._image_dimensions()
            offset_x = float(self.relative_offset_x.value())
            offset_y = float(self.relative_offset_y.value())
            self.current_relative_point = (
                center_x + (offset_x / image_w if image_w else 0.0),
                center_y + (offset_y / image_h if image_h else 0.0),
            )
        else:
            self.current_relative_point = None
        view_button = self.mode_buttons.get("view")
        if view_button is not None:
            view_button.setText("取相对点击点" if relative else "查看")
        if relative and self.current_relative_point is not None and self.node is not None:
            image_w, image_h = self._image_dimensions()
            px = self.current_relative_point[0] * image_w
            py = self.current_relative_point[1] * image_h
            self.relative_point_info.setText(
                f"相对点击点：x={self.current_relative_point[0]:.4f} "
                f"y={self.current_relative_point[1]:.4f}  "
                f"像素=({px:.0f}, {py:.0f})  "
                f"偏移=({self.relative_offset_x.value():.0f}, "
                f"{self.relative_offset_y.value():.0f})"
            )
        else:
            self.relative_point_info.setText("相对点击点：未设置")
        self.target_strict_check.setEnabled(target is not None)
        self.target_simulate_btn.setVisible(target is not None)
        self.target_threshold_row.setVisible(target is not None)
        self.relative_group.setVisible(relative is not None)
        self.image_editor.set_markers(
            self.current_click,
            self.current_roi,
            self.current_target,
            self.current_relative,
            self.current_relative_point,
        )
        self._update_clear_buttons()
        self._update_click_priority()

    def _on_relative_offset_changed(self, *_args) -> None:
        if self._loading:
            return
        self._update_marker_labels()
        self._mark_dirty()

    def _update_kind_constraints(self, *_args) -> None:
        is_detect = self.kind_combo.currentText() == KIND_LABELS["detect"]
        is_ai = self.kind_combo.currentText() == KIND_LABELS["ai"]
        if is_detect or is_ai:
            self.roi_weight.setValue(1.0)
            self.roi_weight.setEnabled(False)
        else:
            self.roi_weight.setEnabled(True)

    def _on_kind_changed(self, *_args) -> None:
        """AI 节点自动切到「AI 决策」动作，改回去时自动切回点击。"""
        if self._loading or self.node is None:
            return
        is_ai = self.kind_combo.currentText() == KIND_LABELS["ai"]
        is_ai_action = self._action_key() == "ai"
        if is_ai and not is_ai_action:
            self.action_combo.setCurrentText(ACTION_LABELS["ai"])
        elif not is_ai and is_ai_action:
            self.action_combo.setCurrentText(ACTION_LABELS["tap"])

    def _threshold_menu(self, position) -> None:
        if self.node is None:
            return
        menu = QMenu(self)
        action = menu.addAction("相似度编辑…")
        chosen = menu.exec(self.threshold.mapToGlobal(position))
        if chosen is action:
            self._open_similarity_editor()

    def _open_similarity_editor(self) -> None:
        if self.node is None:
            QMessageBox.information(self, "提示", "请先选择一个节点")
            return
        self.flush()
        self.edit_similarity_requested.emit(self.node.id)

    def _update_clear_buttons(self) -> None:
        has_click = self.current_click is not None
        has_roi = self.current_roi is not None
        has_target = self.current_target is not None
        has_relative = self.current_relative is not None
        self.clear_click_btn.setEnabled(has_click)
        self.clear_roi_btn.setEnabled(has_roi)
        self.clear_target_btn.setEnabled(has_target)
        self.clear_relative_btn.setEnabled(has_relative)
        self.clear_relative_main_btn.setEnabled(has_relative)

    def _update_click_priority(self) -> None:
        if self.node is None:
            self.click_priority_info.setText("点击方式：屏幕中心")
            return
        if self.node.action_type == "none":
            text = "不模拟点击"
        elif self.node.kind == "ai" or self.node.action_type == "ai":
            text = "AI 决策（模型看屏决定）"
        elif isinstance((self.node.action or {}).get("ui_selector"), dict) and self.node.action["ui_selector"]:
            text = "控件树定位（优先）"
        elif self.node.kind == "detect":
            text = "侧重检测节点：检测重点匹配位置"
        elif self.current_target is not None:
            text = "点击对象（优先）"
        elif self.current_relative is not None:
            text = "相对偏移点击"
        elif self.current_click is not None:
            text = "固定点击点"
        else:
            text = "屏幕中心"
        self.click_priority_info.setText(f"点击方式：{text}")

    def _update_if_group(self, *_args) -> None:
        is_if = self.kind_combo.currentText() == KIND_LABELS["if"]
        self.if_group.setVisible(is_if)
        if is_if:
            self._refresh_if_targets()

    def _flow_nodes(self):
        if self.flow_id is None:
            return []
        return self.db.list_nodes(self.flow_id)

    def _refresh_if_targets(self) -> None:
        current_default = self.if_default_combo.currentData()
        self.if_default_combo.blockSignals(True)
        self.if_default_combo.clear()
        self.if_default_combo.addItem("（无）", None)
        for node in self._flow_nodes():
            if self.node is not None and node.id == self.node.id:
                continue
            self.if_default_combo.addItem(node.name, node.id)
        index = self.if_default_combo.findData(current_default)
        if index >= 0:
            self.if_default_combo.setCurrentIndex(index)
        self.if_default_combo.blockSignals(False)

    def _load_if_branches(self, node: Node) -> None:
        self.if_list.clear()
        action = node.action or {}
        for branch_id in action.get("if_branches") or []:
            try:
                bid = int(branch_id)
            except (TypeError, ValueError):
                continue
            target = self.db.get_node(bid)
            if target is None:
                continue
            item = QListWidgetItem(target.name)
            item.setData(Qt.ItemDataRole.UserRole, bid)
            self.if_list.addItem(item)
        self._refresh_if_targets()
        default_id = action.get("if_default")
        if default_id is not None:
            try:
                index = self.if_default_combo.findData(int(default_id))
            except (TypeError, ValueError):
                index = -1
            if index >= 0:
                self.if_default_combo.setCurrentIndex(index)

    def _if_add_branch(self) -> None:
        existing = {
            self.if_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.if_list.count())
        }
        candidates = [
            node
            for node in self._flow_nodes()
            if (self.node is None or node.id != self.node.id)
            and node.id not in existing
        ]
        if not candidates:
            QMessageBox.information(self, "提示", "没有可添加的分支节点")
            return
        names = [node.name for node in candidates]
        choice, ok = QInputDialog.getItem(
            self,
            "添加 IF 分支",
            "选择作为分支条件的节点（按列表顺序依次判断）：",
            names,
            0,
            False,
        )
        if not ok:
            return
        target = candidates[names.index(choice)]
        item = QListWidgetItem(target.name)
        item.setData(Qt.ItemDataRole.UserRole, target.id)
        self.if_list.addItem(item)
        self._mark_dirty()

    def _if_remove_branch(self) -> None:
        row = self.if_list.currentRow()
        if row < 0:
            return
        self.if_list.takeItem(row)
        self._mark_dirty()

    def _if_move_branch(self, delta: int) -> None:
        row = self.if_list.currentRow()
        target = row + delta
        if row < 0 or not (0 <= target < self.if_list.count()):
            return
        item = self.if_list.takeItem(row)
        self.if_list.insertItem(target, item)
        self.if_list.setCurrentRow(target)
        self._mark_dirty()

    def _mark_dirty(self, *_args) -> None:
        if self._loading:
            return
        self._dirty = True
        if self._auto_save and self.node is not None:
            self._auto_save_timer.start()

    def _auto_save_now(self) -> None:
        if self._loading or self.node is None:
            return
        self.save_node(silent=True)

    def _on_auto_save_toggled(self, checked: bool) -> None:
        self._auto_save = checked
        self._settings.setValue("auto_save", checked)
        if checked and self._dirty and self.node is not None:
            self._auto_save_timer.start(200)

    def _push_undo(self, label: str) -> None:
        if self.undo_callback and self.flow_id is not None:
            self.undo_callback(label)

    def flush(self) -> None:
        if self._dirty and self.node is not None:
            self.save_node(silent=True)

    def _on_click(self, x: float, y: float) -> None:
        if self.image_editor.mode == "relative_point" and self.current_relative is not None:
            relative = self.current_relative
            center_x = float(relative[0]) + float(relative[2]) / 2.0
            center_y = float(relative[1]) + float(relative[3]) / 2.0
            image_w, image_h = self._image_dimensions()
            offset_x = (x - center_x) * image_w
            offset_y = (y - center_y) * image_h
            self.relative_offset_x.setValue(offset_x)
            self.relative_offset_y.setValue(offset_y)
            self.current_relative_point = (x, y)
            self._update_marker_labels()
            self._save_image_state(silent=True)
            self.status_message.emit(
                f"已设置相对点击点 ({x:.4f}, {y:.4f})，"
                f"偏移量 ({offset_x:.0f}, {offset_y:.0f})"
            )
            return
        if not self._image_undo_pushed:
            self._push_undo("设置点击点")
            self._image_undo_pushed = True
        self.current_click = (x, y)
        self._update_marker_labels()
        self._save_image_state(silent=True)

    def _on_roi(self, rect) -> None:
        if not self._image_undo_pushed:
            self._push_undo("设置检测重点")
            self._image_undo_pushed = True
        self.current_roi = (rect.x(), rect.y(), rect.width(), rect.height())
        self._update_marker_labels()
        self._save_image_state(silent=True)

    def _on_target(self, rect) -> None:
        if not self._image_undo_pushed:
            self._push_undo("框选点击对象")
            self._image_undo_pushed = True
        self.current_target = (rect.x(), rect.y(), rect.width(), rect.height())
        self.current_relative = None
        self.relative_offset_x.setValue(0.0)
        self.relative_offset_y.setValue(0.0)
        self.image_editor.set_markers(
            self.current_click, self.current_roi, self.current_target, None
        )
        self._update_marker_labels()
        self._save_image_state(silent=True)
        if self.image_editor.mode == "relative_point":
            self._set_mode("view")
        self.status_message.emit(
            "已设置点击对象（已自动关闭偏移点击）：运行时会识别该区域并点击中心"
        )

    def _on_relative(self, rect) -> None:
        if not self._image_undo_pushed:
            self._push_undo("框选相对点击")
            self._image_undo_pushed = True
        self.current_relative = (rect.x(), rect.y(), rect.width(), rect.height())
        self.current_target = None
        self.relative_offset_x.setValue(0.0)
        self.relative_offset_y.setValue(0.0)
        self.image_editor.set_markers(
            self.current_click, self.current_roi, None, self.current_relative
        )
        self._update_marker_labels()
        self._save_image_state(silent=True)
        self.status_message.emit(
            "已设置相对点击图片（已自动关闭点击对象）：运行时会识别该图片并在中心加上偏移量点击"
        )

    def _clear_click(self) -> None:
        self._push_undo("清除点击点")
        self.current_click = None
        self.image_editor.set_markers(
            None, self.current_roi, self.current_target, self.current_relative
        )
        self._update_marker_labels()
        self._save_image_state(silent=True)
        self.status_message.emit("已清除点击点")

    def _clear_roi(self) -> None:
        self._push_undo("清除检测重点")
        self.current_roi = None
        self.image_editor.set_markers(
            self.current_click, None, self.current_target, self.current_relative
        )
        self._update_marker_labels()
        self._save_image_state(silent=True)

    def _clear_target(self) -> None:
        self._push_undo("清除点击对象")
        self.current_target = None
        self.image_editor.set_markers(
            self.current_click, self.current_roi, None, self.current_relative
        )
        self._update_marker_labels()
        self._save_image_state(silent=True)

    def _clear_relative(self) -> None:
        self._push_undo("清除相对点击")
        self.current_relative = None
        self.current_relative_point = None
        self.relative_offset_x.setValue(0.0)
        self.relative_offset_y.setValue(0.0)
        self.image_editor.set_markers(
            self.current_click, self.current_roi, self.current_target, None
        )
        self._update_marker_labels()
        if self.image_editor.mode == "relative_point":
            self._set_mode("view")
        self._save_image_state(silent=True)
        self.status_message.emit("已清除相对点击")

    def _save_image_state(self, silent: bool = False) -> None:
        self._dirty = True
        self.save_node(silent=silent)

    def _replace_image_menu(self) -> None:
        if self.node is None:
            return
        menu = QMenu(self)
        file_action = menu.addAction("从文件选择…")
        library_action = menu.addAction("从素材库选择…")
        chosen = menu.exec(
            self.replace_image_btn.mapToGlobal(self.replace_image_btn.rect().bottomLeft())
        )
        if chosen is file_action:
            self._replace_image_from_file()
        elif chosen is library_action:
            self._replace_image_from_library()

    def _replace_image_from_file(self) -> None:
        if self.node is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择状态图片", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        self._apply_image_path(path, "更换图片")

    def _replace_image_from_library(self) -> None:
        if self.node is None:
            return
        dialog = AssetPickerDialog(self, "从素材库选择状态图片")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if not dialog.selected_path:
            return
        self._apply_image_path(dialog.selected_path, "更换图片（素材库）")

    def _apply_image_path(self, path: str, undo_label: str = "更换图片") -> None:
        if self.node is None:
            return
        # Persist pending edits (e.g. the node type) so changing the image does
        # not silently discard them when the node is reloaded below.
        if self._dirty:
            self.save_node(silent=True)
        if self.node is None:
            return
        self._push_undo(undo_label)
        source = Path(path)
        target = config.images_dir() / (
            f"flow{self.flow_id}_{uuid4().hex[:10]}{source.suffix.lower() or '.png'}"
        )
        shutil.copy2(source, target)
        pixmap = self._pixmap_for_path(str(target))
        action = dict(self.node.action or {})
        for key in ("target_threshold", "rel_roi", "rel_offset"):
            action.pop(key, None)
        self.db.update_node(
            self.node.id,
            image_path=str(target),
            img_w=pixmap.width(),
            img_h=pixmap.height(),
            click_x=None,
            click_y=None,
            roi_x=None,
            roi_y=None,
            roi_w=None,
            roi_h=None,
            target_x=None,
            target_y=None,
            target_w=None,
            target_h=None,
            action=action,
        )
        try:
            vector = get_embedder().embed(str(target))
            self.db.set_node_embedding(
                self.node.id,
                embed_to_blob(vector),
                get_embedder().kind,
                int(vector.shape[0]),
            )
        except Exception as exc:
            self.status_message.emit(f"向量计算失败：{exc}")
        self.current_click = self.current_roi = self.current_target = None
        self.current_relative = None
        node_id = self.node.id
        self.node_saved.emit(node_id)
        self.node = None
        self.load_node(node_id, force=True)
        self.status_message.emit("图片已更换，请重新取点")

    def _capture_to_node(self) -> None:
        if self.node is None:
            QMessageBox.information(self, "提示", "请先选择一个节点")
            return
        try:
            device = AdbDevice(serial=self.serial)
            device.check()
            frame = device.screenshot()
        except Exception as exc:
            QMessageBox.critical(self, "截图失败", str(exc))
            return
        asset = add_frame_to_assets(frame, prefix=f"node{self.node.id}")
        self._apply_image_path(str(asset), "截取手机画面")
        self.assets_changed.emit()
        self.status_message.emit(f"已截取手机画面更新状态图片，并存入素材库：{asset.name}")

    def _add_image_to_library(self) -> None:
        if self.node is None or not self.node.image_path:
            QMessageBox.information(self, "提示", "当前节点没有状态图片")
            return
        source = Path(self.node.image_path)
        if not source.exists():
            QMessageBox.warning(self, "提示", f"找不到图片文件：\n{source}")
            return
        try:
            target = add_file_to_assets(source)
        except Exception as exc:
            QMessageBox.warning(self, "加入素材库失败", str(exc))
            return
        self.assets_changed.emit()
        self.status_message.emit(f"已把当前图片加入素材库：{target.name}")

    def _on_no_click(self, checked: bool) -> None:
        if self._loading:
            return
        self._syncing_no_click = True
        try:
            if checked:
                if self._action_key() != "none":
                    self._last_action_label = self.action_combo.currentText()
                self.action_combo.setCurrentText(ACTION_LABELS["none"])
            else:
                self.action_combo.setCurrentText(self._last_action_label or ACTION_LABELS["tap"])
        finally:
            self._syncing_no_click = False
        self.simulate_btn.setEnabled(not checked)
        self._push_undo("不模拟点击" if checked else "恢复模拟点击")
        if checked and self.current_click is not None:
            self.current_click = None
            self.image_editor.set_markers(
                None, self.current_roi, self.current_target, self.current_relative
            )
            self._update_marker_labels()
        self._save_image_state(silent=True)
        if checked:
            self.status_message.emit("已设为不模拟点击：该节点运行时不再点击")

    def _on_action_type_changed(self, *_args) -> None:
        if self._loading:
            return
        key = self._action_key()
        if (
            key == "burst"
            and self.node is not None
            and not self.node.has_image
            and self.kind_combo.currentText() != KIND_LABELS["nodetect"]
        ):
            self.kind_combo.setCurrentText(KIND_LABELS["nodetect"])
            self.status_message.emit("极速连滑需要「不检测步骤」，已自动切换")
        if not self._syncing_no_click:
            self.no_click_check.setChecked(key == "none")
            self.simulate_btn.setEnabled(key != "none")
        self._rebuild_action_fields(self.node)
        self._mark_dirty()

    def _action_key(self) -> str:
        return ACTION_VALUES.get(self.action_combo.currentText(), "tap")

    def _rebuild_action_fields(self, node: Node | None) -> None:
        while self.action_form.rowCount():
            self.action_form.removeRow(0)
        self._action_widgets = {}
        if node is None:
            return
        action = node.action or {}
        kind = self._action_key()
        if kind == "tap":
            self.action_form.addRow(QLabel("点击图片上的取点；未取点则点屏幕中心。"))
        elif kind == "swipe":
            for name, label, default in (
                ("x1", "起点 x", 0.5),
                ("y1", "起点 y", 0.8),
                ("x2", "终点 x", 0.5),
                ("y2", "终点 y", 0.2),
                ("duration", "时长 ms", 300),
                ("repeat", "重复次数(0=运行页默认)", 0),
            ):
                edit = QLineEdit(str(action.get(name, default)))
                edit.textChanged.connect(self._mark_dirty)
                self.action_form.addRow(label, edit)
                self._action_widgets[name] = edit
        elif kind == "burst":
            self.action_form.addRow(
                QLabel("Monkey 脚本注入，实测 100+ 次/秒；起终点相同＝连点。手机需解锁。")
            )
            for name, label, default in (
                ("count", "次数", 100),
                ("x1", "起点 x", 0.5),
                ("y1", "起点 y", 0.8),
                ("x2", "终点 x", 0.5),
                ("y2", "终点 y", 0.2),
                ("duration", "每次时长 ms", 50),
            ):
                edit = QLineEdit(str(action.get(name, default)))
                edit.textChanged.connect(self._mark_dirty)
                self.action_form.addRow(label, edit)
                self._action_widgets[name] = edit
        elif kind == "key":
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(["back", "home", "enter", "menu", "power", "volume_up", "volume_down"])
            combo.setCurrentText(str(action.get("key", "back")))
            combo.currentTextChanged.connect(self._mark_dirty)
            self.action_form.addRow("按键", combo)
            self._action_widgets["key"] = combo
        elif kind == "text":
            edit = QLineEdit(str(action.get("text", "")))
            edit.textChanged.connect(self._mark_dirty)
            self.action_form.addRow("文本", edit)
            self._action_widgets["text"] = edit
            clear = QCheckBox("先清空输入框")
            clear.setChecked(bool(action.get("clear", False)))
            clear.toggled.connect(self._mark_dirty)
            submit = QCheckBox("输入后回车")
            submit.setChecked(bool(action.get("submit", False)))
            submit.toggled.connect(self._mark_dirty)
            self.action_form.addRow("", clear)
            self.action_form.addRow("", submit)
            self._action_widgets["clear"] = clear
            self._action_widgets["submit"] = submit
        elif kind == "launch":
            edit = QLineEdit(str(action.get("package", "")))
            edit.textChanged.connect(self._mark_dirty)
            fresh = QCheckBox("先强制停止再启动")
            fresh.setChecked(bool(action.get("fresh", True)))
            fresh.toggled.connect(self._mark_dirty)
            self.action_form.addRow("包名", edit)
            self.action_form.addRow("", fresh)
            self._action_widgets["package"] = edit
            self._action_widgets["fresh"] = fresh
        elif kind == "wait":
            edit = QLineEdit(str(action.get("seconds", 1)))
            edit.textChanged.connect(self._mark_dirty)
            self.action_form.addRow("等待秒数", edit)
            self._action_widgets["seconds"] = edit
        if node.kind == "manual":
            monitor = QCheckBox("执行后检测完成情况")
            monitor.setChecked(bool(action.get("monitor", True)))
            monitor.toggled.connect(self._mark_dirty)
            self.action_form.addRow("", monitor)
            self._action_widgets["monitor"] = monitor
        elif kind == "ai":
            goal = QPlainTextEdit(str(action.get("goal", "")))
            goal.setPlaceholderText("运行到这里时，本地模型会看着当前画面一步步完成这个目标")
            goal.setFixedHeight(90)
            goal.textChanged.connect(self._mark_dirty)
            self.action_form.addRow("任务目标", goal)
            self._action_widgets["goal"] = goal
            detail = QPlainTextEdit(str(action.get("detail", "")))
            detail.setPlaceholderText("可选：额外要求，例如「不要点广告」「优先用搜索框」")
            detail.setFixedHeight(56)
            detail.textChanged.connect(self._mark_dirty)
            self.action_form.addRow("补充说明", detail)
            self._action_widgets["detail"] = detail
            limit = QLineEdit(str(action.get("max_decisions", 0) or 0))
            limit.setToolTip("0 表示使用「AI 设置」里的单节点决策上限")
            limit.textChanged.connect(self._mark_dirty)
            self.action_form.addRow("决策上限(0=全局)", limit)
            self._action_widgets["max_decisions"] = limit
            self.action_form.addRow(
                QLabel("AI 节点不需要截图，运行时由本地 Ollama 看屏决策。")
            )
        else:
            self.action_form.addRow(QLabel("只匹配状态，不执行动作。"))

    def _collect_action(self) -> dict:
        action = {}
        for name, widget in self._action_widgets.items():
            if isinstance(widget, QCheckBox):
                action[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                action[name] = widget.currentText()
            else:
                if name in ("goal", "detail"):
                    text = (
                        widget.toPlainText()
                        if hasattr(widget, "toPlainText")
                        else widget.text()
                    )
                    action[name] = text
                    continue
                value = widget.text()
                if name in {"text", "key", "package"}:
                    action[name] = value
                else:
                    try:
                        action[name] = (
                            int(float(value))
                            if name in ("duration", "max_decisions", "repeat", "count")
                            else float(value)
                        )
                    except ValueError:
                        action[name] = 1 if name == "duration" else 0
        if self.node is not None and self.node.kind == "manual":
            action["manual"] = True
            action["kind"] = self.node.action_type or "tap"
        for name, widget in (("pre_wait", self.pre_wait), ("post_wait", self.post_wait)):
            try:
                action[name] = float(widget.text())
            except ValueError:
                action[name] = 0.0
        return action

    def save_node(self, silent: bool = False) -> None:
        if self._loading or self.node is None:
            return
        self._auto_save_timer.stop()
        if not silent:
            self._push_undo("保存节点")
        kind = KIND_VALUES.get(self.kind_combo.currentText(), self.node.kind)
        target = self.current_target
        roi = self.current_roi
        if kind == "detect" and roi is None:
            self.status_message.emit("提示：侧重检测节点需要先「框选检测重点」才能检测")
        roi_weight = 1.0 if kind == "detect" else self.roi_weight.value()
        action = self._collect_action()
        if kind == "manual" and action.get("kind") == "tap" and self.current_click:
            action["x"] = float(self.current_click[0])
            action["y"] = float(self.current_click[1])
        action["target_strict"] = bool(target is not None and self.target_strict_check.isChecked())
        action["target_threshold"] = float(self.target_threshold.value())
        action["target_min_mr"] = float(self.target_min_mr.value())
        action["rel_min_mr"] = float(self.target_min_mr.value())
        existing_action = self.node.action or {}
        if isinstance(existing_action.get("ui_selector"), dict) and existing_action["ui_selector"]:
            action["ui_selector"] = existing_action["ui_selector"]
        if isinstance(existing_action.get("similarity_model"), dict):
            action["similarity_model"] = existing_action["similarity_model"]
        relative = self.current_relative
        if relative is not None:
            action["rel_roi"] = [float(value) for value in relative]
            action["rel_offset"] = [
                float(self.relative_offset_x.value()),
                float(self.relative_offset_y.value()),
            ]
        else:
            action.pop("rel_roi", None)
            action.pop("rel_offset", None)
        if kind == "if":
            branches = []
            for row in range(self.if_list.count()):
                value = self.if_list.item(row).data(Qt.ItemDataRole.UserRole)
                if value is not None:
                    branches.append(int(value))
            action["if_branches"] = branches
            action["if_default"] = self.if_default_combo.currentData()
        self.db.update_node(
            self.node.id,
            name=self.name_edit.text().strip() or self.node.name,
            kind=kind,
            threshold=self.threshold.value(),
            roi_weight=roi_weight,
            click_x=self.current_click[0] if self.current_click else None,
            click_y=self.current_click[1] if self.current_click else None,
            roi_x=roi[0] if roi else None,
            roi_y=roi[1] if roi else None,
            roi_w=roi[2] if roi else None,
            roi_h=roi[3] if roi else None,
            target_x=target[0] if target else None,
            target_y=target[1] if target else None,
            target_w=target[2] if target else None,
            target_h=target[3] if target else None,
            action_type=self._action_key(),
            action=action,
            is_entry=self.entry_check.isChecked(),
            is_end=self.end_check.isChecked(),
            notes=self.notes.text(),
        )
        self.node = self.db.get_node(self.node.id)
        self._dirty = False
        self.node_saved.emit(self.node.id)
        if not silent:
            self.status_message.emit(f"节点已保存：{self.node.name}")

    def _delete_node(self) -> None:
        if self.node is None:
            return
        answer = QMessageBox.question(
            self, "删除节点", f"删除节点「{self.node.name}」以及相关连线？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._push_undo("删除节点")
        self.delete_requested.emit(self.node.id)

    def _reload_edges(self) -> None:
        self.edges_list.clear()
        if self.node is None:
            return
        for edge in self.db.list_edges(self.node.flow_id):
            if edge.src_id != self.node.id:
                continue
            target = self.db.get_node(edge.dst_id)
            item = QListWidgetItem(
                f"→ {target.name if target else edge.dst_id}   ×{edge.weight:g}"
                + (f"   {edge.label}" if edge.label else "")
            )
            item.setData(Qt.ItemDataRole.UserRole, edge.id)
            self.edges_list.addItem(item)

    def _selected_edge(self):
        item = self.edges_list.currentItem()
        if item is None:
            return None
        return self.db.get_edge(int(item.data(Qt.ItemDataRole.UserRole)))

    def _nodes_of_flow(self) -> list[Node]:
        if self.node is None:
            return []
        return self.db.list_nodes(self.node.flow_id)

    def _add_edge(self) -> None:
        if self.node is None:
            return
        dialog = ConnectDialog(self._nodes_of_flow(), src=self.node.id, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._push_undo("添加连线")
        values = dialog.values()
        self.db.add_edge(
            self.node.flow_id, values["src"], values["dst"], values["weight"], values["label"]
        )
        self._reload_edges()
        self.edges_changed.emit()

    def _edit_edge(self) -> None:
        edge = self._selected_edge()
        if edge is None or self.node is None:
            return
        dialog = ConnectDialog(
            self._nodes_of_flow(), src=edge.src_id, dst=edge.dst_id, parent=self
        )
        dialog.weight.setValue(edge.weight)
        dialog.label.setText(edge.label)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._push_undo("编辑连线")
        values = dialog.values()
        self.db.add_edge(
            self.node.flow_id, values["src"], values["dst"], values["weight"], values["label"]
        )
        self._reload_edges()
        self.edges_changed.emit()

    def _equalize_edge_weights(self) -> None:
        if self.node is None:
            return
        edges = [
            edge
            for edge in self.db.list_edges(self.node.flow_id)
            if edge.src_id == self.node.id
        ]
        if not edges:
            QMessageBox.information(self, "提示", "当前节点没有下一步连线")
            return
        self._push_undo("均分概率")
        weight = round(1.0 / len(edges), 4)
        for edge in edges:
            self.db.update_edge(edge.id, weight=weight)
        self._reload_edges()
        self.edges_changed.emit()
        self.status_message.emit(f"已均分 {len(edges)} 条连线的概率（各 {weight:g}）")

    def _delete_edge(self) -> None:
        edge = self._selected_edge()
        if edge is None:
            return
        self._push_undo("删除连线")
        self.db.delete_edge(edge.id)
        self._reload_edges()
        self.edges_changed.emit()

    def _simulate_click(self) -> None:
        if self.no_click_check.isChecked():
            QMessageBox.information(self, "提示", "当前节点已设置为「不模拟点击」")
            return
        if self.current_click is None:
            QMessageBox.information(self, "提示", "请先在图片上取点击点")
            return
        try:
            device = AdbDevice(serial=self.serial)
            width, height = device.screen_size()
            device.tap(self.current_click[0] * width, self.current_click[1] * height)
        except Exception as exc:
            QMessageBox.critical(self, "模拟点击失败", str(exc))
            return
        self.status_message.emit(
            f"已发送点击：({int(self.current_click[0] * width)}, "
            f"{int(self.current_click[1] * height)})"
        )

    def _simulate_target_click(self) -> None:
        if self.node is None or self.current_target is None:
            QMessageBox.information(self, "提示", "请先框选点击对象")
            return
        try:
            device = AdbDevice(serial=self.serial)
            shot = device.screenshot()
        except Exception as exc:
            QMessageBox.critical(self, "模拟人手点击失败", str(exc))
            return
        node_image = None
        if self.node.image_path and Path(self.node.image_path).exists():
            try:
                node_image = vision.load_rgb(self.node.image_path)
            except Exception:
                node_image = None
        point, _region, found = vision.locate_target(
            shot,
            node_image,
            self.current_target,
            min_score=float(self.target_threshold.value()),
        )
        if point is None:
            QMessageBox.information(self, "提示", "没有可点击的位置")
            return
        if not found and self.target_strict_check.isChecked():
            self.status_message.emit("未识别到点击对象，已跳过（仅识别到才点击）")
            return
        try:
            device.tap(point[0], point[1])
        except Exception as exc:
            QMessageBox.critical(self, "模拟人手点击失败", str(exc))
            return
        if found:
            self.status_message.emit(f"模拟人手点击：已识别并点击 {point}")
        else:
            self.status_message.emit(
                f"未识别到点击对象，已按框选相对位置点击 {point}"
            )

    def _simulate_relative_click(self) -> None:
        if self.node is None or self.current_relative is None:
            QMessageBox.information(self, "提示", "请先框选相对点击图片")
            return
        try:
            device = AdbDevice(serial=self.serial)
            shot = device.screenshot()
        except Exception as exc:
            QMessageBox.critical(self, "模拟偏移点击失败", str(exc))
            return
        node_image = None
        if self.node.image_path and Path(self.node.image_path).exists():
            try:
                node_image = vision.load_rgb(self.node.image_path)
            except Exception:
                node_image = None
        point, _region, found = vision.locate_target(
            shot, node_image, self.current_relative, min_score=0.55
        )
        if point is None:
            QMessageBox.information(self, "提示", "没有可点击的位置")
            return
        offset_x = float(self.relative_offset_x.value())
        offset_y = float(self.relative_offset_y.value())
        height, width = shot.shape[:2]
        tap_x = max(0, min(width - 1, round(point[0] + offset_x)))
        tap_y = max(0, min(height - 1, round(point[1] + offset_y)))
        try:
            device.tap(tap_x, tap_y)
        except Exception as exc:
            QMessageBox.critical(self, "模拟偏移点击失败", str(exc))
            return
        if found:
            self.status_message.emit(
                f"模拟偏移点击：已识别相对图片并点击 ({tap_x}, {tap_y})"
            )
        else:
            self.status_message.emit(
                f"未识别到相对图片，已按框选相对位置偏移点击 ({tap_x}, {tap_y})"
            )
