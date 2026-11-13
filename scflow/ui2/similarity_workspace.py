from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import calibration as calibration_mod
from .. import config, similarity, vision
from ..db import Database
from .image_editor import ImageEditor
from .theme import ACCENT, GREEN, ORANGE

KIND_LABELS = {
    "image": "侧重点检测节点",
    "action": "侧重点检测节点",
    "detect": "侧重检测节点",
    "nodetect": "不检测步骤",
    "if": "IF 条件节点",
}
COMPONENTS = (
    ("color", "颜色"),
    ("texture", "纹理"),
    ("structure", "结构"),
    ("edge", "边缘"),
)
PREVIEW_KEYS = (
    ("base", "状态图片"),
    ("color", "颜色成分"),
    ("texture", "纹理成分"),
    ("structure", "结构成分"),
    ("edge", "边缘成分"),
)


class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ImagePreviewDialog(QDialog):
    def __init__(
        self,
        title: str,
        pixmap: QPixmap,
        clean_pixmap: QPixmap | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.pixmap_marked = pixmap
        self.pixmap_clean = clean_pixmap if clean_pixmap is not None else pixmap
        self.setWindowTitle(f"预览 · {title}")
        self.resize(980, 760)
        layout = QVBoxLayout(self)
        self.view = ImageEditor()
        self.view.set_mode("view")
        self.view.set_image(pixmap)
        layout.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.hide_boxes_check = QCheckBox("选框不可视")
        self.hide_boxes_check.setToolTip("勾选后只显示原始图片，不显示检测重点/点击对象/相对点击选框")
        fit_btn = QPushButton("适应窗口")
        zoom_in = QPushButton("放大")
        zoom_out = QPushButton("缩小")
        close_btn = QPushButton("关闭")
        row.addWidget(self.hide_boxes_check)
        row.addSpacing(12)
        row.addWidget(fit_btn)
        row.addWidget(zoom_in)
        row.addWidget(zoom_out)
        row.addStretch(1)
        row.addWidget(close_btn)
        layout.addLayout(row)
        self.hide_boxes_check.toggled.connect(self._toggle_boxes)
        fit_btn.clicked.connect(self.view.fit_image)
        zoom_in.clicked.connect(self.view.zoom_in)
        zoom_out.clicked.connect(self.view.zoom_out)
        close_btn.clicked.connect(self.accept)
        self.view.fit_image()

    def _toggle_boxes(self, checked: bool) -> None:
        self.view.set_image(self.pixmap_clean if checked else self.pixmap_marked)
        self.view.fit_image()


class SimilarityWorkspace(QWidget):
    status_message = Signal(str)
    back_requested = Signal()

    def __init__(self, db: Database, flow_workspace, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_workspace = flow_workspace
        self.node_id: int | None = None
        self.node = None
        self.image = None
        self.maps: dict[str, np.ndarray] = {}
        self._loading = False
        self._dirty = False
        self.component_locks: set[str] = set()
        self.calibration: dict | None = None
        self._syncing_threshold = False
        self.simulation_frame = None
        self.simulation_point: tuple[int, int] | None = None
        self.simulation_description = ""
        self.simulation_boxes: list[tuple[tuple[int, int, int, int], str, str]] = []
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._refresh_preview)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        self.back_btn = QPushButton("← 返回流程画布")
        self.title_label = QLabel("相似度编辑")
        self.title_label.setStyleSheet("font-size:18px;font-weight:700;")
        self.kind_label = QLabel("")
        self.kind_label.setObjectName("subtitle")
        self.save_btn = QPushButton("保存相似度")
        self.save_btn.setObjectName("primary")
        self.lock_btn = QPushButton("MR成分锁定…")
        self.lock_status = QLabel("未锁定")
        self.lock_status.setObjectName("subtitle")
        top.addWidget(self.back_btn)
        top.addSpacing(10)
        top.addWidget(self.title_label)
        top.addWidget(self.kind_label)
        top.addStretch(1)
        top.addWidget(self.lock_status)
        top.addWidget(self.lock_btn)
        top.addWidget(self.save_btn)
        layout.addLayout(top)

        self.formula_label = QLabel("")
        self.formula_label.setWordWrap(True)
        self.formula_label.setObjectName("subtitle")
        layout.addWidget(self.formula_label)

        self.unsupported = QLabel("该节点类型不参与相似度编辑。")
        self.unsupported.setStyleSheet("font-size:16px;color:#8b95a7;padding:30px;")
        self.unsupported.setVisible(False)
        layout.addWidget(self.unsupported)

        self.content = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.content, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("图片与成分预览（单击任意图片可放大）"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        preview_body = QWidget()
        self.preview_grid = QGridLayout(preview_body)
        self.preview_grid.setContentsMargins(4, 4, 4, 4)
        self.preview_labels: dict[str, ClickableLabel] = {}
        self.preview_cards: dict[str, QWidget] = {}
        self.preview_full_pixmaps: dict[str, QPixmap] = {}
        self.preview_clean_pixmaps: dict[str, QPixmap] = {}
        for index, (key, title) in enumerate(PREVIEW_KEYS):
            card = QWidget()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(2, 2, 2, 2)
            label = QLabel(title)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview = ClickableLabel("无预览")
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setCursor(Qt.CursorShape.PointingHandCursor)
            preview.clicked.connect(lambda key=key: self._open_preview(key))
            preview.setMinimumSize(220, 180)
            preview.setStyleSheet(
                "background:#0d0f13;border:1px solid #333842;color:#5c6575;"
            )
            card_layout.addWidget(label)
            card_layout.addWidget(preview, 1)
            self.preview_grid.addWidget(card, index // 3, index % 3)
            self.preview_cards[key] = card
            self.preview_labels[key] = preview
        scroll.setWidget(preview_body)
        left_layout.addWidget(scroll, 1)
        self.content.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        self.controls_layout.setContentsMargins(4, 4, 4, 4)

        threshold_group = QGroupBox("匹配阈值")
        threshold_form = QFormLayout(threshold_group)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 0.999)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setToolTip(
            "原始空间阈值。启用自动校准后，运行时优先使用校准后的归一化阈值。"
        )
        threshold_form.addRow("匹配阈值 r（原始）", self.threshold_spin)
        self.controls_layout.addWidget(threshold_group)

        self.calibration_group = QGroupBox("阈值自动校准（负样本基线）")
        calibration_layout = QVBoxLayout(self.calibration_group)
        fpr_row = QHBoxLayout()
        fpr_row.addWidget(QLabel("目标误匹配率"))
        self.fpr_combo = QComboBox()
        self.fpr_combo.addItem("1%", 0.01)
        self.fpr_combo.addItem("0.5%", 0.005)
        self.fpr_combo.addItem("0.1%", 0.001)
        self.fpr_combo.setToolTip("负样本中允许被误判为匹配的比例，越小越严格")
        fpr_row.addWidget(self.fpr_combo, 1)
        calibration_layout.addLayout(fpr_row)
        self.calibration_label = QLabel("未校准：运行时使用上面的原始阈值 r。")
        self.calibration_label.setObjectName("subtitle")
        self.calibration_label.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_label)
        calibration_buttons = QHBoxLayout()
        self.calibrate_btn = QPushButton("自动校准阈值…")
        self.calibrate_btn.setToolTip(
            "用同流程其它节点图片作为负样本，计算 μ、σ 与阈值；"
            "无关图会被压到 0 附近。"
        )
        self.clear_calibration_btn = QPushButton("清除校准")
        calibration_buttons.addWidget(self.calibrate_btn)
        calibration_buttons.addWidget(self.clear_calibration_btn)
        calibration_layout.addLayout(calibration_buttons)
        self.controls_layout.addWidget(self.calibration_group)

        self.component_threshold_group = QGroupBox("MR 成分匹配阈值（低于阈值的成分视为不匹配）")
        component_threshold_form = QFormLayout(self.component_threshold_group)
        self.component_threshold_spins: dict[str, QDoubleSpinBox] = {}
        for key, text in COMPONENTS:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._on_component_threshold_changed)
            component_threshold_form.addRow(text, spin)
            self.component_threshold_spins[key] = spin
        self.controls_layout.addWidget(self.component_threshold_group)

        self.detect_group = QGroupBox("侧重检测节点")
        detect_layout = QVBoxLayout(self.detect_group)
        detect_layout.addWidget(QLabel("检测权重 W = 1（固定）"))
        self.detect_group.setVisible(False)
        self.controls_layout.addWidget(self.detect_group)

        self.region_group = QGroupBox("区域权重")
        region_form = QFormLayout(self.region_group)
        self.roi_weight_spin = QDoubleSpinBox()
        self.roi_weight_spin.setRange(0.0, 1.0)
        self.roi_weight_spin.setSingleStep(0.01)
        self.roi_weight_spin.setDecimals(2)
        self.non_roi_weight_label = QLabel("0.28")
        region_form.addRow("W1 检测重点", self.roi_weight_spin)
        region_form.addRow("W2 剩余部分", self.non_roi_weight_label)
        self.controls_layout.addWidget(self.region_group)

        self.roi_components_group, self.roi_sliders, self.roi_labels, self.roi_sum = self._component_group(
            "检测重点 MR 成分", "roi"
        )
        self.non_components_group, self.non_sliders, self.non_labels, self.non_sum = self._component_group(
            "剩余部分 MR非 成分", "non_roi"
        )
        self.target_components_group, self.target_sliders, self.target_labels, self.target_sum = self._component_group(
            "点击对象 MR 成分（用于定位点击对象）", "target"
        )
        self.group_sum_labels = {
            "roi": self.roi_sum,
            "non_roi": self.non_sum,
            "target": self.target_sum,
        }
        self.controls_layout.addWidget(self.roi_components_group)
        self.controls_layout.addWidget(self.non_components_group)
        self.controls_layout.addWidget(self.target_components_group)

        test_group = QGroupBox("测试相似度")
        test_layout = QVBoxLayout(test_group)
        self.load_test_btn = QPushButton("载入图片得分…")
        self.load_test_btn.setToolTip("选择一张本地图片，按当前权重大致计算它在该节点算法下的相似度 s")
        self.phone_score_btn = QPushButton("手机当前画面分值计算")
        self.phone_score_btn.setToolTip("通过 ADB 截取当前手机画面，并按当前权重大致计算相似度 s")
        self.simulate_check = QCheckBox("模拟运行（仅手机当前画面）")
        self.simulate_check.setToolTip("阈值匹配时模拟点击手机；不匹配时提示阈值不匹配")
        self.test_label = QLabel(
            "「载入图片得分…」选择本地图片计算；「手机当前画面分值计算」通过 ADB 截取手机当前画面计算。"
        )
        self.test_label.setWordWrap(True)
        self.observe_btn = QPushButton("观测点击点")
        self.observe_btn.setVisible(False)
        self.clear_cache_btn = QPushButton("清理测试缓存")
        test_layout.addWidget(self.load_test_btn)
        test_layout.addWidget(self.phone_score_btn)
        test_layout.addWidget(self.simulate_check)
        test_layout.addWidget(self.test_label)
        test_layout.addWidget(self.observe_btn)
        test_layout.addWidget(self.clear_cache_btn)
        self.controls_layout.addWidget(test_group)

        buttons = QHBoxLayout()
        self.reset_btn = QPushButton("恢复默认权重")
        self.save_btn2 = QPushButton("保存相似度")
        self.save_btn2.setObjectName("primary")
        buttons.addWidget(self.reset_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.save_btn2)
        self.controls_layout.addLayout(buttons)
        self.controls_layout.addStretch(1)
        right_scroll.setWidget(controls)
        right_layout.addWidget(right_scroll, 1)
        self.content.addWidget(right)
        self.content.setSizes([760, 380])

        self.back_btn.clicked.connect(self._back)
        self.save_btn.clicked.connect(lambda: self._save(silent=False))
        self.save_btn2.clicked.connect(lambda: self._save(silent=False))
        self.reset_btn.clicked.connect(self._reset)
        self.load_test_btn.clicked.connect(self._load_test_image)
        self.phone_score_btn.clicked.connect(self._calc_phone_score)
        self.observe_btn.clicked.connect(self._show_observation)
        self.clear_cache_btn.clicked.connect(self._clear_test_cache)
        self.lock_btn.clicked.connect(self._open_lock_dialog)
        self.calibrate_btn.clicked.connect(self._run_calibration)
        self.clear_calibration_btn.clicked.connect(self._clear_calibration)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self.roi_weight_spin.valueChanged.connect(self._on_region_weight_changed)
        self.clear()

    def _component_group(self, title: str, group_key: str):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        head = QHBoxLayout()
        sum_label = QLabel("合计 1.00")
        sum_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        head.addStretch(1)
        head.addWidget(sum_label)
        layout.addLayout(head)
        sliders = {}
        labels = {}
        for key, text in COMPONENTS:
            row = QHBoxLayout()
            row.addWidget(QLabel(text))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(25)
            value_label = QLabel("0.25")
            value_label.setMinimumWidth(40)
            slider.valueChanged.connect(
                lambda value, name=key, gkey=group_key: self._on_component_changed(
                    gkey, name, value
                )
            )
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            layout.addLayout(row)
            sliders[key] = slider
            labels[key] = value_label
        return group, sliders, labels, sum_label

    def clear(self) -> None:
        self.node_id = None
        self.node = None
        self.image = None
        self.maps = {}
        self._dirty = False
        self.title_label.setText("相似度编辑")
        self.kind_label.setText("")
        self.formula_label.setText(
            "在流程画布中开启「相似度编辑」后点击节点，或右键节点 / 匹配阈值进入。"
        )
        self.unsupported.setVisible(False)
        self.content.setVisible(True)
        self.save_btn.setVisible(True)
        self.calibration = None
        self.component_locks = set()
        self._update_lock_status()
        self._update_calibration_label()
        self.preview_full_pixmaps.clear()
        self.preview_clean_pixmaps.clear()
        for label in self.preview_labels.values():
            label.setText("无预览")
            label.setPixmap(QPixmap())

    def load_node(self, node_id: int, force: bool = False) -> None:
        if not force and node_id == self.node_id:
            return
        if self._dirty:
            self._save(silent=True)
        node = self.db.get_node(node_id)
        if node is None:
            self.clear()
            return
        self.observe_btn.setVisible(False)
        self.simulation_frame = None
        self.simulation_point = None
        self.simulation_description = ""
        self.simulation_boxes = []
        self.node_id = node_id
        self.node = node
        self.component_locks = set()
        self._update_lock_status()
        self._loading = True
        self.title_label.setText(f"相似度编辑 · {node.name}")
        self.kind_label.setText(KIND_LABELS.get(node.kind, node.kind))
        supported = node.kind in ("image", "action", "detect")
        self.unsupported.setVisible(not supported)
        self.content.setVisible(supported)
        self.save_btn.setVisible(supported)
        if not supported:
            self.unsupported.setText(
                f"「{KIND_LABELS.get(node.kind, node.kind)}」不参与相似度编辑，"
                "请返回流程画布继续编辑。"
            )
            self.formula_label.setText("")
            self._loading = False
            return
        self.image = None
        if node.image_path and Path(node.image_path).exists():
            try:
                self.image = vision.load_rgb(node.image_path)
            except Exception:
                self.image = None
        self.threshold_spin.setValue(node.threshold)
        detect = node.kind == "detect"
        self.detect_group.setVisible(detect)
        self.region_group.setVisible(not detect and node.roi is not None)
        self.non_components_group.setVisible(not detect and node.roi is not None)
        has_click_object = node.has_target or node.relative_roi is not None
        self.target_components_group.setVisible(has_click_object)
        if node.has_target:
            self.target_components_group.setTitle("点击对象 MR 成分（框选点击对象）")
        elif node.relative_roi is not None:
            self.target_components_group.setTitle("点击对象 MR 成分（相对点击）")
        else:
            self.target_components_group.setTitle("点击对象 MR 成分")
        model = similarity.get_model(node)
        self._apply_model(model)
        if node.kind == "detect":
            self.formula_label.setText(
                "侧重检测节点：W = 1（固定），s = MR；"
                "MR = 四成分加权几何平均（含短板惩罚）。"
            )
        else:
            self.formula_label.setText(
                "侧重点检测节点：s = (W1·MR + W2·MR非) / (W1 + W2)；"
                "MR / MR非 = 四成分加权几何平均（含短板惩罚）。"
            )
        self._loading = False
        self._dirty = False
        self._schedule_preview()

    def _apply_model(self, model: dict) -> None:
        self._loading = True
        calibration = model.get("calibration")
        self.calibration = (
            dict(calibration)
            if isinstance(calibration, dict) and calibration
            else None
        )
        region = model.get("region_weights", {})
        roi_weight = float(region.get("roi", 1.0 if model.get("mode") == "roi_only" else 0.72))
        self.roi_weight_spin.setValue(roi_weight)
        self.non_roi_weight_label.setText(f"{max(0.0, 1.0 - roi_weight):.2f}")
        thresholds = model.get("component_thresholds") or {}
        for key, spin in self.component_threshold_spins.items():
            try:
                spin.setValue(max(0.0, min(1.0, float(thresholds.get(key, 0.0)))))
            except (TypeError, ValueError):
                spin.setValue(0.0)
        for key in ("roi", "non_roi", "target"):
            sliders = {
                "roi": self.roi_sliders,
                "non_roi": self.non_sliders,
                "target": self.target_sliders,
            }[key]
            labels = {
                "roi": self.roi_labels,
                "non_roi": self.non_labels,
                "target": self.target_labels,
            }[key]
            weights = model.get(key) or similarity.default_component_weights()
            for name, slider in sliders.items():
                value = int(round(float(weights.get(name, 0.25)) * 100))
                slider.setValue(value)
                labels[name].setText(f"{value / 100:.2f}")
        self._loading = False
        self._refresh_sum_labels()
        self._update_calibration_label()

    def _collect_model(self) -> dict:
        detect = self.node is not None and self.node.kind == "detect"
        model = {
            "version": 1,
            "mode": "roi_only" if detect else "roi_plus_full",
            "region_weights": {
                "roi": 1.0 if detect else float(self.roi_weight_spin.value()),
                "non_roi": 0.0 if detect else max(0.0, 1.0 - float(self.roi_weight_spin.value())),
            },
            "roi": {
                name: slider.value() / 100.0
                for name, slider in self.roi_sliders.items()
            },
            "non_roi": {
                name: slider.value() / 100.0
                for name, slider in self.non_sliders.items()
            },
            "component_thresholds": {
                name: spin.value()
                for name, spin in self.component_threshold_spins.items()
            },
        }
        if self.node is not None and (
            self.node.has_target or self.node.relative_roi is not None
        ):
            model["target"] = {
                name: slider.value() / 100.0
                for name, slider in self.target_sliders.items()
            }
        if self.calibration:
            model["calibration"] = dict(self.calibration)
        return model

    def _group_sliders(self, group_key: str) -> dict[str, QSlider]:
        return {
            "roi": self.roi_sliders,
            "non_roi": self.non_sliders,
            "target": self.target_sliders,
        }[group_key]

    def _group_labels(self, group_key: str) -> dict[str, QLabel]:
        return {
            "roi": self.roi_labels,
            "non_roi": self.non_labels,
            "target": self.target_labels,
        }[group_key]

    def _refresh_sum_labels(self) -> None:
        for group_key, label in self.group_sum_labels.items():
            sliders = self._group_sliders(group_key)
            total = sum(slider.value() for slider in sliders.values()) / 100.0
            label.setText(f"合计 {total:.2f}")
            label.setStyleSheet("color:#f85149;" if total > 1.0001 else "")

    def _set_group_values(self, group_key: str, values: dict[str, int]) -> None:
        sliders = self._group_sliders(group_key)
        labels = self._group_labels(group_key)
        self._loading = True
        try:
            for name, slider in sliders.items():
                value = max(0, min(100, int(round(values.get(name, slider.value())))))
                slider.setValue(value)
                labels[name].setText(f"{value / 100:.2f}")
        finally:
            self._loading = False

    def _balance_group(self, group_key: str, changed_key: str | None = None) -> tuple[bool, str]:
        sliders = self._group_sliders(group_key)
        values = {name: slider.value() for name, slider in sliders.items()}
        total = sum(values.values())
        if total <= 100:
            return True, ""
        locked = {name for name in self.component_locks if name in values}
        locked_sum = sum(values[name] for name in locked)
        unlocked = [name for name in values if name not in locked]
        if locked_sum > 100:
            return False, "锁定成分的权重之和已经超过 1"
        if not unlocked:
            return False, "所有成分都已锁定，且总权重超过 1"
        available = 100 - locked_sum
        unlocked_sum = sum(values[name] for name in unlocked)
        if unlocked_sum <= available:
            return True, ""
        if changed_key in unlocked:
            others = [name for name in unlocked if name != changed_key]
            other_sum = sum(values[name] for name in others)
            changed_value = values[changed_key]
            target_other = max(0, available - changed_value)
            if other_sum > 0:
                scale = target_other / float(other_sum)
                for name in others:
                    values[name] = int(round(values[name] * scale))
            else:
                for name in others:
                    values[name] = 0
            if changed_value > available:
                values[changed_key] = available
        else:
            scale = available / float(unlocked_sum)
            for name in unlocked:
                values[name] = int(round(values[name] * scale))
        # Reconcile integer rounding so unlocked values add up exactly.
        unlocked_total = sum(values[name] for name in unlocked)
        remainder = available - unlocked_total
        if remainder != 0:
            target = changed_key if changed_key in unlocked else max(unlocked, key=lambda item: values[item])
            values[target] = max(0, min(100, values[target] + remainder))
        self._set_group_values(group_key, values)
        self._refresh_sum_labels()
        return True, ""

    def _on_component_changed(self, group_key: str, key: str, value: int) -> None:
        if self._loading:
            return
        labels = self._group_labels(group_key)
        labels[key].setText(f"{value / 100:.2f}")
        self._balance_group(group_key, changed_key=key)
        self._refresh_sum_labels()
        self._invalidate_calibration()
        self._dirty = True
        self._schedule_preview()

    def _update_lock_status(self) -> None:
        if not self.component_locks:
            self.lock_status.setText("未锁定")
            self.lock_status.setStyleSheet("")
            return
        names = "、".join(text for key, text in COMPONENTS if key in self.component_locks)
        self.lock_status.setText(f"已锁定：{names}")
        self.lock_status.setStyleSheet("color:#f0883e;")

    def _open_lock_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("MR 成分锁定")
        dialog.resize(320, 260)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "锁定的成分在自动平衡时不会被减少。\n"
            "4 个成分合计不能超过 1；超出的部分会从未锁定成分中扣减。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        boxes = {}
        for key, text in COMPONENTS:
            box = QCheckBox(text)
            box.setChecked(key in self.component_locks)
            layout.addWidget(box)
            boxes[key] = box
        unlock_all = QPushButton("全部解锁")
        layout.addWidget(unlock_all)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)
        unlock_all.clicked.connect(lambda: [box.setChecked(False) for box in boxes.values()])
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.component_locks = {key for key, box in boxes.items() if box.isChecked()}
            self._update_lock_status()
            self._refresh_sum_labels()
            self.status_message.emit("MR 成分锁定已更新")

    def _update_calibration_label(self) -> None:
        if self.calibration:
            mu = float(self.calibration.get("mu", 0.0) or 0.0)
            r_raw = float(self.calibration.get("threshold_raw", 0.0) or 0.0)
            r_norm = float(self.calibration.get("threshold_norm", 0.0) or 0.0)
            fpr = float(self.calibration.get("fpr", 0.01) or 0.01)
            n_pos = int(self.calibration.get("n_pos", 0) or 0)
            n_neg = int(self.calibration.get("n_neg", 0) or 0)
            self.calibration_label.setStyleSheet("color:#3fb950;")
            self.calibration_label.setText(
                f"已校准：μ={mu:.3f}，原始阈值={r_raw:.3f}，"
                f"归一化阈值={r_norm:.3f}，目标误匹配率={fpr * 100:g}%\n"
                f"正样本 {n_pos}，负样本 {n_neg}。运行时使用归一化分数比较。"
            )
        else:
            self.calibration_label.setStyleSheet("")
            self.calibration_label.setText("未校准：运行时使用上面的原始阈值 r。")

    def _invalidate_calibration(self, reason: str = "权重已修改，校准已清除") -> None:
        if not self.calibration:
            return
        self.calibration = None
        self._update_calibration_label()
        self.status_message.emit(reason)

    def _run_calibration(self) -> None:
        if self.node is None or self.node.kind not in ("image", "action", "detect"):
            QMessageBox.information(self, "提示", "请先选择一个可编辑相似度的节点")
            return
        model = self._collect_model()
        model.pop("calibration", None)
        fpr = float(self.fpr_combo.currentData() or 0.01)
        try:
            calibration = calibration_mod.calibrate_node(
                self.db, self.node, model, fpr=fpr
            )
        except Exception as exc:
            QMessageBox.warning(self, "自动校准失败", str(exc))
            return
        self.calibration = calibration
        self._syncing_threshold = True
        try:
            self.threshold_spin.setValue(
                float(calibration.get("threshold_raw", self.threshold_spin.value()))
            )
        finally:
            self._syncing_threshold = False
        self._dirty = True
        self._update_calibration_label()
        self.status_message.emit(
            f"已校准：原始阈值 {calibration.get('threshold_raw', 0):.3f}，"
            f"归一化阈值 {calibration.get('threshold_norm', 0):.3f}"
        )

    def _clear_calibration(self) -> None:
        self.calibration = None
        self._dirty = True
        self._update_calibration_label()
        self.status_message.emit("已清除阈值校准")

    def _on_threshold_changed(self, _value: float) -> None:
        if self._loading or self._syncing_threshold:
            return
        if self.calibration:
            self._invalidate_calibration("阈值已手动修改，校准已清除")
        self._dirty = True

    def _on_component_threshold_changed(self, _value: float) -> None:
        if self._loading:
            return
        self._dirty = True

    def _on_region_weight_changed(self, value: float) -> None:
        if self._loading:
            return
        self.non_roi_weight_label.setText(f"{max(0.0, 1.0 - value):.2f}")
        self._invalidate_calibration()
        self._dirty = True
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        if self._loading:
            return
        self.preview_timer.start(220)

    def _refresh_preview(self) -> None:
        if self.image is None or self.node is None:
            self.preview_full_pixmaps.clear()
            self.preview_clean_pixmaps.clear()
            for label in self.preview_labels.values():
                label.setText("无预览")
                label.setPixmap(QPixmap())
            return
        self.maps = similarity.chart_data(self.image)
        detect = self.node.kind == "detect"
        self.detect_group.setVisible(detect)
        self.region_group.setVisible(not detect and self.node.roi is not None)
        self.non_components_group.setVisible(not detect and self.node.roi is not None)
        self.target_components_group.setVisible(
            self.node.has_target or self.node.relative_roi is not None
        )
        self.preview_cards["base"].setVisible(True)
        self._set_pixmap(
            "base",
            self._base_pixmap(self.image, self.node, draw_regions=True),
            self._base_pixmap(self.image, self.node, draw_regions=False),
        )
        for key in ("color", "texture", "structure", "edge"):
            self.preview_cards[key].setVisible(True)
            self._set_pixmap(
                key,
                self._heatmap_pixmap(self.maps[key], self.node, draw_regions=True),
                self._heatmap_pixmap(self.maps[key], self.node, draw_regions=False),
            )

    def _base_pixmap(self, image, node, draw_regions: bool = True) -> QPixmap:
        pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB").copy()
        if draw_regions:
            draw = ImageDraw.Draw(pil)
            width, height = pil.size
            self._draw_regions(draw, node, width, height)
        return self._pil_to_pixmap(pil)

    def _heatmap_pixmap(
        self,
        values: np.ndarray,
        node,
        base=None,
        mask: np.ndarray | None = None,
        draw_regions: bool = True,
    ) -> QPixmap:
        value = np.clip(values.astype(np.float32), 0.0, 1.0)
        red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
        heat = np.stack([red, green, blue], axis=-1) * 255.0
        if base is None:
            base = self.image
        if base is None:
            base = np.zeros((*value.shape, 3), dtype=np.float32)
        base_pil = Image.fromarray(
            np.clip(base, 0, 255).astype(np.uint8), "RGB"
        ).resize((value.shape[1], value.shape[0]))
        base_array = np.asarray(base_pil, dtype=np.float32)
        dim = np.clip(base_array * 0.32 + 18.0, 0.0, 255.0)
        blended = dim * 0.28 + heat * 0.72
        if mask is not None:
            blended = np.where(mask[..., None], blended, dim)
        pil = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")
        if draw_regions:
            draw = ImageDraw.Draw(pil)
            self._draw_regions(draw, node, pil.width, pil.height)
        return self._pil_to_pixmap(pil)

    def _draw_regions(self, draw, node, width: int, height: int) -> None:
        roi = getattr(node, "roi", None)
        if roi:
            x1 = roi[0] * width
            y1 = roi[1] * height
            x2 = (roi[0] + roi[2]) * width
            y2 = (roi[1] + roi[3]) * height
            draw.rectangle((x1, y1, x2, y2), outline=ACCENT, width=3)
        target = getattr(node, "target_region", None)
        if target:
            x1 = target[0] * width
            y1 = target[1] * height
            x2 = (target[0] + target[2]) * width
            y2 = (target[1] + target[3]) * height
            draw.rectangle((x1, y1, x2, y2), outline=ORANGE, width=3)
        relative = getattr(node, "relative_roi", None)
        if relative:
            x1 = relative[0] * width
            y1 = relative[1] * height
            x2 = (relative[0] + relative[2]) * width
            y2 = (relative[1] + relative[3]) * height
            draw.rectangle((x1, y1, x2, y2), outline=GREEN, width=2)
            offset_x, offset_y = node.relative_offset
            node_width = node.img_w or (
                self.image.shape[1] if self.image is not None else width
            )
            node_height = node.img_h or (
                self.image.shape[0] if self.image is not None else height
            )
            cx = (relative[0] + relative[2] / 2.0) * width + offset_x * width / max(1, node_width)
            cy = (relative[1] + relative[3] / 2.0) * height + offset_y * height / max(1, node_height)
            radius = 9
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=GREEN, width=3)

    def _pil_to_pixmap(self, image: Image.Image) -> QPixmap:
        image = image.convert("RGB")
        data = image.tobytes()
        qimage = QImage(data, image.width, image.height, 3 * image.width, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimage)

    def _set_pixmap(
        self,
        key: str,
        pixmap: QPixmap,
        clean_pixmap: QPixmap | None = None,
    ) -> None:
        self.preview_full_pixmaps[key] = pixmap
        self.preview_clean_pixmaps[key] = clean_pixmap if clean_pixmap is not None else pixmap
        label = self.preview_labels[key]
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        for key, pixmap in self.preview_full_pixmaps.items():
            label = self.preview_labels.get(key)
            if label is None or pixmap.isNull():
                continue
            label.setPixmap(
                pixmap.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def reload_current(self) -> None:
        if self.node_id is None:
            return
        if self._dirty:
            self._save(silent=True)
        self.load_node(self.node_id, force=True)

    def _open_preview(self, key: str) -> None:
        pixmap = self.preview_full_pixmaps.get(key)
        if pixmap is None or pixmap.isNull():
            return
        title = dict(PREVIEW_KEYS).get(key, key)
        clean = self.preview_clean_pixmaps.get(key)
        ImagePreviewDialog(title, pixmap, clean, self).exec()

    def _load_test_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择测试截图", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path or self.node is None:
            return
        try:
            frame = vision.load_rgb(path)
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self._cache_frame(frame, "image")
        self._score_frame(frame, "图片", simulate=False)

    def _calc_phone_score(self) -> None:
        if self.node is None or self.node.kind not in ("image", "action", "detect"):
            QMessageBox.information(self, "提示", "请先选择一个可编辑相似度的节点")
            return
        serial = getattr(self.flow_workspace.inspector, "serial", None)
        try:
            from ..device import AdbDevice

            frame = AdbDevice(serial=serial).screenshot()
        except Exception as exc:
            QMessageBox.critical(self, "获取手机画面失败", str(exc))
            return
        self._cache_frame(frame, "phone")
        self._score_frame(frame, "手机当前画面", simulate=True)

    @staticmethod
    def _score_lines(
        prefix: str,
        raw: float,
        score: float,
        r_raw: float,
        r_effective: float,
        calibration: dict | None,
    ) -> list[str]:
        lines = [f"{prefix} · 检测对象得分 s = {score:.3f}"]
        if isinstance(calibration, dict) and calibration:
            lines.append(f"原始得分 s_raw = {raw:.3f}")
            lines.append(f"匹配阈值 r_norm = {r_effective:.3f}")
        else:
            lines.append(f"匹配阈值 r = {r_raw:.3f}")
        lines.append("结果：" + ("通过" if score >= r_effective else "不通过"))
        return lines

    def _score_frame(self, frame, prefix: str, simulate: bool = False) -> None:
        if self.node is None:
            return
        self.observe_btn.setVisible(False)
        model = self._collect_model()
        calibration = model.get("calibration")
        threshold_raw = float(self.threshold_spin.value())
        threshold_effective = (
            float(calibration.get("threshold_norm", threshold_raw))
            if isinstance(calibration, dict) and calibration
            else threshold_raw
        )
        # 模板定位使用原始阈值；成分得分再应用校准。
        threshold = threshold_raw
        lines: list[str] = []
        detection_score = 0.0
        detection_raw = 0.0
        detection_low: list[str] = []
        detect_region = None
        if self.node.kind == "detect":
            if not self.node.roi or not self.node.image_path:
                self.test_label.setText(f"{prefix}：侧重检测节点缺少检测重点或图片，无法计算。")
                return
            if not vision.cv2_available():
                lines = [
                    f"{prefix} · 检测对象得分 s = 0.000",
                    "未安装 OpenCV，无法进行侧重检测节点的整屏模板定位。",
                ]
            else:
                try:
                    node_image = vision.load_rgb(self.node.image_path)
                    crop = vision.crop_relative(node_image, self.node.roi)
                    found = vision.template_locate(
                        frame, crop, min_score=max(0.45, threshold)
                    )
                except Exception as exc:
                    QMessageBox.critical(self, "计算失败", str(exc))
                    return
                if found is None:
                    detection_raw = 0.0
                    detection_score = similarity.apply_calibration(0.0, calibration)
                    lines = self._score_lines(
                        prefix,
                        detection_raw,
                        detection_score,
                        threshold_raw,
                        threshold_effective,
                        calibration,
                    )
                    lines.append("未定位到检测重点")
                else:
                    x, y, w, h, template_score = found
                    detect_region = (x, y, w, h)
                    components = similarity.region_component_scores(
                        self.node, frame, self.node.roi, (x, y, w, h)
                    )
                    low = similarity.check_component_thresholds(
                        components or {}, model.get("component_thresholds")
                    )
                    detection_low = low
                    detection_raw = 0.0 if low else similarity.weighted_score(
                        components or {}, model["roi"]
                    )
                    detection_score = similarity.apply_calibration(
                        detection_raw, calibration
                    )
                    lines = self._score_lines(
                        prefix,
                        detection_raw,
                        detection_score,
                        threshold_raw,
                        threshold_effective,
                        calibration,
                    )
                    lines.append(f"模板定位相似度 = {template_score:.3f}")
                    if components:
                        lines.append(
                            "MR："
                            + " ".join(
                                f"{key}={value:.3f}" for key, value in components.items()
                            )
                        )
                    if low:
                        lines.append("低于阈值成分：" + "、".join(low) + "，动作相似度低")
        else:
            result = similarity.explain_node(self.node, frame, model)
            if result is None:
                self.test_label.setText(f"{prefix}：无法读取节点图片。")
                return
            detection_score = float(result["score"])
            detection_raw = float(result.get("raw_score", detection_score))
            detection_low = result.get("low_components") or []
            lines = self._score_lines(
                prefix,
                detection_raw,
                detection_score,
                threshold_raw,
                threshold_effective,
                calibration,
            )
            for part in ("roi", "non_roi"):
                if part in result:
                    comps = result[part]
                    lines.append(
                        f"{'MR' if part == 'roi' else 'MR非'}："
                        + " ".join(f"{key}={value:.3f}" for key, value in comps.items())
                    )
            low = result.get("low_components") or []
            if low:
                lines.append("低于阈值成分：" + "、".join(low) + "，动作相似度低")
        click_roi = None
        click_min_score = 0.55
        if self.node.has_target and self.node.target_region is not None:
            click_roi = self.node.target_region
            click_min_score = float(self.node.target_threshold)
        elif self.node.relative_roi is not None:
            click_roi = self.node.relative_roi
            click_min_score = 0.55
        click_score = None
        click_point = None
        click_region = None
        click_found = False
        click_low: list[str] = []
        if click_roi is not None and self.node.image_path and Path(self.node.image_path).exists():
            try:
                node_image = vision.load_rgb(self.node.image_path)
                click_point, click_region, click_found = vision.locate_target(
                    frame, node_image, click_roi, min_score=click_min_score
                )
                weights = model.get("target") or similarity.default_component_weights()
                click_info = similarity.roi_explain(
                    self.node,
                    frame,
                    click_roi,
                    weights,
                    model.get("component_thresholds"),
                    click_region,
                )
                if click_info is not None:
                    click_score = float(click_info["score"])
                    click_low = click_info.get("low_components") or []
            except Exception as exc:
                lines.append(f"点击对象得分计算失败：{exc}")
                click_low = []
            if click_score is not None:
                lines.append(f"{prefix} · 点击对象得分 s = {click_score:.3f}")
                if not click_found:
                    lines.append("点击对象未定位，按相对位置计算")
                if click_low:
                    lines.append(
                        "点击对象低于阈值成分：" + "、".join(click_low) + "，动作相似度低"
                    )
        if click_score is not None and click_found and click_roi is not None:
            action = self.node.action or {}
            if self.node.has_target:
                min_mr = float(action.get("target_min_mr", 0.4) or 0.4)
            else:
                min_mr = float(action.get("rel_min_mr", 0.4) or 0.4)
            if click_score < min_mr:
                click_found = False
                click_region = None
                height, width = frame.shape[:2]
                click_point = self._relative_center(click_roi, width, height)
                lines.append(
                    f"点击对象识别分 {click_score:.3f} < 最低 MR {min_mr:.2f}，按未识别处理"
                )
        boxes: list[tuple[tuple[int, int, int, int], str, str]] = []
        if self.node.kind == "detect" and detect_region is not None:
            boxes.append((detect_region, "detect", ACCENT))
        elif self.node.has_target and click_found and click_region is not None:
            boxes.append((click_region, "target", ORANGE))
        elif self.node.relative_roi is not None and click_found and click_region is not None:
            boxes.append((click_region, "relative", GREEN))
        if simulate and self.simulate_check.isChecked():
            matched = detection_score >= threshold_effective and not detection_low
            if not matched:
                lines.append("模拟运行：阈值不匹配，未点击")
                detail = (
                    f"检测对象得分 {detection_score:.3f}，"
                    f"匹配阈值 {threshold_effective:.3f}"
                )
                if isinstance(calibration, dict) and calibration:
                    detail += f"（原始得分 {detection_raw:.3f}）"
                if detection_low:
                    detail += "；低于阈值成分：" + "、".join(detection_low)
                QMessageBox.information(self, "阈值不匹配", detail)
            else:
                point, description, reason = self._resolve_simulation_point(
                    frame, model, detect_region, click_point, click_found, click_low
                )
                if point is None:
                    lines.append(f"模拟运行：{reason}")
                else:
                    try:
                        from ..device import AdbDevice

                        serial = getattr(self.flow_workspace.inspector, "serial", None)
                        AdbDevice(serial=serial).tap(point[0], point[1])
                        self.simulation_frame = frame.copy()
                        self.simulation_point = point
                        self.simulation_description = description
                        self.simulation_boxes = boxes
                        self.observe_btn.setVisible(True)
                        lines.append(
                            f"模拟运行：已点击 ({point[0]}, {point[1]}) · {description}"
                        )
                    except Exception as exc:
                        lines.append(f"模拟运行：点击失败：{exc}")
        elif self.simulate_check.isChecked() and not simulate:
            lines.append("模拟运行仅对「手机当前画面分值计算」生效")
        self.test_label.setText("\n".join(lines))
        message = f"{prefix}：检测对象 {detection_score:.3f}"
        if click_score is not None:
            message += f" · 点击对象 {click_score:.3f}"
        self.status_message.emit(message)

    @staticmethod
    def _relative_center(roi, width: int, height: int) -> tuple[int, int]:
        x = int(round((roi[0] + roi[2] / 2.0) * width))
        y = int(round((roi[1] + roi[3] / 2.0) * height))
        return (max(0, min(width - 1, x)), max(0, min(height - 1, y)))

    def _resolve_simulation_point(
        self,
        frame,
        model: dict,
        detect_region,
        click_point,
        click_found: bool,
        click_low: list[str],
    ) -> tuple[tuple[int, int] | None, str, str]:
        node = self.node
        if node is None:
            return None, "", "节点不存在"
        if node.action_type != "tap":
            return None, "", f"当前节点动作是「{node.action_type}」，不是点击"
        height, width = frame.shape[:2]
        selector = (node.action or {}).get("ui_selector")
        if isinstance(selector, dict) and selector:
            try:
                from ..device import AdbDevice

                serial = getattr(self.flow_workspace.inspector, "serial", None)
                element = AdbDevice(serial=serial).find_ui_element(selector)
            except Exception as exc:
                return None, "", f"控件树定位失败：{exc}"
            if element is None:
                return None, "", "未找到控件树元素"
            center = element.get("center", [0, 0])
            return (
                (int(center[0]), int(center[1])),
                "控件树点击",
                "",
            )
        if node.kind == "detect":
            if detect_region is None:
                return None, "", "未定位到检测重点"
            x, y, w, h = detect_region
            return (int(x + w // 2), int(y + h // 2)), "检测重点点击", ""
        if node.has_target:
            if click_low:
                return None, "", "点击对象动作相似度低：" + "、".join(click_low)
            center = (
                click_point
                if click_point is not None
                else self._relative_center(node.target_region, width, height)
            )
            if not click_found and (node.action or {}).get("target_strict"):
                return None, "", "未识别到点击对象"
            return (
                (
                    max(0, min(width - 1, int(center[0]))),
                    max(0, min(height - 1, int(center[1]))),
                ),
                "点击对象中心",
                "",
            )
        if node.relative_roi is not None:
            if click_low:
                return None, "", "相对点击动作相似度低：" + "、".join(click_low)
            center = (
                click_point
                if click_point is not None
                else self._relative_center(node.relative_roi, width, height)
            )
            offset_x, offset_y = node.relative_offset
            return (
                (
                    max(0, min(width - 1, round(center[0] + offset_x))),
                    max(0, min(height - 1, round(center[1] + offset_y))),
                ),
                "相对偏移点击",
                "",
            )
        if node.has_click_point:
            return (
                (
                    max(0, min(width - 1, round(node.click_x * width))),
                    max(0, min(height - 1, round(node.click_y * height))),
                ),
                "固定点击点",
                "",
            )
        return (width // 2, height // 2), "屏幕中心", ""

    def _cache_frame(self, frame, prefix: str) -> None:
        try:
            path = config.similarity_cache_dir() / (
                f"{prefix}_{int(time.time() * 1000)}_{uuid4().hex[:6]}.png"
            )
            vision.save_rgb(path, frame)
        except Exception:
            pass

    @staticmethod
    def _hex_rgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
        )

    def _annotated_pixmap(
        self,
        frame,
        point: tuple[int, int],
        boxes: list[tuple[tuple[int, int, int, int], str, str]] | None = None,
        draw_boxes: bool = True,
    ) -> QPixmap:
        pil = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB").copy()
        draw = ImageDraw.Draw(pil)
        if draw_boxes:
            for rect, label, color in boxes or []:
                x, y, w, h = rect
                rgb = self._hex_rgb(color)
                draw.rectangle((x, y, x + w, y + h), outline=rgb, width=4)
                draw.text((x + 4, max(0, y - 16)), label, fill=rgb)
        x, y = point
        radius = 30
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 60, 60),
            width=4,
        )
        draw.line((x - radius - 12, y, x + radius + 12, y), fill=(255, 60, 60), width=2)
        draw.line((x, y - radius - 12, x, y + radius + 12), fill=(255, 60, 60), width=2)
        return self._pil_to_pixmap(pil)

    def _show_observation(self) -> None:
        if self.simulation_frame is None or self.simulation_point is None:
            return
        marked = self._annotated_pixmap(
            self.simulation_frame, self.simulation_point, self.simulation_boxes, True
        )
        clean = self._annotated_pixmap(
            self.simulation_frame, self.simulation_point, self.simulation_boxes, False
        )
        title = "观测点击点"
        if self.simulation_description:
            title += f" · {self.simulation_description}"
        ImagePreviewDialog(title, marked, clean, self).exec()

    def _clear_test_cache(self) -> None:
        directory = config.similarity_cache_dir()
        files = [path for path in directory.glob("*") if path.is_file()]
        count = len(files)
        for path in files:
            try:
                path.unlink()
            except OSError:
                pass
        self.simulation_frame = None
        self.simulation_point = None
        self.simulation_description = ""
        self.simulation_boxes = []
        self.observe_btn.setVisible(False)
        if count:
            self.test_label.setText(f"已清理 {count} 张测试缓存图片。")
            QMessageBox.information(self, "清理测试缓存", f"已删除 {count} 张测试缓存图片。")
        else:
            self.test_label.setText("没有可清理的测试缓存。")
        self.status_message.emit(f"测试缓存清理完成：{count} 张")

    def _save(self, silent: bool = False) -> None:
        if self.node is None or self.node.kind not in ("image", "action", "detect"):
            return
        groups = ["roi"]
        if self.node.kind != "detect" and self.node.roi is not None:
            groups.append("non_roi")
        if self.node.has_target or self.node.relative_roi is not None:
            groups.append("target")
        titles = {
            "roi": "检测重点 MR",
            "non_roi": "剩余部分 MR非",
            "target": "点击对象 MR",
        }
        for group_key in groups:
            ok, reason = self._balance_group(group_key, changed_key=None)
            if not ok:
                QMessageBox.warning(
                    self,
                    "修改失败",
                    f"{titles[group_key]} 的 4 个成分权重之和不能超过 1。\n"
                    f"原因：{reason}。\n"
                    "请解锁部分成分或降低权重后再保存。",
                )
                return
        self._refresh_sum_labels()
        model = self._collect_model()
        node = self.db.get_node(self.node.id)
        if node is None:
            return
        action = dict(node.action or {})
        action["similarity_model"] = model
        detect = node.kind == "detect"
        self.db.update_node(
            node.id,
            threshold=float(self.threshold_spin.value()),
            roi_weight=1.0 if detect else float(self.roi_weight_spin.value()),
            action=action,
            kind="image" if node.kind == "action" else node.kind,
        )
        self._dirty = False
        self.node = self.db.get_node(node.id)
        self.flow_workspace._reload_graph()
        try:
            self.flow_workspace.inspector.load_node(node.id, force=True)
        except Exception:
            pass
        self.flow_workspace.flow_changed.emit()
        if not silent:
            self.status_message.emit(f"相似度已保存：{node.name}")

    def _reset(self) -> None:
        if self.node is None:
            return
        model = similarity.default_model(
            "detect" if self.node.kind == "detect" else "image",
            self.node.roi_weight,
            bool(self.node.has_target or self.node.relative_roi is not None),
        )
        self.component_locks = set()
        self._update_lock_status()
        self._apply_model(model)
        self.threshold_spin.setValue(self.node.threshold)
        self._dirty = True
        self._schedule_preview()
        self.status_message.emit("已恢复默认相似度权重")

    def _back(self) -> None:
        if self._dirty:
            self._save(silent=True)
        self.back_requested.emit()
