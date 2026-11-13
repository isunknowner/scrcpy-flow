from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from PIL import Image
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import config, vision
from ..db import Database
from ..device import (
    AdbDevice,
    DeviceController,
    DryRunDevice,
    list_adb_devices,
    ui_selector_from_element,
)
from ..engine import RunEngine
from ..project_runner import ProjectRunner
from .recording_review import RecordingReviewDialog
from .scrollbar_bar import make_scrollable_bar


def _target_from_combo(combo: QComboBox):
    data = combo.currentData()
    if not data:
        return None
    kind, _, raw = str(data).partition(":")
    try:
        return kind, int(raw)
    except ValueError:
        return None


def _format_event(event: dict) -> str:
    kind = event.get("type")
    if kind == "match":
        return (
            f"匹配「{event.get('state_name', '')}」 "
            f"相似度={float(event.get('similarity', 0) or 0):.3f} "
            f"动作={event.get('action', '')}"
        )
    if kind == "no_match":
        similarity = event.get("similarity")
        text = "-" if similarity is None else f"{float(similarity):.3f}"
        return f"未匹配 重试 {event.get('retries')}/{event.get('max_retries')} 最高相似度={text}"
    if kind == "fallback":
        return f"执行保底 {event.get('index')}/{event.get('max_fallbacks')}"
    if kind == "block_start":
        return f"区块开始：{event.get('name', '')} ({event.get('alias', '')})"
    if kind == "block_finish":
        result = event.get("result")
        if result is not None:
            return f"区块结束：{result.name} {result.status} 步数={result.steps}"
        return "区块结束"
    if kind == "warn":
        return f"提示：{event.get('message', '')}"
    if kind == "error":
        return f"错误：{event.get('message', '')}"
    if kind in ("finish", "project_finish"):
        return (
            f"结束：{event.get('status', '')} 步数={event.get('steps', '')} "
            f"保底={event.get('fallbacks', '')}"
        )
    return str(event)


class BatchRunWidget(QWidget):
    status_message = Signal(str)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.stop_event = threading.Event()
        self.active: dict[str, object] = {}
        self.active_lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self._build()
        self.refresh_targets()
        self.refresh_devices()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(150)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("运行对象"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(280)
        bar.addWidget(self.target_combo)
        self.refresh_target_btn = QPushButton("刷新")
        self.refresh_target_btn.clicked.connect(self.refresh_targets)
        bar.addWidget(self.refresh_target_btn)
        self.dry_check = QCheckBox("模拟运行")
        bar.addWidget(self.dry_check)
        bar.addWidget(QLabel("最大步数"))
        self.max_steps = QSpinBox()
        self.max_steps.setRange(1, 100000)
        self.max_steps.setValue(300)
        bar.addWidget(self.max_steps)
        bar.addWidget(QLabel("并发设备"))
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 8)
        self.concurrency.setValue(2)
        bar.addWidget(self.concurrency)
        bar.addStretch(1)
        self.start_btn = QPushButton("开始批量运行")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        bar.addWidget(self.start_btn)
        bar.addWidget(self.stop_btn)
        layout.addWidget(make_scrollable_bar(bar))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(QLabel("选择设备"))
        self.refresh_device_btn = QPushButton("刷新设备")
        self.select_all_btn = QPushButton("全选")
        self.select_none_btn = QPushButton("全不选")
        head.addStretch(1)
        head.addWidget(self.select_all_btn)
        head.addWidget(self.select_none_btn)
        head.addWidget(self.refresh_device_btn)
        left_layout.addLayout(head)
        self.devices = QListWidget()
        left_layout.addWidget(self.devices, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("批量运行日志"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        right_layout.addWidget(self.log, 1)
        splitter.addWidget(right)
        splitter.setSizes([320, 820])

        self.refresh_device_btn.clicked.connect(self.refresh_devices)
        self.select_all_btn.clicked.connect(lambda: self._set_all(True))
        self.select_none_btn.clicked.connect(lambda: self._set_all(False))
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def refresh_targets(self) -> None:
        current = self.target_combo.currentData()
        self.target_combo.clear()
        for project in self.db.list_projects():
            self.target_combo.addItem(f"项目：{project.name}", f"project:{project.id}")
        for flow in self.db.list_flows():
            self.target_combo.addItem(f"流程：{flow.name}", f"flow:{flow.id}")
        if current is not None:
            index = self.target_combo.findData(current)
            if index >= 0:
                self.target_combo.setCurrentIndex(index)

    def refresh_devices(self) -> None:
        checked = {
            self.devices.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.devices.count())
            if self.devices.item(row).checkState() == Qt.CheckState.Checked
        }
        self.devices.clear()
        for device in list_adb_devices():
            if device["state"] != "device":
                continue
            model = f" · {device['model']}" if device["model"] else ""
            item = QListWidgetItem(f"{device['serial']}{model}")
            item.setData(Qt.ItemDataRole.UserRole, device["serial"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if not checked or device["serial"] in checked
                else Qt.CheckState.Unchecked
            )
            self.devices.addItem(item)

    def _set_all(self, checked: bool) -> None:
        for row in range(self.devices.count()):
            self.devices.item(row).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )

    def _selected_serials(self) -> list[str]:
        serials = []
        for row in range(self.devices.count()):
            item = self.devices.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                serials.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return serials

    def start(self) -> None:
        if self.running:
            return
        target = _target_from_combo(self.target_combo)
        if target is None:
            QMessageBox.warning(self, "提示", "请选择项目或流程")
            return
        serials = self._selected_serials()
        if not serials:
            QMessageBox.warning(self, "提示", "请至少勾选一台设备")
            return
        kind, target_id = target
        if kind == "flow" and not self.db.list_nodes(target_id):
            QMessageBox.warning(self, "提示", "该流程还没有节点")
            return
        if kind == "project" and not self.db.list_project_blocks(target_id):
            QMessageBox.warning(self, "提示", "该项目还没有区块")
            return
        self.log.clear()
        self.running = True
        self.stop_event.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker = threading.Thread(
            target=self._run_all,
            args=(serials, kind, target_id, self.dry_check.isChecked(), self.max_steps.value(), self.concurrency.value()),
            daemon=True,
        )
        self.worker.start()
        self.status_message.emit(f"批量运行开始：{len(serials)} 台设备")

    def stop(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        with self.active_lock:
            runners = list(self.active.values())
        for runner in runners:
            try:
                runner.stop()
            except Exception:
                pass
        self._append("已请求停止所有设备…")

    def _run_all(self, serials, kind, target_id, dry, max_steps, concurrency) -> None:
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
            futures = [
                pool.submit(self._run_one, serial, kind, target_id, dry, max_steps)
                for serial in serials
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    self.queue.put(("log", "", f"设备任务异常：{exc}"))
        self.queue.put(("finished", "", ""))

    def _run_one(self, serial: str, kind: str, target_id: int, dry: bool, max_steps: int) -> None:
        if self.stop_event.is_set():
            return
        device = None
        try:
            self.queue.put(("log", serial, "连接设备…"))
            device = DeviceController(
                serial=serial,
                capture="adb",
                input_mode="adb",
                manage_scrcpy=False,
            )
            device.start()
            inner = DryRunDevice(device) if dry else device

            def callback(event):
                self.queue.put(("log", serial, _format_event(event)))

            if kind == "project":
                runner = ProjectRunner(
                    self.db,
                    target_id,
                    inner,
                    event_cb=callback,
                    max_steps=max_steps,
                )
            else:
                runner = RunEngine(
                    self.db,
                    target_id,
                    inner,
                    event_cb=callback,
                    max_steps=max_steps,
                )
            with self.active_lock:
                self.active[serial] = runner
            result = runner.run()
            self.queue.put(
                (
                    "log",
                    serial,
                    f"运行结束：{result.status} 步数={result.steps} 保底={result.fallbacks}",
                )
            )
        except Exception as exc:
            self.queue.put(("log", serial, f"运行失败：{exc}"))
        finally:
            with self.active_lock:
                self.active.pop(serial, None)
            if device is not None:
                try:
                    device.stop()
                except Exception:
                    pass

    def _poll(self) -> None:
        try:
            while True:
                kind, serial, text = self.queue.get_nowait()
                if kind == "finished":
                    self.running = False
                    self.start_btn.setEnabled(True)
                    self.stop_btn.setEnabled(False)
                    self._append("批量运行结束")
                    self.status_message.emit("批量运行结束")
                    continue
                prefix = f"[{serial}] " if serial else ""
                self._append(prefix + text)
        except queue.Empty:
            pass

    def _append(self, text: str) -> None:
        self.log.appendPlainText(text)


class SmartLocatorWidget(QWidget):
    status_message = Signal(str)

    def __init__(self, db: Database, flow_workspace, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_workspace = flow_workspace
        self.elements: list[dict] = []
        self._build()
        self.refresh_devices()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("设备"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(240)
        bar.addWidget(self.device_combo)
        self.refresh_btn = QPushButton("刷新设备")
        bar.addWidget(self.refresh_btn)
        bar.addWidget(QLabel("OCR 语言"))
        self.ocr_lang = QLineEdit("chi_sim+eng")
        self.ocr_lang.setMaximumWidth(120)
        bar.addWidget(self.ocr_lang)
        bar.addStretch(1)
        self.tree_btn = QPushButton("读取控件树")
        self.tree_btn.setObjectName("primary")
        self.ocr_btn = QPushButton("OCR 识别文字")
        bar.addWidget(self.tree_btn)
        bar.addWidget(self.ocr_btn)
        layout.addWidget(make_scrollable_bar(bar))

        hint = QLabel(
            "控件树通过 uiautomator 读取，不依赖图片；OCR 需要安装 pytesseract 与 tesseract。"
            "选中元素后可测试点击，或写入当前节点的点击点。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("识别结果"))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["来源", "类型", "文本", "描述", "资源 ID", "中心"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        left_layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.test_btn = QPushButton("点击测试")
        self.apply_btn = QPushButton("写入当前节点")
        self.apply_btn.setObjectName("primary")
        buttons.addWidget(self.test_btn)
        buttons.addWidget(self.apply_btn)
        buttons.addStretch(1)
        left_layout.addLayout(buttons)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("屏幕预览"))
        self.preview = QLabel("读取控件树或 OCR 后显示截图")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(240, 300)
        self.preview.setStyleSheet(
            "background:#0d0f13; border:1px solid #333842; color:#5c6575;"
        )
        right_layout.addWidget(self.preview, 1)
        splitter.addWidget(right)
        splitter.setSizes([700, 360])

        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.tree_btn.clicked.connect(self.capture_tree)
        self.ocr_btn.clicked.connect(self.capture_ocr)
        self.test_btn.clicked.connect(self.test_click)
        self.apply_btn.clicked.connect(self.apply_to_node)

    def refresh_devices(self) -> None:
        current = self.device_combo.currentData()
        self.device_combo.clear()
        self.device_combo.addItem("（自动选择）", None)
        for device in list_adb_devices():
            if device["state"] != "device":
                continue
            model = f" · {device['model']}" if device["model"] else ""
            self.device_combo.addItem(f"{device['serial']}{model}", device["serial"])
        if current is not None:
            index = self.device_combo.findData(current)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

    def _device(self) -> AdbDevice:
        return AdbDevice(serial=self.device_combo.currentData())

    def _show_preview(self, frame) -> None:
        if frame is None:
            return
        frame = frame.copy()
        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            3 * width,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def _fill(self, elements: list[dict], source: str) -> None:
        self.elements = elements
        self.table.setRowCount(len(elements))
        for row, element in enumerate(elements):
            values = [
                source,
                str(element.get("class", "")).split(".")[-1],
                element.get("text", ""),
                element.get("description", ""),
                element.get("resource_id", ""),
                f"{element.get('center', ['', ''])[0]}, {element.get('center', ['', ''])[1]}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, element)
                self.table.setItem(row, column, item)
        self.status_message.emit(f"识别到 {len(elements)} 个元素（{source}）")
        if elements:
            self.table.selectRow(0)

    def capture_tree(self) -> None:
        try:
            device = self._device()
            frame = device.screenshot()
            elements = device.ui_elements()
        except Exception as exc:
            QMessageBox.critical(self, "读取控件树失败", str(exc))
            return
        self._show_preview(frame)
        self._fill(elements, "控件树")

    def capture_ocr(self) -> None:
        try:
            import pytesseract
        except ImportError:
            QMessageBox.information(
                self,
                "缺少 OCR 依赖",
                "OCR 需要安装 pytesseract 和 tesseract：\n\n"
                "pip install pytesseract\n"
                "Linux: sudo apt install tesseract-ocr tesseract-ocr-chi-sim\n"
                "macOS: brew install tesseract tesseract-lang\n"
                "Windows: 安装 Tesseract-OCR 并加入 PATH",
            )
            return
        try:
            device = self._device()
            frame = device.screenshot()
            language = self.ocr_lang.text().strip() or "eng"
            try:
                data = pytesseract.image_to_data(
                    Image.fromarray(frame),
                    lang=language,
                    output_type=pytesseract.Output.DICT,
                )
            except Exception:
                data = pytesseract.image_to_data(
                    Image.fromarray(frame),
                    lang="eng",
                    output_type=pytesseract.Output.DICT,
                )
        except Exception as exc:
            QMessageBox.critical(self, "OCR 识别失败", str(exc))
            return
        self._show_preview(frame)
        elements = []
        for index in range(len(data.get("text", []))):
            text = str(data["text"][index]).strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1
            if confidence < 35:
                continue
            x = int(data["left"][index])
            y = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
            if width <= 0 or height <= 0:
                continue
            elements.append(
                {
                    "index": index,
                    "class": "OCR",
                    "text": text,
                    "description": "",
                    "resource_id": "",
                    "clickable": True,
                    "enabled": True,
                    "bounds": [x, y, x + width, y + height],
                    "center": [x + width // 2, y + height // 2],
                    "confidence": confidence,
                }
            )
        self._fill(elements, "OCR")

    def _selected_element(self) -> dict | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def test_click(self) -> None:
        element = self._selected_element()
        if element is None:
            QMessageBox.information(self, "提示", "请先选择一个识别结果")
            return
        try:
            device = self._device()
            center = element.get("center", [0, 0])
            device.tap(center[0], center[1])
        except Exception as exc:
            QMessageBox.critical(self, "点击失败", str(exc))
            return
        self.status_message.emit(f"已点击 {element.get('center')}")

    def apply_to_node(self) -> None:
        element = self._selected_element()
        if element is None:
            QMessageBox.information(self, "提示", "请先选择一个识别结果")
            return
        inspector = self.flow_workspace.inspector
        node = inspector.node
        if node is None:
            QMessageBox.information(self, "提示", "请先在流程画布中选中一个节点")
            return
        source = self.table.item(self.table.currentRow(), 0).text()
        try:
            device = self._device()
            width, height = device.screen_size()
        except Exception as exc:
            QMessageBox.critical(self, "读取屏幕尺寸失败", str(exc))
            return
        try:
            inspector.flush()
            fresh = self.db.get_node(node.id)
            action = dict(fresh.action or {})
            if source == "控件树":
                selector = ui_selector_from_element(element)
                if selector:
                    action["ui_selector"] = selector
                else:
                    action.pop("ui_selector", None)
            else:
                action.pop("ui_selector", None)
            center_x, center_y = element.get("center", [0, 0])
            self.db.update_node(
                node.id,
                click_x=float(center_x) / max(1, width),
                click_y=float(center_y) / max(1, height),
                action=action,
                action_type="tap" if fresh.action_type == "none" else fresh.action_type,
            )
        except Exception as exc:
            QMessageBox.critical(self, "写入节点失败", str(exc))
            return
        inspector.load_node(node.id, force=True)
        self.flow_workspace._reload_graph()
        self.flow_workspace.flow_changed.emit()
        self.status_message.emit(f"已写入节点「{node.name}」的点击位置")


class RecorderWidget(QWidget):
    status_message = Signal(str)

    def __init__(self, db: Database, flow_workspace, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_workspace = flow_workspace
        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.device: DeviceController | None = None
        self.listener = None
        self.capture_thread: threading.Thread | None = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.record_count = 0
        self.record_flow_id: int | None = None
        self.recorded_nodes: list[int] = []
        self.last_recorded_id: int | None = None
        self._build()
        self.refresh_devices()
        self.refresh_flows()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(120)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("设备"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(220)
        bar.addWidget(self.device_combo)
        self.refresh_device_btn = QPushButton("刷新设备")
        bar.addWidget(self.refresh_device_btn)
        bar.addWidget(QLabel("录制到流程"))
        self.flow_combo = QComboBox()
        self.flow_combo.setMinimumWidth(220)
        bar.addWidget(self.flow_combo)
        self.refresh_flow_btn = QPushButton("刷新流程")
        bar.addWidget(self.refresh_flow_btn)
        bar.addStretch(1)
        self.start_btn = QPushButton("开始录制")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QPushButton("停止录制")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.review_btn = QPushButton("编辑录制结果")
        self.review_btn.setEnabled(False)
        bar.addWidget(self.start_btn)
        bar.addWidget(self.stop_btn)
        bar.addWidget(self.review_btn)
        layout.addWidget(make_scrollable_bar(bar))

        hint = QLabel(
            "录制会启动一个受控 scrcpy 窗口；你在窗口中的每次左键点击都会生成一个流程节点。"
            "需要安装 pynput；macOS 需要辅助功能权限，Linux 需要 X11 会话。录制为实验功能，建议每步操作后稍作停顿。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.status = QLabel("未开始录制")
        layout.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        layout.addWidget(self.log, 1)

        self.refresh_device_btn.clicked.connect(self.refresh_devices)
        self.refresh_flow_btn.clicked.connect(self.refresh_flows)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.review_btn.clicked.connect(self.edit_result)

    def refresh_devices(self) -> None:
        current = self.device_combo.currentData()
        self.device_combo.clear()
        self.device_combo.addItem("（自动选择）", None)
        for device in list_adb_devices():
            if device["state"] != "device":
                continue
            model = f" · {device['model']}" if device["model"] else ""
            self.device_combo.addItem(f"{device['serial']}{model}", device["serial"])
        if current is not None:
            index = self.device_combo.findData(current)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

    def refresh_flows(self) -> None:
        current = self.flow_combo.currentData()
        self.flow_combo.clear()
        for flow in self.db.list_flows():
            self.flow_combo.addItem(flow.name, flow.id)
        if current is not None:
            index = self.flow_combo.findData(current)
            if index >= 0:
                self.flow_combo.setCurrentIndex(index)

    def start(self) -> None:
        if self.running:
            return
        flow_id = self.flow_combo.currentData()
        if flow_id is None:
            QMessageBox.warning(self, "提示", "请选择要录制到的流程")
            return
        try:
            from pynput import mouse  # noqa: F401
        except ImportError:
            QMessageBox.information(
                self,
                "缺少录制依赖",
                "录制需要 pynput：\n\npip install -r requirements-pro.txt\n"
                "macOS 还需要在系统设置中授予辅助功能权限。",
            )
            return
        self.running = True
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.record_count = 0
        self.record_flow_id = int(flow_id)
        self.recorded_nodes = []
        self.last_recorded_id = None
        self.review_btn.setEnabled(False)
        self.log.clear()
        self.status.setText("正在启动 scrcpy 窗口…")
        serial = self.device_combo.currentData()
        threading.Thread(
            target=self._prepare,
            args=(serial, int(flow_id)),
            daemon=True,
        ).start()

    def _prepare(self, serial, flow_id: int) -> None:
        try:
            device = DeviceController(
                serial=serial,
                capture="adb",
                input_mode="adb",
                manage_scrcpy=True,
            )
            device.start()
            if not self.running:
                device.stop()
                return
            if device.window is None or not device.window.ready:
                raise RuntimeError("无法控制 scrcpy 窗口，录制需要窗口模式")
            self.device = device
            self.latest_frame = None
            self.capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self.capture_thread.start()
            self._start_listener()
            self.queue.put(("ready", "", f"录制已开始，目标流程 ID={flow_id}"))
        except Exception as exc:
            self.queue.put(("error", "", str(exc)))

    def _start_listener(self) -> None:
        from pynput import mouse

        def on_click(x, y, button, pressed):
            if not pressed or button != mouse.Button.left or not self.running:
                return
            device = self.device
            if device is None or device.window is None:
                return
            try:
                origin_x, origin_y, win_w, win_h = device.window._refresh_geometry()
                dev_w, dev_h = device.screen_size()
            except Exception:
                return
            if not (origin_x <= x <= origin_x + win_w and origin_y <= y <= origin_y + win_h):
                return
            scale = min(win_w / max(1, dev_w), win_h / max(1, dev_h))
            video_w, video_h = dev_w * scale, dev_h * scale
            offset_x = (win_w - video_w) / 2
            offset_y = (win_h - video_h) / 2
            device_x = (x - origin_x - offset_x) / max(1e-6, scale)
            device_y = (y - origin_y - offset_y) / max(1e-6, scale)
            if not (0 <= device_x <= dev_w and 0 <= device_y <= dev_h):
                return
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            self.queue.put(
                (
                    "record",
                    "",
                    (device_x / max(1, dev_w), device_y / max(1, dev_h), frame),
                )
            )

        self.listener = mouse.Listener(on_click=on_click)
        self.listener.start()

    def _capture_loop(self) -> None:
        while self.running:
            device = self.device
            if device is None:
                break
            try:
                frame = device.screenshot()
                with self.frame_lock:
                    self.latest_frame = frame
            except Exception:
                pass
            time.sleep(0.6)

    def stop(self) -> None:
        if not self.running:
            return
        self._finish()
        if self.recorded_nodes:
            self.status.setText(
                f"录制已停止，共 {len(self.recorded_nodes)} 个节点；"
                "可点击「编辑录制结果」调整顺序和内容"
            )
        else:
            self.status.setText("录制已停止")
        self.status_message.emit("录制已停止")

    def edit_result(self) -> None:
        if self.record_flow_id is None or not self.recorded_nodes:
            QMessageBox.information(self, "提示", "还没有录制节点")
            return
        self.flow_workspace.inspector.flush()
        dialog = RecordingReviewDialog(
            self.db,
            self.record_flow_id,
            list(self.recorded_nodes),
            self.flow_workspace,
            self,
        )
        dialog.exec()
        self.status_message.emit("录制结果已更新")

    def _finish(self) -> None:
        self.running = False
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception:
                pass
            self.listener = None
        thread = self.capture_thread
        self.capture_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        device = self.device
        self.device = None
        if device is not None:
            try:
                device.stop()
            except Exception:
                pass
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.review_btn.setEnabled(bool(self.recorded_nodes))

    def _poll(self) -> None:
        try:
            while True:
                kind, _serial, payload = self.queue.get_nowait()
                if kind == "ready":
                    self.status.setText("正在录制：在 scrcpy 窗口中点击手机画面")
                    self.log.appendPlainText(str(payload))
                elif kind == "error":
                    self.log.appendPlainText(f"录制失败：{payload}")
                    QMessageBox.critical(self, "录制失败", str(payload))
                    self._finish()
                elif kind == "record":
                    self._add_node(*payload)
        except queue.Empty:
            pass

    def _add_node(self, rel_x: float, rel_y: float, frame) -> None:
        if frame is None:
            self.log.appendPlainText("已忽略一次点击：还没有抓取到画面")
            return
        flow_id = self.record_flow_id
        if flow_id is None:
            return
        self.record_count += 1
        temp = config.images_dir() / f"rec_{uuid4().hex[:10]}.png"
        try:
            vision.save_rgb(temp, frame)
            self.flow_workspace.inspector.flush()
            if self.flow_workspace.flow_id != int(flow_id):
                self.flow_workspace.refresh_flows(select_id=int(flow_id))
                self.flow_workspace.load_flow(int(flow_id))
            node_id = self.flow_workspace.add_node_from_path(
                str(temp),
                kind="image",
                name=f"录制 {self.record_count}",
            )
            if node_id is not None:
                self.db.update_node(
                    node_id,
                    click_x=float(rel_x),
                    click_y=float(rel_y),
                    action_type="tap",
                    action={"post_wait": 0.6},
                )
                if self.last_recorded_id is not None:
                    try:
                        self.db.add_edge(
                            int(flow_id),
                            self.last_recorded_id,
                            node_id,
                            1.0,
                            "录制",
                        )
                    except Exception as exc:
                        self.log.appendPlainText(f"自动连线失败：{exc}")
                self.last_recorded_id = node_id
                self.recorded_nodes.append(node_id)
                self.review_btn.setEnabled(True)
                self.flow_workspace.inspector.load_node(node_id, force=True)
                self.flow_workspace._reload_graph()
                self.flow_workspace.flow_changed.emit()
                self.log.appendPlainText(
                    f"已录制节点 {self.record_count}：点击 ({rel_x:.3f}, {rel_y:.3f})"
                )
        except Exception as exc:
            self.log.appendPlainText(f"录制节点失败：{exc}")
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


class ProWorkspace(QWidget):
    status_message = Signal(str)

    def __init__(self, db: Database, flow_workspace, parent=None):
        super().__init__(parent)
        self.db = db
        self.flow_workspace = flow_workspace
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.batch = BatchRunWidget(db)
        self.locator = SmartLocatorWidget(db, flow_workspace)
        self.recorder = RecorderWidget(db, flow_workspace)
        self.tabs.addTab(self.batch, "批量设备")
        self.tabs.addTab(self.locator, "智能定位")
        self.tabs.addTab(self.recorder, "录制回放")
        layout.addWidget(self.tabs)
        for child in (self.batch, self.locator, self.recorder):
            child.status_message.connect(self.status_message)

    def refresh_all(self) -> None:
        self.batch.refresh_targets()
        self.batch.refresh_devices()
        self.locator.refresh_devices()
        self.recorder.refresh_devices()
        self.recorder.refresh_flows()

    def stop_all(self) -> None:
        self.batch.stop()
        self.recorder.stop()
