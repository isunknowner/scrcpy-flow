from __future__ import annotations

import random
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import ai as ai_module
from . import config, similarity as sim_model, vision
from .db import Database
from .embeddings import (
    blob_to_embed,
    cosine,
    embed_to_blob,
    get_embedder,
)
from .models import Edge, EngineResult, Flow, Match, Node
from .ollama import OllamaError, load_settings

DEFAULT_FALLBACK = {
    "max_retries": 2,
    "threshold_step": 0.04,
    "min_threshold": 0.55,
    "calibrated_step": 0.02,
    "calibrated_relax": 0.05,
    "target_min_mr": 0.4,
    "rel_min_mr": 0.4,
    "max_fallbacks": 3,
    "retry_wait": 0.6,
    "steps": [
        {"type": "key", "key": "back", "wait": 0.8},
        {"type": "wait", "seconds": 1.0},
    ],
}

DEFAULT_SETTINGS = {
    "beta": 0.12,
    "stay_penalty": 0.06,
    "match_scope": "next",
    "only_reachable": False,
    "selection_mode": "best",
    "min_margin": 0.0,
    "save_screenshots": True,
    "max_run_seconds": 0,
    "fallback": DEFAULT_FALLBACK,
}


def merged_settings(flow: Flow) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    for key, value in (flow.settings or {}).items():
        if key == "fallback" and isinstance(value, dict):
            fallback = dict(DEFAULT_FALLBACK)
            fallback.update(value)
            if "steps" not in value:
                fallback["steps"] = list(DEFAULT_FALLBACK["steps"])
            settings["fallback"] = fallback
        else:
            settings[key] = value
    if "fallback" not in settings:
        settings["fallback"] = dict(DEFAULT_FALLBACK)
    if "match_scope" not in (flow.settings or {}):
        settings["match_scope"] = (
            "reachable" if (flow.settings or {}).get("only_reachable") else "next"
        )
    return settings


def screen_point(node: Node, shot: np.ndarray, node_image: np.ndarray | None) -> tuple:
    click = (node.click_x, node.click_y) if node.has_click_point else None
    if node_image is not None:
        size = (int(node_image.shape[1]), int(node_image.shape[0]))
    elif node.img_w and node.img_h:
        size = (int(node.img_w), int(node.img_h))
    else:
        size = None
    return vision.resolve_point(click, size, shot, node_image)


def _relative(value: float, total: int) -> float:
    return float(value) * total if 0.0 <= float(value) <= 1.0 else float(value)


def _region_center(roi, width: int, height: int) -> tuple[int, int]:
    x = int(round((float(roi[0]) + float(roi[2]) / 2.0) * width))
    y = int(round((float(roi[1]) + float(roi[3]) / 2.0) * height))
    return (max(0, min(width - 1, x)), max(0, min(height - 1, y)))


class RunEngine:
    def __init__(
        self,
        db: Database,
        flow_id: int,
        device,
        event_cb: Callable[[dict], None] | None = None,
        embedder=None,
        max_steps: int = 500,
        brain=None,
        speed: float = 1.0,
        skip_waits: bool = False,
        swipe_repeat: int = 1,
    ):
        self.db = db
        self.flow_id = flow_id
        self.device = device
        self.event_cb = event_cb
        self.embedder = embedder or get_embedder()
        self.max_steps = max_steps
        self._brain_override = brain
        # 加速运行：所有等待按倍率缩短；极速模式直接跳过等待
        self.speed = max(0.1, float(speed or 1.0))
        self.skip_waits = bool(skip_waits)
        self.swipe_repeat = max(1, int(swipe_repeat or 1))

        self.flow: Flow | None = None
        self.nodes: list[Node] = []
        self.image_nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.embeddings: dict[int, np.ndarray] = {}
        self.node_images: dict[int, np.ndarray] = {}
        self.node_roi_embeddings: dict[int, np.ndarray] = {}
        self._live_roi_cache: dict[tuple, np.ndarray] = {}
        self.settings: dict = {}
        self.fallback: dict = {}
        self.max_run_seconds = 0.0
        self._start_time = time.monotonic()

        self.current_state: int | None = None
        self.session_id: int | None = None
        self.log_index = 0
        self.steps = 0
        self.fallbacks = 0
        self.retries = 0
        self.error: str | None = None
        self.finished = False
        self._stop = threading.Event()
        self._screenshot_dir: Path | None = None
        self._warned: set[str] = set()
        # 人工介入（运行面板「人工选择」按钮）
        self._manual_pending = False
        self._human_result: dict | None = None
        self._human_event = threading.Event()

        # 本地 Ollama 大脑（可选；关闭时完全是原来的图片匹配流程）
        self.ai_settings = load_settings()
        self.brain: ai_module.ScreenBrain | None = None
        self._ai_done: set[int] = set()
        self._ai_cycles: dict[int, int] = {}
        self._ai_history: list[dict] = []
        self._ai_last_signature: dict[int, np.ndarray] = {}
        self._ai_last_action: dict[int, tuple] = {}

    def emit(self, event_type: str, **payload) -> None:
        if not self.event_cb:
            return
        event = {"type": event_type, "time": time.time(), **payload}
        try:
            self.event_cb(event)
        except Exception:
            pass

    def warn(self, message: str) -> None:
        if message in self._warned:
            return
        self._warned.add(message)
        self.emit("warn", message=message)

    def _ai_log(self, message: str) -> None:
        self.emit("ai_log", message=message)

    def stop(self) -> None:
        self._stop.set()
        # 唤醒正在等待人工输入的线程
        self._human_event.set()

    # ------------------------------------------------------------- 人工介入
    def request_manual(self) -> None:
        """界面按下「人工选择」：下一步暂停并请求人工动作。"""
        self._manual_pending = True

    def provide_human(self, result: dict | None) -> None:
        """界面提交人工选择（UI 线程调用）。"""
        self._human_result = result
        self._human_event.set()

    def _request_human(
        self,
        shot: np.ndarray,
        context: str,
        allow_ai: bool = True,
        extra: dict | None = None,
    ) -> dict | None:
        self._human_result = None
        self._human_event.clear()
        self.emit(
            "need_human",
            frame=shot,
            context=context,
            allow_ai=allow_ai,
            **(extra or {}),
        )
        while not self._human_event.wait(0.2):
            if self._stop.is_set():
                return None
        if self._stop.is_set():
            return None
        return self._human_result

    @staticmethod
    def _manual_name(action: dict) -> str:
        kind = str(action.get("kind") or "tap")
        if kind == "tap":
            return f"点击 ({float(action.get('x', 0.5)):.2f},{float(action.get('y', 0.5)):.2f})"
        if kind == "swipe":
            return f"滑动 {action.get('direction') or 'up'}"
        if kind == "key":
            return f"按键 {action.get('key') or 'back'}"
        if kind == "text":
            return f"输入 {str(action.get('text') or '')[:12]}"
        if kind == "launch":
            return f"打开 {str(action.get('package') or '')[:24]}"
        if kind == "wait":
            return f"等待 {float(action.get('seconds', 1) or 1):g}s"
        return "人工动作"

    def _perform_manual(self, action: dict, shot: np.ndarray) -> str:
        height, width = shot.shape[:2]
        kind = str(action.get("kind") or "tap")
        if kind == "tap":
            x = int(max(0.0, min(1.0, float(action.get("x", 0.5)))) * width)
            y = int(max(0.0, min(1.0, float(action.get("y", 0.5)))) * height)
            self.device.tap(x, y)
            return f"人工点击 ({x},{y})"
        if kind == "swipe":
            direction = str(action.get("direction") or "up")
            points = {
                "up": (0.5, 0.78, 0.5, 0.26),
                "down": (0.5, 0.26, 0.5, 0.78),
                "left": (0.82, 0.5, 0.18, 0.5),
                "right": (0.18, 0.5, 0.82, 0.5),
            }.get(direction, (0.5, 0.78, 0.5, 0.26))
            duration = int(action.get("duration", 200) or 200)
            self.device.swipe(
                int(points[0] * width),
                int(points[1] * height),
                int(points[2] * width),
                int(points[3] * height),
                duration,
            )
            return f"人工滑动 {direction}"
        if kind == "key":
            key = str(action.get("key") or "back")
            self.device.key(key)
            return f"人工按键 {key}"
        if kind == "text":
            text = str(action.get("text") or "")
            if action.get("clear"):
                clear = getattr(self.device, "clear_text", None)
                if clear:
                    clear()
            self.device.text(text)
            if action.get("submit"):
                self.device.key("enter")
            return f"人工输入 {text[:24]}"
        if kind == "launch":
            package = str(action.get("package") or "").strip()
            if not package:
                raise RuntimeError("打开应用缺少包名")
            launcher = getattr(self.device, "launch_app", None)
            if launcher is None:
                raise RuntimeError("当前设备不支持启动应用")
            launcher(package, bool(action.get("fresh", True)))
            return f"人工打开应用 {package}"
        if kind == "wait":
            seconds = float(action.get("seconds", 1) or 1)
            self._sleep(seconds)
            return f"人工等待 {seconds:g}s"
        return "人工动作（未知类型）"

    def _save_manual_node(self, action: dict) -> Node | None:
        """把人工选择记成流程里的 manual 节点，下次运行当作保底动作。"""
        stored = {
            key: value
            for key, value in action.items()
            if key not in ("remember", "monitor", "kind", "ai", "stop")
        }
        stored.update(
            {
                "manual": True,
                "kind": str(action.get("kind") or "tap"),
                "monitor": bool(action.get("monitor")),
            }
        )
        if action.get("kind") == "tap":
            stored["x"] = float(action.get("x", 0.5))
            stored["y"] = float(action.get("y", 0.5))
        name = "人工：" + self._manual_name(action)
        index = len(self.nodes)
        click_x = float(action.get("x", 0.5)) if action.get("kind") == "tap" else None
        click_y = float(action.get("y", 0.5)) if action.get("kind") == "tap" else None
        try:
            node_id = self.db.add_node(
                self.flow_id,
                name=name[:24],
                kind="manual",
                click_x=click_x,
                click_y=click_y,
                action_type=str(action.get("kind") or "tap"),
                action=stored,
                threshold=0.85,
                is_entry=False,
                is_end=False,
                x=float(70 + (index % 4) * 280),
                y=float(80 + (index // 4) * 170),
            )
        except Exception as exc:  # noqa: BLE001 - 保存失败不影响运行
            self.warn(f"保存人工动作失败：{exc}")
            return None
        node = self.db.get_node(node_id)
        if node is not None:
            self.nodes.append(node)
            self.emit("nodes_changed")
        return node

    def _monitor_manual(self, timeout: float = 6.0) -> bool:
        """人工动作后看画面是否恢复：能重新匹配到节点即视为达标。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return False
            self._sleep(0.35)
            try:
                shot = self.device.screenshot()
                matches = self.rank(shot, self.embedder.embed(shot))
            except Exception:
                return False
            if self._choose(matches) is not None:
                return True
        return False

    def _manual_node(self) -> Node | None:
        candidates = [
            node
            for node in self.nodes
            if node.kind == "manual" and (node.action or {}).get("manual")
        ]
        return max(candidates, key=lambda node: node.id) if candidates else None

    def _run_manual_action(self, action: dict, shot: np.ndarray, source: str) -> bool:
        description = self._perform_manual(action, shot)
        self.db.log_step(
            self.session_id,
            self.log_index,
            self.current_state,
            None,
            None,
            None,
            description,
            source,
            None,
        )
        self.emit("human_applied", action=description, action_data=action)
        if action.get("remember"):
            node = self._save_manual_node(action)
            if node is not None:
                self.emit("manual_saved", node_id=node.id, name=node.name)
        if not action.get("monitor"):
            return True
        if self._monitor_manual():
            return True
        self.emit("human_monitor_failed", action=description)
        retry = self._request_human(
            self.device.screenshot(),
            "手动动作后检测未达标：重新人工选择，或让 AI 托管",
            allow_ai=True,
            extra={"failed_action": description},
        )
        if retry is None:
            return True
        if retry.get("ai"):
            if self.brain is not None and self.ai_settings.ai_fallback:
                recovered = self._ai_recover(self.device.screenshot())
                if recovered is not None:
                    recovered_desc, stop_run = recovered
                    self.emit("human_applied", action=recovered_desc, ai=True)
                    if stop_run:
                        self.finished = True
                        self.emit("end", node=None)
                        return False
                    return True
            self.warn("AI 托管不可用（未启用或失败），已跳过")
            return True
        if retry.get("stop"):
            self.stop()
            return True
        return self._run_manual_action(retry, self.device.screenshot(), "manual")

    def prepare(self) -> None:
        flow = self.db.get_flow(self.flow_id)
        if flow is None:
            raise RuntimeError("流程不存在")
        self.flow = flow
        self.settings = merged_settings(flow)
        self.fallback = self.settings["fallback"]
        self.max_run_seconds = float(self.settings.get("max_run_seconds", 0) or 0)
        self._start_time = time.monotonic()
        self.nodes = self.db.list_nodes(self.flow_id)
        self.edges = self.db.list_edges(self.flow_id)
        self.image_nodes = [
            n
            for n in self.nodes
            if n.image_path or n.kind in ("nodetect", "if", "ai")
        ]
        if not self.image_nodes:
            raise RuntimeError("流程中没有可匹配或可执行的节点")

        self._ai_done.clear()
        self._ai_cycles.clear()
        self._ai_history.clear()
        self._ai_last_signature.clear()
        self._ai_last_action.clear()
        self.brain = None
        if self._brain_override is not None:
            self.brain = self._brain_override
        else:
            try:
                from . import cloud_ai

                candidate = cloud_ai.create_brain(self.ai_settings, log=self._ai_log)
                if candidate is not None and getattr(candidate, "enabled", False):
                    self.brain = candidate
            except Exception as exc:  # 大脑初始化不应拖垮整个运行
                self.warn(f"AI 初始化失败，将只用图片匹配：{exc}")
        if self.brain is None and any(
            node.kind == "ai" or node.action_type == "ai" for node in self.nodes
        ):
            raise RuntimeError(
                "流程里有 AI 决策节点，但 AI 未启用；请在「AI 助手（本地 Ollama）」或"
                "「AI 助手（云端 API）」面板里开启后端，或把这些节点改成普通图片节点"
            )

        stored = self.db.get_embeddings(self.flow_id)
        for node in self.image_nodes:
            embedding = None
            entry = stored.get(node.id)
            if entry:
                blob, kind, dim = entry
                if kind == self.embedder.kind and (dim == self.embedder.dim or self.embedder.dim == 0):
                    embedding = blob_to_embed(blob)
            if embedding is None and node.image_path and Path(node.image_path).exists():
                try:
                    image = vision.load_rgb(node.image_path)
                    self.node_images[node.id] = image
                    embedding = self.embedder.embed(image)
                    self.db.set_node_embedding(
                        node.id,
                        embed_to_blob(embedding),
                        self.embedder.kind,
                        int(embedding.shape[0]),
                    )
                except Exception as exc:
                    self.warn(f"节点 {node.name} 图片读取失败：{exc}")
            if embedding is None and entry:
                embedding = blob_to_embed(entry[0])
            if embedding is not None:
                self.embeddings[node.id] = embedding

        if not self.embeddings and not any(
            n.kind in ("nodetect", "if", "ai") for n in self.image_nodes
        ):
            raise RuntimeError("没有可用的图片向量，请检查节点图片")

        for node in self.nodes:
            if (
                node.kind in ("nodetect", "ai")
                and not node.is_entry
                and not any(edge.dst_id == node.id for edge in self.edges)
            ):
                self.warn(
                    f"不检测步骤「{node.name}」没有上一步连线，不会被执行；请用「连接 / 下一步」接入流程"
                )

        entry_node = next((n for n in self.nodes if n.is_entry), None)
        if entry_node is None:
            entry_node = next((n for n in self.image_nodes if n.id in self.embeddings), None)
        if entry_node is None and self.nodes:
            entry_node = self.nodes[0]
        self.current_state = entry_node.id if entry_node else None
        self.steps = 0
        self.fallbacks = 0
        self.retries = 0
        self.log_index = 0
        self.finished = False
        self.error = None
        self._stop.clear()

    def _reachable(self) -> set[int] | None:
        if self.current_state is None:
            return None
        outgoing: dict[int, list[int]] = defaultdict(list)
        for edge in self.edges:
            outgoing[edge.src_id].append(edge.dst_id)
        if not outgoing.get(self.current_state):
            return {self.current_state}
        seen = {self.current_state}
        stack = [self.current_state]
        while stack:
            current = stack.pop()
            for nxt in outgoing.get(current, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def _next_candidates(self) -> set[int] | None:
        if self.current_state is None:
            return None
        allowed = {self.current_state}
        for edge in self.edges:
            if edge.src_id == self.current_state:
                allowed.add(edge.dst_id)
        return allowed

    def _bonus(self, dst_id: int) -> float:
        if self.current_state is None:
            return 0.0
        weight = 0.0
        for edge in self.edges:
            if edge.src_id == self.current_state and edge.dst_id == dst_id:
                weight = max(weight, edge.weight)
        return weight

    @staticmethod
    def _crop_roi(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x = int(round(roi[0] * width))
        y = int(round(roi[1] * height))
        w = int(round(roi[2] * width))
        h = int(round(roi[3] * height))
        x = max(0, min(width - 2, x))
        y = max(0, min(height - 2, y))
        w = max(2, min(width - x, w))
        h = max(2, min(height - y, h))
        crop = frame[y : y + h, x : x + w]
        return crop if crop.size else None

    def _node_roi_embedding(self, node: Node) -> np.ndarray | None:
        if node.id in self.node_roi_embeddings:
            return self.node_roi_embeddings[node.id]
        roi = node.roi
        image = self._node_image(node)
        if roi is None or image is None:
            return None
        crop = self._crop_roi(image, roi)
        if crop is None:
            return None
        vector = self.embedder.embed(crop)
        self.node_roi_embeddings[node.id] = vector
        return vector

    def _live_roi_embedding(self, shot: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray | None:
        key = tuple(round(value, 4) for value in roi)
        if key in self._live_roi_cache:
            return self._live_roi_cache[key]
        crop = self._crop_roi(shot, roi)
        if crop is None:
            return None
        vector = self.embedder.embed(crop)
        self._live_roi_cache[key] = vector
        return vector

    def _rank_if_branches(
        self, if_node: Node, shot: np.ndarray, screenshot_embedding: np.ndarray
    ) -> list[Match]:
        action = if_node.action or {}
        branch_ids = action.get("if_branches") or []
        default_id = action.get("if_default")
        nodes = {node.id: node for node in self.nodes}
        for branch_id in branch_ids:
            try:
                node_id = int(branch_id)
            except (TypeError, ValueError):
                continue
            node = nodes.get(node_id)
            if node is None or node.id == if_node.id:
                continue
            score = self._branch_similarity(node, shot, screenshot_embedding)
            if score is None:
                continue
            if score >= self._threshold_for(node):
                return [
                    Match(node=node, similarity=score, bonus=1.0, score=score)
                ]
        if default_id:
            try:
                node = nodes.get(int(default_id))
            except (TypeError, ValueError):
                node = None
            if node is not None and node.id != if_node.id:
                return [
                    Match(node=node, similarity=1.0, bonus=0.0, score=1.0)
                ]
        return []

    def _branch_similarity(
        self, node: Node, shot: np.ndarray, screenshot_embedding: np.ndarray
    ) -> float | None:
        if node.kind in ("nodetect", "if"):
            return 1.0
        if node.kind == "detect":
            found = self._detect_focus(shot, node)
            if found is None:
                return 0.0
            x, y, w, h, template_score = found
            model = sim_model.get_model(node)
            if node.roi is not None:
                components = sim_model.region_component_scores(
                    node, shot, node.roi, (x, y, w, h)
                )
                low = sim_model.check_component_thresholds(
                    components or {}, model.get("component_thresholds")
                )
                if low:
                    self.warn(
                        f"节点「{node.name}」动作相似度低："
                        + "、".join(low)
                        + "低于成分阈值"
                    )
                    return 0.0
                return sim_model.calibrated_weighted_score(
                    components or {}, model.get("roi", {}), model
                )
            return float(template_score)
        score = sim_model.score_node(node, shot)
        if score is not None:
            return float(score)
        embedding = self.embeddings.get(node.id)
        if embedding is None:
            return 0.0
        return cosine(screenshot_embedding, embedding)

    def rank(self, shot: np.ndarray, screenshot_embedding: np.ndarray) -> list[Match]:
        beta = float(self.settings.get("beta", 0.12))
        stay_penalty = float(self.settings.get("stay_penalty", 0.06))
        if self.current_state is not None:
            current = next(
                (node for node in self.nodes if node.id == self.current_state), None
            )
            if current is not None and current.kind == "if":
                return self._rank_if_branches(current, shot, screenshot_embedding)
            if (
                current is not None
                and current.kind == "ai"
                and current.id not in self._ai_done
            ):
                # AI 节点没完成前，只允许它自己继续决策，避免被图片节点抢走。
                return [
                    Match(node=current, similarity=1.0, bonus=1.0, score=1.0)
                ]
        scope = str(self.settings.get("match_scope", "next"))
        if scope == "next":
            allowed = self._next_candidates()
        elif scope == "reachable":
            allowed = self._reachable()
        elif self.settings.get("only_reachable"):
            allowed = self._reachable()
        else:
            allowed = None
        matches: list[Match] = []
        for node in self.image_nodes:
            if allowed is not None and node.id not in allowed:
                continue
            if node.kind in ("nodetect", "if", "ai"):
                # 已完成的 AI 节点只有在“还停在它自己身上”时才跳过；
                # 从别的节点绕回来时允许重新进入（支持流程环路）。
                if (
                    node.kind == "ai"
                    and node.id in self._ai_done
                    and node.id == self.current_state
                ):
                    continue
                entry_start = self.steps == 0 and (
                    (self.current_state is None and node.is_entry)
                    or self.current_state == node.id
                )
                if self.steps == 0 and self.current_state is not None and not entry_start:
                    # First step should be the entry node, not one of its successors.
                    continue
                if not entry_start:
                    if self.current_state is None:
                        if not node.is_entry:
                            continue
                    elif not any(
                        edge.src_id == self.current_state and edge.dst_id == node.id
                        for edge in self.edges
                    ):
                        continue
                similarity = 1.0
                bonus = self._bonus(node.id)
                score = similarity + beta * bonus
                matches.append(
                    Match(
                        node=node,
                        similarity=similarity,
                        bonus=bonus,
                        score=score,
                    )
                )
                continue
            if node.kind == "detect":
                found = self._detect_focus(shot, node)
                if found is None:
                    similarity = 0.0
                    roi_similarity = 0.0
                    region = None
                else:
                    x, y, w, h, template_score = found
                    region = (x, y, w, h)
                    model = sim_model.get_model(node)
                    if node.roi is not None:
                        components = sim_model.region_component_scores(
                            node, shot, node.roi, region
                        )
                        low = sim_model.check_component_thresholds(
                            components or {}, model.get("component_thresholds")
                        )
                        if low:
                            self.warn(
                                f"节点「{node.name}」动作相似度低："
                                + "、".join(low)
                                + "低于成分阈值"
                            )
                            similarity = 0.0
                            roi_similarity = 0.0
                        else:
                            component = sim_model.calibrated_weighted_score(
                                components or {}, model.get("roi", {}), model
                            )
                            similarity = component
                            roi_similarity = component
                    else:
                        similarity = float(template_score)
                        roi_similarity = float(template_score)
                bonus = self._bonus(node.id)
                score = similarity + beta * bonus
                if node.id == self.current_state:
                    score -= stay_penalty
                matches.append(
                    Match(
                        node=node,
                        similarity=similarity,
                        bonus=bonus,
                        score=score,
                        roi_similarity=roi_similarity,
                        region=region,
                    )
                )
                continue
            explained = sim_model.explain_node(node, shot)
            if explained is not None:
                low = explained.get("low_components") or []
                if low:
                    self.warn(
                        f"节点「{node.name}」动作相似度低："
                        + "、".join(low)
                        + "低于成分阈值"
                    )
                similarity = float(explained["score"])
                roi_similarity = None
                if "roi" in explained:
                    roi_similarity = sim_model.weighted_score(
                        explained["roi"], sim_model.get_model(node)["roi"]
                    )
                bonus = self._bonus(node.id)
                score = similarity + beta * bonus
                if node.id == self.current_state:
                    score -= stay_penalty
                matches.append(
                    Match(
                        node=node,
                        similarity=similarity,
                        bonus=bonus,
                        score=score,
                        roi_similarity=roi_similarity,
                    )
                )
                continue
            embedding = self.embeddings.get(node.id)
            if embedding is None:
                continue
            full_similarity = cosine(screenshot_embedding, embedding)
            roi_similarity = None
            if node.has_roi:
                node_roi = self._node_roi_embedding(node)
                live_roi = self._live_roi_embedding(shot, node.roi)
                if node_roi is not None and live_roi is not None:
                    roi_similarity = cosine(live_roi, node_roi)
            similarity = full_similarity
            if roi_similarity is not None:
                weight = max(0.0, min(1.0, float(node.roi_weight)))
                similarity = weight * roi_similarity + (1.0 - weight) * full_similarity
            bonus = self._bonus(node.id)
            score = similarity + beta * bonus
            if node.id == self.current_state:
                score -= stay_penalty
            matches.append(
                Match(
                    node=node,
                    similarity=similarity,
                    bonus=bonus,
                    score=score,
                    roi_similarity=roi_similarity,
                )
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches

    def _choose(self, matches: list[Match]) -> Match | None:
        eligible = [
            match for match in matches if match.similarity >= self._threshold_for(match.node)
        ]
        if not eligible:
            return None
        min_margin = float(self.settings.get("min_margin", 0.0) or 0.0)
        if min_margin > 0 and len(matches) >= 2:
            # 最优候选必须比“最难负样本”高出足够多，避免两个相似页面误匹配。
            challenger = max(
                (
                    match.similarity
                    for match in matches
                    if match.node.id != eligible[0].node.id
                ),
                default=0.0,
            )
            if eligible[0].similarity - challenger < min_margin:
                return None
        mode = str(self.settings.get("selection_mode", "best"))
        if mode != "weighted_random" or len(eligible) == 1:
            return eligible[0]
        weights = []
        for match in eligible:
            margin = max(0.05, match.similarity - self._threshold_for(match.node) + 0.1)
            weights.append(max(0.02, match.bonus) * margin)
        return random.choices(eligible, weights=weights, k=1)[0]

    def _threshold_for(self, node: Node) -> float:
        model = sim_model.get_model(node)
        calibration = model.get("calibration")
        # 侧重检测节点没有 ROI 时使用模板分数（原始空间），此时不能套用成分校准阈值。
        if node.kind == "detect" and node.roi is None:
            calibration = None
        if isinstance(calibration, dict) and calibration:
            base = float(
                calibration.get(
                    "threshold_norm", calibration.get("threshold_raw", node.threshold)
                )
            )
            step = float(self.fallback.get("calibrated_step", 0.02))
            relax = float(self.fallback.get("calibrated_relax", 0.05))
            floor = max(0.0, base - relax)
        else:
            base = float(node.threshold)
            step = float(self.fallback.get("threshold_step", 0.04))
            floor = min(base, float(self.fallback.get("min_threshold", 0.55)))
        return max(floor, base - step * self.retries)

    def _node_image(self, node: Node) -> np.ndarray | None:
        if node.id in self.node_images:
            return self.node_images[node.id]
        if node.image_path and Path(node.image_path).exists():
            try:
                image = vision.load_rgb(node.image_path)
                self.node_images[node.id] = image
                return image
            except Exception:
                return None
        return None

    def _detect_focus(self, shot: np.ndarray, node: Node):
        """Search the whole screen for the node's 检测重点 (ROI) template."""
        if not node.has_roi:
            return None
        node_image = self._node_image(node)
        if node_image is None:
            return None
        crop = vision.crop_relative(node_image, node.roi)
        if crop is None or crop.shape[0] < 6 or crop.shape[1] < 6:
            return None
        if not vision.cv2_available():
            self.warn("未安装 OpenCV，重点检测节点无法扫描画面")
            return None
        return vision.template_locate(
            shot, crop, min_score=max(0.45, float(node.threshold))
        )

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep；加速模式下按倍率缩短，极速模式直接跳过。"""
        seconds = float(seconds or 0)
        if seconds <= 0 or self.skip_waits:
            return
        seconds = seconds / self.speed
        deadline = time.monotonic() + seconds
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _execute(self, node: Node, shot: np.ndarray) -> str:
        action = node.action or {}
        pre_wait = float(action.get("pre_wait", 0) or 0)
        default_post = 0.6 if node.action_type == "tap" else 0.3
        post_wait = float(action.get("post_wait", action.get("wait", default_post)) or 0)
        if pre_wait > 0:
            self._sleep(pre_wait)
        if self._stop.is_set():
            return "已停止"
        description = node.action_summary
        kind = node.action_type
        if node.kind == "ai" or kind == "ai":
            description = self._execute_ai_node(node, shot)
            if post_wait > 0:
                self._sleep(post_wait)
            return description
        if kind == "tap":
            node_image = self._node_image(node)
            target_region = node.target_region
            ui_selector = action.get("ui_selector")
            ui_clicked = False
            if isinstance(ui_selector, dict) and ui_selector:
                finder = getattr(self.device, "find_ui_element", None)
                if finder is not None:
                    try:
                        element = finder(ui_selector)
                    except Exception as exc:
                        element = None
                        self.warn(f"节点「{node.name}」控件树定位失败：{exc}")
                    if element:
                        cx, cy = element.get("center", [0, 0])
                        self.device.tap(cx, cy)
                        label = (
                            element.get("text")
                            or element.get("description")
                            or element.get("resource_id")
                            or "元素"
                        )
                        description = f"控件树点击 {label} ({cx}, {cy})"
                        ui_clicked = True
                    else:
                        self.warn(f"节点「{node.name}」未找到控件，已回退图片定位")
            if ui_clicked:
                pass
            elif node.kind == "detect":
                found = self._detect_focus(shot, node)
                if found is not None:
                    dx, dy, dw, dh, _score = found
                    point = (dx + dw // 2, dy + dh // 2)
                    self.device.tap(point[0], point[1])
                    description = f"检测到重点，点击 {point}"
                else:
                    point, _ = screen_point(node, shot, node_image)
                    if point:
                        self.device.tap(point[0], point[1])
                        description = f"未检测到重点，按点击点点击 {point}"
            elif target_region is not None and node_image is not None:
                point, region, found = vision.locate_target(
                    shot,
                    node_image,
                    target_region,
                    min_score=float(node.target_threshold),
                )
                model = (node.action or {}).get("similarity_model")
                target_mr = None
                low_click: list[str] = []
                if isinstance(model, dict) and isinstance(model.get("target"), dict):
                    target_info = sim_model.roi_explain(
                        node,
                        shot,
                        target_region,
                        model["target"],
                        model.get("component_thresholds"),
                        region,
                    )
                    if target_info is not None:
                        target_mr = float(target_info.get("score", 0.0))
                        low_click = target_info.get("low_components") or []
                target_min_mr = float(
                    action.get(
                        "target_min_mr", self.fallback.get("target_min_mr", 0.4)
                    )
                    or 0.4
                )
                if found and target_mr is not None and target_mr < target_min_mr:
                    found = False
                    height, width = shot.shape[:2]
                    point = _region_center(target_region, width, height)
                    region = None
                if low_click:
                    self.warn(
                        f"节点「{node.name}」点击对象动作相似度低："
                        + "、".join(low_click)
                        + "低于成分阈值，已跳过点击"
                    )
                    description = "点击对象动作相似度低，已跳过点击"
                elif point:
                    if not found and action.get("target_strict"):
                        description = "未识别到点击对象，已跳过点击"
                        if not vision.cv2_available():
                            self.warn("未安装 OpenCV，点击对象无法定位，已跳过点击")
                        else:
                            self.warn(f"节点「{node.name}」未定位到点击对象，已跳过点击")
                    else:
                        self.device.tap(point[0], point[1])
                        if found:
                            description = f"点击对象中心 {point}"
                        else:
                            if not vision.cv2_available():
                                self.warn("未安装 OpenCV，点击对象无法定位，已按相对位置点击")
                            else:
                                self.warn(f"节点「{node.name}」未定位到点击对象，已按相对位置点击")
                            description = f"点击对象相对中心 {point}（未定位）"
            elif node.relative_roi is not None and node_image is not None:
                offset_x, offset_y = node.relative_offset
                point, region, found = vision.locate_target(
                    shot,
                    node_image,
                    node.relative_roi,
                    min_score=float(action.get("rel_threshold", 0.55) or 0.55),
                )
                model = (node.action or {}).get("similarity_model")
                low_click: list[str] = []
                if isinstance(model, dict) and isinstance(model.get("target"), dict):
                    rel_info = sim_model.roi_explain(
                        node,
                        shot,
                        node.relative_roi,
                        model["target"],
                        model.get("component_thresholds"),
                        region,
                    )
                    if rel_info is not None:
                        rel_mr = float(rel_info.get("score", 0.0))
                        low_click = rel_info.get("low_components") or []
                        rel_min_mr = float(
                            action.get(
                                "rel_min_mr", self.fallback.get("rel_min_mr", 0.4)
                            )
                            or 0.4
                        )
                        if found and rel_mr < rel_min_mr:
                            found = False
                            height, width = shot.shape[:2]
                            point = _region_center(node.relative_roi, width, height)
                            region = None
                if low_click:
                    self.warn(
                        f"节点「{node.name}」相对点击动作相似度低："
                        + "、".join(low_click)
                        + "低于成分阈值，已跳过点击"
                    )
                    description = "相对点击动作相似度低，已跳过点击"
                elif point:
                    height, width = shot.shape[:2]
                    tap_x = max(0, min(width - 1, round(point[0] + offset_x)))
                    tap_y = max(0, min(height - 1, round(point[1] + offset_y)))
                    self.device.tap(tap_x, tap_y)
                    if found:
                        description = (
                            f"相对图片中心偏移点击 ({tap_x}, {tap_y})"
                            f"（偏移 {offset_x:g},{offset_y:g}）"
                        )
                    else:
                        if not vision.cv2_available():
                            self.warn("未安装 OpenCV，相对点击图片无法定位，已按相对位置偏移点击")
                        else:
                            self.warn(
                                f"节点「{node.name}」未定位到相对点击图片，已按相对位置偏移点击"
                            )
                        description = (
                            f"相对图片相对位置偏移点击 ({tap_x}, {tap_y})（未定位）"
                        )
            else:
                point, _ = screen_point(node, shot, node_image)
                if point:
                    self.device.tap(point[0], point[1])
                    description = f"点击 {point}"
        elif kind == "swipe":
            height, width = shot.shape[:2]
            x1 = _relative(action.get("x1", 0.5), width)
            y1 = _relative(action.get("y1", 0.8), height)
            x2 = _relative(action.get("x2", 0.5), width)
            y2 = _relative(action.get("y2", 0.2), height)
            duration = int(action.get("duration", 300) or 300)
            repeat = 1
            if self.speed > 1 or self.skip_waits or action.get("repeat"):
                repeat = max(1, int(action.get("repeat") or self.swipe_repeat))
            if repeat > 1 or self.speed > 1:
                duration = max(40, int(duration / max(1.0, self.speed)))
            if self.skip_waits and repeat > 1:
                duration = min(duration, 50)
            if repeat > 1:
                batched = getattr(self.device, "swipe_repeat", None)
                if batched is not None:
                    batched(x1, y1, x2, y2, duration, repeat)
                    description = (
                        f"滑动 ({int(x1)},{int(y1)})->({int(x2)},{int(y2)})"
                        f" × {repeat}（批量）"
                    )
                else:
                    for index in range(repeat):
                        if self._stop.is_set():
                            break
                        self.device.swipe(x1, y1, x2, y2, duration)
                        if index < repeat - 1:
                            self._sleep(0.01)
                    description = (
                        f"滑动 ({int(x1)},{int(y1)})->({int(x2)},{int(y2)})"
                        f" × {repeat}"
                    )
            else:
                self.device.swipe(x1, y1, x2, y2, duration)
                description = f"滑动 ({int(x1)},{int(y1)})->({int(x2)},{int(y2)})"
        elif kind == "burst":
            height, width = shot.shape[:2]
            count = max(1, int(action.get("count", 100) or 100))
            x1 = _relative(action.get("x1", 0.5), width)
            y1 = _relative(action.get("y1", 0.8), height)
            x2 = _relative(action.get("x2", 0.5), width)
            y2 = _relative(action.get("y2", 0.2), height)
            duration = int(action.get("duration", 50) or 50)
            burster = getattr(self.device, "burst_swipe", None)
            if burster is None:
                raise RuntimeError("当前设备不支持极速连滑（需要 adb monkey）")
            burster(x1, y1, x2, y2, count, duration)
            description = (
                f"极速连滑 ×{count} ({int(x1)},{int(y1)})->({int(x2)},{int(y2)})"
            )
        elif kind == "key":
            key = str(action.get("key", "back"))
            self.device.key(key)
            description = f"按键 {key}"
        elif kind == "text":
            text = str(action.get("text", ""))
            if action.get("clear"):
                clear = getattr(self.device, "clear_text", None)
                if clear:
                    clear()
            self.device.text(text)
            if action.get("submit"):
                self.device.key("enter")
            description = f"输入 {text[:24]}" + (" + 回车" if action.get("submit") else "")
        elif kind == "launch":
            package = str(action.get("package", "")).strip()
            if not package:
                raise RuntimeError("启动应用缺少 package 参数")
            launcher = getattr(self.device, "launch_app", None)
            if launcher is None:
                raise RuntimeError("当前设备不支持启动应用")
            launcher(package, bool(action.get("fresh", True)))
            description = f"启动应用 {package}"
        elif kind == "wait":
            seconds = float(action.get("seconds", 1) or 1)
            self._sleep(seconds)
            description = f"等待 {seconds}s"
        if post_wait > 0:
            self._sleep(post_wait)
        return description

    # ------------------------------------------------------------------
    # 本地 AI（Ollama）相关
    # ------------------------------------------------------------------
    def _ui_elements(self) -> list[dict]:
        """读取 Android 控件树作为 AI 的上下文；失败不影响运行。"""
        if not self.ai_settings.ui_context:
            return []
        getter = getattr(self.device, "ui_elements", None)
        if getter is None:
            return []
        try:
            return list(getter() or [])
        except Exception as exc:
            self.warn(f"读取 Android 控件树失败，AI 只能看截图：{exc}")
            return []

    def _execute_ai_node(self, node: Node, shot: np.ndarray) -> str:
        if node.id != self.current_state:
            # 从上一个节点进入：重新开始计数，支持流程环路多次执行。
            self._ai_done.discard(node.id)
            self._ai_cycles.pop(node.id, None)
            self._ai_last_signature.pop(node.id, None)
            self._ai_last_action.pop(node.id, None)
        action = node.action or {}
        goal = str(action.get("goal") or node.notes or node.name).strip() or node.name
        limit = int(action.get("max_decisions") or self.ai_settings.max_decisions or 12)
        cycle = self._ai_cycles.get(node.id, 0) + 1
        self._ai_cycles[node.id] = cycle
        if cycle > limit:
            self._ai_done.add(node.id)
            self.warn(f"AI 节点「{node.name}」达到最大决策次数 {limit}，视为完成")
            return f"AI 达到决策上限（{limit} 次）"
        if self.brain is None:
            raise RuntimeError("AI 未启用：请在「AI 设置」中开启，或把节点改回普通图片节点")
        if self._stop.is_set():
            return "已停止"

        hint = ""
        detail = str(action.get("detail") or "").strip()
        if detail:
            hint = f"补充说明：{detail}"
        previous = self._ai_last_signature.get(node.id)
        if previous is not None:
            # 上一步刚点完，页面可能还在切换/加载：等画面稳定再决策，
            # 否则模型会看到「半新半旧」的截图而重复点击。
            shot = self._settle_shot(shot)
        height, width = shot.shape[:2]
        signature = self._shot_signature(shot)
        stalled = previous is not None and self._shot_diff(previous, signature) < 0.015
        self._ai_last_signature[node.id] = signature
        if stalled:
            hint += (
                "\n注意：上一步操作之后画面几乎没有变化，说明那个动作没有生效，"
                "不要重复同样的操作；如果目标其实已经达成，直接返回 done=true。"
            )
        elif previous is not None and self._ai_history:
            hint += (
                "\n上一步操作已经执行（见「最近已经做过的操作」）。"
                "如果目标动作已经完成、画面也已经变化，请返回 action=done、done=true；"
                "如果还在原页面或目标没完成，再继续操作。"
            )
        decision = self.brain.decide(
            goal,
            shot,
            elements=self._ui_elements(),
            history=self._ai_history,
            hint=hint,
            screen_size=(width, height),
        )
        self.emit(
            "ai_decision",
            node_id=node.id,
            node_name=node.name,
            page=decision.page,
            reason=decision.reason,
            action=decision.action,
            done=decision.done,
            confidence=decision.confidence,
            elapsed=decision.elapsed,
            element_index=decision.element_index,
            element_label=decision.element_label,
        )
        if decision.done:
            verdict = None
            if self._ai_history:  # 本轮已经做过动作，值得复核一次
                verdict = self._ai_verify(node, goal, shot)
            self._ai_done.add(node.id)
            self._ai_history.append(
                {
                    "page": decision.page or node.name,
                    "action": "done",
                    "detail": decision.reason[:80],
                }
            )
            self._ai_history = self._ai_history[-8:]
            if verdict is not None and not verdict.get("done"):
                reason = str(verdict.get("reason") or "").strip()
                self.warn(
                    f"AI 节点「{node.name}」声称完成，但复核认为目标未达成：{reason[:60]}"
                )
                return f"AI 判定完成（复核未通过：{reason[:40]}）"
            if verdict is not None:
                self.emit(
                    "ai_verified",
                    node_id=node.id,
                    node_name=node.name,
                    done=True,
                    reason=verdict.get("reason") or "",
                    page=verdict.get("page") or "",
                )
            return f"AI 判定完成（{decision.page or node.name}）"

        signature_key = (
            decision.action,
            decision.element_index,
            None if decision.x is None else round(float(decision.x), 3),
            None if decision.y is None else round(float(decision.y), 3),
        )
        if stalled and self._ai_last_action.get(node.id) == signature_key:
            self._ai_done.add(node.id)
            self.warn(f"AI 节点「{node.name}」连续两次执行同一个动作且画面无变化，已结束该节点")
            self._ai_history.append(
                {
                    "page": decision.page or node.name,
                    "action": decision.action,
                    "detail": "重复动作且画面无变化，节点结束",
                }
            )
            return "AI 连续无效动作，已结束该节点"

        description = self._perform_ai_action(decision, shot)
        self._ai_last_action[node.id] = signature_key
        self._ai_history.append(
            {
                "page": decision.page or node.name,
                "action": decision.action,
                "detail": description,
            }
        )
        self._ai_history = self._ai_history[-8:]
        if cycle >= limit:
            self._ai_done.add(node.id)
            self.warn(f"AI 节点「{node.name}」达到最大决策次数 {limit}，自动收尾")
            description += f"（达到决策上限 {limit}）"
        return description

    def _settle_shot(self, fallback: np.ndarray, max_wait: float = 2.4) -> np.ndarray:
        """等界面稳定：连续两帧几乎一样就返回，最多等 max_wait 秒。"""
        last = fallback
        last_signature = self._shot_signature(last)
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline and not self._stop.is_set():
            self._sleep(0.3)
            try:
                current = self.device.screenshot()
            except Exception:
                break
            if current is None:
                break
            signature = self._shot_signature(current)
            if self._shot_diff(last_signature, signature) < 0.004:
                return current
            last, last_signature = current, signature
        return last

    @staticmethod
    def _shot_signature(shot: np.ndarray) -> np.ndarray:
        """缩成 32x32 灰度图，用于判断两次决策之间画面有没有变化。"""
        try:
            image = vision.to_pil(shot).convert("L").resize((32, 32))
            return np.asarray(image, dtype=np.float32) / 255.0
        except Exception:
            return np.zeros((32, 32), dtype=np.float32)

    @staticmethod
    def _shot_diff(left: np.ndarray, right: np.ndarray) -> float:
        if left is None or right is None or left.shape != right.shape:
            return 1.0
        return float(np.abs(left - right).mean())

    def _ai_verify(self, node: Node, goal: str, shot: np.ndarray) -> dict | None:
        """让模型复核目标是否已达成；大脑不支持时返回 None。"""
        verifier = getattr(self.brain, "verify_goal", None)
        if verifier is None:
            return None
        try:
            return verifier(goal, shot, elements=self._ui_elements())
        except OllamaError as exc:
            self.warn(f"AI 完成复核失败：{exc}")
            return None
        except Exception as exc:  # 复核不能影响主流程
            self.warn(f"AI 完成复核异常：{exc}")
            return None

    def _perform_ai_action(self, decision, shot: np.ndarray) -> str:
        height, width = shot.shape[:2]
        kind = decision.action
        if kind == "tap":
            if decision.x is None or decision.y is None:
                raise RuntimeError("AI 要求点击但没有给出坐标")
            x = max(0, min(width - 1, round(decision.x * width)))
            y = max(0, min(height - 1, round(decision.y * height)))
            self.device.tap(x, y)
            label = f" {decision.element_label}" if decision.element_label else ""
            return f"AI 点击{label} ({x}, {y})"
        if kind == "swipe":
            self.device.swipe(
                width * 0.5, height * 0.78, width * 0.5, height * 0.26, 320
            )
            return "AI 上滑"
        if kind == "back":
            self.device.key("back")
            return "AI 返回"
        if kind == "key":
            self.device.key(decision.key)
            return f"AI 按键 {decision.key}"
        if kind == "text":
            text = decision.text
            if not text:
                return "AI 想输入文字但内容为空"
            if any(ord(char) > 127 for char in text):
                self.warn("ADB 直接输入中文可能失败，建议安装并启用 ADBKeyboard 输入法")
            self.device.text(text)
            return f"AI 输入 {text[:24]}"
        if kind == "wait":
            self._sleep(decision.seconds)
            return f"AI 等待 {decision.seconds:g}s"
        return decision.describe()

    def _ai_goal(self) -> str:
        if self.flow is None:
            return "继续当前手机操作"
        return self.flow.description or self.flow.name or "继续当前手机操作"

    def _ai_rerank(self, shot: np.ndarray, matches: list[Match]) -> Match | None:
        """图片相似度都不够时，让视觉模型在候选节点里挑一个。"""
        if self.brain is None or not self.ai_settings.ai_rerank or not matches:
            return None
        usable = [
            match
            for match in matches
            if match.node.kind in ("image", "detect") and match.similarity >= 0.35
        ][:4]
        if len(usable) < 2:
            return None
        height, width = shot.shape[:2]
        payload = [
            {
                "id": match.node.id,
                "name": match.node.name,
                "notes": match.node.notes,
                "similarity": match.similarity,
            }
            for match in usable
        ]
        goal = self._ai_goal()
        current = next(
            (node for node in self.nodes if node.id == self.current_state), None
        )
        if current is not None:
            goal = f"{goal}（当前在节点「{current.name}」）"
        try:
            result = self.brain.choose_node(
                shot,
                payload,
                goal=goal,
                elements=self._ui_elements(),
                screen_size=(width, height),
            )
        except OllamaError as exc:
            self.warn(f"AI 节点重排失败：{exc}")
            return None
        if result["choice"] < 0 or result["confidence"] < 0.6:
            return None
        chosen = next(
            (match for match in usable if match.node.id == result["choice"]), None
        )
        if chosen is None:
            return None
        self.emit(
            "ai_rerank",
            node_id=chosen.node.id,
            node_name=chosen.node.name,
            confidence=result["confidence"],
            reason=result["reason"],
        )
        return Match(
            node=chosen.node,
            similarity=max(chosen.similarity, 0.0),
            bonus=chosen.bonus,
            score=chosen.score,
            roi_similarity=chosen.roi_similarity,
            region=chosen.region,
        )

    def _ai_recover(self, shot: np.ndarray) -> tuple[str, bool] | None:
        """图片匹配失败时让 AI 接管一步。返回 (描述, 是否结束流程)。"""
        if self.brain is None or not self.ai_settings.ai_fallback:
            return None
        height, width = shot.shape[:2]
        goal = self._ai_goal()
        current = next(
            (node for node in self.nodes if node.id == self.current_state), None
        )
        next_names: list[str] = []
        for edge in self.edges:
            if edge.src_id != self.current_state:
                continue
            target = next((node for node in self.nodes if node.id == edge.dst_id), None)
            if target is not None:
                next_names.append(target.name)
        hint = "图片匹配失败了，请让界面回到流程可以继续的状态。"
        if current is not None:
            hint += f"\n当前卡在节点「{current.name}」。"
        if next_names:
            hint += "\n流程期望的下一步是：" + "、".join(next_names[:6]) + "。"
        try:
            decision = self.brain.decide(
                goal,
                shot,
                elements=self._ui_elements(),
                history=self._ai_history,
                hint=hint,
                screen_size=(width, height),
            )
        except OllamaError as exc:
            self.warn(f"AI 兜底失败：{exc}")
            return None
        self.emit(
            "ai_recover",
            page=decision.page,
            reason=decision.reason,
            action=decision.action,
            done=decision.done,
        )
        if decision.done:
            return ("AI 判定任务已完成", True)
        description = self._perform_ai_action(decision, shot)
        self._ai_history.append(
            {
                "page": decision.page or "未知页面",
                "action": decision.action,
                "detail": decision.reason[:80],
            }
        )
        self._ai_history = self._ai_history[-8:]
        return (description, False)

    def _save_shot(self, shot: np.ndarray) -> str | None:
        if not self.settings.get("save_screenshots", True) or self._screenshot_dir is None:
            return None
        return vision.save_thumbnail(
            self._screenshot_dir / f"step_{self.log_index:04d}.jpg", shot
        )

    def _ensure_session(self) -> None:
        if self.session_id is not None:
            return
        self.session_id = self.db.create_session(self.flow_id, note="stepwise")
        self._screenshot_dir = config.screenshots_dir() / f"session_{self.session_id}"

    def _time_limit_reached(self) -> bool:
        return (
            self.max_run_seconds > 0
            and (time.monotonic() - self._start_time) >= self.max_run_seconds
        )

    def _timeout_end(self) -> bool:
        self.finished = True
        elapsed = time.monotonic() - self._start_time
        self.log_index += 1
        if self.session_id is not None:
            self.db.log_step(
                self.session_id,
                self.log_index,
                self.current_state,
                None,
                None,
                None,
                "运行时间结束",
                f"已运行 {elapsed:.1f}s / {self.max_run_seconds:.0f}s",
                None,
            )
        self.emit("timeout", elapsed=elapsed, limit=self.max_run_seconds)
        return False

    def _manual_step(self) -> bool:
        """用户按了「人工选择」：暂停，等用户决定动作，执行后可检测是否达标。"""
        shot = self.device.screenshot()
        self.emit("screenshot", frame=shot)
        self.log_index += 1
        action = self._request_human(shot, "运行中：请选择下一步动作")
        if action is None:
            return True
        if action.get("stop"):
            self.stop()
            return True
        if action.get("ai"):
            if self.brain is not None and self.ai_settings.ai_fallback:
                recovered = self._ai_recover(shot)
                if recovered is not None:
                    description, stop_run = recovered
                    self.emit("human_applied", action=description, ai=True)
                    if stop_run:
                        self.finished = True
                        self.emit("end", node=None)
                        return False
                    return True
            self.warn("AI 托管不可用（未启用或失败），已跳过")
            return True
        return self._run_manual_action(action, shot, "manual")

    def step(self) -> bool:
        if self._stop.is_set() or self.finished:
            return False
        if self.flow is None:
            self.prepare()
        self._ensure_session()
        if self._time_limit_reached():
            return self._timeout_end()
        if self._manual_pending:
            self._manual_pending = False
            return self._manual_step()
        shot = self.device.screenshot()
        self.emit("screenshot", frame=shot)
        embedding = self.embedder.embed(shot)
        self._live_roi_cache = {}
        matches = self.rank(shot, embedding)
        best = self._choose(matches)
        if best is None:
            best = self._ai_rerank(shot, matches)

        self.log_index += 1
        screenshot_path = self._save_shot(shot)

        if best is None:
            return self._no_match(shot, matches, screenshot_path)

        node = best.node
        self.retries = 0
        self.steps += 1
        try:
            description = self._execute(node, shot)
        except Exception as exc:
            self.error = f"执行动作失败：{exc}"
            self.db.log_step(
                self.session_id,
                self.log_index,
                self.current_state,
                node.id,
                best.similarity,
                best.score,
                node.action_summary,
                f"error: {exc}",
                screenshot_path,
            )
            self.emit("error", message=self.error)
            self.finished = True
            return False

        if self._stop.is_set():
            return False

        previous = self.current_state
        self.current_state = node.id
        self.db.log_step(
            self.session_id,
            self.log_index,
            previous,
            node.id,
            best.similarity,
            best.score,
            description,
            "ok",
            screenshot_path,
        )
        self.emit(
            "match",
            match=best,
            matches=matches[:6],
            previous_state=previous,
            state=node.id,
            state_name=node.name,
            action=description,
            similarity=best.similarity,
            score=best.score,
            roi_similarity=best.roi_similarity,
            screenshot_path=screenshot_path,
        )
        if node.kind == "ai" and node.id in self._ai_done:
            has_next = any(edge.src_id == node.id for edge in self.edges)
            if not has_next:
                self.finished = True
                self.emit("end", node=node)
                return False
        if node.kind == "end" or node.is_end:
            self.finished = True
            self.emit("end", node=node)
            return False
        return True

    def _no_match(self, shot: np.ndarray, matches: list[Match], screenshot_path: str | None) -> bool:
        self.retries += 1
        max_retries = int(self.fallback.get("max_retries", 2))
        if self.skip_waits:
            max_retries = min(max_retries, 1)
        best = matches[0] if matches else None
        similarity = best.similarity if best else None
        score = best.score if best else None
        self.db.log_step(
            self.session_id,
            self.log_index,
            self.current_state,
            None,
            similarity,
            score,
            "未匹配",
            f"retry {self.retries}/{max_retries}",
            screenshot_path,
        )
        self.emit(
            "no_match",
            matches=matches[:6],
            retries=self.retries,
            max_retries=max_retries,
            similarity=similarity,
        )
        if self.retries <= max_retries:
            self._sleep(float(self.fallback.get("retry_wait", 0.6)))
            return True

        self.retries = 0
        self.fallbacks += 1
        max_fallbacks = int(self.fallback.get("max_fallbacks", 3))
        if self.fallbacks > max_fallbacks:
            self.error = "保底次数超过上限"
            self.db.log_step(
                self.session_id,
                self.log_index,
                self.current_state,
                None,
                similarity,
                score,
                "停止",
                self.error,
                screenshot_path,
            )
            self.emit("error", message=self.error)
            self.finished = True
            return False

        if (
            self.brain is not None
            and self.ai_settings.ai_fallback
            and not self._stop.is_set()
        ):
            recovered = self._ai_recover(shot)
            if recovered is not None:
                description, stop_run = recovered
                self.db.log_step(
                    self.session_id,
                    self.log_index,
                    self.current_state,
                    None,
                    similarity,
                    score,
                    description,
                    "ai",
                    screenshot_path,
                )
                if stop_run:
                    self.finished = True
                    self.emit("end", node=None)
                    return False
                return True

        manual_node = self._manual_node()
        if manual_node is not None and not self._stop.is_set():
            # 记住的人工动作优先于静态保底步骤
            action = dict(manual_node.action or {})
            action["manual"] = True
            return self._run_manual_action(action, shot, "manual_fallback")

        self.emit(
            "fallback",
            index=self.fallbacks,
            max_fallbacks=max_fallbacks,
            steps=list(self.fallback.get("steps", [])),
        )
        for fallback_step in self.fallback.get("steps", []):
            try:
                description = self._execute_fallback_step(fallback_step, shot)
            except Exception as exc:
                description = f"保底动作失败：{exc}"
                self.warn(description)
            self.db.log_step(
                self.session_id,
                self.log_index,
                self.current_state,
                None,
                similarity,
                score,
                description,
                "fallback",
                screenshot_path,
            )
        return True

    def _execute_fallback_step(self, step: dict, shot: np.ndarray) -> str:
        height, width = shot.shape[:2]
        kind = step.get("type", "wait")
        wait = float(step.get("wait", step.get("seconds", 0.5)) or 0.5)
        if kind == "tap":
            x = _relative(step.get("x", 0.5), width)
            y = _relative(step.get("y", 0.5), height)
            self.device.tap(x, y)
            description = f"保底点击 ({int(x)},{int(y)})"
        elif kind == "swipe":
            x1 = _relative(step.get("x1", 0.5), width)
            y1 = _relative(step.get("y1", 0.8), height)
            x2 = _relative(step.get("x2", 0.5), height)
            y2 = _relative(step.get("y2", 0.2), height)
            self.device.swipe(x1, y1, x2, y2, int(step.get("duration", 300) or 300))
            description = "保底滑动"
        elif kind == "key":
            key = str(step.get("key", "back"))
            self.device.key(key)
            description = f"保底按键 {key}"
        elif kind == "text":
            text = str(step.get("text", ""))
            self.device.text(text)
            description = f"保底输入 {text[:24]}"
        else:
            description = f"保底等待 {wait}s"
        self._sleep(wait)
        return description

    def run(self) -> EngineResult:
        try:
            self.prepare()
        except Exception as exc:
            self.error = str(exc)
            self.emit("error", message=self.error)
            return EngineResult(status="error", error=self.error)

        self.session_id = self.db.create_session(self.flow_id)
        self._screenshot_dir = config.screenshots_dir() / f"session_{self.session_id}"
        self.emit(
            "start",
            session_id=self.session_id,
            flow=self.flow.name if self.flow else "",
            state=self.current_state,
            settings=self.settings,
        )
        status = "finished"
        try:
            for _ in range(self.max_steps):
                if self._stop.is_set():
                    status = "stopped"
                    break
                if not self.step():
                    break
                if self._stop.is_set():
                    status = "stopped"
                    break
            else:
                status = "max_steps"
        except Exception as exc:
            status = "error"
            self.error = str(exc)
            self.emit("error", message=str(exc), detail=traceback.format_exc())

        self.db.finish_session(
            self.session_id, status, self.steps, self.fallbacks, self.error or ""
        )
        self.emit(
            "finish",
            status=status,
            steps=self.steps,
            fallbacks=self.fallbacks,
            error=self.error,
            session_id=self.session_id,
        )
        return EngineResult(
            status=status,
            steps=self.steps,
            fallbacks=self.fallbacks,
            session_id=self.session_id,
            error=self.error,
        )
