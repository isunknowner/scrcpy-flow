"""把本地 Ollama 变成“看得懂屏幕”的决策大脑。

四个能力：
1. describe   —— 描述当前画面（应用、页面、可见文字、是否加载中）；
2. decide     —— 给定任务目标，输出下一步唯一动作（点击/滑动/输入/返回/等待/完成）；
3. choose_node—— 图片相似度不确定时，在几个候选流程节点里挑一个；
4. plan_flow  —— 把一句中文描述拆成可执行的 AI 流程步骤。

以及一个不依赖流程图的最小智能体 AgentRunner（命令行 --ai-task 使用）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .ollama import (
    OllamaClient,
    OllamaError,
    OllamaSettings,
    extract_json,
    get_client,
    load_settings,
)

SYSTEM = (
    "你是一名严谨的 Android 手机自动化操作员。你能看到手机截图，也能读到 Android "
    "控件树。你的每次回答都必须是一个合法 JSON 对象，不要输出 markdown 代码块、"
    "不要输出任何解释文字。"
)

ACTION_ALIASES = {
    "click": "tap",
    "touch": "tap",
    "press": "tap",
    "tap": "tap",
    "swipe": "swipe",
    "scroll": "swipe",
    "drag": "swipe",
    "back": "back",
    "key": "key",
    "keyevent": "key",
    "text": "text",
    "input": "text",
    "type": "text",
    "wait": "wait",
    "sleep": "wait",
    "done": "done",
    "finish": "done",
    "complete": "done",
    "stop": "done",
}


@dataclass
class Decision:
    """模型给出的一步操作。"""

    action: str = "none"
    x: float | None = None
    y: float | None = None
    element_index: int | None = None
    element_label: str = ""
    text: str = ""
    key: str = "back"
    seconds: float = 1.0
    page: str = ""
    reason: str = ""
    done: bool = False
    confidence: float = 0.0
    elapsed: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def is_noop(self) -> bool:
        return self.action in ("none", "done")

    def describe(self) -> str:
        if self.action == "tap":
            target = self.element_label or f"({(self.x or 0):.2f}, {(self.y or 0):.2f})"
            return f"AI 点击 {target}"
        if self.action == "swipe":
            return "AI 滑动"
        if self.action == "back":
            return "AI 返回"
        if self.action == "key":
            return f"AI 按键 {self.key}"
        if self.action == "text":
            return f"AI 输入 {self.text[:24]}"
        if self.action == "wait":
            return f"AI 等待 {self.seconds:g}s"
        if self.action == "done":
            return "AI 判定任务完成"
        return "AI 无动作"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


CONTAINER_HINTS = (
    "action_bar_root",
    "decor_content_parent",
    "content",
    "root",
    "background",
    "app_bar",
    "toolbar_container",
    "navigationbar",
    "statusbar",
)


def _is_container(element: dict) -> bool:
    """没有文字/描述、不可点、名字又像布局容器的元素，别让模型去点它。"""
    if element.get("text") or element.get("description"):
        return False
    if element.get("clickable"):
        return False
    rid = str(element.get("resource_id") or "").lower()
    class_name = str(element.get("class") or "").lower()
    if any(hint in rid for hint in CONTAINER_HINTS):
        return True
    return class_name.endswith(("layout", "viewgroup", "framelayout", "relativelayout"))


def format_elements(elements: list[dict] | None, limit: int = 50) -> str:
    """把 uiautomator 元素列表压成给模型看的紧凑文本。

    重要：不能按树的原始顺序直接截断——底部标签栏、悬浮按钮这类元素通常在
    控件树的最后，一截断就全丢了，模型只能瞎猜坐标。这里按
    「可点且有文字 > 有文字 > 可点 > 其他」排序后再取前 limit 个。
    """
    if not elements:
        return "（没有控件树信息，请完全依靠截图判断位置）"
    ranked: list[tuple[int, int, dict, str, bool]] = []
    for order, element in enumerate(elements):
        text = str(element.get("text") or "").strip()
        desc = str(element.get("description") or "").strip()
        rid = str(element.get("resource_id") or "").strip()
        clickable = bool(element.get("clickable"))
        if not (text or desc or rid or clickable):
            continue
        if _is_container(element):
            continue
        label = text or desc
        if clickable and label:
            priority = 0
        elif label:
            priority = 1
        elif clickable:
            priority = 2
        else:
            priority = 3
        ranked.append((priority, order, element, label or rid.split("/")[-1], clickable))
    ranked.sort(key=lambda item: (item[0], item[1]))
    rows: list[str] = []
    for _priority, _order, element, label, clickable in ranked[:limit]:
        bounds = element.get("bounds") or [0, 0, 0, 0]
        class_name = str(element.get("class") or "").split(".")[-1]
        flag = "可点" if clickable else "  "
        rows.append(
            f'#{element.get("index")} [{flag}] {class_name} "{label[:40]}" '
            f"bounds=[{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}]"
        )
    return "\n".join(rows) if rows else "（控件树里没有可用的文字或可点击元素）"


class ScreenBrain:
    """对 Ollama 的高层封装，所有方法失败都抛 OllamaError。"""

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        client: OllamaClient | None = None,
        log: Callable[[str], None] | None = None,
    ):
        self.settings = settings or load_settings()
        self.client = client or get_client(self.settings)
        self._log = log
        self._vision_model: str | None = None
        self._text_model: str | None = None

    # -- 基础 ------------------------------------------------------------
    def log(self, message: str) -> None:
        if self._log:
            try:
                self._log(message)
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled)

    def vision_model(self) -> str:
        if self._vision_model is None:
            self._vision_model = self.client.resolve_vision_model()
        return self._vision_model

    def text_model(self) -> str:
        if self._text_model is None:
            self._text_model = self.client.resolve_text_model()
        return self._text_model

    # -- 描述画面 --------------------------------------------------------
    def describe(self, shot: Any, elements: list[dict] | None = None) -> dict:
        prompt = f"""请描述这张手机截图。

可点击控件（可能为空）：
{format_elements(elements)}

输出 JSON：
{{
  "app": "应用名称，不确定就填空字符串",
  "page": "当前页面的简短名称，例如 外卖首页 / 聊天列表 / 商品详情",
  "summary": "一句话说明屏幕内容",
  "key_text": ["屏幕上最重要的 3~8 段文字"],
  "loading": true 或 false
}}"""
        text = self.client.vision(self.vision_model(), SYSTEM, prompt, shot)
        data = extract_json(text)
        if not isinstance(data, dict):
            raise OllamaError("描述画面的返回不是 JSON 对象")
        return data

    # -- 决策 ------------------------------------------------------------
    def decide(
        self,
        goal: str,
        shot: Any,
        *,
        elements: list[dict] | None = None,
        history: list[dict] | None = None,
        hint: str = "",
        screen_size: tuple[int, int] | None = None,
        allow_text: bool | None = None,
    ) -> Decision:
        allow_text = self.settings.allow_text if allow_text is None else allow_text
        width, height = screen_size or (1080, 2400)
        # 不要把模型自己上一轮写的页面名喂回去：那会把它锚定在旧页面上，
        # 导致「已经切换页面却仍认为没切换」而反复点同一个按钮。
        history_text = "无" if not history else "\n".join(
            f"{index + 1}. 执行了 {item.get('action')}：{item.get('detail') or ''}".rstrip(
                "： "
            )
            for index, item in enumerate(history[-6:])
        )
        text_rule = (
            "- 需要输入文字时用 action=text，把内容放进 text；"
            "如果想先清空输入框，把 clear 设为 true。"
            if allow_text
            else "- 本次运行不允许输入文字，请不要使用 action=text。"
        )
        prompt = f"""任务目标：{goal}
{hint}
最近已经做过的操作：
{history_text}

可用控件（来自 Android 控件树，index 用于精确点击）：
{format_elements(elements)}

屏幕像素尺寸：{width} x {height}（相对坐标 0~1 都要以整屏为基准）

请决定下一步只做的一个操作，输出 JSON：
{{
  "page": "当前页面简短名称",
  "thought": "一句话说明你为什么这样操作",
  "action": "tap / swipe / back / text / key / wait / done",
  "element_index": 控件序号或 null,
  "x": 相对横坐标 0~1 或 null,
  "y": 相对纵坐标 0~1 或 null,
  "text": "action=text 时输入的内容，否则空字符串",
  "clear": false,
  "key": "action=key 时的按键，例如 back / home / enter",
  "seconds": 1,
  "done": false,
  "confidence": 0~1 之间的小数
}}

规则：
- 每次决策前先判断「目标是否已经达成」：如果页面已经是目标页面（目标标签已高亮、
  标题或内容已经变化、目标列表已经出现），必须返回 action=done、done=true，
  不要再点一次去「确认」。
- 只点带「可点」标记的控件；没有「可点」标记的是容器或纯文字展示，不要拿它当点击目标。
- 有合适的控件时优先填 element_index，x/y 可以留 null。
- 只有控件树里找不到目标时，才根据截图给出 x/y。
- {text_rule}
- 不要把同一个动作重复做第二次；最近操作里已经点过的目标，除非画面明显变化，
  否则请换一种做法或判定完成。
- 只输出 JSON。"""
        started = time.time()
        raw = self.client.vision(self.vision_model(), SYSTEM, prompt, shot)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise OllamaError("决策返回不是 JSON 对象")
        return self._to_decision(data, elements, width, height, time.time() - started)

    def choose_node(
        self,
        shot: Any,
        candidates: list[dict],
        *,
        goal: str = "",
        elements: list[dict] | None = None,
        screen_size: tuple[int, int] | None = None,
    ) -> dict:
        """在候选流程节点里挑一个。candidates: [{id,name,notes,similarity}]"""
        width, height = screen_size or (1080, 2400)
        lines = []
        for item in candidates:
            notes = str(item.get("notes") or "").replace("\n", " ")[:100]
            lines.append(
                f'- id={item.get("id")} 名称「{item.get("name")}」'
                f' 备注「{notes}」 图片相似度 {float(item.get("similarity") or 0):.3f}'
            )
        prompt = f"""我正在用手截图匹配的方式操作手机，但现在没有节点达到匹配阈值。
任务目标：{goal or "按流程图继续"}

当前画面可能的流程节点：
{chr(10).join(lines) if lines else "（没有候选）"}

屏幕像素尺寸：{width} x {height}

请判断当前截图最像哪一个节点，输出 JSON：
{{
  "choice": 节点 id 的整数，若都不像则填 -1,
  "confidence": 0~1,
  "reason": "一句话理由"
}}"""
        raw = self.client.vision(self.vision_model(), SYSTEM, prompt, shot)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise OllamaError("节点重排返回不是 JSON 对象")
        try:
            choice = int(data.get("choice", -1))
        except (TypeError, ValueError):
            choice = -1
        return {
            "choice": choice,
            "confidence": max(0.0, min(1.0, _as_float(data.get("confidence")) or 0.0)),
            "reason": str(data.get("reason") or ""),
        }

    # -- 完成复核 --------------------------------------------------------
    def verify_goal(
        self, goal: str, shot: Any, *, elements: list[dict] | None = None
    ) -> dict:
        """只回答「目标是否已经达成」，比开放式规划更容易判断对。"""
        prompt = f"""任务目标：{goal}

请只判断当前这张截图是否已经达成上面的目标。
可点击控件（可能为空）：
{format_elements(elements)}

输出 JSON：
{{
  "done": true 或 false,
  "page": "当前页面简短名称",
  "reason": "一句话说明判断依据"
}}

说明：
- 导航类目标（例如「进入电视台页面」）只要当前页面已经是对应页面，就填 done=true，
  即使目标标签/按钮仍然显示在屏幕上。
- 只有确实还没到位时才填 done=false。"""
        raw = self.client.vision(self.vision_model(), SYSTEM, prompt, shot)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise OllamaError("完成复核返回不是 JSON 对象")
        done = data.get("done")
        if isinstance(done, str):
            done = done.strip().lower() in ("true", "1", "yes", "是")
        return {
            "done": bool(done),
            "page": str(data.get("page") or ""),
            "reason": str(data.get("reason") or ""),
        }

    # -- 生成流程 --------------------------------------------------------
    def plan_flow(self, description: str, screen_size: tuple[int, int] | None = None) -> dict:
        width, height = screen_size or (1080, 2400)
        prompt = f"""把下面这个手机自动化需求拆成 1~6 个连续的步骤。
需求：{description}

要求：
- 每个步骤是一个可以由“看图操作”完成的阶段，例如「打开美团」「搜索黄焖鸡」「加入购物车」「提交订单」。
- 不要写代码，不要写坐标，不要假设用户已经打开了某个应用。
- 步骤要少而清晰；简单需求就用 1 个步骤。
- 屏幕尺寸：{width}x{height}

输出 JSON：
{{
  "name": "不超过 12 个字的流程名",
  "description": "一句话说明这个流程做什么",
  "steps": [
    {{"name": "步骤名（不超过 10 字）", "goal": "这个步骤要达成的具体目标（写给执行者看）", "detail": "可选的注意事项"}}
  ]
}}"""
        text = self.client.text(self.text_model(), SYSTEM, prompt)
        data = extract_json(text)
        if not isinstance(data, dict):
            raise OllamaError("流程生成返回不是 JSON 对象")
        steps = []
        for item in data.get("steps") or []:
            if not isinstance(item, dict):
                continue
            goal = str(item.get("goal") or item.get("detail") or item.get("name") or "").strip()
            if not goal:
                continue
            steps.append(
                {
                    "name": str(item.get("name") or f"步骤 {len(steps) + 1}")[:24],
                    "goal": goal,
                    "detail": str(item.get("detail") or "").strip(),
                }
            )
        if not steps:
            steps = [{"name": description[:24] or "AI 任务", "goal": description, "detail": ""}]
        return {
            "name": str(data.get("name") or description[:24] or "AI 流程")[:40],
            "description": str(data.get("description") or description),
            "steps": steps[:8],
        }

    def plan_project(self, description: str, screen_size: tuple[int, int] | None = None) -> dict:
        """把目标拆成“区块项目”：多个顺序执行的流程区块。"""
        width, height = screen_size or (1080, 2400)
        prompt = f"""把下面这个手机自动化目标拆成一个「区块项目」。
目标：{description}

要求：
- 项目由 1~4 个区块组成，每个区块是一条独立流程，按执行顺序排列。
- 例如「打开美团并搜索」「选店加购」「提交订单并支付」可以是 3 个区块。
- 每个区块 1~3 个步骤，步骤是看屏幕就能执行的动作目标（例如「点击搜索框」「输入黄焖鸡」）。
- 不要写坐标、不要写代码、不要假设用户已经打开了某个应用。
- 屏幕尺寸：{width}x{height}

输出 JSON：
{{
  "name": "项目名（不超过 12 个字）",
  "description": "一句话说明这个项目做什么",
  "blocks": [
    {{
      "name": "区块名（不超过 10 个字）",
      "flow_goal": "这个区块整体要完成什么",
      "steps": [
        {{"name": "步骤名（不超过 10 个字）", "goal": "这一步的具体动作目标"}}
      ]
    }}
  ]
}}"""
        text = self.client.text(self.text_model(), SYSTEM, prompt)
        data = extract_json(text)
        if not isinstance(data, dict):
            raise OllamaError("项目规划返回不是 JSON 对象")
        blocks = []
        for item in data.get("blocks") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("flow_goal") or "").strip()
            if not name:
                continue
            steps = []
            for step in item.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                goal = str(step.get("goal") or step.get("name") or "").strip()
                if not goal:
                    continue
                steps.append(
                    {
                        "name": str(step.get("name") or f"步骤 {len(steps) + 1}")[:24],
                        "goal": goal,
                    }
                )
            flow_goal = str(item.get("flow_goal") or item.get("goal") or name).strip()
            if not steps:
                steps = [{"name": name[:24], "goal": flow_goal}]
            blocks.append({"name": name[:40], "flow_goal": flow_goal, "steps": steps[:4]})
        if not blocks:
            blocks = [
                {
                    "name": description[:24] or "AI 任务",
                    "flow_goal": description,
                    "steps": [{"name": "执行目标", "goal": description}],
                }
            ]
        return {
            "name": str(data.get("name") or description[:24] or "AI 项目")[:40],
            "description": str(data.get("description") or description),
            "blocks": blocks[:4],
        }

    # -- 内部 ------------------------------------------------------------
    def _to_decision(
        self,
        data: dict,
        elements: list[dict] | None,
        width: int,
        height: int,
        elapsed: float,
    ) -> Decision:
        action_raw = str(data.get("action") or "none").strip().lower()
        if action_raw == "tap" and str(data.get("done")).lower() in ("true", "1"):
            action_raw = "done"
        action = ACTION_ALIASES.get(action_raw, "none")
        done = bool(data.get("done")) or action == "done"

        x = _as_float(data.get("x"))
        y = _as_float(data.get("y"))
        if x is not None and x > 1.5:
            x = x / max(1, width)
        if y is not None and y > 1.5:
            y = y / max(1, height)
        if x is not None:
            x = _clamp(x)
        if y is not None:
            y = _clamp(y)

        element_index = None
        try:
            if data.get("element_index") is not None:
                element_index = int(data["element_index"])
        except (TypeError, ValueError):
            element_index = None

        element_label = ""
        if element_index is not None and elements:
            for element in elements:
                if int(element.get("index", -1)) != element_index:
                    continue
                bounds = element.get("bounds") or [0, 0, width, height]
                x = _clamp(((bounds[0] + bounds[2]) / 2.0) / max(1, width))
                y = _clamp(((bounds[1] + bounds[3]) / 2.0) / max(1, height))
                element_label = (
                    str(element.get("text") or "")
                    or str(element.get("description") or "")
                    or str(element.get("resource_id") or "").split("/")[-1]
                )
                break
            else:
                element_index = None

        key = str(data.get("key") or "back").strip().lower() or "back"
        seconds = _as_float(data.get("seconds")) or 1.0
        seconds = max(0.1, min(10.0, seconds))

        return Decision(
            action=action,
            x=x,
            y=y,
            element_index=element_index,
            element_label=element_label,
            text=str(data.get("text") or ""),
            key=key,
            seconds=seconds,
            page=str(data.get("page") or ""),
            reason=str(data.get("thought") or data.get("reason") or ""),
            done=done,
            confidence=max(0.0, min(1.0, _as_float(data.get("confidence")) or 0.0)),
            elapsed=elapsed,
            raw=data,
        )


# ---------------------------------------------------------------------------
# 不依赖流程图的极简智能体
# ---------------------------------------------------------------------------


class AgentRunner:
    """看一步、做一步，直到模型说完成或达到步数上限。

    用于命令行 `python -m scflow --ai-task "打开设置看电池电量"`，
    也可以被界面复用。
    """

    def __init__(
        self,
        device,
        goal: str,
        *,
        settings: OllamaSettings | None = None,
        brain: ScreenBrain | None = None,
        event_cb: Callable[[dict], None] | None = None,
        max_steps: int = 25,
        interval: float = 0.8,
    ):
        self.device = device
        self.goal = goal.strip()
        self.settings = settings or load_settings()
        self.brain = brain or ScreenBrain(self.settings)
        self.event_cb = event_cb
        self.max_steps = max(1, int(max_steps))
        self.interval = interval
        self.history: list[dict] = []
        self._stop = False
        self.error = ""
        self.steps = 0

    def emit(self, event_type: str, **payload) -> None:
        if not self.event_cb:
            return
        try:
            self.event_cb({"type": event_type, "time": time.time(), **payload})
        except Exception:
            pass

    def stop(self) -> None:
        self._stop = True

    def _ui_elements(self) -> list[dict]:
        if not self.settings.ui_context:
            return []
        getter = getattr(self.device, "ui_elements", None)
        if getter is None:
            return []
        try:
            return list(getter() or [])
        except Exception as exc:
            self.emit("warn", message=f"读取控件树失败（将只看截图）：{exc}")
            return []

    def _apply(self, decision: Decision) -> str:
        if decision.action == "tap" and decision.x is not None and decision.y is not None:
            height, width = self._last_size
            self.device.tap(decision.x * width, decision.y * height)
            return decision.describe()
        if decision.action == "swipe":
            height, width = self._last_size
            self.device.swipe(width * 0.5, height * 0.78, width * 0.5, height * 0.28, 320)
            return "AI 上滑"
        if decision.action == "back":
            self.device.key("back")
            return "AI 返回"
        if decision.action == "key":
            self.device.key(decision.key)
            return f"AI 按键 {decision.key}"
        if decision.action == "text":
            self.device.text(decision.text)
            return decision.describe()
        if decision.action == "wait":
            time.sleep(decision.seconds)
            return decision.describe()
        return decision.describe()

    def run(self) -> dict:
        self.emit("ai_agent_start", goal=self.goal, max_steps=self.max_steps)
        status = "finished"
        try:
            if not self.brain.enabled:
                raise OllamaError("AI 已在设置中关闭")
            self.brain.vision_model()
        except OllamaError as exc:
            self.error = str(exc)
            self.emit("error", message=self.error)
            return {"status": "error", "steps": 0, "error": self.error}

        for index in range(self.max_steps):
            if self._stop:
                status = "stopped"
                break
            shot = self.device.screenshot()
            height, width = shot.shape[:2]
            self._last_size = (height, width)
            self.emit("screenshot", frame=shot, step=index + 1)
            elements = self._ui_elements()
            try:
                decision = self.brain.decide(
                    self.goal,
                    shot,
                    elements=elements,
                    history=self.history,
                    screen_size=(width, height),
                )
            except OllamaError as exc:
                self.error = str(exc)
                self.emit("error", message=self.error)
                status = "error"
                break
            self.steps += 1
            detail = self._apply(decision) if not decision.done else "任务完成"
            self.history.append(
                {
                    "page": decision.page,
                    "action": decision.action,
                    "detail": detail,
                }
            )
            self.emit(
                "ai_step",
                step=index + 1,
                page=decision.page,
                thought=decision.reason,
                action=detail,
                done=decision.done,
                confidence=decision.confidence,
                elapsed=decision.elapsed,
            )
            if decision.done:
                status = "finished"
                break
            if self._stop:
                status = "stopped"
                break
            time.sleep(self.interval)
        else:
            status = "max_steps"

        self.emit("ai_agent_finish", status=status, steps=self.steps, error=self.error)
        return {"status": status, "steps": self.steps, "error": self.error}
