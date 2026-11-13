"""本地 Ollama 接入层：零第三方依赖的 HTTP 客户端 + 持久化设置。

设计目标：
- 不引入 requests / httpx，只用标准库，保持项目“复制文件夹即可运行”的特性；
- 所有调用都有超时与友好错误；Ollama 没启动时，程序照常按图片流程运行；
- 模型能力（completion / vision / tools / thinking）从 /api/show 自动探测，自动挑选视觉模型。
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import config

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_VISION_MODEL = "qwen3.5:4b"
DEFAULT_TEXT_MODEL = "qwen3.5:4b"

# 名字里带这些片段的模型通常支持看图；真正的判断仍以 /api/show 的 capabilities 为准。
VISION_NAME_HINTS = (
    "vl",
    "vision",
    "llava",
    "moondream",
    "minicpm-v",
    "qwen3.5",
    "qwen3-vl",
    "qwen2.5vl",
    "gemma3",
    "granite3.2-vision",
    "llama3.2-vision",
    "llama4",
    "mistral-small3",
    "pixtral",
    "internvl",
)


class OllamaError(RuntimeError):
    """Ollama 不可用、模型不存在或返回格式错误。"""


@dataclass
class OllamaSettings:
    """全局 AI 设置，保存在 data/ollama.json。"""

    enabled: bool = True
    base_url: str = DEFAULT_BASE_URL
    vision_model: str = ""  # 留空表示自动挑选
    text_model: str = ""  # 留空表示自动挑选（可回退到视觉模型）
    backend: str = "auto"  # local / cloud / auto（auto＝配了云端就优先云端，否则用本地）
    temperature: float = 0.1
    timeout: float = 180.0
    max_image_side: int = 720
    image_quality: int = 80
    think: bool = False
    # 功能开关
    ai_nodes: bool = True  # AI 决策节点
    ai_fallback: bool = True  # 图片匹配失败时由 AI 接管
    ai_rerank: bool = True  # 匹配不确定时由 AI 选节点
    ui_context: bool = True  # 把 uiautomator 控件树作为上下文
    allow_text: bool = True  # 允许 AI 决定输入文本
    max_decisions: int = 12  # 单个 AI 节点最多决策次数
    # 运行期信息（不写入磁盘）
    last_error: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("last_error", None)
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "OllamaSettings":
        settings = cls()
        if not isinstance(data, dict):
            return settings
        for key, value in data.items():
            if key == "last_error" or not hasattr(settings, key):
                continue
            current = getattr(settings, key)
            try:
                if isinstance(current, bool):
                    setattr(settings, key, bool(value))
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(settings, key, int(value))
                elif isinstance(current, float):
                    setattr(settings, key, float(value))
                elif isinstance(current, str):
                    setattr(settings, key, str(value))
            except (TypeError, ValueError):
                continue
        return settings


def settings_path() -> Path:
    return config.home() / "ollama.json"


def load_settings() -> OllamaSettings:
    path = settings_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    settings = OllamaSettings.from_dict(data)

    # 环境变量优先，方便命令行 / 打包后临时覆盖
    env = os.environ
    if env.get("SCFLOW_OLLAMA_URL"):
        settings.base_url = env["SCFLOW_OLLAMA_URL"].strip()
    if env.get("SCFLOW_OLLAMA_MODEL"):
        settings.vision_model = env["SCFLOW_OLLAMA_MODEL"].strip()
    if env.get("SCFLOW_OLLAMA_TEXT_MODEL"):
        settings.text_model = env["SCFLOW_OLLAMA_TEXT_MODEL"].strip()
    if env.get("SCFLOW_OLLAMA_TIMEOUT"):
        try:
            settings.timeout = float(env["SCFLOW_OLLAMA_TIMEOUT"])
        except ValueError:
            pass
    if env.get("SCFLOW_OLLAMA_DISABLE", "").strip().lower() in ("1", "true", "yes", "on"):
        settings.enabled = False
    return settings


def save_settings(settings: OllamaSettings) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# 图片编码
# ---------------------------------------------------------------------------


def encode_image(image: Any, max_side: int = 720, quality: int = 80) -> str:
    """把截图压成 base64 JPEG。图片越小，本地视觉模型越快。"""
    from PIL import Image

    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            pil = handle.convert("RGB")
    else:
        import numpy as np

        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype("uint8")
        if array.shape[-1] == 4:
            array = array[:, :, :3]
        pil = Image.fromarray(array, "RGB")

    width, height = pil.size
    longest = max(width, height)
    if max_side > 0 and longest > max_side:
        scale = max_side / longest
        pil = pil.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=max(30, min(95, int(quality))))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON，容忍 ```json 代码块和前后废话。"""
    if not text:
        raise OllamaError("模型没有返回内容")
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    try:
        return json.loads(raw)
    except ValueError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    snippet = raw[start : index + 1]
                    try:
                        return json.loads(snippet)
                    except ValueError:
                        break
    raise OllamaError("模型返回的不是合法 JSON：" + text[:200])


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class OllamaClient:
    """极简 Ollama REST 客户端，线程安全（内部串行化请求）。"""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        settings: OllamaSettings | None = None,
    ):
        if isinstance(base_url, OllamaSettings):  # 容忍把 settings 当第一个参数传
            settings = base_url
            base_url = None
        self.settings = settings or load_settings()
        self.base_url = (base_url or self.settings.base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout or self.settings.timeout or 180.0)
        self._lock = threading.RLock()
        self._tags_cache: tuple[float, list[dict]] | None = None
        self._capability_cache: dict[str, tuple[float, dict]] = {}

    # -- 低层 ------------------------------------------------------------
    def request(
        self, path: str, payload: dict | None = None, timeout: float | None = None
    ) -> dict:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with self._lock:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as resp:
                    body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise OllamaError(f"Ollama 返回 HTTP {exc.code}：{detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaError(
                f"连不上本地 Ollama（{self.base_url}）：{reason}。"
                "请确认已运行 `ollama serve`，或到「AI 设置」里修改地址。"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama 响应超时（{timeout or self.timeout:.0f}s）") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError as exc:
            raise OllamaError("Ollama 返回了非 JSON 响应") from exc

    # -- 元信息 ----------------------------------------------------------
    def ping(self) -> tuple[bool, str]:
        try:
            body = self.request("/api/version", timeout=5)
            return True, str(body.get("version", "?"))
        except OllamaError as exc:
            return False, str(exc)

    def tags(self, refresh: bool = False) -> list[dict]:
        now = time.time()
        if not refresh and self._tags_cache and now - self._tags_cache[0] < 10:
            return self._tags_cache[1]
        body = self.request("/api/tags", timeout=10)
        models = body.get("models") or []
        self._tags_cache = (now, models)
        return models

    def model_names(self, refresh: bool = False) -> list[str]:
        return [str(item.get("name") or item.get("model") or "") for item in self.tags(refresh)]

    def show(self, model: str) -> dict:
        now = time.time()
        cached = self._capability_cache.get(model)
        if cached and now - cached[0] < 600:
            return cached[1]
        body = self.request("/api/show", {"model": model}, timeout=20)
        self._capability_cache[model] = (now, body)
        return body

    def capabilities(self, model: str) -> set[str]:
        try:
            body = self.show(model)
        except OllamaError:
            return set()
        caps = body.get("capabilities")
        if isinstance(caps, list):
            return {str(item) for item in caps}
        return set()

    def supports_vision(self, model: str) -> bool:
        caps = self.capabilities(model)
        if caps:
            return "vision" in caps
        lowered = model.lower()
        return any(hint in lowered for hint in VISION_NAME_HINTS)

    def resolve_vision_model(self) -> str:
        names = self.model_names()
        wanted = (self.settings.vision_model or "").strip()
        if wanted:
            if not names or wanted in names:
                return wanted
            raise OllamaError(
                f"视觉模型 {wanted} 不在这台机器的 Ollama 里，请先执行：ollama pull {wanted}"
            )
        for name in names:
            if self.supports_vision(name):
                return name
        for hint in (DEFAULT_VISION_MODEL, "qwen2.5vl:3b", "llava:7b", "moondream"):
            if hint in names:
                return hint
        raise OllamaError(
            "没有找到支持看图的本地模型。建议执行：ollama pull qwen2.5vl:3b（或 llava:7b），"
            "再到「AI 设置」里选择视觉模型。"
        )

    def resolve_text_model(self, allow_vision: bool = True) -> str:
        names = self.model_names()
        wanted = (self.settings.text_model or "").strip()
        if wanted:
            if not names or wanted in names:
                return wanted
            raise OllamaError(f"文本模型 {wanted} 不在这台机器的 Ollama 里")
        if allow_vision:
            try:
                return self.resolve_vision_model()
            except OllamaError:
                pass
        if names:
            return names[0]
        raise OllamaError("Ollama 里还没有任何模型，请先执行：ollama pull qwen3.5:4b")

    def status(self) -> dict:
        info: dict = {
            "available": False,
            "base_url": self.base_url,
            "version": "",
            "models": [],
            "vision_model": "",
            "text_model": "",
            "error": "",
        }
        ok, detail = self.ping()
        if not ok:
            info["error"] = detail
            return info
        info["available"] = True
        info["version"] = detail
        try:
            models = self.model_names(refresh=True)
            info["models"] = models
            info["vision_model"] = self.resolve_vision_model()
            info["text_model"] = self.resolve_text_model()
        except OllamaError as exc:
            info["error"] = str(exc)
        return info

    # -- 推理 ------------------------------------------------------------
    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        json_mode: bool = True,
        think: bool | None = None,
        images: list[str] | None = None,
        options: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        messages = [dict(item) for item in messages]
        if images:
            for message in reversed(messages):
                if message.get("role") == "user":
                    message["images"] = list(images)
                    break
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(self.settings.temperature),
                "num_ctx": 8192,
                "num_predict": 512,
                **(options or {}),
            },
        }
        if json_mode:
            payload["format"] = "json"
        payload["think"] = bool(self.settings.think if think is None else think)
        started = time.time()
        body = self.request("/api/chat", payload, timeout=timeout)
        if body.get("error"):
            raise OllamaError(str(body["error"]))
        body["_elapsed"] = time.time() - started
        return body

    def text(
        self,
        model: str,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
        timeout: float | None = None,
    ) -> str:
        body = self.chat(
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            json_mode=json_mode,
            timeout=timeout,
        )
        return str((body.get("message") or {}).get("content") or "")

    def vision(
        self,
        model: str,
        system: str,
        user: str,
        image: Any,
        *,
        json_mode: bool = True,
        timeout: float | None = None,
    ) -> str:
        encoded = encode_image(
            image,
            max_side=int(self.settings.max_image_side),
            quality=int(self.settings.image_quality),
        )
        body = self.chat(
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [encoded]},
            ],
            json_mode=json_mode,
            images=[encoded],
            timeout=timeout,
        )
        return str((body.get("message") or {}).get("content") or "")


def get_client(settings: OllamaSettings | None = None) -> OllamaClient:
    return OllamaClient(settings=settings or load_settings())


def status_text(info: dict) -> str:
    if not info.get("available"):
        suffix = f"：{info['error']}" if info.get("error") else ""
        return f"未连接（{info.get('base_url')}）{suffix}"
    return (
        f"已连接 v{info.get('version')} · 视觉模型 {info.get('vision_model') or '-'} · "
        f"文本模型 {info.get('text_model') or '-'}"
    )
