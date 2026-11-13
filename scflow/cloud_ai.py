"""云端 AI（OpenAI 兼容 API）接入层。

- 只依赖标准库 + Pillow（与本地 Ollama 层保持一致，复制文件夹即可运行）；
- 设置保存在 data/cloud.json，可用环境变量覆盖；
- CloudClient 实现与 OllamaClient 相同的 text/vision/resolve_*_model 接口，
  所以可以直接喂给 ai.ScreenBrain，本地能用的 AI 能力云端也能用。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import config
from .ollama import OllamaError, encode_image

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# 常见服务预设：界面上选一下就能填好地址和模型
PRESETS = {
    "deepseek": ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    "dashscope": ("通义千问（阿里云）", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max"),
    "lmstudio": ("本地 LM Studio", "http://127.0.0.1:1234/v1", "local-model"),
    "ollama-openai": ("本地 Ollama（OpenAI 兼容）", "http://127.0.0.1:11434/v1", "qwen3.5:4b"),
}


def preset_choices() -> list[tuple[str, str]]:
    """给下拉框用的 (显示名, key)。"""
    return [(label, key) for key, (label, _url, _model) in PRESETS.items()]


def preset_values(key: str) -> tuple[str, str] | None:
    item = PRESETS.get(str(key or "").strip().lower())
    if not item:
        return None
    return item[1], item[2]


class CloudError(OllamaError):
    """云端 API 不可用或返回格式错误。"""


@dataclass
class CloudSettings:
    enabled: bool = False
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    vision_model: str = ""
    temperature: float = 0.1
    timeout: float = 180.0
    max_image_side: int = 720
    image_quality: int = 80
    think: bool = False
    # 功能开关，语义与 OllamaSettings 一致
    ai_nodes: bool = True
    ai_fallback: bool = True
    ai_rerank: bool = True
    ui_context: bool = True
    allow_text: bool = True
    max_decisions: int = 12
    last_error: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip()) and bool(self.base_url.strip())

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("last_error", None)
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "CloudSettings":
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
                elif isinstance(current, int):
                    setattr(settings, key, int(value))
                elif isinstance(current, float):
                    setattr(settings, key, float(value))
                elif isinstance(current, str):
                    setattr(settings, key, str(value))
            except (TypeError, ValueError):
                continue
        return settings


def settings_path() -> Path:
    return config.home() / "cloud.json"


def load_settings() -> CloudSettings:
    path = settings_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    settings = CloudSettings.from_dict(data)
    env = os.environ
    if env.get("SCFLOW_CLOUD_URL"):
        settings.base_url = env["SCFLOW_CLOUD_URL"].strip()
    if env.get("SCFLOW_CLOUD_API_KEY"):
        settings.api_key = env["SCFLOW_CLOUD_API_KEY"].strip()
    if env.get("SCFLOW_CLOUD_MODEL"):
        settings.model = env["SCFLOW_CLOUD_MODEL"].strip()
    if env.get("SCFLOW_CLOUD_ENABLED"):
        settings.enabled = env["SCFLOW_CLOUD_ENABLED"].strip().lower() in ("1", "true", "yes", "on")
    return settings


def save_settings(settings: CloudSettings) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


class CloudClient:
    """OpenAI 兼容 /chat/completions 客户端；接口与 OllamaClient 对齐。"""

    def __init__(self, settings: CloudSettings | None = None):
        self.settings = settings or load_settings()

    # -- 基础 ------------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        settings = self.settings
        if not settings.api_key.strip():
            raise CloudError("云端 API 未填写 Key")
        base = settings.base_url.strip().rstrip("/")
        url = f"{base}{path}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.api_key.strip()}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or settings.timeout) as response:
                body = response.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode("utf-8", "ignore")
            raise CloudError(f"云端 API 错误 {exc.code}：{detail}") from exc
        except Exception as exc:  # noqa: BLE001 - 统一成可读错误
            raise CloudError(f"云端 API 连接失败：{exc}") from exc
        try:
            return json.loads(body)
        except ValueError as exc:
            raise CloudError(f"云端 API 返回不是 JSON：{body[:200]}") from exc

    def _chat(self, model: str, messages: list[dict], timeout: float | None = None) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(self.settings.temperature),
            "stream": False,
        }
        data = self._post("/chat/completions", payload, timeout=timeout)
        choices = data.get("choices") or []
        content: Any = ""
        if choices:
            content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        return str(content or "").strip()

    # -- 与 ScreenBrain 对接 ---------------------------------------------
    def resolve_vision_model(self) -> str:
        return self.settings.vision_model.strip() or self.settings.model.strip()

    def resolve_text_model(self, allow_vision: bool = True) -> str:
        return self.settings.model.strip()

    def text(
        self,
        model: str,
        system: str,
        user: str,
        json_mode: bool = True,
        timeout: float | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._chat(model or self.resolve_text_model(), messages, timeout)

    def vision(
        self,
        model: str,
        system: str,
        user: str,
        image: Any,
        json_mode: bool = True,
        timeout: float | None = None,
    ) -> str:
        encoded = encode_image(
            image,
            max_side=int(self.settings.max_image_side or 720),
            quality=int(self.settings.image_quality or 80),
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            },
        ]
        return self._chat(model or self.resolve_vision_model(), messages, timeout)

    # -- 连接测试 --------------------------------------------------------
    def list_models(self) -> list[str]:
        settings = self.settings
        if not settings.configured:
            raise CloudError("云端 API 未配置（需要 Base URL 和 Key）")
        url = settings.base_url.strip().rstrip("/") + "/models"
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {settings.api_key.strip()}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=min(30.0, settings.timeout)) as response:
                data = json.loads(response.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as exc:
            raise CloudError(f"云端 API 错误 {exc.code}") from exc
        except Exception as exc:  # noqa: BLE001
            raise CloudError(f"无法连接云端 API：{exc}") from exc
        items = data.get("data") or data.get("models") or []
        return [str(item.get("id") or item.get("name") or "") for item in items if isinstance(item, dict)]


class FallbackBrain:
    """auto 模式：云端优先，出错自动用本地模型重试。"""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary
        self.settings = primary.settings

    @property
    def enabled(self) -> bool:
        return True

    def _call(self, name: str, *args, **kwargs):
        try:
            return getattr(self.primary, name)(*args, **kwargs)
        except Exception:  # noqa: BLE001 - 云端失败必须回退本地
            return getattr(self.secondary, name)(*args, **kwargs)

    def describe(self, *args, **kwargs):
        return self._call("describe", *args, **kwargs)

    def decide(self, *args, **kwargs):
        return self._call("decide", *args, **kwargs)

    def choose_node(self, *args, **kwargs):
        return self._call("choose_node", *args, **kwargs)

    def verify_goal(self, *args, **kwargs):
        return self._call("verify_goal", *args, **kwargs)

    def plan_flow(self, *args, **kwargs):
        return self._call("plan_flow", *args, **kwargs)

    def plan_project(self, *args, **kwargs):
        return self._call("plan_project", *args, **kwargs)


def create_brain(local_settings=None, log=None):
    """按「AI 后端」选择本地 / 云端 / 自动，返回 ScreenBrain 或 FallbackBrain。"""
    from . import ai as ai_module

    local = local_settings
    if local is None:
        from . import ollama

        local = ollama.load_settings()
    cloud = load_settings()
    backend = str(getattr(local, "backend", "local") or "local")
    cloud_ready = bool(cloud.enabled and cloud.configured)
    if backend in ("cloud", "auto") and cloud_ready:
        primary = ai_module.ScreenBrain(cloud, client=CloudClient(cloud), log=log)
        if backend == "cloud":
            return primary
        if getattr(local, "enabled", False):
            return FallbackBrain(primary, ai_module.ScreenBrain(local, log=log))
        return primary
    if getattr(local, "enabled", False):
        return ai_module.ScreenBrain(local, log=log)
    return None
