from __future__ import annotations

import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from scflow import ollama
from tests.helpers import temp_home


class FakeOllama:
    """最小可用的 Ollama HTTP 假服务，用于测试客户端而不启动真实模型。"""

    def __init__(self):
        self.models = [
            {"name": "gemma:2b", "size": 1},
            {"name": "qwen3.5:4b", "size": 3},
        ]
        self.capabilities = {
            "gemma:2b": ["completion", "tools"],
            "qwen3.5:4b": ["completion", "vision", "thinking"],
        }
        self.chat_content = '{"ok": true}'
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # noqa: D401 - 静默
                return

            def _send(self, payload: dict, status: int = 200):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path == "/api/version":
                    self._send({"version": "0.0-test"})
                elif self.path == "/api/tags":
                    self._send({"models": owner.models})
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                owner.requests.append({"path": self.path, "payload": payload})
                if self.path == "/api/show":
                    model = payload.get("model", "")
                    self._send({"capabilities": owner.capabilities.get(model, ["completion"])})
                elif self.path == "/api/chat":
                    self._send(
                        {
                            "model": payload.get("model"),
                            "message": {"role": "assistant", "content": owner.chat_content},
                            "eval_count": 12,
                        }
                    )
                else:
                    self._send({"error": "not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class OllamaClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.addCleanup(self.fake.close)
        self.settings = ollama.OllamaSettings(base_url=self.fake.base_url, timeout=10)

    def test_ping_and_status(self):
        client = ollama.OllamaClient(settings=self.settings)
        ok, version = client.ping()
        self.assertTrue(ok)
        self.assertEqual(version, "0.0-test")

        info = client.status()
        self.assertTrue(info["available"])
        self.assertEqual(info["vision_model"], "qwen3.5:4b")
        self.assertEqual(info["text_model"], "qwen3.5:4b")
        self.assertEqual(len(info["models"]), 2)

    def test_vision_capability_detection(self):
        client = ollama.OllamaClient(settings=self.settings)
        self.assertTrue(client.supports_vision("qwen3.5:4b"))
        self.assertFalse(client.supports_vision("gemma:2b"))

    def test_resolve_vision_reports_pull_hint(self):
        settings = ollama.OllamaSettings(
            base_url=self.fake.base_url, vision_model="llava:7b"
        )
        client = ollama.OllamaClient(settings=settings)
        with self.assertRaises(ollama.OllamaError) as caught:
            client.resolve_vision_model()
        self.assertIn("ollama pull", str(caught.exception))

    def test_vision_sends_base64_image(self):
        client = ollama.OllamaClient(settings=self.settings)
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        client.vision("qwen3.5:4b", "sys", "user", frame)
        payload = self.fake.requests[-1]["payload"]
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0.1)
        user_message = [m for m in payload["messages"] if m["role"] == "user"][-1]
        self.assertTrue(user_message["images"])
        decoded = Image.open(io.BytesIO(__import__("base64").b64decode(user_message["images"][0])))
        self.assertLessEqual(max(decoded.size), 720)

    def test_connection_error_is_friendly(self):
        settings = ollama.OllamaSettings(base_url="http://127.0.0.1:1", timeout=3)
        client = ollama.OllamaClient(settings=settings)
        with self.assertRaises(ollama.OllamaError) as caught:
            client.model_names(refresh=True)
        self.assertIn("ollama serve", str(caught.exception))


class EncodingTests(unittest.TestCase):
    def test_encode_image_downscales_numpy(self):
        frame = np.full((2400, 1080, 3), 128, dtype=np.uint8)
        encoded = ollama.encode_image(frame, max_side=720, quality=80)
        image = Image.open(io.BytesIO(__import__("base64").b64decode(encoded)))
        self.assertEqual(max(image.size), 720)
        self.assertAlmostEqual(image.size[0] / image.size[1], 1080 / 2400, places=2)

    def test_encode_image_accepts_pil(self):
        encoded = ollama.encode_image(Image.new("RGB", (50, 50), "red"))
        self.assertTrue(encoded)


class JsonExtractionTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(ollama.extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        text = '```json\n{"action": "tap", "x": 0.5}\n```'
        self.assertEqual(ollama.extract_json(text)["action"], "tap")

    def test_json_with_prose_and_braces_in_strings(self):
        text = '好的，结果如下：{"note": "包含 } 和 { 的字符串", "n": 2} 完毕。'
        parsed = ollama.extract_json(text)
        self.assertEqual(parsed["n"], 2)
        self.assertIn("}", parsed["note"])

    def test_array(self):
        self.assertEqual(ollama.extract_json("prefix [1, 2] suffix"), [1, 2])

    def test_invalid_raises(self):
        with self.assertRaises(ollama.OllamaError):
            ollama.extract_json("没有任何 JSON")


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(lambda: os.environ.pop("SCFLOW_HOME", None))
        for key in (
            "SCFLOW_OLLAMA_URL",
            "SCFLOW_OLLAMA_MODEL",
            "SCFLOW_OLLAMA_TEXT_MODEL",
            "SCFLOW_OLLAMA_TIMEOUT",
            "SCFLOW_OLLAMA_DISABLE",
        ):
            os.environ.pop(key, None)

    def test_roundtrip(self):
        settings = ollama.load_settings()
        self.assertTrue(settings.enabled)
        settings.vision_model = "qwen3.5:4b"
        settings.ai_rerank = False
        settings.max_image_side = 640
        path = ollama.save_settings(settings)
        self.assertTrue(path.exists())

        reloaded = ollama.load_settings()
        self.assertEqual(reloaded.vision_model, "qwen3.5:4b")
        self.assertFalse(reloaded.ai_rerank)
        self.assertEqual(reloaded.max_image_side, 640)

    def test_env_overrides(self):
        os.environ["SCFLOW_OLLAMA_URL"] = "http://192.168.1.9:11434"
        os.environ["SCFLOW_OLLAMA_MODEL"] = "llava:7b"
        os.environ["SCFLOW_OLLAMA_DISABLE"] = "true"
        settings = ollama.load_settings()
        self.assertEqual(settings.base_url, "http://192.168.1.9:11434")
        self.assertEqual(settings.vision_model, "llava:7b")
        self.assertFalse(settings.enabled)

    def test_unknown_keys_are_ignored(self):
        settings = ollama.OllamaSettings.from_dict({"enabled": False, "bogus": 1})
        self.assertFalse(settings.enabled)


if __name__ == "__main__":
    unittest.main()
