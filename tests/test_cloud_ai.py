from __future__ import annotations

import io
import json
import os
import unittest
from unittest import mock

import numpy as np

from scflow import ai, cloud_ai, ollama
from tests.helpers import temp_home


class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_settings(**kwargs) -> cloud_ai.CloudSettings:
    settings = cloud_ai.CloudSettings(
        enabled=True,
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="test-model",
    )
    for key, value in kwargs.items():
        setattr(settings, key, value)
    return settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(lambda: os.environ.pop("SCFLOW_HOME", None))

    def test_save_and_load_roundtrip(self):
        settings = make_settings()
        cloud_ai.save_settings(settings)
        loaded = cloud_ai.load_settings()
        self.assertEqual(loaded.base_url, "https://api.example.com/v1")
        self.assertEqual(loaded.api_key, "sk-test")
        self.assertTrue(loaded.enabled)

    def test_configured_requires_key(self):
        self.assertFalse(cloud_ai.CloudSettings().configured)
        self.assertTrue(make_settings().configured)

    def test_invalid_file_is_ignored(self):
        path = cloud_ai.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json", encoding="utf-8")
        loaded = cloud_ai.load_settings()
        self.assertEqual(loaded.model, cloud_ai.DEFAULT_MODEL)


class PresetTests(unittest.TestCase):
    def test_presets_cover_common_services(self):
        keys = [key for _label, key in cloud_ai.preset_choices()]
        for key in ("deepseek", "openai", "lmstudio"):
            self.assertIn(key, keys)
        url, model = cloud_ai.preset_values("deepseek")
        self.assertIn("deepseek", url)
        self.assertTrue(model)
        self.assertIsNone(cloud_ai.preset_values("nope"))

    def test_default_backend_is_auto(self):
        self.assertEqual(ollama.OllamaSettings().backend, "auto")


class ClientTests(unittest.TestCase):
    def test_text_request_shape(self):
        settings = make_settings()
        client = cloud_ai.CloudClient(settings)
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"choices": [{"message": {"content": "hello"}}]})

        with mock.patch.object(cloud_ai.urllib.request, "urlopen", fake_urlopen):
            reply = client.text("test-model", "system", "user")
        self.assertEqual(reply, "hello")
        self.assertEqual(captured["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "test-model")
        self.assertEqual(captured["payload"]["messages"][1]["content"], "user")
        self.assertIn("Authorization", captured["headers"])

    def test_vision_encodes_image(self):
        settings = make_settings()
        client = cloud_ai.CloudClient(settings)
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"choices": [{"message": {"content": "{\"ok\": true}"}}]})

        image = np.zeros((40, 30, 3), dtype=np.uint8)
        with mock.patch.object(cloud_ai.urllib.request, "urlopen", fake_urlopen):
            client.vision("", "system", "看这张图", image)
        content = captured["payload"]["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_http_error_becomes_cloud_error(self):
        settings = make_settings()
        client = cloud_ai.CloudClient(settings)
        error = cloud_ai.urllib.error.HTTPError(
            "https://api.example.com/v1/chat/completions",
            401,
            "unauthorized",
            {},
            io.BytesIO(b'{"error":"bad key"}'),
        )
        with mock.patch.object(cloud_ai.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(cloud_ai.CloudError) as caught:
                client.text("m", "s", "u")
        self.assertIn("401", str(caught.exception))

    def test_list_models(self):
        settings = make_settings()
        client = cloud_ai.CloudClient(settings)

        def fake_urlopen(request, timeout=None):
            return FakeResponse({"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]})

        with mock.patch.object(cloud_ai.urllib.request, "urlopen", fake_urlopen):
            models = client.list_models()
        self.assertEqual(models, ["gpt-4o", "gpt-4o-mini"])


class BrainSelectionTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(lambda: os.environ.pop("SCFLOW_HOME", None))
        cloud_ai.save_settings(make_settings())
        self.local = ollama.OllamaSettings()
        self.local.enabled = True

    def test_cloud_backend(self):
        self.local.backend = "cloud"
        brain = cloud_ai.create_brain(self.local)
        self.assertIsInstance(brain, ai.ScreenBrain)
        self.assertIsInstance(brain.client, cloud_ai.CloudClient)

    def test_auto_backend_wraps_fallback(self):
        self.local.backend = "auto"
        brain = cloud_ai.create_brain(self.local)
        self.assertIsInstance(brain, cloud_ai.FallbackBrain)

    def test_local_backend(self):
        self.local.backend = "local"
        brain = cloud_ai.create_brain(self.local)
        self.assertIsInstance(brain, ai.ScreenBrain)
        self.assertNotIsInstance(brain.client, cloud_ai.CloudClient)

    def test_cloud_disabled_falls_back_to_local(self):
        settings = make_settings(enabled=False)
        cloud_ai.save_settings(settings)
        self.local.backend = "cloud"
        brain = cloud_ai.create_brain(self.local)
        self.assertIsInstance(brain, ai.ScreenBrain)
        self.assertNotIsInstance(brain.client, cloud_ai.CloudClient)

    def test_nothing_enabled_returns_none(self):
        self.local.enabled = False
        self.local.backend = "local"
        self.assertIsNone(cloud_ai.create_brain(self.local))

    def test_fallback_brain_uses_secondary(self):
        class Broken:
            settings = None

            def decide(self, *args, **kwargs):
                raise cloud_ai.CloudError("down")

        class Good:
            settings = None

            def decide(self, *args, **kwargs):
                return {"action": "tap"}

        brain = cloud_ai.FallbackBrain(Broken(), Good())
        self.assertEqual(brain.decide("goal", None), {"action": "tap"})


if __name__ == "__main__":
    unittest.main()
