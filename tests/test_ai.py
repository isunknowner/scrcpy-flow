from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

from scflow import ai, ollama
from scflow.db import Database
from scflow.engine import RunEngine
from tests.helpers import FakeDevice, make_image, save_image, temp_home


class StubClient:
    """按顺序吐出预设 JSON 的假客户端。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def vision(self, model, system, user, image, json_mode=True, timeout=None):
        self.calls.append({"model": model, "system": system, "user": user})
        return self.replies.pop(0) if self.replies else "{}"

    def text(self, model, system, user, json_mode=True, timeout=None):
        self.calls.append({"model": model, "system": system, "user": user})
        return self.replies.pop(0) if self.replies else "{}"

    def resolve_vision_model(self):
        return "qwen3.5:4b"

    def resolve_text_model(self, allow_vision=True):
        return "qwen3.5:4b"


class StubBrain:
    """实现引擎用到的三个方法，不访问网络。"""

    def __init__(self, decisions=None, choice=None, verdict=None):
        self.decisions = list(decisions or [])
        self.choice = choice
        self.verdict = verdict
        self.calls: list[dict] = []

    @property
    def enabled(self):
        return True

    def vision_model(self):
        return "stub"

    def decide(self, goal, shot, **kwargs):
        self.calls.append({"goal": goal, **kwargs})
        if self.decisions:
            return self.decisions.pop(0)
        return ai.Decision(action="done", done=True)

    def verify_goal(self, goal, shot, *, elements=None):
        self.calls.append({"verify": goal})
        if self.verdict is None:
            return {"done": False, "reason": "stub 没有复核结果"}
        return dict(self.verdict)

    def choose_node(self, shot, candidates, **kwargs):
        self.calls.append({"candidates": candidates, **kwargs})
        if self.choice is None:
            return {"choice": -1, "confidence": 0.0, "reason": "stub"}
        return self.choice


def tap_decision(x=0.5, y=0.5, label="按钮") -> ai.Decision:
    return ai.Decision(action="tap", x=x, y=y, element_label=label, reason="点它")


def done_decision() -> ai.Decision:
    return ai.Decision(action="done", done=True, page="完成页", reason="已完成")


class DecisionParsingTests(unittest.TestCase):
    def setUp(self):
        self.settings = ollama.OllamaSettings()
        self.home = temp_home()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(lambda: os.environ.pop("SCFLOW_HOME", None))

    def test_element_index_overrides_coordinates(self):
        reply = json.dumps(
            {
                "page": "外卖首页",
                "thought": "点搜索框",
                "action": "tap",
                "element_index": 3,
                "x": 0.99,
                "y": 0.99,
                "confidence": 0.8,
            }
        )
        brain = ai.ScreenBrain(self.settings, client=StubClient([reply]))
        elements = [
            {"index": 3, "text": "搜索", "bounds": [0, 100, 200, 200], "center": [100, 150]}
        ]
        decision = brain.decide(
            "搜索黄焖鸡", np.zeros((1000, 500, 3), dtype=np.uint8),
            elements=elements, screen_size=(500, 1000),
        )
        self.assertEqual(decision.action, "tap")
        self.assertEqual(decision.element_index, 3)
        self.assertEqual(decision.element_label, "搜索")
        self.assertAlmostEqual(decision.x, 0.2, places=3)
        self.assertAlmostEqual(decision.y, 0.15, places=3)

    def test_pixel_coordinates_are_normalised(self):
        reply = json.dumps({"action": "tap", "x": 540, "y": 1200, "page": "p"})
        brain = ai.ScreenBrain(self.settings, client=StubClient([reply]))
        decision = brain.decide(
            "goal", np.zeros((2400, 1080, 3), dtype=np.uint8),
            screen_size=(1080, 2400),
        )
        self.assertAlmostEqual(decision.x, 0.5, places=3)
        self.assertAlmostEqual(decision.y, 0.5, places=3)

    def test_action_aliases_and_done(self):
        brain = ai.ScreenBrain(self.settings, client=StubClient(["{\"action\": \"click\"}"]))
        self.assertEqual(brain.decide("g", np.zeros((10, 10, 3))).action, "tap")

        brain = ai.ScreenBrain(self.settings, client=StubClient(["{\"action\": \"finish\"}"]))
        decision = brain.decide("g", np.zeros((10, 10, 3)))
        self.assertEqual(decision.action, "done")
        self.assertTrue(decision.done)

    def test_text_action_keeps_text(self):
        reply = json.dumps({"action": "text", "text": "黄焖鸡", "element_index": 2})
        brain = ai.ScreenBrain(self.settings, client=StubClient([reply]))
        decision = brain.decide(
            "g", np.zeros((100, 100, 3)),
            elements=[{"index": 2, "text": "输入框", "bounds": [0, 0, 100, 50]}],
        )
        self.assertEqual(decision.action, "text")
        self.assertEqual(decision.text, "黄焖鸡")

    def test_invalid_json_raises(self):
        brain = ai.ScreenBrain(self.settings, client=StubClient(["不是 JSON"]))
        with self.assertRaises(ollama.OllamaError):
            brain.decide("g", np.zeros((10, 10, 3)))

    def test_format_elements_marks_clickable(self):
        text = ai.format_elements(
            [
                {"index": 0, "text": "确定", "class": "android.widget.Button",
                 "clickable": True, "bounds": [0, 0, 100, 50]},
                {"index": 1, "text": "", "description": "", "class": "android.view.View",
                 "clickable": False, "bounds": [0, 0, 1, 1]},
            ]
        )
        self.assertIn("#0", text)
        self.assertIn("可点", text)
        self.assertIn("确定", text)
        self.assertNotIn("#1", text)

    def test_layout_containers_are_filtered_out(self):
        text = ai.format_elements(
            [
                {
                    "index": 4,
                    "text": "",
                    "description": "",
                    "resource_id": "com.x:id/action_bar_root",
                    "class": "android.widget.FrameLayout",
                    "clickable": False,
                    "bounds": [0, 0, 1080, 200],
                },
                {
                    "index": 5,
                    "text": "电视台",
                    "resource_id": "",
                    "class": "android.widget.TextView",
                    "clickable": True,
                    "bounds": [700, 2200, 900, 2350],
                },
            ]
        )
        # 没有文字、不可点的布局容器不应出现在提示词里
        self.assertNotIn("#4", text)
        self.assertIn("#5", text)
        self.assertIn("可点", text)

    def test_plan_flow_falls_back_to_single_step(self):
        brain = ai.ScreenBrain(self.settings, client=StubClient(["{\"name\":\"\",\"steps\":[]}"]))
        plan = brain.plan_flow("打开设置")
        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(plan["steps"][0]["goal"], "打开设置")

    def test_plan_flow_parses_steps(self):
        reply = json.dumps(
            {
                "name": "点外卖",
                "description": "每天点外卖",
                "steps": [
                    {"name": "打开美团", "goal": "打开美团应用", "detail": ""},
                    {"name": "下单", "goal": "选择第一家常去的店并下单"},
                ],
            }
        )
        brain = ai.ScreenBrain(self.settings, client=StubClient([reply]))
        plan = brain.plan_flow("点一份黄焖鸡")
        self.assertEqual(plan["name"], "点外卖")
        self.assertEqual([step["name"] for step in plan["steps"]], ["打开美团", "下单"])


class AgentRunnerTests(unittest.TestCase):
    def test_runs_until_done(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        device = FakeDevice(frame)
        brain = StubBrain(
            [tap_decision(0.2, 0.3), tap_decision(0.4, 0.5), done_decision()]
        )
        events: list[dict] = []
        runner = ai.AgentRunner(
            device, "打开设置", brain=brain, event_cb=events.append, max_steps=5,
            interval=0,
        )
        result = runner.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["steps"], 3)
        self.assertEqual(device.taps, [(20, 30), (40, 50)])
        self.assertTrue(any(event["type"] == "ai_agent_finish" for event in events))

    def test_stops_on_error(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        device = FakeDevice(frame)

        class Broken(StubBrain):
            def decide(self, goal, shot, **kwargs):
                raise ollama.OllamaError("模型挂了")

        events: list[dict] = []
        runner = ai.AgentRunner(
            device, "目标", brain=Broken(), event_cb=events.append, max_steps=3
        )
        result = runner.run()
        self.assertEqual(result["status"], "error")
        self.assertIn("模型挂了", result["error"])


class EngineAITests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(lambda: os.environ.pop("SCFLOW_HOME", None))
        self.db = Database(Path(self.home.name) / "test.db")
        self.images = Path(self.home.name) / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.settings = ollama.OllamaSettings()
        self.settings.enabled = True

    def _flow(self, name="AI 流程", settings: dict | None = None) -> int:
        ollama.save_settings(self.settings)
        return self.db.create_flow(
            name, description="打开设置并查看电池", settings=settings
        )

    def test_ai_node_runs_until_done(self):
        flow_id = self._flow()
        node_id = self.db.add_node(
            flow_id,
            name="打开设置",
            kind="ai",
            action_type="ai",
            action={"goal": "打开系统设置"},
            is_entry=True,
        )
        frame = np.zeros((800, 400, 3), dtype=np.uint8)
        device = FakeDevice(frame)
        brain = StubBrain([tap_decision(0.5, 0.1), done_decision()])
        engine = RunEngine(self.db, flow_id, device, max_steps=10, brain=brain)
        result = engine.run()

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.steps, 2)
        self.assertEqual(device.taps, [(200, 80)])
        self.assertEqual(engine._ai_done, {node_id})

    def test_ai_node_respects_decision_limit(self):
        flow_id = self._flow()
        self.db.add_node(
            flow_id,
            name="无限任务",
            kind="ai",
            action_type="ai",
            action={"goal": "永远做不完", "max_decisions": 2},
            is_entry=True,
        )
        device = FakeDevice(np.zeros((400, 200, 3), dtype=np.uint8))
        brain = StubBrain([tap_decision(), tap_decision(), tap_decision()])
        engine = RunEngine(self.db, flow_id, device, max_steps=10, brain=brain)
        result = engine.run()
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.status, "finished")

    def test_ai_node_needs_brain_when_disabled(self):
        flow_id = self._flow()
        self.db.add_node(
            flow_id, name="AI", kind="ai", action_type="ai",
            action={"goal": "x"}, is_entry=True,
        )
        self.settings.enabled = False
        ollama.save_settings(self.settings)
        engine = RunEngine(self.db, flow_id, FakeDevice(np.zeros((100, 50, 3), dtype=np.uint8)))
        result = engine.run()
        self.assertEqual(result.status, "error")
        self.assertIn("AI", result.error or "")

    def test_ai_fallback_recovers_when_no_match(self):
        flow_id = self._flow(
            settings={"fallback": {"max_retries": 0, "max_fallbacks": 2}}
        )
        shot = save_image(self.images / "a.png", make_image("blue"))
        self.db.add_node(
            flow_id, name="蓝色页", kind="image", image_path=shot,
            img_w=180, img_h=400, is_entry=True, threshold=0.95,
        )
        # 手机上是一张完全不同的画面，图片匹配必然失败
        device = FakeDevice(np.asarray(make_image("green").convert("RGB")))
        brain = StubBrain([tap_decision(0.3, 0.4)])
        engine = RunEngine(self.db, flow_id, device, max_steps=1, brain=brain)
        result = engine.run()
        self.assertEqual(device.taps, [(54, 160)])
        self.assertTrue(any(call.get("goal") for call in brain.calls))
        self.assertIn(result.status, ("finished", "max_steps", "stopped"))

    def test_ai_rerank_picks_candidate(self):
        flow_id = self._flow()
        blue = save_image(self.images / "blue.png", make_image("blue"))
        green = save_image(self.images / "green.png", make_image("green"))
        source_id = self.db.add_node(
            flow_id, name="蓝色页", kind="image", image_path=blue,
            img_w=180, img_h=400, is_entry=True, threshold=0.99,
        )
        target_id = self.db.add_node(
            flow_id, name="绿色页", kind="image", image_path=green,
            img_w=180, img_h=400, threshold=0.99,
        )
        self.db.add_edge(flow_id, source_id, target_id)
        # 画面接近绿色页，但阈值设得很高，只能靠 AI 重排
        device = FakeDevice(np.asarray(make_image("green").convert("RGB")))
        brain = StubBrain(
            choice={"choice": target_id, "confidence": 0.9, "reason": "看起来是绿色页"}
        )
        engine = RunEngine(self.db, flow_id, device, max_steps=1, brain=brain)
        engine.prepare()
        engine.step()
        self.assertEqual(engine.current_state, target_id)

    def test_ai_node_runs_again_after_flow_loops_back(self):
        flow_id = self._flow()
        a = self.db.add_node(
            flow_id, name="A", kind="ai", action_type="ai",
            action={"goal": "A", "max_decisions": 5}, is_entry=True,
        )
        b = self.db.add_node(
            flow_id, name="B", kind="ai", action_type="ai",
            action={"goal": "B", "max_decisions": 5},
        )
        self.db.add_edge(flow_id, a, b)
        self.db.add_edge(flow_id, b, a)
        brain = StubBrain(
            [tap_decision(), done_decision(), done_decision(), done_decision()]
        )
        engine = RunEngine(
            self.db, flow_id, FakeDevice(np.zeros((100, 100, 3))), max_steps=4, brain=brain
        )
        result = engine.run()
        # A 跑了一轮（tap + done），B done，回到 A 又重新开始计数
        self.assertEqual(result.steps, 4)
        self.assertEqual(engine._ai_cycles.get(a), 1)
        self.assertEqual(brain.calls[0]["goal"], "A")

    def test_ai_node_stops_on_repeated_noop_action(self):
        flow_id = self._flow()
        self.db.add_node(
            flow_id, name="重复无效", kind="ai", action_type="ai",
            action={"goal": "点一个点不动的东西", "max_decisions": 8}, is_entry=True,
        )
        device = FakeDevice(np.zeros((100, 100, 3), dtype=np.uint8))
        brain = StubBrain([tap_decision(0.5, 0.5), tap_decision(0.5, 0.5)])
        engine = RunEngine(self.db, flow_id, device, max_steps=10, brain=brain)
        result = engine.run()
        # 画面没变化 + 同一个动作 = 立刻收尾，不浪费后续决策次数
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.status, "finished")

    def test_ai_node_allows_different_actions_on_static_screen(self):
        flow_id = self._flow()
        self.db.add_node(
            flow_id, name="换位置", kind="ai", action_type="ai",
            action={"goal": "依次点两个地方", "max_decisions": 8}, is_entry=True,
        )
        device = FakeDevice(np.zeros((100, 100, 3), dtype=np.uint8))
        brain = StubBrain(
            [tap_decision(0.2, 0.2), tap_decision(0.8, 0.8), done_decision()]
        )
        engine = RunEngine(self.db, flow_id, device, max_steps=10, brain=brain)
        result = engine.run()
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(device.taps), 2)

    def test_ai_node_done_is_flagged_when_verifier_disagrees(self):
        flow_id = self._flow()
        self.db.add_node(
            flow_id, name="假完成", kind="ai", action_type="ai",
            action={"goal": "进入电视台", "max_decisions": 8}, is_entry=True,
        )
        device = FakeDevice(np.zeros((100, 100, 3), dtype=np.uint8))
        events: list[dict] = []
        brain = StubBrain(
            [tap_decision(0.5, 0.5), done_decision()],
            verdict={"done": False, "reason": "截图还是首页"},
        )
        engine = RunEngine(
            self.db, flow_id, device, max_steps=10, brain=brain, event_cb=events.append
        )
        result = engine.run()
        self.assertEqual(result.status, "finished")
        # 复核不通过时不能记成「已完成」，要留下警告
        self.assertFalse(any(event["type"] == "ai_verified" for event in events))
        self.assertTrue(
            any(
                event["type"] == "warn" and "复核认为目标未达成" in event.get("message", "")
                for event in events
            )
        )

    def test_ai_node_done_accepted_when_verifier_agrees(self):
        flow_id = self._flow()
        self.db.add_node(
            flow_id, name="真完成", kind="ai", action_type="ai",
            action={"goal": "点击电视台", "max_decisions": 8}, is_entry=True,
        )
        device = FakeDevice(np.zeros((100, 100, 3), dtype=np.uint8))
        events: list[dict] = []
        brain = StubBrain(
            [tap_decision(0.5, 0.5), done_decision()],
            verdict={"done": True, "reason": "已经是电视台页面"},
        )
        engine = RunEngine(
            self.db, flow_id, device, max_steps=10, brain=brain, event_cb=events.append
        )
        engine.run()
        self.assertTrue(any(event["type"] == "ai_verified" for event in events))

    def test_ai_disabled_keeps_original_behaviour(self):
        flow_id = self._flow()
        shot = save_image(self.images / "a.png", make_image("blue"))
        self.db.add_node(
            flow_id, name="蓝色页", kind="image", image_path=shot,
            img_w=180, img_h=400, is_entry=True, threshold=0.9,
        )
        self.settings.enabled = False
        ollama.save_settings(self.settings)
        device = FakeDevice(np.asarray(make_image("blue").convert("RGB")))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        result = engine.run()
        self.assertIsNone(engine.brain)
        # 单节点流程没有后继，会一直停在当前节点直到步数上限
        self.assertEqual(result.status, "max_steps")


if __name__ == "__main__":
    unittest.main()
