from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from scflow import ai, ollama
from scflow.ai_project import create_flow_from_plan, create_project_from_plan
from scflow.db import Database
from tests.helpers import temp_home


class StubClient:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def text(self, model, system, user, json_mode=True, timeout=None):
        self.calls.append({"model": model, "user": user})
        return self.replies.pop(0) if self.replies else "{}"

    def vision(self, *args, **kwargs):  # pragma: no cover - 不用
        return "{}"

    def resolve_vision_model(self):
        return "qwen3.5:4b"

    def resolve_text_model(self, allow_vision=True):
        return "qwen3.5:4b"


class PlanProjectTests(unittest.TestCase):
    def setUp(self):
        self.settings = ollama.OllamaSettings()
        self.home = temp_home()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(lambda: os.environ.pop("SCFLOW_HOME", None))

    def test_plan_project_normalizes_blocks(self):
        reply = json.dumps(
            {
                "name": "美团下单",
                "description": "从搜索到提交订单",
                "blocks": [
                    {
                        "name": "打开并搜索",
                        "flow_goal": "打开美团搜索黄焖鸡",
                        "steps": [
                            {"name": "打开美团", "goal": "点击美团图标"},
                            {"name": "搜索", "goal": "搜索黄焖鸡"},
                        ],
                    },
                    {"name": "加购", "flow_goal": "加入购物车", "steps": []},
                ],
            },
            ensure_ascii=False,
        )
        brain = ai.ScreenBrain(self.settings, client=StubClient([reply]))
        plan = brain.plan_project("打开美团点黄焖鸡")
        self.assertEqual(plan["name"], "美团下单")
        self.assertEqual(len(plan["blocks"]), 2)
        self.assertEqual(plan["blocks"][0]["steps"][1]["goal"], "搜索黄焖鸡")
        # 没给 steps 的区块自动补一步
        self.assertEqual(len(plan["blocks"][1]["steps"]), 1)
        self.assertEqual(plan["blocks"][1]["steps"][0]["goal"], "加入购物车")

    def test_plan_project_falls_back_to_one_block(self):
        brain = ai.ScreenBrain(self.settings, client=StubClient(['{"name":"","blocks":[]}']))
        plan = brain.plan_project("打开设置")
        self.assertEqual(len(plan["blocks"]), 1)
        self.assertTrue(plan["blocks"][0]["steps"])

    def test_create_flow_from_plan_creates_chain(self):
        db = Database(Path(self.home.name) / "test.db")
        flow_id = create_flow_from_plan(
            db,
            {
                "name": "测试流程",
                "description": "说明",
                "steps": [
                    {"name": "第一步", "goal": "点 A"},
                    {"name": "第二步", "goal": "点 B"},
                ],
            },
        )
        nodes = db.list_nodes(flow_id)
        self.assertEqual(len(nodes), 2)
        self.assertTrue(nodes[0].is_entry)
        self.assertTrue(nodes[1].is_end)
        edges = db.list_edges(flow_id)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].src_id, nodes[0].id)
        self.assertEqual(nodes[0].kind, "ai")
        self.assertEqual(nodes[0].action.get("goal"), "点 A")

    def test_create_project_from_plan_builds_blocks(self):
        db = Database(Path(self.home.name) / "test.db")
        plan = {
            "name": "两段项目",
            "description": "测试",
            "blocks": [
                {"name": "打开", "flow_goal": "打开应用", "steps": [{"name": "打开", "goal": "点图标"}]},
                {"name": "搜索", "flow_goal": "搜索", "steps": [{"name": "输入", "goal": "输入关键词"}]},
            ],
        }
        project_id = create_project_from_plan(db, plan)
        project = db.get_project(project_id)
        self.assertEqual(project.name, "两段项目")
        blocks = db.list_project_blocks(project_id)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].alias, "打开")
        self.assertEqual(len(db.list_nodes(blocks[1].flow_id)), 1)
        self.assertIn("()", project.python_code)
        self.assertEqual(len(project.python_code.splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
