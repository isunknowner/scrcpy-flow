from __future__ import annotations

import json
import os
import unittest
import zipfile
from pathlib import Path

from scflow.db import Database
from scflow.phone_export import export_phone_package
from tests.helpers import make_image, save_image, temp_home


class PhoneExportTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.db = Database(Path(self.home.name) / "test.db")
        self.images = Path(self.home.name) / "images"
        self.images.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.home.cleanup()
        os.environ.pop("SCFLOW_HOME", None)

    def _add_node(self, flow_id, name, image_name, **kwargs):
        path = save_image(self.images / f"{name}.png", make_image(image_name))
        return self.db.add_node(
            flow_id,
            name=name,
            kind="image",
            image_path=path,
            img_w=180,
            img_h=400,
            action_type="tap",
            **kwargs,
        )

    def test_flow_export_contains_nodes_images_and_actions(self):
        flow_id = self.db.create_flow("手机流程")
        n1 = self._add_node(
            flow_id,
            "入口",
            "blue",
            click_x=0.2,
            click_y=0.3,
            target_x=0.1,
            target_y=0.2,
            target_w=0.3,
            target_h=0.2,
            action={"target_threshold": 0.62, "ui_selector": {"text": "确定"}},
            is_entry=True,
        )
        n2 = self._add_node(flow_id, "结束", "red", is_end=True)
        self.db.add_edge(flow_id, n1, n2, 1.0, "下一步")
        out = Path(self.home.name) / "flow.scflowpkg"
        path = export_phone_package(self.db, out, flow_id=flow_id, app_name="测试应用")
        self.assertTrue(path.exists())
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["format"], "scflow.phone")
            self.assertEqual(manifest["app"]["name"], "测试应用")
            self.assertEqual(manifest["entry"]["type"], "flow")
            flow = manifest["flows"][0]
            self.assertEqual(len(flow["nodes"]), 2)
            self.assertEqual(flow["entryId"], "n0")
            node = flow["nodes"][0]
            self.assertEqual(node["action"]["ui_selector"]["text"], "确定")
            self.assertAlmostEqual(node["targetThreshold"], 0.62)
            self.assertIn(node["image"], archive.namelist())
            self.assertEqual(flow["edges"][0]["src"], "n0")

    def test_flow_with_ai_node_exports_without_image(self):
        flow_id = self.db.create_flow("AI 流程")
        node_id = self.db.add_node(
            flow_id,
            name="AI 打开设置",
            kind="ai",
            action_type="ai",
            action={"goal": "打开系统设置"},
            is_entry=True,
        )
        out = Path(self.home.name) / "ai.scflowpkg"
        path = export_phone_package(self.db, out, flow_id=flow_id)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        node = manifest["flows"][0]["nodes"][0]
        self.assertEqual(node["id"], f"n0")
        self.assertEqual(node["kind"], "ai")
        self.assertIsNone(node["image"])
        self.assertEqual(node["action"]["goal"], "打开系统设置")
        self.assertTrue(node_id)

    def test_project_export_order(self):
        project_id = self.db.create_project("手机项目")
        first = self.db.create_flow("第一步")
        second = self.db.create_flow("第二步")
        self._add_node(first, "一步", "blue", is_entry=True)
        self._add_node(second, "二步", "red", is_entry=True)
        self.db.add_project_block(project_id, first)
        self.db.add_project_block(project_id, second)
        out = Path(self.home.name) / "project.scflowpkg"
        path = export_phone_package(self.db, out, project_id=project_id)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["entry"]["type"], "project")
        self.assertEqual(len(manifest["flows"]), 2)
        self.assertEqual(
            manifest["projects"][0]["flowKeys"],
            [manifest["flows"][0]["key"], manifest["flows"][1]["key"]],
        )


if __name__ == "__main__":
    unittest.main()
