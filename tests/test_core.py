from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from scflow import vision
from scflow.db import Database
from scflow.embeddings import cosine, get_embedder
from scflow.engine import RunEngine
from scflow.flowio import export_flow, import_flow
from tests.helpers import FakeDevice, make_image, save_image, temp_home


def rgb(image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.db = Database(Path(self.home.name) / "test.db")
        self.images = Path(self.home.name) / "images"
        self.images.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.home.cleanup()
        os.environ.pop("SCFLOW_HOME", None)

    def add_node(self, flow_id, name, kind, action="none", **kwargs):
        image = kwargs.pop("image", None)
        image_path = None
        width = height = None
        if image is not None:
            image_path = save_image(self.images / f"{name}.png", image)
            width, height = image.size
        return self.db.add_node(
            flow_id,
            name=name,
            kind=kind,
            image_path=image_path,
            img_w=width,
            img_h=height,
            action_type=action,
            **kwargs,
        )

    def test_embedding_similarity(self):
        embedder = get_embedder("hybrid")
        blue = rgb(make_image("blue"))
        red = rgb(make_image("red"))
        same = cosine(embedder.embed(blue), embedder.embed(blue))
        other = cosine(embedder.embed(blue), embedder.embed(red))
        self.assertAlmostEqual(same, 1.0, places=5)
        self.assertLess(other, 0.9)

    def test_match_and_tap(self):
        flow_id = self.db.create_flow("流程")
        entry = self.add_node(
            flow_id, "入口", "image", action="none", image=make_image("blue"),
            click_x=0.5, click_y=0.5, is_entry=True,
        )
        target = self.add_node(
            flow_id, "下一步", "image", action="tap", image=make_image("red"),
            click_x=0.5, click_y=0.5, is_entry=False,
        )
        self.db.add_edge(flow_id, entry, target, 1.0)
        device = FakeDevice(rgb(make_image("blue")))
        engine = RunEngine(self.db, flow_id, device, max_steps=5)
        self.assertTrue(engine.step())
        self.assertEqual(engine.current_state, entry)
        self.assertEqual(device.taps, [])
        device.set_frame(rgb(make_image("red")))
        self.assertTrue(engine.step())
        self.assertEqual(engine.current_state, target)
        self.assertEqual(device.taps, [(90, 200)])

    def test_next_scope_skips_later_nodes(self):
        flow_id = self.db.create_flow("严格下一步")
        entry = self.add_node(
            flow_id, "入口", "image", image=make_image("blue"), is_entry=True
        )
        nxt = self.add_node(flow_id, "下一步", "image", image=make_image("red"))
        later = self.add_node(flow_id, "更后面", "image", image=make_image("green"))
        self.db.add_edge(flow_id, entry, nxt, 1.0)
        self.db.add_edge(flow_id, nxt, later, 1.0)
        frame = rgb(make_image("green"))
        device = FakeDevice(frame)
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        engine.prepare()
        self.assertEqual(engine.settings.get("match_scope"), "next")
        matches = engine.rank(frame, engine.embedder.embed(frame))
        ids = {match.node.id for match in matches}
        self.assertEqual(ids, {entry, nxt})
        self.assertNotIn(later, ids)
        engine.settings["match_scope"] = "global"
        global_matches = engine.rank(frame, engine.embedder.embed(frame))
        self.assertIn(later, {match.node.id for match in global_matches})

    def test_state_preference(self):
        flow_id = self.db.create_flow("偏好")
        entry = self.add_node(
            flow_id, "入口", "image", image=make_image("blue"), is_entry=True
        )
        image = make_image("red")
        first = self.add_node(flow_id, "高偏好", "image", image=image)
        second = self.add_node(flow_id, "低偏好", "image", image=image)
        self.db.add_edge(flow_id, entry, first, 8.0)
        self.db.add_edge(flow_id, entry, second, 0.5)
        device = FakeDevice(rgb(make_image("blue")))
        engine = RunEngine(self.db, flow_id, device, max_steps=5)
        engine.step()
        device.set_frame(rgb(make_image("red")))
        engine.step()
        self.assertEqual(engine.current_state, first)

    def test_roi_emphasis(self):
        flow_id = self.db.create_flow("重点区")
        self.add_node(
            flow_id,
            "重点",
            "image",
            image=make_image("roi"),
            roi_x=0.0,
            roi_y=0.0,
            roi_w=0.5,
            roi_h=1.0,
            roi_weight=0.95,
            is_entry=True,
        )
        device = FakeDevice(rgb(make_image("roi-changed")))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        engine.prepare()
        frame = device.screenshot()
        full = cosine(engine.embedder.embed(frame), engine.embeddings[engine.current_state])
        matches = engine.rank(frame, engine.embedder.embed(frame))
        self.assertEqual(len(matches), 1)
        self.assertIsNotNone(matches[0].roi_similarity)
        self.assertGreater(matches[0].similarity, full)

    def test_fallback(self):
        flow_id = self.db.create_flow("保底")
        node = self.add_node(
            flow_id, "目标", "image", image=make_image("blue"), is_entry=True
        )
        self.db.update_flow(
            flow_id,
            settings={
                "fallback": {
                    "max_retries": 0,
                    "max_fallbacks": 1,
                    "threshold_step": 0.04,
                    "min_threshold": 0.55,
                    "retry_wait": 0,
                    "steps": [{"type": "key", "key": "back", "wait": 0}],
                }
            },
        )
        device = FakeDevice(rgb(make_image("black")))
        engine = RunEngine(self.db, flow_id, device, max_steps=5)
        engine.prepare()
        self.assertTrue(engine.step())
        self.assertEqual(device.keys, ["back"])
        self.assertFalse(engine.step())
        self.assertIn("保底", engine.error or "")
        self.assertEqual(engine.current_state, node)

    def test_export_import_and_link(self):
        flow_id = self.db.create_flow("导出")
        entry = self.add_node(
            flow_id,
            "入口",
            "image",
            action="tap",
            image=make_image("blue"),
            click_x=0.3,
            click_y=0.4,
            roi_x=0.1,
            roi_y=0.2,
            roi_w=0.5,
            roi_h=0.3,
            roi_weight=0.81,
            target_x=0.2,
            target_y=0.3,
            target_w=0.1,
            target_h=0.1,
            is_entry=True,
        )
        target = self.add_node(flow_id, "结束", "end", image=make_image("green"))
        self.db.add_edge(flow_id, entry, target, 2.5, "偏好")
        exported = export_flow(self.db, flow_id, Path(self.home.name) / "flow.scflow.json")
        self.assertTrue(exported.exists())

        result = import_flow(self.db, exported)
        imported = self.db.get_flow(result["flow_id"])
        self.assertIsNotNone(imported)
        nodes = self.db.list_nodes(result["flow_id"])
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(self.db.list_edges(result["flow_id"])), 1)
        self.assertTrue(any(node.is_entry for node in nodes))
        with_roi = next(node for node in nodes if node.has_roi)
        self.assertAlmostEqual(with_roi.roi_weight, 0.81, places=2)
        self.assertTrue(with_roi.has_target)
        self.assertAlmostEqual(with_roi.target_region[0], 0.2, places=3)
        self.assertTrue(Path(with_roi.image_path).exists())
        self.assertTrue(self.db.get_embeddings(result["flow_id"]))

        host_id = self.db.create_flow("宿主")
        host_node = self.add_node(host_id, "宿主入口", "image", image=make_image("red"), is_entry=True)
        merged = import_flow(
            self.db,
            exported,
            target_flow_id=host_id,
            link_from_node_id=host_node,
        )
        self.assertEqual(merged["flow_id"], host_id)
        self.assertEqual(len(self.db.list_nodes(host_id)), 3)
        edges = self.db.list_edges(host_id)
        self.assertTrue(any(edge.src_id == host_node for edge in edges))

    @unittest.skipIf(not vision.cv2_available(), "需要 OpenCV 才能测试点击对象定位")
    def test_click_target_localization(self):
        def make_screen(target_x, target_y):
            image = Image.new("RGB", (300, 600), (30, 80, 180))
            draw = ImageDraw.Draw(image)
            draw.rectangle((target_x, target_y, target_x + 100, target_y + 80), fill=(230, 50, 50))
            draw.text((target_x + 35, target_y + 30), "OK", fill=(255, 255, 255))
            return image

        node_image = make_screen(40, 80)
        live = make_screen(180, 400)
        flow_id = self.db.create_flow("目标点击")
        self.add_node(
            flow_id,
            "目标",
            "image",
            action="tap",
            image=node_image,
            click_x=0.5,
            click_y=0.5,
            target_x=40 / 300,
            target_y=80 / 600,
            target_w=100 / 300,
            target_h=80 / 600,
            threshold=0.2,
            is_entry=True,
        )
        device = FakeDevice(rgb(live))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        self.assertTrue(engine.step())
        self.assertEqual(len(device.taps), 1)
        tap_x, tap_y = device.taps[0]
        self.assertLessEqual(abs(tap_x - 230), 3)
        self.assertLessEqual(abs(tap_y - 440), 3)

    def test_click_target_fallback(self):
        flow_id = self.db.create_flow("目标回退")
        image = make_image("blue")
        self.add_node(
            flow_id,
            "目标",
            "image",
            action="tap",
            image=image,
            click_x=0.5,
            click_y=0.5,
            target_x=0.2,
            target_y=0.3,
            target_w=0.2,
            target_h=0.2,
            threshold=0.2,
            is_entry=True,
        )
        device = FakeDevice(rgb(make_image("blue")))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        with mock.patch("scflow.vision.template_locate", return_value=None):
            self.assertTrue(engine.step())
        self.assertEqual(len(device.taps), 1)
        self.assertLessEqual(abs(device.taps[0][0] - 54), 3)
        self.assertLessEqual(abs(device.taps[0][1] - 160), 3)

    @unittest.skipIf(not vision.cv2_available(), "需要 OpenCV 才能测试相对偏移点击")
    def test_click_relative_offset(self):
        def make_screen(target_x, target_y):
            image = Image.new("RGB", (300, 600), (30, 80, 180))
            draw = ImageDraw.Draw(image)
            draw.rectangle((target_x, target_y, target_x + 100, target_y + 80), fill=(230, 50, 50))
            draw.text((target_x + 35, target_y + 30), "OK", fill=(255, 255, 255))
            return image

        node_image = make_screen(40, 80)
        live = make_screen(180, 400)
        flow_id = self.db.create_flow("相对偏移点击")
        node_id = self.add_node(
            flow_id,
            "相对",
            "image",
            action="tap",
            image=node_image,
            click_x=0.5,
            click_y=0.5,
            threshold=0.2,
            is_entry=True,
        )
        self.db.update_node(
            node_id,
            action={
                "rel_roi": [40 / 300, 80 / 600, 100 / 300, 80 / 600],
                "rel_offset": [10, -20],
            },
        )
        device = FakeDevice(rgb(live))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        self.assertTrue(engine.step())
        self.assertEqual(len(device.taps), 1)
        tap_x, tap_y = device.taps[0]
        self.assertLessEqual(abs(tap_x - 240), 3)
        self.assertLessEqual(abs(tap_y - 420), 3)

    def test_ui_selector_click_preferred(self):
        flow_id = self.db.create_flow("控件树点击")
        node_id = self.add_node(
            flow_id,
            "控件",
            "image",
            action="tap",
            image=make_image("blue"),
            click_x=0.5,
            click_y=0.5,
            threshold=0.2,
            is_entry=True,
        )
        self.db.update_node(
            node_id,
            action={"ui_selector": {"text": "确定", "clickable_only": True}},
        )

        class UiDevice(FakeDevice):
            def find_ui_element(self, selector):
                return {"center": [120, 340], "text": "确定", "clickable": True}

        device = UiDevice(rgb(make_image("blue")))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        self.assertTrue(engine.step())
        self.assertEqual(device.taps, [(120, 340)])

    def test_click_relative_offset_fallback(self):
        flow_id = self.db.create_flow("相对偏移回退")
        node_id = self.add_node(
            flow_id,
            "相对",
            "image",
            action="tap",
            image=make_image("blue"),
            click_x=0.5,
            click_y=0.5,
            threshold=0.2,
            is_entry=True,
        )
        self.db.update_node(
            node_id,
            action={
                "rel_roi": [0.2, 0.3, 0.2, 0.2],
                "rel_offset": [10, -20],
            },
        )
        device = FakeDevice(rgb(make_image("blue")))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        with mock.patch("scflow.vision.template_locate", return_value=None):
            self.assertTrue(engine.step())
        self.assertEqual(device.taps, [(64, 140)])

    def test_export_import_relative_and_threshold(self):
        flow_id = self.db.create_flow("新字段导出")
        node_id = self.add_node(
            flow_id,
            "节点",
            "image",
            action="tap",
            image=make_image("blue"),
            threshold=0.2,
            is_entry=True,
        )
        self.db.update_node(
            node_id,
            action={
                "rel_roi": [0.1, 0.2, 0.3, 0.4],
                "rel_offset": [12, -8],
                "target_threshold": 0.62,
            },
        )
        exported = export_flow(self.db, flow_id, Path(self.home.name) / "new.scflow.json")
        result = import_flow(self.db, exported)
        imported = self.db.list_nodes(result["flow_id"])[0]
        self.assertEqual(imported.relative_roi, (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(imported.relative_offset, (12.0, -8.0))
        self.assertAlmostEqual(imported.target_threshold, 0.62, places=3)

    def test_old_project_without_relative_fields(self):
        flow_id = self.db.create_flow("旧版")
        node_id = self.add_node(
            flow_id,
            "旧节点",
            "image",
            action="tap",
            image=make_image("blue"),
            target_x=0.2,
            target_y=0.3,
            target_w=0.1,
            target_h=0.1,
            is_entry=True,
        )
        node = self.db.get_node(node_id)
        self.assertIsNone(node.relative_roi)
        self.assertEqual(node.relative_offset, (0.0, 0.0))
        self.assertAlmostEqual(node.target_threshold, 0.55, places=3)

        payload = {
            "format": "scflow.flow",
            "version": 1,
            "flow": {"name": "旧版导入", "description": "", "settings": {}},
            "nodes": [
                {
                    "key": "n1",
                    "name": "旧节点",
                    "kind": "image",
                    "img_w": 180,
                    "img_h": 400,
                    "click_x": 0.5,
                    "click_y": 0.5,
                    "target_x": 0.2,
                    "target_y": 0.3,
                    "target_w": 0.1,
                    "target_h": 0.1,
                    "action_type": "tap",
                    "action": {"target_strict": False},
                    "threshold": 0.85,
                    "is_entry": True,
                    "is_end": False,
                    "x": 0,
                    "y": 0,
                    "notes": "",
                    "image": None,
                }
            ],
            "edges": [],
            "images": {},
        }
        old_file = Path(self.home.name) / "old.scflow.json"
        old_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        result = import_flow(self.db, old_file)
        imported = self.db.list_nodes(result["flow_id"])[0]
        self.assertIsNone(imported.relative_roi)
        self.assertEqual(imported.relative_offset, (0.0, 0.0))
        self.assertAlmostEqual(imported.target_threshold, 0.55, places=3)

    def test_text_clear_and_submit(self):
        flow_id = self.db.create_flow("输入")
        node_id = self.add_node(
            flow_id,
            "搜索",
            "image",
            action="text",
            image=make_image("blue"),
            click_x=0.5,
            click_y=0.5,
            is_entry=True,
            threshold=0.2,
        )
        self.db.update_node(
            node_id, action={"text": "4tb ssd", "clear": True, "submit": True}
        )
        device = FakeDevice(rgb(make_image("blue")))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        self.assertTrue(engine.step())
        self.assertTrue(device.cleared)
        self.assertEqual(device.texts, ["4tb ssd"])
        self.assertEqual(device.keys, ["enter"])

    def test_launch_action(self):
        flow_id = self.db.create_flow("启动")
        node_id = self.add_node(
            flow_id,
            "启动淘宝",
            "image",
            action="launch",
            image=make_image("blue"),
            click_x=0.5,
            click_y=0.5,
            is_entry=True,
            threshold=0.2,
        )
        self.db.update_node(
            node_id, action={"package": "com.taobao.taobao", "fresh": False}
        )
        device = FakeDevice(rgb(make_image("blue")))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        self.assertTrue(engine.step())
        self.assertEqual(device.launched, ["com.taobao.taobao"])

    def test_dry_run_device(self):
        from scflow.device import DryRunDevice

        device = FakeDevice(rgb(make_image("blue")))
        dry = DryRunDevice(device)
        dry.tap(10, 20)
        dry.key("back")
        self.assertEqual(device.taps, [])
        self.assertEqual(device.keys, [])
        self.assertEqual(dry.actions[0], ("tap", 10, 20))


if __name__ == "__main__":
    unittest.main()
