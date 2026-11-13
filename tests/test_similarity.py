from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from scflow import similarity
from scflow.db import Database
from scflow.engine import RunEngine
from tests.helpers import FakeDevice, make_image, save_image, temp_home


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.db = Database(Path(self.home.name) / "test.db")
        self.images = Path(self.home.name) / "images"
        self.images.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.home.cleanup()
        os.environ.pop("SCFLOW_HOME", None)

    def test_component_scores_identical(self):
        image = np.asarray(make_image("blue").convert("RGB"), dtype=np.float32)
        components = similarity.component_scores(image, image)
        for value in components.values():
            self.assertGreater(value, 0.95)

    def test_weighted_score_changes_with_weights(self):
        image = np.asarray(make_image("blue").convert("RGB"), dtype=np.float32)
        other = np.asarray(make_image("red").convert("RGB"), dtype=np.float32)
        components = similarity.component_scores(image, other)
        color_heavy = similarity.weighted_score(
            components, {"color": 1.0, "texture": 0.0, "structure": 0.0, "edge": 0.0}
        )
        texture_heavy = similarity.weighted_score(
            components, {"color": 0.0, "texture": 1.0, "structure": 0.0, "edge": 0.0}
        )
        self.assertNotAlmostEqual(color_heavy, texture_heavy, places=3)

    def test_detect_default_model_weight_is_one(self):
        model = similarity.default_model("detect")
        self.assertEqual(model["mode"], "roi_only")
        self.assertAlmostEqual(model["region_weights"]["roi"], 1.0)
        self.assertAlmostEqual(model["region_weights"]["non_roi"], 0.0)

    def test_click_object_roi_score(self):
        flow_id = self.db.create_flow("点击对象得分")
        path = save_image(self.images / "blue2.png", make_image("blue"))
        node_id = self.db.add_node(
            flow_id,
            "点击对象",
            "image",
            image_path=path,
            img_w=180,
            img_h=400,
            action_type="tap",
            action={"rel_roi": [0.2, 0.2, 0.2, 0.2]},
            is_entry=True,
        )
        node = self.db.get_node(node_id)
        model = similarity.get_model(node)
        same = similarity.roi_score(
            node,
            np.asarray(make_image("blue").convert("RGB")),
            node.relative_roi,
            model["target"],
        )
        other = similarity.roi_score(
            node,
            np.asarray(make_image("red").convert("RGB")),
            node.relative_roi,
            model["target"],
        )
        self.assertIsNotNone(same)
        self.assertIsNotNone(other)
        self.assertGreater(same, other)

    def test_engine_uses_similarity_model(self):
        flow_id = self.db.create_flow("相似度模型")
        path = save_image(self.images / "blue.png", make_image("blue"))
        node_id = self.db.add_node(
            flow_id,
            "模型节点",
            "image",
            image_path=path,
            img_w=180,
            img_h=400,
            action_type="tap",
            action={
                "similarity_model": {
                    "version": 1,
                    "mode": "roi_plus_full",
                    "region_weights": {"roi": 0.8, "non_roi": 0.2},
                    "roi": {"color": 0.7, "texture": 0.1, "structure": 0.1, "edge": 0.1},
                    "non_roi": {"color": 0.25, "texture": 0.25, "structure": 0.25, "edge": 0.25},
                }
            },
            threshold=0.5,
            is_entry=True,
        )
        device = FakeDevice(np.asarray(make_image("blue").convert("RGB")))
        engine = RunEngine(self.db, flow_id, device, max_steps=2)
        engine.prepare()
        frame = device.screenshot()
        matches = engine.rank(frame, engine.embedder.embed(frame))
        self.assertEqual(len(matches), 1)
        self.assertGreater(matches[0].similarity, 0.9)
        self.assertIsNotNone(self.db.get_node(node_id))


if __name__ == "__main__":
    unittest.main()
