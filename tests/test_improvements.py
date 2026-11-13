from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from scflow import similarity
from scflow.db import Database
from scflow.engine import RunEngine
from tests.helpers import FakeDevice, make_image, save_image, temp_home


def rgb(image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.db = Database(Path(self.home.name) / "test.db")
        self.images = Path(self.home.name) / "images"
        self.images.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.home.cleanup()
        os.environ.pop("SCFLOW_HOME", None)

    def test_blank_images_are_not_perfect_match(self):
        blank_a = np.full((200, 100, 3), 255.0, dtype=np.float32)
        blank_b = np.full((200, 100, 3), 255.0, dtype=np.float32)
        components = similarity.component_scores(blank_a, blank_b)
        score = similarity.weighted_score(
            components, similarity.default_component_weights()
        )
        self.assertLess(score, 0.2)

    def test_unrelated_image_scores_lower(self):
        blue = rgb(make_image("blue"))
        red = rgb(make_image("red"))
        weights = similarity.default_component_weights()
        same = similarity.weighted_score(
            similarity.component_scores(blue, blue), weights
        )
        other = similarity.weighted_score(
            similarity.component_scores(blue, red), weights
        )
        self.assertGreater(same, 0.8)
        self.assertLess(other, same - 0.2)

    def test_calibration_pushes_unrelated_to_zero(self):
        positives = [0.95, 0.96, 0.94]
        negatives = [0.60, 0.62, 0.61, 0.63]
        calibration = similarity.build_calibration(
            positives, negatives, fpr=0.01
        )
        self.assertGreater(calibration["threshold_raw"], max(negatives))
        matched = similarity.apply_calibration(0.96, calibration)
        unrelated = similarity.apply_calibration(0.62, calibration)
        self.assertGreater(matched, calibration["threshold_norm"])
        self.assertLess(unrelated, 0.2)

    def test_engine_uses_calibrated_threshold(self):
        flow_id = self.db.create_flow("校准阈值")
        path = save_image(self.images / "blue.png", make_image("blue"))
        calibration = similarity.build_calibration(
            [0.95, 0.96], [0.60, 0.62, 0.61], fpr=0.01
        )
        node_id = self.db.add_node(
            flow_id,
            "校准节点",
            "image",
            image_path=path,
            img_w=180,
            img_h=400,
            action_type="tap",
            action={
                "similarity_model": {
                    "version": 1,
                    "mode": "roi_plus_full",
                    "region_weights": {"roi": 0.72, "non_roi": 0.28},
                    "roi": {"color": 0.25, "texture": 0.25, "structure": 0.25, "edge": 0.25},
                    "non_roi": {"color": 0.25, "texture": 0.25, "structure": 0.25, "edge": 0.25},
                    "calibration": calibration,
                }
            },
            threshold=0.2,
            is_entry=True,
        )
        engine = RunEngine(self.db, self.db.get_flow(flow_id).id, FakeDevice(rgb(make_image("blue"))))
        engine.prepare()
        node = self.db.get_node(node_id)
        self.assertAlmostEqual(
            engine._threshold_for(node), calibration["threshold_norm"], places=4
        )

    def test_raw_score_node_ignores_calibration(self):
        flow_id = self.db.create_flow("原始分")
        path = save_image(self.images / "blue.png", make_image("blue"))
        node_id = self.db.add_node(
            flow_id,
            "节点",
            "image",
            image_path=path,
            img_w=180,
            img_h=400,
            action_type="tap",
            action={
                "similarity_model": {
                    "version": 1,
                    "mode": "roi_plus_full",
                    "region_weights": {"roi": 0.72, "non_roi": 0.28},
                    "roi": {"color": 0.25, "texture": 0.25, "structure": 0.25, "edge": 0.25},
                    "non_roi": {"color": 0.25, "texture": 0.25, "structure": 0.25, "edge": 0.25},
                    "calibration": {
                        "mode": "excess",
                        "mu": 0.5,
                        "threshold_raw": 0.9,
                        "threshold_norm": 0.8,
                    },
                }
            },
            is_entry=True,
        )
        node = self.db.get_node(node_id)
        raw = similarity.raw_score_node(node, rgb(make_image("red")))
        calibrated = similarity.score_node(node, rgb(make_image("red")))
        self.assertIsNotNone(raw)
        self.assertIsNotNone(calibrated)
        self.assertGreater(raw, calibrated)
        self.assertLess(calibrated, 0.5)


if __name__ == "__main__":
    unittest.main()
