from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from scflow.db import Database
from scflow.project_runner import ProjectRunner
from scflow.undo import restore_flow, restore_project, snapshot_flow, snapshot_project
from tests.helpers import ScriptedDevice, make_image, save_image, temp_home


def rgb(kind: str) -> np.ndarray:
    return np.asarray(make_image(kind).convert("RGB"))


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.db = Database(Path(self.home.name) / "test.db")
        self.images = Path(self.home.name) / "images"
        self.images.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.home.cleanup()
        os.environ.pop("SCFLOW_HOME", None)

    def add_node(self, flow_id, name, kind, image, is_entry=False, action="none"):
        path = save_image(self.images / f"{flow_id}-{name}.png", image)
        return self.db.add_node(
            flow_id,
            name=name,
            kind=kind,
            image_path=path,
            img_w=image.size[0],
            img_h=image.size[1],
            action_type=action,
            click_x=0.5,
            click_y=0.5,
            is_entry=is_entry,
        )

    def make_block_flow(self, name, entry_kind, end_kind="green"):
        flow_id = self.db.create_flow(name)
        entry = self.add_node(flow_id, "入口", "image", make_image(entry_kind), is_entry=True)
        end = self.add_node(flow_id, "结束", "end", make_image(end_kind))
        self.db.add_edge(flow_id, entry, end, 1.0)
        return flow_id

    def test_project_blocks_aliases_and_restore(self):
        first = self.db.create_flow("首页 流程")
        second = self.db.create_flow("选菜")
        project_id = self.db.create_project("每日外卖")
        block_a = self.db.add_project_block(project_id, first)
        block_b = self.db.add_project_block(project_id, second)
        blocks = self.db.list_project_blocks(project_id)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[0].alias.isidentifier())
        self.assertNotEqual(blocks[0].alias, blocks[1].alias)
        self.assertEqual(blocks[0].flow_name, "首页 流程")

        snapshot = snapshot_project(self.db, project_id)
        self.db.reorder_project_blocks(project_id, [block_b, block_a])
        self.db.remove_project_block(block_a)
        self.assertEqual(len(self.db.list_project_blocks(project_id)), 1)
        restore_project(self.db, project_id, snapshot)
        restored = self.db.list_project_blocks(project_id)
        self.assertEqual(len(restored), 2)
        self.assertEqual([block.flow_id for block in restored], [first, second])

    def test_flow_snapshot_restore(self):
        flow_id = self.make_block_flow("流程", "blue")
        nodes = self.db.list_nodes(flow_id)
        self.db.update_node(
            nodes[0].id, target_x=0.2, target_y=0.3, target_w=0.1, target_h=0.1
        )
        snapshot = snapshot_flow(self.db, flow_id)
        self.db.update_node(nodes[0].id, name="被改名", target_x=0.5)
        self.db.delete_node(nodes[1].id)
        self.db.add_node(flow_id, "新增", kind="image")
        restore_flow(self.db, flow_id, snapshot)
        restored_nodes = self.db.list_nodes(flow_id)
        self.assertEqual(len(restored_nodes), 2)
        entry = next(node for node in restored_nodes if node.name == "入口")
        self.assertTrue(entry.has_target)
        self.assertAlmostEqual(entry.target_x, 0.2, places=3)
        self.assertEqual(len(self.db.list_edges(flow_id)), 1)

    def test_project_runner_sequence(self):
        flow_a = self.make_block_flow("区块A", "blue")
        flow_b = self.make_block_flow("区块B", "red")
        project_id = self.db.create_project("顺序项目")
        self.db.add_project_block(project_id, flow_a)
        self.db.add_project_block(project_id, flow_b)
        device = ScriptedDevice([rgb("blue"), rgb("green"), rgb("red"), rgb("green")])
        runner = ProjectRunner(self.db, project_id, device)
        result = runner.run()
        self.assertEqual(result.status, "finished", result.error)
        self.assertEqual(result.steps, 4)
        self.assertEqual(len(runner.executed_blocks), 2)
        runs = self.db.list_project_runs(project_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "finished")

    def test_project_runner_python_while_and_variables(self):
        flow_a = self.make_block_flow("区块A", "blue")
        flow_b = self.make_block_flow("区块B", "red")
        project_id = self.db.create_project("Python 项目")
        self.db.add_project_block(project_id, flow_a)
        self.db.add_project_block(project_id, flow_b)
        aliases = [block.alias for block in self.db.list_project_blocks(project_id)]
        code = f"次数 = 0\nwhile 次数 < 2:\n    {aliases[0]}()\n    次数 += 1\n{aliases[1]}()\n"
        self.db.update_project(
            project_id,
            python_code=code,
            variables={"起始": 1},
            settings={"mode": "python", "block_max_steps": 5},
        )
        device = ScriptedDevice(
            [
                rgb("blue"),
                rgb("green"),
                rgb("blue"),
                rgb("green"),
                rgb("red"),
                rgb("green"),
            ]
        )
        runner = ProjectRunner(self.db, project_id, device)
        result = runner.run()
        self.assertEqual(result.status, "finished", result.error)
        self.assertEqual(len(runner.executed_blocks), 3)
        self.assertEqual(runner.runtime_variables.get("次数"), 2)
        self.assertEqual(runner.runtime_variables.get("起始"), 1)


if __name__ == "__main__":
    unittest.main()
