from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from scflow import ollama
from scflow.db import Database
from scflow.engine import RunEngine
from tests.helpers import FakeDevice, make_image, save_image, temp_home


def wait_event(
    events: list[dict], kind: str, timeout: float = 5.0, min_count: int = 1
) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = [event for event in list(events) if event.get("type") == kind]
        if len(found) >= min_count:
            return found[min_count - 1]
        time.sleep(0.02)
    return None


class BaseEngineTest(unittest.TestCase):
    def setUp(self):
        self.home = temp_home()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(lambda: os.environ.pop("SCFLOW_HOME", None))
        self.db = Database(Path(self.home.name) / "test.db")
        settings = ollama.OllamaSettings()
        settings.enabled = False
        ollama.save_settings(settings)

    def make_engine(self, flow_id, device, **kwargs):
        return RunEngine(self.db, flow_id, device, **kwargs)


class SpeedTests(BaseEngineTest):
    def test_speed_scales_sleep(self):
        flow_id = self.db.create_flow("加速测试")
        engine = self.make_engine(flow_id, FakeDevice(np.zeros((100, 100, 3), dtype=np.uint8)), speed=5.0)
        started = time.monotonic()
        engine._sleep(0.5)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.25)
        self.assertGreater(elapsed, 0.05)

    def test_turbo_skips_sleep(self):
        flow_id = self.db.create_flow("极速测试")
        engine = self.make_engine(
            flow_id, FakeDevice(np.zeros((100, 100, 3), dtype=np.uint8)), skip_waits=True
        )
        started = time.monotonic()
        engine._sleep(3.0)
        self.assertLess(time.monotonic() - started, 0.05)

    def test_node_repeat_overrides_global_default(self):
        flow_id = self.db.create_flow("覆盖测试")
        self.db.add_node(
            flow_id,
            name="连滑2次",
            kind="nodetect",
            action_type="swipe",
            action={"x1": 0.5, "y1": 0.8, "x2": 0.5, "y2": 0.2, "duration": 300, "repeat": 2},
            is_entry=True,
            is_end=True,
        )
        device = FakeDevice(np.zeros((600, 300, 3), dtype=np.uint8))
        engine = self.make_engine(flow_id, device, swipe_repeat=7)
        result = engine.run()
        self.assertEqual(result.status, "finished")
        self.assertEqual(len(device.swipes), 2)

    def test_swipe_repeat_runs_multiple_times(self):
        flow_id = self.db.create_flow("连滑测试")
        self.db.add_node(
            flow_id,
            name="连滑",
            kind="nodetect",
            action_type="swipe",
            action={"x1": 0.5, "y1": 0.8, "x2": 0.5, "y2": 0.2, "duration": 300, "repeat": 4},
            is_entry=True,
            is_end=True,
        )
        device = FakeDevice(np.zeros((600, 300, 3), dtype=np.uint8))
        engine = self.make_engine(flow_id, device, skip_waits=True)
        result = engine.run()
        self.assertEqual(result.status, "finished")
        self.assertEqual(len(device.swipes), 4)


class BatchedSwipeTests(unittest.TestCase):
    def test_swipe_repeat_batches_into_one_shell(self):
        from scflow.device import AdbDevice

        adb = AdbDevice("serial")
        calls = []
        adb.run = lambda args, timeout=20, binary=False: calls.append((args, timeout)) or ""
        adb.swipe_repeat(10, 20, 30, 40, 60, 5)
        self.assertEqual(len(calls), 1)
        args, timeout = calls[0]
        self.assertEqual(args[:3], ["shell", "sh", "-c"])
        script = args[3]
        self.assertEqual(script.count("input swipe"), 5)
        self.assertIn("input swipe 10 20 30 40 60", script)
        self.assertGreater(timeout, 10)

    def test_single_swipe_repeat_uses_plain_swipe(self):
        from scflow.device import AdbDevice

        adb = AdbDevice("serial")
        calls = []
        adb.run = lambda args, timeout=20, binary=False: calls.append(args) or ""
        adb.swipe_repeat(1, 2, 3, 4, 300, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:3], ["shell", "input", "swipe"])

    def test_dry_run_records_each_repeat(self):
        from scflow.device import DryRunDevice

        device = DryRunDevice(FakeDevice(np.zeros((100, 100, 3), dtype=np.uint8)))
        device.swipe_repeat(0.5, 0.8, 0.5, 0.2, 60, 3)
        self.assertEqual(len(device.actions), 3)
        self.assertEqual(device.actions[0][0], "swipe")


class BurstInjectionTests(unittest.TestCase):
    def test_build_monkey_script(self):
        from scflow.device import build_monkey_script

        script = build_monkey_script(540, 1800, 540, 1200, count=3)
        lines = [line for line in script.splitlines() if line.startswith("DispatchPointer")]
        self.assertEqual(len(lines), 3 * 3)  # down + move + up
        self.assertIn("count= 9", script)
        self.assertIn("DispatchPointer(0,0,0,540,1800", script)
        self.assertIn("DispatchPointer(0,0,1,540,1200", script)
        # 起终点相同 = 连点
        tap_script = build_monkey_script(100, 200, 100, 200, count=2)
        tap_lines = [line for line in tap_script.splitlines() if line.startswith("DispatchPointer")]
        self.assertEqual(len(tap_lines), 2 * 3)
        self.assertTrue(all(",100,200," in line for line in tap_lines))

    def test_burst_swipe_pushes_and_runs_monkey(self):
        from scflow.device import AdbDevice

        adb = AdbDevice("serial")
        calls = []
        adb.run = lambda args, timeout=20, binary=False: calls.append(args) or "Events injected: 200"
        count = adb.burst_swipe(100, 200, 100, 900, 50, 50)
        self.assertEqual(count, 50)
        self.assertEqual(calls[0][0], "push")
        self.assertEqual(calls[1][:3], ["shell", "monkey", "-f"])
        self.assertTrue(calls[1][3].startswith("/data/local/tmp/scflow_burst"))

    def test_burst_swipe_detects_zero_events(self):
        from scflow.device import AdbDevice, DeviceError

        adb = AdbDevice("serial")
        adb.run = lambda args, timeout=20, binary=False: "Events injected: 0"
        with self.assertRaises(DeviceError):
            adb.burst_swipe(1, 2, 3, 4, 10, 50)

    def test_dry_run_records_burst(self):
        from scflow.device import DryRunDevice

        device = DryRunDevice(FakeDevice(np.zeros((100, 100, 3), dtype=np.uint8)))
        count = device.burst_swipe(10, 20, 10, 900, 30, 50)
        self.assertEqual(count, 30)
        self.assertEqual(len(device.actions), 30)


class BurstEngineTests(BaseEngineTest):
    def test_burst_node_executes_via_monkey(self):
        flow_id = self.db.create_flow("极速连滑测试")
        self.db.add_node(
            flow_id,
            name="极速连滑",
            kind="nodetect",
            action_type="burst",
            action={"count": 100, "x1": 0.5, "y1": 0.8, "x2": 0.5, "y2": 0.2, "duration": 50},
            is_entry=True,
            is_end=True,
        )
        device = FakeDevice(np.zeros((800, 400, 3), dtype=np.uint8))
        engine = self.make_engine(flow_id, device)
        result = engine.run()
        self.assertEqual(result.status, "finished")
        self.assertEqual(len(device.bursts), 1)
        x1, y1, x2, y2, count, duration = device.bursts[0]
        self.assertEqual(count, 100)
        self.assertEqual(duration, 50)
        self.assertEqual((x1, y1, x2, y2), (200, 640, 200, 160))
        self.assertEqual(device.swipes, [])  # 不走逐条 swipe


class HumanInterventionTests(BaseEngineTest):
    def _flow_with_unmatched_node(self, settings=None):
        """放一个蓝色图片节点，设备画面是纯黑：永远匹配不上。"""
        image = make_image("blue")
        path = save_image(Path(self.home.name) / "images" / "blue.png", image)
        flow_id = self.db.create_flow("人工测试", settings=settings)
        self.db.add_node(
            flow_id,
            name="蓝色按钮",
            kind="image",
            image_path=path,
            img_w=image.width,
            img_h=image.height,
            action_type="tap",
            click_x=0.5,
            click_y=0.5,
            is_entry=True,
            is_end=True,
        )
        return flow_id

    def test_manual_step_applies_and_remembers(self):
        flow_id = self._flow_with_unmatched_node()
        device = FakeDevice(np.zeros((200, 400, 3), dtype=np.uint8))
        events: list[dict] = []
        engine = self.make_engine(flow_id, device, event_cb=events.append)
        engine.prepare()
        engine.request_manual()
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(engine.step()), daemon=True)
        worker.start()
        self.assertIsNotNone(wait_event(events, "need_human"))
        engine.provide_human(
            {"kind": "tap", "x": 0.5, "y": 0.5, "remember": True, "monitor": False}
        )
        worker.join(timeout=5)
        self.assertTrue(result and result[0])
        self.assertEqual(device.taps, [(200, 100)])
        manual = [
            node
            for node in self.db.list_nodes(flow_id)
            if node.kind == "manual"
        ]
        self.assertEqual(len(manual), 1)
        self.assertTrue(manual[0].action.get("manual"))
        self.assertTrue(any(event.get("type") == "manual_saved" for event in events))

    def test_remembered_manual_node_is_used_on_no_match(self):
        flow_id = self._flow_with_unmatched_node(
            settings={
                "fallback": {
                    "max_retries": 0,
                    "retry_wait": 0,
                    "max_fallbacks": 3,
                    "steps": [],
                }
            }
        )
        self.db.add_node(
            flow_id,
            name="人工：按键",
            kind="manual",
            action_type="key",
            action={"manual": True, "kind": "key", "key": "back", "monitor": False},
        )
        device = FakeDevice(np.zeros((200, 400, 3), dtype=np.uint8))
        engine = self.make_engine(flow_id, device)
        engine.prepare()
        engine.step()
        self.assertEqual(device.keys, ["back"])

    def test_monitor_failure_asks_again_and_allows_ai(self):
        flow_id = self._flow_with_unmatched_node()
        device = FakeDevice(np.zeros((200, 400, 3), dtype=np.uint8))
        events: list[dict] = []
        engine = self.make_engine(flow_id, device, event_cb=events.append)
        engine.prepare()
        engine._monitor_manual = lambda timeout=6.0: False  # 模拟检测不达标
        engine.request_manual()
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(engine.step()), daemon=True)
        worker.start()
        self.assertIsNotNone(wait_event(events, "need_human"))
        engine.provide_human(
            {"kind": "key", "key": "back", "remember": False, "monitor": True}
        )
        # 检测必然不通过（画面里没有可匹配节点），应再次请求人工
        wait_event(events, "need_human", timeout=5, min_count=2)
        # 第二次 need_human 应带有失败动作信息
        retries = [event for event in events if event.get("type") == "need_human"]
        self.assertGreaterEqual(len(retries), 2)
        self.assertTrue(retries[-1].get("failed_action"))
        engine.provide_human({"ai": True})
        worker.join(timeout=6)
        self.assertTrue(any(event.get("type") == "human_monitor_failed" for event in events))
        self.assertEqual(device.keys, ["back"])

    def test_cancel_manual_keeps_running(self):
        flow_id = self._flow_with_unmatched_node()
        device = FakeDevice(np.zeros((200, 400, 3), dtype=np.uint8))
        events: list[dict] = []
        engine = self.make_engine(flow_id, device, event_cb=events.append)
        engine.prepare()
        engine.request_manual()
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(engine.step()), daemon=True)
        worker.start()
        self.assertIsNotNone(wait_event(events, "need_human"))
        engine.provide_human(None)
        worker.join(timeout=5)
        self.assertTrue(result and result[0])
        self.assertEqual(device.taps, [])


if __name__ == "__main__":
    unittest.main()
