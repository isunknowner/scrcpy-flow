from __future__ import annotations

import unittest

from scflow.device import match_ui_element, parse_bounds, parse_ui_dump, ui_selector_from_element

UI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout" package="com.demo" clickable="false" enabled="true" bounds="[0,0][1080,2400]">
    <node index="1" text="确定" resource-id="com.demo:id/ok" class="android.widget.Button" package="com.demo" clickable="true" enabled="true" bounds="[700,2000][1000,2100]" />
    <node index="2" text="取消" resource-id="com.demo:id/cancel" class="android.widget.Button" package="com.demo" clickable="true" enabled="true" bounds="[100,2000][400,2100]" />
  </node>
</hierarchy>
"""


class DeviceUiTreeTests(unittest.TestCase):
    def test_parse_bounds(self):
        self.assertEqual(parse_bounds("[10,20][110,220]"), (10, 20, 110, 220))
        self.assertIsNone(parse_bounds("[10,20][10,220]"))
        self.assertIsNone(parse_bounds("bad"))

    def test_parse_dump(self):
        elements = parse_ui_dump(UI_XML)
        self.assertEqual(len(elements), 3)
        ok = elements[1]
        self.assertEqual(ok["text"], "确定")
        self.assertEqual(ok["center"], [850, 2050])
        self.assertTrue(ok["clickable"])

    def test_match_selector_prefers_clickable(self):
        elements = parse_ui_dump(UI_XML)
        element = match_ui_element(elements, {"text": "确定"})
        self.assertIsNotNone(element)
        self.assertEqual(element["resource_id"], "com.demo:id/ok")
        self.assertIsNone(match_ui_element(elements, {"text": "不存在"}))
        selector = ui_selector_from_element(element)
        self.assertEqual(selector["text"], "确定")
        self.assertTrue(selector["clickable_only"])


if __name__ == "__main__":
    unittest.main()
