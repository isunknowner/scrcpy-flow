# 流程文件格式（scflow.flow v1）

导出文件是一个 UTF-8 JSON，扩展名建议 `.scflow.json`。图片以 base64 内嵌，单文件即可迁移。

```json
{
  "format": "scflow.flow",
  "version": 1,
  "exported_at": "2026-10-29T23:30:00",
  "flow": {
    "name": "每天外卖",
    "description": "首页 -> 选择餐厅 -> 确认",
    "settings": {
      "beta": 0.12,
      "stay_penalty": 0.06,
      "only_reachable": false,
      "selection_mode": "best",
      "save_screenshots": true,
      "fallback": {
        "max_retries": 2,
        "threshold_step": 0.04,
        "min_threshold": 0.55,
        "max_fallbacks": 3,
        "retry_wait": 0.6,
        "steps": [
          {"type": "key", "key": "back", "wait": 0.8},
          {"type": "wait", "seconds": 1.0}
        ]
      }
    }
  },
  "nodes": [
    {
      "key": "n1",
      "name": "外卖首页",
      "kind": "image",
      "image": "n1_home.png",
      "img_w": 1080,
      "img_h": 2400,
      "click_x": 0.5,
      "click_y": 0.82,
      "roi_x": 0.1,
      "roi_y": 0.1,
      "roi_w": 0.8,
      "roi_h": 0.4,
      "roi_weight": 0.72,
      "target_x": 0.42,
      "target_y": 0.88,
      "target_w": 0.16,
      "target_h": 0.05,
      "action_type": "tap",
      "action": {"pre_wait": 0, "post_wait": 0.6},
      "threshold": 0.88,
      "is_entry": true,
      "x": 70,
      "y": 70,
      "notes": "首页识别"
    }
  ],
  "edges": [
    {"src": "n1", "dst": "n2", "weight": 1.5, "label": "常点"}
  ],
  "images": {
    "n1_home.png": {"mime": "image/png", "data": "iVBORw0KGgo..."}
  }
}
```

## 字段说明

### flow

- `name`：导入到新流程时使用；合并时不会覆盖目标流程名。
- `settings`：运行参数，缺省值由 `engine.DEFAULT_SETTINGS` 补齐。

### nodes

- `key`：文件内唯一标识，导入时重新映射为数据库 ID。
- `kind`：`image` 侧重点检测节点（兼容旧 `action`）、`detect` 侧重检测节点、`nodetect` 不检测步骤、`if` 条件节点、`end` 命中后停止。
- `image` / `images`：图片文件名与 base64 数据。
- `img_w` / `img_h`：原图尺寸，用于点击点与模板定位。
- `click_x` / `click_y`：相对图片的点击位置，0–1；为空时点击屏幕中心。
- `roi_*`：检测重点相对矩形；`roi_weight` 0–1，越大越只看该区域。
- `target_*`：重点点击对象相对矩形；运行时会用模板匹配定位它并点击中心。
- `action_type`：`tap` / `swipe` / `key` / `text` / `wait` / `none`。
- `action`：
  - `tap`：可含 `pre_wait`、`post_wait`；
  - `swipe`：`x1,y1,x2,y2` 相对坐标 + `duration` 毫秒；
  - `key`：`key`，如 `back`、`home`、`enter`；
  - `text`：`text`；
  - `wait`：`seconds`；
  - `similarity_model`：可选的成分相似度模型（颜色/纹理/结构/边缘权重、区域权重、点击对象成分权重）。
- `threshold`：匹配阈值 r，默认 0.88；与 `similarity_model` 的成分相似度一起使用。
- `is_entry`：入口节点；导入时若目标流程已有入口则不会抢占。
- `x` / `y`：编辑画布坐标。
- `notes`：备注。

### edges

- `src` / `dst`：节点 key。
- `weight`：当前状态下的偏好权重。
- `label`：说明。

### 保底步骤

- `key`：`{"type":"key","key":"back","wait":0.8}`
- `wait`：`{"type":"wait","seconds":1.0}`
- `tap`：`{"type":"tap","x":0.5,"y":0.9,"wait":0.5}`
- `swipe`：`{"type":"swipe","x1":0.5,"y1":0.8,"x2":0.5,"y2":0.2,"duration":300,"wait":0.5}`
- `text`：`{"type":"text","text":"收到","wait":0.5}`

## 导入衔接

导入时可指定：

- `link_from`：当前流程的某个节点 → 导入流程入口；
- `link_to`：导入流程的出口（无内部出边的节点）→ 当前流程的某个节点；
- `link_weight`：衔接边的权重。

这两个选择在导入对话框中人工完成。合并后仍是一个普通流程，可继续编辑。
