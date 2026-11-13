# 架构设计

## 总览

scrcpy-flow 把“手机当前画面”当作状态，把“截图 + 点击位置”当作节点，把“带权重的有向边”当作状态转移偏好。

```
截图 ──→ 图片向量 ──→ 与所有图片节点比较 ──→ 阈值过滤 ──→ 加权偏好排序 ──→ 执行动作
                                                    │
                                                    └──→ 无命中 → 重试 / 保底
```

## 模块

| 模块 | 职责 |
| --- | --- |
| `scflow/config.py` | 便携数据目录、adb / scrcpy 路径探测 |
| `scflow/db.py` | SQLite 表与 DAO，连接按操作创建，支持 GUI 与运行线程并发 |
| `scflow/models.py` | `Flow` / `Node` / `Edge` / `Match` 等数据模型 |
| `scflow/embeddings.py` | 混合图片向量；可选 CLIP |
| `scflow/vision.py` | 解码、缩放、模板定位、点击点映射、截图保存 |
| `scflow/device.py` | ADB 设备、scrcpy 窗口、DryRun 包装、跨平台窗口 API |
| `scflow/engine.py` | 运行状态机、候选排序、动作执行、保底、日志 |
| `scflow/ollama.py` | 本地 Ollama 客户端：模型能力探测、图片编码、JSON 抽取、AI 设置 |
| `scflow/ai.py` | 看屏决策、候选节点重排、一句话生成流程、极简智能体 |
| `scflow/flowio.py` | 流程 JSON 导入导出、图片打包、衔接 |
| `scflow/gui/` | Tkinter 界面：运行、编辑、历史 |

### 本地 AI 与图片匹配的关系

图片匹配是主路径，本地视觉模型是可选增强，两者通过 `RunEngine` 汇合：

```
rank()                       _choose()                 step()
截图 ──→ 候选 + 分数 ──→ 阈值/偏好过滤 ──→ 命中 ──→ 执行节点动作
               │ 都不命中                 │
               │                          └── AI 重排（ai_rerank）──→ 命中
               ├── AI 决策节点（kind=ai）──→ 每次只问模型“下一步做什么”
               └── 仍无命中 ──→ AI 接管一步（ai_fallback）──→ 回到主循环
```

- AI 关闭（`enabled=false`）时，`RunEngine.brain` 为 `None`，所有 AI 分支短路，行为与 1.1.0 完全一致。
- `brain` 可以在 `RunEngine(..., brain=...)` 注入，测试里用 `StubBrain` 替换，不访问网络。
- 模型返回的是动作描述，不直接接触设备；点击/滑动仍由 `device` 层执行，`DryRunDevice` 下自动变成空操作。

## 数据表

- `flows`：流程，`settings` 保存运行参数与保底配置。
- `nodes`：节点，含图片路径、图片尺寸、点击点、ROI、动作、阈值、入口标记、画布坐标、向量缓存。
- `edges`：`src_id -> dst_id`，`weight` 是当前状态的偏好权重。
- `sessions`：一次运行。
- `step_logs`：每一步的当前状态、匹配节点、相似度、动作、结果、截图路径。

## 图片向量

默认 `HybridEmbedder` 将截图统一到 RGB，并组合四个块：

| 块 | 维度 | 权重 | 作用 |
| --- | --- | --- | --- |
| dHash | 64 | 0.15 | 结构/布局的快速指纹 |
| HSV 直方图 | 512 | 0.25 | 颜色分布 |
| 4×4×8 梯度方向直方图 | 128 | 0.25 | 纹理与边缘 |
| 16×16 灰度缩略图 | 256 | 0.35 | 整体版面 |

四块分别归一化后按 `sqrt(weight)` 缩放拼接，再做 L2 归一化。相似度为余弦相似度。
新的「相似度编辑」模型会优先使用颜色/纹理/结构/边缘成分匹配；旧的混合向量保留用于兼容和回退。

可通过 `SCFLOW_EMBEDDER=clip` 使用 CLIP：

```bash
pip install -r requirements-clip.txt
SCFLOW_EMBEDDER=clip python -m scflow
```

节点向量缓存在数据库；向量类型不匹配时会自动重算。

## 检测重点（ROI）

节点可以保存一个相对矩形 `(roi_x, roi_y, roi_w, roi_h)` 和权重 `roi_weight`。

运行时：

```
roi_similarity = cos(截图同位置裁剪向量, 节点同位置裁剪向量)
similarity = roi_weight * roi_similarity + (1 - roi_weight) * full_similarity
```

默认 `roi_weight = 0.72`，即重点区占主导。适合：

- 消息回复：框住消息气泡或联系人标题；
- 点外卖：框住餐厅名/菜品区域；
- 游戏或工具：框住关键数字或按钮。

## 点击对象定位

节点可以额外保存一个“点击对象”矩形 `target_*`。运行时：

1. 从节点图片中裁出该矩形；
2. 用 OpenCV 多尺度模板匹配在当前截图中寻找它；
3. 找到后点击匹配区域中心；
4. 找不到时回退到该矩形在节点图片中的相对中心，并记录提示。

这与“检测重点 ROI”相互独立：ROI 决定“这是哪个状态”，点击对象决定“点哪里”。

## 候选范围与选择

默认 `match_scope = next`：只比较当前状态本身和它的直连下一步（出边目标）。其他节点（包括其检测重点 ROI）完全不会参与相似度计算，因此不会在未到步骤时被误触发。

可选：

- `reachable`：当前节点沿连线可达的所有节点；
- `global`：所有图片节点，适合弹窗、公共页面或允许跳步的流程。

对候选中的每个图片节点：

```
score = similarity + beta * max(edge_weight(current_state -> node))
if node == current_state:
    score -= stay_penalty
```

- `beta` 默认 0.12，连线的偏好权重越大，越容易被选中。
- `stay_penalty` 防止点击失败时永远停留在当前画面。
- 先按每个节点的阈值过滤，再按 `score` 排序。
- `selection_mode = weighted_random` 时，在达标候选里按偏好权重和相似度余量做加权随机，用于“每天换一家”。

## 保底机制

每次未匹配：

1. 记录一次 `no-match`，提高重试计数；
2. 重试计数未超限：等待 `retry_wait` 后重新截图，同时按 `threshold_step` 降低阈值，最低到 `min_threshold`；
3. 重试超限：进入保底，执行配置的步骤（按键 / 等待 / 点击 / 滑动 / 输入）；
4. 保底次数超过 `max_fallbacks`：停止并报错。

保底步骤与重试参数都在编辑模式中编辑，保存在 `flows.settings.fallback`。

## 并发

- 运行引擎在后台线程执行，GUI 通过 `queue.Queue` 接收事件。
- SQLite 连接短生命周期、WAL 模式，GUI 与运行线程各自打开连接。
- 运行截图保存为缩小后的 JPEG，长跑时可关闭。

## 相似度模型

节点可以在动作 JSON 的 `similarity_model` 中保存新的相似度参数：

- 侧重检测节点（kind=detect）：`s = W·MR`，W 固定为 1，MR 为检测重点区域成分匹配。
- 侧重点检测节点（kind=image，兼容旧 action）：`s = W1·MR + W2·MR非`。
- MR / MR非 都由颜色、纹理、结构、边缘四类成分按权重合成。
- 点击对象可单独保存一组成分权重，用于点击对象定位时的校验。
- UI 入口：属性面板「相似度编辑…」、右键匹配阈值、流程节点右键、流程画布「相似度编辑」模式。
- 未设置 `similarity_model` 的旧节点继续使用原有向量匹配，保证兼容。

## 专业版工具

- `ui2/pro_workspace.py`：批量设备、智能定位、录制回放。
- 批量设备：每台设备独立 `DeviceController`，共用 `Database`，通过线程池并发执行 `RunEngine` / `ProjectRunner`。
- 智能定位：`AdbDevice.ui_elements()` 解析 `uiautomator dump`；可选 OCR 使用 `pytesseract`。选中元素后可测试点击或写入节点的 `click_x/click_y` 与 `action.ui_selector`。
- 录制回放：受控 scrcpy 窗口 + `pynput` 全局鼠标监听，把点击映射为设备坐标并自动创建流程节点（实验功能）。

## 手机文件与原生 APK

- `phone_export.py`：把单个流程或项目导出为 `.scflowpkg`（zip）。
  - `manifest.json`：应用信息、入口、流程节点/连线、动作、点击和控件选择器。
  - `images/`：节点状态图。
- `flow-to-apk-sdk/phone-runner`：原生 Android 运行器。
  - `PhonePackage` 解析手机文件。
  - `FlowAccessibilityService` 使用无障碍服务截图、点击、滑动、输入文本和读取控件树。
  - `FlowEngine` 执行状态匹配、图片/ROI/点击对象/相对偏移/控件树定位、IF 分支与保底。
  - `build_flow_apk.sh` 用 SDK 工具链把手机文件嵌入 APK；`install_apk.sh` 通过 adb 安装。

## 跨平台

- 截图：ADB `exec-out screencap -p`；scrcpy 窗口用 `mss` 或 `ImageGrab`。
- 窗口定位：
  - Linux：`xdotool` / `xwininfo`；
  - Windows：Win32 `EnumWindows` + `GetClientRect` + `ClientToScreen`；
  - macOS：AppleScript / System Events 读取窗口位置与大小。
- 输入：
  - ADB `input tap/swipe/text/keyevent`（默认）；
  - scrcpy 窗口点击：Linux 用 `xdotool`，Windows 用 `SetCursorPos` + `mouse_event`，macOS 用 `cliclick` 或 System Events。
- scrcpy 窗口画面会按手机宽高比裁掉黑边并缩放到设备分辨率，保证节点图片与运行截图一致。
