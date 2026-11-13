# 本地 AI（Ollama）集成说明

本文说明 scrcpy-flow 如何把本机 Ollama 变成一个“看得懂屏幕”的决策器，以及怎样调参、排错和扩展。

## 为什么接本地模型

原来的流程是“截图 → 图片向量 → 阈值匹配 → 执行动作”。它稳定、可解释，但有一个前提：**每个可能出现的重要页面都要先截一张图**。页面改版、弹窗、首次引导都会让流程卡住。

本地视觉模型补上了另一条路：给它当前截图和任务目标，它直接告诉你“下一步点哪里”。两者并存：

```
             ┌── 图片匹配（快、可解释、离线）
截图 ────────┤
             └── 本地视觉模型（慢、通用、能处理没见过的页面）
```

AI 失败或关闭时，程序完全走原来的逻辑，不会因为 AI 不可用而无法运行。

## 四个能力

| 能力 | 入口 | 代码位置 |
| --- | --- | --- |
| AI 决策节点（kind=`ai`） | 流程画布 →「添加 AI 节点」 | `engine.RunEngine._execute_ai_node` |
| 失败接管 | 自动触发（图片匹配重试失败后） | `engine.RunEngine._ai_recover` |
| 节点重排 | 自动触发（候选都低于阈值） | `engine.RunEngine._ai_rerank` |
| 一句话生成流程 | 「AI 助手」→「AI 生成流程…」 | `ai.ScreenBrain.plan_flow` |

另外 `python -m scflow --ai-task "…"` 可以不建流程直接执行（`ai.AgentRunner`）。

## 一次决策的完整数据流

```
截图(np.ndarray, RGB)
   │  encode_image(): 缩到最长边 720px + JPEG q80 + base64
   ▼
OllamaClient.vision()
   │  POST /api/chat  { model, messages[{role,content,images[]}],
   │                    format:"json", think:false, options{temperature,...} }
   ▼
本机视觉模型（如 qwen3.5:4b / qwen2.5vl:3b）
   │  返回一个 JSON 字符串
   ▼
ollama.extract_json()  →  ai.ScreenBrain._to_decision()
   │  校验 action 白名单、把 element_index 映射成控件中心、
   │  把像素坐标换算成 0~1 相对坐标、夹紧越界值
   ▼
Decision(action, x, y, text, done, ...)
   ▼
engine._perform_ai_action() → device.tap / swipe / key / text / _sleep
```

发给模型的控件树来自 `adb shell uiautomator dump`，被压成紧凑文本：

```
#31 [可点] ImageView "返回" bounds=[0,60,120,180]
#12 [可点] TextView "搜索" bounds=[120,90,900,150]
```

有控件时模型会优先回 `element_index`，程序用控件中心点击——比让模型“估坐标”准得多。

### 模型输出契约

```json
{
  "page": "外卖首页",
  "thought": "需要先点搜索框",
  "action": "tap",
  "element_index": 12,
  "x": null,
  "y": null,
  "text": "",
  "clear": false,
  "key": "back",
  "seconds": 1,
  "done": false,
  "confidence": 0.8
}
```

- `action` 只接受 `tap / swipe / back / text / key / wait / done`（`click`、`input`、`finish` 等别名会被归一化）。
- `element_index` 有效时覆盖 `x/y`。
- `x/y` 允许给像素值（>1.5 时按屏幕宽高换算）。
- `done=true` 或 `action=done` 表示该节点目标完成。

## 配置

设置文件：`data/ollama.json`（在工程目录内，可随工程迁移）。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 总开关；关闭后不创建大脑、不发起任何请求 |
| `base_url` | `http://127.0.0.1:11434` | Ollama 地址，可指向局域网另一台机器 |
| `vision_model` | 空=自动 | 需要 `vision` 能力 |
| `text_model` | 空=自动 | 生成流程用 |
| `temperature` | `0.1` | 越低越稳，建议 0~0.3 |
| `timeout` | `180` | 单次请求超时（秒） |
| `max_image_side` | `720` | 截图最长边，越小越快 |
| `image_quality` | `80` | JPEG 质量 |
| `think` | `false` | 让支持思考的模型先推理（更准但慢 2~3 倍） |
| `ai_nodes` | `true` | 允许运行 AI 决策节点 |
| `ai_fallback` | `true` | 匹配失败时 AI 接管 |
| `ai_rerank` | `true` | 相似度不足时 AI 选节点 |
| `ui_context` | `true` | 把控件树发给模型 |
| `allow_text` | `true` | 允许模型决定输入文字 |
| `max_decisions` | `12` | 单个 AI 节点一轮最多做多少动作（节点里可单独覆盖） |

环境变量优先级高于文件：`SCFLOW_OLLAMA_URL`、`SCFLOW_OLLAMA_MODEL`、`SCFLOW_OLLAMA_TEXT_MODEL`、`SCFLOW_OLLAMA_TIMEOUT`、`SCFLOW_OLLAMA_DISABLE=1`。

命令行也支持临时覆盖：`--ai-model qwen2.5vl:3b`。

## 怎样写好 AI 节点的目标

这是实践中最影响成败的一点。

**推荐：写成动作。**

- 「点击底部标签栏的「电视台」」
- 「在搜索框输入黄焖鸡米饭，然后点「搜索」」
- 「点击右下角的「加入购物车」」

动作类目标做完就能确认，模型不容易反复。

**不推荐：只写结果。**

- 「进入电视台页面」
- 「买好外卖」

结果类目标要求模型理解“这个页面是不是目标页面”。实测 3B~4B 模型容易两极化：要么完成后不肯收手，要么把别的页面误判成目标页面。如果确实要描述结果，请补上可判定的特征，例如
「点击底部「电视台」标签，直到页面顶部出现「第一频道」「视野」「联播频道」」。

**经验值**

- 一个节点只做一件事；多件事请拆成多个 AI 节点。
- 需要确认成功时，让“成功”在画面上有明确文字（标题、按钮文案），比看图标高亮可靠。
- 「决策上限」默认 12，简单导航给 3~5 就够，能避免模型反复试错。

## 模型选择

- 必须支持 `vision`。用 `curl -s http://127.0.0.1:11434/api/show -d '{"model":"..."}'` 可以看到 `capabilities` 里是否有 `vision`。
- 中文界面优先 `qwen2.5vl:3b` / `qwen3.5:4b`；英文界面 `llava:7b`、`moondream` 也可用。
- 纯文本模型（`gemma:2b` 等）不能看图，程序会在「AI 设置」里标红并拒绝把它当视觉模型。
- 显存/内存紧张时：把 `max_image_side` 降到 600~720，关闭 `think`，一次只跑一个任务（Ollama 默认并行度为 1）。

## 性能

实测（CPU、4B 级模型、720px 截图、`think=false`）：

| 阶段 | 耗时 |
| --- | --- |
| 首次加载模型 | 3~8 秒 |
| 单次决策 | 1.5~5 秒 |
| 读取控件树（uiautomator） | 0.5~2 秒 |

因此 AI 节点适合“每步决策”的中低频任务；高频固定点击仍建议用图片节点。

`uiautomator dump` 在播放视频、动画不停时可能失败并报 `could not get idle state`。程序会捕获并只靠截图决策，不会中断运行。

## 排错

| 现象 | 原因 / 处理 |
| --- | --- |
| `连不上本地 Ollama` | 没运行 `ollama serve`，或地址/端口不对；`ollama list` 能列出模型说明服务正常 |
| `没有找到支持看图的本地模型` | 只装了文本模型；`ollama pull qwen2.5vl:3b` |
| `视觉模型 xxx 不在...` | 配置里写死的模型没拉取；改回自动或先 pull |
| 决策总是 `done` | 目标描述太抽象，写具体一点，例如“点击左上角返回按钮” |
| 点击位置偏移 | 模型估坐标不准；尽量让控件树可用，或改用 `element_index` 能覆盖的按钮 |
| 中文输入失败 | `adb shell input text` 不支持中文，装 ADBKeyboard 并切换输入法 |
| 每次都超时 | 模型太大或机器忙；换 2B/3B 视觉模型、降低 `max_image_side`、加大 `timeout` |

## 隐私与安全

- 所有请求发往 `base_url`，默认是本机回环地址；程序不会主动上报任何数据。
- 如果把 `base_url` 改成局域网地址，截图会发到那台机器，请自行评估。
- AI 只返回“动作建议”，真正的点击仍由 `device` 层执行；`--dry-run` 下不会真的点击手机。
- 不做验证码破解、风控绕过；请只在合法、自己有权操作的设备上使用。

## 扩展点

- 新增模型能力：`ollama.OllamaClient.chat()` 支持 `options` 与 `think`，可直接复用。
- 新增动作类型：在 `ai.ACTION_ALIASES` 里加别名，在 `engine._perform_ai_action` 里实现执行。
- 自定义提示词：在 `ai.ScreenBrain.decide()` 里改 `prompt`；保持“只输出 JSON”的约束。
- 测试：`tests/test_ollama.py` 用本地假 HTTP 服务，`tests/test_ai.py` 用 `StubBrain`，都不需要真实模型。
