from __future__ import annotations

import argparse
import ctypes
import os
import sys

from . import __version__


def _resolve_flow(db, value: str):
    flow = None
    try:
        flow = db.get_flow(int(value))
    except ValueError:
        flow = db.get_flow_by_name(value)
    if flow is None:
        flow = db.get_flow_by_name(value)
    return flow


def _resolve_project(db, value: str):
    project = None
    try:
        project = db.get_project(int(value))
    except ValueError:
        project = db.get_project_by_name(value)
    if project is None:
        project = db.get_project_by_name(value)
    return project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scflow",
        description="scrcpy-flow · 基于图片识别与流程图的手机自动化工具",
    )
    parser.add_argument("--version", action="version", version=f"scrcpy-flow {__version__}")
    parser.add_argument("--data-dir", help="数据目录（默认使用程序旁的 data/）")
    parser.add_argument("--embedder", choices=["hybrid", "clip"], help="图片向量方式")
    parser.add_argument("--list-devices", action="store_true", help="列出 ADB 设备")
    parser.add_argument("--list-flows", action="store_true", help="列出流程")
    parser.add_argument("--list-projects", action="store_true", help="列出区块项目")
    parser.add_argument("--headless", metavar="流程名或ID", help="无界面运行指定流程或项目")
    parser.add_argument("--ai-status", action="store_true", help="检查本地 Ollama 与模型状态")
    parser.add_argument(
        "--ai-task",
        metavar="任务描述",
        help="不建流程，直接让本地 AI 看屏执行任务，例如：打开设置查看电池电量",
    )
    parser.add_argument(
        "--ai-model", metavar="模型名", help="临时覆盖视觉模型（等同 SCFLOW_OLLAMA_MODEL）"
    )
    parser.add_argument("--classic", action="store_true", help="启动经典 Tkinter 界面")
    parser.add_argument("--open-project", metavar="项目名", help="启动界面并打开指定区块项目")
    parser.add_argument("--open-flow", metavar="流程名", help="启动界面并打开指定流程")
    parser.add_argument("--steps", type=int, default=200, help="最大运行步数")
    parser.add_argument("--serial", help="ADB 设备序列号")
    parser.add_argument("--capture", choices=["scrcpy", "adb"], default="scrcpy")
    parser.add_argument("--input", dest="input_mode", choices=["adb", "scrcpy"], default="adb")
    parser.add_argument("--no-scrcpy", action="store_true", help="不启动 scrcpy 窗口")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不真正点击")
    parser.add_argument("--auto-run", metavar="项目名或流程名", help="界面启动后自动运行")
    args = parser.parse_args(argv)

    if args.data_dir:
        os.environ["SCFLOW_HOME"] = args.data_dir
    if args.embedder:
        os.environ["SCFLOW_EMBEDDER"] = args.embedder
    if args.ai_model:
        os.environ["SCFLOW_OLLAMA_MODEL"] = args.ai_model

    from .db import Database

    db = Database()

    if args.list_devices:
        from .device import list_adb_devices

        devices = list_adb_devices()
        if not devices:
            print("未发现 ADB 设备")
        for device in devices:
            print(f"{device['serial']}\t{device['state']}\t{device['model']}")
        return 0

    if args.list_flows:
        flows = db.list_flows()
        if not flows:
            print("还没有流程，请先启动图形界面创建")
        for flow in flows:
            count = len(db.list_nodes(flow.id))
            print(f"{flow.id}\t{flow.name}\t{count} 个节点")
        return 0

    if args.list_projects:
        projects = db.list_projects()
        if not projects:
            print("还没有区块项目")
        for project in projects:
            count = len(db.list_project_blocks(project.id))
            print(f"{project.id}\t{project.name}\t{count} 个区块")
        return 0

    if args.ai_status:
        return _ai_status()

    if args.ai_task:
        return _ai_task(args)

    if args.headless:
        return _headless(db, args)

    if not args.classic:
        try:
            from .ui2 import launch as launch2
        except ImportError as exc:
            print(f"PySide6 不可用，回退经典界面：{exc}", file=sys.stderr)
        else:
            return launch2(db, args)
    return _gui(db, args.open_project, args.open_flow)


def _headless(db, args) -> int:
    from .device import DeviceController, DryRunDevice
    from .engine import RunEngine
    from .project_runner import ProjectRunner

    project = _resolve_project(db, args.headless)
    flow = None if project is not None else _resolve_flow(db, args.headless)
    if project is None and flow is None:
        print(f"找不到流程或项目：{args.headless}", file=sys.stderr)
        return 2
    device = None
    try:
        device = DeviceController(
            serial=args.serial,
            capture=args.capture,
            input_mode=args.input_mode,
            manage_scrcpy=not args.no_scrcpy,
        )
        device.start()
        for warning in device.warnings:
            print(f"[提示] {warning}")
        inner = DryRunDevice(device) if args.dry_run else device

        def printer(event):
            kind = event.get("type")
            if kind == "match":
                print(
                    f"步 {event.get('state')} 匹配 {event.get('state_name')} "
                    f"相似度={event.get('similarity', 0):.3f} 动作={event.get('action')}"
                )
            elif kind == "no_match":
                print(
                    f"未匹配 重试 {event.get('retries')}/{event.get('max_retries')} "
                    f"最高相似度={event.get('similarity')}"
                )
            elif kind == "fallback":
                print(f"执行保底 {event.get('index')}/{event.get('max_fallbacks')}")
            elif kind == "project_start":
                print(f"项目开始：{event.get('name')} · {len(event.get('blocks') or [])} 个区块 · {event.get('mode')}")
            elif kind == "block_start":
                print(f"区块开始：{event.get('name')} ({event.get('alias')})")
            elif kind == "block_finish":
                result = event.get("result")
                if result is not None:
                    print(f"区块结束：{result.name} {result.status} 步数={result.steps} 保底={result.fallbacks}")
            elif kind == "project_log":
                print(f"[脚本] {event.get('message')}")
            elif kind == "project_finish":
                print(
                    f"项目结束：{event.get('status')} 区块={event.get('blocks')} "
                    f"步数={event.get('steps')} 保底={event.get('fallbacks')}"
                )
            elif kind == "warn":
                print(f"[提示] {event.get('message')}")
            elif kind == "ai_decision":
                print(
                    f"[AI] 页面「{event.get('page') or '?'}」-> {event.get('action')}"
                    f" · {event.get('reason') or ''} ({event.get('elapsed', 0):.1f}s)"
                )
            elif kind == "ai_recover":
                print(f"[AI 接管] {event.get('reason') or ''} -> {event.get('action')}")
            elif kind == "ai_rerank":
                print(
                    f"[AI 选节点] {event.get('node_name')} "
                    f"置信度={event.get('confidence')} {event.get('reason') or ''}"
                )
            elif kind == "ai_verified":
                print(f"[AI 复核] 目标已达成：{event.get('reason') or ''}")
            elif kind == "error":
                print(f"[错误] {event.get('message')}", file=sys.stderr)
            elif kind == "finish":
                print(
                    f"结束：{event.get('status')} 步数={event.get('steps')} "
                    f"保底={event.get('fallbacks')}"
                )

        if project is not None:
            runner = ProjectRunner(
                db,
                project.id,
                inner,
                event_cb=printer,
                max_steps=args.steps,
            )
            result = runner.run()
        else:
            engine = RunEngine(
                db,
                flow.id,
                inner,
                event_cb=printer,
                max_steps=args.steps,
            )
            result = engine.run()
        return 0 if result.status in ("finished", "stopped") else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    finally:
        if device is not None:
            device.stop()


def _ai_status() -> int:
    """打印本地 Ollama 的连接状态、可用模型和自动挑选的视觉模型。"""
    from . import ollama

    settings = ollama.load_settings()
    print(f"配置文件：{ollama.settings_path()}")
    print(f"启用：{'是' if settings.enabled else '否'}")
    print(f"地址：{settings.base_url}")
    info = ollama.get_client(settings).status()
    if not info.get("available"):
        print(f"未连接：{info.get('error') or '请先运行 ollama serve'}")
        return 1
    print(f"版本：{info.get('version')}")
    print(f"模型：{', '.join(info.get('models') or []) or '（无）'}")
    print(f"视觉模型：{info.get('vision_model') or '-'}")
    print(f"文本模型：{info.get('text_model') or '-'}")
    if info.get("error"):
        print(f"提示：{info['error']}")
        return 1
    return 0


def _ai_task(args) -> int:
    """不依赖流程图，直接让本地 AI 看屏执行一句任务描述。"""
    from . import ai, ollama
    from .device import DeviceController, DryRunDevice

    settings = ollama.load_settings()
    if not settings.enabled:
        print("AI 已在设置中关闭（可用 python -m scflow --ai-status 查看）", file=sys.stderr)
        return 2

    def printer(event):
        kind = event.get("type")
        if kind == "ai_agent_start":
            print(
                f"AI 任务开始：{event.get('goal')} · 最多 {event.get('max_steps')} 步"
            )
        elif kind == "ai_step":
            print(
                f"步 {event.get('step')} · 页面「{event.get('page') or '?'}」"
                f" -> {event.get('action')} · {event.get('thought') or ''}"
            )
        elif kind == "warn":
            print(f"[提示] {event.get('message')}")
        elif kind == "error":
            print(f"[错误] {event.get('message')}", file=sys.stderr)
        elif kind == "ai_agent_finish":
            print(
                f"AI 任务结束：{event.get('status')} · 共 {event.get('steps')} 步"
            )

    device = None
    try:
        device = DeviceController(
            serial=args.serial,
            capture=args.capture,
            input_mode=args.input_mode,
            manage_scrcpy=not args.no_scrcpy,
        )
        device.start()
        for warning in device.warnings:
            print(f"[提示] {warning}")
        inner = DryRunDevice(device) if args.dry_run else device
        runner = ai.AgentRunner(
            inner,
            args.ai_task,
            settings=settings,
            event_cb=printer,
            max_steps=args.steps,
        )
        result = runner.run()
        return 0 if result["status"] in ("finished", "stopped") else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    finally:
        if device is not None:
            device.stop()


def _gui(db, open_project: str | None = None, open_flow: str | None = None) -> int:
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    try:
        from .gui import App
    except ImportError as exc:
        print(f"无法加载图形界面：{exc}", file=sys.stderr)
        print("请确认已安装 Python 的 Tk 支持（Windows 官方 Python 默认包含）", file=sys.stderr)
        return 1
    app = App(db)
    if open_project:
        app.after(300, lambda: app.open_project(open_project))
    if open_flow:
        app.after(300, lambda: app.open_flow(open_flow))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
