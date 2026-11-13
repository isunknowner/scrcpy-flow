from __future__ import annotations

import builtins
import json
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .db import Database
from .engine import RunEngine
from .models import EngineResult, Project, ProjectBlock

DEFAULT_PROJECT_SETTINGS = {
    "mode": "sequence",
    "stop_on_error": True,
    "block_max_steps": 100,
    "max_block_runs": 200,
    "script_timeout": 3600,
}


class ProjectStopped(Exception):
    pass


@dataclass
class BlockResult:
    block_id: int
    flow_id: int
    name: str
    alias: str
    status: str
    steps: int = 0
    fallbacks: int = 0
    session_id: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("finished", "max_steps") and not self.error

    def __bool__(self) -> bool:
        return self.ok


class BlockHandle:
    def __init__(self, runner: "ProjectRunner", block: ProjectBlock):
        self.runner = runner
        self.block = block
        self.alias = block.alias
        self.name = block.flow_name
        self.runs = 0
        self.last: BlockResult | None = None

    def __call__(self, max_steps: int | None = None) -> BlockResult:
        result = self.runner.run_block(self.block, max_steps)
        self.runs += 1
        self.last = result
        return result

    def __bool__(self) -> bool:
        return bool(self.last and self.last.ok)

    @property
    def ok(self) -> bool:
        return bool(self.last and self.last.ok)

    @property
    def status(self) -> str | None:
        return self.last.status if self.last else None

    @property
    def steps(self) -> int:
        return self.last.steps if self.last else 0

    @property
    def error(self) -> str | None:
        return self.last.error if self.last else None


class ProjectRunner:
    def __init__(
        self,
        db: Database,
        project_id: int,
        device,
        event_cb: Callable[[dict], None] | None = None,
        max_steps: int | None = None,
        speed: float = 1.0,
        skip_waits: bool = False,
        swipe_repeat: int = 1,
    ):
        self.db = db
        self.project_id = project_id
        self.device = device
        self.event_cb = event_cb
        self.max_steps_override = max_steps
        self.speed = float(speed or 1.0)
        self.skip_waits = bool(skip_waits)
        self.swipe_repeat = max(1, int(swipe_repeat or 1))

        self.project: Project | None = None
        self.blocks: list[ProjectBlock] = []
        self.settings: dict = {}
        self.handles: dict[str, BlockHandle] = {}
        self.handle_by_block: dict[int, BlockHandle] = {}
        self.variables: dict[str, Any] = {}
        self.runtime_variables: dict[str, Any] = {}

        self.stop_event = threading.Event()
        self.current_engine: RunEngine | None = None
        self.project_run_id: int | None = None
        self.run_count = 0
        self.total_steps = 0
        self.total_fallbacks = 0
        self.executed_blocks: list[BlockResult] = []
        self.error: str | None = None
        self._script_start = 0.0
        self._script_timeout = 3600.0
        self._max_block_runs = 200
        self._block_max_steps = 100

    def emit(self, event_type: str, **payload) -> None:
        if not self.event_cb:
            return
        event = {"type": event_type, "project": True, "time": time.time(), **payload}
        try:
            self.event_cb(event)
        except Exception:
            pass

    def _check_timeout(self) -> None:
        if self._script_start and time.time() - self._script_start > self._script_timeout:
            raise TimeoutError("Python 编排脚本运行超时")

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise ProjectStopped()
        self._check_timeout()

    def prepare(self) -> None:
        project = self.db.get_project(self.project_id)
        if project is None:
            raise RuntimeError("项目不存在")
        self.project = project
        self.blocks = self.db.list_project_blocks(self.project_id)
        if not self.blocks:
            raise RuntimeError("项目还没有区块，请先在区块编排中添加流程区块")
        settings = dict(DEFAULT_PROJECT_SETTINGS)
        settings.update(project.settings or {})
        self.settings = settings
        self.variables = dict(project.variables or {})
        self.runtime_variables = dict(self.variables)
        self._block_max_steps = int(
            self.max_steps_override or settings.get("block_max_steps") or 100
        )
        self._max_block_runs = max(1, int(settings.get("max_block_runs") or 200))
        self._script_timeout = float(settings.get("script_timeout") or 3600)
        self.run_count = 0
        self.total_steps = 0
        self.total_fallbacks = 0
        self.executed_blocks = []
        self.error = None
        self.stop_event.clear()
        self._script_start = time.time()

        self.handles = {}
        self.handle_by_block = {}
        for block in self.blocks:
            handle = BlockHandle(self, block)
            self.handles[block.alias] = handle
            self.handle_by_block[block.id] = handle
            if (
                block.flow_name
                and block.flow_name.isidentifier()
                and block.flow_name not in self.handles
            ):
                self.handles[block.flow_name] = handle

    def stop(self) -> None:
        self.stop_event.set()
        if self.current_engine is not None:
            self.current_engine.stop()

    def request_manual(self) -> None:
        if self.current_engine is not None:
            self.current_engine.request_manual()

    def provide_human(self, result: dict | None) -> None:
        if self.current_engine is not None:
            self.current_engine.provide_human(result)

    def _forward_event(self, block: ProjectBlock, event: dict) -> None:
        kind = event.get("type")
        if kind in ("start", "finish"):
            return
        forwarded = dict(event)
        forwarded["block"] = block.flow_name
        forwarded["block_alias"] = block.alias
        forwarded["flow_id"] = block.flow_id
        self.emit(kind or "event", **{k: v for k, v in forwarded.items() if k != "type"})

    def run_block(self, block: ProjectBlock, max_steps: int | None = None) -> BlockResult:
        self._check_stop()
        self.run_count += 1
        if self.run_count > self._max_block_runs:
            raise RuntimeError("区块执行次数超过上限，请检查 while 循环")

        emit_payload = {
            "block_id": block.id,
            "flow_id": block.flow_id,
            "name": block.flow_name,
            "alias": block.alias,
            "run_index": self.run_count,
            "max_runs": self._max_block_runs,
        }
        self.emit("block_start", **emit_payload)

        engine = RunEngine(
            self.db,
            block.flow_id,
            self.device,
            event_cb=lambda event: self._forward_event(block, event),
            max_steps=int(max_steps or self._block_max_steps),
            speed=self.speed,
            skip_waits=self.skip_waits,
            swipe_repeat=self.swipe_repeat,
        )
        self.current_engine = engine
        try:
            result = engine.run()
        finally:
            self.current_engine = None
        block_result = BlockResult(
            block_id=block.id,
            flow_id=block.flow_id,
            name=block.flow_name,
            alias=block.alias,
            status=result.status,
            steps=result.steps,
            fallbacks=result.fallbacks,
            session_id=result.session_id,
            error=result.error,
        )
        self.total_steps += result.steps
        self.total_fallbacks += result.fallbacks
        self.executed_blocks.append(block_result)
        handle = self.handle_by_block.get(block.id)
        if handle is not None:
            handle.last = block_result
        self.emit("block_finish", result=block_result, **emit_payload)
        if result.status == "stopped" or self.stop_event.is_set():
            raise ProjectStopped()
        return block_result

    def _run_sequence(self) -> str:
        stop_on_error = bool(self.settings.get("stop_on_error", True))
        status = "finished"
        for block in self.blocks:
            self._check_stop()
            result = self.run_block(block)
            if not result.ok:
                message = f"区块「{block.flow_name}」未完成：{result.status}"
                if result.error:
                    message += f" · {result.error}"
                self.error = message
                if stop_on_error:
                    status = "error"
                    break
        return status

    def _safe_builtins(self) -> dict:
        names = [
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "float",
            "format",
            "int",
            "isinstance",
            "len",
            "list",
            "max",
            "min",
            "range",
            "reversed",
            "round",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "zip",
            "Exception",
            "ValueError",
            "RuntimeError",
            "KeyError",
            "IndexError",
            "TypeError",
            "StopIteration",
        ]
        safe = {name: getattr(builtins, name) for name in names}
        safe["print"] = self._python_print
        return safe

    def _python_print(self, *args, **kwargs) -> None:
        self._check_stop()
        text = " ".join(str(arg) for arg in args)
        self.emit("project_log", message=text)

    def _python_sleep(self, seconds: float) -> None:
        end = time.time() + float(seconds)
        while time.time() < end:
            self._check_stop()
            time.sleep(min(0.1, max(0.0, end - time.time())))

    def _python_stop(self) -> None:
        self.stop_event.set()

    def _python_run_block(self, target) -> BlockResult:
        if isinstance(target, BlockHandle):
            return target()
        if isinstance(target, str):
            handle = self.handles.get(target) or self.handles.get(target.strip())
            if handle:
                return handle()
            for block in self.blocks:
                if block.flow_name == target:
                    return self.run_block(block)
        raise ValueError(f"找不到区块：{target}")

    def _python_set_var(self, name: str, value: Any) -> None:
        self.runtime_variables[str(name)] = value

    def _python_get_var(self, name: str, default: Any = None) -> Any:
        return self.runtime_variables.get(name, default)

    def _python_launch_app(self, package: str, fresh: bool = True) -> None:
        self._check_stop()
        launcher = getattr(self.device, "launch_app", None)
        if launcher is None:
            raise RuntimeError("当前设备不支持启动应用")
        launcher(str(package), bool(fresh))

    def _python_key(self, key: str) -> None:
        self._check_stop()
        self.device.key(str(key))

    def _python_tap(self, x: float, y: float) -> None:
        self._check_stop()
        self.device.tap(float(x), float(y))

    def _run_python(self) -> str:
        code = (self.project.python_code if self.project else "") or ""
        if not code.strip():
            return self._run_sequence()
        env: dict[str, Any] = {
            "__builtins__": self._safe_builtins(),
            "变量": self.runtime_variables,
            "vars": self.runtime_variables,
            "run_block": self._python_run_block,
            "set_var": self._python_set_var,
            "get_var": self._python_get_var,
            "log": self._python_print,
            "sleep": self._python_sleep,
            "stop": self._python_stop,
            "launch_app": self._python_launch_app,
            "key": self._python_key,
            "tap": self._python_tap,
        }
        for name, handle in self.handles.items():
            env[name] = handle
        for name, value in self.variables.items():
            if (
                isinstance(name, str)
                and name.isidentifier()
                and name not in env
            ):
                env[name] = value
        self._script_start = time.time()
        try:
            compiled = compile(code, f"<project:{self.project.name if self.project else ''}>", "exec")
            exec(compiled, env, env)
        except ProjectStopped:
            return "stopped"
        except Exception as exc:
            self.error = f"Python 编排出错：{exc}"
            self.emit("error", message=self.error, detail=traceback.format_exc())
            return "error"
        finally:
            self.runtime_variables = self._collect_variables(env)
        return "stopped" if self.stop_event.is_set() else "finished"

    def _collect_variables(self, env: dict) -> dict:
        result: dict[str, Any] = dict(self.runtime_variables)
        skip = {
            "__builtins__",
            "变量",
            "vars",
            "run_block",
            "set_var",
            "get_var",
            "log",
            "sleep",
            "stop",
            "launch_app",
            "key",
            "tap",
        }
        skip.update(self.handles.keys())
        for key, value in env.items():
            if key in skip or key.startswith("__"):
                continue
            if isinstance(value, (str, int, float, bool, type(None), list, dict, tuple)):
                try:
                    json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError):
                    result[key] = str(value)
                else:
                    result[key] = value
        return result

    def run(self) -> EngineResult:
        try:
            self.prepare()
        except Exception as exc:
            self.error = str(exc)
            self.emit("error", message=self.error)
            self.emit("project_finish", status="error", steps=0, blocks=0, error=self.error, variables={})
            return EngineResult(status="error", error=self.error)

        if self.project is None:
            return EngineResult(status="error", error="项目不存在")
        self.project_run_id = self.db.create_project_run(self.project.id, self.project.name)
        self.emit(
            "project_start",
            project_id=self.project.id,
            name=self.project.name,
            mode=self.settings.get("mode", "sequence"),
            blocks=[
                {"id": block.id, "name": block.flow_name, "alias": block.alias}
                for block in self.blocks
            ],
            variables=dict(self.variables),
            run_id=self.project_run_id,
        )
        status = "finished"
        try:
            mode = str(self.settings.get("mode", "sequence"))
            if mode == "python":
                status = self._run_python()
            else:
                status = self._run_sequence()
        except ProjectStopped:
            status = "stopped"
        except Exception as exc:
            status = "error"
            self.error = str(exc)
            self.emit("error", message=self.error, detail=traceback.format_exc())
        if status == "stopped":
            self.error = self.error or "已手动停止"
        self.db.finish_project_run(
            self.project_run_id,
            status,
            len(self.executed_blocks),
            self.total_steps,
            self.error or "",
            self.runtime_variables,
        )
        self.emit(
            "project_finish",
            status=status,
            steps=self.total_steps,
            fallbacks=self.total_fallbacks,
            blocks=len(self.executed_blocks),
            error=self.error,
            variables=self.runtime_variables,
            run_id=self.project_run_id,
        )
        return EngineResult(
            status=status,
            steps=self.total_steps,
            fallbacks=self.total_fallbacks,
            session_id=self.project_run_id,
            error=self.error,
        )
