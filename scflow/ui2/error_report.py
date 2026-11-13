from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path

from ..db import Database

_MAX_CONSOLE_LINES = 800
_SESSION_LIMIT = 30


def _config_home() -> Path:
    from .. import config

    return Path(config.home())


def _flow_name(db: Database, flow_id: int) -> str:
    flow = db.get_flow(flow_id)
    return flow.name if flow else f"#{flow_id}"


def _log_line(row) -> str:
    parts = [f"  步 {row['idx']}"]
    if row["current_name"]:
        parts.append(f"当前={row['current_name']}")
    if row["matched_name"]:
        parts.append(f"匹配={row['matched_name']}")
    if row["similarity"] is not None:
        parts.append(f"相似度={row['similarity']:.3f}")
    if row["score"] is not None:
        parts.append(f"得分={row['score']:.3f}")
    if row["action"]:
        parts.append(f"动作={row['action']}")
    parts.append(f"结果={row['result']}")
    if row["screenshot_path"]:
        parts.append(f"截图={row['screenshot_path']}")
    return " | ".join(parts)


def _is_problem_log(row) -> bool:
    result = str(row["result"] or "").strip().lower()
    return result not in ("", "ok")


def build_error_report(
    db: Database,
    console_text: str,
    engineering_dir: str | Path,
    version: str,
) -> str:
    lines: list[str] = []
    lines.append("scrcpy-flow 错误日志")
    lines.append("=" * 64)
    lines.append(f"导出时间   : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"程序版本   : {version}")
    lines.append(f"工程目录   : {engineering_dir}")
    lines.append(f"操作系统   : {platform.platform()}")
    lines.append(f"Python     : {sys.version.split()[0]}")
    lines.append("")

    lines.append("-" * 64)
    lines.append("[控制台输出（最近部分）]")
    lines.append("-" * 64)
    console_lines = (console_text or "").splitlines()
    if len(console_lines) > _MAX_CONSOLE_LINES:
        console_lines = console_lines[-_MAX_CONSOLE_LINES:]
    lines.extend(console_lines if console_lines else ["（无）"])
    lines.append("")

    lines.append("-" * 64)
    lines.append("[运行错误 / 未匹配 / 保底记录]")
    lines.append("-" * 64)
    session_blocks = 0
    for session in db.list_sessions(limit=_SESSION_LIMIT):
        logs = db.list_logs(session.id)
        problem_logs = [row for row in logs if _is_problem_log(row)]
        if session.status == "finished" and not problem_logs and not session.note:
            continue
        session_blocks += 1
        lines.append(
            f"会话 {session.id} | 流程: {_flow_name(db, session.flow_id)} | "
            f"状态: {session.status} | 步数: {session.steps} | 保底: {session.fallbacks}"
        )
        lines.append(f"  开始: {session.started_at}  结束: {session.ended_at or '-'}")
        if session.note:
            lines.append(f"  备注: {session.note}")
        rows = logs if session.status != "finished" else problem_logs
        if rows:
            for row in rows:
                lines.append(_log_line(row))
        else:
            lines.append("  （该会话没有错误记录）")
        lines.append("")
    if not session_blocks:
        lines.append("（无）")
        lines.append("")

    lines.append("-" * 64)
    lines.append("[项目运行错误]")
    lines.append("-" * 64)
    project_blocks = 0
    for row in db.list_project_runs(limit=_SESSION_LIMIT):
        error = str(row["error"] or "").strip()
        status = str(row["status"] or "")
        if not error and status == "finished":
            continue
        project_blocks += 1
        lines.append(
            f"项目运行 {row['id']} | 项目: {row['project_name']} | 状态: {status} | "
            f"区块: {row['blocks']} | 步数: {row['steps']}"
        )
        lines.append(f"  开始: {row['started_at']}  结束: {row['ended_at'] or '-'}")
        if error:
            lines.append(f"  错误: {error}")
        lines.append("")
    if not project_blocks:
        lines.append("（无）")
        lines.append("")

    return "\n".join(lines) + "\n"
