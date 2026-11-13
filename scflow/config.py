from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

APP_NAME = "scflow"
APP_TITLE = "scrcpy-flow"


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / APP_TITLE
        return Path(sys.executable).resolve().parent
    project = Path(__file__).resolve().parent.parent
    if any((project / marker).exists() for marker in ("pyproject.toml", "run.sh", "main.py")):
        return project
    return Path.home() / ".scrcpy-flow"


def home() -> Path:
    override = os.environ.get("SCFLOW_HOME")
    root = Path(override).expanduser() if override else app_root() / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_file() -> Path:
    return home() / "scflow.db"


def images_dir() -> Path:
    path = home() / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def screenshots_dir() -> Path:
    path = home() / "screenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def assets_dir() -> Path:
    path = home() / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def exports_dir() -> Path:
    path = home() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def similarity_cache_dir() -> Path:
    path = home() / "similarity_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tools_dir() -> Path:
    path = app_root() / "tools"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _platform_tokens() -> list[str]:
    if os.name == "nt":
        return ["windows", "win64", "win32"]
    if sys.platform == "darwin":
        return ["macos", "mac"]
    return ["linux"]


def _score_candidate(path: Path, name: str) -> int:
    text = str(path).lower()
    score = len(path.parts) * 2
    if os.name == "nt":
        if "win64" in text or "x86_64" in text:
            score -= 8
        if "win32" in text:
            score -= 3
        if "arm" in text and platform.machine().lower() not in ("arm64", "aarch64"):
            score += 20
    elif sys.platform == "darwin":
        if "macos" in text or "mac" in text:
            score -= 6
    else:
        if "linux" in text:
            score -= 6
        if "arm" in text and platform.machine().lower() not in ("arm64", "aarch64"):
            score += 20
    if name == "scrcpy" and (path.parent / "scrcpy-server").exists():
        score -= 3
    return score


def _find_tool(name: str) -> str | None:
    executable = _exe(name)
    base = tools_dir()
    candidates: list[Path] = []
    for directory in (base, *[base / token for token in _platform_tokens()]):
        candidates.extend(
            [
                directory / executable,
                directory / name,
                directory / "platform-tools" / executable,
                directory / "scrcpy" / executable,
            ]
        )
    try:
        candidates.extend(path for path in base.rglob(executable) if path.is_file())
    except OSError:
        pass
    seen: set[str] = set()
    found: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            found.append(candidate)
    for candidate in sorted(found, key=lambda item: _score_candidate(item, name)):
        return str(candidate)
    return None


def _system_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if sys.platform != "darwin":
        return None
    candidates = [
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
        Path("/opt/local/bin") / name,
        Path.home() / "Library/Android/sdk/platform-tools" / name,
    ]
    for base in (os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT")):
        if base:
            candidates.append(Path(base) / "platform-tools" / name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_adb() -> str:
    override = os.environ.get("SCFLOW_ADB")
    if override:
        return override
    bundled = _find_tool("adb")
    if bundled:
        return bundled
    return _system_tool("adb") or "adb"


def resolve_scrcpy() -> str:
    override = os.environ.get("SCFLOW_SCRCPY")
    if override:
        return override
    bundled = _find_tool("scrcpy")
    if bundled:
        return bundled
    return _system_tool("scrcpy") or "scrcpy"


def platform_label() -> str:
    if os.name == "nt":
        return "Windows"
    if sys.platform == "darwin":
        return "macOS"
    return "Linux"
