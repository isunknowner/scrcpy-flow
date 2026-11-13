from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings

RECENT_KEY = "engineering_recent"
CURRENT_KEY = "engineering_current"
MAX_RECENT = 12


class EngineeringManager:
    """Track the current engineering directory and recently opened ones."""

    def __init__(self):
        self.settings = QSettings("scrcpy-flow", "workspace-2")

    def current(self) -> str | None:
        value = self.settings.value(CURRENT_KEY, "")
        return str(value) if value else None

    def set_current(self, path: str | Path) -> None:
        self.settings.setValue(CURRENT_KEY, str(Path(path).expanduser().resolve()))

    def _load_recent(self) -> list[str]:
        raw = self.settings.value(RECENT_KEY, "[]")
        try:
            data = json.loads(raw) if isinstance(raw, str) else []
        except (TypeError, ValueError):
            data = []
        if not isinstance(data, list):
            return []
        return [str(item) for item in data if isinstance(item, str)]

    def _save_recent(self, items: list[str]) -> None:
        self.settings.setValue(RECENT_KEY, json.dumps(items, ensure_ascii=False))

    def add(self, path: str | Path) -> None:
        resolved = str(Path(path).expanduser().resolve())
        items = [item for item in self._load_recent() if item != resolved]
        items.insert(0, resolved)
        self._save_recent(items[:MAX_RECENT])

    def recent_paths(self) -> list[Path]:
        paths = []
        for item in self._load_recent():
            path = Path(item)
            if path.is_dir():
                paths.append(path)
        return paths
