"""Filesystem registry for Python strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from research_lab.backend.core.python_strategies.models import PythonStrategyMetadata
from research_lab.backend.settings import get_settings


class PythonStrategyRegistry:
    """Persist and retrieve Python strategy metadata in a JSON index."""

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.registry_dir = Path(registry_dir) if registry_dir else settings.python_strategies_dir
        self.index_file = self.registry_dir / "registry.json"
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register_strategy(self, metadata: PythonStrategyMetadata) -> None:
        """Register or update a strategy in the registry."""

        index = self._load_index()
        index[metadata.key] = metadata.model_dump()
        self._save_index(index)

    def get_strategy(self, key: str) -> Optional[PythonStrategyMetadata]:
        """Return strategy metadata for the given key."""

        index = self._load_index()
        entry = index.get(key)
        return PythonStrategyMetadata(**entry) if entry else None

    def list_strategies(self) -> List[PythonStrategyMetadata]:
        """List all registered strategies."""

        index = self._load_index()
        return [PythonStrategyMetadata(**data) for data in index.values()]

    def _load_index(self) -> Dict[str, dict]:
        if not self.index_file.exists():
            return {}
        with self.index_file.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupted strategy registry index: {exc}") from exc
        return data if isinstance(data, dict) else {}

    def _save_index(self, index: Dict[str, dict]) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_file.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
        tmp_path.replace(self.index_file)


__all__ = ["PythonStrategyRegistry"]
