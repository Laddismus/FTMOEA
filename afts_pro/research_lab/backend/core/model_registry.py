"""Filesystem-backed model registry stub for the research backend."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from research_lab.backend.settings import get_settings

logger = logging.getLogger(__name__)


class ModelRegistryBase(ABC):
    """Interface for the model registry."""

    @abstractmethod
    def register_model(self, name: str, version: str, path: Path, metadata: dict[str, Any] | None = None) -> None:
        """Register a model artifact."""

    @abstractmethod
    def get_model(self, name: str, version: str | None = None) -> Path | None:
        """Return a model artifact path."""

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """List registered models."""


class FileSystemModelRegistry(ModelRegistryBase):
    """Lightweight filesystem registry storing an index JSON alongside artifacts."""

    def __init__(self, registry_root: Path | None = None) -> None:
        settings = get_settings()
        self.registry_root = Path(registry_root) if registry_root else settings.model_registry_root
        self.index_file = settings.model_registry_index if registry_root is None else self.registry_root / "registry.json"

    def register_model(self, name: str, version: str, path: Path, metadata: dict[str, Any] | None = None) -> None:
        """Persist a model entry in the registry index."""

        entries = self._load_index()
        normalized = {
            "name": name,
            "version": version,
            "path": str(Path(path)),
            "metadata": metadata or {},
        }
        # Replace existing entry for same name/version.
        entries = [entry for entry in entries if not (entry["name"] == name and entry["version"] == version)]
        entries.append(normalized)
        self._save_index(entries)
        logger.info("Model registered | name=%s version=%s path=%s", name, version, path)

    def get_model(self, name: str, version: str | None = None) -> Path | None:
        """Return the model path for the latest or specified version."""

        entries = self._load_index()
        for entry in reversed(entries):
            if entry["name"] == name and (version is None or entry["version"] == version):
                return Path(entry["path"])
        return None

    def list_models(self) -> list[dict[str, Any]]:
        """List all models with their metadata."""

        entries = self._load_index()
        return [
            {
                "name": entry["name"],
                "version": entry["version"],
                "path": Path(entry["path"]),
                "metadata": entry.get("metadata", {}),
            }
            for entry in entries
        ]

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_file.exists():
            return []
        with self.index_file.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupted registry index at {self.index_file}: {exc}") from exc
        return data if isinstance(data, list) else []

    def _save_index(self, entries: list[dict[str, Any]]) -> None:
        self.registry_root.mkdir(parents=True, exist_ok=True)
        with self.index_file.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)


__all__ = ["ModelRegistryBase", "FileSystemModelRegistry"]
