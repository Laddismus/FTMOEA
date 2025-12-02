"""Research-specific configuration loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from research_lab.backend.settings import get_settings

logger = logging.getLogger(__name__)


class ResearchConfigLoader:
    """Loads YAML configurations from the research config directory."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        settings = get_settings()
        self.base_path: Path = Path(base_path) if base_path else settings.config_root

    def load_config(self, name: str) -> dict[str, Any]:
        """Load a YAML config by name without extension."""

        path = self.base_path / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Research config '{name}' not found at {path}")

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            message = f"Invalid YAML in research config '{name}': {exc}"
            logger.error(message)
            raise ValueError(message) from exc

        logger.debug("Research config loaded | name=%s path=%s", name, path)
        return data

    def list_configs(self) -> list[str]:
        """Return all available config names (without extension)."""

        if not self.base_path.exists():
            return []

        return sorted([config.stem for config in self.base_path.glob("*.yaml")])


__all__ = ["ResearchConfigLoader"]
