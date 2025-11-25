from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    logger.debug("YAML loaded | path=%s", resolved)
    return data


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    resolved = Path(path).resolve()
    with resolved.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp)
    logger.debug("YAML saved | path=%s", resolved)


def reload_global_config():
    """
    Hot-reload helper that avoids circular imports.
    """
    from afts_pro.config.global_config import load_all_configs_into_global

    logger.info("Reloading GlobalConfig from disk.")
    return load_all_configs_into_global()


def get_file_mtimes(paths: List[Path]) -> Dict[Path, float]:
    mtimes: Dict[Path, float] = {}
    for path in paths:
        try:
            mtimes[path] = path.stat().st_mtime
        except FileNotFoundError:
            mtimes[path] = 0.0
    return mtimes
