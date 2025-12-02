"""Loader utilities for Python strategies."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Type

from research_lab.backend.core.python_strategies.interface import BasePythonStrategy, PythonStrategyInterface
from research_lab.backend.core.python_strategies.models import PythonStrategyMetadata


def import_strategy_class(module_path: str, class_name: str) -> Type[PythonStrategyInterface]:
    """Import a strategy class and ensure it adheres to the strategy interface."""

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - exercised in tests via exception path
        raise ImportError(f"Failed to import module '{module_path}': {exc}") from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'.") from exc

    if not inspect.isclass(cls):
        raise TypeError(f"Attribute '{class_name}' in module '{module_path}' is not a class.")
    if not issubclass(cls, PythonStrategyInterface):
        raise TypeError(f"Class '{class_name}' must implement PythonStrategyInterface.")

    return cls


def extract_metadata(cls: Type[PythonStrategyInterface]) -> PythonStrategyMetadata:
    """Extract metadata from a strategy class."""

    if not issubclass(cls, PythonStrategyInterface):
        raise TypeError("Class does not implement PythonStrategyInterface.")

    instance = cls() if issubclass(cls, BasePythonStrategy) else None
    base_metadata = instance.get_metadata() if instance else {}

    strategy_key = getattr(cls, "strategy_key", None) or base_metadata.get("key") or f"{cls.__module__}.{cls.__name__}"
    strategy_name = getattr(cls, "strategy_name", None) or base_metadata.get("name") or cls.__name__
    strategy_version = getattr(cls, "strategy_version", None) or base_metadata.get("version") or "1.0.0"
    strategy_description = getattr(cls, "strategy_description", None) or base_metadata.get("description")
    strategy_tags = getattr(cls, "strategy_tags", None) or base_metadata.get("tags") or []
    params_schema = getattr(cls, "strategy_params_schema", None) or base_metadata.get("params_schema") or {}

    return PythonStrategyMetadata(
        key=strategy_key,
        name=strategy_name,
        version=strategy_version,
        description=strategy_description,
        module_path=cls.__module__,
        class_name=cls.__name__,
        tags=strategy_tags,
        params_schema=params_schema,
    )


__all__ = ["import_strategy_class", "extract_metadata"]
