"""Pydantic models for Python strategy metadata and requests."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PythonStrategyMetadata(BaseModel):
    """Metadata describing a Python strategy."""

    key: str
    name: str
    version: str
    description: Optional[str] = None
    module_path: str
    class_name: str
    tags: list[str] = Field(default_factory=list)
    params_schema: dict[str, Any] = Field(default_factory=dict)


class PythonStrategyRegistrationRequest(BaseModel):
    """Request payload to validate/register a Python strategy."""

    module_path: str
    class_name: str
    key: Optional[str] = None
    override_metadata: Optional[dict[str, Any]] = None


class PythonStrategyValidationResult(BaseModel):
    """Result of a validation/import attempt."""

    valid: bool
    error: Optional[str] = None
    metadata: Optional[PythonStrategyMetadata] = None


__all__ = [
    "PythonStrategyMetadata",
    "PythonStrategyRegistrationRequest",
    "PythonStrategyValidationResult",
]
