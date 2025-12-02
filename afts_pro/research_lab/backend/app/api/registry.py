"""Model registry endpoints."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from research_lab.backend.core.model_registry import FileSystemModelRegistry

router = APIRouter(tags=["registry"])
registry = FileSystemModelRegistry()


class ModelRegistrationRequest(BaseModel):
    """Request payload for registering a model."""

    name: str = Field(..., description="Name of the model.")
    version: str = Field(..., description="Version identifier.")
    path: str = Field(..., description="Filesystem path to the model artifact.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata to persist with the model."
    )


@router.post("/models", status_code=201)
def register_model(payload: ModelRegistrationRequest) -> dict[str, Any]:
    """Register a model in the filesystem registry stub."""

    registry.register_model(
        name=payload.name,
        version=payload.version,
        path=Path(payload.path),
        metadata=payload.metadata,
    )
    return {"name": payload.name, "version": payload.version, "path": payload.path, "metadata": payload.metadata or {}}


@router.get("/models")
def list_registered_models() -> list[dict[str, Any]]:
    """Return all registered models."""

    entries = registry.list_models()
    return [
        {"name": entry["name"], "version": entry["version"], "path": str(entry["path"]), "metadata": entry.get("metadata", {})}
        for entry in entries
    ]


@router.get("/models/{name}")
def get_model(name: str, version: str | None = None) -> dict[str, Any]:
    """Retrieve a model path by name and optional version."""

    entries = [
        entry for entry in registry.list_models() if entry["name"] == name and (version is None or entry["version"] == version)
    ]
    if not entries:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found.")

    entry = entries[-1]
    model_path = registry.get_model(name, entry["version"])
    if model_path is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found.")

    return {"name": name, "version": entry["version"], "path": str(model_path), "metadata": entry.get("metadata", {})}


__all__ = ["router"]
