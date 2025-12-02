"""API endpoints for Python strategy registration and validation."""

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from research_lab.backend.core.python_strategies import (
    PythonStrategyRegistry,
    PythonStrategyValidationResult,
    PythonStrategyMetadata,
    PythonStrategyRegistrationRequest,
    import_strategy_class,
    extract_metadata,
)

router = APIRouter(prefix="/python-strategies", tags=["python-strategies"])

registry = PythonStrategyRegistry()


class PythonStrategiesResponse(BaseModel):
    strategies: List[PythonStrategyMetadata]


@router.get("", response_model=PythonStrategiesResponse)
def list_strategies() -> PythonStrategiesResponse:
    """List all registered Python strategies."""

    return PythonStrategiesResponse(strategies=registry.list_strategies())


@router.post("/validate-import", response_model=PythonStrategyValidationResult)
def validate_strategy(request: PythonStrategyRegistrationRequest) -> PythonStrategyValidationResult:
    """Validate that a strategy can be imported and conforms to the interface."""

    try:
        cls = import_strategy_class(request.module_path, request.class_name)
        metadata = extract_metadata(cls)
        if request.override_metadata:
            metadata_dict = metadata.model_dump()
            metadata_dict.update(request.override_metadata)
            metadata = PythonStrategyMetadata(**metadata_dict)
        if request.key:
            metadata = metadata.model_copy(update={"key": request.key})
        return PythonStrategyValidationResult(valid=True, metadata=metadata)
    except Exception as exc:  # pragma: no cover - error path validated in tests
        return PythonStrategyValidationResult(valid=False, error=str(exc))


@router.post("/register", response_model=PythonStrategyMetadata)
def register_strategy(request: PythonStrategyRegistrationRequest) -> PythonStrategyMetadata:
    """Validate and register a Python strategy."""

    validation = validate_strategy(request)
    if not validation.valid or validation.metadata is None:
        raise HTTPException(status_code=400, detail=validation.error or "Validation failed.")

    metadata = validation.metadata
    registry.register_strategy(metadata)
    return metadata


__all__ = ["router"]
