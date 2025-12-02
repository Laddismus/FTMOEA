"""Health endpoint for the Research Lab backend."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight health status."""

    return {"status": "ok", "service": "research_lab_backend"}


__all__ = ["router"]
