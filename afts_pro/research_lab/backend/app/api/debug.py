"""Debug endpoints to exercise error handling."""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/http-error")
async def trigger_http_error():
    raise HTTPException(status_code=400, detail="forced error")


@router.get("/generic-error")
async def trigger_generic_error():
    raise ValueError("boom")


__all__ = ["router"]
