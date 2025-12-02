"""Global error handling utilities for the Research Lab backend."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("research_lab")


class ErrorResponse(BaseModel):
    """Standardized error response envelope."""

    detail: str
    error_code: str
    request_id: str | None = None


def _error_code_from_status(status_code: int) -> str:
    if status_code == 404:
        return "not_found"
    if status_code == 400:
        return "bad_request"
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 422:
        return "validation_error"
    return "http_error"


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for standardized responses."""

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        detail = exc.detail if exc.detail else "HTTP error"
        if not isinstance(detail, str):
            detail = str(detail)
        error = ErrorResponse(detail=detail, error_code=_error_code_from_status(exc.status_code), request_id=request_id)
        logger.warning("HTTPException: %s", error.model_dump(), extra={"request_id": request_id})
        return JSONResponse(status_code=exc.status_code, content=error.model_dump())

    @app.exception_handler(Exception)
    async def _handle_generic_exception(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        error = ErrorResponse(detail="Internal server error", error_code="internal_error", request_id=request_id)
        logger.exception("Unhandled exception", extra={"request_id": request_id})
        return JSONResponse(status_code=500, content=error.model_dump())


def get_request_id(request: Request) -> str:
    """Generate or retrieve a request correlation id."""

    if getattr(request.state, "request_id", None):
        return request.state.request_id
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


__all__ = ["ErrorResponse", "register_exception_handlers", "get_request_id"]
