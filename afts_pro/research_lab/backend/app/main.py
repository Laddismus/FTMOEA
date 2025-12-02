"""FastAPI entrypoint for the Research Lab backend."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from research_lab.backend.app.api import api_router
from research_lab.backend.app.errors import register_exception_handlers
from research_lab.backend.app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("research_lab")

app = FastAPI(title="AFTS-PRO Research Lab Backend", version="0.1.0")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception during request", extra={"request_id": request_id})
        from research_lab.backend.app.errors import ErrorResponse  # local import to avoid cycle

        error = ErrorResponse(detail="Internal server error", error_code="internal_error", request_id=request_id)
        return JSONResponse(status_code=500, content=error.model_dump())
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request completed",
        extra={"request_id": request_id, "method": request.method, "path": request.url.path, "status_code": response.status_code, "duration_ms": duration_ms},
    )
    response.headers["X-Request-ID"] = request_id
    return response


register_exception_handlers(app)
app.include_router(api_router, prefix="/api")

__all__ = ["app"]
