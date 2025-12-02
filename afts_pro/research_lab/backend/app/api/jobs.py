"""Job runner endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from research_lab.backend.core.job_runner import InMemoryJobRunner

router = APIRouter(tags=["jobs"])
job_runner = InMemoryJobRunner()


class JobCreateRequest(BaseModel):
    """Request payload for submitting a job."""

    job_type: str = Field(..., description="Job category identifier.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Payload for the job.")


@router.post("", status_code=201)
def submit_job(payload: JobCreateRequest) -> dict[str, Any]:
    """Submit a new job to the in-memory runner."""

    job_id = job_runner.submit_job(payload.job_type, payload.payload)
    return job_runner.get_status(job_id)


@router.get("/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    """Return status information for a job."""

    try:
        return job_runner.get_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("")
def list_jobs(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    """List the most recent jobs."""

    return job_runner.list_jobs(limit=limit)


__all__ = ["router"]
