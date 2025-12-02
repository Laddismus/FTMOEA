"""In-memory job runner stub for the research backend."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any


class JobRunnerBase(ABC):
    """Interface for submitting and tracking jobs."""

    @abstractmethod
    def submit_job(self, job_type: str, payload: dict[str, Any]) -> str:
        """Submit a job and return its identifier."""

    @abstractmethod
    def get_status(self, job_id: str) -> dict[str, Any]:
        """Return the status payload for a job."""

    @abstractmethod
    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent jobs up to the provided limit."""


class InMemoryJobRunner(JobRunnerBase):
    """Simple in-memory job runner suitable for skeleton usage."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def submit_job(self, job_type: str, payload: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        job_info = {
            "job_id": job_id,
            "job_type": job_type,
            "payload": payload,
            "status": "queued",
            "created_at": time.time(),
            "result": None,
        }
        self._jobs[job_id] = job_info
        self._order.append(job_id)
        return job_id

    def get_status(self, job_id: str) -> dict[str, Any]:
        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' not found")
        return self._jobs[job_id]

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        recent_ids = self._order[-limit:]
        return [self._jobs[job_id] for job_id in reversed(recent_ids)]

    def complete_job(self, job_id: str, result: Any) -> None:
        """Mark a job as completed and store its result."""

        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' not found")
        self._jobs[job_id]["status"] = "completed"
        self._jobs[job_id]["result"] = result


__all__ = ["JobRunnerBase", "InMemoryJobRunner"]
