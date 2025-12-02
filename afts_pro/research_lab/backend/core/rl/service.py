"""Service layer orchestrating RL runs and job lifecycle."""

from __future__ import annotations

import uuid
from typing import Tuple

from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl.models import RLRunRequest, RLRunResult, RLRunStatus
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class RLService:
    """Coordinate RL runs and track status via the job runner."""

    def __init__(self, job_runner: InMemoryJobRunner, rl_runner: RLRunner) -> None:
        self.job_runner = job_runner
        self.rl_runner = rl_runner

    def run_sync(self, request: RLRunRequest) -> RLRunResult:
        """Run RL training synchronously."""

        return self.rl_runner.run(request)

    def submit_job(self, request: RLRunRequest) -> str:
        """Submit an RL job, run immediately, and store result."""

        job_id = self.job_runner.submit_job("rl_run", request.model_dump())
        result = self.rl_runner.run(request)
        self.job_runner.complete_job(job_id, result.model_dump())
        return job_id

    def get_job_result(self, job_id: str) -> Tuple[RLRunStatus, RLRunResult | None, str | None]:
        """Return status, result, and error (if any) for a submitted RL job."""

        try:
            status = self.job_runner.get_status(job_id)
        except KeyError:
            return RLRunStatus.FAILED, None, "Job not found"
        job_status = status.get("status", "pending")
        result_payload = status.get("result")
        result = RLRunResult(**result_payload) if isinstance(result_payload, dict) else None
        error = status.get("error")
        try:
            rl_status = RLRunStatus(job_status)
        except ValueError:
            rl_status = RLRunStatus.PENDING
        return rl_status, result, error


__all__ = ["RLService"]
