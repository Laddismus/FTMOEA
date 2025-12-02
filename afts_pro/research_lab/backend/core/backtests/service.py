"""Backtest service coordinating engine execution and job runner."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

from research_lab.backend.core.backtests.engine import BacktestEngineInterface
from research_lab.backend.core.backtests.models import BacktestRequest, BacktestResult, BacktestIndexEntry
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.job_runner import InMemoryJobRunner


class BacktestService:
    """Service that executes backtests and manages job lifecycle."""

    def __init__(self, job_runner: InMemoryJobRunner, engine: BacktestEngineInterface, persistence: BacktestPersistence) -> None:
        self.job_runner = job_runner
        self.engine = engine
        self.persistence = persistence

    def run_sync(self, request: BacktestRequest) -> BacktestResult:
        """Run a backtest synchronously."""

        result = self.engine.run_backtest(request)
        self._ensure_identity(result)
        self.persistence.save_result(result)
        return result

    def submit_job(self, request: BacktestRequest) -> str:
        """Submit a backtest job and return its ID (runs immediately for stub)."""

        job_id = self.job_runner.submit_job("backtest", request.model_dump())
        result = self.engine.run_backtest(request)
        self._ensure_identity(result)
        self.persistence.save_result(result)
        self.job_runner.complete_job(job_id, result.model_dump())
        return job_id

    def get_job_result(self, job_id: str) -> dict | None:
        """Return job status and result if available."""

        try:
            return self.job_runner.get_status(job_id)
        except KeyError:
            return None

    def list_runs(self) -> list[BacktestIndexEntry]:
        """List stored backtest runs."""

        return self.persistence.list_runs()

    def get_run(self, run_id: str) -> BacktestResult | None:
        """Load a stored backtest run by ID."""

        return self.persistence.load_result(run_id)

    def _ensure_identity(self, result: BacktestResult) -> None:
        if not result.id:
            result.id = str(uuid.uuid4())
        if result.created_at is None:
            result.created_at = datetime.now(tz=timezone.utc)


__all__ = ["BacktestService"]
