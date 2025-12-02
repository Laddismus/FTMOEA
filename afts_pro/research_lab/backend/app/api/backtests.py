"""Backtest API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from research_lab.backend.core.backtests import (
    BacktestRequest,
    BacktestResult,
    RollingKpiBacktestEngine,
    BacktestService,
    BacktestIndexEntry,
)
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend import settings as settings_module

router = APIRouter(prefix="/backtests", tags=["backtests"])

_job_runner = InMemoryJobRunner()
_engine = RollingKpiBacktestEngine()
_persistence = BacktestPersistence(settings_module.get_settings().backtests_dir)
_service = BacktestService(job_runner=_job_runner, engine=_engine, persistence=_persistence)


class BacktestJobSubmitResponse(BaseModel):
    job_id: str


class BacktestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None


class BacktestRunsResponse(BaseModel):
    runs: list[BacktestIndexEntry]


@router.post("/run-sync", response_model=BacktestResult)
def run_sync(request: BacktestRequest) -> BacktestResult:
    """Run a backtest synchronously and return the result."""

    return _service.run_sync(request)


@router.post("/submit", response_model=BacktestJobSubmitResponse)
def submit_backtest(request: BacktestRequest) -> BacktestJobSubmitResponse:
    """Submit a backtest job and return its identifier."""

    job_id = _service.submit_job(request)
    return BacktestJobSubmitResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=BacktestJobStatusResponse)
def get_job_status(job_id: str) -> BacktestJobStatusResponse:
    """Return job status and result."""

    status = _service.get_job_result(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return BacktestJobStatusResponse(job_id=job_id, status=status["status"], result=status.get("result"))


@router.get("/runs", response_model=BacktestRunsResponse)
def list_runs() -> BacktestRunsResponse:
    """List persisted backtest runs."""

    return BacktestRunsResponse(runs=_service.list_runs())


@router.get("/runs/{run_id}", response_model=BacktestResult)
def get_run(run_id: str) -> BacktestResult:
    """Fetch a persisted backtest run by ID."""

    result = _service.get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    return result


__all__ = ["router"]
