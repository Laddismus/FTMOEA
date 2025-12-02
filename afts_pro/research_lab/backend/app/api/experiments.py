"""Experiment orchestration API endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from research_lab.backend import settings as settings_module
from research_lab.backend.core.backtests import BacktestRequest, BacktestService, RollingKpiBacktestEngine
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.experiments import (
    ExperimentConfig,
    ExperimentRunStatus,
    ExperimentStatus,
    ExperimentSummary,
    ExperimentParamPoint,
    ExperimentStrategyRef,
    ExperimentService,
    ExperimentLeaderboard,
    ExperimentScorer,
)
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.job_runner import InMemoryJobRunner


router = APIRouter(prefix="/experiments", tags=["experiments"])

_settings = settings_module.get_settings()
_job_runner = InMemoryJobRunner()
_engine = RollingKpiBacktestEngine()
_backtest_persistence = BacktestPersistence(_settings.backtests_dir)
_backtest_service = BacktestService(job_runner=_job_runner, engine=_engine, persistence=_backtest_persistence)
_experiment_persistence = ExperimentPersistence(_settings.experiments_dir)
_experiment_scorer = ExperimentScorer(backtest_persistence=_backtest_persistence)
_experiment_service = ExperimentService(
    backtest_service=_backtest_service,
    experiment_persistence=_experiment_persistence,
    scorer=_experiment_scorer,
)


class ExperimentCreateRequest(BaseModel):
    name: str
    description: str | None = None
    strategy: ExperimentStrategyRef
    base_backtest: BacktestRequest
    param_grid: List[ExperimentParamPoint] = []
    tags: List[str] = []
    metadata: dict = {}


class ExperimentDetailResponse(BaseModel):
    config: ExperimentConfig
    runs: List[ExperimentRunStatus]


@router.post("", response_model=ExperimentConfig)
def create_experiment(request: ExperimentCreateRequest) -> ExperimentConfig:
    """Create a new experiment with pending runs."""

    config = ExperimentConfig(
        id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        created_at=datetime.now(timezone.utc),
        strategy=request.strategy,
        base_backtest=request.base_backtest,
        param_grid=request.param_grid,
        tags=request.tags,
        metadata=request.metadata,
    )
    _experiment_service.create_experiment(config)
    return config


@router.post("/{experiment_id}/launch", response_model=ExperimentDetailResponse)
def launch_experiment(experiment_id: str) -> ExperimentDetailResponse:
    """Launch all runs for an experiment."""

    try:
        config, runs = _experiment_service.launch_experiment(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentDetailResponse(config=config, runs=runs)


@router.get("", response_model=List[ExperimentSummary])
def list_experiments() -> List[ExperimentSummary]:
    """List all experiments."""

    return _experiment_service.list_experiments()


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(experiment_id: str, refresh: bool = True) -> ExperimentDetailResponse:
    """Return experiment details (optionally refreshing job statuses)."""

    if refresh:
        try:
            config, runs = _experiment_service.refresh_status(experiment_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return ExperimentDetailResponse(config=config, runs=runs)
    loaded = _experiment_service.get_experiment(experiment_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    config, runs = loaded
    return ExperimentDetailResponse(config=config, runs=runs)


@router.get("/{experiment_id}/leaderboard", response_model=ExperimentLeaderboard)
def get_leaderboard(experiment_id: str) -> ExperimentLeaderboard:
    """Return leaderboard for a specific experiment."""

    leaderboard = _experiment_service.get_leaderboard(experiment_id)
    if leaderboard is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return leaderboard


__all__ = ["router"]
