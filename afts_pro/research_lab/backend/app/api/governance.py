"""API endpoints for governance model hub."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from research_lab.backend import settings as settings_module
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.governance import (
    GovernanceService,
    GovernanceRegistry,
    ModelEntry,
    ModelEntrySummary,
    ModelStage,
    ModelType,
)
from research_lab.backend.core.governance.models import GovernanceTag
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl.policy_loader import RLPolicyLoader
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl.models import RLRunStatus


router = APIRouter(prefix="/governance", tags=["governance"])

_settings = settings_module.get_settings()
_job_runner = InMemoryJobRunner()
_rl_service = RLService(job_runner=_job_runner, rl_runner=RLRunner(policies_dir=_settings.rl_policies_dir, verifier=RLRewardVerifier()))
_registry = GovernanceRegistry(_settings.governance_dir)
_service = GovernanceService(
    registry=_registry,
    backtest_persistence=BacktestPersistence(_settings.backtests_dir),
    experiment_persistence=ExperimentPersistence(_settings.experiments_dir),
    rl_service=_rl_service,
    rl_experiment_persistence=RlExperimentPersistence(_settings.rl_experiments_dir),
)


class RegisterBacktestRequest(BaseModel):
    name: str
    backtest_id: str
    experiment_id: str | None = None
    experiment_run_id: str | None = None
    initial_stage: ModelStage = ModelStage.CANDIDATE
    tags: List[GovernanceTag] = []


class RegisterRlRequest(BaseModel):
    name: str
    rl_run_id: str
    rl_experiment_id: str | None = None
    rl_experiment_run_id: str | None = None
    initial_stage: ModelStage = ModelStage.CANDIDATE
    tags: List[GovernanceTag] = []


class PromoteRequest(BaseModel):
    new_stage: ModelStage
    note: str | None = None


class ScoreUpdateRequest(BaseModel):
    composite_score: float
    note: str | None = None


@router.post("/models/from-backtest", response_model=ModelEntry)
def register_from_backtest(request: RegisterBacktestRequest) -> ModelEntry:
    try:
        return _service.register_from_backtest(
            name=request.name,
            backtest_id=request.backtest_id,
            experiment_id=request.experiment_id,
            experiment_run_id=request.experiment_run_id,
            initial_stage=request.initial_stage,
            tags=request.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/models/from-rl", response_model=ModelEntry)
def register_from_rl(request: RegisterRlRequest) -> ModelEntry:
    try:
        return _service.register_from_rl_run(
            name=request.name,
            rl_run_id=request.rl_run_id,
            rl_experiment_id=request.rl_experiment_id,
            rl_experiment_run_id=request.rl_experiment_run_id,
            initial_stage=request.initial_stage,
            tags=request.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/models/{model_id}/promote", response_model=ModelEntry)
def promote_model(model_id: str, request: PromoteRequest) -> ModelEntry:
    try:
        return _service.promote(model_id=model_id, new_stage=request.new_stage, note=request.note)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/models/{model_id}/score", response_model=ModelEntry)
def score_model(model_id: str, request: ScoreUpdateRequest) -> ModelEntry:
    try:
        return _service.update_score(model_id=model_id, composite_score=request.composite_score, note=request.note)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/models", response_model=List[ModelEntrySummary])
def list_models(stage: ModelStage | None = Query(default=None), type: ModelType | None = Query(default=None)) -> List[ModelEntrySummary]:
    return _service.list_models(stage=stage, type=type)


@router.get("/models/{model_id}", response_model=ModelEntry)
def get_model(model_id: str) -> ModelEntry:
    entry = _service.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return entry


__all__ = ["router"]
