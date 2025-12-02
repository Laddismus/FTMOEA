"""API endpoints for RL experiments and leaderboards."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from research_lab.backend import settings as settings_module
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl import RLRunner, RLRewardVerifier, RLService
from research_lab.backend.core.rl.policy_loader import RLPolicyLoader
from research_lab.backend.core.rl_experiments import (
    RlExperimentConfig,
    RlExperimentLeaderboard,
    RlExperimentParamPoint,
    RlExperimentRunStatus,
    RlExperimentService,
    RlExperimentStatus,
    RlExperimentScorer,
)
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence
from research_lab.backend.core.rl.models import RLTrainingConfig, RLEnvRef, RLAlgo, RLRewardCheckConfig


router = APIRouter(prefix="/rl-experiments", tags=["rl-experiments"])

_settings = settings_module.get_settings()
_job_runner = InMemoryJobRunner()
_reward_verifier = RLRewardVerifier()
_rl_runner = RLRunner(policies_dir=_settings.rl_policies_dir, verifier=_reward_verifier)
_rl_service = RLService(job_runner=_job_runner, rl_runner=_rl_runner)
_rl_experiment_persistence = RlExperimentPersistence(_settings.rl_experiments_dir)
_rl_experiment_scorer = RlExperimentScorer(rl_service=_rl_service)
_rl_experiment_service = RlExperimentService(
    rl_service=_rl_service, persistence=_rl_experiment_persistence, scorer=_rl_experiment_scorer
)


class RlExperimentCreateRequest(BaseModel):
    name: str
    description: str | None = None
    env: RLEnvRef
    algo: RLAlgo
    base_training: RLTrainingConfig
    param_grid: List[RlExperimentParamPoint] = []
    reward_checks: List[RLRewardCheckConfig] = []
    tags: List[str] = []
    metadata: dict = {}


class RlExperimentDetailResponse(BaseModel):
    config: RlExperimentConfig
    runs: List[RlExperimentRunStatus]


@router.post("", response_model=RlExperimentConfig)
def create_experiment(request: RlExperimentCreateRequest) -> RlExperimentConfig:
    config = RlExperimentConfig(
        id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        created_at=datetime.now(timezone.utc),
        env=request.env,
        algo=request.algo,
        base_training=request.base_training,
        param_grid=request.param_grid,
        reward_checks=request.reward_checks,
        tags=request.tags,
        metadata=request.metadata,
    )
    _rl_experiment_service.create_experiment(config)
    return config


@router.post("/{experiment_id}/launch", response_model=RlExperimentDetailResponse)
def launch_experiment(experiment_id: str) -> RlExperimentDetailResponse:
    try:
        config, runs = _rl_experiment_service.launch_experiment(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return RlExperimentDetailResponse(config=config, runs=runs)


@router.get("/{experiment_id}", response_model=RlExperimentDetailResponse)
def get_experiment(experiment_id: str) -> RlExperimentDetailResponse:
    loaded = _rl_experiment_service.get_experiment(experiment_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    config, runs = loaded
    return RlExperimentDetailResponse(config=config, runs=runs)


@router.get("/{experiment_id}/leaderboard", response_model=RlExperimentLeaderboard)
def get_leaderboard(experiment_id: str) -> RlExperimentLeaderboard:
    leaderboard = _rl_experiment_service.get_leaderboard(experiment_id)
    if leaderboard is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return leaderboard


__all__ = ["router"]
