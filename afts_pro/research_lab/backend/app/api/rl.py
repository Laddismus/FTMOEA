"""RL experimentation API endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from research_lab.backend import settings as settings_module
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl import (
    RLRunRequest,
    RLRunResult,
    RLRunStatus,
    RLPolicyRef,
    RLRewardCheckConfig,
    RLRewardCheckResult,
    RLTrainingMetrics,
)
from research_lab.backend.core.rl.models import RLTrainingConfig
from research_lab.backend.core.rl.policy_loader import RLPolicyLoader
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.service import RLService


router = APIRouter(prefix="/rl", tags=["rl"])

_settings = settings_module.get_settings()
_job_runner = InMemoryJobRunner()
_reward_verifier = RLRewardVerifier()
_rl_runner = RLRunner(policies_dir=_settings.rl_policies_dir, verifier=_reward_verifier)
_policy_loader = RLPolicyLoader(policies_dir=_settings.rl_policies_dir)
_rl_service = RLService(job_runner=_job_runner, rl_runner=_rl_runner)


class RLRunCreateRequest(BaseModel):
    config: RLTrainingConfig
    reward_check: RLRewardCheckConfig | None = None
    notes: str | None = None
    tags: List[str] = []


class RLJobSubmitResponse(BaseModel):
    job_id: str
    run_id: str


class RLJobStatusResponse(BaseModel):
    status: RLRunStatus
    result: RLRunResult | None = None
    error: str | None = None


@router.post("/runs/sync", response_model=RLRunResult)
def run_rl_sync(request: RLRunCreateRequest) -> RLRunResult:
    """Run RL training synchronously."""

    run_request = RLRunRequest(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        config=request.config,
        reward_check=request.reward_check,
        notes=request.notes,
        tags=request.tags,
    )
    return _rl_service.run_sync(run_request)


@router.post("/runs", response_model=RLJobSubmitResponse)
def submit_rl_run(request: RLRunCreateRequest) -> RLJobSubmitResponse:
    """Submit an RL training job."""

    run_request = RLRunRequest(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        config=request.config,
        reward_check=request.reward_check,
        notes=request.notes,
        tags=request.tags,
    )
    job_id = _rl_service.submit_job(run_request)
    return RLJobSubmitResponse(job_id=job_id, run_id=run_request.id)


@router.get("/jobs/{job_id}", response_model=RLJobStatusResponse)
def get_rl_job(job_id: str) -> RLJobStatusResponse:
    """Return status and result for an RL job."""

    status, result, error = _rl_service.get_job_result(job_id)
    if status == RLRunStatus.FAILED and error == "Job not found":
        raise HTTPException(status_code=404, detail="Job not found")
    return RLJobStatusResponse(status=status, result=result, error=error)


@router.get("/policies", response_model=List[RLPolicyRef])
def list_policies() -> List[RLPolicyRef]:
    """List known RL policies."""

    return _policy_loader.list_policies()


@router.get("/policies/{key}", response_model=RLPolicyRef)
def get_policy(key: str) -> RLPolicyRef:
    """Fetch a policy by key."""

    policy = _policy_loader.get_policy(key)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


class RewardCheckRequest(BaseModel):
    metrics: RLTrainingMetrics
    config: RLRewardCheckConfig


@router.post("/reward-check", response_model=RLRewardCheckResult)
def verify_reward(request: RewardCheckRequest) -> RLRewardCheckResult:
    """Verify reward metrics against thresholds."""

    return _reward_verifier.verify(metrics=request.metrics, config=request.config)


__all__ = ["router"]
