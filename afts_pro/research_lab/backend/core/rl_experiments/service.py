"""Service orchestrating RL experiments and job lifecycle."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Any

from research_lab.backend.core.rl.models import RLRunRequest, RLTrainingConfig, RLRunStatus
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl_experiments.models import (
    RlExperimentConfig,
    RlExperimentLeaderboard,
    RlExperimentParamPoint,
    RlExperimentRunStatus,
    RlExperimentStatus,
)
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence
from research_lab.backend.core.rl_experiments.scoring import RlExperimentScorer
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class RlExperimentService:
    """Manage RL experiments: creation, launching runs, refreshing status, and leaderboards."""

    def __init__(self, rl_service: RLService, persistence: RlExperimentPersistence, scorer: RlExperimentScorer) -> None:
        self.rl_service = rl_service
        self.persistence = persistence
        self.scorer = scorer

    def create_experiment(self, config: RlExperimentConfig) -> RlExperimentConfig:
        runs = self._initial_runs(config.param_grid)
        self.persistence.save_experiment(config, runs)
        return config

    def launch_experiment(self, experiment_id: str) -> Tuple[RlExperimentConfig, List[RlExperimentRunStatus]]:
        loaded = self.persistence.load_experiment(experiment_id)
        if loaded is None:
            raise ValueError("Experiment not found")
        config, runs = loaded
        updated_runs: List[RlExperimentRunStatus] = []
        for idx, run in enumerate(runs):
            if run.status not in {RlExperimentStatus.PENDING, RlExperimentStatus.FAILED}:
                updated_runs.append(run)
                continue
            training = self._build_training_config(config.base_training, config.param_grid[idx].values if idx < len(config.param_grid) else {})
            run_request = RLRunRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(timezone.utc),
                config=training,
                reward_check=config.reward_checks[0] if config.reward_checks else None,
                notes=None,
                tags=config.tags,
            )
            job_id = self.rl_service.submit_job(run_request)
            updated_runs.append(
                RlExperimentRunStatus(
                    run_id=run.run_id,
                    job_id=job_id,
                    status=RlExperimentStatus.RUNNING,
                    error=None,
                    rl_run_id=run_request.id,
                )
            )
        self.persistence.save_experiment(config, updated_runs)
        return config, updated_runs

    def refresh_status(self, experiment_id: str) -> Tuple[RlExperimentConfig, List[RlExperimentRunStatus]]:
        loaded = self.persistence.load_experiment(experiment_id)
        if loaded is None:
            raise ValueError("Experiment not found")
        config, runs = loaded
        refreshed: List[RlExperimentRunStatus] = []
        for run in runs:
            if run.job_id is None:
                refreshed.append(run)
                continue
            status, result, error = self.rl_service.get_job_result(run.job_id)
            if status == RLRunStatus.COMPLETED:
                refreshed.append(
                    RlExperimentRunStatus(
                        run_id=run.run_id,
                        job_id=run.job_id,
                        status=RlExperimentStatus.COMPLETED,
                        error=None,
                        rl_run_id=result.id if result else run.rl_run_id,
                    )
                )
            elif status == RLRunStatus.FAILED:
                refreshed.append(
                    RlExperimentRunStatus(
                        run_id=run.run_id,
                        job_id=run.job_id,
                        status=RlExperimentStatus.FAILED,
                        error=error,
                        rl_run_id=run.rl_run_id,
                    )
                )
            else:
                refreshed.append(run)
        self.persistence.save_experiment(config, refreshed)
        return config, refreshed

    def get_experiment(self, experiment_id: str) -> Optional[Tuple[RlExperimentConfig, List[RlExperimentRunStatus]]]:
        return self.persistence.load_experiment(experiment_id)

    def get_leaderboard(self, experiment_id: str) -> Optional[RlExperimentLeaderboard]:
        loaded = self.persistence.load_experiment(experiment_id)
        if loaded is None:
            return None
        config, runs = loaded
        return self.scorer.build_leaderboard(experiment_id, config, runs)

    def _initial_runs(self, param_grid: List[RlExperimentParamPoint]) -> List[RlExperimentRunStatus]:
        if not param_grid:
            return [
                RlExperimentRunStatus(run_id=str(uuid.uuid4()), status=RlExperimentStatus.PENDING, job_id=None, error=None, rl_run_id=None)
            ]
        return [
            RlExperimentRunStatus(run_id=str(uuid.uuid4()), status=RlExperimentStatus.PENDING, job_id=None, error=None, rl_run_id=None)
            for _ in param_grid
        ]

    def _build_training_config(self, base: RLTrainingConfig, overrides: dict[str, Any]) -> RLTrainingConfig:
        training = base.model_copy(deep=True)
        if overrides:
            merged_hparams = dict(training.hyperparams)
            merged_hparams.update(overrides)
            training.hyperparams = merged_hparams
            # apply direct fields if matching keys exist in overrides
            for key, value in overrides.items():
                if hasattr(training, key):
                    setattr(training, key, value)
        return training


__all__ = ["RlExperimentService"]
