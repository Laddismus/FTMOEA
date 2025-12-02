"""Service orchestrating experiments composed of multiple backtests."""

from __future__ import annotations

import uuid
from typing import List, Tuple, Any

from research_lab.backend.core.backtests.models import BacktestRequest
from research_lab.backend.core.backtests.service import BacktestService
from research_lab.backend.core.experiments.models import (
    ExperimentConfig,
    ExperimentParamPoint,
    ExperimentRunStatus,
    ExperimentStatus,
    ExperimentSummary,
    ExperimentLeaderboard,
)
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.experiments.scoring import ExperimentScorer


class ExperimentService:
    """Manage experiment lifecycle and orchestrate backtest runs."""

    def __init__(
        self,
        backtest_service: BacktestService,
        experiment_persistence: ExperimentPersistence,
        scorer: ExperimentScorer,
    ) -> None:
        self.backtest_service = backtest_service
        self.experiment_persistence = experiment_persistence
        self.scorer = scorer

    def create_experiment(self, config: ExperimentConfig) -> ExperimentConfig:
        """Persist a new experiment with pending runs."""

        runs = self._initial_runs(config.param_grid)
        self.experiment_persistence.save_experiment(config, runs)
        return config

    def launch_experiment(self, experiment_id: str) -> Tuple[ExperimentConfig, List[ExperimentRunStatus]]:
        """Launch all runs of an experiment by submitting backtests."""

        loaded = self.experiment_persistence.load_experiment(experiment_id)
        if loaded is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        config, runs = loaded

        updated_runs: List[ExperimentRunStatus] = []
        for idx, run in enumerate(runs):
            if run.status not in {ExperimentStatus.PENDING, ExperimentStatus.FAILED}:
                updated_runs.append(run)
                continue
            param_values = config.param_grid[idx].values if idx < len(config.param_grid) else {}
            request = self._build_request(config.base_backtest, param_values)
            job_id = self.backtest_service.submit_job(request)
            updated_runs.append(
                ExperimentRunStatus(
                    run_id=run.run_id,
                    job_id=job_id,
                    status=ExperimentStatus.RUNNING,
                    backtest_id=None,
                    error=None,
                )
            )

        self.experiment_persistence.save_experiment(config, updated_runs)
        return config, updated_runs

    def refresh_status(self, experiment_id: str) -> Tuple[ExperimentConfig, List[ExperimentRunStatus]]:
        """Refresh run statuses from the underlying job runner and persist changes."""

        loaded = self.experiment_persistence.load_experiment(experiment_id)
        if loaded is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        config, runs = loaded
        refreshed: List[ExperimentRunStatus] = []
        for run in runs:
            if not run.job_id:
                refreshed.append(run)
                continue
            status = self.backtest_service.get_job_result(run.job_id)
            if status is None:
                refreshed.append(run)
                continue
            if status.get("status") == "completed" and status.get("result"):
                refreshed.append(
                    ExperimentRunStatus(
                        run_id=run.run_id,
                        job_id=run.job_id,
                        backtest_id=status["result"].get("id"),
                        status=ExperimentStatus.COMPLETED,
                        error=None,
                    )
                )
            elif status.get("status") == "failed":
                refreshed.append(
                    ExperimentRunStatus(
                        run_id=run.run_id,
                        job_id=run.job_id,
                        backtest_id=None,
                        status=ExperimentStatus.FAILED,
                        error=status.get("error"),
                    )
                )
            else:
                refreshed.append(run)

        self.experiment_persistence.save_experiment(config, refreshed)
        return config, refreshed

    def get_experiment(self, experiment_id: str) -> Tuple[ExperimentConfig, List[ExperimentRunStatus]] | None:
        """Return full experiment configuration and runs."""

        loaded = self.experiment_persistence.load_experiment(experiment_id)
        return loaded

    def list_experiments(self) -> List[ExperimentSummary]:
        """List experiment summaries."""

        return self.experiment_persistence.list_experiments()

    def get_leaderboard(self, experiment_id: str) -> ExperimentLeaderboard | None:
        """Return leaderboard data for a given experiment."""

        loaded = self.experiment_persistence.load_experiment(experiment_id)
        if loaded is None:
            return None
        config, runs = loaded
        return self.scorer.build_leaderboard(experiment_id, config, runs)

    def _initial_runs(self, param_grid: List[ExperimentParamPoint]) -> List[ExperimentRunStatus]:
        if not param_grid:
            return [
                ExperimentRunStatus(
                    run_id=str(uuid.uuid4()),
                    status=ExperimentStatus.PENDING,
                    job_id=None,
                    backtest_id=None,
                    error=None,
                )
            ]
        return [
            ExperimentRunStatus(
                run_id=str(uuid.uuid4()),
                status=ExperimentStatus.PENDING,
                job_id=None,
                backtest_id=None,
                error=None,
            )
            for _ in param_grid
        ]

    def _build_request(self, base: BacktestRequest, param_values: dict[str, Any]) -> BacktestRequest:
        """Clone base request and merge provided parameter values into strategy_params."""

        request = base.model_copy(deep=True)
        if param_values:
            updated_params = dict(request.strategy_params)
            updated_params.update(param_values)
            request.strategy_params = updated_params
        return request
