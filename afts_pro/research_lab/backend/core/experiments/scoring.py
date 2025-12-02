"""Scoring utilities to build experiment leaderboards from backtest results."""

from __future__ import annotations

from typing import List

from research_lab.backend.core.backtests.models import BacktestRequest, BacktestResult
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.experiments.models import (
    ExperimentConfig,
    ExperimentLeaderboard,
    ExperimentRunScore,
    ExperimentRunStatus,
    ExperimentStatus,
)


class ExperimentScorer:
    """Build leaderboards for experiments based on stored backtest results."""

    def __init__(self, backtest_persistence: BacktestPersistence) -> None:
        self.backtest_persistence = backtest_persistence

    def build_leaderboard(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        runs: List[ExperimentRunStatus],
    ) -> ExperimentLeaderboard:
        """Construct leaderboard data for an experiment."""

        scored_runs: list[ExperimentRunScore] = []
        pass_count = 0
        param_base = config.base_backtest.strategy_params or {}

        for idx, run in enumerate(runs):
            if run.backtest_id is None:
                continue
            result = self.backtest_persistence.load_result(run.backtest_id)
            if result is None:
                continue
            params = dict(param_base)
            if idx < len(config.param_grid):
                params.update(config.param_grid[idx].values)
            score = self._build_run_score(run_id=run.run_id, backtest_id=run.backtest_id, params=params, result=result)
            if score.ftmo_passed:
                pass_count += 1
            scored_runs.append(score)

        # sort: ftmo_passed True first, then total_return desc, then profit_factor desc
        scored_runs.sort(
            key=lambda s: (
                0 if s.ftmo_passed else 1,
                -(s.total_return if s.total_return is not None else float("-inf")),
                -(s.profit_factor if s.profit_factor is not None else float("-inf")),
            )
        )
        for idx, run in enumerate(scored_runs, start=1):
            run.rank = idx

        ftmo_pass_rate = None
        if scored_runs:
            ftmo_pass_rate = pass_count / len(scored_runs)

        leaderboard = ExperimentLeaderboard(
            experiment_id=experiment_id,
            name=config.name,
            created_at=config.created_at,
            total_runs=len(scored_runs),
            ftmo_pass_rate=ftmo_pass_rate,
            runs=scored_runs,
        )
        return leaderboard

    def _build_run_score(self, run_id: str, backtest_id: str, params: dict, result: BacktestResult) -> ExperimentRunScore:
        kpi = result.kpi_summary
        ftmo = result.ftmo_risk_summary
        ftmo_passed = ftmo.passed if ftmo is not None else None
        ftmo_breach = ftmo.first_breach.breach_type if ftmo and ftmo.first_breach else None
        max_dd = getattr(kpi, "max_drawdown", None)
        return ExperimentRunScore(
            run_id=run_id,
            backtest_id=backtest_id,
            params=params,
            total_return=kpi.total_return,
            profit_factor=kpi.profit_factor,
            max_drawdown=max_dd,
            ftmo_passed=ftmo_passed,
            ftmo_breach_type=ftmo_breach,
        )


__all__ = ["ExperimentScorer"]
