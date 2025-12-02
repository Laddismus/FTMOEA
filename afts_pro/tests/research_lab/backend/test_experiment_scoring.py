from datetime import datetime, timezone
from pathlib import Path

import pytest

from research_lab.backend.core.backtests.models import BacktestEngineDetail, BacktestKpiSummary, BacktestResult
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.experiments.models import (
    ExperimentConfig,
    ExperimentParamPoint,
    ExperimentRunStatus,
    ExperimentStatus,
    ExperimentStrategyRef,
)
from research_lab.backend.core.experiments.scoring import ExperimentScorer


def _save_result(persistence: BacktestPersistence, result_id: str, total_return: float, pf: float, ftmo_passed: bool) -> None:
    result = BacktestResult(
        id=result_id,
        created_at=datetime.now(timezone.utc),
        mode="graph",
        kpi_summary=BacktestKpiSummary(
            total_return=total_return,
            mean_return=0.0,
            std_return=0.0,
            profit_factor=pf,
            win_rate=0.0,
            max_drawdown=0.0,
            trade_count=1,
        ),
        engine_detail=BacktestEngineDetail(window_kpis=[]),
        metadata={},
        ftmo_risk_summary={"passed": ftmo_passed, "first_breach": None, "worst_daily_drawdown_pct": 0.0, "worst_total_drawdown_pct": 0.0, "config": {}},
    )
    persistence.save_result(result)


def test_build_leaderboard_ranks_and_pass_rate(tmp_path: Path) -> None:
    backtests_dir = tmp_path / "backtests"
    persistence = BacktestPersistence(backtests_dir)
    _save_result(persistence, "b1", total_return=0.2, pf=2.0, ftmo_passed=True)
    _save_result(persistence, "b2", total_return=0.1, pf=1.5, ftmo_passed=False)
    _save_result(persistence, "b3", total_return=0.25, pf=1.2, ftmo_passed=True)

    config = ExperimentConfig(
        id="exp",
        name="exp",
        description=None,
        created_at=datetime.now(timezone.utc),
        strategy=ExperimentStrategyRef(mode="graph"),
        base_backtest={"mode": "graph", "returns": [0.1], "window": 1},
        param_grid=[ExperimentParamPoint(values={"size": 1.0}), ExperimentParamPoint(values={"size": 2.0}), ExperimentParamPoint(values={"size": 3.0})],
        tags=[],
        metadata={},
    )
    runs = [
        ExperimentRunStatus(run_id="r1", backtest_id="b1", job_id=None, status=ExperimentStatus.COMPLETED),
        ExperimentRunStatus(run_id="r2", backtest_id="b2", job_id=None, status=ExperimentStatus.COMPLETED),
        ExperimentRunStatus(run_id="r3", backtest_id="b3", job_id=None, status=ExperimentStatus.COMPLETED),
    ]

    scorer = ExperimentScorer(backtest_persistence=persistence)
    leaderboard = scorer.build_leaderboard("exp", config, runs)

    assert leaderboard.total_runs == 3
    assert leaderboard.ftmo_pass_rate == pytest.approx(2 / 3)
    assert [r.rank for r in leaderboard.runs] == [1, 2, 3]
    # best run should be ftmo passed and highest return (b3)
    assert leaderboard.runs[0].backtest_id == "b3"
