from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.backtests.models import BacktestEngineDetail, BacktestKpiSummary, BacktestResult
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.backtests.service import BacktestService
from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.experiments.models import (
    ExperimentConfig,
    ExperimentParamPoint,
    ExperimentRunStatus,
    ExperimentStatus,
    ExperimentStrategyRef,
)
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.experiments.service import ExperimentService
from research_lab.backend.core.experiments.scoring import ExperimentScorer
from research_lab.backend.core.job_runner import InMemoryJobRunner


def _persist_result(persistence: BacktestPersistence, result_id: str, total_return: float, pf: float, ftmo_passed: bool) -> None:
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


def test_service_get_leaderboard(tmp_path: Path) -> None:
    backtest_persistence = BacktestPersistence(tmp_path / "backtests")
    experiment_persistence = ExperimentPersistence(tmp_path / "experiments")
    scorer = ExperimentScorer(backtest_persistence=backtest_persistence)
    backtest_service = BacktestService(
        job_runner=InMemoryJobRunner(), engine=RollingKpiBacktestEngine(), persistence=backtest_persistence
    )
    service = ExperimentService(backtest_service=backtest_service, experiment_persistence=experiment_persistence, scorer=scorer)

    config = ExperimentConfig(
        id="exp",
        name="exp",
        description=None,
        created_at=datetime.now(timezone.utc),
        strategy=ExperimentStrategyRef(mode="graph"),
        base_backtest={"mode": "graph", "returns": [0.1], "window": 1},
        param_grid=[ExperimentParamPoint(values={"size": 1.0}), ExperimentParamPoint(values={"size": 2.0})],
        tags=[],
        metadata={},
    )
    runs = [
        ExperimentRunStatus(run_id="r1", status=ExperimentStatus.COMPLETED, job_id=None, backtest_id="b1"),
        ExperimentRunStatus(run_id="r2", status=ExperimentStatus.COMPLETED, job_id=None, backtest_id="b2"),
    ]
    experiment_persistence.save_experiment(config, runs)
    _persist_result(backtest_persistence, "b1", 0.1, 1.0, True)
    _persist_result(backtest_persistence, "b2", 0.05, 0.8, False)

    leaderboard = service.get_leaderboard("exp")

    assert leaderboard is not None
    assert leaderboard.experiment_id == "exp"
    assert leaderboard.total_runs == 2
    assert leaderboard.runs[0].backtest_id == "b1"
