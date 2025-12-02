from datetime import datetime, timezone, timedelta
from pathlib import Path

from research_lab.backend.core.backtests.models import (
    BacktestEngineDetail,
    BacktestKpiSummary,
    BacktestResult,
)
from research_lab.backend.core.backtests.persistence import BacktestPersistence


def make_result(run_id: str) -> BacktestResult:
    return BacktestResult(
        id=run_id,
        created_at=datetime.now(timezone.utc),
        mode="graph",
        kpi_summary=BacktestKpiSummary(
            total_return=1.0,
            mean_return=0.1,
            std_return=0.2,
            profit_factor=2.0,
            win_rate=0.6,
            max_drawdown=0.5,
            trade_count=10,
        ),
        engine_detail=BacktestEngineDetail(window_kpis=[]),
        metadata={"tag": "test"},
    )


def test_save_and_load_result(tmp_path: Path) -> None:
    persistence = BacktestPersistence(tmp_path)
    result = make_result("run1")

    path = persistence.save_result(result)
    assert path.exists()

    loaded = persistence.load_result("run1")
    assert loaded is not None
    assert loaded.id == result.id
    assert loaded.kpi_summary.total_return == result.kpi_summary.total_return
    assert loaded.created_at.tzinfo is not None
    assert loaded.created_at.utcoffset() == timedelta(0)


def test_list_runs(tmp_path: Path) -> None:
    persistence = BacktestPersistence(tmp_path)
    persistence.save_result(make_result("run1"))
    persistence.save_result(make_result("run2"))

    runs = persistence.list_runs()
    assert len(runs) == 2
    ids = {run.id for run in runs}
    assert {"run1", "run2"}.issubset(ids)
    assert all(run.created_at.tzinfo is not None for run in runs)
