from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.backtests.models import BacktestRequest
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.backtests.service import BacktestService
from research_lab.backend.core.experiments.models import (
    ExperimentConfig,
    ExperimentParamPoint,
    ExperimentStatus,
    ExperimentStrategyRef,
)
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.experiments.service import ExperimentService
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.experiments.scoring import ExperimentScorer


def _make_config(exp_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        id=exp_id,
        name="exp",
        description=None,
        created_at=datetime.now(timezone.utc),
        strategy=ExperimentStrategyRef(mode="graph"),
        base_backtest=BacktestRequest(mode="graph", returns=[0.1, -0.05], window=2),
        param_grid=[ExperimentParamPoint(values={"size": 1.0}), ExperimentParamPoint(values={"size": 2.0})],
        tags=[],
        metadata={},
    )


def build_service(tmp_path: Path) -> ExperimentService:
    job_runner = InMemoryJobRunner()
    backtest_service = BacktestService(
        job_runner=job_runner,
        engine=RollingKpiBacktestEngine(),
        persistence=BacktestPersistence(tmp_path / "backtests"),
    )
    experiment_persistence = ExperimentPersistence(tmp_path / "experiments")
    scorer = ExperimentScorer(backtest_persistence=backtest_service.persistence)
    return ExperimentService(backtest_service=backtest_service, experiment_persistence=experiment_persistence, scorer=scorer)


def test_create_experiment(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    cfg = _make_config("exp1")

    created = service.create_experiment(cfg)

    assert created.id == "exp1"
    loaded = service.get_experiment("exp1")
    assert loaded is not None


def test_launch_and_refresh_experiment(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    cfg = _make_config("exp2")
    service.create_experiment(cfg)

    config, runs = service.launch_experiment("exp2")
    assert len(runs) == len(cfg.param_grid)
    assert all(r.status == ExperimentStatus.RUNNING for r in runs)

    config2, refreshed = service.refresh_status("exp2")
    assert config2.id == config.id
    assert any(r.status == ExperimentStatus.COMPLETED for r in refreshed)
    assert all(r.backtest_id is not None for r in refreshed if r.status == ExperimentStatus.COMPLETED)
